#!/usr/bin/env python
"""Cache builder for the paper's *redshift dependence* subsection (Figs. R1-R4).

One GPU pass over the multi-redshift held-out test set with the
``--condition_redshift`` model (``fm_redshift``), producing two caches:

``zcache.npz``  — per-patch truth-vs-BIND summaries at every training snapshot:
    integrated masses (full patch and within a projected R200c aperture),
    radial surface-density profiles on both a fixed comoving grid and an
    r/R200c grid, and aperture-integrated thermo scalars. This backs
    Figs. R1 (accuracy vs z), R2 (profiles vs z) and R3 (evolution).

``zsweep.npz``  — the redshift analogue of the 1P butterfly figure: a fixed set
    of z=0 DMO patches (structure + astrophysical parameters held fixed) with
    only the conditioning scale factor a=1/(1+z) swept, at the *same* initial
    ODE noise. Includes off-grid z values that appear in no training snapshot,
    which is what separates smooth interpolation in z from memorization of the
    discrete snapshot set. This backs Fig. R4.

Both caches are pure numpy and are consumed by ``make_z_figures.py``.

Usage
-----
    python tools/paper_cache/build_zcache.py --n_per_z 600
    python tools/paper_cache/build_zcache.py --run_name fm_redshift_loo052 \
        --tag loo052 --snaps 52          # held-out-snapshot variant
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bind.data import (  # noqa: E402
    PIX_MPC_H,
    RHO_CRIT_0,
    SNAPSHOT_REDSHIFTS,
    THERMO_KEYS,
    AstroDataset,
    NormStats,
    a_to_z,
)
from bind.inference.pipeline import _denormalize_to_physical  # noqa: E402
from bind.train import FlowMatchingLit  # noqa: E402

# ── Geometry / binning ──────────────────────────────────────────────────────
N_PIX = 128
MASS_NAMES = ["DM_hydro", "Gas", "Stars"]
THERMO_NAMES = list(THERMO_KEYS)

# Fixed comoving radial grid, out to the patch half-width (3.125 Mpc/h). Matches
# the r [Mpc/h] x-axis of the paper's z=0 profile figure, so R2 is the same
# estimator with a different grouping. The innermost bin is [0, 1.5 px] rather
# than a log bin so the central pixels are captured and no annulus is empty (a
# purely log grid starting at one pixel leaves the first two bins pixel-free).
R_EDGES_COM = np.concatenate([[0.0],
                              np.logspace(np.log10(1.5 * PIX_MPC_H),
                                          np.log10(3.0), 16)])
# Scale-free grid in units of R200c, so a cross-z comparison is not confounded
# by the growth of R200c itself. R200c spans ~9-12 px here, so 0.2 R200c is the
# smallest annulus that stays populated at the small-R200c end.
RR_EDGES = np.concatenate([[0.0],
                           np.logspace(np.log10(0.2), np.log10(4.0), 16)])


def _bin_centers(edges):
    """Geometric centers, with the innermost (zero-anchored) bin at half-width."""
    cen = np.sqrt(edges[:-1] * edges[1:])
    cen[0] = 0.5 * edges[1]
    return cen


R_CEN_COM = _bin_centers(R_EDGES_COM)
RR_CEN = _bin_centers(RR_EDGES)
N_RBIN = len(R_CEN_COM)

# Param-vector column indices (see src/bind/assets/SB35_param_minmax.csv).
I_OMEGA_M, I_OMEGA_B = 0, 6


# ── Redshift-aware R200c ────────────────────────────────────────────────────
def r200c_comoving(m200c, z, omega_m):
    """Projected R200c in *comoving* Mpc/h — the units the maps are gridded in.

    M200c = 200 rho_c(z) (4pi/3) R_phys^3 with rho_c(z) = rho_c0 E^2(z), and
    R_com = R_phys (1+z), so

        R_com = [3M / (4pi 200 rho_c0)]^(1/3) * (1+z) / E(z)^(2/3).

    At z=0 this reduces to ``bind.data.m200c_to_r200c``, which is what the
    paper's z=0 figures use — the two stay consistent by construction.
    """
    m200c = np.asarray(m200c, np.float64)
    z = np.asarray(z, np.float64)
    omega_m = np.asarray(omega_m, np.float64)
    e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    r0 = np.cbrt(m200c / (4.0 / 3.0 * np.pi * 200.0 * RHO_CRIT_0))
    return r0 * (1.0 + z) / e2 ** (1.0 / 3.0)


# ── Radial binning helpers ──────────────────────────────────────────────────
_c = (N_PIX - 1) / 2.0
_yy, _xx = np.mgrid[0:N_PIX, 0:N_PIX]
RPIX_COM = (np.hypot(_xx - _c, _yy - _c) * PIX_MPC_H).astype(np.float64)  # Mpc/h
_RFLAT = RPIX_COM.ravel()

_IDX_COM = np.digitize(_RFLAT, R_EDGES_COM) - 1
_VALID_COM = (_IDX_COM >= 0) & (_IDX_COM < N_RBIN)
_IDX_COM_V = _IDX_COM[_VALID_COM]
_CNT_COM = np.bincount(_IDX_COM_V, minlength=N_RBIN).astype(np.float64)


def _profile(img_flat, idx, valid, cnt):
    """Azimuthally averaged surface density in preset annuli (mean per pixel)."""
    out = np.bincount(idx, weights=img_flat[valid], minlength=len(cnt))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cnt > 0, out / cnt, np.nan)


def profiles_for_patch(maps3, r200):
    """(3, n_r) profiles on the fixed comoving grid and on the r/R200c grid."""
    idx_rr = np.digitize(_RFLAT
                         / max(float(r200), 1e-6), RR_EDGES) - 1
    valid_rr = (idx_rr >= 0) & (idx_rr < N_RBIN)
    idx_rr_v = idx_rr[valid_rr]
    cnt_rr = np.bincount(idx_rr_v, minlength=N_RBIN).astype(np.float64)

    p_com = np.empty((3, N_RBIN))
    p_rr = np.empty((3, N_RBIN))
    for c in range(3):
        f = maps3[c].ravel().astype(np.float64)
        p_com[c] = _profile(f, _IDX_COM_V, _VALID_COM, _CNT_COM)
        p_rr[c] = _profile(f, idx_rr_v, valid_rr, cnt_rr)
    return p_com, p_rr


def aperture_mask(r200):
    return RPIX_COM < float(r200)


# ── File indexing ───────────────────────────────────────────────────────────
def _read_meta(f):
    """(halo_mass, redshift, Omega_m, Omega_b, has_thermo) for one patch file."""
    try:
        with np.load(f) as d:
            if not all(k in d.files for k in THERMO_KEYS):
                return None
            p = d["params"]
            return (
                float(d["halo_mass"]),
                float(d["redshift"]) if "redshift" in d.files else np.nan,
                float(p[I_OMEGA_M]),
                float(p[I_OMEGA_B]),
            )
    except Exception:
        return None


def build_index(data_root, split, cache_path, n_workers=32):
    """Index every test patch once (halo mass, z, cosmology); cache to npz."""
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        print(f"  index: loaded {len(d['files'])} patches from {cache_path.name}")
        return {k: d[k] for k in d.files}

    root = Path(data_root) / split
    files = sorted(str(p) for p in root.glob("sim_*/snap_*/*.npz"))
    print(f"  index: scanning {len(files)} files under {root} ...")
    with ThreadPoolExecutor(n_workers) as ex:
        metas = list(ex.map(_read_meta, files))

    keep, hm, zz, om, ob, snap = [], [], [], [], [], []
    for f, m in zip(files, metas):
        if m is None:
            continue
        s = re.search(r"snap_?(\d+)", f)
        if s is None or int(s.group(1)) not in SNAPSHOT_REDSHIFTS:
            continue
        keep.append(f)
        hm.append(m[0])
        z = m[1] if np.isfinite(m[1]) else SNAPSHOT_REDSHIFTS[int(s.group(1))]
        zz.append(z)
        om.append(m[2])
        ob.append(m[3])
        snap.append(int(s.group(1)))

    out = dict(
        files=np.array(keep),
        halo_mass=np.array(hm, np.float64),
        redshift=np.array(zz, np.float64),
        omega_m=np.array(om, np.float64),
        omega_b=np.array(ob, np.float64),
        snap=np.array(snap, np.int64),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f"  index: kept {len(keep)}/{len(files)} thermo-complete patches "
          f"-> {cache_path.name}")
    return out


# ── Model ───────────────────────────────────────────────────────────────────
def load_model(run_dir, ckpt_name, device):
    ckpt = Path(run_dir) / "checkpoints" / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval().to(device)
    ns = NormStats.load(Path(run_dir) / "norm_stats.npz")
    if not ns.predict_thermo:
        raise RuntimeError(f"{run_dir}: norm_stats has no thermo stats")
    if not bool(getattr(model.hparams, "condition_redshift", False)):
        raise RuntimeError(f"{run_dir}: model is not redshift-conditioned")
    # Matches bind.inference.runner: raw checkpoint weights, never EMA, so these
    # figures use the identical weights as every other figure in the paper.
    return model, ns


@torch.no_grad()
def generate(model, ns, files, batch_size, n_steps, n_workers, device, seed=1234):
    """Sample DMO->hydro conditioned on each patch's own a=1/(1+z)."""
    ds = AstroDataset(files, ns, condition_redshift=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, pin_memory=True,
                        persistent_workers=(n_workers > 0))
    torch.manual_seed(seed)
    real, gen, sf = [], [], []
    for bi, batch in enumerate(loader):
        cond = batch["condition"].to(device)
        ls = batch["large_scale"].to(device)
        params = batch["params"].to(device)
        a = batch["scale_factor"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            g = model.fm.sample(cond, ls, params, n_steps=n_steps, scale_factor=a)
        real.append(_denormalize_to_physical(batch["target"].numpy().copy(), ns))
        gen.append(_denormalize_to_physical(g.float().cpu().numpy(), ns))
        sf.append(batch["scale_factor"].numpy().copy())
        if bi % 10 == 0:
            print(f"    batch {bi}/{len(loader)}", flush=True)
    return (np.concatenate(real), np.concatenate(gen), np.concatenate(sf))


# ── Per-patch reduction ─────────────────────────────────────────────────────
def reduce_patches(real, gen, r200):
    """Integrated masses, aperture masses, profiles and thermo scalars."""
    n = len(real)
    out = dict(
        m_full_t=np.zeros((n, 3)), m_full_g=np.zeros((n, 3)),
        m_ap_t=np.zeros((n, 3)), m_ap_g=np.zeros((n, 3)),
        prof_com_t=np.zeros((n, 3, N_RBIN)), prof_com_g=np.zeros((n, 3, N_RBIN)),
        prof_rr_t=np.zeros((n, 3, N_RBIN)), prof_rr_g=np.zeros((n, 3, N_RBIN)),
        th_ap_t=np.zeros((n, len(THERMO_NAMES))),
        th_ap_g=np.zeros((n, len(THERMO_NAMES))),
        y_ap_t=np.zeros(n), y_ap_g=np.zeros(n),
    )
    pix_area = PIX_MPC_H ** 2
    for i in range(n):
        mask = aperture_mask(r200[i])
        rt, gt = real[i], gen[i]
        out["m_full_t"][i] = rt[:3].sum(axis=(-2, -1))
        out["m_full_g"][i] = gt[:3].sum(axis=(-2, -1))
        out["m_ap_t"][i] = (rt[:3] * mask).sum(axis=(-2, -1))
        out["m_ap_g"][i] = (gt[:3] * mask).sum(axis=(-2, -1))
        out["prof_com_t"][i], out["prof_rr_t"][i] = profiles_for_patch(rt[:3], r200[i])
        out["prof_com_g"][i], out["prof_rr_g"][i] = profiles_for_patch(gt[:3], r200[i])
        # Y_200: aperture-integrated Compton-y (extensive).
        out["y_ap_t"][i] = rt[3][mask].sum() * pix_area
        out["y_ap_g"][i] = gt[3][mask].sum() * pix_area
        # T / K / P_e: mean over positive pixels in the aperture (intensive).
        for j in range(len(THERMO_NAMES)):
            for tag, arr in (("t", rt), ("g", gt)):
                v = arr[3 + j][mask]
                v = v[v > 0]
                out[f"th_ap_{tag}"][i, j] = v.mean() if v.size else np.nan
    return out


# ── Fixed-DMO redshift sweep (Fig. R4) ──────────────────────────────────────
@torch.no_grad()
def redshift_sweep(model, ns, files, z_grid, n_steps, device, seed=777):
    """Hold structure + parameters fixed, sweep only the conditioning a=1/(1+z).

    The initial ODE noise is reseeded identically before every z, so a given
    halo starts from the same latent at all redshifts and any difference in the
    output is attributable to the redshift label alone rather than to sampling
    scatter.
    """
    ds = AstroDataset(files, ns, condition_redshift=True)
    batch = next(iter(DataLoader(ds, batch_size=len(files), shuffle=False)))
    cond = batch["condition"].to(device)
    ls = batch["large_scale"].to(device)
    params = batch["params"].to(device)
    nz, nh = len(z_grid), len(files)

    fields = np.zeros((nz, nh, 3 + len(THERMO_NAMES), N_PIX, N_PIX), np.float32)
    for zi, z in enumerate(z_grid):
        a = torch.full((nh,), 1.0 / (1.0 + float(z)),
                       dtype=torch.float32, device=device)
        torch.manual_seed(seed)          # identical latent at every z
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            g = model.fm.sample(cond, ls, params, n_steps=n_steps, scale_factor=a)
        fields[zi] = _denormalize_to_physical(g.float().cpu().numpy(), ns)
        print(f"    sweep z={z:.3f} done", flush=True)
    return fields


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root",
                    default="/mnt/home/mlee1/ceph/train_data_multiz_128_cpu")
    ap.add_argument("--runs_dir", default="/mnt/home/mlee1/ceph/fm_runs")
    ap.add_argument("--run_name", default="fm_redshift")
    ap.add_argument("--ckpt", default="last.ckpt")
    ap.add_argument("--out_dir", default="/mnt/home/mlee1/ceph/paper_cache/fm_redshift")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--n_per_z", type=int, default=600)
    ap.add_argument("--mass_min", type=float, default=1e13,
                    help="trained-regime cut, matching the rest of the paper")
    ap.add_argument("--snaps", type=int, nargs="*", default=None,
                    help="restrict to these snapshot numbers")
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sweep_halos", type=int, default=24)
    ap.add_argument("--skip_sweep", action="store_true")
    ap.add_argument("--n_hero", type=int, default=3,
                    help="full maps kept per redshift for the visual panels")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    print(f"[1/5] indexing {args.data_root}")
    idx = build_index(args.data_root, "test",
                      out_dir / "zindex_test.npz")

    print(f"[2/5] selecting <= {args.n_per_z} patches/snapshot "
          f"with M200c >= {args.mass_min:.1e}")
    rng = np.random.RandomState(args.seed)
    snaps = sorted(set(idx["snap"].tolist()), reverse=True)
    if args.snaps:
        snaps = [s for s in snaps if s in set(args.snaps)]
    sel = []
    for s in snaps:
        cand = np.where((idx["snap"] == s) & (idx["halo_mass"] >= args.mass_min))[0]
        if len(cand) == 0:
            print(f"    snap_{s:03d} (z={SNAPSHOT_REDSHIFTS[s]:.3f}): 0 halos — skipped")
            continue
        pick = cand if len(cand) <= args.n_per_z else \
            cand[rng.permutation(len(cand))[:args.n_per_z]]
        sel.append(np.sort(pick))
        print(f"    snap_{s:03d} (z={SNAPSHOT_REDSHIFTS[s]:.3f}): "
              f"{len(pick)}/{len(cand)} selected")
    sel = np.concatenate(sel)

    files = list(idx["files"][sel])
    halo_mass = idx["halo_mass"][sel]
    omega_m = idx["omega_m"][sel]
    omega_b = idx["omega_b"][sel]
    snap = idx["snap"][sel]
    # Each SB35 simulation has its own cosmology, so a given snapshot number
    # lands at a slightly different redshift in each (drifting up to ~6e-3 by
    # z=4). Grouping on the per-file redshift would shatter every level into
    # near-duplicates, so group on the snapshot number and label it with the
    # nominal redshift -- but use each patch's *own* redshift wherever the value
    # enters physics, i.e. in R200c(z).
    z_file = idx["redshift"][sel]
    zlab = np.array([SNAPSHOT_REDSHIFTS[s] for s in snap], np.float64)
    r200 = r200c_comoving(halo_mass, z_file, omega_m)
    print("    per-snapshot offset of the nominal label from the file redshift:")
    for s in sorted(set(snap.tolist()), reverse=True):
        m = snap == s
        print(f"      snap_{s:03d}: nominal z={SNAPSHOT_REDSHIFTS[s]:.4f}  "
              f"file z in [{z_file[m].min():.4f}, {z_file[m].max():.4f}]  "
              f"max|offset|={np.abs(z_file[m] - zlab[m]).max():.1e}")

    print(f"[3/5] loading {args.run_name}/{args.ckpt}")
    model, ns = load_model(Path(args.runs_dir) / args.run_name, args.ckpt, device)

    print(f"[4/5] generating {len(files)} patches ({args.n_steps} ODE steps)")
    real, gen, sf = generate(model, ns, files, args.batch_size, args.n_steps,
                             args.n_workers, device, seed=args.seed)
    # The conditioning the model actually saw is the dataset's own a=1/(1+z);
    # confirm it round-trips to the redshift stored in each file.
    assert np.allclose(a_to_z(sf), z_file, atol=1e-4), "scale-factor/file mismatch"

    red = reduce_patches(real, gen, r200)

    # Keep a few full maps per redshift (most massive halos) for visual panels.
    hero_idx, hero_z = [], []
    for s in sorted(set(snap.tolist()), reverse=True):
        w = np.where(snap == s)[0]
        top = w[np.argsort(-halo_mass[w])[:args.n_hero]]
        hero_idx.extend(top.tolist())
        hero_z.extend([SNAPSHOT_REDSHIFTS[s]] * len(top))
    hero_idx = np.array(hero_idx, np.int64)

    payload = dict(
        files=np.array(files), halo_mass=halo_mass, z=zlab, z_file=z_file,
        snap=snap, omega_m=omega_m, omega_b=omega_b, r200=r200,
        r_cen_com=R_CEN_COM, rr_cen=RR_CEN,
        mass_names=np.array(MASS_NAMES), thermo_names=np.array(THERMO_NAMES),
        hero_idx=hero_idx, hero_z=np.array(hero_z),
        hero_truth=real[hero_idx], hero_gen=gen[hero_idx],
        n_steps=args.n_steps, mass_min=args.mass_min,
        run_name=args.run_name, ckpt=args.ckpt,
        **red,
    )
    p = out_dir / f"zcache{sfx}.npz"
    np.savez_compressed(p, **payload)
    print(f"    wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    if args.skip_sweep:
        print("[5/5] sweep skipped")
        return

    print(f"[5/5] fixed-DMO redshift sweep on {args.n_sweep_halos} z=0 halos")
    z0 = np.where(snap == 90)[0]
    if len(z0) == 0:
        print("    no z=0 patches selected — sweep skipped")
        return
    # Log-uniform spread in halo mass so the sweep is not all cluster-scale.
    order = z0[np.argsort(halo_mass[z0])]
    take = np.unique(np.linspace(0, len(order) - 1, args.n_sweep_halos).astype(int))
    sweep_idx = order[take]
    sweep_files = [files[i] for i in sweep_idx]

    z_train = np.array([SNAPSHOT_REDSHIFTS[s] for s in
                        sorted(SNAPSHOT_REDSHIFTS, reverse=True) if s >= 44])
    # Off-grid values that correspond to no training snapshot: if the model has
    # merely memorized the discrete snapshots these will not land on a smooth
    # monotonic sequence.
    z_off = np.array([0.10, 0.33, 0.70, 1.25, 1.75])
    z_grid = np.unique(np.concatenate([z_train, z_off]))
    is_train_z = np.isin(np.round(z_grid, 4), np.round(z_train, 4))

    fields = redshift_sweep(model, ns, sweep_files, z_grid, args.n_steps, device)

    pix_area = PIX_MPC_H ** 2
    nz, nh = fields.shape[:2]
    m_ap = np.zeros((nz, nh, 3))
    m_full = fields[:, :, :3].sum(axis=(-2, -1))
    y_ap = np.zeros((nz, nh))
    th_ap = np.zeros((nz, nh, len(THERMO_NAMES)))
    # Aperture follows the swept label: R200c(z) at the halo's fixed M200c.
    r200_sweep = np.array([[r200c_comoving(halo_mass[j], z, omega_m[j])
                            for j in sweep_idx] for z in z_grid])
    for zi in range(nz):
        for hi in range(nh):
            mask = aperture_mask(r200_sweep[zi, hi])
            m_ap[zi, hi] = (fields[zi, hi, :3] * mask).sum(axis=(-2, -1))
            y_ap[zi, hi] = fields[zi, hi, 3][mask].sum() * pix_area
            for j in range(len(THERMO_NAMES)):
                v = fields[zi, hi, 3 + j][mask]
                v = v[v > 0]
                th_ap[zi, hi, j] = v.mean() if v.size else np.nan

    p = out_dir / f"zsweep{sfx}.npz"
    np.savez_compressed(
        p, z_grid=z_grid, is_train_z=is_train_z,
        halo_mass=halo_mass[sweep_idx], omega_m=omega_m[sweep_idx],
        omega_b=omega_b[sweep_idx], r200=r200_sweep,
        files=np.array(sweep_files), fields=fields,
        m_ap=m_ap, m_full=m_full, y_ap=y_ap, th_ap=th_ap,
        mass_names=np.array(MASS_NAMES), thermo_names=np.array(THERMO_NAMES),
    )
    print(f"    wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
