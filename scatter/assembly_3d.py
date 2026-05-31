"""scatter/assembly_3d.py — does halo assembly (3D structure) set the baryon scatter?

Large-N test on the 27 CAMELS-IllustrisTNG CV simulations (all fiducial; ~1100 halos between them).
Anchored on OUR pipeline halos (fm_testsuite/CV/sim_*), so we attach, per halo:

  * baryon content from BOTH truth and the emulator (full_maps + generated_halos), and
  * 3D ASSEMBLY structure from the DMO / N-body Subfind catalog (clean, baryon-free):
        c_V    = Vmax / V200             concentration proxy (formation-time tracer; Prada+2012)
        lambda = Bullock spin            |j| / (sqrt(2) V200 R200)
        veldisp                          subhalo velocity dispersion
        rhalf  = SubhaloHalfmassRad/R200 compactness

We then correlate the mass-detrended baryon residual (the 'assembly' scatter) against each structural
property, for truth AND emulator. Truth establishes the real physical dependence; the emulator
agreeing validates that the decomposition's 'assembly' term is identified structure, not a black box.

Conventions: catalogs in Gadget units (1e10 Msun/h, ckpc/h, km/s); halo_positions in Mpc/h.
DMO runs replace 'IllustrisTNG' with 'IllustrisTNG_DM'. fm_testsuite sim_i <-> CAMELS CV_i.

Usage:
    python -m scatter.assembly_3d                       # all CV sims, M>1e13
    python -m scatter.assembly_3d --cv-max 5
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

from scatter.obs_common import (
    observables_from_phys, axis_ratio_q, PATCH_PIX, BOX_SIZE, N_PIX_FULL, OMEGA_B_FIXED,
)

FM_CV_ROOT = "/mnt/home/mlee1/ceph/fm_testsuite/CV"            # our pipeline (truth + emulator)
DMO_ROOT   = "/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/CV"  # assembly structure
SNAP = "groups_090"; SUB = "snap_090/mass_threshold_1p000e13"
BOX_KPC_H = BOX_SIZE * 1000.0                                  # 50000 ckpc/h
G_KPC = 4.300917270e-6                                         # kpc (km/s)^2 / Msun (h cancels)
OUT_DIR = pathlib.Path("outputs/scatter_diagnostics"); FIG_DIR = pathlib.Path("figures/scatter_diagnostics")

ASSEMBLY_PROPS = ["c_V", "lambda", "veldisp", "rhalf", "z_form"]
FOCUS = ["M_gas", "M_star", "f_b"]
LOG_FOCUS = {"M_gas", "M_star"}


# ── DMO assembly structure (shared CAMELS catalogs) ──────────────────────────
def _read_cat(cat_file, group, fields):
    """Read fields from a single-file Subfind catalog (DM N-body layout)."""
    if not pathlib.Path(cat_file).exists():
        return {f: np.array([]) for f in fields}
    with h5py.File(cat_file, "r") as h:
        if group not in h:
            return {f: np.array([]) for f in fields}
        return {f: (h[group][f][:] if f in h[group] else np.array([])) for f in fields}


def dmo_assembly(cat_file, m200_min):
    g = _read_cat(cat_file, "Group",
                  ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"])
    s = _read_cat(cat_file, "Subhalo",
                  ["SubhaloVmax", "SubhaloSpin", "SubhaloVelDisp", "SubhaloHalfmassRad"])
    if g["Group_M_Crit200"].size == 0:
        return {}
    M200 = g["Group_M_Crit200"] * 1e10; R200 = g["Group_R_Crit200"]
    keep = (M200 > m200_min) & (R200 > 0) & (g["GroupFirstSub"] >= 0)
    fs = g["GroupFirstSub"][keep].astype(int)
    M200, R200, pos = M200[keep], R200[keep], g["GroupPos"][keep]
    V200 = np.sqrt(G_KPC * M200 / R200)
    spin = np.linalg.norm(s["SubhaloSpin"][fs], axis=1)
    return {"M200": M200, "pos": pos,
            "c_V": s["SubhaloVmax"][fs] / V200,
            "lambda": spin / (np.sqrt(2.0) * V200 * R200 + 1e-30),
            "veldisp": s["SubhaloVelDisp"][fs],
            "rhalf": s["SubhaloHalfmassRad"][fs] / R200}


def _group_snap(cat_file):
    """(M200 [Msun/h], pos [ckpc/h], scale_factor a) for one snapshot file, or None."""
    if not pathlib.Path(cat_file).exists():
        return None
    with h5py.File(cat_file, "r") as h:
        a = float(h["Header"].attrs.get("Time", np.nan))
    g = _read_cat(cat_file, "Group", ["Group_M_Crit200", "GroupPos"])
    if g["Group_M_Crit200"].size == 0:
        return None
    return g["Group_M_Crit200"] * 1e10, g["GroupPos"].astype(float), a


def formation_redshift(dmo_dir, pos0_kpc, M200_0, snap_max=89, snap_min=33,
                       tol_kpc=400.0, mass_floor_frac=0.1, step=1):
    """z at which each halo's main progenitor first reached 0.5 * M200(z=0).

    Tree-free main-branch trace: step back through DMO snapshots, following each halo by taking the
    most-massive group within tol of its current position (position-tracked so the progenitor can
    drift), until its M200 drops below half the z=0 value; interpolate the crossing scale factor.
    """
    Nh = len(M200_0)
    cur_pos = (pos0_kpc % BOX_KPC_H).astype(float)
    cur_M = M200_0.astype(float).copy()
    a_prev = np.ones(Nh); a_form = np.full(Nh, np.nan); active = np.ones(Nh, bool)
    floor = mass_floor_frac * float(np.nanmin(M200_0)); half = 0.5 * M200_0
    for snap in range(snap_max, snap_min - 1, -step):
        if not active.any():
            break
        gs = _group_snap(f"{dmo_dir}/fof_subhalo_tab_{snap:03d}.hdf5")
        if gs is None:
            continue
        M, pos, a = gs
        sel = M > floor
        if sel.sum() == 0:
            continue
        Msel = M[sel]; psel = pos[sel] % BOX_KPC_H
        tree = cKDTree(psel, boxsize=BOX_KPC_H)
        for i in np.where(active)[0]:
            nb = tree.query_ball_point(cur_pos[i], tol_kpc)
            Mprog = float(Msel[nb].max()) if nb else 0.0
            if Mprog < half[i]:
                denom = cur_M[i] - Mprog
                frac = (cur_M[i] - half[i]) / denom if denom > 0 else 0.0
                a_form[i] = a_prev[i] + (a - a_prev[i]) * frac
                active[i] = False
            else:
                k = nb[int(np.argmax(Msel[nb]))]
                cur_pos[i] = psel[k]; cur_M[i] = Mprog; a_prev[i] = a
    return 1.0 / np.clip(a_form, 1e-3, None) - 1.0     # z_form (NaN where never crossed)


def _match(pos_a_kpc, pos_b_kpc, tol_kpc=200.0):
    idx = np.full(len(pos_a_kpc), -1, dtype=int)
    for i, p in enumerate(pos_a_kpc):
        d = pos_b_kpc - p; d -= BOX_KPC_H * np.round(d / BOX_KPC_H)
        r2 = np.einsum("ij,ij->i", d, d); j = int(np.argmin(r2))
        if r2[j] < tol_kpc ** 2:
            idx[i] = j
    return idx


# ── our pipeline halos: truth + emulator observables (my dir) ────────────────
def _patch(field_2d, cx, cy):
    n = field_2d.shape[0]; half = PATCH_PIX // 2
    ix = (cx - half + np.arange(PATCH_PIX)) % n; iy = (cy - half + np.arange(PATCH_PIX)) % n
    return field_2d[np.ix_(ix, iy)]


def ingest_cv(sim_idx):
    base = pathlib.Path(FM_CV_ROOT) / f"sim_{sim_idx}"
    fm_p, cat_p = base / "snap_090/full_maps.npz", base / SUB / "halo_catalog.npz"
    cut_p, gen_p = base / SUB / "halo_cutouts.npz", base / SUB / "fm_two_head/generated_halos.npz"
    if not all(p.exists() for p in (fm_p, cat_p, cut_p, gen_p)):
        return None
    fm = np.load(fm_p, allow_pickle=True); cat = np.load(cat_p, allow_pickle=True)
    cuts = np.load(cut_p, allow_pickle=True); gen = np.load(gen_p, allow_pickle=True)["generated"]
    masses = cat["halo_masses"].astype(np.float64)
    pos_mpc = cat["halo_positions"].astype(np.float64)
    ppm = N_PIX_FULL / BOX_SIZE
    cpix = (cat["centers"] * ppm).astype(np.int64) % N_PIX_FULL
    params = cat["params"].astype(np.float64); tmaps = fm["truth_maps"]; cond = cuts["condition"]
    n = len(masses); truth = {o: np.full(n, np.nan) for o in FOCUS}; genr = {o: np.full(n, np.nan) for o in FOCUS}
    from scatter.obs_common import r200c_pix
    for i in range(n):
        om = float(params[i, 0]) if params[i, 0] > 0 else 0.3
        r200p = r200c_pix(float(masses[i])); fb = OMEGA_B_FIXED / max(om, 1e-10)
        r_aper = max(min(r200p, PATCH_PIX / 2 - 2), 4.0)
        q = axis_ratio_q(np.maximum(cond[i].astype(np.float64), 0.0), r_aper)
        tp = np.stack([_patch(tmaps[c], cpix[i, 0], cpix[i, 1]) for c in range(3)]).astype(np.float64)
        t = observables_from_phys(tp, r200p, fb, q); ge = observables_from_phys(gen[i].astype(np.float64), r200p, fb, q)
        for o in FOCUS:
            truth[o][i] = t[o]; genr[o][i] = ge[o]
    return {"pos_mpc": pos_mpc, "logM": np.log10(masses), "truth": truth, "gen": genr, "N": n}


def residual(O, logM, is_log):
    y = np.log10(np.clip(O, 1e-30, None)) if is_log else np.asarray(O, float)
    ok = np.isfinite(y) & np.isfinite(logM)
    if ok.sum() < 5:
        return np.full_like(y, np.nan)
    b, a = np.polyfit(logM[ok], y[ok], 1); return y - (a + b * logM)


PART_KEYS = ASSEMBLY_PROPS + ["logM"] + [f"{s}_{o}" for s in ("truth", "gen") for o in FOCUS]


def process_sim(i, args) -> dict | None:
    """All matched-halo columns for ONE CV sim (the parallelizable unit). None if data missing."""
    cv = ingest_cv(i)
    if cv is None:
        print(f"  sim_{i}: pipeline products missing, skip"); return None
    dmo = dmo_assembly(f"{DMO_ROOT}/CV_{i}/fof_subhalo_tab_090.hdf5", args.m200_min)
    if not dmo:
        print(f"  sim_{i}: DMO catalog missing, skip"); return None
    if args.no_formation:
        dmo["z_form"] = np.full(len(dmo["M200"]), np.nan)
    else:
        dmo["z_form"] = formation_redshift(f"{DMO_ROOT}/CV_{i}", dmo["pos"], dmo["M200"],
                                           snap_min=args.snap_min, tol_kpc=args.form_tol_kpc)
        print(f"  sim_{i}: z_form median = {np.nanmedian(dmo['z_form']):.2f} "
              f"({np.isfinite(dmo['z_form']).sum()}/{len(dmo['z_form'])} traced)")
    j = _match(cv["pos_mpc"] * 1000.0, dmo["pos"], args.match_tol_kpc)
    ok = j >= 0
    out = {p: dmo[p][j[ok]] for p in ASSEMBLY_PROPS}
    out["logM"] = cv["logM"][ok]
    for o in FOCUS:
        out[f"truth_{o}"] = cv["truth"][o][ok]; out[f"gen_{o}"] = cv["gen"][o][ok]
    print(f"  sim_{i}: {int(ok.sum())}/{cv['N']} halos matched to DMO assembly")
    return out


def analyze(data: dict) -> None:
    """Reduction: correlate baryon residuals vs structure, write json + figure."""
    N = len(data["logM"]); print(f"\nTotal matched halos: {N}")
    if N < 20:
        print("Too few halos — aborting."); return
    results = {"n_halos": N, "obs": {}}
    for o in FOCUS:
        ent = {}
        for src in ("truth", "gen"):
            res = residual(data[f"{src}_{o}"], data["logM"], o in LOG_FOCUS)
            sp = {}
            for p in ASSEMBLY_PROPS:
                x = data[p]; ok = np.isfinite(x) & np.isfinite(res)
                rho, pv = spearmanr(x[ok], res[ok]) if ok.sum() > 10 else (np.nan, np.nan)
                sp[p] = {"rho": float(rho), "p": float(pv)}
            usable = [p for p in ASSEMBLY_PROPS if np.isfinite(data[p]).sum() > 10]  # drop all-NaN cols
            X = np.column_stack([data[p] for p in usable]) if usable else np.empty((len(res), 0))
            ok = np.all(np.isfinite(X), axis=1) & np.isfinite(res)
            r2 = np.nan
            if X.shape[1] and ok.sum() > X.shape[1] + 5:
                Xo = np.column_stack([np.ones(ok.sum()), X[ok]])
                beta, *_ = np.linalg.lstsq(Xo, res[ok], rcond=None); pred = Xo @ beta
                r2 = float(1 - np.sum((res[ok]-pred)**2) / np.sum((res[ok]-res[ok].mean())**2))
            ent[src] = {"residual_std": float(np.nanstd(res)), "spearman": sp, "explained_fraction": r2}
        results["obs"][o] = ent
        print(f"\n{o}: truth structure explains {ent['truth']['explained_fraction']*100:.0f}% "
              f"| emulator {ent['gen']['explained_fraction']*100:.0f}%")
        for p in ASSEMBLY_PROPS:
            print(f"    {p:9s} truth ρ={ent['truth']['spearman'][p]['rho']:+.2f}  "
                  f"emul ρ={ent['gen']['spearman'][p]['rho']:+.2f}")
    (OUT_DIR / "assembly_3d.json").write_text(json.dumps(results, indent=2))
    print(f"\n[assembly3d] wrote {OUT_DIR / 'assembly_3d.json'}")
    # per-halo table (aligned halos) so the notebook can plot residual-vs-structure directly
    np.savez_compressed(OUT_DIR / "assembly_3d_perhalo.npz", **{k: np.asarray(v) for k, v in data.items()})
    print(f"[assembly3d] wrote {OUT_DIR / 'assembly_3d_perhalo.npz'}  ({data['logM'].size} halos)")
    _plot(data, results, FIG_DIR / "assembly_3d")


def _merge(dicts: list[dict]) -> dict:
    dicts = [d for d in dicts if d]
    return {k: (np.concatenate([d[k] for d in dicts]) if dicts else np.array([])) for k in PART_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Assembly (DMO 3D structure) vs baryon scatter, CV halos.")
    ap.add_argument("--cv-max", type=int, default=27)
    ap.add_argument("--m200-min", type=float, default=1e13)
    ap.add_argument("--match-tol-kpc", type=float, default=300.0)
    ap.add_argument("--no-formation", action="store_true", help="skip formation-time tracing")
    ap.add_argument("--snap-min", type=int, default=33, help="earliest snapshot to trace back to")
    ap.add_argument("--form-tol-kpc", type=float, default=400.0, help="progenitor search radius per step")
    # parallel / map-reduce controls
    ap.add_argument("--mpi", action="store_true", help="distribute sims over MPI ranks (mpi4py)")
    ap.add_argument("--nproc", type=int, default=1, help="in-process worker pool over sims")
    ap.add_argument("--only-sim", type=int, default=None, help="compute one sim -> partial npz (disBatch task)")
    ap.add_argument("--reduce", action="store_true", help="combine partials in --partial-dir and analyze")
    ap.add_argument("--partial-dir", default="outputs/scatter_diagnostics/assembly_parts")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdir = pathlib.Path(args.partial_dir)

    # --- MPI: round-robin sims across ranks, gather to rank 0, reduce there ---
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD; rank, size = comm.Get_rank(), comm.Get_size()
        mine = [process_sim(i, args) for i in range(args.cv_max) if i % size == rank]
        gathered = comm.gather(mine, root=0)
        if rank == 0:
            parts = [d for sub in gathered for d in sub]
            analyze(_merge(parts))
        return

    # --- task-parallel modes (disBatch / job array): one sim per task, then reduce ---
    if args.only_sim is not None:
        pdir.mkdir(parents=True, exist_ok=True)
        d = process_sim(args.only_sim, args)
        if d is not None:
            np.savez_compressed(pdir / f"part_{args.only_sim:03d}.npz", **d)
            print(f"[assembly3d] wrote {pdir / f'part_{args.only_sim:03d}.npz'}")
        return
    if args.reduce:
        parts = [dict(np.load(p)) for p in sorted(glob.glob(f"{pdir}/part_*.npz"))]
        print(f"[assembly3d] reducing {len(parts)} partials from {pdir}")
        analyze(_merge(parts)); return

    # --- single-process or in-process pool ---
    if args.nproc > 1:
        from multiprocessing import Pool
        with Pool(args.nproc) as pool:
            parts = pool.starmap(process_sim, [(i, args) for i in range(args.cv_max)])
    else:
        parts = [process_sim(i, args) for i in range(args.cv_max)]
    analyze(_merge(parts))


def _plot(data, results, stub):
    fig, axs = plt.subplots(len(FOCUS), len(ASSEMBLY_PROPS),
                            figsize=(3.0*len(ASSEMBLY_PROPS), 2.7*len(FOCUS)), squeeze=False)
    for r, o in enumerate(FOCUS):
        res = residual(data[f"truth_{o}"], data["logM"], o in LOG_FOCUS)
        for cc, p in enumerate(ASSEMBLY_PROPS):
            ax = axs[r][cc]; ax.scatter(data[p], res, s=6, alpha=0.4, color="#455A64")
            rho = results["obs"][o]["truth"]["spearman"][p]["rho"]
            ax.set_title(f"{o} vs {p}: ρ={rho:+.2f}", fontsize=9)
            if cc == 0: ax.set_ylabel(f"{o} residual (truth)")
            if r == len(FOCUS)-1: ax.set_xlabel(p)
    fig.suptitle(f"Assembly attribution: baryon residual vs DMO 3D structure "
                 f"({results['n_halos']} CV halos)", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stub.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight"); print(f"  saved {stub.with_suffix('.'+ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
