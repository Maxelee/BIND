"""Sobol map of the self-similar-residual -> matter-suppression calibration.

Regenerates the SAME ~1154 CV halos across a Sobol grid in the 30 astrophysical
parameters (cosmology held at the CV fiducial), so the
``Delta_SS = log10(Y200/Y200^SS)`` -> ``P_hydro/P_DMO(k~10)`` relation
(tsz_wl_calibration.ipynb, ``fig_money``) can be refit at every grid point and
its fit parameters correlated with the feedback knobs.

Design points are deterministic from --seed (Sobol scramble), so every SLURM
array task recomputes the identical 256-point design and processes only its
``--chunk_id`` slice -- no manifest/lock needed.  Per-halo flow-matching noise
is held fixed across design points (common random numbers) by re-seeding torch
before each generation, which isolates the parameter response.

Outputs (default ``/mnt/home/mlee1/ceph/sobol_ss_cv``):
  design.npz                      shared design metadata
  maps/gen_design{d:04d}.npz      (N_halo, 7, 128, 128) generated maps  [DM,Gas,Stars,Y,T,S,P]
  obs/cube_design{d:04d}.npz      reduced per-halo observables for design d
  cube.npz                        (--reduce) stacked obs (N_design, N_halo, n_obs)

Usage
-----
    # one array task (slice chunk_id of n_chunks):
    python sobol_ss_generation.py --n_chunks 16 --chunk_id 0
    # local smoke test on a few explicit design ids:
    python sobol_ss_generation.py --design_ids 0,1,128 --out_root /tmp/sobol_smoke
    # reduce after the array finishes:
    python sobol_ss_generation.py --reduce
"""

from __future__ import annotations

import argparse
import io
import math
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from data import NormStats, N_THERMO  # noqa: F401  (N_THERMO documents the 4 thermo channels)
from metrics import power_spectrum_2d
from test_suite.artifacts import load_halo_cutouts
from test_suite.pipeline import generate_halo_patches
from test_suite.runner import load_model_bundle
from test_suite.schemas import RunConfig

# ----------------------------------------------------------------------------- geometry
BOX_SIZE = 50.0
N_PIX_FULL = 1024
PATCH_PIX = 128
PATCH_BOX = BOX_SIZE * PATCH_PIX / N_PIX_FULL          # 6.25 Mpc/h
PIX_MPC = PATCH_BOX / PATCH_PIX
PIX_KPC = PIX_MPC * 1000.0                             # 48.83 kpc/h per pixel (R200 is kpc/h)
PIX_AREA_MPC2 = PIX_MPC ** 2
K_TARGET = 10.0                                        # h/Mpc; matches fig_money default
SUPP_BAND = (0.1, 0.5)                                 # r/R200 band for the profile-ratio metric

# Cosmological parameter indices in the 35-vector (held fixed at CV fiducial):
# 0 Omega_m, 1 sigma8, 6 Omega_b, 7 h, 8 n_s.  The other 30 are astrophysical.
COSMO_IDX = (0, 1, 6, 7, 8)
ASTRO_IDX = np.array([i for i in range(35) if i not in COSMO_IDX], dtype=int)  # 30

NOISE_SEED = 20250528  # common-random-numbers seed for the per-halo flow-matching noise

CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
SNAP = "snap_090"
MASS_TAG = "mass_threshold_1p000e13"
RUN_DIR_DEFAULT = Path("/mnt/home/mlee1/ceph/fm_runs/fm_thermo")
CKPT_DEFAULT = RUN_DIR_DEFAULT / "checkpoints/kept/keep_epoch064_ema.ckpt"
OUT_ROOT_DEFAULT = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
SB35_CSV = Path("/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv")

OBS_NAMES = ["Y200", "T", "S", "P", "f_gas", "m_gen", "supp_k10", "supp_prof"]

# aperture / radial grids (replicate tsz_wl_calibration.ipynb)
_yy, _xx = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
_RR = np.sqrt((_xx - PATCH_PIX / 2.0) ** 2 + (_yy - PATCH_PIX / 2.0) ** 2)  # aperture (center 64.0)
# Radial-profile integer-radius map, centered at (N-1)/2 to match the notebook's
# radial_profile_2d, precomputed once for speed.
_cP = (PATCH_PIX - 1) / 2.0
_R_INT = np.sqrt((_xx - _cP) ** 2 + (_yy - _cP) ** 2).astype(int)
_R_MAX = int(_R_INT.max())
_R_AXIS = np.arange(_R_MAX + 1)
RAD_GRID = np.linspace(0.05, 1.5, 30)


# ----------------------------------------------------------------------------- design
def fiducial_params() -> np.ndarray:
    """CV fiducial 35-vector (physical units) from the SB35 min/max table."""
    import pandas as pd
    meta = pd.read_csv(SB35_CSV)
    return meta["FiducialVal"].to_numpy(dtype=np.float64)


def build_design(seed: int, n_design: int, ns: NormStats) -> dict:
    """Sobol design over the 30 astro params, inverted through the model's
    normalized box (log-flag aware) to physical units.  Cosmology fixed."""
    sampler = qmc.Sobol(d=len(ASTRO_IDX), scramble=True, seed=seed)
    if n_design & (n_design - 1) == 0:                 # power of two -> balanced
        u = sampler.random_base2(int(round(math.log2(n_design))))
    else:
        u = sampler.random(n_design)
    u = u[:n_design]                                   # (n_design, 30) in [0,1]

    pmin = np.asarray(ns.param_min, float)[ASTRO_IDX]  # log10 where log_flag==1
    pmax = np.asarray(ns.param_max, float)[ASTRO_IDX]
    lf = np.asarray(ns.param_log_flag, float)[ASTRO_IDX]
    box = pmin[None] + u * (pmax - pmin)[None]         # normalized-box coordinate
    astro_phys = box.copy()                            # (n_design, 30) physical
    log_cols = lf == 1                                 # exponentiate only log-flag params
    astro_phys[:, log_cols] = 10.0 ** box[:, log_cols]

    fid = fiducial_params()
    design_phys = np.tile(fid, (n_design, 1))          # (n_design, 35)
    design_phys[:, ASTRO_IDX] = astro_phys

    import pandas as pd
    names = pd.read_csv(SB35_CSV)["ParamName"].tolist()
    return {
        "design_phys": design_phys.astype(np.float64),         # (n_design, 35)
        "design_astro_phys": astro_phys.astype(np.float64),    # (n_design, 30)
        "design_norm": u.astype(np.float64),                   # (n_design, 30) Sobol [0,1]
        "astro_idx": ASTRO_IDX,
        "param_names": np.array(names, dtype=object),
        "astro_names": np.array([names[i] for i in ASTRO_IDX], dtype=object),
        "fiducial": fid,
        "seed": seed,
    }


# ----------------------------------------------------------------------------- CV halos
def load_cv_halos() -> dict:
    """Load all CV halo cutouts + catalog scalars, concatenated across sims.

    Returns cutouts (list of dicts), and per-halo M200/R200/sim_id plus the DMO
    condition patch stack used for the P(k) suppression.
    """
    cutouts: list[dict] = []
    masses, radii, sim_ids, dmo = [], [], [], []
    for sd in sorted(p for p in CV_ROOT.iterdir() if p.is_dir()):
        base = sd / SNAP / MASS_TAG
        cut_path, cat_path = base / "halo_cutouts.npz", base / "halo_catalog.npz"
        if not (cut_path.exists() and cat_path.exists()):
            print(f"[skip] {sd.name}: missing cutouts/catalog")
            continue
        cat = np.load(cat_path)
        if "radii" not in cat.files or len(cat["radii"]) == 0:
            print(f"[skip] {sd.name}: missing/empty radii")
            continue
        hc = load_halo_cutouts(cut_path)
        if len(hc) != len(cat["masses"]):
            print(f"[skip] {sd.name}: cutout/catalog count mismatch "
                  f"({len(hc)} vs {len(cat['masses'])})")
            continue
        cutouts.extend(hc)
        masses.append(np.asarray(cat["masses"], float))
        radii.append(np.asarray(cat["radii"], float))
        dmo.extend(h["condition"] for h in hc)
        sim_ids.extend([sd.name] * len(hc))
    out = {
        "cutouts": cutouts,
        "M200": np.concatenate(masses),
        "R200": np.concatenate(radii),
        "sim_id": np.array(sim_ids, dtype=object),
        "dmo": np.stack(dmo).astype(np.float32),
    }
    print(f"Loaded {len(cutouts)} CV halos from "
          f"{len(set(sim_ids))} sims")
    return out


# ----------------------------------------------------------------------------- reductions
def _pk(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with redirect_stdout(io.StringIO()):
        return power_spectrum_2d(field, box_size=PATCH_BOX)


def _radial_profile(arr: np.ndarray) -> np.ndarray:
    """Azimuthal mean per integer-pixel radius (center (N-1)/2), using the
    precomputed _R_INT map.  Returns profile aligned with _R_AXIS."""
    flat = arr.ravel()
    m = np.isfinite(flat)
    sums = np.bincount(_R_INT.ravel()[m], weights=flat[m], minlength=_R_MAX + 1)
    cnt = np.bincount(_R_INT.ravel()[m], minlength=_R_MAX + 1)
    prof = np.full(_R_MAX + 1, np.nan)
    good = cnt > 0
    prof[good] = sums[good] / cnt[good]
    return prof


def _supp_prof(tot: np.ndarray, dmo_prof: np.ndarray, r200_pix: float) -> float:
    """Profile-ratio suppression over SUPP_BAND in r/R200.  dmo_prof is the
    precomputed DMO radial profile for this halo (fixed across designs)."""
    pn = _radial_profile(tot)
    x = _R_AXIS / max(r200_pix, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pn / dmo_prof
    ok = np.isfinite(ratio) & (dmo_prof > 0)
    rr = np.interp(RAD_GRID, x[ok], ratio[ok], left=np.nan, right=np.nan)
    band = (RAD_GRID >= SUPP_BAND[0]) & (RAD_GRID <= SUPP_BAND[1])
    return float(np.nanmean(rr[band]))


def reduce_design(gen: np.ndarray, halos: dict, pk_dmo_k10: np.ndarray,
                  dmo_profiles: np.ndarray, k_idx: int) -> np.ndarray:
    """Per-halo observables for one design.  gen: (N,7,128,128) physical units.
    pk_dmo_k10 / dmo_profiles are the fixed DMO P(k~10) and radial profiles."""
    n = gen.shape[0]
    obs = np.full((n, len(OBS_NAMES)), np.nan, dtype=np.float64)
    for i in range(n):
        g = gen[i]
        r200_pix = max(halos["R200"][i] / PIX_KPC, 1.0)
        ap = _RR <= r200_pix
        tot = g[0] + g[1] + g[2]
        gas = g[1]
        w = gas[ap]
        sw = float(w.sum())
        mtot = float(tot[ap].sum())
        obs[i, 0] = float(g[3][ap].sum()) * PIX_AREA_MPC2          # Y200
        if sw > 0:
            obs[i, 1] = float((g[4][ap] * w).sum() / sw)           # T
            obs[i, 2] = float((g[5][ap] * w).sum() / sw)           # S
            obs[i, 3] = float((g[6][ap] * w).sum() / sw)           # P
        obs[i, 4] = sw / mtot if mtot > 0 else np.nan              # f_gas
        obs[i, 5] = mtot                                           # m_gen
        _, pk_gen = _pk(tot)
        obs[i, 6] = (pk_gen[k_idx] / pk_dmo_k10[i]
                     if pk_dmo_k10[i] > 0 else np.nan)             # supp_k10
        obs[i, 7] = _supp_prof(tot, dmo_profiles[i], r200_pix)     # supp_prof
    return obs


# ----------------------------------------------------------------------------- driver
def select_design_ids(args, n_design: int) -> list[int]:
    if args.design_ids:
        return [int(x) for x in args.design_ids.split(",") if x != ""]
    ids = list(range(n_design))
    return ids[args.chunk_id::args.n_chunks]


def run_campaign(args) -> None:
    out_root = Path(args.out_root)
    (out_root / "maps").mkdir(parents=True, exist_ok=True)
    (out_root / "obs").mkdir(parents=True, exist_ok=True)

    rc = RunConfig(run_dir=Path(args.run_dir), checkpoint_path=Path(args.checkpoint_path),
                   output_root=out_root, model_name="sobol_ss",
                   n_steps=args.n_steps, batch_size=args.batch_size,
                   use_amp=True, device="cuda")
    ns, fm, dev, pidx, no_ls, predict_thermo = load_model_bundle(rc)
    assert predict_thermo, "checkpoint must predict thermo (need compton_y for Y200)"

    design = build_design(args.seed, args.n_design, ns)
    design_path = out_root / "design.npz"
    if not design_path.exists():
        tmp = out_root / f"design.tmp{os.getpid()}.npz"
        np.savez(tmp, **design)
        os.replace(tmp, design_path)                  # atomic; harmless if another task raced
    print(f"Design: {design['design_phys'].shape[0]} points x {len(ASTRO_IDX)} astro params "
          f"(seed={args.seed})")

    halos = load_cv_halos()
    # Precompute the DMO P(k) at k~10 once (fixed across design points).
    k_ref, _ = _pk(halos["dmo"][0])
    k_idx = int(np.argmin(np.abs(k_ref - K_TARGET)))
    print(f"k bin nearest {K_TARGET}: idx {k_idx}, k={k_ref[k_idx]:.3f} h/Mpc")
    pk_dmo_k10 = np.array([_pk(d)[1][k_idx] for d in halos["dmo"]], dtype=np.float64)
    dmo_profiles = np.stack([_radial_profile(d) for d in halos["dmo"]])  # fixed across designs

    design_ids = select_design_ids(args, args.n_design)
    print(f"This task processes design ids: {design_ids[:6]}{'...' if len(design_ids) > 6 else ''} "
          f"({len(design_ids)} total)")

    for d in design_ids:
        obs_path = out_root / "obs" / f"cube_design{d:04d}.npz"
        maps_path = out_root / "maps" / f"gen_design{d:04d}.npz"
        done = obs_path.exists() and (args.no_maps or maps_path.exists())
        if done and not args.regenerate:
            print(f"[design {d:04d}] exists, skip")
            continue
        params_phys = design["design_phys"][d].astype(np.float32)
        torch.manual_seed(NOISE_SEED)                  # common random numbers across designs
        gen = generate_halo_patches(
            halos["cutouts"], ns, params_phys, fm, dev,
            n_steps=args.n_steps, batch_size=args.batch_size, use_amp=True,
            param_indices=pidx, no_large_scale=no_ls)
        obs = reduce_design(gen, halos, pk_dmo_k10, dmo_profiles, k_idx)

        np.savez(obs_path, obs=obs, obs_names=np.array(OBS_NAMES, dtype=object),
                 M200=halos["M200"], R200=halos["R200"], sim_id=halos["sim_id"],
                 design_id=d, params_phys=params_phys,
                 design_norm=design["design_norm"][d])
        msg = f"[design {d:04d}] obs {obs.shape} -> {obs_path.name}"
        if not args.no_maps:
            np.savez(maps_path, generated=gen.astype(np.float16 if args.fp16 else np.float32))
            msg += f"; maps {gen.shape} -> {maps_path.name}"
        msg += f"  (med supp_k10={np.nanmedian(obs[:, 6]):.3f})"
        print(msg)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_reduce(args) -> None:
    out_root = Path(args.out_root)
    shards = sorted((out_root / "obs").glob("cube_design*.npz"))
    if not shards:
        raise SystemExit(f"no obs shards in {out_root/'obs'}")
    ids, rows = [], []
    M200 = R200 = sim_id = obs_names = None
    for sp in shards:
        z = np.load(sp, allow_pickle=True)
        ids.append(int(z["design_id"]))
        rows.append(z["obs"])
        if M200 is None:
            M200, R200, sim_id = z["M200"], z["R200"], z["sim_id"]
            obs_names = z["obs_names"]
    order = np.argsort(ids)
    cube = np.stack([rows[i] for i in order])          # (n_present, N_halo, n_obs)
    design = np.load(out_root / "design.npz", allow_pickle=True)
    out = out_root / "cube.npz"
    np.savez(out, obs=cube, design_id=np.array(ids)[order], obs_names=obs_names,
             M200=M200, R200=R200, sim_id=sim_id,
             design_phys=design["design_phys"], design_astro_phys=design["design_astro_phys"],
             design_norm=design["design_norm"], astro_idx=design["astro_idx"],
             astro_names=design["astro_names"], param_names=design["param_names"])
    print(f"Reduced {cube.shape[0]} designs -> {out}  cube {cube.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--run_dir", default=str(RUN_DIR_DEFAULT))
    ap.add_argument("--checkpoint_path", default=str(CKPT_DEFAULT))
    ap.add_argument("--seed", type=int, default=12345, help="Sobol scramble seed")
    ap.add_argument("--n_design", type=int, default=256)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--design_ids", default="", help="explicit comma list (overrides chunking)")
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="store generated maps as float16")
    ap.add_argument("--no_maps", action="store_true",
                    help="skip saving full maps (obs cube only) -- fast/light preview")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--reduce", action="store_true", help="concatenate obs shards into cube.npz")
    args = ap.parse_args()

    if args.reduce:
        run_reduce(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
