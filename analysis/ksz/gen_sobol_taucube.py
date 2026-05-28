"""Generate the fixed-halo Sobol τ cube for kSZ SBI.

BIND's advantage for inference: paint the *same* CV halos at many astrophysical
parameter points with cosmology fixed, turning ~100 confounded sims into
thousands of clean, paired forward evaluations.  This script realises that:

  1. Build a Sobol design over the 30 ASTRO parameters (cosmology held at the
     SB35 fiducial), using the SB35 bounds + log-flags.
  2. Load the CV halo cube-cutouts once (the DMO conditioning for ~1154 halos,
     all at fiducial cosmology — already cached by the cube test suite).
  3. For each design, paint all halos with the cube model (K stochastic draws)
     and reduce to the canonical stacked CAP τ(M) observable.

Outputs one shard npz per design range so a SLURM array can parallelise over
designs.  `--reduce` concatenates shards into a single training file for
`npe_tau.py`.

The observable here is the *projected aperture* CAP τ from the 6.25 Mpc/h cube
model — the faithful kSZ observable, not the 3D M_gas proxy used in the pilot.

Usage (one shard):
    python -m analysis.ksz.gen_sobol_taucube \\
        --run_dir /mnt/home/mlee1/ceph/fm_runs/fm_cube_two_head \\
        --checkpoint /mnt/home/mlee1/ceph/fm_runs/fm_cube_two_head/checkpoints/last.ckpt \\
        --testsuite_root /mnt/home/mlee1/ceph/fm_testsuite_cube --suite CV \\
        --n_design 4096 --design_start 0 --design_end 128 --n_draws 8 \\
        --out_dir /mnt/home/mlee1/ceph/ksz_sobol/fm_cube_two_head

Reduce:
    python -m analysis.ksz.gen_sobol_taucube --reduce \\
        --out_dir /mnt/home/mlee1/ceph/ksz_sobol/fm_cube_two_head \\
        --reduced_out analysis_physics_cache/ksz_sobol_taucube_fm_cube_two_head.npz
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from .param_meta import COSMO_IDX, load_param_meta
from .tau_utils import per_halo_tau

GAS_CH = 1


# --------------------------------------------------------------------------- #
# Sobol design over the 30 astro params (cosmology fixed)
# --------------------------------------------------------------------------- #
def build_design(n_design: int, seed: int):
    """Return (theta35 (n_design,35), astro (n_design,30), scan_idx (30,)).

    Cosmo indices held at SB35 fiducial; astro indices Sobol-sampled in the
    (log-aware) SB35 box.
    """
    from scipy.stats import qmc

    meta = load_param_meta()
    n = len(meta.names)
    scan_idx = np.array([i for i in range(n) if i not in COSMO_IDX], dtype=np.int64)
    d = len(scan_idx)

    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random(n_design)                          # (n_design, d) in [0,1]

    lo = meta.minv[scan_idx].copy()
    hi = meta.maxv[scan_idx].copy()
    logm = meta.logflag[scan_idx].astype(bool)
    astro = np.empty_like(u)
    # log-flagged dims sample uniformly in log10, matching SB35 sampling
    llo, lhi = np.log10(np.clip(lo, 1e-30, None)), np.log10(np.clip(hi, 1e-30, None))
    astro[:, logm] = 10.0 ** (llo[logm] + u[:, logm] * (lhi[logm] - llo[logm]))
    astro[:, ~logm] = lo[~logm] + u[:, ~logm] * (hi[~logm] - lo[~logm])

    theta35 = np.tile(meta.fiducial.astype(np.float64), (n_design, 1))
    theta35[:, scan_idx] = astro
    return theta35, astro, scan_idx


# --------------------------------------------------------------------------- #
# Model + halo loading
# --------------------------------------------------------------------------- #
def load_cube_model(run_dir: Path, checkpoint: Path, device):
    """Load NormStats + cube FlowMatching model; return (ns, fm, param_indices)."""
    from data import NormStats
    from train import FlowMatchingLit

    ns = NormStats.load(run_dir / "norm_stats.npz")
    model = FlowMatchingLit.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval(); model.to(device)
    _COSMO_TRAIN = [0, 1, 7, 8]   # what --exclude_cosmo_params drops (keeps Ω_b)
    n_params = int(getattr(model.hparams, "n_params", 35))
    param_indices = (
        np.array([i for i in range(35) if i not in _COSMO_TRAIN])
        if n_params < 35 else None
    )
    no_large_scale = bool(getattr(model.hparams, "no_large_scale", True))
    return ns, model.fm, param_indices, no_large_scale


def load_cv_halos(testsuite_root: Path, suite: str, halo_mass_min: float,
                  max_halos: int | None):
    """Concatenate CV halo cube-cutouts + masses across all sims (fiducial cosmo)."""
    from test_suite.artifacts import load_halo_cutouts
    from ._io import find_sim_dirs, format_mass_tag

    cutouts: list[dict] = []
    masses: list[float] = []
    for sd in find_sim_dirs(testsuite_root, suite):
        snaps = sorted(p for p in sd.iterdir() if p.is_dir() and p.name.startswith("snap_"))
        if not snaps:
            continue
        mass_dir = snaps[0] / f"mass_threshold_{format_mass_tag(halo_mass_min)}"
        cut_path = mass_dir / "halo_cutouts_cube.npz"
        cat_path = mass_dir / "halo_catalog.npz"
        if not (cut_path.exists() and cat_path.exists()):
            continue
        cuts = load_halo_cutouts(cut_path)
        cat = np.load(cat_path)
        m = cat["halo_masses"] if "halo_masses" in cat.files else cat["masses"]
        if len(cuts) != len(m):
            continue
        cutouts.extend(cuts)
        masses.extend(np.asarray(m, dtype=np.float64).tolist())
    if not cutouts:
        raise SystemExit(f"No CV cube cutouts under {testsuite_root / suite}")
    masses = np.asarray(masses, dtype=np.float64)
    if max_halos is not None and len(cutouts) > max_halos:
        cutouts = cutouts[:max_halos]
        masses = masses[:max_halos]
    return cutouts, masses


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_shard(args) -> None:
    import torch
    from test_suite.pipeline import generate_halo_patches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta35, astro, scan_idx = build_design(args.n_design, args.seed)
    d0, d1 = args.design_start, min(args.design_end, args.n_design)
    if d0 >= d1:
        raise SystemExit(f"empty design range [{d0},{d1})")

    ns, fm, param_indices, no_large_scale = load_cube_model(
        args.run_dir, args.checkpoint, device)
    cutouts, masses = load_cv_halos(args.testsuite_root, args.suite,
                                    args.halo_mass_min, args.max_halos)
    pix_size = args.patch_size_mpc_h / int(cutouts[0]["condition"].shape[-1])
    r_ap_pix = args.r_ap_mpc_h / pix_size
    edges = np.asarray(args.mass_bins, dtype=np.float64)
    nb = len(edges) - 1
    bin_idx = np.digitize(masses, edges) - 1
    print(f"[info] designs[{d0}:{d1}] of {args.n_design}; {len(cutouts)} halos; "
          f"K={args.n_draws}; {nb} mass bins; device={device.type}")

    n_loc = d1 - d0
    tau_stack = np.full((n_loc, args.n_draws, nb), np.nan, dtype=np.float32)   # mean τ / bin
    tau_scat = np.full((n_loc, args.n_draws, nb), np.nan, dtype=np.float32)    # frac scatter / bin

    for li, d in enumerate(range(d0, d1)):
        for k in range(args.n_draws):
            gen = generate_halo_patches(
                cutouts, ns, theta35[d], fm, device,
                n_steps=args.n_steps, batch_size=args.batch_size,
                use_amp=args.use_amp, param_indices=param_indices,
                no_large_scale=no_large_scale,
            )
            tau = per_halo_tau(gen[:, GAS_CH], r_ap_pix, pix_size, args.hubble,
                               estimator=args.aperture)
            for b in range(nb):
                sel = (bin_idx == b) & np.isfinite(tau)
                if sel.any():
                    m = float(np.mean(tau[sel]))
                    tau_stack[li, k, b] = m
                    tau_scat[li, k, b] = float(np.std(tau[sel]) / m) if m != 0 else np.nan
        if (li + 1) % 10 == 0 or li == n_loc - 1:
            print(f"  design {d0+li+1}/{d1} done", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard = args.out_dir / f"sobol_shard_{d0:06d}_{d1:06d}.npz"
    np.savez(
        shard,
        design_start=d0, design_end=d1, n_design=args.n_design,
        theta35=theta35[d0:d1], astro=astro[d0:d1], scan_idx=scan_idx,
        tau_stack=tau_stack, tau_scat=tau_scat,
        mass_edges=edges, n_per_bin=np.bincount(bin_idx[bin_idx >= 0], minlength=nb),
        aperture=args.aperture, r_ap_mpc_h=args.r_ap_mpc_h, n_draws=args.n_draws,
    )
    print(f"[save] {shard}")


def reduce_shards(args) -> None:
    files = sorted(glob.glob(str(args.out_dir / "sobol_shard_*.npz")))
    if not files:
        raise SystemExit(f"No shards under {args.out_dir}")
    astro, tau_stack, tau_scat, theta35 = [], [], [], []
    scan_idx = edges = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        astro.append(z["astro"]); theta35.append(z["theta35"])
        tau_stack.append(z["tau_stack"]); tau_scat.append(z["tau_scat"])
        scan_idx = z["scan_idx"]; edges = z["mass_edges"]
    astro = np.concatenate(astro); theta35 = np.concatenate(theta35)
    tau_stack = np.concatenate(tau_stack); tau_scat = np.concatenate(tau_scat)
    Path(args.reduced_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.reduced_out,
        astro=astro, theta35=theta35, scan_idx=scan_idx,
        tau_stack=tau_stack, tau_scat=tau_scat, mass_edges=edges,
    )
    print(f"[save] {args.reduced_out}  ({len(astro)} designs, "
          f"{tau_stack.shape[1]} draws, {tau_stack.shape[2]} bins)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reduce", action="store_true", help="Concatenate shards → training npz.")
    ap.add_argument("--run_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--testsuite_root", type=Path)
    ap.add_argument("--suite", default="CV")
    ap.add_argument("--halo_mass_min", type=float, default=1e13)
    ap.add_argument("--n_design", type=int, default=4096)
    ap.add_argument("--design_start", type=int, default=0)
    ap.add_argument("--design_end", type=int, default=4096)
    ap.add_argument("--n_draws", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--max_halos", type=int, default=None, help="Cap halos (smoke tests).")
    ap.add_argument("--patch_size_mpc_h", type=float, default=6.25)
    ap.add_argument("--r_ap_mpc_h", type=float, default=0.5)
    ap.add_argument("--aperture", choices=["disk", "cap"], default="cap")
    ap.add_argument("--hubble", type=float, default=0.6711)
    ap.add_argument("--mass_bins", nargs="+", type=float, default=[1e13, 3e13, 1e14, 1e15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--reduced_out", type=Path)
    args = ap.parse_args()

    if args.reduce:
        if args.reduced_out is None:
            raise SystemExit("--reduce requires --reduced_out")
        reduce_shards(args)
        return
    for req in ("run_dir", "checkpoint", "testsuite_root"):
        if getattr(args, req) is None:
            raise SystemExit(f"generation requires --{req}")
    generate_shard(args)


if __name__ == "__main__":
    main()
