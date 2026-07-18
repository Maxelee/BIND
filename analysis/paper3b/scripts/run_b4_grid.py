"""WP-B4 grid driver: matched mock measurement over the atlas (MPI over runs).

Work units = atlas runs: the 60 twobound feedback runs + the bind fiducial +
the TNG300-hydro truth lightcone (R2 selection validation). Per unit, per
(realization, noise seed) patch, the full frozen chain of
`mocks.measure.measure_mock_patch` (survey-realistic DES n(z) weighting +
shape noise + PEAK_DEFINITION smoothing/peaks/nu + ACT beam + the imported B2
CAP operators). Output: one npz per unit under <out>/grid/ with the per-patch
tables + pooled summary.

MPI: ranks stride the unit list (each unit is single-rank — the per-unit
ensemble loop is the parallel grain). Serial run = same code path, unit list
unstrided. Reproducible: every patch's rng is seeded (seed0, unit_index,
realization, noise_seed); re-running any unit standalone reproduces bit-
identical tables.

Launch (Max submits; see run_b4_grid.sh):
    mpirun -np $SLURM_NTASKS python -u -m analysis.paper3b.scripts.run_b4_grid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from analysis.paper3b.mocks.beam import ACT_DR6_YMAP_FWHM_ARCMIN, BeamConfig
from analysis.paper3b.mocks.measure import MockEnsembleAccumulator, measure_mock_patch
from analysis.paper3b.mocks.nz import SourcePlaneWeighting
from analysis.paper3b.mocks.patch import PatchGeometry
from analysis.paper3b.mocks.shape_noise import (
    DESY3_NEFF_TOTAL_ARCMIN2,
    DESY3_SIGMA_E,
    ShapeNoiseConfig,
)
from analysis.paper3b.mocks.smoothing import FIDUCIAL_SMOOTHING_ARCMIN, SmoothingConfig
from analysis.paper3b.mocks.transfer import load_transfer
from analysis.paper3b.stack.mpiutil import mpi_comm

RUNS_ROOT = Path("/mnt/home/mlee1/ceph/bind_science/runs")
DEFAULT_OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")
DEFAULT_N_SEEDS = 8
DEFAULT_SEED0 = 20260717


def unit_list(categories) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for cat in categories:
        base = RUNS_ROOT / cat
        runs = sorted(p.name for p in base.iterdir()
                      if p.is_dir() and (p / "kappa_maps.npz").exists()
                      and (p / "y_maps.npz").exists())
        units += [(cat, r) for r in runs]
    return units


def load_weighting(weights_npz: Path, scheme: str) -> SourcePlaneWeighting:
    d = np.load(weights_npz)
    key = {"default": "w_default", "variant8": "w_variant8", "plain": "w_plain"}[scheme]
    return SourcePlaneWeighting(np.asarray(d["source_z"], dtype=float),
                                np.asarray(d[key], dtype=float))


def process_unit(cat: str, run: str, weighting: SourcePlaneWeighting,
                 n_seeds: int, seed0: int, unit_idx: int, out_dir: Path,
                 scheme: str, transfer=None, transfer_name: str = "none",
                 quantize_arcmin: float | None = None) -> dict:
    t0 = time.time()
    rd = RUNS_ROOT / cat / run
    kz = np.load(rd / "kappa_maps.npz")
    kappa = kz["kappa"]                                   # (n_real, K, n, n) f32
    source_z = np.asarray(kz["source_redshifts"], dtype=float)
    if not np.allclose(source_z, weighting.source_z):
        raise RuntimeError(f"{cat}/{run}: source planes {source_z} != weighting "
                           f"{weighting.source_z}")
    y_total = np.load(rd / "y_maps.npz")["y"][:, -1]      # (n_real, n, n) deepest plane
    n_real, _, npix = kappa.shape[:3]
    geom = PatchGeometry(fov_deg=float(kz["fov_deg"]), npix=int(npix))

    noise_cfg = ShapeNoiseConfig(DESY3_NEFF_TOTAL_ARCMIN2, geom.pixel_area_arcmin2(),
                                 DESY3_SIGMA_E)
    smooth_cfg = SmoothingConfig(FIDUCIAL_SMOOTHING_ARCMIN, geom.arcmin_per_pixel(), "wrap")
    beam_cfg = BeamConfig(ACT_DR6_YMAP_FWHM_ARCMIN, geom.arcmin_per_pixel(), "wrap")

    acc = MockEnsembleAccumulator()
    real_idx, seed_idx = [], []
    for r in range(n_real):
        planes = kappa[r].astype(np.float64)
        ymap = y_total[r].astype(np.float64)
        for s in range(n_seeds):
            rng = np.random.default_rng((seed0, unit_idx, r, s))
            acc.add(measure_mock_patch(planes, ymap, weighting, noise_cfg,
                                       smooth_cfg, beam_cfg, rng, geom=geom,
                                       transfer=transfer,
                                       quantize_arcmin=quantize_arcmin))
            real_idx.append(r)
            seed_idx.append(s)

    params_file = rd / "params.npy"
    params = np.load(params_file) if params_file.exists() else np.array([])
    summ = acc.summary()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{cat}_{run}.npz",
             category=cat, run=run, params=params,
             transfer=transfer_name,
             quantize_arcmin=(0.0 if quantize_arcmin is None else quantize_arcmin),
             weight_scheme=scheme, weights=weighting.weights,
             n_seeds=n_seeds, seed0=seed0, unit_idx=unit_idx,
             neff_arcmin2=DESY3_NEFF_TOTAL_ARCMIN2, sigma_e=DESY3_SIGMA_E,
             beam_fwhm_arcmin=ACT_DR6_YMAP_FWHM_ARCMIN,
             smoothing_arcmin=FIDUCIAL_SMOOTHING_ARCMIN,
             real_idx=np.asarray(real_idx), seed_idx=np.asarray(seed_idx),
             y_mean=summ["y_mean"], y_mc_err=summ["y_mc_err"],
             n_per_bin=summ["n_per_bin"],
             **acc.arrays())
    return {"unit": f"{cat}/{run}", "minutes": (time.time() - t0) / 60.0,
            "n_patches": len(real_idx),
            "n_peaks": int(np.asarray(acc.counts).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="+", default=["twobound", "bind", "truth"])
    ap.add_argument("--weights", default=str(DEFAULT_OUT / "weights_desy3.npz"))
    ap.add_argument("--weight-scheme", default="default",
                    choices=["default", "variant8", "plain"])
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--seed0", type=int, default=DEFAULT_SEED0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--transfer", default="none",
                    choices=["none", "wiener", "glimpse"],
                    help="reconstruction transfer T(ell) applied to noisy mock "
                         "kappa (B5 design decision); 'none' = intrinsic "
                         "convention (the pre-decision grid)")
    ap.add_argument("--transfer-npz", default=str(DEFAULT_OUT / "transfer_desy3.npz"),
                    help="persisted output of build_b5_transfer.py")
    ap.add_argument("--quantize-nside1024", action="store_true",
                    help="snap mock peak positions to the Nside=1024 pixel "
                         "pitch (3.435') — the DATA side's position "
                         "quantization (B5 matched-convention fix); output "
                         "dir gains suffix _q1024")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    comm = mpi_comm()
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)

    weights_npz = Path(args.weights)
    if not weights_npz.exists():
        raise SystemExit(f"weights bundle missing: {weights_npz} — run "
                         "build_b4_weights.py first (no silently recomputed weights)")
    weighting = load_weighting(weights_npz, args.weight_scheme)

    transfer = None
    if args.transfer != "none":
        transfer = load_transfer(args.transfer_npz, args.transfer)

    from analysis.paper3b.mocks.patch import HEALPIX_NSIDE1024_PIX_ARCMIN
    quant = HEALPIX_NSIDE1024_PIX_ARCMIN if args.quantize_nside1024 else None

    units = unit_list(args.categories)
    suffix = "" if args.transfer == "none" else f"_tf{args.transfer}"
    suffix += "_q1024" if quant is not None else ""
    suffix += "" if args.weight_scheme == "default" else f"_{args.weight_scheme}"
    out_dir = Path(args.out) / f"grid{suffix}"
    if rank == 0:
        print(f"[b4-grid] {len(units)} units, {size} ranks, scheme={args.weight_scheme}, "
              f"transfer={args.transfer}, n_seeds={args.n_seeds}, out={out_dir}", flush=True)

    for u in range(rank, len(units), size):
        cat, run = units[u]
        if not args.overwrite and (out_dir / f"{cat}_{run}.npz").exists():
            print(f"[rank {rank}] skip {cat}/{run} (exists)", flush=True)
            continue
        info = process_unit(cat, run, weighting, args.n_seeds, args.seed0, u,
                            out_dir, args.weight_scheme,
                            transfer=transfer, transfer_name=args.transfer,
                            quantize_arcmin=quant)
        print(f"[rank {rank}] {info['unit']}: {info['n_patches']} patches, "
              f"{info['n_peaks']} peaks, {info['minutes']:.1f} min", flush=True)

    if comm is not None:
        comm.Barrier()
    if rank == 0:
        print("[b4-grid] all units done", flush=True)


if __name__ == "__main__":
    main()
