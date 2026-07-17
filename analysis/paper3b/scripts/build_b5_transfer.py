"""Derive the B5 reconstruction transfer T(ell) — data map spectra vs mock (WP-B4/B5).

Serial, ~3 min. Implements the B5 design decision (bind-paper3-plans,
`projectB/wp5-inference-decomposition/DESIGN_DECISION_FILTER.md`):

  T_variant(ell) = sqrt( C_ell^{variant map, common footprint}
                         / C_ell^{bind-fiducial kappa_eff + DES shape noise} )

Numerator: pseudo-C_ell of the released Wiener / GLIMPSE map over the
harmonized common footprint (apodized weight; uniform f_sky_2 = <w^2>
normalization — an approximation whose overall amplitude error is irrelevant
because nu is map-normalized; only the SHAPE of T enters the measurement).
Denominator: mean flat-sky periodogram of the bind-fiducial atlas run through
the frozen n(z) weights + DES shape noise (the exact grid-chain convention).

Outputs (all under --out, default ~/ceph/paper3/B/wp4_mocks/):
  transfer_desy3.npz        — ell_/t_ for both variants + input spectra + meta
  b5_transfer_summary.json  — validation: data vs mock nu>=4 abundances
                              (unfiltered / T-filtered), provenance
  figures/b5_transfer.png   — spectra + T(ell) + validation panel

Uses ONLY the kappa maps' 2-pt statistics (public) — never the y map or the
frozen stacks (blinding-compatible; see the decision doc).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.paper3b.maps.loaders import load_des_map
from analysis.paper3b.mocks.nz import SourcePlaneWeighting
from analysis.paper3b.mocks.patch import PatchGeometry, find_peaks_flat
from analysis.paper3b.mocks.shape_noise import (
    DESY3_NEFF_TOTAL_ARCMIN2,
    DESY3_SIGMA_E,
    ShapeNoiseConfig,
    noise_map,
)
from analysis.paper3b.mocks.smoothing import FIDUCIAL_SMOOTHING_ARCMIN, SmoothingConfig, smooth_flat_sky
from analysis.paper3b.mocks.transfer import (
    ELL_MAX_NSIDE1024,
    ELL_MIN,
    build_empirical_transfer,
    flat_sky_power,
)
from analysis.paper3b.scripts.run_b4_grid import RUNS_ROOT, load_weighting

DEFAULT_OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks")
FOOTPRINT_NPZ = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps/common_footprint_nside1024.npz")
PEAKS_DIR = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")
SEED0 = 20260717
NOISE_STREAM = 999           # distinct from the grid's (seed0, unit, r, s) streams
DEG2_SKY = 4.0 * np.pi * (180.0 / np.pi) ** 2


def data_pseudo_cl(variant: str, weight: np.ndarray, lmax: int) -> np.ndarray:
    import healpy as hp

    kappa = load_des_map(variant)                     # off-footprint -> 0
    cl = hp.anafast(np.asarray(kappa, float) * weight, lmax=lmax)
    return cl / np.mean(weight ** 2)                  # uniform pseudo-Cl norm


def mock_mean_power(kappa: np.ndarray, weighting: SourcePlaneWeighting,
                    noise_cfg: ShapeNoiseConfig, geom: PatchGeometry,
                    n_seeds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    cls = []
    for r in range(kappa.shape[0]):
        keff = weighting.effective_map(kappa[r].astype(np.float64), plane_axis=0)
        for s in range(n_seeds):
            rng = np.random.default_rng((SEED0, NOISE_STREAM, r, s))
            ell, cl, _ = flat_sky_power(keff + noise_map(keff.shape, noise_cfg, rng=rng),
                                        geom.fov_deg)
            cls.append(cl)
    return ell, np.nanmean(cls, axis=0)


def mock_abundance_nu4(kappa: np.ndarray, weighting: SourcePlaneWeighting,
                       noise_cfg: ShapeNoiseConfig, smooth_cfg: SmoothingConfig,
                       geom: PatchGeometry, transfer=None,
                       n_real: int = 12, n_seeds: int = 2) -> float:
    """Peaks-only chain (no y stacking) -> nu>=4 peaks per deg^2."""
    n4, area = 0, 0.0
    for r in range(min(n_real, kappa.shape[0])):
        keff = weighting.effective_map(kappa[r].astype(np.float64), plane_axis=0)
        for s in range(n_seeds):
            rng = np.random.default_rng((SEED0, NOISE_STREAM + 1, r, s))
            kn = keff + noise_map(keff.shape, noise_cfg, rng=rng)
            if transfer is not None:
                kn = transfer.apply(kn, geom.fov_deg)
            ksm = smooth_flat_sky(kn, smooth_cfg)
            nu = (ksm - ksm.mean()) / (ksm.std() + 1e-30)
            n4 += int((nu[find_peaks_flat(ksm)] >= 4.0).sum())
            area += geom.fov_deg ** 2
    return n4 / area


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--weights", default=str(DEFAULT_OUT / "weights_desy3.npz"))
    ap.add_argument("--mock-run", default="bind/run_0000",
                    help="atlas run for the denominator spectrum (grid center)")
    ap.add_argument("--n-seeds", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    lmax = int(ELL_MAX_NSIDE1024)

    t0 = time.time()
    fp = np.load(FOOTPRINT_NPZ, allow_pickle=True)
    weight = np.asarray(fp["weight"], dtype=np.float64)
    area_deg2 = float(np.asarray(fp["binary"], bool).mean() * DEG2_SKY)

    weighting = load_weighting(Path(args.weights), "default")
    kz = np.load(RUNS_ROOT / args.mock_run / "kappa_maps.npz")
    kappa = kz["kappa"]
    geom = PatchGeometry(fov_deg=float(kz["fov_deg"]), npix=int(kappa.shape[-1]))
    noise_cfg = ShapeNoiseConfig(DESY3_NEFF_TOTAL_ARCMIN2, geom.pixel_area_arcmin2(),
                                 DESY3_SIGMA_E)
    smooth_cfg = SmoothingConfig(FIDUCIAL_SMOOTHING_ARCMIN, geom.arcmin_per_pixel(), "wrap")

    print(f"[b5-transfer] mock spectrum from {args.mock_run} "
          f"({kappa.shape[0]} real x {args.n_seeds} seeds)", flush=True)
    ell_mock, cl_mock = mock_mean_power(kappa, weighting, noise_cfg, geom,
                                        n_seeds=args.n_seeds)

    save: dict = {"ell_zero": ELL_MAX_NSIDE1024, "ell_mock": ell_mock,
                  "cl_mock": cl_mock, "mock_run": args.mock_run,
                  "weights_scheme": "default", "f_sky2": float(np.mean(weight ** 2)),
                  "footprint_npz": str(FOOTPRINT_NPZ)}
    summary: dict = {"area_deg2_common_binary": area_deg2,
                     "mock_run": args.mock_run,
                     "abundance_nu4_per_deg2": {}}

    print("[b5-transfer] unfiltered mock nu>=4 abundance...", flush=True)
    ab_raw = mock_abundance_nu4(kappa, weighting, noise_cfg, smooth_cfg, geom)
    summary["abundance_nu4_per_deg2"]["mock_unfiltered"] = ab_raw

    transfers = {}
    for variant in ("wiener", "glimpse"):
        print(f"[b5-transfer] data pseudo-Cl: {variant}", flush=True)
        cl_data = data_pseudo_cl(variant, weight, lmax)
        ell_data = np.arange(lmax + 1, dtype=float)
        tf = build_empirical_transfer(ell_data[2:], cl_data[2:], ell_mock, cl_mock,
                                      ell_min=ELL_MIN, ell_max=ELL_MAX_NSIDE1024)
        transfers[variant] = tf
        save[f"ell_{variant}"] = tf.ell
        save[f"t_{variant}"] = tf.t
        save[f"cl_data_{variant}"] = cl_data

        pk = np.load(PEAKS_DIR / f"peaks_{variant}_sm2am.npz", allow_pickle=True)
        n4_data = int((np.asarray(pk["nu"]) >= 4.0).sum())
        ab_filt = mock_abundance_nu4(kappa, weighting, noise_cfg, smooth_cfg,
                                     geom, transfer=tf)
        summary["abundance_nu4_per_deg2"][f"data_{variant}"] = n4_data / area_deg2
        summary["abundance_nu4_per_deg2"][f"mock_tf{variant}"] = ab_filt
        summary[f"n_peaks_nu4_data_{variant}"] = n4_data

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "transfer_desy3.npz", **save)

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ok = np.isfinite(cl_mock)
    ax.loglog(ell_mock[ok], cl_mock[ok], "k-", label="mock $\\kappa_{\\rm eff}$+noise")
    for variant, color in (("wiener", "C0"), ("glimpse", "C1")):
        cd = save[f"cl_data_{variant}"]
        ax.loglog(np.arange(len(cd))[2:], cd[2:], color=color, alpha=0.7,
                  label=f"data {variant} (pseudo)")
    ax.set_xlabel("$\\ell$"); ax.set_ylabel("$C_\\ell$"); ax.legend(fontsize=8)
    ax.set_xlim(60, 4e3); ax.set_title("input spectra")

    ax = axes[1]
    for variant, color in (("wiener", "C0"), ("glimpse", "C1")):
        tf = transfers[variant]
        ax.semilogx(tf.ell, tf.t, color=color, label=f"$T_{{\\rm {variant}}}(\\ell)$")
    ax.axhline(1.0, color="gray", lw=0.5); ax.set_ylim(0, None)
    ax.set_xlabel("$\\ell$"); ax.set_ylabel("$T(\\ell)$"); ax.legend(fontsize=8)
    ax.set_title("empirical transfer (0 above $\\ell=3071$)")

    ax = axes[2]
    ab = summary["abundance_nu4_per_deg2"]
    names = ["mock_unfiltered", "mock_tfwiener", "data_wiener",
             "mock_tfglimpse", "data_glimpse"]
    vals = [ab[n] for n in names]
    ax.bar(range(len(names)), vals,
           color=["0.5", "C0", "lightsteelblue", "C1", "navajowhite"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("$\\nu\\geq4$ peaks / deg$^2$")
    ax.set_title("validation: abundance closure")
    fig.tight_layout()
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "b5_transfer.png", dpi=150)

    summary["minutes"] = (time.time() - t0) / 60.0
    with open(out / "b5_transfer_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", out / "transfer_desy3.npz")


if __name__ == "__main__":
    main()
