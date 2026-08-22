"""One common nu axis for every threshold statistic in the Paper I figure package.

WHY. figs 4, 5, 7 and 8 all plot statistics against the peak-height variable nu, but until
now on THREE different grids: the Minkowski functionals on linspace(-3, 4, 29), peaks and
minima on linspace(-4.875, 11.875, 68), and the PDF on a +-6-sigma kappa grid with 41 bins.
Panels sat side by side on axes that were not the same axis. A first fix unified them onto
linspace(-3, 8, 45) (dnu=0.25); the author has since RULED a second, canonical convention
that supersedes it:

    NU_EDGES = arange(-3.0, 8.0, 0.5) + [8.0]     23 edges, width 0.5
    NU       = bin centres of NU_EDGES            22 centres, -2.75 .. 7.75

Motivation for 0.5-wide bins (vs the earlier 0.25): the counting statistics (peaks, minima,
PDF) then hold an integer number of realization-summed counts per bin, so each point carries
a well-defined Poisson error. All six statistics -- pdf, peak_counts, minima_counts, mf_v0,
mf_v1, mf_v2 -- share this ONE grid, for every run, so the released dataset and every figure
agree.

EDGES vs CENTRES -- the two families are not treated the same way, by design:
  * PDF / peak_counts / minima_counts are HISTOGRAMS: a pixel (or peak, or minimum) with
    S/N nu falls into bin i iff NU_EDGES[i] <= nu < NU_EDGES[i+1] (last bin closed at the
    top edge). These three are always computed with NU_EDGES (23 edges -> 22 bins), and
    the reported abscissa is the bin CENTRE, NU[i].
  * mf_v0 / mf_v1 / mf_v2 (the Minkowski functionals) are THRESHOLD FUNCTIONALS, not
    histograms: V_k(nu) is evaluated AT a threshold nu, one map-wide number per threshold,
    not a count of pixels falling in a bin. These are computed with NU itself (the 22
    centres) passed directly as evaluation points -- NU_EDGES plays no role for the MFs.
  Both families end up living on the same 22-point abscissa NU, which is the point of this
  module, but only the histogram trio actually "bins" anything.

WHAT IT DOES NOT DO. It does not overwrite anything. Per-run shards go to
`bind_sb35/nu05_shards/<run>.npz` (never replacing the earlier `nu8_shards/` or the released
`peak_counts.npz` / `nongaussian_stats.npz`), and the assembled product is a NEW dataset
file; the released `emulator_dataset_xpkfix.npz` is left exactly as it is.

A CAVEAT THAT SURVIVES THIS CHANGE. nu is a per-statistic S/N variable: the PDF is measured
on the unsmoothed map, the Minkowski functionals at 1', peaks and minima at 2'. Putting them
on one numerical grid makes the axes comparable to read, NOT physically interchangeable.
Fixing a single smoothing across all six would change the statistics' definitions and is a
separate decision.

BUILD:  sbatch papers/01_pipeline/run_nu_cache.sbatch
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"

# the one grid: 0.5-wide bins from -3 to 8 (23 edges), NU = the 22 bin centres.
# Histograms (pdf, peak_counts, minima_counts) bin on NU_EDGES; the Minkowski functionals
# are threshold functionals evaluated directly AT the centres NU (see module docstring).
# Two campaigns, two axes.  The 50-real Paper I package (default) keeps the
# original centres -2.75 .. 7.75; the N=1000 campaign standardised on centres
# -2.5 .. 8.0 (bind.inference.stats.NU_CANON), which is what every n1000
# product -- per-run peak_counts/nongaussian, the field cache and the nu shards
# -- is binned on.  Both are 22 points at width 0.5; they differ by a quarter
# bin, so mixing them would misalign every threshold statistic.  Select with
# BIND_CAMPAIGN=n1000.
if os.environ.get("BIND_CAMPAIGN", "sci50") == "n1000":
    NU = np.linspace(-2.5, 8.0, 22)                         # 22 centres, -2.5 .. 8.0
    NU_EDGES = np.concatenate([NU - 0.25, [NU[-1] + 0.25]])  # 23 edges, width 0.5
else:
    NU_EDGES = np.arange(-3.0, 8.0 + 0.5 / 2, 0.5)          # 23 edges, width 0.5
    NU = 0.5 * (NU_EDGES[:-1] + NU_EDGES[1:])               # 22 centres, -2.75 .. 7.75
DNU = float(NU[1] - NU[0])                                  # 0.5

FOV_DEG = 5.0
SMOOTH_PK = 2.0        # peaks/minima, as released
N_PLANES = 5
SHARD_DIR = SB35 / "nu05_shards"
DATASET_IN = SB35 / "emulator_dataset_xpkfix.npz"
DATASET_OUT = SB35 / "emulator_dataset_nu05.npz"

KEYS = ("peak_counts", "minima_counts", "pdf", "mf_v0", "mf_v1", "mf_v2")


def compute(kappa_path: Path, per_real: bool = False) -> dict[str, np.ndarray]:
    """All six nu statistics on NU, for one run's kappa cube, all source planes.

    Returns per-plane realization means plus the realization SE for the counts (the
    released dataset carries `err` only for peaks/minima, so we match that).

    per_real=True additionally keeps the (n_real, plane, nu) draws for every statistic.
    Only the fiducial BIND/truth pair needs this — fig 4's paired +-1 sigma/sqrt(50)
    bands are built from the realization-by-realization DIFFERENCE, which a mean cannot
    reconstruct. It is ~0.5 MB per run, so it is not worth doing for all 253.
    """
    import gc

    from bind.inference.stats import nongaussian_stats, peak_counts

    a = np.load(kappa_path)
    K = a["kappa"]
    del a
    nr = K.shape[0]
    out = {k: np.empty((N_PLANES, len(NU))) for k in KEYS}
    err = {k: np.empty((N_PLANES, len(NU))) for k in ("peak_counts", "minima_counts")}
    real = {k: np.empty((nr, N_PLANES, len(NU))) for k in KEYS} if per_real else {}

    for zi in range(N_PLANES):
        Kz = K[:, zi]
        # peaks / minima on the shared edges (nu_norm='map', as released)
        pc = peak_counts(Kz[:, None], fov_deg=FOV_DEG, smoothing_arcmin=SMOOTH_PK,
                         nu_bins=NU_EDGES, nu_norm="map", return_realizations=True)
        for dst in ("peak_counts", "minima_counts"):
            r = np.asarray(pc[f"{dst}_real"])[:, 0]
            out[dst][zi] = r.mean(0)
            err[dst][zi] = r.std(0) / np.sqrt(nr)
            if per_real:
                real[dst][:, zi] = r
        # Minkowski functionals on NU itself (thresholds, not edges), 1' as released
        ngk = nongaussian_stats(Kz[:, None], fov_deg=FOV_DEG,
                                smoothing_scales_arcmin=(1.0,), mf_thresholds=NU,
                                return_realizations=per_real)
        for k in ("V0", "V1", "V2"):
            out[f"mf_v{k[-1]}"][zi] = np.asarray(ngk[k])[0]
            if per_real:
                real[f"mf_v{k[-1]}"][:, zi] = np.asarray(ngk[f"{k}_real"])[:, 0]
        # PDF: histogram the UNSMOOTHED map in its own S/N units on the same edges.
        # NB this makes nu the map's OWN rms, retiring the earlier mix of unsmoothed
        # kappa bins divided by the 2'-smoothed sigma0.
        acc = np.zeros(len(NU))
        for r in range(nr):
            m = Kz[r].astype(float)
            h = np.histogram((m - m.mean()) / m.std(), bins=NU_EDGES, density=True)[0]
            acc += h
            if per_real:
                real["pdf"][r, zi] = h
        out["pdf"][zi] = acc / nr
        del Kz, pc, ngk; gc.collect()
    del K; gc.collect()
    res = {**out, **{f"{k}_err": v for k, v in err.items()}, "nu": NU}
    res.update({f"{k}_real": v for k, v in real.items()})
    return res
