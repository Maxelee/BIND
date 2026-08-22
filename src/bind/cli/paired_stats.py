"""``bind-paired-stats``: shared-sky (paired) response errors for WL statistics.

The lightcone realizations of a run and the fiducial are random rotations of the
*same* DMO box, so cosmic variance cancels in the per-realization ratio
``S_r(run)/S_r(fid)``.  The paired error ``std_r(ratio)/sqrt(N)`` is ~10x smaller
than the marginal error stored in ``Cl_kappa.npz`` / ``peak_counts*.npz`` (which
quadrature-adds two independent means).  This CLI reduces a run and the fiducial
*per realization* and writes the paired response **and** paired error for each WL
statistic into ``<run_dir>/paired_stats.npz``.

    bind-paired-stats --run_dir .../twobound/run_0013 --fid_dir .../bind/run_0000

The fiducial per-realization cube is cached once at
``<fid_dir>/paired_perreal_fid.npz`` and reused across all runs.  Statistics
(all at z_s bins of the maps; the dashboard uses z_s=1):

* ``clk``        — WL auto power, paired ratio response vs ell
* ``pk``/``min`` — fixed-nu peak / minima counts (nu = kappa/sigma_fid), ratio
* ``V0/V1/V2``   — Minkowski functionals, (run-fid)/max|fid| (matches dashboard)
* ``sk``         — skewness per smoothing scale, (run-fid)

After running for all runs:  ``python examples/dashboard_precompute.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from bind.inference import stats as S

PEAK_SMOOTH = 2.0          # arcmin, scalar (matches peak_counts_nufid)


def _sigma0(fid_kappa: np.ndarray, fov: float, n: int = 20) -> np.ndarray:
    """Fixed nu: ONE sigma per source bin from the fiducial (first n realizations)."""
    nref = min(fid_kappa.shape[0], n)
    return np.array([np.mean([S._gaussian_smooth(fid_kappa[r, i], PEAK_SMOOTH, fov).std()
                              for r in range(nref)])
                     for i in range(fid_kappa.shape[1])])


def _perreal(kappa: np.ndarray, fov: float, sig0: np.ndarray) -> dict:
    """Per-realization WL statistic cubes for one run's kappa maps."""
    n_real, n_src = kappa.shape[:2]
    clk, ell = [], None                                 # auto power only (paired needs no cross)
    for r in range(n_real):
        row = []
        for i in range(n_src):
            ell, cl = S.power_spectrum(kappa[r, i], fov_deg=fov)
            row.append(cl)
        clk.append(row)
    pk = S.peak_counts(kappa, fov_deg=fov, smoothing_arcmin=PEAK_SMOOTH,
                       nu_norm="fixed", nu_sigma0=sig0, return_realizations=True)
    ng = S.nongaussian_stats(kappa, fov_deg=fov, return_realizations=True)
    return {
        "ell": ell, "nu": pk["nu"], "mf_nu": ng["mf_nu"],
        "mf_scales": ng["smoothing_scales_arcmin"],
        "clk": np.asarray(clk),                         # (n_real, n_src, n_ell)
        "pk": pk["peak_counts_real"], "min": pk["minima_counts_real"],
        "V0": ng["V0_real"], "V1": ng["V1_real"], "V2": ng["V2_real"],
        "sk": ng["skewness_real"],                      # (n_real, n_src, n_scale)
    }


def _load_or_build_fid(fid_dir: Path, fov: float, n_real) -> dict:
    cache = fid_dir / "paired_perreal_fid.npz"
    if cache.exists():
        return dict(np.load(cache))
    fk = np.load(fid_dir / "kappa_maps.npz")["kappa"]
    if n_real:
        fk = fk[:n_real]
    sig0 = _sigma0(fk, fov)
    print(f"[paired] building fiducial per-realization cube from {fid_dir} "
          f"(n_real={fk.shape[0]}, sigma0={np.array2string(sig0, precision=4)})")
    fid = _perreal(fk, fov, sig0)
    fid["sigma0"] = sig0
    tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")     # atomic vs concurrent array tasks
    np.savez_compressed(tmp, **fid)
    os.replace(tmp, cache)
    print(f"[paired] cached {cache.name}")
    return fid


def _ratio_resp(run, fid, fill=np.nan):
    """Paired response and error for a ratio statistic, robust to empty bins.

    Uses the per-realization *difference* ``run_r - fid_r`` (cosmic variance cancels)
    normalized by the fiducial mean — well-defined even where a single realization has
    zero counts (unlike a per-realization ratio).  Bins with zero fiducial mean get
    ``fill``.  NOTE: band significance for counts must band-INTEGRATE the per-realization
    arrays (see ``*_real`` outputs), not band-average these per-bin numbers.
    """
    n = min(len(run), len(fid))
    r, f = run[:n], fid[:n]
    fm = f.mean(0)
    ok = fm != 0
    resp = np.full_like(fm, fill, dtype=float)
    err = np.full_like(fm, fill, dtype=float)
    resp[ok] = (r.mean(0)[ok] - fm[ok]) / fm[ok]
    err[ok] = (r - f).std(0)[ok] / np.sqrt(n) / fm[ok]
    return resp, err


def _diff_resp(run, fid, norm):
    """Paired difference response normalized by `norm`, and its error."""
    n = min(len(run), len(fid))
    d = run[:n] - fid[:n]
    return d.mean(0) / norm, d.std(0) / np.sqrt(n) / norm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--fid_dir", type=Path, required=True,
                   help="Fiducial run dir (defines sigma_fid and the shared sky).")
    p.add_argument("--fov_deg", type=float, default=5.0)
    p.add_argument("--n_real", type=int, default=None, help="Cap realizations.")
    p.add_argument("--out_name", default="paired_stats.npz")
    a = p.parse_args()

    fid = _load_or_build_fid(a.fid_dir, a.fov_deg, a.n_real)
    if a.run_dir.resolve() == a.fid_dir.resolve():
        print("[paired] run_dir == fid_dir: fiducial cube cached, no response to write.")
        return

    rk = np.load(a.run_dir / "kappa_maps.npz")["kappa"]
    if a.n_real:
        rk = rk[:a.n_real]
    run = _perreal(rk, a.fov_deg, fid["sigma0"])

    n = min(len(rk), len(fid["clk"]))
    out = {"ell": fid["ell"], "nu": fid["nu"], "mf_nu": fid["mf_nu"],
           "mf_scales": fid["mf_scales"], "nu_fixed": np.array(1), "n_real": np.array(n)}
    # clk: smooth (no empty bins) -> per-bin response + paired err is fine to band-average
    out["clk_resp"], out["clk_err"] = _ratio_resp(run["clk"], fid["clk"])
    # peaks/minima: store per-realization counts so the notebook band-INTEGRATES them
    # (rare-event counts: band-averaging per-bin fractional errors is wrong). Also keep
    # a per-bin response/err for the panel curves (fill empty bins with 0).
    for key in ("pk", "min"):
        out[f"{key}_resp"], out[f"{key}_err"] = _ratio_resp(run[key], fid[key], fill=0.0)
        out[f"{key}_real"] = run[key].astype(np.float32)          # (n_real, n_src, n_nu)
    for key in ("V0", "V1", "V2"):                                # normalize per source bin
        norm = np.max(np.abs(fid[key].mean(0)), axis=-1, keepdims=True)
        out[f"{key}_resp"], out[f"{key}_err"] = _diff_resp(run[key], fid[key], norm)
    out["sk_resp"], out["sk_err"] = _diff_resp(run["sk"], fid["sk"], 1.0)

    np.savez_compressed(a.run_dir / a.out_name, **out)
    src = 1 if run["pk"].shape[1] > 1 else 0                       # band-integrated z_s=1 check
    nu = out["nu"]
    m = (nu >= 3) & (nu < 6)
    rb = run["pk"][:n, src][:, m].sum(1)
    fb = fid["pk"][:n, src][:, m].sum(1)
    resp = rb.mean() / fb.mean() - 1
    err = (rb - fb).std() / fb.mean() / np.sqrt(n)
    print(f"[paired] wrote {a.run_dir.name}/{a.out_name}  (N={n})  band-integrated "
          f"N_pk(nu>3) z_s=1: {resp*100:+.1f}% +/- {err*100:.2f}%  "
          f"({abs(resp)/max(err,1e-12):.1f}sigma)")


if __name__ == "__main__":
    main()
