"""WP-A6 task 5: the Pandey gamma_t x y posterior-predictive comparison.

Model side (per A5-posterior sample — the Hankel mixes ell, so quantiles
are taken AFTER the transform, never before):

  C_ell^{kappa_eff y} = sum_i w_i C_ell^{kappa(z_i) y}      (B4 NNLS weights)
  xi^{gamma_t y}(theta) = Int dell ell/(2pi) C_ell J2(ell theta) B_ell

with B_ell the ACT DR6 y-map beam (1.6' Gaussian — the released beam,
audit-verified on the B side) forward-modeled onto the mock spectra
(the painted maps are beam-free). Flat-sky J2 kernel; valid for the
theta << 1 rad range used here.

Data side: the frozen Pandey et al. 2025 `compton_shear` vector (80
points, 4 source bins x 20 theta bins) combined across source bins with
the SAME n_eff weights the frozen B4 source-plane weights were built
for (neff_perbin in weights_desy3.npz), covariance propagated exactly
(C_comb = A C A^T on the 80x80 block).

Support mask: our mock ell grid starts at ell_min ~ 87; per theta bin the
low-ell leakage fraction is estimated by power-law extension of the
fiducial spectrum and bins with > MAX_LEAK of the integral outside the
grid are EXCLUDED (large theta), not extrapolated.

Consistency metric: chi2 of (data_comb - model) over valid bins with
C_comb + the posterior-predictive model covariance (sample covariance of
the transformed model across the 2000 draws), for both the posterior
median/draws and the TNG fiducial.

Run:  python analysis/paper3a/scripts/run_a6_pandey_compare.py
Out:  wp6_propagation/a6_pandey_compare.{npz,json}
      wp6_propagation/figures/a6_pandey_compare.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.data_vectors.data_vectors import load_kappa_y_pandey2025  # noqa: E402
from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.scripts.run_a6_posterior_stats import (  # noqa: E402
    load_posterior_thin,
)

B_MOCKS = Path("/mnt/ceph/users/mlee1/paper3/B/wp4_mocks")
ARCMIN = np.pi / (180.0 * 60.0)
BEAM_FWHM_ARCMIN = 1.6          # ACT DR6 y map (audit: released beam == 1.6' Gaussian)
MAX_LEAK = 0.05                 # low-ell leakage tolerance per theta bin

# NORMALIZATION FIX (2026-07-18, calibrated on rusty): the sb35_stats
# cross-spectra were measured with Pylians XPk_plane, which returns values
# a factor npix^2/fov_rad SMALLER than Pk_plane — verified by crossing a
# map with ITSELF at npix = 128/256/512/1024 (auto/cross = 11.459156 *
# npix^2 = npix^2/deg2rad(5), constant to 1e-10). The auto spectra
# (Pk_plane) match theory expectations; the crosses are the broken ones.
# This is a pure constant -> emulation/ratios unaffected; absolute cross
# amplitudes need this factor. Upstream fix flagged to Popeye
# (bind.inference.stats.power_spectrum, cross branch + re-assembly).
XPK_CROSS_FIX = 1024.0**2 / np.deg2rad(5.0)      # = 1.201580e7


def hankel_kernel(ell: np.ndarray, theta_rad: np.ndarray,
                  beam_fwhm_arcmin: float | None) -> np.ndarray:
    """K (n_theta, n_ell): xi(theta) = K @ C_ell (trapezoid on the grid)."""
    dl = np.gradient(ell)
    b = np.ones_like(ell)
    if beam_fwhm_arcmin:
        sig = beam_fwhm_arcmin * ARCMIN / np.sqrt(8.0 * np.log(2.0))
        b = np.exp(-0.5 * ell * (ell + 1.0) * sig**2)
    return (jv(2, np.outer(theta_rad, ell))
            * (ell * b * dl / (2.0 * np.pi))[None, :])


def low_ell_leakage(ell: np.ndarray, cl_fid: np.ndarray,
                    theta_rad: np.ndarray) -> np.ndarray:
    """Fraction of |xi(theta)| carried by ell < ell_min, from a power-law
    extension of the fiducial spectrum over its first decade."""
    sel = ell < ell[0] * 10
    slope, lnA = np.polyfit(np.log(ell[sel]), np.log(np.abs(cl_fid[sel])), 1)
    ell_lo = np.geomspace(2.0, ell[0], 200)
    cl_lo = np.exp(lnA) * ell_lo**slope
    k_lo = hankel_kernel(ell_lo, theta_rad, BEAM_FWHM_ARCMIN)
    k_hi = hankel_kernel(ell, theta_rad, BEAM_FWHM_ARCMIN)
    xi_lo = k_lo @ cl_lo
    xi_hi = k_hi @ cl_fid
    return np.abs(xi_lo) / np.maximum(np.abs(xi_lo) + np.abs(xi_hi), 1e-300)


def main() -> None:
    emu = StatsEmulator.load()
    post = load_posterior_thin()
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    wz = np.load(B_MOCKS / "weights_desy3.npz", allow_pickle=True)
    w = np.asarray(wz["w_default"], float)
    neff = np.asarray(wz["neff_perbin"], float)
    alpha = neff / neff.sum()

    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)
    ell = np.asarray(raw["a__cl_kappa_y__ell"], float)

    # ---- data: neff-weighted source-bin combination ------------------------
    from astropy.io import fits as afits

    dv = load_kappa_y_pandey2025("compton_shear")
    with afits.open("/mnt/ceph/users/mlee1/paper3/A/wp1_data/"
                    "kappa_y_pandey_2506.07432/DES_ACT_full_data_theorycov_2.5.fits") as f:
        t = f["compton_shear"].data
        bin1 = np.asarray(t["BIN1"], int)
        ang = np.asarray(t["ANG"], float)
        val = np.asarray(t["VALUE"], float)
    theta_arcmin = np.unique(ang)
    theta_arcmin.sort()
    n_th = len(theta_arcmin)
    A = np.zeros((n_th, len(val)))
    for j, th in enumerate(theta_arcmin):
        for b in (1, 2, 3, 4):
            i = np.flatnonzero((bin1 == b) & np.isclose(ang, th))
            A[j, i] = alpha[b - 1]
    d_comb = A @ val
    cov_comb = A @ dv.covariance @ A.T

    # ---- model: per-sample weighted spectra -> Hankel ----------------------
    theta_rad = theta_arcmin * ARCMIN
    K = hankel_kernel(ell, theta_rad, BEAM_FWHM_ARCMIN)

    # The log-transform re-fit masks the ~2.6% of (z, ell) dims whose
    # training values crossed zero (noise-dominated tail) — predictions
    # are NaN there. Zero-fill before the Hankel: those dims are
    # consistent with zero and the fill error is far below the ~9%
    # emulation floor.
    fid_pred = np.nan_to_num(emu.predict(u_fid, "cl_kappa_y"), nan=0.0) * XPK_CROSS_FIX
    n_masked = int(np.sum(~np.isfinite(emu.predict(u_fid, "cl_kappa_y"))))
    print(f"masked (zero-filled) dims: {n_masked} / {fid_pred.size}")
    cl_fid = np.tensordot(fid_pred, w, axes=([0], [0]))
    leak = low_ell_leakage(ell, cl_fid, theta_rad)
    valid = leak < MAX_LEAK

    pred = np.nan_to_num(emu.predict(post, "cl_kappa_y"), nan=0.0) * XPK_CROSS_FIX
    cl_w = np.tensordot(pred, w, axes=([1], [0]))          # (N, L)
    xi = cl_w @ K.T                                        # (N, n_th)
    xi_fid = K @ cl_fid
    qs = np.nanpercentile(xi, [2.5, 16, 50, 84, 97.5], axis=0)
    cov_model = np.cov(xi, rowvar=False)

    # ---- consistency over valid bins --------------------------------------
    v = valid
    c_tot = cov_comb[np.ix_(v, v)] + cov_model[np.ix_(v, v)]
    r_med = (d_comb - qs[2])[v]
    chi2_med = float(r_med @ np.linalg.solve(c_tot, r_med))
    r_fid = (d_comb - xi_fid)[v]
    chi2_fid = float(r_fid @ np.linalg.solve(cov_comb[np.ix_(v, v)], r_fid))
    ratio = d_comb / qs[2]

    out = {
        "n_theta_valid": int(v.sum()),
        "theta_valid_arcmin": theta_arcmin[v].tolist(),
        "max_leak_tolerance": MAX_LEAK,
        "chi2_posterior_median_totcov": chi2_med,
        "chi2_fiducial_datacov": chi2_fid,
        "data_over_model_median": {f"{theta_arcmin[j]:.1f}": float(ratio[j])
                                   for j in np.flatnonzero(v)},
        "beam_fwhm_arcmin": BEAM_FWHM_ARCMIN,
        "weights": "w_default (B4 NNLS) for the model planes; neff_perbin "
                   "for the data source-bin combination",
        "status": "first-contact frame-matched comparison; flat-sky J2, "
                  "beam forward-modeled; large-theta bins masked by the "
                  "low-ell leakage criterion",
        "xpk_cross_fix": XPK_CROSS_FIX,
        "sanity": {"xi_fid_at_2.8am": float(xi_fid[0]),
                   "data_comb_at_2.8am": float(d_comb[0])},
    }
    (WP6 / "a6_pandey_compare.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        WP6 / "a6_pandey_compare.npz", theta_arcmin=theta_arcmin,
        data_comb=d_comb, cov_comb=cov_comb, q=qs.astype(np.float32),
        xi_fid=xi_fid, valid=valid, leak=leak, cov_model=cov_model)
    print(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar(theta_arcmin[v], d_comb[v] * 1e9,
                yerr=np.sqrt(np.diag(cov_comb))[v] * 1e9, fmt="o", ms=4,
                color="#D55E00", label="Pandey+25 (neff-combined)")
    ax.errorbar(theta_arcmin[~v], d_comb[~v] * 1e9,
                yerr=np.sqrt(np.diag(cov_comb))[~v] * 1e9, fmt="o", ms=4,
                mfc="none", color="#bdbdbd", label="excluded (low-ell leak)")
    ax.plot(theta_arcmin, qs[2] * 1e9, color="#0072B2", lw=1.6,
            label="A5-posterior median")
    ax.fill_between(theta_arcmin, qs[1] * 1e9, qs[3] * 1e9, color="#0072B2",
                    alpha=0.3)
    ax.plot(theta_arcmin, xi_fid * 1e9, color="#009E73", ls="--", lw=1.2,
            label="TNG fiducial")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi^{\gamma_t y}(\theta) \times 10^9$")
    ax.legend(fontsize=7)
    ax.set_title("A6: gamma_t x y posterior-predictive vs Pandey+25")
    fig.tight_layout()
    figdir = WP6 / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "a6_pandey_compare.png", dpi=150)
    print(f"wrote {figdir}/a6_pandey_compare.png")


if __name__ == "__main__":
    main()
