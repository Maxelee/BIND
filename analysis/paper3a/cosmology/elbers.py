"""WP-A7 task 4: the fixed-cosmology (Elbers) systematic band.

Our feedback calibration is performed at the FIXED TNG300 cosmology
(params_meta.FIXED_COSMOLOGY). A survey analyst applying it at a
different (Omega_m, sigma_8) incurs an error, because baryonic feedback
couples to cosmology. Elbers et al. 2024 (arXiv:2403.12967, FLAMINGO)
quantify that coupling; this module turns it into a band on the A7
safe-scale tables.

SOURCE NUMBERS — re-verified against the paper body 2026-07-19 (NOT
from memory; the abstract alone is insufficient, it carries neither
alpha nor the xi^2 definition):

  Eq. (16)  xi^2 = f_b / c_v^2            (c_v = V_max/V_200c)
  Eq. (23)  c_v  = 1.24 (Omega_m sigma_8)^(1/8)
  Eq. (22)  Delta_F_b / (1 - F_b) = -alpha Delta(xi^2),
            alpha = 13.8 +/- 0.6
  Eq. (24)  equivalently, in terms of standard parameters,
            Delta_F_b/(1-F_b) = -alpha' Delta[f_b (Omega_m sigma_8)^-1/4],
            alpha' = 0.65 alpha = 8.98
  Eq. (20)  F_b = P_hydro / P_DMO  (so 1 - F_b is the suppression DEPTH)
  Appendix B1: a wider-range two-parameter form,
            Delta_F_b/(1-F_b) = -alpha_1 Delta(xi^2) + alpha_2 [Delta(xi^2)]^2,
            alpha_1 = 18.1 +/- 3.5, alpha_2 = 19.9 +/- 10.1
            (fitted including decaying-dark-matter models; used here
            only as a stress variant, since our range is inside the
            LCDM regime Eq. 22 was fitted on)

CORRECTION TO THE WP-A7 REPORT SCAFFOLD. The scaffold recorded
"xi^2 = f_b/(Omega_m sigma_8)^(1/4), alpha = 13.8 +/- 0.6". alpha is
right; the identity is not. xi^2 is f_b/c_v^2, and substituting Eq. (23)
gives c_v^2 = 1.5376 (Omega_m sigma_8)^(1/4), i.e.
    xi^2 = 0.65 f_b / (Omega_m sigma_8)^(1/4),
so the remembered form omits the 0.65. Using it unscaled would have
overstated Delta(xi^2) — and hence this systematic — by ~1.54x.

APPLICATION. Their Delta_F_b/(1-F_b) is the RELATIVE change in
suppression depth, so the induced error on our vs-DMO ratio R(ell) is

    Delta_R(ell) = -alpha Delta(xi^2) * (1 - R(ell)),

i.e. it scales with the LOCAL suppression depth: negligible where our
envelope is flat, largest where suppression bites. This is added to the
task-1 bias budget COHERENTLY (one-signed in ell, like the feedback
band itself) and the safe scales are recomputed with it included.

TWO LIMITS OF VALIDITY, both stated rather than absorbed:
 (a) Eq. (22) is fitted to the suppression AVERAGED over
     0.1 <= k <= 10 h/Mpc at z = 0. Applying it per-ell assumes the
     coupling is scale-independent, which the paper explicitly says it
     is not ("the cosmological coupling is more important on small
     scales"). Our per-ell application is therefore an approximation
     that is most trustworthy at the low-ell end of our range, which is
     also where the safe-scale crossings occur — but it is an
     approximation, and the alpha_1/alpha_2 stress variant brackets it.
 (b) It is a z = 0 fit; our source planes run to z_s = 2.44. No
     redshift correction is applied (the paper defers time dependence),
     so this band is a z = 0 estimate.

Run: python -m analysis.paper3a.cosmology.elbers
Out: wp7_cosmology/elbers_band.json + figures/elbers_band.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.paper3a.cosmology.safescale import (
    CRIT, ELL_MAX, ELL_MIN, WP7, _crossing, band_sources, load_bands,
    zs_interp,
)
from analysis.paper3a.cosmology.surveys import SURVEYS
from analysis.paper3a.emulator import params_meta as pm
from analysis.paper3a.emulator.statsemu import DATASET, StatsEmulator

ALPHA, ALPHA_ERR = 13.8, 0.6            # Eq. (22)
ALPHA1, ALPHA2 = 18.1, 19.9             # Eq. (B1) stress variant
C_V_NORM, C_V_POW = 1.24, 0.125         # Eq. (23)
XI2_FIT_LO, XI2_FIT_HI = 0.135, 0.155   # the xi^2 span of their Figs. 4-5

# Survey-relevant cosmology range for the band (a deliberately generous
# box spanning current WL and CMB constraints; the paper's own Fig. 6
# compares KiDS/Planck contours over a similar range).
OM_RANGE = (0.24, 0.40)
S8_RANGE = (0.70, 0.90)


def f_baryon(omega_b, omega_c):
    """f_b = Omega_b / (Omega_b + Omega_c) — cold species only (their fn 12)."""
    return omega_b / (omega_b + omega_c)


def c_v(omega_m, sigma8):
    return C_V_NORM * (omega_m * sigma8) ** C_V_POW


def xi2(omega_b, omega_c, omega_m, sigma8):
    return f_baryon(omega_b, omega_c) / c_v(omega_m, sigma8) ** 2


def fiducial_xi2():
    fc = pm.FIXED_COSMOLOGY
    om, s8, ob = fc["Omega0"], fc["sigma8"], fc["OmegaBaryon"]
    return xi2(ob, om - ob, om, s8), om, s8, ob


def rel_suppression_change(om, s8, ob_over_om=None, alpha=ALPHA):
    """Delta_F_b/(1-F_b) at (om, s8) relative to the TNG fiducial.

    Omega_b h^2 is far better constrained than Omega_c h^2 (their §4.1),
    so we hold Omega_b FIXED at the TNG value and let Omega_c absorb the
    change in Omega_m — the same logic the paper uses to eliminate
    Omega_b.
    """
    x_fid, om_f, s8_f, ob_f = fiducial_xi2()
    ob = ob_f if ob_over_om is None else ob_over_om * om
    x = xi2(ob, om - ob, om, s8)
    return -alpha * (x - x_fid), x - x_fid


def main() -> None:
    x_fid, om_f, s8_f, ob_f = fiducial_xi2()

    # ---- the (Omega_m, sigma_8) map of the induced error ------------
    om = np.linspace(*OM_RANGE, 81)
    s8 = np.linspace(*S8_RANGE, 81)
    OM, S8 = np.meshgrid(om, s8, indexing="ij")
    rel, dxi = rel_suppression_change(OM, S8)

    # ---- corner values a survey analyst would actually hit ----------
    probes = {
        "TNG300 fiducial (calibration point)": (om_f, s8_f),
        "Planck-like (0.315, 0.811)": (0.315, 0.811),
        "DES-Y3-like (0.339, 0.733)": (0.339, 0.733),
        "KiDS-like (0.305, 0.760)": (0.305, 0.760),
        "low-S8 corner of the box": (OM_RANGE[0], S8_RANGE[0]),
        "high-S8 corner of the box": (OM_RANGE[1], S8_RANGE[1]),
    }
    point_tbl = {}
    for name, (o, s) in probes.items():
        r, d = rel_suppression_change(o, s)
        r_lo, _ = rel_suppression_change(o, s, alpha=ALPHA - ALPHA_ERR)
        r_hi, _ = rel_suppression_change(o, s, alpha=ALPHA + ALPHA_ERR)
        r_b1 = -ALPHA1 * d + ALPHA2 * d ** 2
        x_here = x_fid + d
        point_tbl[name] = {
            "Omega_m": float(o), "sigma_8": float(s),
            "delta_xi2": float(d),
            "xi2": float(x_here),
            # Eq. (22) was fitted over xi^2 in ~[0.135, 0.155] (their
            # Figs. 4-5); outside that, the quoted number is an
            # EXTRAPOLATION of their fit, not a result of it.
            "outside_elbers_fitted_range": bool(not
                                                XI2_FIT_LO <= x_here
                                                <= XI2_FIT_HI),
            "rel_suppression_change": float(r),
            "rel_change_alpha_lo": float(r_lo),
            "rel_change_alpha_hi": float(r_hi),
            "rel_change_appendixB1_variant": float(r_b1),
        }

    # ---- propagate into the safe-scale tables -----------------------
    emu = StatsEmulator.load()
    raw = np.load(DATASET, allow_pickle=False)
    ell_all = np.asarray(raw["a__suppression__ell"], float)
    cl_dmo = np.asarray(raw["cl_dmo"], float)
    zs_grid = np.asarray(raw["source_redshifts"], float)
    srcs = band_sources()
    bands, fid = load_bands(emu, srcs)

    sel = (ell_all >= ELL_MIN) & (ell_all <= ELL_MAX)
    ell = ell_all[sel]
    dl = np.gradient(ell)

    # the systematic a survey at the DES-Y3-like point would incur
    rel_des = point_tbl["DES-Y3-like (0.339, 0.733)"]["rel_suppression_change"]

    prop = {}
    for sv in SURVEYS:
        r_fid = zs_interp(fid, zs_grid, sv.z_eff)[sel]      # R(ell) vs DMO
        cl = zs_interp(cl_dmo, zs_grid, sv.z_eff)[sel] * r_fid
        w = ((cl / (cl + sv.noise_cl)) ** 2 * (2 * ell + 1) * dl
             * sv.f_sky / 2.0)
        depth = np.clip(1.0 - r_fid, 0.0, None)
        d_sys = np.abs(rel_des) * depth                     # |Delta R|

        prop[sv.name] = {}
        for name, q in bands.items():
            width = 0.5 * np.abs(zs_interp(q[2] - q[0], zs_grid,
                                           sv.z_eff))[sel]
            cw = np.cumsum(w)
            base = np.cumsum(w * width) / np.sqrt(cw)
            withsys = np.cumsum(w * (width + d_sys)) / np.sqrt(cw)
            l0, _ = _crossing(ell, base, CRIT)
            l1, _ = _crossing(ell, withsys, CRIT)
            prop[sv.name][name] = {
                "lmax_safe_no_elbers": l0,
                "lmax_safe_with_elbers": l1,
                "lmax_shift": l1 - l0,
                "sys_frac_of_band_median": float(
                    np.nanmedian(d_sys / np.where(width > 0, width, np.nan))),
            }

    out = {
        "source": "Elbers et al. 2024, arXiv:2403.12967 (FLAMINGO); "
                  "Eqs. 16, 20, 22, 23, 24, B1 — re-verified against the "
                  "paper body 2026-07-19",
        "alpha": ALPHA, "alpha_err": ALPHA_ERR,
        "xi2_identity": "xi^2 = f_b/c_v^2 = 0.65 f_b (Omega_m sigma_8)^-1/4 "
                        "(the 0.65 was MISSING from the A7 scaffold)",
        "fiducial": {"Omega_m": om_f, "sigma_8": s8_f, "Omega_b": ob_f,
                     "f_b": float(f_baryon(ob_f, om_f - ob_f)),
                     "c_v": float(c_v(om_f, s8_f)), "xi2": float(x_fid)},
        "validity": "Eq.22 is a k-averaged (0.1-10 h/Mpc), z=0 LCDM fit; "
                    "per-ell and z_s>0 application is an approximation "
                    "(see module docstring limits a and b)",
        "points": point_tbl,
        "safescale_propagation_at_DES_Y3_like": prop,
        "box": {"Omega_m": OM_RANGE, "sigma_8": S8_RANGE},
    }
    WP7.mkdir(exist_ok=True)
    (WP7 / "elbers_band.json").write_text(json.dumps(out, indent=2))

    print(f"fiducial TNG300: Om={om_f} s8={s8_f} f_b="
          f"{f_baryon(ob_f, om_f-ob_f):.5f} c_v={c_v(om_f,s8_f):.4f} "
          f"xi2={x_fid:.6f}")
    print("\n=== induced relative change in suppression depth ===")
    for name, d in point_tbl.items():
        flag = "  [EXTRAPOLATED beyond the Eq.22 fit range]" \
            if d["outside_elbers_fitted_range"] else ""
        print(f"  {name:38s} {100*d['rel_suppression_change']:+7.2f}%  "
              f"(alpha range {100*d['rel_change_alpha_lo']:+.2f}..."
              f"{100*d['rel_change_alpha_hi']:+.2f}%){flag}")
    print("\n=== safe-scale shift when the Elbers band is included "
          "(at the DES-Y3-like point) ===")
    for sv in SURVEYS:
        p = prop[sv.name]
        print(f"  {sv.name:9s} prior {p['prior']['lmax_safe_no_elbers']:7.1f}"
              f" -> {p['prior']['lmax_safe_with_elbers']:7.1f} | "
              f"joint {p['joint_post']['lmax_safe_no_elbers']:7.1f}"
              f" -> {p['joint_post']['lmax_safe_with_elbers']:7.1f}  "
              f"(sys/band = "
              f"{p['joint_post']['sys_frac_of_band_median']:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cs = axes[0].contourf(OM, S8, 100 * rel, levels=21, cmap="RdBu_r")
    axes[0].contour(OM, S8, 100 * np.abs(rel), levels=[10.0],
                    colors="k", linewidths=1.5)
    axes[0].plot([om_f], [s8_f], "k*", ms=14)
    plt.colorbar(cs, ax=axes[0], label=r"$\Delta F_b/(1-F_b)$  [%]")
    axes[0].set_xlabel(r"$\Omega_m$")
    axes[0].set_ylabel(r"$\sigma_8$")
    axes[0].set_title("Fixed-cosmology error (star: TNG300 calibration;\n"
                      "black: the paper's 10% contour)", fontsize=9)

    for sv in SURVEYS:
        p = prop[sv.name]["joint_post"]
        axes[1].bar(sv.name, p["lmax_shift"], color="#D55E00")
    axes[1].set_ylabel(r"$\Delta \ell_{\max}$ from the Elbers band")
    axes[1].set_title("Safe-scale cost of the fixed-cosmology\n"
                      "systematic (joint-A+B band)", fontsize=9)
    fig.tight_layout()
    (WP7 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP7 / "figures" / "elbers_band.png", dpi=150)
    print(f"\nwrote {WP7}/elbers_band.json + figures/elbers_band.png")


if __name__ == "__main__":
    main()
