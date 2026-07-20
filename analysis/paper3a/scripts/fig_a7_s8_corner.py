"""The off-manifold S8 bias as a corner plot — the standard way.

The bias is a feedback<->S8 degeneracy: an analyst who clamps the feedback
model to the TNG posterior band (Delta ln M_gas ~ -0.35) when the truth is
off-manifold (~ -1.22) must absorb the leftover suppression into a LOW S8.
A corner in (Delta ln M_gas, S8) shows this directly: the biased posterior
and the true one, with marginals.

CONSTRUCTION, self-consistent at the van Daalen+2020 pivot k = 0.5 h/Mpc
(where vd20 is valid). Matching the observed suppression amplitude,

    S8_model = S8_fid * sqrt( [1 + dP/P(f_true)] / [1 + dP/P(f_model)] ),
    dP/P(f) = -exp(d*ftilde_bar + e),  ftilde_bar(Delta) = ftilde_bar_fid*exp(Delta),

so an analyst whose feedback prior is WEAKER than the truth (higher
ftilde_bar, less negative dP/P) recovers S8 biased LOW. This is the
pivot-scale illustration of the degeneracy, NOT the full multi-scale C_ell
fit (vd20 is k<1); the direction and rough magnitude carry, the exact
number is scale-dependent. Every constant is frozen/sourced (p5 vd20 +
ftilde_bar; joint_ab posterior width; s8_exercise S8 sigma).

Two analysts are drawn:
  - TNG-calibrated (the paper's current A7 analyst): feedback prior =
    the joint A+B posterior N(-0.355, 0.024). Recovers biased S8.
  - correct feedback model: feedback prior brackets the data-required
    -1.22. Recovers S8 ~ truth. Shown as the reference.

Run: python analysis/paper3a/scripts/fig_a7_s8_corner.py
Out: bind-paper3-plans/systematics-hunt/figures/09_s8_corner.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
from analysis.paper3a.style import COL, W_SINGLE, apply  # noqa: E402

apply()
OUT = Path("/mnt/home/mlee1/bind-paper3-plans/systematics-hunt/figures")

P5 = json.loads((Path("/mnt/ceph/users/mlee1/paper3/B/wp5_inference")
                 / "p5_pk_feedback_implication.json").read_text())
JSUM = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp5_chains")
                   / "joint_ab_summary.json").read_text())
S8X = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
                  / "s8_exercise.json").read_text())
OFFM = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp7_cosmology")
                   / "s8_offmanifold.json").read_text())

S8_FID = 0.83
D = P5["van_daalen_2020"]["coefficients_k0p5"]["d"]     # -5.99
E = P5["van_daalen_2020"]["coefficients_k0p5"]["e"]     # -0.5107
# fiducial baryon fraction (remeasured catalog, nu3 mass-matched) and the
# data-required Delta ln M_gas (p5 acceptance)
FBAR_FID = P5["absolute_branch_BOTH_CATALOGS"]["remeasured_d1ce1133"][
    "per_bin_mass_matched"]["nu3"]["ftilde_bar_TNG"]
DLN_DATA = -1.214            # p5 acceptance required_dln_mgas nu3 (~ -1.21)
# conservative branch: if sigma8/sigma_pos nuisances absorb part of the
# deficit, the residual TRUE feedback is milder. ftilde_bar from p5.
FBAR_CONS = OFFM["conservative_after_nuisance_nu3"]["remeasured_d1ce1133"][
    "sigma8_DESY3_0p776"]["ftilde_bar_data"]                  # ~0.424
DLN_CONS = float(np.log(FBAR_CONS / FBAR_FID))               # ~ -0.78

# TNG-calibrated analyst's feedback prior = the joint A+B posterior
DLN_TNG_MU = JSUM["coords_posterior"]["dln_mgas"]["p50"]         # -0.353
DLN_TNG_SD = 0.5 * (JSUM["coords_posterior"]["dln_mgas"]["p84"]
                    - JSUM["coords_posterior"]["dln_mgas"]["p16"])  # ~0.024
# S8 statistical width: DES-Y6 calibrated sigma from the C_ell exercise
S8_SD = S8X["surveys"]["DES-Y6"]["calibrated"]["sigma_S8"]        # ~0.0054


def dP_P(dln_mgas):
    fbar = FBAR_FID * np.exp(dln_mgas)
    return -np.exp(D * fbar + E)


def s8_of(dln_model, dln_true):
    """Recovered S8 when the analyst models feedback at dln_model but the
    truth is dln_true, matching the k=0.5 suppression amplitude."""
    return S8_FID * np.sqrt((1.0 + dP_P(dln_true)) / (1.0 + dP_P(dln_model)))


def sample(dln_mu, dln_sd, n, rng, dln_true=DLN_DATA):
    dln = rng.normal(dln_mu, dln_sd, n)
    s8 = s8_of(dln, dln_true) + rng.normal(0.0, S8_SD, n)
    return dln, s8


def contour(ax, x, y, color, fill=False):
    h, xe, ye = np.histogram2d(x, y, bins=70)
    h = h.T
    hs = np.sort(h.ravel())[::-1]
    cum = np.cumsum(hs) / hs.sum()
    lv = [hs[np.searchsorted(cum, f)] for f in (0.95, 0.68)]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if fill:
        ax.contourf(xc, yc, h, levels=lv + [h.max() + 1], colors=color,
                    alpha=0.28)
    ax.contour(xc, yc, h, levels=lv, colors=color, linewidths=1.3)


def main():
    rng = np.random.default_rng(7)
    n = 300_000
    dx_t, s8_t = sample(DLN_DATA, 0.10, n, rng)         # correct analyst
    dx_b, s8_b = sample(DLN_TNG_MU, DLN_TNG_SD, n, rng)  # TNG-clamped, full deficit
    # TNG-clamped analyst, but part of the deficit is nuisance (milder truth)
    dx_c, s8_c = sample(DLN_TNG_MU, DLN_TNG_SD, n, rng, dln_true=DLN_CONS)

    fig = plt.figure(figsize=(W_SINGLE * 1.35, W_SINGLE * 1.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                          wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    # 2D
    contour(ax, dx_t, s8_t, COL["truth"], fill=True)
    contour(ax, dx_c, s8_c, COL["aux"], fill=True)
    contour(ax, dx_b, s8_b, COL["data"], fill=True)
    ax.axhline(S8_FID, color=COL["ref"], ls=":", lw=1.1)
    ax.axvline(DLN_DATA, color=COL["ref"], ls=":", lw=1.1)
    ax.plot(DLN_DATA, S8_FID, "*", ms=15, color=COL["ref"], mec="white",
            mew=0.8, zorder=6)
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$  (feedback strength)")
    ax.set_ylabel(r"$S_8$")
    ax.set_xlim(-1.45, -0.05)
    ax.set_ylim(0.773, 0.851)
    ax.tick_params(top=False, right=False)

    # marginals
    for d, c in ((dx_t, COL["truth"]), (dx_b, COL["data"])):
        axx.hist(d, 90, density=True, histtype="step", color=c, lw=1.5)
    axx.axvline(DLN_DATA, color=COL["ref"], ls=":", lw=1.1)
    axx.set_axis_off()
    for s, c in ((s8_t, COL["truth"]), (s8_c, COL["aux"]), (s8_b, COL["data"])):
        axy.hist(s, 90, density=True, histtype="step", color=c, lw=1.5,
                 orientation="horizontal")
    axy.axhline(S8_FID, color=COL["ref"], ls=":", lw=1.1)
    axy.set_axis_off()

    bias_b = (float(np.median(s8_b)) - S8_FID) / S8_SD
    bias_c = (float(np.median(s8_c)) - S8_FID) / S8_SD
    ax.annotate(f"if the full deficit\nis feedback:  ${bias_b:+.0f}\\sigma$",
                xy=(DLN_TNG_MU, np.median(s8_b)), xytext=(-0.98, 0.784),
                fontsize=7, color="#8a3000", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["data"]))
    ax.annotate(f"if nuisances absorb\npart of it:  ${bias_c:+.0f}\\sigma$",
                xy=(DLN_TNG_MU, np.median(s8_c)), xytext=(-0.98, 0.818),
                fontsize=7, color="#1f6f8f", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color=COL["aux"]))
    ax.annotate("truth", xy=(DLN_DATA, S8_FID), xytext=(-0.78, 0.842),
                fontsize=7, ha="center", color="#444444",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888888"))
    axx.text(0.02, 0.7, "feedback prior", transform=axx.transAxes,
             fontsize=6.8, color="#666666")

    import matplotlib.patches as mp
    ax.legend(handles=[
        mp.Patch(fc=COL["truth"], alpha=0.5, label="correct feedback model"),
        mp.Patch(fc=COL["data"], alpha=0.5,
                 label="TNG-clamped, full deficit real"),
        mp.Patch(fc=COL["aux"], alpha=0.5,
                 label="TNG-clamped, part is nuisance")],
        loc="lower right", fontsize=6.6, framealpha=0.95)
    axx.set_title(r"$S_8$ bias from an insufficient feedback prior "
                  r"(vd20 pivot $k=0.5$)", fontsize=7.5, pad=3)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"09_s8_corner.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"TNG-clamped, full deficit: S8 = {np.median(s8_b):.4f}  "
          f"({bias_b:+.1f} sigma)")
    print(f"TNG-clamped, conservative: S8 = {np.median(s8_c):.4f}  "
          f"({bias_c:+.1f} sigma)  [true dln = {DLN_CONS:.2f}]")
    print(f"correct-model:             S8 = {np.median(s8_t):.4f}")
    print("wrote 09_s8_corner.pdf/.png")


if __name__ == "__main__":
    main()
