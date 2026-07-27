#!/usr/bin/env python
"""P6b Fig M4 (docs/ksz_lightcone_map_plan.md §3-P6 stage B): the ELG z-shell
(z=1.16, snap046) money plot -- map-level tau-CAP(theta) and f~gas(theta) in
BIND's **patch-reuse regime** (ELG hosts, logM200 12.0-12.6, are below
BIND's painted floor of 1e13 -- physical only where they happen to fall
inside an existing >=1e13 patch's 6.25 Mpc/h cutout, ~29% of the ELG sample
here -- ``lowmass-reuse-capture`` memory / plan §1.5).

No DESI(BGS)xACT-style external kSZ data exists for ELG in this pipeline
(the per-halo capstone's ELG comparison in ``_build_ksz_paper_nb.py``
§6b/f6b_lowmass used a *different*, non-CAP, background-subtracted-aperture
estimator at a single, non-r200-matched data point ~2.8*r200 -- not
apples-to-apples with this phase's CAP observable) -- this figure is
therefore an INTERNAL consistency + caveat figure (fiducial + 253-node
envelope, patch-reuse shading), not a data confrontation.

Panels
------
(a) ELG tau-CAP(theta): fiducial + 16-84% realization band + 253 SB35 nodes.
(b) ELG f~gas(theta) = CAP_tau(ELG)/CAP_mat(ELG)/F_B (massplane denominator,
    P6b), same styling. ELG's selection (mass-proxy logM200 window) is
    node-INDEPENDENT (verified byte-identical across nodes in
    ``lightcone_capmat_merge.py``, unlike BGS's per-node M*-cut, which
    needed a per-node CAP_mat fix -- see that module's defect-and-fix note),
    so a single fiducial-computed CAP_mat is exactly correct here and is
    used for every node; NOT an instance of the BGS bug.
(c) Deflection-smearing check: fiducial tau-CAP, plain (Born positions) vs
    P2's sigma=0.222' Gaussian-smeared variant (``elgtau_smear_lightcone.npz``,
    beam_fwhm=0.52' -- the deflection budget measured at the ELG-z shell,
    P2 verdict) -- quantifies how much the (expected, small) extra
    ray-deflection blurring changes the stacked profile.

Both (a) and (b) carry the grey "patch reuse" shading + green "in_patch
fraction" annotation, mirroring the per-halo low-mass figure's
`ax.axvspan(12.0, 13.0, color="0.93")` / `axvspan(..., color="tab:green")`
motif (``_build_ksz_paper_nb.py`` §6b) -- adapted from a mass axis (that
figure) to this theta axis (this one): here the ENTIRE curve is in the
patch-reuse regime (ELG hosts are uniformly below the painted floor), so the
whole plot panel gets the grey wash rather than a sub-range.

Usage
-----
    python examples/lightcone_m4_elg_zshell.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightcone_m2_ycap_liu import mean_theta200_arcmin, N_XB

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PATCH_REUSE_GREY = "0.93"
ELG_GREEN = "tab:green"


def in_patch_fraction() -> float:
    cat = np.load(LIGHTCONE / "catalogs/desi_mock_snap046.npz", allow_pickle=True)
    mask = cat["elg_sel"] & cat["in_crop"]
    return float(cat["in_patch"][mask].mean())


def main():
    taucap = np.load(LIGHTCONE / "elgtau_lightcone.npz", allow_pickle=True)
    smear = np.load(LIGHTCONE / "elgtau_smear_lightcone.npz", allow_pickle=True)
    capmat = np.load(LIGHTCONE / "capmat_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])

    thr = mean_theta200_arcmin("elg")
    xb = taucap["theta_value"][:N_XB]
    node_theta = xb * thr
    frac_patch = in_patch_fraction()
    print(f"[M4] ELG mean theta200={thr:.3f}'; in_patch fraction={frac_patch:.3f}")

    fid_tau_mean = taucap["fid_mean_elg"][:N_XB]
    fid_tau_real = taucap["fid_real_elg"][:, :N_XB]
    sb35_tau_mean = taucap["sb35_mean_elg"][:, :N_XB]

    # ELG selection is node-independent (verified in lightcone_capmat_merge.py) --
    # capmat's 'fid_mean_elg' is the single correct denominator for every node,
    # unlike BGS which needed a per-node fix (see module docstring).
    mat_mean = capmat["fid_mean_elg"][:N_XB].astype(np.float64)
    fid_fgas_mean = fid_tau_mean / mat_mean / F_B
    fid_fgas_real = fid_tau_real / mat_mean[None, :] / F_B
    sb35_fgas_mean = sb35_tau_mean / mat_mean[None, :] / F_B

    # Honest denominator-noise disclosure (P6b follow-up point 4): CAP_mat's
    # OWN 47-realization scatter is large for ELG (~97% at theta/theta200=1,
    # vs ~12-23% for BGS/LRG) because so few galaxies land in a single
    # 25deg^2 realization at this z-shell/mass -- same root cause as panel
    # (c)'s smearing sensitivity. f~gas here still uses the realization-MEAN
    # denominator (the plan's error convention: numerator carries the
    # reported scatter), but that denominator noise floor is disclosed on
    # the figure rather than silently absorbed. Read the precomputed value
    # (computed on the FULL r200mult grid, not the xb-only slice used here --
    # the xb grid's nearest-to-1.0 bin is itself NaN for ELG, a sub-pixel-
    # aperture artifact, see note_subpixel_gap below) straight from
    # capmat_lightcone.npz rather than recomputing on a grid that doesn't
    # have a finite point there.
    _capmat_samples = list(capmat["capmat_realization_scatter_pct_samples"])
    capmat_scatter_pct = float(
        capmat["capmat_realization_scatter_pct_at_theta200eq1"][_capmat_samples.index("elg")])

    fig = plt.figure(figsize=(15.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])
    ax_tau, ax_fgas, ax_smear = (fig.add_subplot(gs[i]) for i in range(3))

    metrics = {"theta200_elg_arcmin": thr, "in_patch_fraction": frac_patch,
               "capmat_realization_scatter_pct_at_theta200eq1": capmat_scatter_pct}

    # ---------------- panel (a): tau-CAP ----------------
    for ax, mean_curve, real_curve, sb35_curve, ylabel, tag in [
        (ax_tau, fid_tau_mean, fid_tau_real, sb35_tau_mean, r"$\tau$-CAP [raw pixel-sum]", "tau"),
        (ax_fgas, fid_fgas_mean, fid_fgas_real, sb35_fgas_mean, r"$\tilde f_{\rm gas}$", "fgas"),
    ]:
        ax.axvspan(node_theta.min() if np.isfinite(node_theta).any() else 0, node_theta.max(),
                   color=PATCH_REUSE_GREY, zorder=0)

        n_plotted = 0
        for row in sb35_curve:
            if np.any(np.isfinite(row)):
                ax.plot(node_theta, row, "-", color=ELG_GREEN, lw=0.4, alpha=0.10, zorder=1)
                n_plotted += 1
        ax.plot([], [], "-", color=ELG_GREEN, lw=1.0, alpha=0.5, label=f"SB35 ({n_plotted})")

        lo16, hi84 = np.nanpercentile(real_curve, [16, 84], axis=0)
        ax.fill_between(node_theta, lo16, hi84, color="0.3", alpha=0.20, zorder=2,
                         label="fid 16-84% (real.)")
        ax.plot(node_theta, mean_curve, "-o", color="k", lw=2.2, ms=3.5, zorder=4,
                label="BIND fiducial")
        if tag == "fgas":
            ax.axhline(1, color="k", ls=":", lw=0.6, alpha=0.6)
        ax.set_xlabel(r"$\theta$ [arcmin]  ($\overline{\theta_{200}}$" f"={thr:.2f}')")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="lower right")
        ax.text(0.03, 0.95, "patch-reuse regime\n(ELG hosts below\nBIND's painted floor)",
                transform=ax.transAxes, fontsize=6.5, ha="left", va="top", color="0.4")
        ax.text(0.03, 0.72, f"in_patch: {frac_patch*100:.1f}%", transform=ax.transAxes,
                fontsize=8, ha="left", va="top", color=ELG_GREEN, fontweight="bold")
        if tag == "fgas":
            ax.text(0.97, 0.95, f"CAP_mat realization\nscatter @$\\theta_{{200}}$: "
                    f"{capmat_scatter_pct:.0f}%\n(denominator noise floor,\nnot in the "
                    "plotted band)", transform=ax.transAxes, fontsize=6.3, ha="right",
                    va="top", color="0.35",
                    bbox=dict(boxstyle="round", fc="0.97", ec="0.7", lw=.4))

    ax_tau.set_title("(a) ELG $\\tau$-CAP(theta), z=1.16", fontsize=10)
    ax_fgas.set_title("(b) ELG $\\tilde f_{\\rm gas}(\\theta)$ (massplane denominator)", fontsize=10)

    finite_theta = node_theta[np.isfinite(fid_tau_mean)]
    metrics["finite_theta_range_arcmin"] = [float(finite_theta.min()), float(finite_theta.max())] \
        if len(finite_theta) else None
    metrics["note_subpixel_gap"] = ("theta200 is only ~0.36' (~1.2 map px) at z=1.16 -- the "
        "smallest few xb/r200mult apertures fall below the disk>=3px CAP gate and are NaN "
        "(NOT a data-quality issue, an aperture-vs-pixel-size floor, same failure mode as "
        "LRG's finest bin); only theta >~0.4' (xb>=1.09) is measurable here.")

    # ---------------- panel (c): plain vs deflection-smeared ----------------
    ax = ax_smear
    fid_plain = taucap["fid_mean_elg"][:N_XB]
    fid_smear = smear["fid_mean_elg"][:N_XB]
    ax.plot(node_theta, fid_plain, "-o", color="k", lw=1.8, ms=3.5, label="plain (Born positions)")
    ax.plot(node_theta, fid_smear, "-s", color="tab:red", lw=1.8, ms=3.5,
            label="deflection-smeared ($\\sigma$=0.222')")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\tau$-CAP [raw pixel-sum] (fiducial)")
    ax.set_title("(c) deflection-smearing check (P2 budget)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    with np.errstate(invalid="ignore", divide="ignore"):
        pct_diff = 100.0 * (fid_smear - fid_plain) / fid_plain
    finite = np.isfinite(pct_diff)
    smear_summary = {"pct_diff_per_theta": np.where(finite, pct_diff, np.nan).tolist(),
                      "mean_abs_pct": float(np.nanmean(np.abs(pct_diff))),
                      "max_abs_pct": float(np.nanmax(np.abs(pct_diff))) if finite.any() else None}
    metrics["elg_smearing_effect_pct"] = smear_summary
    ax.text(0.97, 0.05, f"mean |dtau/tau|={smear_summary['mean_abs_pct']:.1f}%\n"
            f"max={smear_summary['max_abs_pct']:.1f}%",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="0.96", ec="0.7", lw=.4))

    fig.suptitle("M4 -- ELG (z=1.16) map-level $\\tau$-CAP / $\\tilde f_{\\rm gas}$ "
                 "-- patch-reuse regime, no external kSZ-CAP data available",
                 fontsize=12.5, fontweight="bold", y=1.03)
    caption = (
        f"ELG hosts (logM200 12.0-12.6) sit below BIND's painted 1e13 floor; only "
        f"in_patch={frac_patch*100:.1f}% land inside an existing >=1e13 patch's 6.25 Mpc/h "
        "cutout and are physical there (memory: lowmass-reuse-capture). No map-level DESI-ELG "
        "kSZ-CAP data exists in this pipeline (the per-halo ELG point in "
        "_build_ksz_paper_nb.py f6b_lowmass used a different, non-CAP estimator at a single "
        "non-r200-matched aperture -- not overlaid here to avoid a false apples-to-apples "
        f"read). Deflection smearing (panel c) is NOT negligible here -- mean "
        f"{smear_summary['mean_abs_pct']:.0f}%, up to {smear_summary['max_abs_pct']:.0f}% "
        "suppression at the smallest apertures -- because the P2 deflection budget "
        "(sigma=0.222' at this z-shell) is a large FRACTION of ELG's own theta200 (~0.36'), "
        "unlike BGS/LRG where theta200 is several arcmin and the same smearing is negligible."
    )
    import textwrap
    fig.text(0.5, -0.04, "\n".join(textwrap.wrap(caption, width=175)), ha="center", va="top", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.90])
    fig_path = FIG_DIR / "M4_elg_zshell.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[M4] wrote {fig_path}")

    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
