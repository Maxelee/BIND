#!/usr/bin/env python
"""P6b Fig M2 (docs/ksz_lightcone_map_plan.md §3-P6 stage B, REGENERATED): the
map-level tSZ money plot -- BEAMED map-level y-CAP(theta) vs the official
Liu+2025 DESI-LRG y-CAP data.

**What changed vs the P6a version of this script.** P6a's version compared
Liu+2025's DESI-LRG data against BIND's *BGS* sample (mean logM200 13.6-13.82,
z~0.26) purely because no map-level LRG sample existed yet in P3/P5 -- an
explicit apples-to-oranges mismatch (BGS vs LRG mass+z), documented as the
leading explanation for the >10x gap found there. P3-LRG (this same session)
built a genuine map-level LRG mock (``desi_mock_snap067.npz``, mass-proxy
selected to logM200=13.181, matching Liu's ACT-CMB-lensing host mass
13.18 -- Sailer+24 -- to 0.001 dex) and P5's ELG/LRG sweep produced the full
253-node LRG y-CAP shard set. **This script promotes the mass-matched LRG
comparison to the PRIMARY panel** (the trustworthy replacement for both the
distrusted per-halo tSZ figure AND P6a's mismatched BGS-vs-LRG version), and
keeps the old BGS-vs-Liu comparison as a labelled SECONDARY panel purely to
show how much of the apparent gap was mass/z mismatch vs real tension.

Conventions (both panels; replicated from ``examples/_build_ksz_paper_nb.py``
the "distrusted" per-halo money plot, ``examples/_reduce_ycap_lrg.py`` the
per-halo LRG comparison, and ``examples/_build_ksz_paper2_nb.py`` §F the
official Liu+2025 loader):

1. **Data + bin matching.** `examples/figures_ksz2/tsz_liu2025_official.npz`
   (Liu+2025 ACT DR6 x DESI LRG y-CAP release, `fig3.csv`, theta grid
   ``[1.0, 1.625, ..., 6.0]`` arcmin -- identical to `_reduce_ycap_lrg.py`'s
   `RAP` per-halo LRG aperture grid). It has 4 photo-z-bin columns
   `pz{1..4}_fiducial(_err)`; **as downloaded these are numerically identical**
   (P6a finding, re-confirmed here) -- an export bug in the release CSV, not
   a modeling choice, so `pz1` is used throughout with no loss of generality.
2. **Units.** The shard's raw pixel-sum compensated aperture (dimensionless
   "y*pixel") is converted to the Liu/per-halo `y*arcmin^2` flux convention
   by multiplying by the map pixel solid angle `DTHETA_ARCMIN**2` (a
   Riemann-sum discretization of the same 2-D compensated-aperture integral).
3. **theta grid.** The shard's dimensionless `xb` grid (theta_d/theta200,
   [0.3,3.0]) is converted to absolute arcmin via the FIDUCIAL sample's MEAN
   theta200 (`r200_comoving/chi[plane]`), applied uniformly to the fiducial
   and all 253 SB35 node curves -- the same convention P6a established for
   the BGS panel, now applied to LRG too.

Usage
-----
    python examples/lightcone_m2_ycap_liu.py
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

from lightcone_cap_stack import GEOM_PATH, select_sample
from bind.inference.lux_geometry import DTHETA_RAD, load_geometry

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPO = Path("/mnt/home/mlee1/BIND-ksz2")
LIU_NPZ = REPO / "examples/figures_ksz2/tsz_liu2025_official.npz"

BGS_SAMPLES = ["bgs110", "bgs1125"]
SAMPLE_LABEL = {"bgs110": r"BGS $M_\star{>}10^{11.0}$ (z~0.26)",
                "bgs1125": r"BGS $M_\star{>}10^{11.25}$ (z~0.26)",
                "lrg": r"LRG mass-matched ($\log M_{200}{=}13.18$, z=0.503)"}
SAMPLE_COLOR = {"bgs110": "tab:blue", "bgs1125": "tab:green", "lrg": "tab:purple"}

DTHETA_ARCMIN = np.degrees(DTHETA_RAD) * 60.0
PIX_AREA_ARCMIN2 = DTHETA_ARCMIN ** 2   # raw pixel-sum CAP -> y*arcmin^2 flux

N_XB = 18   # lightcone_cap_stack.XB length; theta_value[:N_XB] are the xb grid


def mean_theta200_arcmin(sample: str) -> float:
    """Mean theta(r200) [arcmin] of the fiducial's selected+in_crop galaxies
    for `sample` -- the single conversion scalar applied to the whole
    sample's dimensionless theta grid (plan: 'via the sample mean theta200')."""
    geom = load_geometry(GEOM_PATH)
    sel = select_sample(sample, "fid")
    chi_p = geom.chi[sel["plane_p"]]
    theta200 = np.degrees(sel["r200"] / chi_p) * 60.0
    return float(theta200.mean())


def load_liu_pz1():
    d = np.load(LIU_NPZ)
    theta = d["theta"]
    y = d["pz1_fiducial"]
    yerr = d["pz1_fiducial_err"]
    ok = np.isfinite(theta) & np.isfinite(y)
    all_pz_identical = all(
        np.allclose(d[f"pz{k}_fiducial"][ok], y[ok]) for k in [2, 3, 4] if f"pz{k}_fiducial" in d
    )
    return theta[ok], y[ok], yerr[ok], bool(all_pz_identical)


def _panel(ax, sample: str, product_path: Path, liu_theta, liu_y, liu_yerr, *,
           show_sb35=True, title_suffix=""):
    theta200 = mean_theta200_arcmin(sample)
    d = np.load(product_path, allow_pickle=True)
    node_theta = d["theta_value"][:N_XB] * theta200
    fid_mean = d[f"fid_mean_{sample}"][:N_XB] * PIX_AREA_ARCMIN2
    fid_real = d[f"fid_real_{sample}"][:, :N_XB] * PIX_AREA_ARCMIN2
    lo16, hi84 = np.nanpercentile(fid_real, [16, 84], axis=0)
    color = SAMPLE_COLOR[sample]

    n_nodes_plotted = 0
    if show_sb35:
        sb35_mean = d[f"sb35_mean_{sample}"][:, :N_XB] * PIX_AREA_ARCMIN2
        for row in sb35_mean:
            if np.all(np.isfinite(row)):
                ax.plot(node_theta, row, "-", color=color, lw=0.4, alpha=0.10, zorder=1)
                n_nodes_plotted += 1
        ax.plot([], [], "-", color=color, lw=1.0, alpha=0.5, label=f"SB35 ({n_nodes_plotted})")

    ax.fill_between(node_theta, lo16, hi84, color=color, alpha=0.20, zorder=2,
                     label="fid 16-84% (real.)")
    ax.plot(node_theta, fid_mean, "-o", color=color, lw=2.6, ms=4.5, zorder=4,
            label="BIND fiducial")
    ax.errorbar(liu_theta, liu_y, yerr=liu_yerr, fmt="s", color="k", mfc="white", ms=5,
                 capsize=2, zorder=5, label="Liu+2025 (pz1, ACT DR6 x DESI LRG)")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$y$-CAP  [$y\cdot$arcmin$^2$]")
    ax.set_title(f"{SAMPLE_LABEL[sample]}{title_suffix}\n"
                  f"($\\overline{{\\theta_{{200}}}}$={theta200:.2f}')", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")

    fid_at_liu = np.interp(liu_theta, node_theta, fid_mean)
    ratio = fid_at_liu / liu_y
    return {"theta200_arcmin": theta200, "theta_arcmin": liu_theta.tolist(),
            "fiducial_over_liu": ratio.tolist(), "min": float(np.min(ratio)),
            "max": float(np.max(ratio)), "n_sb35_plotted": n_nodes_plotted}


def main():
    liu_theta, liu_y, liu_yerr, liu_degenerate = load_liu_pz1()
    print(f"[M2] Liu pz1 points: {liu_theta} arcmin; pz1..pz4 numerically identical = {liu_degenerate}")

    metrics = {"liu_pz_bins_numerically_identical": liu_degenerate}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ------------------------------------------------------------------
    # PRIMARY panel: mass-matched LRG (the trustworthy comparison)
    # ------------------------------------------------------------------
    ax = axes[0]
    r_lrg = _panel(ax, "lrg", LIGHTCONE / "lrgy_beam_lightcone.npz", liu_theta, liu_y, liu_yerr,
                   title_suffix="  [PRIMARY -- mass-matched to Liu]")
    ax.set_title(ax.get_title(), fontsize=10, fontweight="bold")
    ax.text(0.03, 0.03, f"fid/Liu at data apertures: {r_lrg['min']:.2f}-{r_lrg['max']:.2f}x",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="0.96", ec="0.7", lw=.4))
    metrics["lrg_ycap_fid_over_liu_range"] = r_lrg

    # ------------------------------------------------------------------
    # SECONDARY panel: old BGS comparison (mass/z-mismatched), both cuts
    # ------------------------------------------------------------------
    ax = axes[1]
    bgs_results = {}
    for sample in BGS_SAMPLES:
        r = _panel(ax, sample, LIGHTCONE / "ycap_beam_lightcone.npz", liu_theta, liu_y, liu_yerr,
                   show_sb35=False)
        bgs_results[sample] = r
    ax.set_title("SECONDARY -- BGS vs Liu (mass/z-MISMATCHED)\nsee caveat below", fontsize=9.5,
                 color="0.25")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    metrics["bgs_vs_liu_ratio_range_mismatched"] = bgs_results

    lrg_ratio_str = f"{r_lrg['min']:.2f}-{r_lrg['max']:.2f}x"
    bgs_ratio_strs = ", ".join(f"{s}={bgs_results[s]['min']:.0f}-{bgs_results[s]['max']:.0f}x"
                                for s in BGS_SAMPLES)
    liu_bin_note = ("pz1 used as the Liu bin (pz1-pz4 numerically identical in the downloaded "
                     "fig3.csv release -- an export/bin-degeneracy bug, not a modeling choice; "
                     "P3lrg/P6a).")
    primary_note = (f"PRIMARY result: mass-matched LRG fid/Liu = {lrg_ratio_str} -- an O(1) "
                     "offset, not the >10x seen when comparing BGS (wrong mass+z) to Liu's LRG "
                     "data. Most of the OLD gap was mass/z mismatch (Y~M^5/3: BGS's 0.4-0.6 dex "
                     f"higher host mass alone predicts ~5-11x); secondary panel keeps that "
                     f"mismatched comparison (BGS fid/Liu = {bgs_ratio_strs}) for reference.")
    metrics["primary_note"] = primary_note

    import textwrap
    wrap = lambda s: "\n".join(textwrap.wrap(s, width=175))
    fig.suptitle("M2 -- map-level BEAMED (1.6') $y$-CAP vs Liu+2025 ACT$\\times$DESI-LRG $y$-CAP "
                 "(LRG mass-matched, PRIMARY)", fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, 0.955, wrap(liu_bin_note), ha="center", va="top", fontsize=7.5)
    fig.text(0.5, -0.03, wrap(primary_note), ha="center", va="top", fontsize=7.5, color="0.15")
    fig.tight_layout(rect=[0, 0.01, 1, 0.88])
    fig_path = FIG_DIR / "M2_ycap_vs_liu.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[M2] wrote {fig_path}")

    metrics["liu_bin_note"] = liu_bin_note
    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
