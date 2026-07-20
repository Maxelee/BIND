"""The systematics hunt, told as figures (2026-07-20).

Seven panels, in narrative order: the deficit, why it is not a unit error,
what the six-domain audit found, Popeye's absolute anchor, the one A-side
defect that mattered, the invariance of the kappa-peak block, and the
coverage argument.

Every number is sourced inline to the artifact or findings file it came
from. Nothing here is recomputed -- this script only draws.

House style is mandatory (FIGURE_STYLE.md): `apply()`, palette by ROLE
never cycled, sequential = one hue light->dark, one y-axis per panel, no
grids, ticks inward.

Run: python analysis/paper3a/scripts/fig_syshunt_story.py
Out: bind-paper3-plans/systematics-hunt/figures/*.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))

from analysis.paper3a.style import (COL, W_DOUBLE, W_SINGLE,  # noqa: E402
                                    apply, fig_single)

apply()

OUT = Path("/mnt/home/mlee1/bind-paper3-plans/systematics-hunt/figures")
OUT.mkdir(parents=True, exist_ok=True)

# one-hue sequential ramp built from the model blue (house rule 4)
SEQ = LinearSegmentedColormap.from_list("bind_seq", ["#f3f8fc", COL["model"]])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------------------
# 1. THE DEFICIT -- data/model across (nu, radius).  FINDINGS_units.md sec 1
# ---------------------------------------------------------------------------
RADII = [2, 3, 4, 6, 8]
NU = ["1-2", "2-3", "3-4", r"$\geq 4$"]
RATIO = np.array([                      # rows = radius, cols = nu bin
    [0.046, 0.334, 0.237, 0.066],
    [0.216, 0.268, 0.259, 0.104],
    [0.313, 0.315, 0.282, 0.147],
    [0.370, 0.461, 0.361, 0.198],
    [0.549, 0.507, 0.466, 0.227],
])


def fig1():
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.0))
    im = ax.imshow(RATIO, cmap=SEQ, vmin=0, vmax=1.0, aspect="auto",
                   origin="lower")
    for i in range(RATIO.shape[0]):
        for j in range(RATIO.shape[1]):
            v = RATIO[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.42 else "#333333")
    ax.set_xticks(range(len(NU)), NU)
    ax.set_yticks(range(len(RADII)), [f"{r}'" for r in RADII])
    ax.set_xlabel(r"$\nu$ bin")
    ax.set_ylabel("CAP aperture")
    ax.tick_params(top=False, right=False)
    # the ramp deliberately runs to 1.0 so the distance from parity is the
    # visual message. An explicit "parity" tag at 1.0 collides with the top
    # tick, so it goes in the label instead.
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label("data / model   (1 = parity)", fontsize=8)
    save(fig, "01_the_deficit")


# ---------------------------------------------------------------------------
# 2. NOT A UNIT ERROR -- the ratio is not constant.  chi2 = 103.0/19
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = fig_single(2.9)
    shades = [SEQ(x) for x in (0.42, 0.60, 0.78, 0.96)]
    for j, (nu, c) in enumerate(zip(NU, shades)):
        ax.plot(RADII, RATIO[:, j], "-o", color=c, lw=2.0, ms=5,
                mec="white", mew=0.8, zorder=3)
        ax.annotate(rf"$\nu\,{nu}$", (RADII[-1], RATIO[-1, j]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=c, va="center")
    ax.axhline(0.2654, color=COL["ref"], ls=":", lw=1.2, zorder=2)
    ax.text(8.4, 0.2654, "best single\nconstant", fontsize=6.5,
            va="center", ha="left")
    ax.set_xlabel("CAP aperture [arcmin]")
    ax.set_ylabel("data / model")
    ax.set_xlim(1.4, 10.6)
    ax.set_ylim(0, 0.62)
    ax.text(0.03, 0.94, r"one constant: $\chi^2 = 103.0/19$",
            transform=ax.transAxes, fontsize=7.5, va="top")
    save(fig, "02_not_a_unit_error")


# ---------------------------------------------------------------------------
# 3. THE AUDIT -- every lever found, against what is needed
#    magnitudes from the six FINDINGS_*.md null tables
# ---------------------------------------------------------------------------
CAND = [   # (label, factor, wrong_sign)
    (r"$Y_{\rm CAP}$ units / CAP operator", 1.000, False),
    ("mass-definition relabel",    1.000, False),
    ("single-sample smoothing",    1.018, False),
    ("ILC beam (any plausible)",   1.020, False),
    ("reference offset",           1.031, False),
    (r"$\nu$ normalisation domain", 1.080, False),
    ("shared BIND amplitude",      1.110, False),
    ("halo-pasting ceiling",       1.110, True),
    ("mock z-truncation",          1.200, True),
    ("transfer residual",          1.630, True),
    (r"Wiener $\rightarrow$ GLIMPSE", 2.000, True),
    ("absolute y calibration",     2.000, False),
]


def fig3():
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.4))
    ax.axvspan(3.2, 6.8, color=COL["shade"], alpha=0.35, zorder=0)
    ax.text(4.66, len(CAND) - 0.4, "needed to explain\nthe deficit",
            ha="center", va="top", fontsize=7.5, color="#444444")

    for i, (lab, f, wrong) in enumerate(CAND):
        c = COL["variant"] if wrong else COL["model"]
        ax.barh(i, f - 1.0, left=1.0, height=0.62, color=c,
                alpha=0.95 if not wrong else 0.75,
                hatch="///" if wrong else None,
                edgecolor="white", lw=0.8, zorder=3)
        ax.text(f + 0.06, i, f"{f:.2f}" + (r"$\times$" if f > 1 else ""),
                va="center", fontsize=6.8, color="#333333")
    ax.set_yticks(range(len(CAND)), [c[0] for c in CAND])
    ax.set_xscale("log")
    ax.set_xlim(0.97, 8.2)
    ax.set_xticks([1, 1.5, 2, 3, 5, 8], ["1", "1.5", "2", "3", "5", "8"])
    # a log axis relabels its minors as 4x10^0 etc., which collides with the
    # explicit major labels above -- suppress them
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel(r"largest lever the mechanism can produce [$\times$]")
    ax.set_ylim(-0.7, len(CAND) - 0.15)
    ax.tick_params(top=False, right=False)
    ax.axvline(1.0, color=COL["ref"], ls=":", lw=1.2, zorder=2)

    h = [mpl.patches.Patch(facecolor=COL["model"], label="reduces the deficit"),
         mpl.patches.Patch(facecolor=COL["variant"], alpha=0.75, hatch="///",
                           label="WRONG SIGN (deepens it)")]
    ax.legend(handles=h, loc="lower right", fontsize=7)
    save(fig, "03_the_audit")


# ---------------------------------------------------------------------------
# 4. THE ABSOLUTE ANCHOR -- Popeye T1a.  T1a_meanY_anchor_RESULT.md
# ---------------------------------------------------------------------------
def fig4():
    fig, ax = fig_single(2.5)
    ax.axhspan(0.7, 1.7, color=COL["shade"], alpha=0.35, zorder=0)
    ax.text(2.5, 1.66, "pre-registered healthy band", fontsize=7,
            ha="center", va="top", color="#444444")
    pts = [("BIND\nfiducial", 1.075, COL["model"]),
           ("bind_science", 1.058, COL["model"]),
           ("TNG300-hydro\ntruth", 1.064, COL["truth"])]
    for i, (lab, v, c) in enumerate(pts):
        ax.plot(i + 1, v, "o", ms=9, color=c, mec="white", mew=1.0, zorder=4)
        ax.text(i + 1, v + 0.10, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(1.6, color=COL["ref"], ls=":", lw=1.2)
    ax.text(3.45, 1.6, "published\n$\\langle y\\rangle$", fontsize=6.5,
            va="center", ha="left")
    ax.axhspan(2.5, 3.2, color=COL["variant"], alpha=0.18, zorder=0)
    ax.text(2.0, 2.85, "a projection bug would land here",
            fontsize=7, ha="center", va="center", color="#8a5a00")
    ax.set_xticks([1, 2, 3], [p[0] for p in pts])
    ax.set_ylabel(r"$\langle y\rangle$ monopole  [$10^{-6}$]")
    ax.set_xlim(0.4, 3.4)
    ax.set_ylim(0.3, 3.25)
    ax.tick_params(top=False, right=False)
    save(fig, "04_absolute_anchor")


# ---------------------------------------------------------------------------
# 5. THE A-SIDE DEFECT -- c2s_massdep.  a8_variant_chi2.json
# ---------------------------------------------------------------------------
# n folded into the tick label: a separate row of "n=" text collided with the
# two-line block names
BLOCKS = [r"X-ray $f_{\rm gas}$" "\n" r"$n=5$",
          "kSZ\n" r"$n=9$",
          r"$\kappa$-peaks $\times\,y$" "\n" r"$n=4$"]
FID = [19.8, 37.6, 106.6]
C2S = [1.8, 17.3, 86.3]
NDOF = [5, 9, 4]


def fig5():
    fig, ax = fig_single(3.0)
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, FID, w, color=COL["model"], edgecolor="white", lw=0.8,
           label="fiducial", zorder=3)
    ax.bar(x + w / 2, C2S, w, color=COL["variant"], edgecolor="white", lw=0.8,
           label="mass-dependent CylToSph", zorder=3)
    for xi, (a, b) in enumerate(zip(FID, C2S)):
        ax.text(xi - w / 2, a + 3, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, b + 3, f"{b:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, BLOCKS)
    ax.set_ylabel(r"block $\chi^2$ at its own MAP")
    ax.set_ylim(0, 128)
    ax.tick_params(top=False, right=False)
    ax.legend(loc="upper left", fontsize=7)
    # the empty region above the kSZ pair is the only collision-free spot
    ax.annotate("two blocks collapse;\nthe third does not",
                xy=(2 - w / 2 - 0.06, 92), xytext=(0.52, 62),
                fontsize=7.5, color="#333333", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#666666"))
    save(fig, "05_the_a_side_defect")


# ---------------------------------------------------------------------------
# 6. THE ANCHOR RESULT -- the kappa-peak block, every configuration ever run
# ---------------------------------------------------------------------------
CONFIGS = [("fiducial", 106.6), ("emul2x", 107.6), ("fgas_model2x", 95.9),
           ("b_coordsys2x", 94.7), ("no_ksz", 88.4), ("no_fgas", 83.1),
           ("c2s_massdep", 86.3), ("ksz_wp2bias", 101.8),
           ("ksz_wp2bias_all", 103.2)]


def fig6():
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.7))
    names = [c[0] for c in CONFIGS]
    vals = [c[1] for c in CONFIGS]
    ax.axhspan(min(vals), max(vals), color=COL["model"], alpha=0.12, zorder=0)
    ax.plot(range(len(vals)), vals, "o", ms=9, color=COL["model"],
            mec="white", mew=1.0, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.6, f"{v:.1f}", ha="center", fontsize=7)
    ax.axhline(9.49, color=COL["ref"], ls=":", lw=1.2, zorder=2)
    ax.text(len(vals) - 0.4, 13, r"$\chi^2_{95\%}$ for 4 d.o.f.",
            fontsize=7, ha="right")
    ax.set_xticks(range(len(names)), names, rotation=28, ha="right")
    ax.set_ylabel(r"$\kappa$-peak block $\chi^2$  (4 d.o.f.)")
    ax.set_ylim(0, 122)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.tick_params(top=False, right=False)
    ax.text(0.015, 0.93, "range 83.1 - 107.6 across every configuration run",
            transform=ax.transAxes, fontsize=7.5, va="top")
    save(fig, "06_the_anchor_result")


# ---------------------------------------------------------------------------
# 7. THE COVERAGE ARGUMENT -- where the model can reach vs where the data are
# ---------------------------------------------------------------------------
def fig7():
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.5))
    rows = [
        ("twobound grid",  -0.171, 0.097, COL["shade"]),
        ("Sobol box (253 units,\nfull 30-dim astro prior)", -0.3218, 0.1357,
         COL["model"]),
    ]
    for i, (lab, lo, hi, c) in enumerate(rows):
        ax.barh(i, hi - lo, left=lo, height=0.42, color=c,
                alpha=0.55 if i == 0 else 0.85, edgecolor="white", lw=0.8,
                zorder=3)
        ax.text(hi + 0.02, i, lab, va="center", fontsize=7.5)

    # the two posteriors are only 0.2 apart in x; side-by-side labels collide,
    # so stagger them vertically with leader lines
    ax.plot(-0.353, 2.0, "o", ms=9, color=COL["model"], mec="white", mew=1,
            zorder=5)
    ax.plot([-0.353, -0.353], [2.10, 2.28], lw=0.7, color="#999999", zorder=4)
    ax.text(-0.353, 2.32, r"joint posterior  $-0.353$", ha="center",
            fontsize=7, va="bottom")

    ax.plot(-0.554, 2.0, "o", ms=9, color=COL["variant"], mec="white", mew=1,
            zorder=5)
    ax.plot([-0.554, -0.554], [2.10, 2.68], lw=0.7, color="#999999", zorder=4)
    ax.text(-0.554, 2.72, r"CylToSph fix  $-0.554$", ha="center",
            fontsize=7, va="bottom", color="#8a5a00")

    ax.plot(-1.216, 2.0, "*", ms=16, color=COL["data"], mec="white", mew=0.8,
            zorder=5)
    ax.text(-1.216, 1.62, "what the data\nrequire:  $-1.22$", ha="center",
            fontsize=7.5, va="top", color=COL["data"])

    ax.annotate("", xy=(-1.216, 1.15), xytext=(-0.3218, 1.15),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color=COL["data"]))
    ax.text(-0.77, 1.22, r"factor $\sim3.8$ in required evacuation",
            ha="center", fontsize=7.5, color=COL["data"])

    ax.axvline(0.0, color=COL["ref"], ls=":", lw=1.2, zorder=2)
    ax.text(0.02, -0.55, "TNG fiducial", fontsize=6.5)
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$")
    ax.set_yticks([])
    ax.set_xlim(-1.42, 0.60)
    ax.set_ylim(-0.75, 3.35)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    # which="both": the house style puts minor ticks on all four sides, and
    # majors-only suppression leaves a stray minor rail along the top
    ax.tick_params(which="both", top=False, right=False, left=False)
    save(fig, "07_the_coverage_argument")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print("done")
