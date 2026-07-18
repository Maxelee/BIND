"""BIND Paper 3 figure style: palette roles, sizes, and plot helpers.

Usage::

    from analysis.paper3a.style import apply, COL, fig_single, data_points
    apply()                       # loads paper3.mplstyle
    fig, ax = fig_single()        # one MNRAS column

Color is assigned BY ROLE, never cycled (validated Okabe-Ito subset,
CVD-safe; the three light hues carry a contrast warning against white and
are used only as fills/bands or labeled markers, never bare thin lines).
The full human guide lives in the plans repo: ``FIGURE_STYLE.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_STYLE = Path(__file__).parent / "paper3.mplstyle"

# ---------------------------------------------------------------- palette --
# roles are fixed across BOTH papers; never reassign, never cycle
COL = {
    "model":     "#0072B2",   # BIND / model curves & bands (blue)
    "data":      "#D55E00",   # measured data points (vermilion)  [or black]
    "data_alt":  "#000000",   # data when overplotted on blue bands (stars)
    "truth":     "#009E73",   # hydro-truth / validation reference (green)
    "variant":   "#E69F00",   # secondary model variant (orange; fills/labeled)
    "aux":       "#56B4E9",   # auxiliary probe (sky; fills/labeled only)
    "extra":     "#CC79A7",   # last resort (pink; fills/labeled only)
    "shade":     "#bdbdbd",   # caveat/exclusion regions (grey fill, alpha .35)
    "ref":       "#000000",   # dotted reference lines (cosmic mean, unity)
}

# journal column widths (MNRAS: 240 pt / 504 pt)
W_SINGLE = 3.33     # inches
W_DOUBLE = 7.00
H_STD = 2.9         # default single-panel height


def apply() -> None:
    plt.style.use(str(_STYLE))


def fig_single(height: float = H_STD):
    return plt.subplots(figsize=(W_SINGLE, height))

def fig_double(height: float = H_STD, ncols: int = 2, **kw):
    return plt.subplots(1, ncols, figsize=(W_DOUBLE, height), **kw)


# ---------------------------------------------------------------- helpers --

def spaghetti(ax, x, curves, color=COL["model"], alpha=0.18, lw=0.5, label=None):
    """The design-family background: many thin transparent curves."""
    for i, c in enumerate(np.atleast_2d(curves)):
        ax.plot(x, c, color=color, alpha=alpha, lw=lw,
                label=label if i == 0 else None, zorder=1)


def band(ax, x, lo, hi, color=COL["model"], alpha=0.25, label=None):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, label=label,
                    zorder=2)


def fiducial_line(ax, x, y, color=COL["model"], label="BIND fiducial"):
    ax.plot(x, y, color=color, lw=2.2, marker="o", ms=4.5, label=label, zorder=4)


def data_points(ax, x, y, yerr=None, color=COL["data_alt"], marker="*",
                ms=9, label=None):
    """Measured data: always with error bars and caps, always on top."""
    ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=ms, color=color,
                elinewidth=1.1, capsize=2.5, zorder=6, label=label)


def refline(ax, y=None, x=None, label=None):
    kw = dict(color=COL["ref"], lw=0.9, ls=":", zorder=3)
    if y is not None:
        ax.axhline(y, **kw)
        if label:
            ax.annotate(label, (0.99, y), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7)
    if x is not None:
        ax.axvline(x, **kw)


def caveat_region(ax, x0, x1, label=None):
    ax.axvspan(x0, x1, color=COL["shade"], alpha=0.35, lw=0, zorder=0)
    if label:
        ax.annotate(label, ((x0 + x1) / 2, 0.03), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="#666666")


def condition_tag(ax, text):
    """Top-center in-axes context label (the 'z = 0.18 - BGS' convention)."""
    ax.annotate(text, (0.5, 0.97), xycoords="axes fraction",
                ha="center", va="top", fontsize=8.5)
