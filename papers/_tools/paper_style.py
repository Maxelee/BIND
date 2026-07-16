"""Shared publication style for ALL BIND Lightcone Suite figures.

Every figure script must use this module — it is what makes the five papers
look like one suite. Run with the BIND venv python:
    /mnt/home/mlee1/venvs/BIND_env/bin/python fig_scripts/fig01_slug.py

Usage:
    import sys
    sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
    from paper_style import setup, save, panel_label, COLORS, ONE_COL, TWO_COL

    setup()
    fig, ax = plt.subplots(figsize=ONE_COL)
    ...  # NO titles. Concise legends. Units on every axis label.
    save(fig, "figs/fig01_slug")   # -> figs/fig01_slug.pdf + figs_preview PNG
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: F401,E402

# Figure sizes in inches (width, height) for a standard 2-column preprint.
ONE_COL = (3.5, 2.63)
ONE_COL_SQ = (3.5, 3.4)
TWO_COL = (7.2, 3.1)
TWO_COL_TALL = (7.2, 5.6)

# Fixed semantic colors — identical meaning in every paper of the suite.
COLORS = {
    "bind": "#0C5DA5",  # BIND / painted / emulated
    "truth": "#111111",  # hydro truth
    "dmo": "#949494",  # dark-matter-only baseline
    "highlight": "#FF2C00",  # the single curve/point being called out
    "secondary": "#00B945",  # a second model/variant when needed
}
BAND_ALPHA = 0.25  # ensemble bands: fill_between(..., color=COLORS[...], alpha=BAND_ALPHA)


def setup():
    """Apply the suite style: scienceplots 'science' (mathtext, not usetex)."""
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "axes.grid": False,
            "image.cmap": "cividis",
        }
    )


def panel_label(ax, text, loc="upper left", color="k"):
    """'(a)'-style panel tag INSIDE the axes. Use instead of any title."""
    pos = {
        "upper left": (0.04, 0.96),
        "upper right": (0.96, 0.96),
        "lower left": (0.04, 0.04),
        "lower right": (0.96, 0.04),
    }[loc]
    ax.text(
        *pos,
        text,
        transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
    )


def save(fig, stem):
    """stem = 'figs/fig01_slug' (relative to the paper dir, no extension).

    Writes the vector PDF used by the paper plus a PNG preview (for reviewers)
    in figs_preview/ (gitignored).
    """
    p = pathlib.Path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    prev = p.parent.parent / "figs_preview"
    prev.mkdir(exist_ok=True)
    fig.savefig(prev / f"{p.name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"wrote {stem}.pdf (+ preview {prev / (p.name + '.png')})")
