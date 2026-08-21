#!/usr/bin/env python
"""REPLACEMENT for fig_radial_pct_diff.pdf (paper Sec. "Radial Profiles").

3 rows (DM, Gas, Stars) x 2 columns, built from the $PAPER_MODEL_TAG cache
(see tools/paper_cache/paper_config.py env overrides) profile partials:

    left  : median per-halo fractional error  (Sigma_gen - Sigma_hydro)/Sigma_hydro
            vs projected radius, 16-84 halo-to-halo band, per suite
            (the current paper style, previously a 3x1 figure);
    right : median |Sigma_gen/Sigma_hydro - 1| vs radius on a log y-axis,
            with 10% and 20% reference lines, per suite.

Estimator copied from tools/paper_cache/build_metric.py::_profile_bands and the
fig_radial_pct_diff cell of examples/paper_figures2_lowmass.ipynb:
zero-truth annuli (Sigma_hydro <= 0) are set to NaN and excluded via
nanmedian/nanpercentile.  Binning (build_metric.py compute_profiles ->
bind.metrics.radial_profile(n_bins=32, logspace=False)): 32 LINEAR annuli,
bin edges linspace(0, 64, 33) patch pixels, centres 0.0488-3.0762 Mpc/h.

Reads  : $PAPER_CACHE partials/profiles/*.npz  (per-sim per-halo profiles)
Writes : examples/paper_figures/fig_radial_pct_diff.{pdf,png}
"""
from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))
import paper_config as C  # noqa: E402

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
PARTIALS = C.PARTIAL_DIR / "profiles"

SUITE_COLORS = C.SUITE_COLORS          # CV green, 1P blue, Test/SB35 red
SUITE_DISPLAY = C.SUITE_DISPLAY
CH_NAMES = ["DM (hydro)", "Gas", "Stars"]

# ── style (identical to examples/_build_paper_nbs.py SETUP) ─────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook"])
except Exception:
    pass


def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", name)


# ── load + pool the spine profile partials per suite ────────────────────────
def load_pooled():
    pooled = defaultdict(lambda: {"t": [], "g": [], "mb": []})
    r = None
    for f in sorted(glob.glob(str(PARTIALS / "*.npz"))):
        z = np.load(f)
        if bool(z["_skip"]):
            continue
        suite = Path(f).name.split("__")[0]
        if r is None:
            r = z["r"]
        else:
            assert np.allclose(r, z["r"])
        pooled[suite]["t"].append(z["truth"])
        pooled[suite]["g"].append(z["gen"])
        pooled[suite]["mb"].append(z["mass_bin"])
    out = {}
    for s, d in pooled.items():
        t, g = np.concatenate(d["t"]), np.concatenate(d["g"])
        mb = np.concatenate(d["mb"])
        # spine cache is already cut at 1e13 (mass_threshold_1p000e13):
        # every halo must land in a >=1e13 mass bin (edges 13.0/13.5/14.0/15.5)
        assert (mb >= 2).all(), f"{s}: halo below the 1e13 training cut"
        out[s] = (t, g)
    # confirm the binning claim: 32 linear annuli on the 128^2 patch
    edges_pix = np.linspace(0, 64, 33)
    cen = 0.5 * (edges_pix[:-1] + edges_pix[1:]) * C.MPC_PER_PIX_PATCH
    assert np.allclose(r, cen), "r grid is not 32 linear annuli"
    return r, out


def pdiff_of(t, g):
    """Per-halo fractional error; zero-truth annuli -> NaN (as in the builder)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, (g - t) / t, np.nan)


# ── stats (printed; captions are written from these) ────────────────────────
def print_stats(r, pooled):
    print(f"\nannuli: {len(r)} linear, centres {r[0]:.4f}-{r[-1]:.4f} Mpc/h")
    for s in ("CV", "1P", "Test"):
        t, g = pooled[s]
        pd_ = pdiff_of(t, g)
        print(f"\n== {SUITE_DISPLAY[s]}  N={len(t)} halos ==")
        for c, ch in enumerate(CH_NAMES):
            x = pd_[:, c, :]
            med = np.nanmedian(x, 0)
            p16 = np.nanpercentile(x, 16, 0)
            p84 = np.nanpercentile(x, 84, 0)
            am = np.nanmedian(np.abs(x), 0)
            fin = np.isfinite(x)
            f10 = np.mean(np.abs(x[fin]) <= 0.10)
            f20 = np.mean(np.abs(x[fin]) <= 0.20)
            nz = int((t[:, c, :] <= 0).sum())
            print(f"  {ch:11s} inner med {med[0]:+.3f} | band envelope "
                  f"[{p16.min():+.3f},{p84.max():+.3f}] | med|.| "
                  f"[{am.min():.3f},{am.max():.3f}] | within 10/20%: "
                  f"{f10:.1%}/{f20:.1%} | zero-truth annuli {nz}"
                  f" ({nz / x.size:.2e})")


# ── the figure ──────────────────────────────────────────────────────────────
def make_fig(r, pooled):
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 9.9), sharex=True,
                             gridspec_kw={"wspace": 0.24, "hspace": 0.08})
    for c, ch in enumerate(CH_NAMES):
        axL, axR = axes[c]
        for s in ("CV", "1P", "Test"):
            t, g = pooled[s]
            x = pdiff_of(t, g)[:, c, :]
            med = np.nanmedian(x, 0)
            p16 = np.nanpercentile(x, 16, 0)
            p84 = np.nanpercentile(x, 84, 0)
            am = np.nanmedian(np.abs(x), 0)
            col = SUITE_COLORS[s]
            axL.fill_between(r, p16, p84, color=col, alpha=0.15, lw=0)
            axL.plot(r, med, color=col, lw=1.8,
                     label=SUITE_DISPLAY[s] if c == 0 else None)
            axR.plot(r, am, color=col, lw=1.8)
        # left column — current paper style
        axL.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        axL.set_ylim(-1, 1)
        axL.set_ylabel(rf"$\Delta \Sigma_{{\rm {ch}}} / \Sigma_{{\rm {ch}}}$")
        # right column — median absolute error, log axis + 10/20% references
        axR.set_yscale("log")
        axR.set_ylim(1.5e-2, 1.2)
        for lev, ls in ((0.10, ":"), (0.20, "--")):
            axR.axhline(lev, color="0.35", lw=1.0, ls=ls, alpha=0.8)
        if c == 0:
            axR.text(r[-1], 0.104, r"$10\%$", ha="right", va="bottom",
                     fontsize=11, color="0.35")
            axR.text(r[-1], 0.208, r"$20\%$", ha="right", va="bottom",
                     fontsize=11, color="0.35")
        axR.set_ylabel(r"med $|\Sigma_{\rm gen}/\Sigma_{\rm hydro} - 1|$")
        axR.text(0.04, 0.90, ch, transform=axR.transAxes, ha="left",
                 va="top", fontsize=13)
        for ax in (axL, axR):
            ax.set_xscale("log")
            ax.grid(which="both", alpha=0.2)
            ax.set_axisbelow(True)
        if c == 0:
            axL.legend(loc="upper right", fontsize=12)
        if c == 2:
            for ax in (axL, axR):
                ax.set_xlabel(r"$r$ [Mpc $h^{-1}$]")
    save_fig(fig, "fig_radial_pct_diff")
    return fig


if __name__ == "__main__":
    r, pooled = load_pooled()
    print_stats(r, pooled)
    make_fig(r, pooled)
