#!/usr/bin/env python3
"""Rebuild fig_mass_error (integrated-mass residual violins) from the paper cache.

Model-agnostic: the cache is selected entirely by the PAPER_* env vars (see
below); the mass_table.pkl schema is shared across model caches.

Plotting code copied verbatim from examples/paper_figures.ipynb cell 16
(the violin variant): rows = DM (hydro) / Gas / Stars / Total, columns =
r <= R200c / full patch, CV/1P/SB35 violins per panel, white dot = median,
thick bar = 25-75, residuals clipped to [-1, 1] for display.

Run:
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    export PAPER_SUITE_ROOT=... PAPER_MODEL_SUBDIR=... PAPER_MASS_DIR=... PAPER_MODEL_TAG=...
    python fig_mass_error.py
"""
import os
import pickle
import sys
from pathlib import Path

# Model selection comes from the environment (must be set before importing
# paper_config) — no hardcoded model defaults.
_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR", "PAPER_MASS_DIR", "PAPER_MODEL_TAG")
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    sys.exit(f"set env vars before running: {' '.join(_missing)}")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_config as C

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])
except Exception:
    pass

SUITES = C.SUITES
SUITE_COLORS = C.SUITE_COLORS
SUITE_DISPLAY = C.SUITE_DISPLAY

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        out = FIG_DIR / f"{name}.{e}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")


# ── plotting code copied from examples/paper_figures.ipynb cell 16 ──────────
def fig2_mass_error(tbl, resid_lim=1.0, save=True):
    suites = list(SUITES)
    col_keys   = ['DM_hydro', 'Gas', 'Stars', 'Total']
    col_titles = ['DM (hydro)', 'Gas', 'Stars', 'Total']
    row_suffixes = ['_rvir', '']
    row_titles   = [r'$r\leq R_\mathrm{200}$', 'Full patch']

    fig, axes = plt.subplots(4, 2, figsize=(8.6, 4.3 * 4), sharex=True, sharey=True)

    for col, (suffix, col_title) in enumerate(zip(row_suffixes, row_titles)):
        for row, (ch, ch_title) in enumerate(zip(col_keys, col_titles)):
            ax = axes[row, col]

            data_per_suite = []
            for s in suites:
                sub = tbl[tbl['suite'] == s]
                if ch == 'Total':
                    t = (sub[f'truth_DM_hydro{suffix}']
                         + sub[f'truth_Gas{suffix}']
                         + sub[f'truth_Stars{suffix}']).to_numpy()
                    g = (sub[f'gen_DM_hydro{suffix}']
                         + sub[f'gen_Gas{suffix}']
                         + sub[f'gen_Stars{suffix}']).to_numpy()
                else:
                    t = sub[f'truth_{ch}{suffix}'].to_numpy()
                    g = sub[f'gen_{ch}{suffix}'].to_numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = (g - t) / t
                r = r[np.isfinite(r)]
                r = np.clip(r, -resid_lim, resid_lim)
                data_per_suite.append(r)

            positions = list(range(len(suites)))

            parts = ax.violinplot(
                data_per_suite, positions=positions,
                widths=0.7, showmedians=False, showextrema=False,
            )
            for pc, s in zip(parts['bodies'], suites):
                pc.set_facecolor(SUITE_COLORS[s])
                pc.set_edgecolor('k')
                pc.set_linewidth(0.6)
                pc.set_alpha(0.65)

            for i, (r, s) in enumerate(zip(data_per_suite, suites)):
                if len(r) == 0:
                    continue
                q1, med, q3 = np.percentile(r, [25, 50, 75])
                ax.plot([i, i], [q1, q3], color='k', lw=3, zorder=5,
                        solid_capstyle='round')
                ax.scatter([i], [med], color='white', s=35, zorder=6,
                           edgecolors='k', linewidths=1.0)

            ax.axhline(0.0, color='k', lw=0.8, ls='--', alpha=0.6)
            ax.set_ylim(-resid_lim * 1.05, resid_lim * 1.05)
            ax.grid(axis='y', alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)

            # column headers on top row
            if row == 0:
                ax.set_title(col_title)
            # row labels on left column
            if col == 0:
                ax.set_ylabel(
                    ch_title + '\n' + r'$\Delta M / M_\mathrm{Truth}$',

                )
            # x tick labels on bottom row only
            if row == len(col_keys) - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels([SUITE_DISPLAY[s] for s in suites])
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels([])

    plt.tight_layout()
    if save:
        save_fig(fig, 'fig_mass_error')
    return fig


def main():
    tbl = pickle.load(open(C.CACHE_DIR / "mass_table.pkl", "rb"))
    # Spine table is already restricted to the trained regime (>= 1e13);
    # verify instead of re-cutting blindly.
    assert (tbl["log_m200c"] >= 13.0).all(), "spine mass_table has halos below 1e13"
    n = tbl["suite"].value_counts().to_dict()
    print(f"spine mass_table: {len(tbl)} halos  {n}")

    # Print the per-panel medians used by the caption
    for suffix, ap in [("_rvir", "r<=R200c"), ("", "full patch")]:
        for ch in ["DM_hydro", "Gas", "Stars", "Total"]:
            row = []
            for s in SUITES:
                sub = tbl[tbl["suite"] == s]
                if ch == "Total":
                    t = sum(sub[f"truth_{c}{suffix}"] for c in C.MASS_CHANNELS).to_numpy()
                    g = sum(sub[f"gen_{c}{suffix}"] for c in C.MASS_CHANNELS).to_numpy()
                else:
                    t = sub[f"truth_{ch}{suffix}"].to_numpy()
                    g = sub[f"gen_{ch}{suffix}"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = (g - t) / t
                r = r[np.isfinite(r)]
                row.append(f"{SUITE_DISPLAY[s]} {100 * np.median(r):+.2f}%")
            print(f"  {ap:11s} {ch:9s} median: " + "  ".join(row))

    fig2_mass_error(tbl)


if __name__ == "__main__":
    main()
