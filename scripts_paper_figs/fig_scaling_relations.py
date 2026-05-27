"""Scaling relations figure for the BIND paper (§5.1).

Truth vs BIND on five canonical halo scaling relations, evaluated on the
held-out SB35 Test suite. Reports best-fit slope, intercept, scatter, and
the per-halo residual Spearman rho — the quantities described in
scatter.ipynb cells 9 and 13.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Data load

CACHE = "analysis_physics_cache/obs_fm_two_head.npz"
OUT_PDF = "paper_figures/fig_scaling_relations.pdf"
OUT_PNG = "paper_figures/fig_scaling_relations.png"

d = np.load(CACHE, allow_pickle=True)
suite = d["suite"]
logM = d["logM"]
m_dm = d["truth_M_dm"]
m_gas_t = d["truth_M_gas"]
m_star_t = d["truth_M_star"]
m_gas_g = d["gen_M_gas"]
m_star_g = d["gen_M_star"]
m_dm_g = d["gen_M_dm"]

# Use the held-out SB35 Test set; it has the broadest parameter coverage
sel = suite == "Test"
M200 = 10 ** logM[sel]
Mgas_t = m_gas_t[sel]
Mstar_t = m_star_t[sel]
Mgas_g = m_gas_g[sel]
Mstar_g = m_star_g[sel]
Mdm_t = d["truth_M_dm"][sel]
Mdm_g = m_dm_g[sel]
Mbar_t = Mgas_t + Mstar_t
Mbar_g = Mgas_g + Mstar_g


# ---------------------------------------------------------------------------
# Helpers

def fit_loglog(x, y, mask=None, pivot=None):
    """OLS log-log fit pivoted at log10(x)=pivot.

    Returns (alpha, beta_at_pivot, sigma). With a pivot the intercept is the
    median fit value at log10(x)=pivot, so Δβ between two fits is the actual
    vertical offset at that x, not the rotation-confounded zero-intercept.
    """
    if mask is None:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    if pivot is None:
        pivot = float(np.median(lx))
    A = np.vstack([lx - pivot, np.ones_like(lx)]).T
    alpha, beta_p = np.linalg.lstsq(A, ly, rcond=None)[0]
    sigma = float(np.std(ly - (alpha * (lx - pivot) + beta_p)))
    return float(alpha), float(beta_p), sigma, mask, pivot


def residual_spearman(x, y_t, y_g):
    """Per-halo residual Spearman between truth and BIND."""
    mask = np.isfinite(x) & np.isfinite(y_t) & np.isfinite(y_g) & (x > 0) & (y_t > 0) & (y_g > 0)
    a_t, b_t, _, _, piv = fit_loglog(x, y_t, mask)
    a_g, b_g, _, _, _ = fit_loglog(x, y_g, mask, pivot=piv)
    lx = np.log10(x[mask]) - piv
    r_t = np.log10(y_t[mask]) - (a_t * lx + b_t)
    r_g = np.log10(y_g[mask]) - (a_g * lx + b_g)
    rho, _ = spearmanr(r_t, r_g)
    return float(rho)


RELS = [
    ("SHMR", M200, Mstar_t, Mstar_g, r"$M_{200c}\,[M_\odot/h]$", r"$M_\star\,[M_\odot/h]$"),
    ("GasFrac", M200, Mgas_t, Mgas_g, r"$M_{200c}\,[M_\odot/h]$", r"$M_{\rm gas}\,[M_\odot/h]$"),
    ("Mgas-Mstar", Mstar_t, Mgas_t, Mgas_g, r"$M_\star\,[M_\odot/h]$", r"$M_{\rm gas}\,[M_\odot/h]$"),
    ("BaryonFrac", M200, Mbar_t, Mbar_g, r"$M_{200c}\,[M_\odot/h]$", r"$M_{\rm bar}\,[M_\odot/h]$"),
]


# For the Mgas-Mstar panel we must use matched stellar mass on both sides;
# use TRUTH M_star as x for both truth and BIND y because the question is
# "given the same halo, do truth and BIND populate the same Mgas at fixed
# stellar mass?". Using BIND M_star on the x-axis would conflate two errors.
RELS[2] = ("Mgas-Mstar", Mstar_t, Mgas_t, Mgas_g, r"$M_\star\,[M_\odot/h]$",
           r"$M_{\rm gas}\,[M_\odot/h]$")


# ---------------------------------------------------------------------------
# Figure

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=False, sharey=False)

for ax, (label, x, y_t, y_g, xlabel, ylabel) in zip(axes, RELS):
    mask = np.isfinite(x) & np.isfinite(y_t) & np.isfinite(y_g) & (x > 0) & (y_t > 0) & (y_g > 0)
    a_t, b_t, s_t, _, piv = fit_loglog(x, y_t, mask)
    a_g, b_g, s_g, _, _ = fit_loglog(x, y_g, mask, pivot=piv)
    rho = residual_spearman(x, y_t, y_g)

    # Background scatter (down-sample to ~1500 for legibility)
    rng = np.random.default_rng(0)
    idx = np.where(mask)[0]
    if idx.size > 1500:
        idx = rng.choice(idx, 1500, replace=False)
    ax.scatter(x[idx], y_t[idx], s=4, c="0.55", alpha=0.4, rasterized=True, label="truth")
    ax.scatter(x[idx], y_g[idx], s=4, c="tab:red", alpha=0.4, rasterized=True, label="BIND")

    # Best-fit lines over the overlap support
    xs = np.geomspace(x[mask].min(), x[mask].max(), 64)
    lxs = np.log10(xs) - piv
    ax.plot(xs, 10 ** (a_t * lxs + b_t), "-", color="0.10", lw=2, zorder=5,
            label=f"truth fit: $\\alpha={a_t:.2f}$")
    ax.plot(xs, 10 ** (a_g * lxs + b_g), "--", color="tab:red", lw=2, zorder=5,
            label=f"BIND fit: $\\alpha={a_g:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(label.replace("Mgas-Mstar", r"$M_{\rm gas}$–$M_\star$"), fontsize=11)
    ax.grid(alpha=0.2, ls=":")

    # Annotate metrics in a transparent box
    txt = (
        f"$\\Delta\\alpha={a_g-a_t:+.3f}$\n"
        f"$\\Delta\\beta={b_g-b_t:+.3f}$ dex\n"
        f"$\\Delta\\sigma={s_g-s_t:+.3f}$ dex\n"
        f"$\\rho_{{\\rm res}}={rho:.2f}$"
    )
    ax.text(
        0.04, 0.96, txt, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    if ax is axes[0]:
        ax.legend(loc="lower right", fontsize=8, frameon=False)

fig.suptitle(
    "Truth vs BIND scaling relations on the held-out SB35 Test set "
    f"($N_{{\\rm halos}}={int(np.isfinite(M200).sum())}$)",
    fontsize=11, y=1.00,
)
fig.tight_layout()
os.makedirs("paper_figures", exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")
