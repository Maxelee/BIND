"""scatter/fig_sensitivity_mass.py  (Deliverable #1: mass dependence)

How does the parameter sensitivity of f_gas and M_star change with halo mass?

Reuses the joint design (no new sampling). For four halo-mass bins we compute the
standardized sensitivity (SRC, bootstrapped) of group/cluster gas fraction and
stellar mass to the 30 astro params, and cross-check the ranking with a model-free
distance correlation + a CV surrogate comparison (see scatter/sensitivity.py).

Output: paper_figures/fig_sensitivity_mass.{pdf,png} + ..._data.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scatter.sensitivity import (                       # noqa: E402
    load_joint_cube, population_targets, src_bootstrap, dcor_all, cv_r2_compare,
)

OUT_DIR = ROOT / "paper_figures"
MASS_EDGES = [13.0, 13.2, 13.4, 13.7, 14.8]
SN_PARAMS = {"A_SN1", "A_SN2", "WindSpecMom", "WindFreeTravelDens", "MinWindVel",
             "WindEnergyReduction", "WindEnergyReductionZ", "WindEnergyReductionExp",
             "WindDumpFac", "ThermalWind"}
AGN_PARAMS = {"A_AGN1", "A_AGN2", "BHAccretion", "BHEddington", "BHFeedback",
              "BHRadEff", "QuasarThreshold", "QuasarThreshPow", "SeedBHMass"}


def main():
    Y, masses, sub, names = load_joint_cube()
    bins = list(zip(MASS_EDGES[:-1], MASS_EDGES[1:]))
    centers = np.array([0.5 * (a + b) for a, b in bins])

    # per-bin SRC (+CI), dCor, CV R^2 for f_gas and logM*
    keys = ["f_gas", "logM_star"]
    beta = {k: np.zeros((len(bins), 30)) for k in keys}
    lo = {k: np.zeros((len(bins), 30)) for k in keys}
    hi = {k: np.zeros((len(bins), 30)) for k in keys}
    dcor = {k: np.zeros((len(bins), 30)) for k in keys}
    cvr2 = {k: [] for k in keys}
    rho_agree = {k: [] for k in keys}
    nhalo = []

    for bi, mb in enumerate(bins):
        tg, n = population_targets(Y, masses, mb)
        nhalo.append(n)
        for k in keys:
            b, R2, l, h = src_bootstrap(sub, tg[k])
            beta[k][bi], lo[k][bi], hi[k][bi] = b, l, h
            dc = dcor_all(sub, tg[k]); dcor[k][bi] = dc
            cvr2[k].append(cv_r2_compare(sub, tg[k]))
            rho_agree[k].append(spearmanr(np.abs(b), dc).correlation)
        print(f"bin {mb} (N={n}): "
              + "  ".join(f"{k} CV[lin/GP/GBM]="
                          f"{cvr2[k][bi]['linear']:.2f}/{cvr2[k][bi]['GP']:.2f}/{cvr2[k][bi]['GBM']:.2f}"
                          f" rank-agree(SRC,dCor)={rho_agree[k][bi]:.2f}"
                          for k in keys))

    # pick top drivers by max |SRC| over bins (per target)
    def top_drivers(k, n_top=7):
        imp = np.abs(beta[k]).max(0)
        return np.argsort(imp)[::-1][:n_top]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True)
    titles = {"f_gas": r"$f_{\rm gas}$", "logM_star": r"$\log M_\star$"}
    cmap = matplotlib.colormaps["tab10"]

    for ax, k in zip(axes, keys):
        drivers = top_drivers(k)
        for j, p in enumerate(drivers):
            c = cmap(j % 10)
            ls = "-" if names[p] in SN_PARAMS else ("--" if names[p] in AGN_PARAMS else ":")
            ax.plot(centers, beta[k][:, p], ls, marker="o", color=c, lw=2, ms=5,
                    label=names[p])
            ax.fill_between(centers, lo[k][:, p], hi[k][:, p], color=c, alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\log_{10} M_{\rm 200c}$ (bin centre)")
        ax.set_ylabel("standardized sensitivity (SRC)")
        ax.set_title(f"Sensitivity of {titles[k]} vs halo mass")
        ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)
        ax.grid(alpha=0.25)
        # robustness footnote
        cvmin = min(cvr2[k][bi]["linear"] for bi in range(len(bins)))
        ax.text(0.02, 0.02,
                f"linestyle: — SN  ·· AGN  : other\n"
                f"linear CV $R^2\\geq${cvmin:.2f}; SRC–dCor rank-agree "
                f"{np.mean(rho_agree[k]):.2f}",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    nstr = ", ".join(f"{c:.2f}:N={n}" for c, n in zip(centers, nhalo))
    fig.suptitle("Mass dependence of feedback sensitivity (group → cluster scale)   "
                 f"[halos per bin — {nstr}]", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_sensitivity_mass.pdf"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"[saved] {pdf} (+ .png)")

    np.savez_compressed(
        OUT_DIR / "fig_sensitivity_mass_data.npz",
        param_names=np.array(names), mass_edges=np.array(MASS_EDGES),
        bin_centers=centers, nhalo=np.array(nhalo),
        src_fgas=beta["f_gas"], src_logMstar=beta["logM_star"],
        dcor_fgas=dcor["f_gas"], dcor_logMstar=dcor["logM_star"],
    )
    print(f"[saved] {OUT_DIR / 'fig_sensitivity_mass_data.npz'}")


if __name__ == "__main__":
    main()
