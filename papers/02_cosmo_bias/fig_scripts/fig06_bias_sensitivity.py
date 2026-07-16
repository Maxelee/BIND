#!/usr/bin/env python
"""Fig 6 -- Sobol sensitivity of the WL bias Delta_S8 to the 30 astrophysical
feedback parameters (99 runs, ell_max=3000).

Left: ranked total-order (S_Ti) and first-order-binned (S_i) sensitivity
indices for the 12 most important parameters. Right: Delta_S8 vs the top
driver (IMFslope, normalized to [0,1]), colored by group f_gas.

Data (already-cached; no surrogate refit, no re-derivation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/bias_sensitivity.npz --
    names(30,), S_total(30,), S_first_bin(30,) [left panel, drawn directly].
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/transfer_Sell.npz --
    sobol_params/astro_idx/param_names/param_lo/param_hi/param_log/run_idx,
    used only to reconstruct the normalized top-driver column X[:, j0] (same
    load_design() transform as the source script).
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/cosmo_bias.npz --
    dS8_3000, fgas [right panel y and color axes].

Source: examples/lightcone_bias_sensitivity.py (wl-cosmo-bias worktree),
load_design() lines ~29-44, main() plotting block lines ~117-141.
Placeholder this replaces: figs/fig06_bias_sensitivity.png (md5-identical
to examples/figures_lightcone/bias_sensitivity.png).

Note: the placeholder's title carried the surrogate cross-validated R^2
(0.38); per house style (no titles), that number is dropped from the
figure itself -- it is already stated in the main text (Sec. 3.6) and is
not referenced by the figure caption.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

CACHE = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache")
ELL_MAX = 3000


def load_design():
    d = np.load(CACHE / "transfer_Sell.npz", allow_pickle=True)
    b = np.load(CACHE / "cosmo_bias.npz")
    assert np.array_equal(d["run_idx"], b["run_idx"])
    names = np.array([str(n) for n in d["param_names"]])
    ai = d["astro_idx"]
    P = d["sobol_params"][d["run_idx"]][:, ai].astype(float)
    lo, hi = d["param_lo"][ai].astype(float), d["param_hi"][ai].astype(float)
    log = d["param_log"][ai].astype(bool)
    X = P.copy()
    X[:, log] = np.log10(X[:, log])
    loN, hiN = lo.copy(), hi.copy()
    loN[log], hiN[log] = np.log10(lo[log]), np.log10(hi[log])
    X = (X - loN) / (hiN - loN)
    Y = b[f"dS8_{ELL_MAX}"]
    return X, Y, names[ai], b["fgas"]


def main() -> None:
    setup()

    s = np.load(CACHE / "bias_sensitivity.npz", allow_pickle=True)
    names_s = np.array([str(n) for n in s["names"]])
    STi, S1 = s["S_total"], s["S_first_bin"]
    order = np.argsort(STi)[::-1]
    top = order[:12][::-1]

    X, Y, names_x, fgas = load_design()
    assert list(names_x) == list(names_s)  # same 30-param ordering
    j0 = order[0]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TWO_COL[0], 3.2),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))

    yy = np.arange(len(top))
    ax.barh(yy + 0.18, STi[top], height=0.36, color=COLORS["highlight"],
            label=r"total $S_{T_i}$")
    ax.barh(yy - 0.18, S1[top], height=0.36, color=COLORS["bind"],
            label=r"first-order $S_i$")
    ax.set_yticks(yy)
    ax.set_yticklabels([names_s[j] for j in top])
    ax.set_xlabel("Sobol sensitivity index")
    ax.legend(loc="lower right")
    panel_label(ax, "(a)")

    sc = ax2.scatter(X[:, j0], Y, c=fgas, s=22, edgecolor="k", lw=0.3)
    ax2.axhline(0, color="0.6", lw=0.7)
    ax2.set_xlabel(f"{names_s[j0]} (normalized)")
    ax2.set_ylabel(r"$\Delta S_8$")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.02)
    cb.set_label(r"group $f_{\rm gas}$")
    panel_label(ax2, "(b)")

    fig.tight_layout()
    save(fig, "figs/fig06_bias_sensitivity")


if __name__ == "__main__":
    main()
