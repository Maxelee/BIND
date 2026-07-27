#!/usr/bin/env python
"""P6a Fig M3 (docs/ksz_lightcone_map_plan.md §3-P6): the kappa-CAP mass
anchor. kappa (source plane z_s=1.0, just behind the BGS shell) is a
feedback-blind tracer of total matter -- its Sobol-node spread should be
small compared to tau/y (which respond to the 30-dim CAMELS-TNG feedback
sweep), and its amplitude should (a) be positive at small aperture and
(b) grow with the M* cut (bgs1125 selects more massive hosts than bgs110).

Panel (a): kappa-CAP(theta) -- 253-node envelope + fiducial, both BGS
samples overlaid (no external data; kappa isn't confronted against a
lensing measurement here, this is an internal consistency/anchor check).
Panel (b): fractional Sobol-node 16-84% spread, (P84-P16)/median at
theta/theta200=1.0, for kappa vs tau vs y (beamless merges, so tau's only
available variant sets the comparison basis), grouped by sample -- the
plan's headline "kappa spread << tau/y spread" claim, quantified.

Uses the raw shard CAP units (dimensionless pixel-sum compensated aperture,
`lightcone_cap_stack.cap_on_map` convention) throughout -- no arcmin^2 flux
conversion is needed here since only *relative* (fractional) spread and
*sign/ordering* checks are made, both invariant to an overall unit choice.

Usage
-----
    python examples/lightcone_m3_kappa_anchor.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = ["bgs110", "bgs1125"]
SAMPLE_LABEL = {"bgs110": r"BGS $M_\star{>}10^{11.0}$", "bgs1125": r"BGS $M_\star{>}10^{11.25}$"}
SAMPLE_COLOR = {"bgs110": "tab:blue", "bgs1125": "tab:green"}
MAP_COLOR = {"kappa": "tab:purple", "tau": "tab:orange", "y": "tab:red"}
N_XB = 18


def load(name):
    return np.load(LIGHTCONE / name, allow_pickle=True)


def main():
    kcap = load("kcap_lightcone.npz")
    taucap = load("taucap_lightcone.npz")
    ycap = load("ycap_lightcone.npz")   # beamless, matches tau (no systematic tau-beam sweep)

    theta_value = kcap["theta_value"]
    theta_kind = kcap["theta_kind"]
    assert np.array_equal(theta_value, taucap["theta_value"]) and np.array_equal(theta_value, ycap["theta_value"])
    r200_idx = np.nonzero(theta_kind == "r200mult")[0]
    r200_grid = theta_value[r200_idx]
    idx_theta1 = r200_idx[np.argmin(np.abs(r200_grid - 1.0))]  # global index of theta/theta200=1.0

    metrics = {"theta_value_xb": theta_value[:N_XB].tolist(), "theta200_index_used": int(idx_theta1)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ------------------------------------------------------------------
    # panel (a): kappa-CAP(theta) envelope + fiducial, both samples
    # ------------------------------------------------------------------
    ax = axes[0]
    sanity = {}
    for sample in SAMPLES:
        xb_theta = theta_value[:N_XB]
        sb35_mean = kcap[f"sb35_mean_{sample}"][:, :N_XB]
        fid_mean = kcap[f"fid_mean_{sample}"][:N_XB]
        fid_real = kcap[f"fid_real_{sample}"][:, :N_XB]
        lo16, hi84 = np.nanpercentile(fid_real, [16, 84], axis=0)

        n_plotted = 0
        for row in sb35_mean:
            if np.all(np.isfinite(row)):
                ax.plot(xb_theta, row, "-", color=SAMPLE_COLOR[sample], lw=0.4, alpha=0.10, zorder=1)
                n_plotted += 1
        ax.plot([], [], "-", color=SAMPLE_COLOR[sample], lw=1.0, alpha=0.5, label=f"{sample} SB35 ({n_plotted})")
        ax.fill_between(xb_theta, lo16, hi84, color=SAMPLE_COLOR[sample], alpha=0.20, zorder=2)
        ax.plot(xb_theta, fid_mean, "-o", color=SAMPLE_COLOR[sample], lw=2.4, ms=4, zorder=4,
                label=f"{sample} fiducial")

        sanity[sample] = {
            "fid_value_at_smallest_theta": float(fid_mean[0]),
            "positive_at_smallest_theta": bool(fid_mean[0] > 0),
            "fid_value_at_theta200_1": float(np.interp(1.0, xb_theta, fid_mean)),
        }
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$\theta_d/\theta_{200}$")
    ax.set_ylabel(r"$\kappa$-CAP  [raw pixel-sum, $z_s{=}1.0$]")
    ax.set_title("(a) kappa mass anchor: 253-node envelope + fiducial", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    sanity["grows_with_Mstar_cut"] = bool(
        sanity["bgs1125"]["fid_value_at_theta200_1"] > sanity["bgs110"]["fid_value_at_theta200_1"])
    sanity["both_positive_at_smallest_theta"] = bool(
        sanity["bgs110"]["positive_at_smallest_theta"] and sanity["bgs1125"]["positive_at_smallest_theta"])
    metrics["sanity"] = sanity
    tick = "OK" if (sanity["both_positive_at_smallest_theta"] and sanity["grows_with_Mstar_cut"]) else "CHECK"

    # ------------------------------------------------------------------
    # panel (b): fractional 16-84% Sobol-node spread at theta200=1,
    # kappa vs tau vs y, grouped by sample
    # ------------------------------------------------------------------
    ax = axes[1]
    products = {"kappa": kcap, "tau": taucap, "y": ycap}
    spread = {s: {} for s in SAMPLES}
    x = np.arange(len(SAMPLES))
    width = 0.25
    for k, (map_type, prod) in enumerate(products.items()):
        vals = []
        for sample in SAMPLES:
            node_vals = prod[f"sb35_mean_{sample}"][:, idx_theta1]
            node_vals = node_vals[np.isfinite(node_vals)]
            p16, p50, p84 = np.nanpercentile(node_vals, [16, 50, 84])
            frac = (p84 - p16) / abs(p50) if p50 != 0 else np.nan
            spread[sample][map_type] = {"p16": float(p16), "p50": float(p50), "p84": float(p84),
                                          "frac_16_84_over_median": float(frac), "n_nodes": int(len(node_vals))}
            vals.append(frac)
        ax.bar(x + (k - 1) * width, vals, width, color=MAP_COLOR[map_type], label=map_type)
    ax.set_xticks(x); ax.set_xticklabels([SAMPLE_LABEL[s] for s in SAMPLES], fontsize=8)
    ax.set_ylabel(r"fractional 16-84% Sobol spread, $(P_{84}-P_{16})/P_{50}$")
    ax.set_title(r"(b) node-to-node spread @ $\theta/\theta_{200}{=}1$: $\kappa \ll \tau, y$?", fontsize=10)
    ax.legend(fontsize=8)
    for k, sample in enumerate(SAMPLES):
        kap = spread[sample]["kappa"]["frac_16_84_over_median"]
        tmax = max(spread[sample]["tau"]["frac_16_84_over_median"], spread[sample]["y"]["frac_16_84_over_median"])
        ax.text(k, ax.get_ylim()[1] * 0.02, f"{tmax/kap:.0f}x" if kap else "n/a",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    metrics["kappa_vs_tau_vs_y_spread"] = spread

    fig.suptitle("M3 -- kappa-CAP mass anchor: feedback-blind vs tau/y feedback-sensitive spread", fontsize=12, fontweight="bold", y=1.02)
    caption = (f"sanity [{tick}]: kappa>0 at smallest theta = {sanity['both_positive_at_smallest_theta']}; "
               f"bgs1125 > bgs110 CAP @ theta200=1 = {sanity['grows_with_Mstar_cut']}.  "
               "High-tail outlier lines in (a) include runs 64/87, the same extreme-feedback nodes with 0 "
               "bgs1125 galaxies (P5) -- likely M*-cut selection bias toward rarer/more-massive halos at "
               "those nodes, not a units bug.")
    import textwrap
    fig.text(0.5, -0.02, "\n".join(textwrap.wrap(caption, width=175)), ha="center", va="top", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig_path = FIG_DIR / "M3_kappa_anchor.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[M3] wrote {fig_path}")

    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
