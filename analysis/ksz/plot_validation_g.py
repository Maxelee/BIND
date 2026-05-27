"""Plot validation G — HMF coverage / mass-range constraint.

Shows per-suite halo counts per mass bin (totals across the suite + per-sim
distribution), and flags the median halos-per-sim per bin.  Bins with
median < ``--min_per_sim`` are highlighted as HMF-limited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min_per_sim", type=float, default=1.0,
                   help="Median halos/sim required per bin to be considered "
                        "well-resolved (default 1.0).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.input, allow_pickle=True)
    centers = d["mass_centers"]
    edges = d["mass_edges"]
    suites = d["suites"].astype(str).tolist()

    fig, (ax_tot, ax_med) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    width = 0.8 / len(suites)
    colors = [f"C{i}" for i in range(len(suites))]

    summary_lines = ["# Validation G — HMF coverage"]
    for i, s in enumerate(suites):
        counts = d[f"counts_{s}"]
        sim_counts = d[f"sim_counts_{s}"]  # (n_sims, n_bins)
        x = np.arange(len(centers)) + (i - (len(suites) - 1) / 2) * width
        ax_tot.bar(x, counts, width, color=colors[i], label=f"{s} (Σ over sims)")

        if sim_counts.size:
            med = np.median(sim_counts, axis=0)
            lo = np.percentile(sim_counts, 16, axis=0)
            hi = np.percentile(sim_counts, 84, axis=0)
        else:
            med = lo = hi = np.zeros(len(centers))
        ax_med.errorbar(
            x, med, yerr=[med - lo, hi - med], fmt="o", color=colors[i],
            label=f"{s} (median ± 16-84%)",
        )

        for b in range(len(centers)):
            flag = " ⚠ HMF-LIMITED" if med[b] < args.min_per_sim else ""
            summary_lines.append(
                f"{s}  logM~{np.log10(centers[b]):.2f}  "
                f"total={int(counts[b])}  median/sim={med[b]:.1f}"
                f" (16-84%={lo[b]:.1f}-{hi[b]:.1f}){flag}"
            )

    for ax in (ax_tot, ax_med):
        ax.set_xticks(np.arange(len(centers)))
        ax.set_xticklabels([f"{np.log10(c):.1f}" for c in centers])
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.2)
        ax.legend(fontsize=8, frameon=False)

    ax_med.axhline(args.min_per_sim, color="k", lw=0.5, ls="--",
                   label=f"min/sim = {args.min_per_sim}")
    ax_tot.set_ylabel("# halos (sum over sims)")
    ax_med.set_ylabel("# halos per sim")
    ax_med.set_xlabel(r"$\log_{10}(M / [M_\odot/h])$ bin centre")
    fig.suptitle("Validation G — HMF coverage")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")

    txt = "\n".join(summary_lines)
    print(txt)
    args.out.with_suffix(".txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
