"""Plot validation A: per-halo BIND τ vs truth τ, coloured by halo mass.

Reads the npz produced by analysis.ksz.validation_a and writes one PDF + a
short text summary of the bias/scatter per mass bin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="npz from analysis.ksz.validation_a")
    p.add_argument("--out", type=Path, required=True, help="Output PDF path.")
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 3e13, 1e14, 1e15],
                   help="Halo mass bin edges [Msun/h].")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.input, allow_pickle=True)
    m = d["halo_mass_msun_h"]
    tau_bind = d["bind_tau_ap"]
    tau_true = d["truth_tau_ap"]

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    bin_idx = np.digitize(m, edges) - 1  # 0..len(edges)-2 for in-range

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    cmap = plt.get_cmap("viridis")
    n_bins = len(edges) - 1
    summary_lines = ["# Validation A — per-halo τ recovery"]
    for b in range(n_bins):
        sel = bin_idx == b
        if not sel.any():
            continue
        t_b = tau_bind[sel]
        t_t = tau_true[sel]
        valid = (t_t > 0) & (t_b > 0)
        if not valid.any():
            continue
        color = cmap(b / max(1, n_bins - 1))
        label = f"$\\log M \\in [{np.log10(edges[b]):.2f},{np.log10(edges[b+1]):.2f}]$  (n={sel.sum()})"
        ax.scatter(t_t[valid], t_b[valid], s=8, alpha=0.5, color=color, label=label)
        # log-residual stats
        d_log = np.log10(t_b[valid] / t_t[valid])
        summary_lines.append(
            f"bin {b}: n={int(valid.sum())}  median_dex={np.median(d_log):+.3f}  "
            f"std_dex={np.std(d_log):.3f}  mean_dex={np.mean(d_log):+.3f}"
        )

    # 1:1 line
    finite = np.isfinite(tau_true) & np.isfinite(tau_bind) & (tau_true > 0) & (tau_bind > 0)
    if finite.any():
        lo = max(np.percentile(tau_true[finite], 1), 1e-12)
        hi = np.percentile(tau_true[finite], 99)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{\rm truth}$  (aperture)")
    ax.set_ylabel(r"$\tau_{\rm BIND}$  (aperture)")
    ax.set_title("Validation A — per-halo τ recovery")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    txt = "\n".join(summary_lines)
    print(txt)
    args.out.with_suffix(".txt").write_text(txt + "\n")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
