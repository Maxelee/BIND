"""Plot validation C — 35-bar τ–parameter Spearman comparison (BIND vs truth).

Renders the all-halo and per-mass-bin entries from analysis.ksz.validation_c
as side-by-side bar charts, with truth in C0 and BIND in C3.  The off-diagonal
discrepancy is the diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_only", action="store_true",
                   help="Plot only the all-halo panel.")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.input, allow_pickle=False)
    n_p = int(d["n_params"])
    idx = d["param_idx"]
    labels = d["labels"].astype(str)
    rho_b = d["rho_bind"]
    rho_t = d["rho_truth"]
    n_per_bin = d["n_per_bin"]

    n_panels = 1 if args.top_only else len(labels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(0.30 * n_p + 2, 2.6 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]
    width = 0.4

    for i, ax in enumerate(axes):
        ax.bar(idx - width / 2, rho_t[i], width, color="C0", alpha=0.85,
               label="truth")
        ax.bar(idx + width / 2, rho_b[i], width, color="C3", alpha=0.85,
               label="BIND")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title(f"{labels[i]}  (n={int(n_per_bin[i])})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper right")

    axes[-1].set_xticks(idx)
    axes[-1].set_xlabel("CAMELS parameter index (0–34)")
    fig.suptitle("Validation C — Spearman τ–parameter sensitivity")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")

    # Quick text summary: top-5 most-correlated params for truth and BIND
    top_n = 5
    rho_t_all = rho_t[0]
    rho_b_all = rho_b[0]
    order_t = np.argsort(-np.abs(np.nan_to_num(rho_t_all)))[:top_n]
    order_b = np.argsort(-np.abs(np.nan_to_num(rho_b_all)))[:top_n]
    lines = ["# Top-N |Spearman ρ| with τ (all halos)"]
    lines.append(f"truth: " + ", ".join(f"p{j}({rho_t_all[j]:+.2f})" for j in order_t))
    lines.append(f"BIND : " + ", ".join(f"p{j}({rho_b_all[j]:+.2f})" for j in order_b))
    txt = "\n".join(lines)
    print(txt)
    args.out.with_suffix(".txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
