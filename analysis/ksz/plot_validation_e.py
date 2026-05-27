"""Plot validation E — per-parameter coverage vs nominal level."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARAM_NAMES = [f"p{i:02d}" for i in range(35)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.input, allow_pickle=True)
    coverage = z["coverage"]
    cov_err = z["coverage_err"]
    level = float(z["nominal_level"])
    abs_bias = z["abs_bias_in_sigma"]
    constraint = z["constraint"] if "constraint" in z.files else np.zeros_like(coverage)
    n_sims = int(z["n_sims"])
    n_real = int(z["n_realizations"])

    p = len(coverage)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0], "hspace": 0.10},
    )

    xs = np.arange(p)
    informed = constraint > 0.1
    colors = ["C0" if inf else "0.7" for inf in informed]
    ax1.bar(xs, coverage, yerr=cov_err, color=colors, edgecolor="k", lw=0.4)
    ax1.axhline(level, color="k", ls="--", lw=1.0,
                label=f"nominal = {level:.3f}")
    ax1.set_ylabel("coverage of 1-σ CI")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title(
        f"Validation E — LOO coverage  ({n_sims} sims × {n_real} reals)  "
        f"— grey = prior-dominated (constraint ≤ 0.1)"
    )
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(xs, constraint, color="C0", edgecolor="k", lw=0.4)
    ax2.axhline(0.1, color="k", ls=":", lw=0.8)
    ax2.set_ylabel(r"$1-\sigma_\mathrm{post}/\sigma_\mathrm{prior}$")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.3, axis="y")

    ax3.bar(xs, abs_bias, color="C2", edgecolor="k", lw=0.4)
    ax3.axhline(1.0, color="k", ls=":", lw=0.8)
    ax3.set_ylabel(r"$\langle|\mu-\theta_\mathrm{t}|/\sigma\rangle$")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(PARAM_NAMES, rotation=90, fontsize=7)
    ax3.set_xlabel("parameter index")
    ax3.grid(alpha=0.3, axis="y")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")

    # text summary alongside the PDF
    txt = args.out.with_suffix(".txt")
    with open(txt, "w") as fh:
        fh.write(f"# Validation E — coverage at nominal {level:.4f}\n")
        fh.write(f"# n_sims={n_sims}  n_realizations={n_real}\n")
        fh.write(f"# overall mean coverage = {coverage.mean():.3f}\n")
        n_inf = int((constraint > 0.1).sum())
        fh.write(f"# {n_inf}/{p} params have constraint > 0.1 (data-informed)\n")
        bad = np.where(
            (constraint > 0.1) & (np.abs(coverage - level) > 2 * cov_err)
        )[0]
        fh.write(f"# {len(bad)} informed params more than 2σ from nominal\n")
        for j in range(p):
            tag = ""
            if constraint[j] <= 0.1:
                tag = "  prior-dominated"
            elif j in bad:
                tag = "  *"
            fh.write(f"p{j:02d}  cov={coverage[j]:.3f} ± {cov_err[j]:.3f}  "
                     f"constraint={constraint[j]:.2f}  |bias|/σ={abs_bias[j]:.2f}{tag}\n")
    print(f"[save] {txt}")


if __name__ == "__main__":
    main()
