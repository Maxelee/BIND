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
    sbc_ranks = z["sbc_ranks"] if "sbc_ranks" in z.files else None
    sbc_ks_p = z["sbc_ks_p"] if "sbc_ks_p" in z.files else None

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

    # ── SBC rank histogram (separate fig): the money plot for calibration ──
    # For a calibrated posterior the ranks Φ(θ_true) are Uniform(0,1).  Only the
    # data-informed params carry information; prior-dominated params pile at 0.5.
    if sbc_ranks is not None:
        informed = constraint > 0.1
        figs, (axa, axb) = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, mask, ttl in (
            (axa, informed, "data-informed params (constraint > 0.1)"),
            (axb, ~informed, "prior-dominated params"),
        ):
            r = sbc_ranks[:, mask].ravel()
            r = r[np.isfinite(r)]
            if r.size:
                ax.hist(r, bins=10, range=(0, 1), color="C0",
                        edgecolor="k", lw=0.4, density=True)
            ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (calibrated)")
            ax.set_xlim(0, 1)
            ax.set_xlabel(r"SBC rank  $\Phi((\theta_\mathrm{true}-\mu)/\sigma)$")
            ax.set_ylabel("density")
            ax.set_title(ttl, fontsize=9)
            ax.legend(fontsize=8, frameon=False)
        n_inf = int(informed.sum())
        figs.suptitle(f"Validation E — SBC rank uniformity ({n_inf} informed params)")
        figs.tight_layout()
        sbc_out = args.out.with_name(args.out.stem + "_sbc" + args.out.suffix)
        figs.savefig(sbc_out)
        plt.close(figs)
        print(f"[save] {sbc_out}")

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
            ks = ""
            if sbc_ks_p is not None and np.isfinite(sbc_ks_p[j]):
                ks = f"  SBC-KS_p={sbc_ks_p[j]:.2f}"
            fh.write(f"p{j:02d}  cov={coverage[j]:.3f} ± {cov_err[j]:.3f}  "
                     f"constraint={constraint[j]:.2f}  |bias|/σ={abs_bias[j]:.2f}{ks}{tag}\n")
    print(f"[save] {txt}")


if __name__ == "__main__":
    main()
