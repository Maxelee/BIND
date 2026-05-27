"""Plot validation F — coverage vs v_los systematic level."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.input, allow_pickle=True)
    sigmas = z["vlos_sigmas"]
    cov = z["coverage"]                  # (S, P)
    cov_err = z["coverage_err"]
    bias = z["abs_bias_in_sigma"]
    level = float(z["nominal_level"])
    n_sims = int(z["n_sims"])
    n_real = int(z["n_realizations"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: mean coverage across params vs σ_v, with shaded 16-84% across params
    mean_cov = cov.mean(axis=1)
    lo = np.percentile(cov, 16, axis=1)
    hi = np.percentile(cov, 84, axis=1)
    ax1.fill_between(sigmas, lo, hi, color="C0", alpha=0.25,
                     label="16–84% across params")
    ax1.plot(sigmas, mean_cov, "o-", color="C0", label="mean across params")
    ax1.axhline(level, color="k", ls="--", lw=1, label=f"nominal = {level:.3f}")
    ax1.set_xlabel(r"$\sigma_{v_\mathrm{los}}$ (fractional multiplicative)")
    ax1.set_ylabel("coverage of 1-σ CI")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title(f"Validation F — coverage vs v_los systematic\n"
                  f"({n_sims} sims × {n_real} reals)")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)

    # Right: heatmap of per-param coverage vs σ_v
    im = ax2.imshow(
        cov, aspect="auto", origin="lower",
        extent=[-0.5, cov.shape[1] - 0.5, sigmas.min() - 0.005, sigmas.max() + 0.005],
        cmap="RdBu", vmin=0, vmax=1,
    )
    ax2.set_yticks(sigmas)
    ax2.set_xlabel("parameter index")
    ax2.set_ylabel(r"$\sigma_{v_\mathrm{los}}$")
    ax2.set_title("per-parameter coverage")
    fig.colorbar(im, ax=ax2, label="coverage")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")

    txt = args.out.with_suffix(".txt")
    with open(txt, "w") as fh:
        fh.write(f"# Validation F — coverage vs v_los systematic  level={level:.4f}\n")
        fh.write(f"# n_sims={n_sims}  n_realizations={n_real}\n")
        fh.write("# sigma_v  mean_cov  16%  84%  mean_|bias|/sigma\n")
        for i, sv in enumerate(sigmas):
            fh.write(f"{sv:.3f}  {mean_cov[i]:.3f}  "
                     f"{lo[i]:.3f}  {hi[i]:.3f}  {bias[i].mean():.2f}\n")
    print(f"[save] {txt}")


if __name__ == "__main__":
    main()
