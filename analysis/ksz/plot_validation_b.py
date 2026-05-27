"""Plot validation B — mean per-halo τ(R/R200) profiles, BIND vs truth.

Aggregates per-halo annular profiles from analysis.ksz.validation_b into mean
(median, 16/84 band) curves per mass bin and writes one comparison PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mass_bins", nargs="+", type=float,
                   default=[1e13, 3e13, 1e14, 1e15])
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.input, allow_pickle=False)
    m = d["halo_mass_msun_h"]
    rc = d["r_centers_r200"]
    tau_b = d["bind_tau_profile"]
    tau_t = d["truth_tau_profile"]

    edges = np.asarray(args.mass_bins, dtype=np.float64)
    n_bins = len(edges) - 1
    bin_idx = np.digitize(m, edges) - 1

    fig, axes = plt.subplots(1, n_bins, figsize=(4.0 * n_bins, 4.0),
                             sharey=True)
    if n_bins == 1:
        axes = [axes]

    for b, ax in enumerate(axes):
        sel = bin_idx == b
        if not sel.any():
            ax.set_title(f"empty bin {b}")
            continue
        tb = tau_b[sel]
        tt = tau_t[sel]

        def _pcts(a):
            med = np.nanmedian(a, axis=0)
            lo = np.nanpercentile(a, 16, axis=0)
            hi = np.nanpercentile(a, 84, axis=0)
            return med, lo, hi

        bm, blo, bhi = _pcts(tb)
        tm, tlo, thi = _pcts(tt)

        ax.fill_between(rc, tlo, thi, alpha=0.25, color="C0", label="truth 16-84%")
        ax.plot(rc, tm, "C0-", lw=2, label="truth median")
        ax.fill_between(rc, blo, bhi, alpha=0.25, color="C3", label="BIND 16-84%")
        ax.plot(rc, bm, "C3--", lw=2, label="BIND median")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$R / R_{200c}$")
        if b == 0:
            ax.set_ylabel(r"$\tau(R)$")
        ax.set_title(
            f"$\\log M \\in [{np.log10(edges[b]):.2f},{np.log10(edges[b+1]):.2f}]$  "
            f"(n={int(sel.sum())})"
        )
        ax.grid(True, which="both", alpha=0.2)
        if b == 0:
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Validation B — annular τ profile match (BIND vs truth)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
