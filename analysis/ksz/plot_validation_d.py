"""Plot validation D — stacked τ(M) in ACT-DR6-like apertures, BIND vs truth."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.input, allow_pickle=True)
    centers = d["mass_centers"]
    edges = d["mass_edges"]
    n = d["n_per_bin"]
    bm = d["bind_tau_mean"]; be = d["bind_tau_sem"]
    tm = d["truth_tau_mean"]; te = d["truth_tau_sem"]
    bmed = d["bind_tau_median"]; tmed = d["truth_tau_median"]
    meta = ast.literal_eval(str(d["meta"]))

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: stacked τ(M)
    ax.errorbar(centers, tm, yerr=te, fmt="o-", color="C0", label="truth (mean)",
                lw=2, ms=6)
    ax.plot(centers, tmed, "C0:", alpha=0.5, label="truth (median)")
    ax.errorbar(centers, bm, yerr=be, fmt="s--", color="C3", label="BIND (mean)",
                lw=2, ms=6)
    ax.plot(centers, bmed, "C3:", alpha=0.5, label="BIND (median)")

    ax.set_xscale("log")
    use_log_y = bool(np.all(np.isfinite(np.concatenate([tm, bm])) & (np.concatenate([tm, bm]) > 0)))
    if use_log_y:
        ax.set_yscale("log")
    ax.set_ylabel(r"$\tau_{\rm stack}$  (" + str(meta.get("aperture")) + r" aperture, $R_{\rm ap}="
                  + f"{meta.get('r_ap_mpc_h')}" + r"\,$Mpc$/h$)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_title(f"Validation D — stacked τ(M)  ({meta.get('model')})")

    # Bottom: fractional residual (BIND − truth)/truth
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = (bm - tm) / tm
        resid_err = be / np.abs(tm)
    ax_r.errorbar(centers, resid, yerr=resid_err, fmt="o-", color="C3")
    ax_r.axhline(0, color="k", lw=0.5)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"halo mass $M\ [M_\odot/h]$")
    ax_r.set_ylabel(r"$(\tau_{\rm BIND} - \tau_{\rm truth}) / \tau_{\rm truth}$")
    ax_r.grid(True, which="both", alpha=0.2)

    # Bin-count annotations
    finite = np.isfinite(tm)
    ymin = np.nanmin(tm[finite]) if finite.any() else 0
    for x, nb in zip(centers, n):
        ax.text(x, ymin if use_log_y else 0,
                f"n={int(nb)}", ha="center", va="bottom", fontsize=7, alpha=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[save] {args.out}")

    # Text summary
    lines = ["# Validation D — stacked τ(M) (BIND vs truth)",
             f"# aperture={meta.get('aperture')}  R_ap={meta.get('r_ap_mpc_h')} Mpc/h"]
    for i, c in enumerate(centers):
        if n[i] == 0:
            continue
        lines.append(
            f"logM~{np.log10(c):.2f}  n={int(n[i])}  truth={tm[i]:.3e}±{te[i]:.1e}  "
            f"BIND={bm[i]:.3e}±{be[i]:.1e}  frac_resid={resid[i]:+.3f}"
        )
    txt = "\n".join(lines)
    print(txt)
    args.out.with_suffix(".txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
