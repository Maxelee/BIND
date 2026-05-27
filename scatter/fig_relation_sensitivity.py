"""scatter/fig_relation_sensitivity.py  (Deliverable #2: sensitivity of the RELATION)

The f_gas-M_star relation is not a single curve: feedback can move halos ALONG it
(shift the mean f_gas and M_star) or RESHAPE it (change the intrinsic scatter at
fixed mass and the f_gas-M_star coupling). We ask which of the 30 astro params do
which.

For each of the 128 joint-design points we characterise the relation among
group-scale halos (10^13-10^13.5) after removing the halo-mass trend:
  normalization : median f_gas, median log M_star      (move ALONG)
  scatter       : sigma(f_gas|M), sigma(log M_star|M)  (RESHAPE)
  coupling      : corr of the mass-residuals           (RESHAPE / re-tilt)
then take the standardized sensitivity (SRC) of each descriptor to every param
(robustness validated in scatter/sensitivity.py: linear ~ GP > GBM in CV R^2).

Output: paper_figures/fig_relation_sensitivity.{pdf,png} + ..._data.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scatter.sensitivity import (                       # noqa: E402
    load_joint_cube, relation_descriptors, src_bootstrap, fgas_cube, I_MSTAR,
)

OUT_DIR = ROOT / "paper_figures"
MASS_BIN = (13.0, 13.5)
SN_PARAMS = {"A_SN1", "A_SN2", "WindSpecMom", "WindFreeTravelDens", "MinWindVel",
             "WindEnergyReduction", "WindEnergyReductionZ", "WindEnergyReductionExp",
             "WindDumpFac", "ThermalWind"}
AGN_PARAMS = {"A_AGN1", "A_AGN2", "BHAccretion", "BHEddington", "BHFeedback",
              "BHRadEff", "QuasarThreshold", "QuasarThreshPow", "SeedBHMass"}


def _residual_clouds(Y, masses, mass_bin, design_mask):
    """Pool mass-detrended (f_gas, logM*) residuals over the selected designs."""
    logm = np.log10(masses)
    sel = (logm >= mass_bin[0]) & (logm < mass_bin[1])
    xd = logm[sel] - logm[sel].mean()
    fg = fgas_cube(Y)[:, sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        lMs = np.log10(Y[:, sel, I_MSTAR])
    rg_all, rm_all = [], []
    for d in np.where(design_mask)[0]:
        m = np.isfinite(fg[d]) & np.isfinite(lMs[d])
        if m.sum() < 10:
            continue
        bg = np.polyfit(xd[m], fg[d][m], 1)
        bm = np.polyfit(xd[m], lMs[d][m], 1)
        rg_all.append(fg[d][m] - np.polyval(bg, xd[m]))
        rm_all.append(lMs[d][m] - np.polyval(bm, xd[m]))
    return np.concatenate(rg_all), np.concatenate(rm_all)


def main():
    Y, masses, sub, names = load_joint_cube()
    desc, n = relation_descriptors(Y, masses, MASS_BIN)
    print(f"group bin N={n}; median coupling r={np.nanmedian(desc['coupling']):.3f}")

    descriptors = ["med_fgas", "med_logMstar", "scat_fgas", "scat_logMstar", "coupling"]
    beta = {}
    for k in descriptors:
        b, R2, lo, hi = src_bootstrap(sub, desc[k])
        beta[k] = dict(b=b, R2=R2, lo=lo, hi=hi)

    # role axes per param
    norm_sens = np.sqrt(beta["med_fgas"]["b"] ** 2 + beta["med_logMstar"]["b"] ** 2)
    reshape_sens = np.sqrt(beta["scat_fgas"]["b"] ** 2 + beta["scat_logMstar"]["b"] ** 2
                           + beta["coupling"]["b"] ** 2)

    cat_color = {p: ("#c0392b" if p in SN_PARAMS else
                     "#2c7fb8" if p in AGN_PARAMS else "0.5") for p in names}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.2))

    # ---- Panel A: role plot (move ALONG vs RESHAPE) --------------------------
    lim = 1.05 * max(norm_sens.max(), reshape_sens.max())
    axA.plot([0, lim], [0, lim], ls="--", color="0.7", lw=1)
    for i, p in enumerate(names):
        axA.scatter(norm_sens[i], reshape_sens[i], s=80, color=cat_color[p],
                    edgecolor="k", lw=0.5, zorder=3)
        if max(norm_sens[i], reshape_sens[i]) > 0.30:
            axA.annotate(p, (norm_sens[i], reshape_sens[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8)
    axA.set_xlim(0, lim); axA.set_ylim(0, lim)
    axA.set_xlabel(r"normalization sensitivity  $\sqrt{{\rm SRC}^2_{\bar f_{\rm gas}}+{\rm SRC}^2_{\bar M_\star}}$"
                   "\n(move ALONG the relation)")
    axA.set_ylabel("reshape sensitivity  "
                   r"$\sqrt{\sum {\rm SRC}^2_{\sigma,\,{\rm coupling}}}$"
                   "\n(change scatter / coupling)")
    axA.set_title("(A)  Does a parameter move halos along the relation, or reshape it?")
    axA.text(0.97, 0.05, "below line: mover\nabove line: reshaper",
             transform=axA.transAxes, ha="right", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    axA.legend(handles=[Patch(color="#c0392b", label="SN winds"),
                        Patch(color="#2c7fb8", label="AGN / BH"),
                        Patch(color="0.5", label="other")],
               fontsize=8, loc="upper left")
    axA.grid(alpha=0.25)

    # ---- Panel B: the dominant reshaper re-tilts the relation -----------------
    reshaper = names[int(np.argmax(reshape_sens))]
    ip = names.index(reshaper)
    q = sub[:, ip]
    lo_mask = q <= np.quantile(q, 0.25)
    hi_mask = q >= np.quantile(q, 0.75)
    rg_lo, rm_lo = _residual_clouds(Y, masses, MASS_BIN, lo_mask)
    rg_hi, rm_hi = _residual_clouds(Y, masses, MASS_BIN, hi_mask)
    r_lo = np.corrcoef(rg_lo, rm_lo)[0, 1]
    r_hi = np.corrcoef(rg_hi, rm_hi)[0, 1]

    for rg, rm, col, lbl, r in [
            (rg_lo, rm_lo, "#4575b4", f"low {reshaper}", r_lo),
            (rg_hi, rm_hi, "#d73027", f"high {reshaper}", r_hi)]:
        axB.scatter(rm, rg, s=4, color=col, alpha=0.10, zorder=2)
        # robust trend line
        b = np.polyfit(rm, rg, 1)
        xs = np.linspace(np.percentile(rm, 1), np.percentile(rm, 99), 50)
        axB.plot(xs, np.polyval(b, xs), color=col, lw=2.5, zorder=4,
                 label=f"{lbl}:  r={r:+.2f}")
    axB.axhline(0, color="k", lw=0.6); axB.axvline(0, color="k", lw=0.6)
    axB.set_xlabel(r"$\Delta \log M_\star$ at fixed $M_{\rm halo}$ (residual)")
    axB.set_ylabel(r"$\Delta f_{\rm gas}$ at fixed $M_{\rm halo}$ (residual)")
    axB.set_xlim(np.percentile(np.r_[rm_lo, rm_hi], 1), np.percentile(np.r_[rm_lo, rm_hi], 99))
    axB.set_ylim(np.percentile(np.r_[rg_lo, rg_hi], 1), np.percentile(np.r_[rg_lo, rg_hi], 99))
    axB.set_title(f"(B)  The top reshaper ({reshaper}) re-tilts the\n"
                  r"intrinsic $f_{\rm gas}$-$M_\star$ relation at fixed halo mass")
    axB.legend(fontsize=9, loc="upper right")
    axB.grid(alpha=0.25)

    fig.suptitle("Which feedback parameters set vs. reshape the "
                 r"$f_{\rm gas}$-$M_\star$ relation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_relation_sensitivity.pdf"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"[saved] {pdf} (+ .png)")

    # ---- table ----
    print(f"\n=== SRC of relation descriptors (group scale), * = 16-84% excludes 0 ===")
    hdr = "param".ljust(22) + "".join(f"{k:>15s}" for k in descriptors)
    print(hdr)
    order = np.argsort(norm_sens + reshape_sens)[::-1]
    for i in order[:12]:
        row = names[i].ljust(22)
        for k in descriptors:
            b = beta[k]["b"][i]
            sig = "*" if (beta[k]["lo"][i] > 0 or beta[k]["hi"][i] < 0) else " "
            row += f"{b:+8.2f}{sig}      "
        print(row)
    print(f"\ntop reshaper = {reshaper}: coupling r {r_lo:+.2f} (low) -> {r_hi:+.2f} (high)")

    np.savez_compressed(
        OUT_DIR / "fig_relation_sensitivity_data.npz",
        param_names=np.array(names), mass_bin=np.array(MASS_BIN),
        **{f"src_{k}": beta[k]["b"] for k in descriptors},
        **{f"src_{k}_lohi": np.c_[beta[k]["lo"], beta[k]["hi"]] for k in descriptors},
        norm_sens=norm_sens, reshape_sens=reshape_sens,
        top_reshaper=reshaper, coupling_lo=r_lo, coupling_hi=r_hi,
    )
    print(f"[saved] {OUT_DIR / 'fig_relation_sensitivity_data.npz'}")


if __name__ == "__main__":
    main()
