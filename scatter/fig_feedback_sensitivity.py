"""scatter/fig_feedback_sensitivity.py

Which of the 30 astro parameters drive the feedback <-> (f_gas, M_star) relations?

Main point: the gas fraction and stellar mass of a group-scale halo respond to
baryonic feedback, and the f_gas-M_star anticorrelation is *generated* by a shared
set of feedback knobs that push gas and stars in opposite directions. We quantify
this with a GLOBAL sensitivity analysis over all 30 astrophysical parameters.

Data (already generated, NO new sampling): the joint Sobol design from
scatter/scatter_decomposition.py, stored as per-halo chunks:
    outputs/scatter_diagnostics/chunks_joint_cv/joint_part_*.npz
each with cube (128 design pts, 40 halos, 12 noise draws, 16 obs), shared design
`sub` (128, 30) in [0,1], and `scan_idx` (the 30 astro params; cosmology fixed).

Method: all halos see the SAME 128-point design, so averaging the observable over
halos (within a mass bin) and noise draws isolates the parameter response,
Y[design] (128,). We summarise sensitivity two ways:
  - SRC: standardized regression coefficients (signed, linear). Valid because the
    linear model explains R^2 ~ 0.9 of the population-median variance.
  - S1: first-order Sobol index (variance-based) via the binning estimator, as a
    model-free cross-check.
Bootstrap over the 128 design points gives confidence intervals.

Usage:  python scatter/fig_feedback_sensitivity.py
Output: paper_figures/fig_feedback_sensitivity.{pdf,png} + ..._data.npz
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHUNK_GLOB = str(ROOT / "outputs/scatter_diagnostics/chunks_joint_cv/joint_part_*.npz")
JOINT_NPZ = ROOT / "outputs/scatter_diagnostics/scatter_decomposition_joint_cv.npz"
OUT_DIR = ROOT / "paper_figures"

MASS_BIN = (13.0, 13.5)        # group scale where feedback dominates
N_BOOT = 2000
RNG = np.random.default_rng(0)

# coarse category for colouring (by SB35 ParamName semantics)
SN_PARAMS = {"A_SN1", "A_SN2", "WindSpecMom", "WindFreeTravelDens", "MinWindVel",
             "WindEnergyReduction", "WindEnergyReductionZ", "WindEnergyReductionExp",
             "WindDumpFac", "ThermalWind"}
AGN_PARAMS = {"A_AGN1", "A_AGN2", "BHAccretion", "BHEddington", "BHFeedback",
              "BHRadEff", "QuasarThreshold", "QuasarThreshPow", "SeedBHMass"}


def reconstruct_cube():
    """Return Y (128, N_h, 16) [median over K], masses (N_h,), sub (128, 30), names(30)."""
    fs = sorted(glob.glob(CHUNK_GLOB))
    if not fs:
        raise FileNotFoundError(f"no joint chunks at {CHUNK_GLOB}")
    Ys, Ms, sub = [], [], None
    for f in fs:
        d = np.load(f)
        Ys.append(np.nanmedian(d["cube"], axis=2))     # (128, n_chunk, 16)
        Ms.append(d["masses"])
        if sub is None:
            sub = d["sub"]
    Y = np.concatenate(Ys, axis=1)
    masses = np.concatenate(Ms)
    names = list(np.load(JOINT_NPZ)["param_names"])
    return Y, masses, sub, names


def population_targets(Y, masses, mass_bin):
    """Per-design population-median observables in a mass bin.

    Returns dict of (128,) arrays for f_gas, log10 M_star, f_b.
    """
    logm = np.log10(masses)
    sel = (logm >= mass_bin[0]) & (logm < mass_bin[1])
    Mdm, Mg, Ms = Y[:, sel, 0], Y[:, sel, 1], Y[:, sel, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        fgas = Mg / (Mdm + Mg + Ms)
        lMs = np.log10(Ms)
    return {
        "f_gas": np.nanmedian(fgas, axis=1),
        "log M_star": np.nanmedian(lMs, axis=1),
        "f_b": np.nanmedian(Y[:, sel, 3], axis=1),
    }, int(sel.sum())


def src(X, y):
    """Standardized regression coefficients and R^2."""
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    A = np.c_[np.ones(len(ys)), Xs]
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    yhat = A @ beta
    R2 = 1.0 - np.sum((ys - yhat) ** 2) / np.sum(ys ** 2)
    return beta[1:], R2


def src_bootstrap(X, y, n_boot=N_BOOT):
    """SRC with bootstrap over design points -> (beta, R2, beta_lo, beta_hi)."""
    beta, R2 = src(X, y)
    n = len(y)
    boots = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots[b], _ = src(X[idx], y[idx])
    lo, hi = np.percentile(boots, [16, 84], axis=0)
    return beta, R2, lo, hi


def sobol_first_order(X01, y, n_bins=8):
    """First-order Sobol S_i = Var(E[y|x_i]) / Var(y) via binning."""
    n, d = X01.shape
    vy = np.var(y)
    S = np.zeros(d)
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(d):
        b = np.clip(np.digitize(X01[:, i], edges[1:-1]), 0, n_bins - 1)
        cond = np.array([y[b == k].mean() if (b == k).any() else y.mean()
                         for k in range(n_bins)])
        w = np.array([(b == k).sum() for k in range(n_bins)]) / n
        S[i] = max(np.sum(w * (cond - y.mean()) ** 2) / vy, 0.0)
    return S


def main():
    Y, masses, sub, names = reconstruct_cube()
    targets, n_halo = population_targets(Y, masses, MASS_BIN)
    print(f"joint cube {Y.shape}; {n_halo} halos in 10^{MASS_BIN[0]}-10^{MASS_BIN[1]}")

    res = {}
    for key, y in targets.items():
        beta, R2, lo, hi = src_bootstrap(sub, y)
        S1 = sobol_first_order(sub, y)
        res[key] = dict(beta=beta, R2=R2, lo=lo, hi=hi, S1=S1)
        print(f"  {key:12s} linear R^2={R2:.2f}")

    bf, bm = res["f_gas"]["beta"], res["log M_star"]["beta"]
    # rank params by combined importance on the two relations
    importance = np.maximum(np.abs(bf), np.abs(bm))
    order = np.argsort(importance)[::-1]
    top = order[:12][::-1]                     # ascending for horizontal bars

    cat_color = {p: ("#c0392b" if p in SN_PARAMS else
                     "#2c7fb8" if p in AGN_PARAMS else "0.5") for p in names}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.2))

    # ---- Panel A: grouped horizontal bars, f_gas vs M_star SRC ---------------
    yy = np.arange(len(top))
    h = 0.38
    axA.barh(yy + h / 2, bf[top], height=h, color="#1b9e77",
             xerr=[bf[top] - res["f_gas"]["lo"][top], res["f_gas"]["hi"][top] - bf[top]],
             error_kw=dict(lw=0.8, ecolor="0.3"), label=r"$f_{\rm gas}$")
    axA.barh(yy - h / 2, bm[top], height=h, color="#d95f02",
             xerr=[bm[top] - res["log M_star"]["lo"][top], res["log M_star"]["hi"][top] - bm[top]],
             error_kw=dict(lw=0.8, ecolor="0.3"), label=r"$\log M_\star$")
    axA.axvline(0, color="k", lw=0.8)
    axA.set_yticks(yy)
    axA.set_yticklabels([names[i] for i in top], fontsize=9)
    for tick, i in zip(axA.get_yticklabels(), top):
        tick.set_color(cat_color[names[i]])
    axA.set_xlabel("standardized sensitivity (SRC)")
    axA.set_title(f"(A)  Sensitivity of group-scale $f_{{\\rm gas}}$ and $M_\\star$\n"
                  f"to the 30 astro params  ($R^2$={res['f_gas']['R2']:.2f}, "
                  f"{res['log M_star']['R2']:.2f})")
    axA.legend(loc="lower right", fontsize=9)
    axA.grid(axis="x", alpha=0.25)

    # ---- Panel B: (SRC_fgas, SRC_Mstar) plane -> shared-driver structure -----
    rho = np.corrcoef(bf, bm)[0, 1]
    for i, p in enumerate(names):
        axB.scatter(bf[i], bm[i], s=70, color=cat_color[p],
                    edgecolor="k", lw=0.5, zorder=3)
        if importance[i] > 0.18:
            axB.annotate(p, (bf[i], bm[i]), textcoords="offset points",
                         xytext=(5, 4), fontsize=8)
    lim = 1.05 * max(np.abs(bf).max(), np.abs(bm).max())
    axB.plot([-lim, lim], [lim, -lim], ls="--", color="0.6", lw=1,
             label="perfect anti-alignment")
    axB.axhline(0, color="k", lw=0.6)
    axB.axvline(0, color="k", lw=0.6)
    axB.set_xlim(-lim, lim)
    axB.set_ylim(-lim, lim)
    axB.set_xlabel(r"SRC on $f_{\rm gas}$")
    axB.set_ylabel(r"SRC on $\log M_\star$")
    axB.set_title(f"(B)  Shared feedback drives both, oppositely\n"
                  rf"(param-wise corr $\rho$={rho:+.2f})")
    handles = [Patch(color="#c0392b", label="SN winds"),
               Patch(color="#2c7fb8", label="AGN / BH"),
               Patch(color="0.5", label="other")]
    axB.legend(handles=handles + [plt.Line2D([], [], ls="--", color="0.6",
               label="anti-alignment")], fontsize=8, loc="upper right")
    axB.grid(alpha=0.25)

    fig.suptitle("Feedback parameters jointly set group-scale gas fraction and stellar mass",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_feedback_sensitivity.pdf"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"[saved] {pdf}  (+ .png)")

    # ---- table -----
    print(f"\n=== Top drivers (group scale), SRC [16-84%], first-order Sobol ===")
    print(f"{'param':22s} {'SRC f_gas':>16s} {'SRC logM*':>16s} {'S1 f_gas':>9s} {'S1 M*':>7s}")
    for i in order[:12]:
        print(f"{names[i]:22s} "
              f"{bf[i]:+6.3f}[{res['f_gas']['lo'][i]:+.2f},{res['f_gas']['hi'][i]:+.2f}] "
              f"{bm[i]:+6.3f}[{res['log M_star']['lo'][i]:+.2f},{res['log M_star']['hi'][i]:+.2f}] "
              f"{res['f_gas']['S1'][i]:9.3f} {res['log M_star']['S1'][i]:7.3f}")
    print(f"\nparam-wise corr(SRC_fgas, SRC_Mstar) = {rho:+.2f}  "
          f"(negative => shared knobs act oppositely on gas vs stars)")

    np.savez_compressed(
        OUT_DIR / "fig_feedback_sensitivity_data.npz",
        param_names=np.array(names), mass_bin=np.array(MASS_BIN),
        src_fgas=bf, src_logMstar=bm, src_fb=res["f_b"]["beta"],
        src_fgas_lohi=np.c_[res["f_gas"]["lo"], res["f_gas"]["hi"]],
        src_logMstar_lohi=np.c_[res["log M_star"]["lo"], res["log M_star"]["hi"]],
        s1_fgas=res["f_gas"]["S1"], s1_logMstar=res["log M_star"]["S1"],
        R2_fgas=res["f_gas"]["R2"], R2_logMstar=res["log M_star"]["R2"],
        corr_src=rho,
    )
    print(f"[saved] {OUT_DIR / 'fig_feedback_sensitivity_data.npz'}")


if __name__ == "__main__":
    main()
