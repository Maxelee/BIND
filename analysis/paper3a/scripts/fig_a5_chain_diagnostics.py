"""WP-A5 FINAL-chain diagnostics: trace, R-hat, corner, posterior-predictive.

`run_a5_chains_final.py --assemble` writes `a5_final_summary.json` but never
actually rendered the figure its own docstring promises
(`figures/a5_posterior_final.png`) -- the only posterior figure that existed
on disk (`a5_posterior_fgas.png`) is stale, from the pre-convergence
EXPLORATORY fit (`a5_fit_summary.json`, MAP chi2=18.9, different corner).
This script builds the actual convergence-diagnostic + posterior figures for
the FINAL chains (4 seeds x 128 walkers x 30k steps, R-hat <= 1.0006) so Max
can review before ruling on decisions 1/3.

Run:  python analysis/paper3a/scripts/fig_a5_chain_diagnostics.py
Out:  wp5_chains/figures/a5_final_trace.png
      wp5_chains/figures/a5_final_rhat.png
      wp5_chains/figures/a5_final_corner.png
      wp5_chains/figures/a5_final_posterior_predictive.png
      wp5_chains/a5_final_ess.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.style import COL, apply, band, condition_tag, data_points  # noqa: E402

OUT = CHAINS / "figures"
N_SEEDS = 4
SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]   # role-neutral, per-seed only


def load_chains() -> list[np.ndarray]:
    return [np.load(CHAINS / f"a5_final_seed{k}.npz")["chain"].astype(float)
            for k in range(N_SEEDS)]


def cross_chain_rhat(chains: list[np.ndarray]) -> np.ndarray:
    """R-hat treating each seed's flattened samples as one chain (matches
    `run_a5_chains_final.cross_chain_rhat`, reimplemented to avoid importing
    a `__main__`-guarded script as a module)."""
    x = np.stack([c.reshape(-1, c.shape[-1]) for c in chains])
    n = x.shape[1]
    m = x.mean(axis=1)
    w = x.var(axis=1, ddof=1).mean(axis=0)
    b = n * m.var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def integrated_autocorr_time(chains: list[np.ndarray]) -> np.ndarray:
    """Per-parameter integrated autocorrelation time, averaged over seeds
    (emcee's FFT estimator, one per seed's (steps, walkers, dim) array)."""
    import emcee.autocorr as ac

    taus = []
    for c in chains:
        try:
            taus.append(ac.integrated_time(c, tol=20, quiet=True))
        except Exception:
            taus.append(ac.integrated_time(c, tol=0, quiet=True))
    return np.mean(taus, axis=0)


def main() -> None:
    apply()
    OUT.mkdir(parents=True, exist_ok=True)
    chains = load_chains()                                  # 4 x (S, W, 30)
    n_steps, n_walkers, ndim = chains[0].shape
    rhat = cross_chain_rhat(chains)
    tau = integrated_autocorr_time(chains)
    ess_per_seed = (n_steps * n_walkers) / (2.0 * tau + 1.0)
    ess_total = N_SEEDS * ess_per_seed

    top = np.argsort(-rhat)[:8]
    print("Highest R-hat dims:", [(pm.ASTRO_NAMES[i], round(float(rhat[i]), 5)) for i in top])

    blk = FgasBlock()
    flat = np.concatenate([c.reshape(-1, ndim) for c in chains])
    logp_all = np.concatenate([np.load(CHAINS / f"a5_final_seed{k}.npz")["logp"].astype(float).ravel()
                                for k in range(N_SEEDS)])
    imap = int(np.argmax(logp_all))
    u_map = flat[imap]

    # subsample for anything that needs a FgasBlock.predict() pass (cheap
    # but no need to run it on all 750k samples)
    sub = flat[:: max(1, len(flat) // 40_000)]
    f_group = blk.predict(sub)[:, 0]

    # ---------------------------------------------------------- 1. trace ---
    key_params = ["QuasarThresholdPower", "RadioFeedbackReiorientationFactor",
                  "IMFslope", "WindEnergyIn1e51erg"]
    fig, axes = plt.subplots(len(key_params) + 1, 1, figsize=(7.0, 2.0 * (len(key_params) + 1)),
                             sharex=True)
    step = np.arange(n_steps)
    for ax, name in zip(axes[:-1], key_params):
        i = pm.ASTRO_NAMES.index(name)
        for k, c in enumerate(chains):
            ax.plot(step, c[:, :, i].mean(axis=1), color=SEED_COLORS[k], lw=0.7,
                    alpha=0.85, label=f"seed {k}" if name == key_params[0] else None)
        ax.set_ylabel(name, fontsize=7)
        ax.set_ylim(0, 1)
    for k, c in enumerate(chains):
        axes[-1].plot(step, c[:, :, pm.ASTRO_NAMES.index("QuasarThresholdPower")],
                      color=SEED_COLORS[k], lw=0.15, alpha=0.25)
    axes[-1].set_ylabel("QuasarThresholdPower\n(all 128 walkers)", fontsize=7)
    axes[-1].set_xlabel("thinned step (x15 -> true step)")
    axes[0].legend(loc="upper right", fontsize=6, ncol=4)
    fig.suptitle("WP-A5 FINAL chains: per-seed walker-mean traces (top) + walker spaghetti (bottom)",
                fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a5_final_trace.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------- 2. rhat ---
    order = np.argsort(rhat)
    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    ax.barh(np.arange(ndim), rhat[order], color=COL["model"])
    ax.set_yticks(np.arange(ndim))
    ax.set_yticklabels([pm.ASTRO_NAMES[i] for i in order], fontsize=6)
    ax.axvline(1.0, color="k", lw=0.6)
    ax.axvline(1.01, color=COL["data"], lw=0.8, ls="--", label="R-hat = 1.01")
    ax.set_xlabel(r"cross-chain $\hat R$ (4 seeds)")
    ax.legend(fontsize=7)
    ax.set_title(f"WP-A5 FINAL chains: max R-hat = {rhat.max():.4f} (all 30 dims)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a5_final_rhat.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------- 3. corner ---
    corner_params = ["QuasarThresholdPower", "RadioFeedbackReiorientationFactor",
                     "IMFslope", "VariableWindVelFactor", "WindEnergyIn1e51erg"]
    idx = [pm.ASTRO_NAMES.index(n) for n in corner_params]
    d = len(idx)
    thin_sub = flat[:: max(1, len(flat) // 20_000)][:, idx]
    u_map_sub = u_map[idx]
    fig, axes = plt.subplots(d, d, figsize=(1.55 * d, 1.55 * d))
    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:
                ax.hist(thin_sub[:, r], bins=60, range=(0, 1), color=COL["model"], alpha=0.75)
                ax.axvline(u_map_sub[r], color=COL["data"], lw=1.0)
                ax.set_yticks([])
            else:
                ax.hexbin(thin_sub[:, c], thin_sub[:, r], gridsize=40, extent=(0, 1, 0, 1),
                          cmap="Blues", mincnt=1)
                ax.scatter([u_map_sub[c]], [u_map_sub[r]], color=COL["data"], marker="*", s=40, zorder=5)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            if r == d - 1:
                ax.set_xlabel(corner_params[c], fontsize=6.5, rotation=30, ha="right")
            else:
                ax.set_xticklabels([])
            if c == 0 and r != 0:
                ax.set_ylabel(corner_params[r], fontsize=6.5)
            else:
                ax.set_yticklabels([])
    fig.suptitle("WP-A5 FINAL posterior: unit-cube corner (MAP = red star), edge-piling visible", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a5_final_corner.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------- 4. posterior predictive ---
    pred_sub = blk.predict(sub)                              # (n, B)
    p16, p50, p84 = np.percentile(pred_sub, [16, 50, 84], axis=0)
    pred_map = blk.predict(u_map[None, :])[0]
    chi2_map = float(blk.chi2(u_map)[0]) if hasattr(blk, "chi2") else None
    x = blk.data.bin_centers
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    band(ax, x, p16, p84, color=COL["model"], alpha=0.30, label="A5 FINAL posterior 16-84%")
    ax.plot(x, p50, color=COL["model"], lw=1.2, label="posterior median")
    ax.plot(x, pred_map, color=COL["model"], lw=1.0, ls="--", label="MAP")
    data_points(ax, x, blk.data.values, yerr=blk.data.stat_err, label="eRASS1 medians")
    ax.set_xlabel(r"$\log_{10} M_{500}\ [M_\odot/h]$")
    ax.set_ylabel(r"$f_{\rm gas,sph}$")
    title = "WP-A5 FINAL posterior-predictive vs eRASS1"
    if chi2_map is not None:
        title += f" (MAP $\\chi^2$={chi2_map:.2f}/{len(x)})"
    condition_tag(ax, title)
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "a5_final_posterior_predictive.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------- ESS summary --
    ess_summary = {
        "n_steps_per_seed": int(n_steps), "n_walkers": int(n_walkers), "n_seeds": N_SEEDS,
        "n_samples_total": int(N_SEEDS * n_steps * n_walkers),
        "tau_by_param": {pm.ASTRO_NAMES[i]: float(tau[i]) for i in range(ndim)},
        "ess_total_by_param": {pm.ASTRO_NAMES[i]: float(ess_total[i]) for i in range(ndim)},
        "ess_total_min": float(ess_total.min()),
        "ess_total_median": float(np.median(ess_total)),
        "worst_param": pm.ASTRO_NAMES[int(np.argmin(ess_total))],
        "rhat_max": float(rhat.max()),
    }
    (CHAINS / "a5_final_ess.json").write_text(json.dumps(ess_summary, indent=2))
    print(json.dumps({k: v for k, v in ess_summary.items() if k not in
                      ("tau_by_param", "ess_total_by_param")}, indent=2))
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
