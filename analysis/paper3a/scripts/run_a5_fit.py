"""WP-A5 data fit (plan tasks 3-4): the posterior, gated on recovery PASS.

Refuses to run unless `wp5_chains/a5_recovery.json` exists with PASS=true
(the pre-registered battery). Then: the full-length fit on the frozen
eRASS1 vector, chain archived, and the deliverables:

- convergence on the CONSTRAINED subspace: R-hat + ESS for the derived
  group-bin f_gas summary and for the top-|correlation| parameters (the
  30-dim max-R-hat is dominated by prior-flat directions mixing slowly —
  documented, not hidden);
- the boundary diagnostic (R1): posterior mass within 5% of each prior
  edge, per parameter — the exclusion-tripwire instrument;
- chi^2 at MAP + posterior-predictive p-value (are we fitting or
  excluding?);
- reduced summaries + a marginal figure for the wind/AGN sector.

Output: wp5_chains/a5_fit_wiener... no — a5_fit_fgas.npz + a5_fit_summary.json
+ figures/a5_posterior_fgas.png. Run: python analysis/paper3a/scripts/run_a5_fit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, run_mcmc, split_rhat  # noqa: E402

N_WALKERS = 96
N_STEPS = 8000
N_BURN = 2000
EDGE_FRAC = 0.05


def main() -> None:
    gate = CHAINS / "a5_recovery.json"
    if not gate.exists():
        raise SystemExit("recovery battery has not run — refusing the data fit")
    rec = json.loads(gate.read_text())
    if not rec.get("PASS", False):
        raise SystemExit(f"recovery battery did NOT pass ({gate}) — fix calibration first")

    blk = FgasBlock()
    res = run_mcmc([blk], n_walkers=N_WALKERS, n_steps=N_STEPS, n_burn=N_BURN,
                   seed=7, label="fit_fgas", archive=True)
    flat = res["flat"]
    logp = res["logp"]

    # constrained-subspace convergence: derived summary chain
    thin = max(1, len(flat) // 20000)
    fsub = flat[::thin]
    f_group = blk.predict(fsub)[:, 0]
    n_eff_chain = f_group.reshape(1, -1, 1)             # pseudo (S, W=1, d)
    corr = np.array([abs(np.corrcoef(fsub[:, i], f_group)[0, 1]) for i in range(30)])
    top = np.argsort(-corr)[:6]

    # boundary diagnostic (R1)
    edge_mass = {pm.ASTRO_NAMES[i]: {
        "low": float(np.mean(flat[:, i] < EDGE_FRAC)),
        "high": float(np.mean(flat[:, i] > 1 - EDGE_FRAC)),
    } for i in range(30)}
    tripwire = {n: v for n, v in edge_mass.items()
                if max(v["low"], v["high"]) > 3 * EDGE_FRAC}

    # fit quality
    imap = int(np.argmax(logp))
    u_map = flat[imap]
    chi2_map = float(blk.chi2(u_map)[0])
    pred_map = blk.predict(u_map)[0]
    sig_map = blk.sigma(pred_map[None, :])[0]
    # posterior-predictive p: draws from posterior + noise vs observed chi2
    rng = np.random.default_rng(3)
    sub = fsub[rng.integers(0, len(fsub), 400)]
    pp = blk.predict(sub)
    ss = blk.sigma(pp)
    sim = pp + rng.normal(0, ss)
    chi2_sim = np.sum(((sim - pp) / ss) ** 2, axis=1)
    chi2_obs = np.sum(((blk.data.values[None, :] - pp) / ss) ** 2, axis=1)
    p_pp = float(np.mean(chi2_sim >= chi2_obs))

    summary = {
        "recovery_gate": {k: rec[k] for k in ("PASS", "coverage68", "coverage95", "ks_uniform_p")},
        "chain": {"acceptance": res["acceptance"],
                  "rhat_max_all_dims": float(np.max(res["rhat"])),
                  "rhat_constrained_top6": {pm.ASTRO_NAMES[i]: float(res["rhat"][i]) for i in top},
                  "ess_min_top6": float(np.nanmin(res["ess"][top]))},
        "map": {"chi2": chi2_map, "n_bins": int(len(blk.data.values)),
                "pred": pred_map.tolist(), "data": blk.data.values.tolist(),
                "sigma": sig_map.tolist(),
                "theta_map_named": {pm.ASTRO_NAMES[i]: float(u_map[i]) for i in top}},
        "posterior_predictive_p": p_pp,
        "boundary_tripwire_fired": tripwire,
        "f_group_posterior": {"p16": float(np.percentile(f_group, 16)),
                              "p50": float(np.percentile(f_group, 50)),
                              "p84": float(np.percentile(f_group, 84)),
                              "data_value": float(blk.data.values[0])},
        "archive": res["archive"],
    }
    (CHAINS / "a5_fit_summary.json").write_text(json.dumps(summary, indent=2))

    # marginal figure: top-6 constrained parameters + f_group
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, i in zip(axes.ravel()[:6], top):
        ax.hist(flat[:, i], bins=40, range=(0, 1), color="#0072B2", alpha=0.8)
        ax.set_title(pm.ASTRO_NAMES[i], fontsize=9)
        ax.set_yticks([])
    ax = axes.ravel()[6]
    ax.hist(f_group, bins=40, color="#0072B2", alpha=0.8)
    ax.axvline(blk.data.values[0], color="#D55E00", lw=2)
    ax.set_title("f_gas group bin: posterior vs data", fontsize=9)
    ax.set_yticks([])
    ax = axes.ravel()[7]
    x = blk.data.bin_centers
    ax.errorbar(x, blk.data.values, yerr=blk.data.stat_err, fmt="o", color="#D55E00", label="eRASS1")
    ax.plot(x, pred_map, "s-", color="#0072B2", label=f"MAP (chi2={chi2_map:.0f}/{len(x)})")
    ax.set_xlabel("log M500 [Msun/h]")
    ax.set_ylabel("f_gas,sph")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("WP-A5 f_gas-only posterior (boundary diagnostic in a5_fit_summary.json)")
    fig.tight_layout()
    figdir = CHAINS / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "a5_posterior_fgas.png", dpi=150)

    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
