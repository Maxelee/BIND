"""WP-A5 convergence-grade chains (decision 2): one seed per invocation.

Fixes the exploratory run's pathologies for publication numbers:
- differential-evolution move mixture (DEMove 80% / DESnookerMove 20%) —
  the stretch move's ~6% acceptance in 30-dim is the textbook case for DE;
- 128 walkers x 30_000 steps, 8_000 burn, thinned archive;
- 4 independent seeds (disjoint initializations) -> cross-chain R-hat on
  the constrained directions in the assembly step, not split-chain only.

Usage:   python run_a5_chains_final.py SEED          (SEED in 0..3)
Assemble: python run_a5_chains_final.py --assemble   (after all four)

Outputs: wp5_chains/a5_final_seed{K}.npz, then a5_final_summary.json +
figures/a5_posterior_final.png (assembly re-uses the exploratory summary
code paths; the exploratory chain stays archived for the record).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, log_prob_factory  # noqa: E402

N_WALKERS = 128
N_STEPS = 30_000
N_BURN = 8_000
THIN = 15


def run_seed(seed: int) -> None:
    import emcee

    blk = FgasBlock()
    rng = np.random.default_rng(100 + seed)
    p0 = rng.uniform(0.02, 0.98, size=(N_WALKERS, 30))
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(N_WALKERS, 30, log_prob_factory([blk]),
                                    vectorize=True, moves=moves)
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    chain = sampler.get_chain(discard=N_BURN, thin=THIN)          # (S, W, 30)
    logp = sampler.get_log_prob(discard=N_BURN, thin=THIN)
    CHAINS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHAINS / f"a5_final_seed{seed}.npz",
        chain=chain.astype(np.float32), logp=logp.astype(np.float32),
        acceptance=np.mean(sampler.acceptance_fraction),
        settings=np.array(json.dumps({"n_walkers": N_WALKERS, "n_steps": N_STEPS,
                                      "n_burn": N_BURN, "thin": THIN,
                                      "moves": "DE 0.8 + DESnooker 0.2",
                                      "seed": seed})))
    print(f"seed {seed}: acceptance {np.mean(sampler.acceptance_fraction):.3f}, "
          f"archived {chain.shape}")


def cross_chain_rhat(chains: list[np.ndarray]) -> np.ndarray:
    """R-hat treating each seed's flattened samples as one chain: (K, N, d)."""
    x = np.stack([c.reshape(-1, c.shape[-1]) for c in chains])    # (K, N, d)
    n = x.shape[1]
    m = x.mean(axis=1)                                            # (K, d)
    w = x.var(axis=1, ddof=1).mean(axis=0)
    b = n * m.var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def assemble() -> None:
    blk = FgasBlock()
    chains, logps, accs = [], [], []
    for k in range(4):
        f = np.load(CHAINS / f"a5_final_seed{k}.npz", allow_pickle=False)
        chains.append(f["chain"].astype(float))
        logps.append(f["logp"].astype(float))
        accs.append(float(f["acceptance"]))
    rhat = cross_chain_rhat(chains)
    flat = np.concatenate([c.reshape(-1, 30) for c in chains])
    logp = np.concatenate([lp.reshape(-1) for lp in logps])

    sub = flat[:: max(1, len(flat) // 40_000)]
    f_group = blk.predict(sub)[:, 0]
    # cross-chain R-hat of the derived summary
    fg_chains = [blk.predict(c.reshape(-1, 30)[:: max(1, c.size // 30 // 8000)])[:, 0]
                 for c in chains]
    nmin = min(len(g) for g in fg_chains)
    rhat_fg = float(cross_chain_rhat([g[:nmin, None] for g in fg_chains])[0])

    imap = int(np.argmax(logp))
    u_map = flat[imap]
    chi2_map = float(blk.chi2(u_map)[0])
    corr = np.array([abs(np.corrcoef(sub[:, i], f_group)[0, 1]) for i in range(30)])
    top = np.argsort(-corr)[:6]
    edge = {pm.ASTRO_NAMES[i]: {"low": float(np.mean(flat[:, i] < 0.05)),
                                "high": float(np.mean(flat[:, i] > 0.95))}
            for i in range(30)}
    summary = {
        "acceptance_by_seed": accs,
        "cross_chain_rhat_max": float(rhat.max()),
        "cross_chain_rhat_top6": {pm.ASTRO_NAMES[i]: float(rhat[i]) for i in top},
        "cross_chain_rhat_f_group": rhat_fg,
        "map": {"chi2": chi2_map, "pred": blk.predict(u_map)[0].tolist(),
                "data": blk.data.values.tolist(),
                "theta_map_named": {pm.ASTRO_NAMES[i]: float(u_map[i]) for i in top}},
        "f_group_posterior": {"p16": float(np.percentile(f_group, 16)),
                              "p50": float(np.percentile(f_group, 50)),
                              "p84": float(np.percentile(f_group, 84)),
                              "data_value": float(blk.data.values[0])},
        "boundary_tripwire_fired": {n: v for n, v in edge.items()
                                    if max(v.values()) > 0.15},
        "n_samples": int(len(flat)),
    }
    (CHAINS / "a5_final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "--assemble":
        assemble()
    else:
        run_seed(int(sys.argv[1]))
