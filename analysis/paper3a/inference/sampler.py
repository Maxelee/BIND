"""MCMC over the 30-dim SB35 unit cube (emcee) + chain diagnostics.

Prior: uniform on [0, 1]^30 — exactly the space CAMELS sampled (log-flagged
parameters are log-uniform there by construction, so the unit-cube uniform
IS the SB35 prior). Walkers initialized from the prior; log-prob is the sum
of the active blocks' Gaussian log-likelihoods (vectorized over walkers).

Diagnostics: split-chain R-hat and a crude ESS from emcee's integrated
autocorrelation time; both recorded in the chain archive. The archive npz
is the one-command reproducibility surface the plan requires.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

CHAINS = Path("/mnt/ceph/users/mlee1/paper3/A/wp5_chains")


def log_prob_factory(blocks):
    def log_prob(U):
        U = np.atleast_2d(U)
        lp = np.zeros(len(U))
        bad = np.any((U < 0.0) | (U > 1.0), axis=1)
        lp[bad] = -np.inf
        good = ~bad
        if good.any():
            total = np.zeros(good.sum())
            for b in blocks:
                total += b.loglike(U[good])
            lp[good] = total
        return lp
    return log_prob


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """Gelman-Rubin over split walker chains: chain (nsteps, nwalkers, d)."""
    n = chain.shape[0] // 2
    halves = np.concatenate([chain[:n], chain[n:2 * n]], axis=1)  # (n, 2W, d)
    m = halves.mean(axis=0)                                       # (2W, d)
    w = halves.var(axis=0, ddof=1).mean(axis=0)
    b = n * m.var(axis=0, ddof=1)
    var = (n - 1) / n * w + b / n
    return np.sqrt(var / w)


def run_mcmc(blocks, n_walkers: int = 64, n_steps: int = 3000,
             n_burn: int = 500, seed: int = 0, label: str = "fit",
             archive: bool = True, ndim: int = 30) -> dict:
    """ndim = 30 for theta-only likelihoods; 31 when a block carries the
    f_sat nuisance in column 30 (see `inference.ksz.KszBlock`)."""
    import emcee

    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.02, 0.98, size=(n_walkers, ndim))
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_factory(blocks),
                                    vectorize=True)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain()                                   # (S, W, d)
    post = chain[n_burn:]
    flat = post.reshape(-1, ndim)
    rhat = split_rhat(post)
    try:
        tau = sampler.get_autocorr_time(discard=n_burn, tol=0)
        ess = post.shape[0] * n_walkers / np.maximum(tau, 1.0)
    except Exception:
        tau = np.full(ndim, np.nan)
        ess = np.full(ndim, np.nan)

    out = {
        "flat": flat, "rhat": rhat, "tau": tau, "ess": ess,
        "logp": sampler.get_log_prob()[n_burn:].reshape(-1),
        "acceptance": float(np.mean(sampler.acceptance_fraction)),
    }
    if archive:
        CHAINS.mkdir(parents=True, exist_ok=True)
        path = CHAINS / f"a5_{label}_seed{seed}.npz"
        np.savez_compressed(
            path, chain=post.astype(np.float32), rhat=rhat, tau=tau, ess=ess,
            logp=out["logp"].astype(np.float32),
            provenance=np.array(json.dumps({
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "n_walkers": n_walkers, "n_steps": n_steps, "n_burn": n_burn,
                "seed": seed, "acceptance": out["acceptance"],
                "blocks": [type(b).__name__ for b in blocks]})))
        out["archive"] = str(path)
    return out
