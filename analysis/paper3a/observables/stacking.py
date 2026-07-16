"""Stacking + generative-scatter bookkeeping (WP-A2 task 6 support).

`stack_profiles` is the one weighted mean every operator funnels through.
`multi_sample_convergence` is the stochasticity control: BIND is generative
(per-halo scatter is real signal), and SHARED_CONTEXT caveat 4 warns that a
*single* sample over-smooths small scales — this measures how the stacked
observable moves as the per-halo sample count grows, so WP-A2 can fix the
sample count per operator with evidence instead of folklore.
"""

from __future__ import annotations

import numpy as np


def stack_profiles(profiles: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean profile over halos: (N_halos, N_bins) -> (N_bins,)."""
    p = np.asarray(profiles, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"profiles must be (N_halos, N_bins), got shape {p.shape}")
    if weights is None:
        return p.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    if w.shape != (p.shape[0],):
        raise ValueError(f"weights shape {w.shape} != (N_halos,) = ({p.shape[0]},)")
    if w.sum() <= 0:
        raise ValueError("weights must have positive sum")
    return (w[:, None] * p).sum(axis=0) / w.sum()


def multi_sample_convergence(
    per_sample_profiles: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Convergence of the stack vs per-halo generative sample count.

    per_sample_profiles: (N_samples, N_halos, N_bins) — the same halos
    painted N_samples times.

    Returns
    -------
    dict with:
      "stack_full"    : (N_bins,) stack using all samples per halo
      "stack_vs_k"    : (N_samples, N_bins) stack using the first k samples
      "frac_dev_vs_k" : (N_samples,) max |stack_k/stack_full - 1| over bins
                        with non-negligible amplitude
      "per_halo_scatter" : (N_bins,) mean-over-halos std-over-samples — the
                        intrinsic generative scatter entering Sigma_model
    """
    x = np.asarray(per_sample_profiles, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"expected (N_samples, N_halos, N_bins), got shape {x.shape}")
    n_s = x.shape[0]
    stack_full = stack_profiles(x.mean(axis=0), weights)
    stack_vs_k = np.stack([stack_profiles(x[: k + 1].mean(axis=0), weights) for k in range(n_s)])
    scale = np.max(np.abs(stack_full))
    meaningful = np.abs(stack_full) > 0.01 * scale if scale > 0 else np.zeros(len(stack_full), bool)
    if meaningful.any():
        frac_dev = np.max(
            np.abs(stack_vs_k[:, meaningful] / stack_full[meaningful] - 1.0), axis=1
        )
    else:
        frac_dev = np.zeros(n_s)
    return {
        "stack_full": stack_full,
        "stack_vs_k": stack_vs_k,
        "frac_dev_vs_k": frac_dev,
        "per_halo_scatter": x.std(axis=0, ddof=1).mean(axis=0),
    }
