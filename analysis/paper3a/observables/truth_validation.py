"""Truth-validation bookkeeping: painted-vs-truth residuals -> Sigma_model (WP-A2 task 5).

The Popeye half of WP-A2 runs the operators twice per halo — once on
TNG300-hydro truth projections, once on BIND-painted maps of the matching
DMO halos — and hands the per-halo observable arrays to this module. What
comes out is the model-error term WP-A5 adds to every likelihood:

- the stacked bias vector (does the painted stack reproduce the truth stack
  within the plan's tolerance?), and
- Sigma_model, the covariance of that stacked residual, estimated by
  bootstrap over halos (captures both the mean-bias uncertainty and the
  halo-to-halo residual correlation across bins).

Pure numpy; no bind import. Persist results with `save_sigma_model` so the
provenance (which truth run, which checkpoint, which operator config)
travels with the matrix into A5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stacking import stack_profiles


def stacked_residual_bootstrap(
    per_halo_pred: np.ndarray,
    per_halo_truth: np.ndarray,
    weights: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bias and covariance of stack(pred) - stack(truth) over matched halos.

    per_halo_pred / per_halo_truth : (N_halos, N_bins), same halos, same
        operator config, same bin grid (pred may be pre-averaged over
        generative samples — do that upstream with `multi_sample_convergence`
        so the sample-count choice is explicit).

    Returns dict with "bias" (N_bins,), "sigma_model" (N_bins, N_bins),
    "frac_bias" (bias / |truth stack|, NaN where truth ~ 0), "n_halos".
    """
    pred = np.asarray(per_halo_pred, dtype=float)
    truth = np.asarray(per_halo_truth, dtype=float)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(f"pred {pred.shape} and truth {truth.shape} must be identical (N_halos, N_bins)")
    n_halos = pred.shape[0]
    resid = pred - truth

    bias = stack_profiles(resid, weights)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, pred.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_halos, n_halos)
        w = None if weights is None else np.asarray(weights, dtype=float)[idx]
        boots[b] = stack_profiles(resid[idx], w)
    sigma_model = np.cov(boots, rowvar=False)

    truth_stack = stack_profiles(truth, weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_bias = np.where(np.abs(truth_stack) > 0, bias / np.abs(truth_stack), np.nan)

    return {
        "bias": bias,
        "sigma_model": np.atleast_2d(sigma_model),
        "frac_bias": frac_bias,
        "n_halos": n_halos,
    }


def save_sigma_model(path: str | Path, result: dict, provenance: dict) -> None:
    """Persist a Sigma_model result + its provenance next to each other.

    Writes <path>.npz (arrays) and <path>.provenance.json (what produced it:
    truth run, checkpoint, operator config, script, date). Refuses to write
    without at least those keys named — an unattributed Sigma_model is
    exactly the kind of artifact SHARED_CONTEXT bans.
    """
    required = {"truth_inputs", "model_or_checkpoint", "operator_config", "script", "date"}
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"provenance missing required keys: {sorted(missing)}")
    path = Path(path)
    np.savez(
        path.with_suffix(".npz"),
        bias=result["bias"],
        sigma_model=result["sigma_model"],
        frac_bias=result["frac_bias"],
        n_halos=result["n_halos"],
    )
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
