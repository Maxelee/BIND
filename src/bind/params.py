"""Parameter helpers for BIND.

The 35-dim CAMELS SB35 parameter vector is the conditioning input to the
flow-matching emulator. Bounds, fiducial values, log-sampling flags, and
human-readable descriptions are bundled in
``src/bind/assets/SB35_param_minmax.csv`` and exposed here as numpy arrays
plus three convenience functions:

- :func:`fiducial_params` — the fiducial CAMELS-IllustrisTNG vector.
- :func:`random_params`   — uniform sample from the prior box (log10-uniform
  for parameters with ``LogFlag == 1``).
- :func:`vary_param`      — fiducial vector with a single parameter set to a
  given value (or fraction of its range).

Use :func:`param_dataframe` to inspect names/ranges interactively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_ASSETS_DIR = Path(__file__).parent / "assets"
SB35_CSV = _ASSETS_DIR / "SB35_param_minmax.csv"

_meta = pd.read_csv(SB35_CSV)

PARAM_NAMES: list[str] = _meta["ParamName"].tolist()                    # len 35
PARAM_DESCRIPTIONS: list[str] = _meta["Description"].tolist()
PARAM_FIDUCIAL: np.ndarray = _meta["FiducialVal"].to_numpy(np.float64)  # (35,)
PARAM_MIN: np.ndarray = _meta["MinVal"].to_numpy(np.float64)            # (35,)
PARAM_MAX: np.ndarray = _meta["MaxVal"].to_numpy(np.float64)            # (35,)
PARAM_LOG_FLAG: np.ndarray = _meta["LogFlag"].to_numpy(np.int32)        # (35,)
N_PARAMS = len(PARAM_NAMES)

_NAME_TO_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}


def param_dataframe() -> pd.DataFrame:
    """Return the full parameter metadata table (a copy)."""
    return _meta.copy()


def _resolve_index(name_or_idx: str | int) -> int:
    if isinstance(name_or_idx, str):
        if name_or_idx not in _NAME_TO_IDX:
            raise KeyError(
                f"unknown parameter '{name_or_idx}'. "
                f"Valid names: {PARAM_NAMES}"
            )
        return _NAME_TO_IDX[name_or_idx]
    i = int(name_or_idx)
    if not 0 <= i < N_PARAMS:
        raise IndexError(f"parameter index {i} out of range [0, {N_PARAMS})")
    return i


def fiducial_params() -> np.ndarray:
    """Return the fiducial 35-dim parameter vector (float64 copy)."""
    return PARAM_FIDUCIAL.copy()


def random_params(
    n: int = 1,
    *,
    rng: np.random.Generator | int | None = None,
    fix: dict[str | int, float] | None = None,
) -> np.ndarray:
    """Draw ``n`` parameter vectors uniformly from the SB35 prior box.

    Parameters with ``LogFlag == 1`` are sampled log10-uniformly (matching the
    CAMELS SB35 / Sobol design). Pass ``fix={name_or_idx: value, ...}`` to pin
    individual parameters; everything else is randomized.

    Returns shape ``(35,)`` if ``n == 1``, else ``(n, 35)``.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    n_safe = max(int(n), 1)
    out = np.empty((n_safe, N_PARAMS), dtype=np.float64)

    log_mask = PARAM_LOG_FLAG == 1
    u = rng.uniform(0.0, 1.0, size=(n_safe, N_PARAMS))

    # log10-uniform for log-flagged params
    log_lo = np.log10(np.where(log_mask, PARAM_MIN, 1.0))
    log_hi = np.log10(np.where(log_mask, PARAM_MAX, 1.0))
    log_vals = 10 ** (log_lo + u * (log_hi - log_lo))

    # linear-uniform for the rest
    lin_vals = PARAM_MIN + u * (PARAM_MAX - PARAM_MIN)

    out[:] = np.where(log_mask, log_vals, lin_vals)

    if fix:
        for k, v in fix.items():
            out[:, _resolve_index(k)] = float(v)

    return out[0] if n == 1 else out


def vary_param(
    name_or_idx: str | int,
    value: float | None = None,
    *,
    fraction: float | None = None,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Return a parameter vector equal to ``base`` (default: fiducial) with a
    single parameter overridden.

    Specify the new value either as ``value`` (raw units) or as
    ``fraction`` ∈ [0, 1] interpolating between MinVal and MaxVal in the
    parameter's native sampling space (log10-uniform if ``LogFlag == 1``).
    """
    if (value is None) == (fraction is None):
        raise ValueError("provide exactly one of `value` or `fraction`")

    idx = _resolve_index(name_or_idx)
    p = (fiducial_params() if base is None else np.asarray(base, dtype=np.float64).copy())

    if fraction is not None:
        f = float(fraction)
        if PARAM_LOG_FLAG[idx] == 1:
            lo, hi = np.log10(PARAM_MIN[idx]), np.log10(PARAM_MAX[idx])
            p[idx] = 10 ** (lo + f * (hi - lo))
        else:
            p[idx] = PARAM_MIN[idx] + f * (PARAM_MAX[idx] - PARAM_MIN[idx])
    else:
        p[idx] = float(value)

    return p


def vary_params(
    overrides: dict[str | int, float], *, base: np.ndarray | None = None
) -> np.ndarray:
    """Like :func:`vary_param` but for several parameters at once.

    ``overrides`` maps name (or index) → raw value. Everything else stays at
    ``base`` (default fiducial).
    """
    p = fiducial_params() if base is None else np.asarray(base, dtype=np.float64).copy()
    for k, v in overrides.items():
        p[_resolve_index(k)] = float(v)
    return p


__all__ = [
    "PARAM_NAMES",
    "PARAM_DESCRIPTIONS",
    "PARAM_FIDUCIAL",
    "PARAM_MIN",
    "PARAM_MAX",
    "PARAM_LOG_FLAG",
    "N_PARAMS",
    "SB35_CSV",
    "param_dataframe",
    "fiducial_params",
    "random_params",
    "vary_param",
    "vary_params",
]
