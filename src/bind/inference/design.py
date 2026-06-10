"""Stage 0 — experimental design for the BIND astro-parameter science runs.

Builds the parameter matrices that drive 1a (WL+tSZ cross-correlation vs astro
params) and 1b (non-Gaussian WL stats emulator):

* :func:`sobol_design`     — quasi-random Sobol coverage of the 30 astro params
  (the emulator training set), cosmology pinned at the TNG300 fiducial.
* :func:`oneparam_design`  — one-at-a-time sweep, ``levels`` per astro param,
  for sensitivity / derivative diagnostics.
* :func:`fiducial_vector`  — the single TNG300-fiducial vector (astro at the
  IllustrisTNG/CAMELS fiducial, cosmo at TNG300), used for the first validation
  against TNG300 full hydro and for the high-N_real fiducial run.

All vectors are full 35-dim BIND parameter vectors (so they drop straight into
``bind.paint`` / the lightcone pipeline).  The 5 cosmology slots
(:data:`COSMO_PARAM_INDICES`) are held at :data:`TNG300_COSMO`; the remaining 30
astro slots (:data:`ASTRO_PARAM_INDICES`) are the design variables, sampled in
their native CAMELS SB35 space (log10-uniform where ``LogFlag == 1``, else
linear) within the SB35 ``[MinVal, MaxVal]`` box.
"""

from __future__ import annotations

import numpy as np

from bind.params import (
    PARAM_NAMES,
    PARAM_MIN,
    PARAM_MAX,
    PARAM_LOG_FLAG,
    N_PARAMS,
    fiducial_params,
    vary_params,
)

# ── Parameter partition ───────────────────────────────────────────────────────
# Cosmology slots in the 35-dim SB35 vector (held fixed across the design).
_COSMO_NAMES = ("Omega0", "sigma8", "OmegaBaryon", "HubbleParam", "n_s")
COSMO_PARAM_INDICES: tuple[int, ...] = tuple(
    PARAM_NAMES.index(n) for n in _COSMO_NAMES
)
ASTRO_PARAM_INDICES: tuple[int, ...] = tuple(
    i for i in range(N_PARAMS) if i not in COSMO_PARAM_INDICES
)
assert len(ASTRO_PARAM_INDICES) == 30, len(ASTRO_PARAM_INDICES)

# TNG300-1 (IllustrisTNG, Planck15) cosmology — the lightcone is painted on the
# TNG300-Dark box, so the conditioning cosmology is pinned here.
TNG300_COSMO: dict[str, float] = {
    "Omega0": 0.3089,
    "sigma8": 0.8159,
    "OmegaBaryon": 0.0486,
    "HubbleParam": 0.6774,
    "n_s": 0.9667,
}


def _base_vector() -> np.ndarray:
    """Fiducial astro (CAMELS/IllustrisTNG) with cosmology set to TNG300."""
    return vary_params(TNG300_COSMO, base=fiducial_params())


def _unit_to_native(u: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Map unit-cube samples ``u`` (..., n) to native param values for ``idx``.

    Log-flagged params are mapped log10-uniformly, matching the CAMELS SB35 /
    Sobol prior used to train the model.
    """
    log_mask = PARAM_LOG_FLAG[idx] == 1
    lo, hi = PARAM_MIN[idx], PARAM_MAX[idx]
    log_lo = np.log10(np.where(log_mask, lo, 1.0))
    log_hi = np.log10(np.where(log_mask, hi, 1.0))
    log_vals = 10.0 ** (log_lo + u * (log_hi - log_lo))
    lin_vals = lo + u * (hi - lo)
    return np.where(log_mask, log_vals, lin_vals)


# ── Sobol design (emulator training set) ──────────────────────────────────────

def sobol_design(n: int = 256, *, seed: int = 0) -> np.ndarray:
    """``(n, 35)`` Sobol design over the 30 astro params; cosmo at TNG300.

    ``n`` should be a power of two for a balanced Sobol sequence (a warning is
    emitted otherwise).  The astro columns are drawn in the unit cube by
    ``scipy.stats.qmc.Sobol`` then mapped to native SB35 ranges.
    """
    from scipy.stats import qmc

    astro = np.asarray(ASTRO_PARAM_INDICES)
    sampler = qmc.Sobol(d=len(astro), scramble=True, seed=seed)
    m = int(round(np.log2(n)))
    if 2 ** m != n:
        import warnings
        warnings.warn(f"Sobol n={n} is not a power of 2; balance is best at 2^m.")
        u = sampler.random(n)
    else:
        u = sampler.random_base2(m)

    out = np.broadcast_to(_base_vector(), (n, N_PARAMS)).copy()
    out[:, astro] = _unit_to_native(u, astro)
    return out


# ── One-parameter sweep (sensitivity) ─────────────────────────────────────────

def oneparam_design(levels: int = 5) -> tuple[np.ndarray, list[dict]]:
    """One-at-a-time sweep: each astro param across ``levels`` values.

    Returns ``(params, meta)`` where ``params`` is ``(30*levels, 35)`` and
    ``meta[k] = {"param": name, "index": i, "level": j, "fraction": f}`` labels
    each row.  Levels span ``[MinVal, MaxVal]`` uniformly in the param's native
    sampling space (``fraction`` in ``[0, 1]``); all other params stay fiducial.
    """
    base = _base_vector()
    fracs = np.linspace(0.0, 1.0, levels)
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for i in ASTRO_PARAM_INDICES:
        native = _unit_to_native(fracs, np.array([i]))
        for j, (f, v) in enumerate(zip(fracs, native)):
            vec = base.copy()
            vec[i] = v
            rows.append(vec)
            meta.append({"param": PARAM_NAMES[i], "index": int(i),
                         "level": int(j), "fraction": float(f), "value": float(v)})
    return np.asarray(rows), meta


def twobound_design() -> tuple[np.ndarray, list[dict]]:
    """One-at-a-time **prior-bound** design: each astro param at Min then Max.

    Returns ``(params, meta)`` with ``params`` shape ``(60, 35)`` (30 astro params
    × {lower, upper} bound) and ``meta[k] = {"param", "index", "bound", "value"}``.
    All other params stay fiducial; cosmology fixed at TNG300.  Rows are ordered
    ``[p0_lo, p0_hi, p1_lo, p1_hi, ...]``.
    """
    base = _base_vector()
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for i in ASTRO_PARAM_INDICES:
        for bound, val in (("lower", PARAM_MIN[i]), ("upper", PARAM_MAX[i])):
            vec = base.copy()
            vec[i] = val
            rows.append(vec)
            meta.append({"param": PARAM_NAMES[i], "index": int(i),
                         "bound": bound, "value": float(val)})
    return np.asarray(rows), meta


def fiducial_vector() -> np.ndarray:
    """The single TNG300-fiducial 35-dim vector (astro fiducial, cosmo TNG300)."""
    return _base_vector()
