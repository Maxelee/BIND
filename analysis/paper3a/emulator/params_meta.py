"""SB35 parameter metadata without pandas.

``bind.params`` reads ``SB35_param_minmax.csv`` through pandas, which is
broken in the torch3 venv (GLIBCXX mismatch with the nix-store build). This
module reads the same CSV with the stdlib ``csv`` reader and exposes the same
arrays, plus the unit-cube transforms the emulator uses. The cosmology
indices and the fixed TNG fiducial cosmology match ``bind.wlemu.fit``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SB35_CSV = Path(__file__).resolve().parents[3] / "src" / "bind" / "assets" / "SB35_param_minmax.csv"

# The five SB35 parameters held fixed (IllustrisTNG fiducial cosmology) in
# the painted suite; the remaining 30 astro parameters are emulator inputs.
COSMO_IDX = (0, 1, 6, 7, 8)  # Omega0, sigma8, OmegaBaryon, HubbleParam, n_s
FIXED_COSMOLOGY = {"Omega0": 0.3089, "sigma8": 0.8159, "OmegaBaryon": 0.0486,
                   "HubbleParam": 0.6774, "n_s": 0.9667}


def _load():
    with open(SB35_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["ParamName"] for r in rows]
    lo = np.array([float(r["MinVal"]) for r in rows])
    hi = np.array([float(r["MaxVal"]) for r in rows])
    log = np.array([int(r["LogFlag"]) for r in rows], dtype=bool)
    fid = np.array([float(r["FiducialVal"]) for r in rows])
    desc = [r["Description"] for r in rows]
    return names, lo, hi, log, fid, desc


PARAM_NAMES, PARAM_MIN, PARAM_MAX, PARAM_LOG, PARAM_FIDUCIAL, PARAM_DESCRIPTIONS = _load()
N_PARAMS = len(PARAM_NAMES)
ASTRO_IDX = np.array([i for i in range(N_PARAMS) if i not in COSMO_IDX])

ASTRO_NAMES = [PARAM_NAMES[i] for i in ASTRO_IDX]
ASTRO_MIN = PARAM_MIN[ASTRO_IDX]
ASTRO_MAX = PARAM_MAX[ASTRO_IDX]
ASTRO_LOG = PARAM_LOG[ASTRO_IDX]
ASTRO_FIDUCIAL = PARAM_FIDUCIAL[ASTRO_IDX]


def astro_physical_to_unit(phys: np.ndarray) -> np.ndarray:
    """Map physical astro values ``(..., 30)`` to the unit cube."""
    p = np.asarray(phys, float)
    lo, hi = ASTRO_MIN, ASTRO_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        u_log = (np.log10(p) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    u_lin = (p - lo) / (hi - lo)
    return np.where(ASTRO_LOG, u_log, u_lin)


def astro_unit_to_physical(u: np.ndarray) -> np.ndarray:
    """Map unit-cube astro values ``(..., 30)`` to physical values."""
    u = np.asarray(u, float)
    lo, hi = ASTRO_MIN, ASTRO_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        p_log = 10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
    p_lin = lo + u * (hi - lo)
    return np.where(ASTRO_LOG, p_log, p_lin)


def table_params_to_unit(params35: np.ndarray) -> np.ndarray:
    """Extract the astro block of a stored 35-dim physical vector and map it
    to the unit cube. Raises if the cosmology block is not the fixed TNG
    fiducial (all painted bundles share it)."""
    p = np.asarray(params35, float)
    cosmo = p[..., list(COSMO_IDX)]
    expect = np.array(list(FIXED_COSMOLOGY.values()))
    if not np.allclose(cosmo, expect, rtol=1e-4):
        raise ValueError(f"cosmology block {cosmo} != fixed TNG fiducial {expect}")
    return astro_physical_to_unit(p[..., ASTRO_IDX])
