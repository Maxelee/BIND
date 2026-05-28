"""SB35 35-parameter metadata (names, fiducial, bounds, log-flag, aliases).

Reads the same SB35 CSV that `data.py` uses for normalization, but without
importing torch.  Provides short human-readable aliases for the well-known
cosmological + headline-feedback axes so the kSZ sensitivity/information plots
are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SB35_CSV = "/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv"

# Short aliases for the canonical CAMELS axes (index → label).  Everything else
# falls back to the CSV ParamName.
_ALIAS = {
    0: "Ω_m", 1: "σ8", 2: "A_SN1", 3: "A_AGN1", 4: "A_SN2", 5: "A_AGN2",
    6: "Ω_b", 7: "h", 8: "n_s", 11: "IMF_slope",
    22: "BH_acc", 23: "BH_Edd", 24: "A_AGN_kin", 25: "BH_radeff",
}

# Indices that are cosmological (not subgrid feedback).  The science question
# is about the *feedback* axes, so these are flagged separately.
COSMO_IDX = (0, 1, 6, 7, 8)


@dataclass(frozen=True)
class ParamMeta:
    names: list[str]          # full ParamName per index
    labels: list[str]         # short alias or ParamName
    fiducial: np.ndarray      # (35,)
    minv: np.ndarray          # (35,)
    maxv: np.ndarray          # (35,)
    logflag: np.ndarray       # (35,) int: 1 → sampled in log10
    is_cosmo: np.ndarray      # (35,) bool

    def normalized(self, theta: np.ndarray) -> np.ndarray:
        """Map raw θ to [0,1] using the (log-aware) SB35 bounds, for slopes
        that are comparable across parameters with very different ranges."""
        theta = np.asarray(theta, dtype=np.float64)
        lo, hi = self.minv.copy(), self.maxv.copy()
        t = theta.copy()
        logm = self.logflag.astype(bool)
        # guard against non-positive values under log
        with np.errstate(invalid="ignore", divide="ignore"):
            t[..., logm] = np.log10(np.clip(t[..., logm], 1e-30, None))
            lo[logm] = np.log10(np.clip(lo[logm], 1e-30, None))
            hi[logm] = np.log10(np.clip(hi[logm], 1e-30, None))
        span = np.where((hi - lo) == 0, 1.0, hi - lo)
        return (t - lo) / span


def load_param_meta() -> ParamMeta:
    m = pd.read_csv(SB35_CSV)
    names = m["ParamName"].astype(str).tolist()
    labels = [_ALIAS.get(i, names[i]) for i in range(len(names))]
    n = len(names)
    is_cosmo = np.zeros(n, dtype=bool)
    for i in COSMO_IDX:
        if i < n:
            is_cosmo[i] = True
    return ParamMeta(
        names=names,
        labels=labels,
        fiducial=m["FiducialVal"].to_numpy(dtype=np.float64),
        minv=m["MinVal"].to_numpy(dtype=np.float64),
        maxv=m["MaxVal"].to_numpy(dtype=np.float64),
        logflag=m["LogFlag"].to_numpy(dtype=np.int32),
        is_cosmo=is_cosmo,
    )
