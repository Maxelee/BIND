"""The FROZEN WP-B3 measurement — the only data-side entry point for B5.

Freeze signed by Max E. Lee 2026-07-17 (plans repo
`projectB/wp3-null-systematics/FREEZE.md`): Wiener headline + GLIMPSE
cross-check; Sigma_sys = the DECOMPOSED (beta/T-axis) rank-1 CIB term; the
nu 0-1 bin is diagnostic-only and excluded from the B5 vector.

`load_frozen()` verifies the sha256 of every file against the freeze record
before returning — a hash mismatch means the data side changed after the
freeze and raises instead of loading (blinding discipline).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .results import load_result
from .stacker import FIDUCIAL_RADIUS_INDEX

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
FROZEN_HASHES = {
    "wp2_measurement/stack_wiener_sm2am_fid.npz":
        "1b545069324051d4d36113a4f819583b47a39243208fd964a290c859a2617ac8",
    "wp2_measurement/stack_glimpse_sm2am_fid.npz":
        "0cef5a5b251a6ed5db4db80d7a626fe9cb1590e5e153fc81ea6343bc8377da66",
    "wp3_nulls/sigma_sys_wiener_decomposed.npz":
        "b53e5bb6583901d172ac23b2be77fd56b71c31fbe8143e4a310c2ed1eaa93e8a",
    "wp3_nulls/sigma_sys_glimpse_decomposed.npz":
        "60f5cfef6f1d45107de9609e2327b325c614f0e3c41b75571349712ba2fc8b3f",
}
B5_BINS = slice(1, 5)   # nu >= 1; the 0-1 bin is diagnostic-only (FREEZE.md)


@dataclass
class FrozenMeasurement:
    """One variant's frozen vector + total covariance (stat + sys)."""

    variant: str
    nu_edges: np.ndarray        # (5,) edges of the 4 B5 bins
    y: np.ndarray               # (4,) <Y_CAP(4')> per bin [y arcmin^2]
    cov_stat: np.ndarray        # (4,4) jackknife (nside_jk=8), 4' radius
    cov_sys: np.ndarray         # (4,4) rank-1 CIB term (decomposed)
    n_per_bin: np.ndarray

    @property
    def cov_total(self) -> np.ndarray:
        return self.cov_stat + self.cov_sys


def _verify(relpath: str) -> Path:
    path = B_ROOT / relpath
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != FROZEN_HASHES[relpath]:
        raise RuntimeError(
            f"FROZEN-FILE HASH MISMATCH for {relpath}: the data side changed "
            f"after the 2026-07-17 freeze (got {h[:12]}..., frozen "
            f"{FROZEN_HASHES[relpath][:12]}...)")
    return path


def load_frozen(variant: str = "wiener") -> FrozenMeasurement:
    """The signed measurement (headline: 'wiener'; cross-check: 'glimpse')."""
    if variant not in ("wiener", "glimpse"):
        raise ValueError(f"variant {variant!r} not in the freeze")
    _verify(f"wp2_measurement/stack_{variant}_sm2am_fid.npz")
    sys_path = _verify(f"wp3_nulls/sigma_sys_{variant}_decomposed.npz")

    res = load_result(f"{variant}_sm2am_fid", B_ROOT / "wp2_measurement")
    J = FIDUCIAL_RADIUS_INDEX
    s = np.load(sys_path)
    return FrozenMeasurement(
        variant=variant,
        nu_edges=res.nu_edges[1:],                       # edges of bins 1..4
        y=res.y_mean[B5_BINS, J].copy(),
        cov_stat=_stat_block(res, J),
        cov_sys=np.asarray(s["sigma_sys"])[1:5, 1:5],
        n_per_bin=res.n_per_bin[B5_BINS].copy(),
    )


def _stat_block(res, J: int) -> np.ndarray:
    """(4,4) statistical covariance at the headline radius.

    The per-bin jackknifes are independent across nu bins (disjoint peak
    sets grouped by the same patches — cross-bin patch correlations are
    second-order and were not persisted per-pair), so the frozen Sigma_stat
    is diagonal by construction; recorded as such in FREEZE.md conventions.
    """
    return np.diag([res.y_cov[b][J, J] for b in range(1, 5)])
