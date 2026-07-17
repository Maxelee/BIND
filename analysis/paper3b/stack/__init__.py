"""WP-B2 stacking pipeline: y stacked at κ-peak positions (the measurement).

Produced by Project B WP-B2 (see bind-paper3-plans, private repo; conventions
frozen in that WP's MEASUREMENT_SPEC.md). The chain is (map, catalog)-agnostic
— B4 imports run_peak_stack unchanged for mocks. Peak inputs come from WP-B1
(`analysis/paper3b/maps`); nothing here redefines peaks.
"""

from .cap import cap_filter, cap_filter_multi, cap_masks
from .covariance import eigenvalue_drift, jackknife_mean_cov, patch_ids
from .results import ARCHIVE_DIR, load_result, save_result
from .stacker import (
    CAP_RADII_ARCMIN,
    FIDUCIAL_RADIUS_INDEX,
    NSIDE_JK_DEFAULT,
    NSIDE_JK_STABILITY,
    NU_STACK_EDGES,
    PeakStackResult,
    run_peak_stack,
)

__all__ = [
    "cap_masks", "cap_filter", "cap_filter_multi",
    "patch_ids", "jackknife_mean_cov", "eigenvalue_drift",
    "ARCHIVE_DIR", "save_result", "load_result",
    "CAP_RADII_ARCMIN", "FIDUCIAL_RADIUS_INDEX", "NU_STACK_EDGES",
    "NSIDE_JK_DEFAULT", "NSIDE_JK_STABILITY",
    "PeakStackResult", "run_peak_stack",
]
