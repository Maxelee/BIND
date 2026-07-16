"""WP-B1 data-side map layer: loaders, footprint harmonization, peaks, stacking.

Produced by Project B WP-B1 (see bind-paper3-plans, private repo). The peak
chain implements the frozen `PEAK_DEFINITION.md` verbatim — B4's mock chain
must stay identical. Survey-file paths live in `loaders`; the peak and stack
logic is pure healpy/pixell+numpy and unit-tested on synthetic maps.
"""

from .footprint import (
    DEFAULT_ACT_THRESHOLD,
    FULL_SKY_DEG2,
    HarmonizedFootprint,
    act_mask_to_healpix,
    harmonize,
)
from .loaders import (
    DES_NSIDE,
    DES_VARIANTS,
    DR5Catalog,
    load_act_mask,
    load_act_ymap,
    load_des_map,
    load_des_mask,
    load_dr5_catalog,
)
from .peaks import (
    FIDUCIAL_SMOOTHING_ARCMIN,
    NU_BIN_EDGES,
    SMOOTHING_SET_ARCMIN,
    PeakCatalog,
    build_peak_catalog,
    exclusion_keep,
    local_maxima,
    map_nu_stats,
    smooth_masked,
)
from .stack import (
    StackResult,
    extract_thumbnails,
    radial_profile,
    random_positions_in_mask,
    stack_catalog,
    stack_thumbnails,
)

__all__ = [
    # loaders
    "DES_NSIDE", "DES_VARIANTS", "DR5Catalog",
    "load_des_mask", "load_des_map", "load_act_ymap", "load_act_mask", "load_dr5_catalog",
    # footprint
    "FULL_SKY_DEG2", "DEFAULT_ACT_THRESHOLD", "HarmonizedFootprint",
    "act_mask_to_healpix", "harmonize",
    # peaks
    "FIDUCIAL_SMOOTHING_ARCMIN", "SMOOTHING_SET_ARCMIN", "NU_BIN_EDGES", "PeakCatalog",
    "smooth_masked", "local_maxima", "map_nu_stats", "exclusion_keep", "build_peak_catalog",
    # stack
    "StackResult", "extract_thumbnails", "stack_thumbnails", "radial_profile",
    "stack_catalog", "random_positions_in_mask",
]
