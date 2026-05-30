"""BIND — flow-matching emulator that paints baryonic fields onto DMO maps."""

__version__ = "0.1.0"

from bind.inference.paint import (  # noqa: E402
    NATIVE_PIXEL_SIZE_MPCH,
    NATIVE_SLAB_DEPTH_MPCH,
    PATCH_PIX,
    Model,
    PaintResult,
    Simulation,
    extract_halo_cutouts,
    paint,
)
from bind.params import (  # noqa: E402
    PARAM_NAMES,
    N_PARAMS,
    fiducial_params,
    param_dataframe,
    random_params,
    vary_param,
    vary_params,
)

__all__ = [
    "Simulation",
    "Model",
    "paint",
    "PaintResult",
    "extract_halo_cutouts",
    "NATIVE_PIXEL_SIZE_MPCH",
    "NATIVE_SLAB_DEPTH_MPCH",
    "PATCH_PIX",
    "PARAM_NAMES",
    "N_PARAMS",
    "fiducial_params",
    "random_params",
    "vary_param",
    "vary_params",
    "param_dataframe",
    "__version__",
]
