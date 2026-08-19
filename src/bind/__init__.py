"""BIND — flow-matching emulator that paints baryonic fields onto DMO maps."""

__version__ = "0.1.0"

from bind.data import N_THERMO, THERMO_KEYS  # noqa: E402
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
from bind.inference.paint_stages import (  # noqa: E402
    generate_from_stage1,
    project_and_extract,
    recomposite_from_saved,
    recomposite_slab,
)
from bind.params import (  # noqa: E402
    N_PARAMS,
    PARAM_NAMES,
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
    "project_and_extract",
    "generate_from_stage1",
    "recomposite_slab",
    "recomposite_from_saved",
    "NATIVE_PIXEL_SIZE_MPCH",
    "NATIVE_SLAB_DEPTH_MPCH",
    "PATCH_PIX",
    "N_THERMO",
    "THERMO_KEYS",
    "PARAM_NAMES",
    "N_PARAMS",
    "fiducial_params",
    "random_params",
    "vary_param",
    "vary_params",
    "param_dataframe",
    "__version__",
]
