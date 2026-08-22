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
from bind.params import (  # noqa: E402
    N_PARAMS,
    PARAM_NAMES,
    TNG300_COSMOLOGY,
    fiducial_params,
    param_dataframe,
    random_params,
    tng300_params,
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
    "N_THERMO",
    "THERMO_KEYS",
    "PARAM_NAMES",
    "N_PARAMS",
    "TNG300_COSMOLOGY",
    "fiducial_params",
    "tng300_params",
    "random_params",
    "vary_param",
    "vary_params",
    "param_dataframe",
    "Emulator",
    "__version__",
]


def __getattr__(name: str):
    # Lazy: the emulator pulls in scikit-learn / zuko / kymatio, so only import on use.
    if name == "Emulator":
        from bind.emulator import Emulator
        return Emulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
