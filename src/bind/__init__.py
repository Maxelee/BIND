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

__all__ = [
    "Simulation",
    "Model",
    "paint",
    "PaintResult",
    "extract_halo_cutouts",
    "NATIVE_PIXEL_SIZE_MPCH",
    "NATIVE_SLAB_DEPTH_MPCH",
    "PATCH_PIX",
    "__version__",
]
