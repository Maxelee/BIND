"""WP-B4 survey-realism layer: raytraced mock κ/y maps -> DES×ACT-realistic maps.

Produced by Project B WP-B4 (forward model; see bind-paper3-plans, private
repo). Map-level post-processing only — pure numpy/scipy (healpy lazily, for
the curved-sky smoothing wrapper); no bind/torch import — so the same layer
runs where the raytraced maps live (Popeye) and inside any mock-measurement
loop. Each module documents the published spec it targets (with arXiv ID) and
its convention choices; where a real survey-parameter file is needed but not
yet available it is accepted as an explicit input with a documented TODO, never
hard-coded silently.

Pipeline order for a mock κ+y pair (matched to the B1 data chain):
  κ:  source-plane weight (nz) -> add shape noise (shape_noise) -> smooth (smoothing)
  y:  source-plane weight (nz) -> apply ACT beam (beam)
then the imported B2 peak-find + stack (NOT part of this layer; needs B2's code
path, which does not exist yet — do not stub it here).
"""

from .beam import ACT_DR6_YMAP_FWHM_ARCMIN, BeamConfig, apply_beam
from .nz import (
    DESY3_BIN_MEAN_Z,
    SourcePlaneWeighting,
    smail_nz,
    source_plane_weights,
)
from .shape_noise import (
    DESY3_NEFF_PERBIN_ARCMIN2,
    DESY3_NEFF_TOTAL_ARCMIN2,
    DESY3_SIGMA_E,
    ShapeNoiseConfig,
    add_shape_noise,
    noise_map,
)
from .smoothing import (
    FIDUCIAL_SMOOTHING_ARCMIN,
    PAPER2_SMOOTHING_ARCMIN,
    SmoothingConfig,
    smooth_curved_sky,
    smooth_flat_sky,
)

__all__ = [
    # nz
    "DESY3_BIN_MEAN_Z",
    "SourcePlaneWeighting",
    "smail_nz",
    "source_plane_weights",
    # shape_noise
    "DESY3_SIGMA_E",
    "DESY3_NEFF_TOTAL_ARCMIN2",
    "DESY3_NEFF_PERBIN_ARCMIN2",
    "ShapeNoiseConfig",
    "add_shape_noise",
    "noise_map",
    # smoothing
    "FIDUCIAL_SMOOTHING_ARCMIN",
    "PAPER2_SMOOTHING_ARCMIN",
    "SmoothingConfig",
    "smooth_flat_sky",
    "smooth_curved_sky",
    # beam
    "ACT_DR6_YMAP_FWHM_ARCMIN",
    "BeamConfig",
    "apply_beam",
]
