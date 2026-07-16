"""WP-A2 observable-matching operators: BIND maps -> mock data vectors.

Produced by Project A WP-A2 (see bind-paper3-plans, private repo). Pure
numpy — no bind/torch import — so the same operators run identically on
rusty (development, CAMELS-scale checks) and Popeye (TNG300 truth
validation), and A4 can call them inside its emulator training loop without
dragging in the painting stack.

Operator contract: every operator takes (map(s), PatchGeometry, config) and
returns arrays aligned bin-for-bin with the corresponding WP-A1
``DataVector.bins`` — configs pull radii/apertures from the A1 loaders
rather than retyping them.
"""

from .constants import TNG_H, TNG_OMEGA_M, X_E_FULLY_IONIZED
from .fgas import CylToSphCorrection, aperture_gas_mass_msunh, fgas_cylindrical
from .filters import (
    aperture_sum,
    cap_photometry,
    cap_weight_map,
    disk_weight_map,
    gaussian_beam_convolve,
    require_fits_in_patch,
)
from .geometry import FlatLCDM, PatchGeometry
from .ksz import (
    KSZOperatorConfig,
    ksz_cap_profile,
    minimum_cutout_extent_hmpc,
    tau_map_from_gas,
)
from .mock_sample import MockSampleConfig, draw_center_offsets_hmpc, sample_weights
from .stacking import multi_sample_convergence, stack_profiles
from .truth_validation import save_sigma_model, stacked_residual_bootstrap
from .ym import YMRelation, fit_ym_relation, y_aperture_mpc2

__all__ = [
    "FlatLCDM",
    "PatchGeometry",
    "TNG_H",
    "TNG_OMEGA_M",
    "X_E_FULLY_IONIZED",
    "CylToSphCorrection",
    "aperture_gas_mass_msunh",
    "fgas_cylindrical",
    "aperture_sum",
    "cap_photometry",
    "cap_weight_map",
    "disk_weight_map",
    "gaussian_beam_convolve",
    "require_fits_in_patch",
    "KSZOperatorConfig",
    "ksz_cap_profile",
    "minimum_cutout_extent_hmpc",
    "tau_map_from_gas",
    "MockSampleConfig",
    "draw_center_offsets_hmpc",
    "sample_weights",
    "multi_sample_convergence",
    "stack_profiles",
    "save_sigma_model",
    "stacked_residual_bootstrap",
    "YMRelation",
    "fit_ym_relation",
    "y_aperture_mpc2",
]
