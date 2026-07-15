"""analysis/paper3a/data_vectors — WP-A1 frozen observational data vectors.

Produced by Project A / WP1 (Observational Data Assembly & Freeze). See
``DATA_VECTOR_FREEZE.md`` (private plans tree,
``projectA/wp1-data-assembly/DATA_VECTOR_FREEZE.md``) for full provenance,
unit conventions, and mass-floor bookkeeping. This subpackage has no BIND
model dependency — it is a pure data-loading layer consumed by A3 (gate) and
A5 (likelihood).
"""

from .data_vectors import (
    DataVector,
    GasFractionCatalog,
    data_root,
    load_egas_erass1,
    load_kappa_y_pandey2025,
    load_kappa_y_pandey2025_full_covariance,
    load_kappa_y_pandey2025_nz,
    load_ksz_hadzhiyska2024,
    load_ksz_hadzhiyska2024_mass_bins,
    load_ksz_hadzhiyska2026_bgs_elg,
    load_ksz_qu2026_lrg_by_mass,
    load_ksz_qu2026_lrg_fiducial,
    load_ksz_ried_guachalla2025,
    load_ksz_ried_guachalla2025_correlation,
)

__all__ = [
    "DataVector",
    "GasFractionCatalog",
    "data_root",
    "load_egas_erass1",
    "load_kappa_y_pandey2025",
    "load_kappa_y_pandey2025_full_covariance",
    "load_kappa_y_pandey2025_nz",
    "load_ksz_hadzhiyska2024",
    "load_ksz_hadzhiyska2024_mass_bins",
    "load_ksz_hadzhiyska2026_bgs_elg",
    "load_ksz_qu2026_lrg_by_mass",
    "load_ksz_qu2026_lrg_fiducial",
    "load_ksz_ried_guachalla2025",
    "load_ksz_ried_guachalla2025_correlation",
]
