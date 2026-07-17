"""Physical constants for the WP-A2 observable operators.

Values are copied verbatim from ``bind.inference.pipeline`` (which itself
ports them from ``make_train_data/add_gas_thermo_maps.py``) so that any
quantity this module derives from a painted or truth map uses *exactly* the
constants that produced the map in the first place. Do not "improve" a value
here without changing it in both places — a mismatch shows up as a fake
sub-percent bias in the WP-A2 truth validation.

This module deliberately does not import ``bind`` (torch/Pylians/h5py chain);
the observables package is pure numpy, like ``analysis.paper3a.data_vectors``.
"""

GAMMA = 5.0 / 3.0
X_H = 0.76                          # hydrogen mass fraction
M_PROTON_KG = 1.6726219e-27         # kg
K_B_J_PER_K = 1.380649e-23          # J/K
SIGMA_T_M2 = 6.6524587e-29          # Thomson cross-section, m^2
M_E_C2_J = 8.187105776e-14          # electron rest energy, J
KPC_IN_M = 3.085677581e19           # m per kpc
MPC_IN_M = KPC_IN_M * 1e3           # m per Mpc
MSUN_KG = 1.989e30                  # kg
KEV_IN_J = 1.602176634e-16          # J per keV

C_KM_S = 299792.458                 # speed of light, km/s
T_CMB_UK = 2.7255e6                 # CMB monopole temperature, microkelvin

# Fully-ionized electron abundance x_e = n_e / (X_H * n_H_total) for
# primordial composition: n_e = n_H + 2 n_He = (X_H + Y_He/2) * rho / m_p
# with Y_He = 1 - X_H  ->  x_e = (X_H + (1-X_H)/2) / X_H.
# This matches the TNG convention where ElectronAbundance == 1.158 for
# fully ionized gas at X_H = 0.76. The kSZ tau operator uses this as the
# default ionization state of painted gas (no per-pixel x_e channel exists);
# the WP-A2 truth validation quantifies the residual from cold/star-forming
# gas where this overestimates n_e.
X_E_FULLY_IONIZED = (X_H + 0.5 * (1.0 - X_H)) / X_H  # = 1.1578...

# TNG / CAMELS-TNG fiducial cosmology (Planck 2015); SB35 varies Omega_m and
# h across its cosmology dimensions, so operators take cosmology as an input
# with this as the default rather than hard-coding it downstream.
TNG_OMEGA_M = 0.3089
TNG_H = 0.6774
TNG_OMEGA_B = 0.0486

# Critical density today in h-units: 2.775e11 (Msun/h)/(Mpc/h)^3 (comoving,
# h-independent in these units). Used for the cosmic-mean gas surface-density
# background that composite gas maps need for compensated-aperture work
# (2026-07-17 gate dry-run finding: without it, CAP annuli compensate
# against zeros between pasted apertures).
RHO_CRIT0_MSUNH_PER_MPCH3 = 2.775e11
