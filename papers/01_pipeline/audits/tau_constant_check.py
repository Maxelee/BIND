#!/usr/bin/env python
"""Investigation 1 (small_items_2026-08-03.md, item 1) -- TAU-CONVERSION PIN.

Derives, from first principles, the physical assumptions baked into the tau
(electron-column / kSZ optical depth) conversion used throughout the BIND
lightcone pipeline:

    tau = sigma_T * (x_e/m_p) * Sigma_gas            [x_e = 0.88]

and numerically reproduces the specific constant printed by the fig-6 cell of
papers/01_pipeline/_build_figures_nb.py,

    tau = 2.219785e-14 * Sigma_gas[Msun/h per patch pixel]   at snap 96 (a=0.96738)

Sources cross-checked (grep'd, not re-typed from memory):
  - src/bind/inference/lightcone_maps.py  lines 41-57  (SIGMA_T, M_P, MSUN_G,
    MPC_CM, PC_CM, X_E_PER_MASS, HUBBLE_H, TAU_PER_DM, _tau_per_gas_pixel)
  - src/bind/cli/paint_tauplane.py        lines 10-16, 52-56 (_tauslab; same
    formula, applied per lux tau-plane at painting time)
  - papers/01_pipeline/_build_figures_nb.py line 920-923 (fig-6 cell's
    K_TAU, the literal 2.219785e-14 provenance) and line 595 (the printed
    tau equation in the §1d markdown)

KEY FINDING: 2.219785e-14 is NOT a separate hardcoded pipeline constant. It is
the generic formula in _tau_per_gas_pixel(box_size, n_grid, a_l), evaluated at
ONE particular snapshot's geometry (snap 96: box_size=6.25 Mpc/h patch size,
n_grid=128, a_l=0.96738). The pipeline (lightcone_maps.py / paint_tauplane.py)
recomputes this factor per lens plane from six physical constants plus that
plane's own (box_size, n_grid, a_l) -- it is never frozen into a single number.
"""
from __future__ import annotations

import numpy as np

# ── Physical constants exactly as hardcoded in src/bind/inference/lightcone_maps.py:41-49 ──
SIGMA_T = 6.6524e-25            # Thomson cross-section [cm^2]  (CODATA 6.6524587e-25; 5-digit truncation)
M_P = 1.6726e-24                # proton mass [g]                (CODATA 1.67262192e-24; 5-digit truncation)
MSUN_G = 1.989e33               # solar mass [g]
MPC_CM = 3.0857e24              # Mpc -> cm
PC_CM = 3.0857e18               # pc -> cm
HUBBLE_H = 0.6774               # TNG/Planck-2015 h

# ── Composition + ionization ASSUMPTIONS (this is the physics this script pins down) ──
X_H = 0.76      # primordial hydrogen mass fraction (no metals: X_H + Y_He = 1)
Y_HE = 0.24     # primordial helium mass fraction
# Fully ionized: H -> p+ + e- (1 free electron per proton mass unit),
# He -> alpha^2+ + 2e- (2 free electrons per ~4 proton masses, i.e. per Y_He/4
# nuclei each contributing 2 electrons -> Y_He/2 electrons per proton mass unit).
X_E_DERIVED = X_H + Y_HE / 2.0
print(f"x_e derived from X_H={X_H}, Y_He={Y_HE}, H singly + He doubly ionized: "
      f"x_e = X_H + Y_He/2 = {X_E_DERIVED}")
assert np.isclose(X_E_DERIVED, 0.88), "derived x_e does not match the pipeline's hardcoded 0.88"

# Equivalent standard-cosmology form: mean molecular weight per free electron
MU_E = 2.0 / (1.0 + X_H)
print(f"equivalent mu_e = 2/(1+X_H) = {MU_E:.6f}  ->  x_e = 1/mu_e = {1/MU_E:.6f}")
assert np.isclose(1 / MU_E, 0.88, atol=2e-4)

X_E_PER_MASS = X_E_DERIVED / M_P   # free electrons per gram
print(f"X_E_PER_MASS = x_e/m_p = {X_E_PER_MASS:.6e} electrons/gram")

# ── TAU_PER_DM: tau = TAU_PER_DM * DM[pc/cm^3] (src/bind/inference/lightcone_maps.py:49) ──
# DM (dispersion measure) convention: DM[pc cm^-3] = integral n_e dl with dl in pc.
# integral n_e dl[cm] = DM[pc/cm^3] * PC_CM  =>  tau = sigma_T * DM[pc/cm^3] * PC_CM.
TAU_PER_DM = SIGMA_T * PC_CM
print(f"\nTAU_PER_DM = SIGMA_T * PC_CM = {TAU_PER_DM:.6e}  (tau per unit DM[pc/cm^3]; "
      "distinct from the gas-surface-density constant below -- do not confuse the two)")

# ── Fig-6 cell's K_TAU: tau = K_TAU * Sigma_gas[Msun/h per PATCH pixel], snap 96 ──
# Reproduces papers/01_pipeline/_build_figures_nb.py:917-923 exactly, and is
# algebraically the same expression as _tau_per_gas_pixel() in lightcone_maps.py:52-57
# (box_size=6.25 Mpc/h "patch" in place of the full lightcone-plane box_size,
# n_grid=128 in place of the plane's n_grid).
PIX_MPCH = 6.25 / 128.0                  # comoving Mpc/h per patch pixel
Z_SNAP96 = 0.0337243718735154            # bind_science/profiles/fid_snap096.npz "z" (exact)
a_snap = 1.0 / (1.0 + Z_SNAP96)
print(f"\nsnap 96: z = {Z_SNAP96:.10f}  ->  a = {a_snap:.10f}")

# Physical (proper) pixel side length: comoving Mpc/h -> Mpc (divide by h) ->
# physical Mpc (multiply by a_l, since physical = comoving * a) -> cm.
pix_phys_cm = (PIX_MPCH / HUBBLE_H) * a_snap * MPC_CM
print(f"physical pixel side at a={a_snap:.5f}: {pix_phys_cm:.6e} cm "
      f"({pix_phys_cm/MPC_CM*1000:.4f} kpc proper)")

# Msun/h -> g needs /h (removing the h in the mass unit) * MSUN_G; the
# resulting electron column [cm^-2] * SIGMA_T is dimensionless tau. No net h
# dependence survives (one h from the mass, one from the pixel length, and
# they cancel: h_mass^-1 * h_length^-2 combine but pix_phys_cm already had
# its own /h, so the two length "h"s are the SAME h**1 factor -- see below).
K_TAU = SIGMA_T * X_E_PER_MASS * MSUN_G / HUBBLE_H / pix_phys_cm ** 2
print(f"\nK_TAU = SIGMA_T * (x_e/m_p) * MSUN_G / h / pix_phys_cm^2")
print(f"      = {K_TAU:.10e}")
print(f"      = {K_TAU:.6e}   (printed-precision form)")

PRINTED = 2.219785e-14
rel_err = abs(K_TAU - PRINTED) / PRINTED
print(f"\nnotebook-printed constant (papers/01_pipeline/_build_figures_nb.py fig-6 cell "
      f"output; also FIGURE_NUMBERS.md:50): {PRINTED:.6e}")
print(f"relative difference: {rel_err:.3e}")
assert rel_err < 1e-6, "recomputed K_TAU does not match the printed digits"
print("VERDICT: matches to all 6 printed significant figures. PASS.")

# ── h-factor sanity note ──────────────────────────────────────────────────
# pix_phys_cm ∝ (comoving_pix/h); pix_phys_cm**2 ∝ h^-2. The mass term carries
# an explicit extra /h (Msun/h -> Msun). So K_TAU's mass-side /h and the
# 1/pix_phys_cm^2 ∝ h^2 together give a net tau ∝ (1/h) * h^2 = h -- i.e. tau
# computed this way is NOT strictly h-independent per unit of *cached comoving
# gas mass in Msun/h*; rather, gas[Msun/h] * K_TAU is h-independent because
# gas[Msun/h] itself already carries the compensating h^-1 -> the PRODUCT
# tau = K_TAU * gas[Msun/h] is what is h-independent, not K_TAU alone. This
# matches the standard cosmological-simulation convention (masses in Msun/h,
# lengths in Mpc/h) and is not a bug.
print("\n(h-factor note: K_TAU itself is not h-independent; the PRODUCT "
      "K_TAU * gas[Msun/h per pixel] = tau is, since the comoving Msun/h mass "
      "unit already carries the compensating h^-1. See comment above.)")
