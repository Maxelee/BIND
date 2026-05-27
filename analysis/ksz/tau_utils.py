"""Optical-depth (τ) utilities for the kSZ analysis.

The BIND Gas channel is the projected gas-mass surface density on a
``patch_pix × patch_pix`` map of physical side ``patch_size_mpc_h`` (in Mpc/h),
with per-pixel value = total gas mass [Msun/h] in that pixel.  For the
validation plots we collapse each patch to a per-halo aperture-integrated
electron column density τ_ap using the standard fully-ionised primordial
conversion

    τ_ap = (σ_T X_H / m_p) · ⟨Σ_gas⟩_ap

with ⟨Σ_gas⟩_ap the gas-mass surface density averaged inside the aperture.

Approximations (acceptable for validation plot A):
  * Fully-ionised hydrogen, x_e ≈ 1, neglecting the (1 + X_H) helium term.
    A more careful conversion uses μ_e ≈ 1.14 m_p (so n_e = ρ_gas / (μ_e m_p)).
    Switchable via ``electron_per_proton`` below; default 1.17 is closer to
    primordial + first-ionisation-helium.
  * No temperature dependence (τ is linear in n_e by definition).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI unless noted)
# ---------------------------------------------------------------------------
SIGMA_T_M2 = 6.6524587e-29      # Thomson cross-section [m^2]
M_PROTON_KG = 1.6726219e-27     # proton mass [kg]
MSUN_KG = 1.989e30              # Msun [kg]
MPC_IN_M = 3.085677581e22       # 1 Mpc in metres
X_H = 0.76                      # primordial hydrogen mass fraction

# Default electron-per-proton ratio for fully ionised primordial plasma
# (H + He, single ionisation):  n_e / n_p ≈ (X_H + Y_He/2) / X_H ≈ 1.17
ELECTRON_PER_PROTON = 1.17


def gas_surface_density_to_tau(
    sigma_gas_msun_h_per_mpc_h2: np.ndarray,
    hubble: float = 0.6711,
    electron_per_proton: float = ELECTRON_PER_PROTON,
) -> np.ndarray:
    """Convert gas-mass surface density [Msun/h per (Mpc/h)^2] to τ (unitless).

    τ = σ_T · n_e,col   with   n_e,col = (X_H · electron_per_proton / m_p) · Σ_gas.
    Both Σ_gas (Msun/h per (Mpc/h)^2) and the conversion are unit-corrected
    using ``hubble`` (h70 → physical Msun/Mpc^2 via 1/h factors).

    Mass units:  M[Msun] = M[Msun/h] / h
    Area units:  A[Mpc^2] = A[(Mpc/h)^2] / h^2  ⇒  Σ[Msun/Mpc^2] = Σ[Msun/h /(Mpc/h)^2] · h
    """
    sigma_msun_per_mpc2 = sigma_gas_msun_h_per_mpc_h2 * hubble
    sigma_kg_per_m2 = sigma_msun_per_mpc2 * MSUN_KG / MPC_IN_M**2
    n_e_col = X_H * electron_per_proton * sigma_kg_per_m2 / M_PROTON_KG  # 1/m^2
    return SIGMA_T_M2 * n_e_col


def gas_mass_to_tau_in_aperture(
    gas_mass_msun_h_in_ap: np.ndarray,
    aperture_area_mpc_h2: float,
    hubble: float = 0.6711,
    electron_per_proton: float = ELECTRON_PER_PROTON,
) -> np.ndarray:
    """Aperture-averaged τ from total gas mass [Msun/h] inside that aperture."""
    sigma = gas_mass_msun_h_in_ap / aperture_area_mpc_h2
    return gas_surface_density_to_tau(
        sigma, hubble=hubble, electron_per_proton=electron_per_proton
    )


# ---------------------------------------------------------------------------
# Aperture geometry on patch maps
# ---------------------------------------------------------------------------
def _radius_pixels(patch_pix: int) -> np.ndarray:
    """Pixel distance from the patch centre (centre = (patch_pix-1)/2)."""
    c = (patch_pix - 1) / 2.0
    y, x = np.indices((patch_pix, patch_pix))
    return np.sqrt((x - c) ** 2 + (y - c) ** 2)


def disk_aperture_mask(patch_pix: int, r_pix: float) -> np.ndarray:
    """Boolean disk of radius r_pix [pixels], centred on the patch."""
    return _radius_pixels(patch_pix) <= float(r_pix)


def compensated_aperture_weights(
    patch_pix: int, r_inner_pix: float, sqrt2: bool = True
) -> np.ndarray:
    """ACT-DR6-style compensated aperture photometry (CAP) weights.

    Inner disk of radius r_inner_pix has weight +1; surrounding annulus out
    to r_outer = sqrt(2) * r_inner has weight −(A_inner / A_annulus).  Summing
    the patch against these weights yields the background-subtracted signal in
    the disk.  Areas balance exactly (∑ w = 0).
    """
    r = _radius_pixels(patch_pix)
    r_outer = (np.sqrt(2.0) if sqrt2 else 2.0) * r_inner_pix
    inner = r <= r_inner_pix
    annulus = (r > r_inner_pix) & (r <= r_outer)
    n_in = inner.sum()
    n_ann = annulus.sum()
    w = np.zeros_like(r, dtype=np.float64)
    if n_in == 0 or n_ann == 0:
        return w
    w[inner] = 1.0
    w[annulus] = -float(n_in) / float(n_ann)
    return w


# ---------------------------------------------------------------------------
# Per-halo aperture integrals
# ---------------------------------------------------------------------------
def aperture_gas_mass(
    gas_patches_msun_h: np.ndarray,  # (N, patch_pix, patch_pix), pixel-sum = mass
    r_pix: float,
) -> np.ndarray:
    """Sum gas mass inside a centred disk of radius r_pix for each patch.

    Patches are assumed to store *mass per pixel* (the convention produced by
    ``pixelize_z_projection``), so a sum is the total mass inside the aperture.
    """
    if gas_patches_msun_h.ndim != 3:
        raise ValueError(f"expected (N, P, P), got {gas_patches_msun_h.shape}")
    patch_pix = gas_patches_msun_h.shape[-1]
    mask = disk_aperture_mask(patch_pix, r_pix)
    return gas_patches_msun_h[:, mask].sum(axis=1).astype(np.float64)


def aperture_cap_signal(
    patches: np.ndarray,            # (N, patch_pix, patch_pix)
    r_inner_pix: float,
    sqrt2: bool = True,
) -> np.ndarray:
    """Background-subtracted aperture sum (CAP filter) per patch."""
    if patches.ndim != 3:
        raise ValueError(f"expected (N, P, P), got {patches.shape}")
    patch_pix = patches.shape[-1]
    w = compensated_aperture_weights(patch_pix, r_inner_pix, sqrt2=sqrt2)
    return (patches * w).sum(axis=(1, 2)).astype(np.float64)


def r200_to_pixels(r200_mpc_h: np.ndarray, patch_size_mpc_h: float, patch_pix: int) -> np.ndarray:
    """Convert R_200c [Mpc/h] to a pixel radius on a patch of side patch_size_mpc_h."""
    return np.asarray(r200_mpc_h, dtype=np.float64) * (patch_pix / float(patch_size_mpc_h))


# ---------------------------------------------------------------------------
# Single source of truth for the per-halo aperture τ (used by A/C/D/E/F)
# ---------------------------------------------------------------------------
def per_halo_tau(
    patches: np.ndarray,            # (N, P, P) gas mass per pixel [Msun/h]
    r_ap_pix,                       # scalar or (N,) aperture radius in pixels
    pix_size_mpc_h: float,
    hubble: float = 0.6711,
    *,
    estimator: str = "cap",
    electron_per_proton: float = ELECTRON_PER_PROTON,
    sqrt2: bool = True,
) -> np.ndarray:
    """Aperture-averaged τ per halo, with a single shared definition.

    ``estimator``:
      * ``"disk"`` — mean Σ_gas in a disk of radius ``r_ap_pix`` converted to τ.
        This integrates the *entire* line-of-sight column inside the patch
        (the BIND/truth projection depth), with **no background subtraction**.
      * ``"cap"`` — ACT-DR6-style compensated aperture: τ ∝ Σ_disk − Σ_annulus
        with the annulus out to √2·r_ap_pix.  A spatially uniform line-of-sight
        background cancels exactly (∑w = 0), so this is the kSZ-canonical
        estimator and the recommended default (see ``docs/paper2_ksz_plan.md``
        §6.1).  CAP τ can be negative for low-S/N halos.

    ``r_ap_pix`` may be a scalar (one mask, fast path) or a per-halo array
    (looped — cost is negligible vs downstream inference).  For ``"cap"`` it is
    the inner-disk radius; for ``"disk"`` it is the disk radius.
    """
    if patches.ndim != 3:
        raise ValueError(f"expected (N, P, P), got {patches.shape}")
    if estimator not in ("disk", "cap"):
        raise ValueError(f"unknown estimator {estimator!r} (use 'disk' or 'cap')")
    n = patches.shape[0]
    r_arr = np.atleast_1d(np.asarray(r_ap_pix, dtype=np.float64))
    scalar_r = r_arr.size == 1
    pix_area = pix_size_mpc_h ** 2

    tau = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = float(r_arr[0] if scalar_r else r_arr[i])
        disk_area_mpc_h2 = np.pi * (r * pix_size_mpc_h) ** 2
        if disk_area_mpc_h2 <= 0:
            tau[i] = np.nan
            continue
        if estimator == "disk":
            mass = aperture_gas_mass(patches[i:i + 1], r)[0]
            sigma = mass / disk_area_mpc_h2
        else:  # cap: ∑w·M / A_disk = Σ_disk − Σ_annulus
            cap_mass = aperture_cap_signal(patches[i:i + 1], r, sqrt2=sqrt2)[0]
            sigma = cap_mass / disk_area_mpc_h2
        tau[i] = gas_surface_density_to_tau(
            np.asarray(sigma), hubble=hubble, electron_per_proton=electron_per_proton
        )
    return tau
