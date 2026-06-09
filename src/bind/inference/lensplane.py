"""Convert BIND composite mass maps to lux-compatible lensplane files.

Pipeline per snapshot:

  1. ``mass_map_to_delta_scaled`` — projected mass map (Msun/h/pixel) →
     lux density field  ``(Σ/Σ̄ − 1) × slab_depth``  [Mpc/h].

  2. ``density_to_lensplane`` — 2D Poisson FFT matching lux's ``Fourier_trs()``:
     solves  ∇²ψ = 3 Ωₘ (H₀/c)² δ_scaled / k²  and returns the 5 partial
     derivatives of ψ packed as a ``(N, N, 5)`` array.

  3. ``write_lensplane`` — writes ``lenspot{PP:02d}.dat`` in the lux binary
     format: ``int32(N) | float64[N×N×5] | int32(N)``.

  4. ``write_lux_config`` — writes ``config.dat`` so lux raytracing knows the
     comoving geometry, box sizes, and per-snapshot transforms.

Conventions match lux exactly so that lensplane files written here are
drop-in replacements for those lux would write itself.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ── Physical constants (matching lux source) ──────────────────────────────────
C_KMS = 299792.458          # speed of light [km/s]
# Critical density today in lux units: [10^10 Msun/h / (Mpc/h)^3]
# multiply by 1e10 to get [Msun/h / (Mpc/h)^3]
RHO_C0 = 27.7536627e10     # [Msun/h / (Mpc/h)^3]


# ── Step 1: mass map → lux density field ──────────────────────────────────────

def mass_map_to_delta_scaled(
    mass_map: np.ndarray,
    *,
    box_size: float,
    slab_depth: float,
    Omega_m: float,
) -> np.ndarray:
    """Convert a projected mass map to the lux-normalised density field.

    Matches lux lenspot.cpp (lines 362-368)::

        bar = rho_c0 * Omega_m * Lt^2 * Ll_slab / LPmesh^2
        delta = mass / bar - 1.0
        delta_scaled = delta * Ll_slab

    Parameters
    ----------
    mass_map : (N, N) float array
        CIC-projected mass per pixel in **Msun/h**.
    box_size : float
        Transverse box size ``Lt`` in Mpc/h.  For non-stacked snapshots
        this equals the full box side length ``L``.
    slab_depth : float
        LOS depth of this slab ``Ll/pps`` in Mpc/h.
    Omega_m : float
        Matter density parameter (from snapshot Header/Omega0).

    Returns
    -------
    delta_scaled : (N, N) float64 array  [Mpc/h]
    """
    N = mass_map.shape[0]
    pixel_area = (box_size / N) ** 2           # (Mpc/h)^2
    bar = RHO_C0 * Omega_m * pixel_area * slab_depth  # Msun/h per pixel
    delta_scaled = (mass_map.astype(np.float64) / bar - 1.0) * slab_depth
    return delta_scaled


# ── Step 2: density field → 5 lensing potential derivatives ──────────────────

def density_to_lensplane(
    delta_scaled: np.ndarray,
    *,
    box_size: float,
    Omega_m: float,
) -> np.ndarray:
    """Solve the 2-D Poisson equation to get lensing potential derivatives.

    Matches lux ``Fourier_trs()`` (fourier.cpp) exactly.

    The 5 output fields (``phi[:,:,f]``) are:

    =========  =================  ==========
    f          field              lux case
    =========  =================  ==========
    0          ∂ψ/∂x             deflection x
    1          ∂ψ/∂y             deflection y
    2          ∂²ψ/∂x²           hessian xx
    3          ∂²ψ/∂x∂y          hessian xy
    4          ∂²ψ/∂y²           hessian yy
    =========  =================  ==========

    Kernel (in Fourier space, wave number **k** in [Mpc/h]⁻¹)::

        f=0: phi_k = −i kₓ / k² × prefactor × δ_k
        f=1: phi_k = −i k_y / k² × prefactor × δ_k
        f=2: phi_k =    kₓ² / k² × prefactor × δ_k
        f=3: phi_k =  kₓk_y / k² × prefactor × δ_k
        f=4: phi_k =   k_y² / k² × prefactor × δ_k

        prefactor = 3 Ωₘ (100/c)²   [1/(Mpc/h)²]

    Note that lux uses unnormalised IFFT (FFTW_BACKWARD) and absorbs the
    1/N² normalisation into the kernel.  Using numpy's normalised ``ifft2``
    we drop that 1/N² and the results are identical.

    Parameters
    ----------
    delta_scaled : (N, N) float array  [Mpc/h]
        Output of :func:`mass_map_to_delta_scaled`.
    box_size : float
        Transverse box size ``Lt`` in Mpc/h.
    Omega_m : float

    Returns
    -------
    phi : (N, N, 5) float64 array
        Lensing potential derivatives in lux storage order (f-index last,
        varying fastest — i.e. C-order last axis).
    """
    N = delta_scaled.shape[0]
    ku = 2.0 * np.pi / box_size             # fundamental mode [1/Mpc/h]
    prefactor = 3.0 * Omega_m * (100.0 / C_KMS) ** 2   # [1/(Mpc/h)^2]

    # Wave number grids — lux: I = i if i<=N/2 else i-N  (same as fftfreq)
    freq = np.fft.fftfreq(N, d=1.0 / N).astype(np.int64)
    kx, ky = np.meshgrid(ku * freq, ku * freq, indexing="ij")  # (N,N)

    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0   # avoid /0; DC term is zeroed out after division

    delta_k = np.fft.fft2(delta_scaled.astype(np.float64))

    phi = np.empty((N, N, 5), dtype=np.float64)

    # f=0: −i kₓ / k²
    _phi_k = -1j * kx / k2 * prefactor * delta_k
    _phi_k[0, 0] = 0.0
    phi[:, :, 0] = np.fft.ifft2(_phi_k).real

    # f=1: −i k_y / k²
    _phi_k = -1j * ky / k2 * prefactor * delta_k
    _phi_k[0, 0] = 0.0
    phi[:, :, 1] = np.fft.ifft2(_phi_k).real

    # f=2: kₓ² / k²
    _phi_k = kx ** 2 / k2 * prefactor * delta_k
    _phi_k[0, 0] = 0.0
    phi[:, :, 2] = np.fft.ifft2(_phi_k).real

    # f=3: kₓ k_y / k²
    _phi_k = kx * ky / k2 * prefactor * delta_k
    _phi_k[0, 0] = 0.0
    phi[:, :, 3] = np.fft.ifft2(_phi_k).real

    # f=4: k_y² / k²
    _phi_k = ky ** 2 / k2 * prefactor * delta_k
    _phi_k[0, 0] = 0.0
    phi[:, :, 4] = np.fft.ifft2(_phi_k).real

    return phi


# ── Step 3: write lensplane binary ────────────────────────────────────────────

def write_lensplane(phi: np.ndarray, path: str | Path) -> None:
    """Write a lux ``lenspot{PP:02d}.dat`` binary file.

    Binary layout (little-endian, matching lux's write_phi())::

        int32   N          (grid dimension)
        float64 phi[N,N,5] (C-order; f-index last/fastest — lux: phi[f+5*(j+N*i)])
        int32   N

    Parameters
    ----------
    phi : (N, N, 5) float64 array
    path : output file path
    """
    N = phi.shape[0]
    if phi.shape != (N, N, 5):
        raise ValueError(f"phi must have shape (N, N, 5), got {phi.shape}")
    with open(path, "wb") as fp:
        fp.write(struct.pack("<i", N))
        fp.write(np.asarray(phi, dtype=np.float64).tobytes())  # C-order
        fp.write(struct.pack("<i", N))


def read_lensplane(path: str | Path) -> np.ndarray:
    """Read a lux lensplane file back into a ``(N, N, 5)`` array."""
    with open(path, "rb") as fp:
        (N,) = struct.unpack("<i", fp.read(4))
        phi = np.frombuffer(fp.read(N * N * 5 * 8), dtype=np.float64).reshape(N, N, 5)
        fp.read(4)  # trailing int32
    return phi.copy()


# ── Step 4: write lux config.dat ──────────────────────────────────────────────

def comoving_distance_from_a(
    a: float | np.ndarray,
    Omega_m: float,
    Omega_L: float | None = None,
    n_steps: int = 10_000,
) -> float | np.ndarray:
    """Compute comoving distance χ(a) by numerical integration.

    Flat ΛCDM: ``E(a) = sqrt(Ωₘ a⁻³ + Ω_Λ)``.

    Returns chi in Mpc/h (same units as box sizes).
    """
    if Omega_L is None:
        Omega_L = 1.0 - Omega_m

    scalar = np.ndim(a) == 0
    a_arr = np.atleast_1d(np.asarray(a, dtype=np.float64))
    chi_arr = np.zeros_like(a_arr)

    for idx, a_end in enumerate(a_arr):
        if a_end >= 1.0:
            chi_arr[idx] = 0.0
            continue
        a_int = np.linspace(a_end, 1.0, n_steps + 1)
        E_a = np.sqrt(Omega_m * a_int ** (-3) + Omega_L)
        integrand = (C_KMS / 100.0) / (a_int ** 2 * E_a)
        chi_arr[idx] = np.trapz(integrand, a_int)

    return float(chi_arr[0]) if scalar else chi_arr


def build_lightcone_geometry(
    *,
    snapshot_scale_factors: list[float],
    Ll: list[float],
    Lt: list[float],
    planes_per_snapshot: int,
    Omega_m: float,
    Omega_L: float | None = None,
) -> dict:
    """Compute chi, a, chi_out arrays for lux config.dat.

    Snapshots must be ordered from **low-z to high-z** (closest first),
    matching lux's stacking convention where each box is placed end-to-end
    in comoving distance starting from the observer.

    The ``chi`` and ``a`` values in the returned dict are computed from the
    cumulative box depth (as lux does), NOT from the actual snapshot
    cosmological coordinates.  This is the standard approximation used in
    shell-stacking lightcone codes.

    Parameters
    ----------
    snapshot_scale_factors : list[float]
        Scale factor ``a = 1/(1+z)`` for each snapshot (low-z first).
    Ll : list[float]
        LOS box size per snapshot [Mpc/h].
    Lt : list[float]
        Transverse box size per snapshot [Mpc/h].
    planes_per_snapshot : int
        Number of lensplanes per snapshot (``pps`` in lux).
    Omega_m, Omega_L : float
        Cosmological parameters.

    Returns
    -------
    dict with keys ``chi`` (Np+1,), ``a`` (Np+1,), ``chi_out`` (Np,),
    where ``Np = Ns * pps``.
    """
    Ns = len(snapshot_scale_factors)
    pps = planes_per_snapshot
    Np = Ns * pps

    # Cumulative LOS distances (matching lux chi[] computation)
    chi = np.zeros(Np + 1)
    for p in range(1, Np + 1):
        s = (p - 1) // pps
        for ss in range(s):
            chi[p] += Ll[ss]
        chi[p] += Ll[s] / pps / 2.0 + Ll[s] / pps * ((p - 1) % pps)

    chi_out = np.zeros(Np)
    for p in range(Np):
        s = p // pps
        for ss in range(s):
            chi_out[p] += Ll[ss]
        chi_out[p] += Ll[s] / pps * (p % pps + 1)

    # Scale factors at each plane (invert chi → a numerically)
    a_planes = np.zeros(Np + 1)
    a_planes[0] = 1.0  # observer at z=0
    for i in range(1, Np + 1):
        a_planes[i] = _find_scale_factor(chi[i], Omega_m, Omega_L or 1.0 - Omega_m)

    return {
        "chi": chi,
        "a": a_planes,
        "chi_out": chi_out,
    }


def _find_scale_factor(
    chi_target: float,
    Omega_m: float,
    Omega_L: float,
    tol: float = 1e-10,
) -> float:
    """Find scale factor a such that chi(a) = chi_target (Newton method, matching lux)."""
    if chi_target <= 0.0:
        return 1.0
    a = 0.5  # initial guess
    for _ in range(200):
        E_a = np.sqrt(Omega_m * a ** (-3) + Omega_L)
        chi_a = comoving_distance_from_a(a, Omega_m, Omega_L)
        dchi_da = -(C_KMS / 100.0) / (a ** 2 * E_a)  # dchi/da < 0
        a_new = a - (chi_a - chi_target) / dchi_da
        a_new = max(0.01, min(1.0, a_new))
        if abs(a - a_new) < tol:
            return a_new
        a = a_new
    return a


def write_lux_config(
    path: str | Path,
    *,
    chi: np.ndarray,
    a: np.ndarray,
    chi_out: np.ndarray,
    Ll: np.ndarray,
    Lt: np.ndarray,
    proj_dirs: np.ndarray,
    disp: np.ndarray,
    flip: np.ndarray,
) -> None:
    """Write lux ``config.dat`` binary file.

    Matches lux ``write_config()`` in ``lenspot.cpp`` (lines 471-496).

    The ``disp`` and ``flip`` arrays are indexed ``[snapshot, original_axis]``
    (shape ``(Ns, 3)``) and are written in C-order, matching lux's
    ``disp[j + 3*s]`` / ``flip[j + 3*s]`` flat layout.

    Parameters
    ----------
    path : output file path
    chi  : (Np+1,) float64 — comoving distances to plane edges [Mpc/h]
    a    : (Np+1,) float64 — scale factors at plane edges
    chi_out : (Np,) float64 — comoving distances to plane centres
    Ll   : (Ns,) float64 — LOS box sizes [Mpc/h]
    Lt   : (Ns,) float64 — transverse box sizes [Mpc/h]
    proj_dirs : (Ns,) int32
    disp : (Ns, 3) float64 — displacement [Mpc/h] per original axis
    flip : (Ns, 3) bool
    """
    Np = len(chi) - 1
    Ns = len(Ll)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<i", Np))
        fp.write(struct.pack("<i", Ns))
        fp.write(np.asarray(a,       dtype=np.float64).tobytes())   # (Np+1)
        fp.write(np.asarray(chi,     dtype=np.float64).tobytes())   # (Np+1)
        fp.write(np.asarray(chi_out, dtype=np.float64).tobytes())   # (Np)
        fp.write(np.asarray(Ll,      dtype=np.float64).tobytes())   # (Ns)
        fp.write(np.asarray(Lt,      dtype=np.float64).tobytes())   # (Ns)
        fp.write(np.asarray(proj_dirs, dtype=np.int32).tobytes())   # (Ns)
        fp.write(np.asarray(disp,    dtype=np.float64).tobytes())   # (Ns,3) C-order
        fp.write(np.asarray(flip,    dtype=np.uint8).tobytes())     # (Ns,3) 1 byte/bool
