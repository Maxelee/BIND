"""Stage 2 — flat-sky lightcone assembly of convergence (kappa) and tSZ (y) maps.

Stacks the per-snapshot BIND composite slabs into angular maps on a **common
grid**, so the WL (kappa) and tSZ (y) maps are pixel-aligned for the 1a
cross-correlation.  This is the self-contained Born-approximation path (no lux
raytracing); lux remains available for high-fidelity kappa validation.

- **kappa** (per source redshift):
  ``kappa(theta) = sum_planes W(chi_l, chi_s) * delta_scaled(theta)`` with
  ``W = (3/2) Omega_m (H0/c)^2 chi_l (chi_s - chi_l) / (chi_s a_l)`` and
  ``delta_scaled = (Sigma/Sigma_bar - 1) * slab_depth`` from the composite mass
  map (matching ``bind.inference.lensplane`` / the diagnostics notebook).
- **y** (tSZ Compton-y): additive along the LOS — ``y(theta) = sum_planes
  y_plane(theta)`` from the composited ``compton_y`` channel.  No lensing kernel.
  ⚠ z>0 thermo a-factors are taken as the model emits them (see CLAUDE.md caveat).
- **tau** (kSZ optical depth / FRB DM): additive along the LOS from the composited
  **gas** mass channel as an electron column, ``tau = sigma_T x_e Sigma_gas/m_p``,
  with the per-plane physical-area factor (scale factor a_l).  ``DM[pc/cm^3] =
  tau / TAU_PER_DM``.  No lensing kernel.  Pixel-aligned with kappa/y, so kappa x
  tau and y x tau cross-spectra are immediate.  For additive LOS probes (y, tau)
  Born projection along the unperturbed ray is exact to first order; ray deflection
  (the lux path) only matters for kappa.

Each plane is periodically tiled to fill the field of view (``fov_deg``) at its
comoving distance and resampled (periodic bilinear) onto ``npix^2``.  A random
per-snapshot transverse shift decorrelates snapshots and, varied by ``seed``,
generates the ``N_real`` stochastic lightcone realizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lensplane import C_KMS, comoving_distance_from_a, mass_map_to_delta_scaled
from .pipeline import circular_taper_weight, paste_halos_2d, square_taper_weight

# Electron-column (kSZ tau / FRB DM) physical constants -------------------------
SIGMA_T = 6.6524e-25            # Thomson cross-section [cm^2]
M_P = 1.6726e-24               # proton mass [g]
MSUN_G = 1.989e33              # solar mass [g]
MPC_CM = 3.0857e24             # Mpc -> cm
PC_CM = 3.0857e18              # pc -> cm
X_E_PER_MASS = 0.88 / M_P      # free electrons per gram (X=0.76, Y_He=0.24, ionized)
HUBBLE_H = 0.6774              # h (manifests carry no hubble; matches tau_profiles/morphology)
TAU_PER_DM = SIGMA_T * PC_CM   # tau = TAU_PER_DM * DM[pc/cm^3];  DM = tau / TAU_PER_DM


def _tau_per_gas_pixel(box_size: float, n_grid: int, a_l: float) -> float:
    """Gas mass [Msun/h] per plane pixel -> kSZ optical depth tau (intensive surface
    density).  Physical pixel area = (comoving pixel * a / h)^2; mass /h to Msun.
    Independent of the angular output pixel size (tau is a sky surface quantity)."""
    pix_phys_cm = (box_size / n_grid) / HUBBLE_H * a_l * MPC_CM       # physical pixel side
    return SIGMA_T * X_E_PER_MASS * MSUN_G / HUBBLE_H / pix_phys_cm ** 2


# ── plane geometry ────────────────────────────────────────────────────────────

def _slab_chi(chi_snap: float, n_slabs: int, sl: int, slab_depth: float) -> float:
    """Comoving distance to slab ``sl`` of a snapshot (matching the notebook)."""
    return chi_snap - (n_slabs / 2 - sl - 0.5) * slab_depth


def _periodic_bilinear(plane: np.ndarray, box_size: float, L_ang: float,
                       npix: int, shift_x: float, shift_y: float) -> np.ndarray:
    """Sample an ``L_ang`` (Mpc/h) FOV from a periodic plane onto ``npix^2``.

    Periodic bilinear interpolation; tiles automatically when ``L_ang`` exceeds
    the box (coordinates wrap).  ``shift`` decorrelates snapshots / realizations.
    """
    N = plane.shape[0]
    u = (np.arange(npix) + 0.5) / npix * L_ang
    cx = ((shift_x + u) % box_size) / box_size * N
    cy = ((shift_y + u) % box_size) / box_size * N
    x0 = np.floor(cx).astype(np.int64)
    fx = cx - x0
    y0 = np.floor(cy).astype(np.int64)
    fy = cy - y0
    x0 %= N
    x1 = (x0 + 1) % N
    y0 %= N
    y1 = (y0 + 1) % N
    p00 = plane[np.ix_(x0, y0)]
    p01 = plane[np.ix_(x0, y1)]
    p10 = plane[np.ix_(x1, y0)]
    p11 = plane[np.ix_(x1, y1)]
    fx = fx[:, None]
    fy = fy[None, :]
    return ((1 - fx) * (1 - fy) * p00 + (1 - fx) * fy * p01
            + fx * (1 - fy) * p10 + fx * fy * p11)


def _thermo_y_from_patches(d, r200_factor: float, taper_frac: float) -> np.ndarray:
    """Fallback compton-y map from saved ``thermo_patches`` (older composites).

    Mirrors :func:`bind.inference.pipeline.build_bind_composite`'s gas-thermo
    blending (``alpha * weighted-paste``) for files written before thermo
    compositing existed.
    """
    thermo = np.asarray(d["thermo_patches"], dtype=np.float32)   # (n, N_THERMO, pp, pp)
    n, _, pp, _ = thermo.shape
    box = float(d["box_size"])
    npix = d["composite"].shape[-1]
    centers, r200 = d["halo_centers"], d["halo_r200"]
    halos = [{"halo_center": centers[i]} for i in range(n)]
    sq = square_taper_weight(pp, taper_frac=taper_frac)
    if r200_factor > 0:
        ppm = npix / box
        wl = [circular_taper_weight(pp, r_pix=float(r200[i]) * ppm * r200_factor,
                                    taper_frac=taper_frac) for i in range(n)]
        canvas, wacc = paste_halos_2d(npix, box, halos, thermo, sq, weights_list=wl)
    else:
        canvas, wacc = paste_halos_2d(npix, box, halos, thermo, sq)
    alpha = np.clip(wacc, 0.0, 1.0)
    return (alpha * canvas[0]).astype(np.float64)   # compton_y channel only


# ── main assembler ────────────────────────────────────────────────────────────

def assemble_lightcone(
    snap_root: Path | str,
    snapshots,
    *,
    field: str = "bind",
    mass_key: str = "composite",
    source_redshifts=(1.0,),
    fov_deg: float = 5.0,
    npix: int = 1024,
    Omega_m: float = 0.3089,
    want_y: bool = True,
    want_tau: bool = True,
    r200_factor: float = 4.0,
    taper_frac: float = 0.15,
    seed: int = 0,
    manifest_root: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Assemble one lightcone realization of kappa (per source z) and y.

    Parameters
    ----------
    snap_root : root holding ``snap_<NNN>/{stage1/stage1_manifest.json,
        composite_slab*.npz}`` (a per-run dir, or the shared lightcone tree).
    field : ``"bind"`` (composite total matter) or ``"dmo"`` (DMO background) for
        the lensing source — use ``"dmo"`` to get the same-pipeline DMO kappa for
        the artifact-cancelling ratio.
    source_redshifts : tuple of source-plane redshifts (tomographic bins).
    seed : varies the per-snapshot transverse shift -> a distinct realization.
    manifest_root : where to find ``snap_<NNN>/stage1/stage1_manifest.json`` when
        it lives apart from the composites (e.g. per-run science dirs read the
        composites from ``snap_root`` but the shared manifests from here).
        Defaults to ``snap_root``.

    Returns dict: ``kappa`` (n_src, npix, npix), ``y`` (n_src, npix, npix) or None
    (tomographic: cumulative Compton-y to each source plane), ``tau`` (n_src, npix,
    npix) or None (cumulative kSZ optical depth / FRB electron column; only for
    ``field="bind"``), ``source_redshifts``, ``fov_deg``, ``npix``, ``n_planes``,
    ``field``, ``seed``.
    """
    snap_root = Path(snap_root)
    man_root = Path(manifest_root) if manifest_root is not None else snap_root
    zs = np.atleast_1d(np.asarray(source_redshifts, dtype=np.float64))
    chi_s = np.array([comoving_distance_from_a(1.0 / (1.0 + z), Omega_m) for z in zs])
    fov = np.deg2rad(fov_deg)
    prefac = 1.5 * Omega_m * (100.0 / C_KMS) ** 2
    rng = np.random.default_rng(seed)

    kappa = np.zeros((len(zs), npix, npix), dtype=np.float64)
    ymap = np.zeros((len(zs), npix, npix), dtype=np.float64) if want_y else None
    taumap = np.zeros((len(zs), npix, npix), dtype=np.float64) if want_tau else None
    n_planes = 0

    for s in snapshots:
        snap_dir = snap_root / f"snap_{s:03d}"
        man_path = man_root / f"snap_{s:03d}" / "stage1" / "stage1_manifest.json"
        if not man_path.exists():
            if verbose:
                print(f"[lightcone] skip snap {s}: no manifest")
            continue
        man = json.loads(man_path.read_text())
        a_l = float(man["scale_factor"])
        box = float(man["box_size"])
        n_slabs = int(man["n_slabs"])
        slab_depth = float(man["slab_depth"])
        Om = float(man.get("Omega_m", Omega_m))
        chi_snap = comoving_distance_from_a(a_l, Om)
        shift = rng.uniform(0.0, box, size=2)   # decorrelation / realization

        for sl in range(n_slabs):
            f2 = snap_dir / f"composite_slab{sl:02d}.npz"
            if not f2.exists():
                continue
            chi_l = _slab_chi(chi_snap, n_slabs, sl, slab_depth)
            if chi_l <= 0:
                continue
            L_ang = fov * chi_l            # comoving FOV size at this plane [Mpc/h]
            d = np.load(f2)

            mass = (d[mass_key].sum(0) if field == "bind"
                    else d["dmo"]).astype(np.float64)
            delta = mass_map_to_delta_scaled(mass, box_size=box,
                                             slab_depth=slab_depth, Omega_m=Om)
            plane_k = _periodic_bilinear(delta, box, L_ang, npix, *shift)
            for j in range(len(zs)):
                if chi_l >= chi_s[j]:
                    continue
                W = prefac * chi_l * (chi_s[j] - chi_l) / (chi_s[j] * a_l)
                kappa[j] += W * plane_k

            if want_y:
                if "composite_thermo" in d.files:
                    yslab = d["composite_thermo"][0].astype(np.float64)
                elif "thermo_patches" in d.files:
                    yslab = _thermo_y_from_patches(d, r200_factor, taper_frac)
                else:
                    yslab = None
                if yslab is not None:
                    yplane = _periodic_bilinear(yslab, box, L_ang, npix, *shift)
                    # tomographic: cumulative y to each source plane (additive, no kernel)
                    for j in range(len(zs)):
                        if chi_l < chi_s[j]:
                            ymap[j] += yplane

            if want_tau and field == "bind":
                # kSZ optical depth / FRB DM: electron column of the composited gas
                # channel, additive along the LOS with the per-plane a_l area factor.
                gas = d["composite"][1].astype(np.float64)
                tau_slab = _tau_per_gas_pixel(box, gas.shape[0], a_l) * gas
                tplane = _periodic_bilinear(tau_slab, box, L_ang, npix, *shift)
                for j in range(len(zs)):
                    if chi_l < chi_s[j]:
                        taumap[j] += tplane
            n_planes += 1
        if verbose:
            print(f"[lightcone] snap {s} (z={1/a_l-1:.2f}) done; planes so far {n_planes}")

    return {
        "kappa": kappa, "y": ymap, "tau": taumap,
        "source_redshifts": zs, "fov_deg": fov_deg, "npix": npix,
        "n_planes": n_planes, "field": field, "seed": seed,
    }
