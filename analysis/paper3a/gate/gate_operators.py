"""Per-(run, snapshot) operator evaluation for the A3 prior-support gate.

For one painted design point (a run dir of composite_slab*.npz files, SB35
Sobol / twobound 1P / fiducial — identical formats) at one snapshot, compute:

- stacked kSZ CAP T_kSZ profiles in the data-relevant M200c bins, measured
  on the pasted gas composite (the WP-A2 patch-size finding: CAP annuli do
  not fit in bare core patches at LRG redshifts);
- per-halo cylindrical f_gas(<R500c-proj) and Y(<R500c-proj) from the bare
  patches (R500c apertures fit comfortably inside the 6.25 h^-1 Mpc core),
  plus the fitted Y-M relation.

Outputs one compact npz per (run, snap) with everything the A3 overlay
needs, so the Slurm array is embarrassingly parallel and resumable.

Conventions and their caveats (gate-level, to be superseded by the Popeye
truth validation):
- R500c is approximated from the catalog R200c via a fixed NFW concentration
  (see `r500c_from_r200c`); the ~few-% aperture error this makes is far
  below the SB35 envelope width the gate cares about. Flagged in the output.
- f_gas denominator M500c is approximated by the same NFW rescaling of the
  catalog M200c.
- Bare patches are used un-rescaled (no DMO mass matching) for the per-halo
  observables; the composite path uses the standard Paper-2 conventions
  (mass match + circular taper). The truth validation quantifies both.
- f_gas / Y here are *cylindrical*; comparison to spherical data values goes
  through the truth-calibrated CylToSphCorrection at A3 aggregation time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.paper3a.observables import (
    FlatLCDM,
    KSZOperatorConfig,
    PatchGeometry,
    fgas_cylindrical,
    ksz_cap_profile,
    stack_profiles,
    y_aperture_mpc2,
)
from analysis.paper3a.observables.constants import (
    RHO_CRIT0_MSUNH_PER_MPCH3,
    TNG_OMEGA_B,
)

# ACT DR6 hILC effective beam, confirmed against all four kSZ papers
# (WP-A1 freeze doc section 6a, audit 2026-07-16).
HILC_BEAM_FWHM_ARCMIN = 1.6


def cosmic_mean_gas_per_pixel(slab_depth_hmpc: float, pixel_mpch: float,
                              omega_b: float = TNG_OMEGA_B) -> float:
    """Cosmic-mean gas surface mass per pixel [Msun/h] for a slab projection.

    Sigma_bar = Omega_b * rho_crit * depth * pix_area (comoving h-units;
    diffuse baryons outside halos are essentially all gas, so no stellar
    correction). This is the background the composite gas map blends in
    where alpha < 1 — the 2026-07-17 dry-run finding: without it the CAP
    annulus compensates against zeros between pasted apertures and the
    filter's background rejection is broken.
    """
    return omega_b * RHO_CRIT0_MSUNH_PER_MPCH3 * slab_depth_hmpc * pixel_mpch**2


def blend_gas_background(gas_canvas: np.ndarray, alpha: np.ndarray, sigma_bar_pix: float) -> np.ndarray:
    """gas = alpha * painted + (1 - alpha) * cosmic-mean background.

    A painted map that equals the background everywhere comes out exactly
    uniform regardless of alpha, so CAP filters it to zero — restoring the
    compensation property the kSZ operator relies on.
    """
    return alpha * gas_canvas + (1.0 - alpha) * sigma_bar_pix

# Mass bins for the kSZ stacks: the two above-floor Qu et al. 2026 quartile
# ticks (log10 M200c = 13.43, 14.57) get a bin each.
DEFAULT_KSZ_MASS_BINS = ((13.2, 13.7), (13.7, 15.5))


def r500c_from_r200c(r200c: np.ndarray, c200: float = 5.0) -> np.ndarray:
    """R500c from R200c for an NFW halo with concentration c200.

    Solves m(x)/x^3 = 2.5 * m(c)/c^3 with m(x) = ln(1+x) - x/(1+x) on a
    lookup grid. At c200 = 5, R500c/R200c ~= 0.66; the ratio moves by only
    ~2% over c200 in [4, 7], which is why a fixed concentration is adequate
    for gate-level apertures (see module docstring).
    """
    x = np.linspace(0.05, 1.0, 2000)
    m = lambda y: np.log(1.0 + y) - y / (1.0 + y)
    lhs = m(c200 * x) / x**3
    ratio = x[np.argmin(np.abs(lhs - 2.5 * m(c200)))]
    return np.asarray(r200c, dtype=float) * ratio


def m500c_from_m200c(m200c: np.ndarray, c200: float = 5.0) -> np.ndarray:
    """M500c = M200c * (2.5 * (R500/R200)^3) by the overdensity definitions."""
    ratio = r500c_from_r200c(np.array([1.0]), c200)[0]
    return np.asarray(m200c, dtype=float) * 2.5 * ratio**3


@dataclass(frozen=True)
class GateConfig:
    radii_arcmin: np.ndarray            # CAP radii, from the A1 loader (Qu grid)
    ksz_mass_bins: tuple = DEFAULT_KSZ_MASS_BINS
    v_rms_over_c: float | None = None   # None -> tau_CAP output (gate default)
    cutout_pix: int = 361
    max_halos_per_bin: int = 200        # runtime control for the CAP stacks
    c200_for_apertures: float = 5.0
    cosmology: FlatLCDM = field(default_factory=FlatLCDM)


def _load_slabs(run_dir: Path, snap: str) -> list[dict]:
    files = sorted((run_dir / f"snap_{snap}").glob("composite_slab*.npz"))
    if not files:
        raise FileNotFoundError(f"no composite_slab*.npz under {run_dir}/snap_{snap}")
    return [dict(np.load(f)) for f in files]


def process_run_snapshot(
    run_dir: str | Path,
    snap: str,
    z_snap: float,
    out_path: str | Path,
    config: GateConfig,
    box_size: float = 205.0,
    canvas_pix: int = 4198,
    patch_pix: int = 128,
) -> dict:
    """Compute all gate observables for one painted run at one snapshot.

    Writes ``out_path`` (npz) and returns the dict. Skips nothing itself —
    resumability (skip existing outputs) belongs to the driver script.
    """
    # local import: keeps the observables package torch-free for Popeye reuse
    from bind.inference.pipeline import (
        circular_taper_weight,
        extract_periodic_cutout,
        paste_halos_2d,
        square_taper_weight,
    )

    run_dir = Path(run_dir)
    slabs = _load_slabs(run_dir, snap)
    geom = PatchGeometry(pixel_mpch=box_size / canvas_pix, z=z_snap, cosmology=config.cosmology)
    patch_geom = PatchGeometry(pixel_mpch=6.25 / patch_pix, z=z_snap, cosmology=config.cosmology)
    ksz_cfg = KSZOperatorConfig(
        radii_arcmin=config.radii_arcmin,
        beam_fwhm_arcmin=HILC_BEAM_FWHM_ARCMIN,
        z_eff=z_snap,
        v_rms_over_c=config.v_rms_over_c,
    )
    ppm = canvas_pix / box_size
    sq = square_taper_weight(patch_pix, taper_frac=0.15)

    # per-halo observables from bare patches (all slabs pooled)
    logm200, fgas_cyl, y_cyl, m500 = [], [], [], []
    # kSZ stacks per mass bin, accumulated across slabs
    ksz_profiles: dict[int, list[np.ndarray]] = {i: [] for i in range(len(config.ksz_mass_bins))}

    for slab in slabs:
        patches = slab["generated_patches"].astype(np.float32)
        thermo = slab["thermo_patches"].astype(np.float32)
        centers = slab["halo_centers"]
        masses = slab["halo_masses"].astype(float)
        r200 = slab["halo_r200"].astype(float)
        cond = slab["condition_sums"].astype(float)
        n = len(masses)

        r500 = r500c_from_r200c(r200, config.c200_for_apertures)
        m500_slab = m500c_from_m200c(masses, config.c200_for_apertures)
        for i in range(n):
            logm200.append(np.log10(masses[i]))
            m500.append(m500_slab[i])
            fgas_cyl.append(
                fgas_cylindrical(patches[i, 1], patch_geom, r500_hmpc=r500[i], m500_msunh=m500_slab[i])
            )
            y_cyl.append(y_aperture_mpc2(thermo[i, 0], patch_geom, r_hmpc=r500[i]))

        # gas composite for the CAP stacks (standard paste conventions)
        pmatch = patches.copy()
        for i in range(n):
            pmatch[i] *= cond[i] / (pmatch[i].sum() + 1e-30)
        halos = [{"halo_center": centers[i], "r200": float(r200[i])} for i in range(n)]
        weights_list = [
            circular_taper_weight(patch_pix, r_pix=float(r200[i]) * ppm * 4.0, taper_frac=0.15)
            for i in range(n)
        ]
        canvas, w_accum = paste_halos_2d(canvas_pix, box_size, halos, pmatch, sq, weights_list=weights_list)
        alpha = np.clip(w_accum, 0.0, 1.0)
        sigma_bar = cosmic_mean_gas_per_pixel(box_size / int(slab["n_slabs"]), box_size / canvas_pix)
        gas_map = blend_gas_background(canvas[1], alpha, sigma_bar)

        for bi, (lo, hi) in enumerate(config.ksz_mass_bins):
            sel = np.where((masses >= 10**lo) & (masses < 10**hi))[0][: config.max_halos_per_bin]
            for i in sel:
                cx = int(centers[i][0] * ppm) % canvas_pix
                cy = int(centers[i][1] * ppm) % canvas_pix
                cut = extract_periodic_cutout(gas_map, cx, cy, config.cutout_pix)
                ksz_profiles[bi].append(ksz_cap_profile(cut, geom, ksz_cfg))

    result = {
        "run": run_dir.name,
        "snap": snap,
        "z_snap": z_snap,
        "radii_arcmin": np.asarray(config.radii_arcmin, dtype=float),
        "logm200": np.array(logm200),
        "m500_msunh": np.array(m500),
        "fgas_cyl_r500": np.array(fgas_cyl),
        "y_cyl_r500_mpc2": np.array(y_cyl),
        "params": np.load(run_dir / "params.npy"),
        "conventions": json.dumps(
            {
                "beam_fwhm_arcmin": HILC_BEAM_FWHM_ARCMIN,
                "output": "tau_CAP arcmin^2" if config.v_rms_over_c is None else "T_kSZ muK arcmin^2",
                "r500_m500": f"NFW c200={config.c200_for_apertures} rescaling of catalog R200c/M200c (gate approximation)",
                "fgas_y": "cylindrical, bare un-rescaled patches; CylToSph correction applied downstream",
                "ksz": (
                    "composite cutouts, mass-matched circular-taper paste (Paper-2 "
                    "standard) + cosmic-mean gas background where alpha<1 "
                    "(v2, 2026-07-17 — restores CAP background compensation); "
                    "velocity decorrelation of the LOS column still pending (A4)"
                ),
                "table_version": 2,
            }
        ),
    }
    for bi, (lo, hi) in enumerate(config.ksz_mass_bins):
        profs = np.array(ksz_profiles[bi])
        result[f"ksz_bin{bi}_range"] = np.array([lo, hi])
        result[f"ksz_bin{bi}_nhalos"] = np.array(len(profs))
        result[f"ksz_bin{bi}_stack"] = stack_profiles(profs) if len(profs) else np.full(len(config.radii_arcmin), np.nan)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    return result
