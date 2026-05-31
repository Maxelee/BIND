"""extract_dmo_structure.py — per-halo DMO structural features for the SS-calibration study.

The Sobol cube (`sobol_ss_generation.py`) stores per-halo *outputs* (Y200, thermo, f_gas,
suppression) but not the structure of the DMO *input* the halos were painted onto. That input
(`condition` map) is **fixed across all 256 designs**, so its morphology is a design-independent
per-halo covariate — exactly the "DMO inheritance" features the BIND scatter analysis
(`analysis/2d`, `scatter.ipynb`) found to be the master predictors of scaling-relation scatter
("concentration is the strongest |rho| across most relations").

This script computes those features once, reusing the SAME halo loader as the cube so rows align
1:1 with `cube.npz['M200']`, and saves them next to the cube. They feed §6.6 of
`tsz_ss_calibration_sobol.ipynb`: does DMO structure explain the (weak) per-halo money plot?

Features per halo (all from the fixed DMO condition map, within catalog R200c):
  c_core    log10( M_DMO(<0.15 R200) / M_DMO(<R200) )   — central concentration
  r_half    half-mass radius within R200, in units of R200 (small => concentrated)
  q_DMO     projected axis ratio (minor/major) at R200c  — DMO shape
  q_DMO_in  projected axis ratio at 0.5 R200c            — inner DMO shape

Run (no GPU needed):
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    python extract_dmo_structure.py            # -> <OUT_ROOT>/dmo_structure.npz
"""
from __future__ import annotations

import numpy as np

from sobol_ss_generation import load_cv_halos, OUT_ROOT_DEFAULT, PIX_KPC
from scatter.obs_common import aperture_sum, axis_ratio_q, _RR_PIX

FEAT_NAMES = ["c_core", "r_half", "q_DMO", "q_DMO_in"]


def _half_mass_radius_frac(dmo: np.ndarray, r200_pix: float) -> float:
    """Radius enclosing 50% of the DMO mass within R200, in units of R200."""
    tot = aperture_sum(dmo, r200_pix)
    if tot <= 0:
        return np.nan
    radii = np.linspace(0.05, 1.0, 40) * r200_pix
    cum = np.array([aperture_sum(dmo, r) for r in radii]) / tot
    return float(np.interp(0.5, cum, radii / r200_pix))


def features_for_halo(dmo: np.ndarray, r200_pix: float) -> np.ndarray:
    r200_pix = max(float(r200_pix), 2.0)
    m_core = aperture_sum(dmo, 0.15 * r200_pix)
    m_tot = aperture_sum(dmo, r200_pix)
    c_core = np.log10(m_core / m_tot) if (m_core > 0 and m_tot > 0) else np.nan
    r_half = _half_mass_radius_frac(dmo, r200_pix)
    q_out = axis_ratio_q(dmo, r200_pix)
    q_in = axis_ratio_q(dmo, 0.5 * r200_pix)
    return np.array([c_core, r_half, q_out, q_in], dtype=np.float64)


def main() -> None:
    halos = load_cv_halos()
    dmo = halos["dmo"]                       # (N, 128, 128) physical DMO condition maps
    M200 = np.asarray(halos["M200"], float)
    R200 = np.asarray(halos["R200"], float)  # kpc/h
    n = len(M200)
    print(f"Extracting DMO structure for {n} halos; map shape {dmo.shape[1:]}")

    feats = np.full((n, len(FEAT_NAMES)), np.nan)
    for i in range(n):
        feats[i] = features_for_halo(dmo[i], R200[i] / PIX_KPC)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")

    out = OUT_ROOT_DEFAULT / "dmo_structure.npz"
    np.savez(out, feats=feats, feat_names=np.array(FEAT_NAMES, dtype=object),
             M200=M200, R200=R200, sim_id=halos["sim_id"])
    print(f"\nwrote {out}")
    for j, name in enumerate(FEAT_NAMES):
        v = feats[:, j]
        print(f"  {name:9s} median={np.nanmedian(v):+.3f}  "
              f"[{np.nanpercentile(v, 16):+.3f}, {np.nanpercentile(v, 84):+.3f}]  "
              f"({np.isfinite(v).sum()}/{n} finite)")


if __name__ == "__main__":
    main()
