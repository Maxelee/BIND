"""WP-A7 survey configurations (task 1) — published values only.

Provenance (fetched + verified 2026-07-19 via the papers' own text):
- DES-Y6:  Yamamoto et al. 2025 (arXiv:2501.05665), Table 3, H12
  definitions: n_eff = 8.22 arcmin^-2, sigma_e = 0.289/component,
  area 4422 deg^2. z_eff ~ 0.63 adopted from the DES-Y3 source n(z)
  (Y6 n(z) similar; our frozen weights_desy3 convention).
- HSC-Y3:  Li et al. 2022 (arXiv:2107.00136): n_eff = 19.9 arcmin^-2
  (Chang+13 definition), area 433.48 deg^2; sigma_e = 0.26/component
  adopted in SHEAR units (the paper's e_RMS is in distortion units;
  0.26 is the standard shear-unit equivalent). z_eff ~ 0.80 adopted
  (deeper than DES; approximate, flagged).
- LSST-Y1: DESC SRD v1 (arXiv:1809.01669), App. D2/F2/F4: n_eff = 10
  arcmin^-2, sigma_e = 0.26/component, mean source z = 0.85 (their
  Fig. F4 fit); area = 12,300 deg^2 (the SRD WFD footprint at Y1
  depth — Y1 is shallower, not smaller).

sigma_e is PER COMPONENT; the convergence noise convention used in
safescale.py is N_ell = sigma_e^2 / nbar with nbar per steradian.
"""

from __future__ import annotations

from dataclasses import dataclass

ARCMIN2_PER_SR = (180.0 * 60.0 / 3.141592653589793) ** 2
DEG2_PER_SKY = 41252.96


@dataclass(frozen=True)
class Survey:
    name: str
    n_eff_arcmin2: float
    sigma_e: float
    area_deg2: float
    z_eff: float
    provenance: str

    @property
    def f_sky(self) -> float:
        return self.area_deg2 / DEG2_PER_SKY

    @property
    def noise_cl(self) -> float:
        return self.sigma_e**2 / (self.n_eff_arcmin2 * ARCMIN2_PER_SR)


SURVEYS = [
    Survey("DES-Y6", 8.22, 0.289, 4422.0, 0.63,
           "Yamamoto+25 2501.05665 Tab.3 (H12); z_eff from Y3 n(z)"),
    Survey("HSC-Y3", 19.9, 0.26, 433.48, 0.80,
           "Li+22 2107.00136 (C13 n_eff); sigma_e shear-units adopted"),
    Survey("LSST-Y1", 10.0, 0.26, 12300.0, 0.85,
           "DESC SRD 1809.01669 App.D2/F2/F4"),
]
