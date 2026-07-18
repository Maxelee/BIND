"""Mock LRG-host sample machinery: mass selection, miscentering, satellites (WP-A2 task 1b).

The kSZ stacks are galaxy-weighted, not halo-weighted: the operator must
stack painted halos with weights matching the *observed sample's* host-mass
distribution (GGL-calibrated, Bigwood et al. 2510.15822 methodology), apply
miscentering between the galaxy and the halo center, and place a satellite
fraction off-center. All three pieces are config inputs traced to A1
metadata / paper prose — nothing here hard-codes an HMF or a specific
paper's numbers (acceptance criterion: "no hard-coded HMF sampling").

Conventions:
- host mass distribution: lognormal in log10 M200c, (mu, sigma) per sample
  bin. The A1 freeze carries per-bin M200c "ticks" (Qu et al. 2026 fig02);
  sigma comes from the paper text / Bigwood methodology when the audit
  closes and is an explicit UNVERIFIED config until then.
- miscentering: with probability f_mis the stack center is offset by a
  2D-Gaussian (Rayleigh-modulus) draw of scale sigma_mis_hmpc, the standard
  cluster/CMB-cross convention.
- satellites: with probability f_sat the "galaxy" sits on a satellite at a
  Rayleigh radius of scale r_sat_hmpc (an effective halo-scale offset —
  refine to an NFW-weighted draw if the truth validation shows it matters).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# GGL-calibrated mean host-halo masses per DESI+ACT kSZ sample bin, from
# Siegel et al. 2025 (arXiv:2509.10455), as quoted in prose (not the garbled
# table conversion) by Bigwood et al. 2025 (arXiv:2510.15822): Sec 4.2
# "log10(<M500[Msun]>) = 13.31 for the BGS sample and 13.30 for the LRG
# sample"; Sec 4.3 "log10(<M500[Msun]>) = 12.91, 13.16" for LRG M1/M2; Sec 4.2
# discussion (Appendix B lead-in) "log10(<M500[Msun]>) = 13.41, 13.57, 13.81
# ... for LRG M3, LRG M3 Photo-z, ... and LRG M4". Verified against the arXiv
# HTML full text 2026-07-18 (WP-A5 kSZ sample-model task). These are M500,
# GGL-selected -- NOT the abundance-matching-derived M200c "ticks" WP-A1
# recorded from Qu et al. 2026's own release (11.86/12.49/13.43/14.57): the
# two methodologies disagree by up to several tenths of a dex at the high-mass
# end (WP1/METHODS.md Sec 2); this is the R4/tension caveat, not a bug to
# reconcile silently. Only bins whose target sits inside the emulator's kSZ
# range (logM500 in [13.2, 15.5], `gasemu.logm500_bin_edges` via
# `ksz_bin{0,1}_range`) are usable by the A5 forward model; LRG M1 (12.91) and
# M2 (13.16) fall below/at the group-bin floor -- informational only, do not
# feed them into the likelihood without a documented extrapolation.
SIEGEL_GGL_LOGM500_TARGETS = {
    "bgs": 13.31,
    "lrg_full": 13.30,
    "lrg_m1": 12.91,          # sub-floor: informational only
    "lrg_m2": 13.16,          # below the emulator's 13.2 group-bin edge
    "lrg_m3_spec": 13.41,
    "lrg_m3_photo": 13.57,
    "lrg_m4": 13.81,
}

# Satellite fraction: Bigwood et al. 2025 Sec 3.5 states the GGL-matched
# simulated samples carry satellite fractions "of ~10-30%" (a qualitative
# range confirmed in the paper's prose; the per-simulation/per-bin table in
# the same section did not survive the arXiv HTML table conversion cleanly
# enough to trust individual digits, so no precise per-bin numbers are
# hard-coded here -- only the range the text itself commits to). The paper
# also notes real DESI-sample satellite fractions are themselves uncertain
# ("there remains uncertainty in the satellite fractions of the observed DESI
# samples"). Treat as a uniform PRIOR range for `f_sat`, mass-independent by
# default (the qualitative trend is a *decrease* toward higher mass bins per
# Sec 4.3, but that trend was not quantified in verifiable prose).
BIGWOOD_SATELLITE_FRACTION_RANGE = (0.10, 0.30)

# Mis-centering: Bigwood et al. 2025 Sec 3.6 finds mis-centering is "not a
# major source of concern" for hydro-sim-vs-data comparison (self-consistent
# sims already place galaxies realistically relative to the hot gas); the one
# quantified test displaces the CAP center by a *uniform* draw of width 0.5'
# (the ACT map's pixel size) in each sky coordinate and finds "a negligible
# impact on the shape of the inner kSZ profile". Default f_mis = 0 here; the
# 0.5' pixel-quantization scale is exposed as `PIXEL_QUANT_SIGMA_MIS_ARCMIN`
# for callers who want to bound the (expected-negligible) systematic
# explicitly rather than assume it away.
PIXEL_QUANT_SIGMA_MIS_ARCMIN = 0.5


@dataclass(frozen=True)
class MockSampleConfig:
    """Selection + centering model for one observed sample bin."""

    logm200c_mean: float
    logm200c_sigma: float
    f_mis: float = 0.0
    sigma_mis_hmpc: float = 0.0
    f_sat: float = 0.0
    r_sat_hmpc: float = 0.0
    notes: str = ""

    def __post_init__(self):
        for name in ("f_mis", "f_sat"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.logm200c_sigma <= 0:
            raise ValueError(f"logm200c_sigma must be > 0, got {self.logm200c_sigma}")


def sample_weights(logm200c_catalog: np.ndarray, config: MockSampleConfig) -> np.ndarray:
    """Importance weights turning a catalog's halo-mass distribution into the sample's.

    w_i = target(logM_i) / catalog_density(logM_i), normalized to sum to 1.
    The catalog density is estimated with a simple histogram (Freedman-
    Diaconis-ish fixed rule) — adequate because downstream stacks are
    insensitive to smooth reweighting details, and it keeps the function
    dependency-free. Halos far outside the target (>5 sigma) get weight 0.

    Warns loudly (ValueError) if the effective sample size collapses below
    ~20 halos — a sign the catalog does not cover the requested mass range
    and the stack would be dominated by a handful of objects.
    """
    logm = np.asarray(logm200c_catalog, dtype=float)
    if logm.ndim != 1 or len(logm) == 0:
        raise ValueError("logm200c_catalog must be a non-empty 1D array")
    target = np.exp(-0.5 * ((logm - config.logm200c_mean) / config.logm200c_sigma) ** 2)
    target[np.abs(logm - config.logm200c_mean) > 5 * config.logm200c_sigma] = 0.0

    n_bins = max(8, int(np.sqrt(len(logm))))
    hist, edges = np.histogram(logm, bins=n_bins)
    density = hist[np.clip(np.digitize(logm, edges) - 1, 0, n_bins - 1)].astype(float)
    density[density <= 0] = np.inf  # no support -> zero weight

    w = target / density
    total = w.sum()
    if total <= 0:
        raise ValueError(
            f"no catalog halos within 5 sigma of logM200c = {config.logm200c_mean} "
            f"+/- {config.logm200c_sigma} — catalog does not cover this sample bin"
        )
    w = w / total
    n_eff = 1.0 / np.sum(w**2)
    if n_eff < 20:
        raise ValueError(
            f"effective sample size {n_eff:.1f} < 20 for logM200c = "
            f"{config.logm200c_mean} +/- {config.logm200c_sigma} — the stack would "
            "be dominated by a few halos; widen the catalog or flag this bin"
        )
    return w


def draw_center_offsets_hmpc(
    n: int,
    config: MockSampleConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n, 2) comoving offsets of the stack center from the halo center.

    Order of application: satellite assignment first (f_sat), then
    miscentering of the remainder (f_mis). Offsets add nothing for the
    well-centered central fraction (1 - f_sat)(1 - f_mis).
    """
    offsets = np.zeros((n, 2))
    is_sat = rng.random(n) < config.f_sat
    is_mis = (~is_sat) & (rng.random(n) < config.f_mis)
    for mask, scale in ((is_sat, config.r_sat_hmpc), (is_mis, config.sigma_mis_hmpc)):
        k = int(mask.sum())
        if k and scale > 0:
            offsets[mask] = rng.normal(0.0, scale, size=(k, 2))
    return offsets
