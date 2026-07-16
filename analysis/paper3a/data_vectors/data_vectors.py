"""WP-A1 loader module: frozen observational data vectors for Project A.

Every function here reads a raw, externally-sourced data product from
``data_root()`` (default ``/mnt/ceph/users/mlee1/paper3/A/wp1_data/``,
override with ``$BIND_PAPER3A_DATA_ROOT``) and returns it as a
:class:`DataVector` (or :class:`GasFractionCatalog` for the per-cluster
eROSITA product) carrying values, covariance, bin definitions, and
provenance together — never bare arrays. This module does not import
``bind`` and has no model dependency; it is pure data plumbing consumed by
A3 (prior-support gate) and A5 (inference likelihood).

Full provenance (DOIs, retrieval dates, license, exact file→figure mapping)
lives in the WP's ``provenance.json`` files (one per source directory under
``data_root()``) and is narrated in
``bind-paper3-plans/projectA/wp1-data-assembly/DATA_VECTOR_FREEZE.md``
(private plans repo — not in this tree). Read that file before trusting a
"fiducial" default chosen below: several sources publish multiple pipeline
variants (e.g. DR9 vs DR10, corrected vs uncorrected) and the default
argument here is WP1's best-effort read of the paper's stated fiducial
choice, flagged for a second pass in the freeze doc, not an independently
re-derived result.

Units caveat: the source ``.npz``/``.fits`` files carry no embedded unit
metadata. Bin axes (arcmin, multipole, log10 mass) are labeled from the
originating figure's axis. Session-2 audit (2026-07-16) closed the signal
units and h-conventions against the actual paper PDFs: all CAP kSZ vectors
are muK arcmin^2 on the ACT DR6 hILC dr6.01 map (0.5' pixels, effective
Gaussian beam FWHM 1.6'), tau conversion tau_CAP = T_kSZ/T_CMB * c/v_rms;
Pandey's compton_shear is a dimensionless xi^{gamma_t y}; the Hadzhiyska
2026 Part II "ratio" is a hybrid data/model f~gas(theta) — see each
loader's ``value_units``/``notes``. Remaining flags are stated per-loader
(e.g. the Fig. 9 harmonic-space units, per-bin sigma_true values). Do not
feed a vector whose value_units still says "unverified" into a likelihood
without closing that flag first.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_DATA_ROOT = Path("/mnt/ceph/users/mlee1/paper3/A/wp1_data")


def data_root() -> Path:
    """Root directory of raw WP1 data products (override via env var)."""
    return Path(os.environ.get("BIND_PAPER3A_DATA_ROOT", str(DEFAULT_DATA_ROOT)))


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataVector:
    """A single binned observational data vector with its covariance.

    Attributes
    ----------
    name : short identifier, e.g. "ksz_hadzhiyska2024_pzbin1"
    source : human-readable citation, e.g. "Hadzhiyska et al. 2024, arXiv:2407.07152"
    bins : 1D array of bin centers (radial aperture, multipole, or log-mass)
    bin_type : what `bins` indexes, e.g. "theta_arcmin", "R_arcmin", "ell", "log10_Mhalo_Msun"
    bin_units : units string for `bins`
    values : 1D array of measured values, same length as `bins`
    value_units : units string for `values` (or explicit "unverified ..." flag)
    covariance : (N,N) array or None
    errors : 1D array; derived from sqrt(diag(covariance)) if not given directly
    mass_definition : e.g. "M200c", "M500c", "GGL-calibrated halo mass (Bigwood-style)"
    aperture : filter/aperture definition, e.g. "CAP, theta_out sqrt(2)*theta_in"
    redshift_range : (z_low, z_high) or None
    sample : short description of the galaxy/cluster sample
    h_convention : "h-free" | "h=1" | "unspecified" — verify before combining vectors
    notes : free-text caveats (fiducial-variant ambiguity, unit-verification status, etc.)
    provenance : dict with doi/url, retrieval_date, license, source_file
    """

    name: str
    source: str
    bins: np.ndarray
    bin_type: str
    bin_units: str
    values: np.ndarray
    value_units: str
    covariance: Optional[np.ndarray] = None
    errors: Optional[np.ndarray] = None
    mass_definition: Optional[str] = None
    aperture: Optional[str] = None
    redshift_range: Optional[tuple] = None
    sample: str = ""
    h_convention: str = "unspecified"
    notes: str = ""
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bins = np.asarray(self.bins, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
            if self.errors is None:
                self.errors = np.sqrt(np.diag(self.covariance))
        elif self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)
        if self.bins.shape != self.values.shape:
            raise ValueError(
                f"{self.name}: bins shape {self.bins.shape} != values shape {self.values.shape}"
            )
        if self.covariance is not None and self.covariance.shape != (len(self.values),) * 2:
            raise ValueError(
                f"{self.name}: covariance shape {self.covariance.shape} != "
                f"({len(self.values)}, {len(self.values)})"
            )

    def check_covariance(self, rtol: float = 1e-5, psd_atol: float = 1e-8) -> None:
        """Assert covariance is square, symmetric, and positive semi-definite.

        Raises AssertionError with a descriptive message on failure. No-op if
        this vector carries no covariance (errors-only).
        """
        if self.covariance is None:
            return
        cov = self.covariance
        assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], f"{self.name}: covariance not square"
        assert np.allclose(cov, cov.T, rtol=rtol, atol=1e-12), f"{self.name}: covariance not symmetric"
        eigvals = np.linalg.eigvalsh(cov)
        scale = max(abs(eigvals.max()), 1e-300)
        assert eigvals.min() > -psd_atol * scale, (
            f"{self.name}: covariance not positive semi-definite (min eig={eigvals.min():.3e}, "
            f"max eig={eigvals.max():.3e})"
        )

    def mass_floor_mask(self, log10_mhalo_min: float = 13.0) -> np.ndarray:
        """Boolean mask of bins at/above the M>=1e13 h^-1 Msun floor (R2).

        Only meaningful when `bin_type` is a log-mass axis. Raises
        NotImplementedError otherwise — the caller must supply an external
        per-bin mass proxy (e.g. from the paper's stated bin->mass mapping)
        for angular/multipole-binned vectors.
        """
        if self.bin_type != "log10_Mhalo_Msun":
            raise NotImplementedError(
                f"{self.name}: bin_type={self.bin_type!r} has no intrinsic mass axis; "
                "supply an external bin->mass mapping (see DATA_VECTOR_FREEZE.md)."
            )
        return self.bins >= log10_mhalo_min


@dataclass
class GasFractionCatalog:
    """Per-object (cluster/group) catalog: mass + gas-fraction pairs, not a binned vector.

    Attributes mirror DataVector's provenance conventions but the payload is
    a per-object table rather than a single binned data vector, since the
    eROSITA product is a cluster catalog, not a pre-binned f_gas(M) relation.
    """

    name: str
    source: str
    log10_m500_msun: Optional[np.ndarray]  # None if this catalog vintage lacks a mass column
    log10_m500_err: Optional[np.ndarray]
    f_gas500: Optional[np.ndarray]
    f_gas500_err: Optional[np.ndarray]
    redshift: np.ndarray
    mass_definition: str
    n_objects: int
    sample: str = ""
    notes: str = ""
    provenance: dict = field(default_factory=dict)

    def mass_floor_mask(self, log10_mhalo_min: float = 13.0) -> np.ndarray:
        """True where log10(M500) >= log10_mhalo_min. NaN (unconstrained mass) -> False."""
        if self.log10_m500_msun is None:
            raise ValueError(f"{self.name}: this catalog vintage has no M500 column (see notes)")
        return self.log10_m500_msun >= log10_mhalo_min  # NaN comparisons are False, not "sub-floor" — see mass_known_mask

    def mass_known_mask(self) -> np.ndarray:
        """True where a finite M500 estimate exists (~14.7% of erass1cl_primary rows are unconstrained)."""
        if self.log10_m500_msun is None:
            raise ValueError(f"{self.name}: this catalog vintage has no M500 column (see notes)")
        return np.isfinite(self.log10_m500_msun)

    def sub_floor_fraction(self, log10_mhalo_min: float = 13.0) -> float:
        """Fraction of MASS-KNOWN objects below the M>=1e13 floor (R2 bookkeeping).

        Objects with no mass estimate are excluded from both numerator and
        denominator (reported separately by `missing_mass_fraction`) rather
        than silently counted as sub-floor.
        """
        known = self.mass_known_mask()
        n_known = known.sum()
        if n_known == 0:
            return float("nan")
        sub = (self.log10_m500_msun[known] < log10_mhalo_min).sum()
        return sub / n_known

    def missing_mass_fraction(self) -> float:
        """Fraction of objects with no finite M500 estimate at all."""
        return 1.0 - self.mass_known_mask().sum() / self.n_objects


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def _provenance(subdir: str, source_file: str, extra: Optional[dict] = None) -> dict:
    """Look up this file's entry in <data_root>/<subdir>/provenance.json, if present."""
    prov_path = data_root() / subdir / "provenance.json"
    entry = {"source_file": str(data_root() / subdir / source_file)}
    if prov_path.exists():
        try:
            manifest = json.loads(prov_path.read_text())
            entry["provenance_json"] = str(prov_path)
            entry["manifest_entry"] = manifest.get("files", {}).get(source_file)
        except (json.JSONDecodeError, OSError):
            pass
    if extra:
        entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# 1. kSZ — Hadzhiyska et al. 2024, arXiv:2407.07152 (DESI photometric x ACT, 13 sigma)
# ---------------------------------------------------------------------------

_HADZ2024_SUBDIR = "ksz_hadzhiyska_2407.07152"
_HADZ2024_ZBIN_RANGES = {  # Table I; approximate photo-z bin edges — verify against Table I before use
    "pzbin1": (0.4, 0.55),
    "pzbin2": (0.55, 0.7),
    "pzbin3": (0.7, 0.85),
    "pzbin4": (0.85, 1.0),
}


def load_ksz_hadzhiyska2024(
    pzbin: int = 1,
    variant: str = "extended_dr10_allfoot_perbin_dr6_corr",
) -> DataVector:
    """CAP-filtered kSZ profile in one DESI photo-z bin (Hadzhiyska et al. 2024).

    Parameters
    ----------
    pzbin : 1-4, photometric redshift bin (see Table I of the paper).
    variant : pipeline variant encoded in the filename. Default is WP1's read
        of the paper's stated fiducial pipeline ("DR10, outlier + photo-z
        correction", text refs near "DR10: Outlier and photo-z correction
        (fiducial)"); the "extended" vs "main" LRG-sample choice was NOT
        independently confirmed against Fig. 1 — verify before A5 use.
        Other files present under the same dir: drop "extended_" or
        "dr10_" or "_corr" to get the alternative pipeline variants; see
        `provenance.json` for the full file list.

    Notes
    -----
    Values are the CAP-filtered stacked kSZ amplitude vs aperture radius;
    `value_units` is left unverified (see module docstring) — the paper
    reports units in the Fig. 1 caption, not embedded in the .npz.
    """
    key = f"pzbin{pzbin}"
    if key not in _HADZ2024_ZBIN_RANGES:
        raise ValueError(f"pzbin must be 1-4, got {pzbin}")
    fname = f"Fig1_Fig8_{variant}_{key}.npz"
    path = data_root() / _HADZ2024_SUBDIR / fname
    d = np.load(path)
    return DataVector(
        name=f"ksz_hadzhiyska2024_{key}_{variant}",
        source="Hadzhiyska et al. 2024, arXiv:2407.07152 (PRL submission)",
        bins=d["theta_arcmins"],
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=d["prof"],
        value_units="muK arcmin^2 (CAP T_kSZ; confirmed against arXiv:2407.07152 figure axes + estimator section, audit 2026-07-16)",
        covariance=d["cov"],
        mass_definition="stellar-mass-proxy LRG host halo (no explicit M500/M200 calibration in this paper)",
        aperture="Compensated Aperture Photometry (CAP) filter",
        redshift_range=_HADZ2024_ZBIN_RANGES[key],
        sample="DESI imaging (photometric) LRGs x ACT DR6",
        h_convention="n/a for the angular vector; halo-mass bin-edge h-convention still unaudited",
        notes=(
            "Map/beam convention (audit 2026-07-16): ACT DR6 hILC dr6.01, 0.5' pixels, "
            "effective Gaussian beam FWHM 1.6', ~15 muK-arcmin white noise; "
            "tau_CAP = T_kSZ/T_CMB * c/v_rms per the paper's own conversion. "
            "13-sigma combined detection; this file is one of 4 photo-z bins. "
            "Filename variant selects DR9/DR10 x extended/main x corrected/raw pipeline; "
            "default is WP1's best read of the paper's fiducial choice, not independently verified."
        ),
        provenance=_provenance(_HADZ2024_SUBDIR, fname),
    )


def load_ksz_hadzhiyska2024_mass_bins(
    variant: str = "extended_dr10_allfoot_perbin_dr6_corr",
) -> dict:
    """CAP kSZ profiles split by (stellar-mass-proxy) halo-mass bin.

    Returns a dict keyed by the paper's log10(M_halo/Msun) bin string, e.g.
    "11.25_11.50". Only the top bin (12.00-13.50) straddles the R2 M>=1e13
    floor; the two lower bins are entirely sub-floor.
    """
    out = {}
    for lo, hi in [(11.25, 11.50), (11.50, 12.00), (12.00, 13.50)]:
        fname = f"Fig4_mass_{lo:.2f}_{hi:.2f}.npz"
        path = data_root() / _HADZ2024_SUBDIR / fname
        if not path.exists():
            continue
        d = np.load(path)
        keys = list(d.keys())
        out[f"{lo:.2f}_{hi:.2f}"] = DataVector(
            name=f"ksz_hadzhiyska2024_mass_{lo:.2f}_{hi:.2f}",
            source="Hadzhiyska et al. 2024, arXiv:2407.07152",
            bins=d[keys[0]] if "theta_arcmins" not in d else d["theta_arcmins"],
            bin_type="theta_arcmin",
            bin_units="arcmin",
            values=d["prof"] if "prof" in d else d[keys[1]],
            value_units="muK arcmin^2 (CAP T_kSZ; same map/estimator conventions as the pzbin loader, audit 2026-07-16)",
            covariance=d["cov"] if "cov" in d else None,
            mass_definition="log10(M_halo/Msun) in (%.2f, %.2f), stellar-mass-proxy assignment" % (lo, hi),
            aperture="CAP filter",
            sample="DESI photometric LRGs x ACT DR6, split by inferred halo mass",
            notes="Raw keys: %s — cross-check against %s if this differs from the pzbin loader's schema."
            % (keys, fname),
            provenance=_provenance(_HADZ2024_SUBDIR, fname),
        )
    return out


# ---------------------------------------------------------------------------
# 2. kSZ — Ried Guachalla et al. 2025, arXiv:2503.19870 (PRD 112, 103512)
#    DESI Y1 spectroscopic LRGs x ACT DR6, S/N ~ 10
# ---------------------------------------------------------------------------

_RIED_SUBDIR = "ksz_ried_guachalla_2503.19870"


def load_ksz_ried_guachalla2025(binned_by: str = "mass") -> dict:
    """kSZ amplitude profiles binned by redshift or stellar mass (Ried Guachalla et al. 2025).

    Parameters
    ----------
    binned_by : "mass" (fig12_ksz_mass.npz, 4 stellar-mass quartiles) or
        "redshift" (fig11_ksz_z.npz, 4 redshift bins).

    Notes
    -----
    The correlation matrix (fig18_cor.npz, 9x9) in this release is a single
    shared shape-9 matrix; WP1 did NOT confirm from the paper text whether
    it applies identically to every mass/z bin or only to the fiducial
    (all-sample) stack — flagged in DATA_VECTOR_FREEZE.md as needing a
    second pass / author contact. `errors` below are therefore independent
    (paper-reported) errors, and `covariance` is left as None pending that
    confirmation rather than silently applying a possibly-mismatched matrix.
    """
    if binned_by not in ("mass", "redshift"):
        raise ValueError("binned_by must be 'mass' or 'redshift'")
    fname = "fig12_ksz_mass.npz" if binned_by == "mass" else "fig11_ksz_z.npz"
    prefix = "mass" if binned_by == "mass" else "z"
    path = data_root() / _RIED_SUBDIR / fname
    d = np.load(path)
    out = {}
    for i in range(1, 5):
        vkey, ekey = f"{prefix}_{i}", f"{prefix}_{i}_error"
        out[f"{prefix}_{i}"] = DataVector(
            name=f"ksz_riedguachalla2025_{prefix}{i}",
            source="Ried Guachalla et al. 2025, arXiv:2503.19870 (PRD 112, 103512)",
            bins=d["R"],
            bin_type="R_arcmin",
            bin_units="arcmin",
            values=d[vkey],
            value_units="muK arcmin^2 (CAP T_kSZ; confirmed against arXiv:2503.19870 Fig. 8/11/12 captions, audit 2026-07-16)",
            covariance=None,
            errors=d[ekey],
            mass_definition=(
                "stellar-mass quartile, log10(M*/Msun) edges [10.5, 11.2, 11.4, 11.6, ...] "
                "(paper Sec. on sample splits; h-free Msun) — no explicit M500/M200 in this file"
            ),
            aperture="CAP filter (ACT DR6 hILC dr6.01 map, 0.5' pixels, beam FWHM 1.6'; audit 2026-07-16)",
            redshift_range=None if binned_by == "mass" else None,  # not encoded in this npz; see paper Table
            sample=f"DESI Y1 spectroscopic LRGs x ACT DR6, {binned_by} bin {i}/4",
            h_convention="h-free (stellar masses in Msun)",
            notes=(
                "Independent (diagonal) errors only; a shared 9x9 correlation matrix "
                "(fig18_cor.npz) exists in this release but its applicability to this "
                "specific bin was not confirmed against the paper text — see "
                "load_ksz_ried_guachalla2025_correlation()."
            ),
            provenance=_provenance(_RIED_SUBDIR, fname),
        )
    return out


def load_ksz_ried_guachalla2025_correlation() -> np.ndarray:
    """Raw 9x9 correlation matrix bundled with the release (fig18_cor.npz).

    Applicability to a specific mass/z bin above is UNCONFIRMED — see the
    notes field of `load_ksz_ried_guachalla2025`. Returned as a bare array
    (not wrapped in DataVector) because it is not yet tied to one specific
    value vector.
    """
    path = data_root() / _RIED_SUBDIR / "fig18_cor.npz"
    return np.load(path)["cor"]


# ---------------------------------------------------------------------------
# 3. kSZ — Qu et al. 2026, arXiv:2604.19744 (Part I, LRGs, DESI DR2 x ACT DR6, 18 sigma)
#    — current best kSZ data vector per REFERENCES.md
# ---------------------------------------------------------------------------

_QU_SUBDIR = "ksz_qu_2604.19744"


def load_ksz_qu2026_lrg_fiducial() -> DataVector:
    """Full-sample CAP kSZ profile + covariance, the fig07 correlation-matrix release.

    This is the headline 18-sigma combined LRG measurement (all z, all mass).
    """
    path = data_root() / _QU_SUBDIR / "fig07_correlation_matrix.npz"
    d = np.load(path)
    cov = d["covariance"]
    # fig07 stores covariance+correlation but not the central values; pull
    # values from fig12 (cap_vs_simulations), which shares the same R grid.
    d12 = np.load(data_root() / _QU_SUBDIR / "fig12_cap_vs_simulations.npz")
    return DataVector(
        name="ksz_qu2026_lrg_fiducial",
        source="Qu et al. 2026, arXiv:2604.19744 (Part I: LRGs)",
        bins=d12["R_arcmin"],
        bin_type="R_arcmin",
        bin_units="arcmin",
        values=d12["T_ksz"],
        value_units=(
            "muK arcmin^2 (confirmed: arXiv:2604.19744 Fig. 12 caption 'mean stacked kSZ "
            "signal in muK arcmin2', audit 2026-07-16). Estimator (their Eq. 30): "
            "velocity-weighted uniform-mean, normalized so E[T_hat] = T_CMB*(sigma_true/c)*tau_CAP "
            "with the r ~= 0.65 reconstruction-fidelity correction already applied; "
            "sigma_true per bin from AbacusSummit (values not yet extracted — A5 to-do)"
        ),
        covariance=cov,
        mass_definition="full DESI DR2 LRG sample; see load_ksz_qu2026_lrg_by_mass for per-bin M200c ticks",
        aperture="CAP filter (ACT DR6 hILC dr6.01 map, 0.5' pixels, Gaussian beam FWHM 1.6'; audit 2026-07-16)",
        redshift_range=(0.4, 1.1),
        sample="DESI DR2 spectroscopic LRGs x ACT DR6 (all z, all mass; 18-sigma combined)",
        h_convention="h-free (M200c ticks and stellar masses in Msun; paper uses Msun throughout)",
        notes=(
            "Values pulled from fig12_cap_vs_simulations.npz. Bin-ordering cross-check "
            "CLOSED 2026-07-16: fig12.T_ksz_err == sqrt(diag(fig07.covariance)) to machine "
            "precision, fig12.covariance == fig07.covariance exactly, and both share the "
            "same R_arcmin grid. fig12 also carries the paper's simulation curves "
            "(DM, Illustris z0.5/z0.8, TNG z0.8) — useful for the A3 overlay sanity check."
        ),
        provenance=_provenance(_QU_SUBDIR, "fig07_correlation_matrix.npz"),
    )


def load_ksz_qu2026_lrg_by_mass() -> dict:
    """CAP kSZ profile + covariance per stellar-mass quartile (Qu et al. 2026, Part I).

    Also returns the paper's own log10(M200c/Msun) "tick" (central value per
    bin, from abundance-matching / HOD, not a per-galaxy calibration) via the
    `mass_definition` field, sourced from fig02_stellar_mass_distribution.npz.

    log10(M*) bin edges: [10.5, 11.2, 11.4, 11.6, 12.5]
    approx log10(M200c) ticks: [11.86, 12.49, 13.43, 14.57]  (bins 1-4)
    -> only bins 3 and 4 clear the M>=1e13 h^-1 Msun floor (R2); bins 1-2 are sub-floor.
    """
    path = data_root() / _QU_SUBDIR / "fig15_mass_dependence.npz"
    d = np.load(path)
    mass_meta = np.load(data_root() / _QU_SUBDIR / "fig02_stellar_mass_distribution.npz")
    m200c_ticks = mass_meta["m200c_ticks"]
    mstar_edges = mass_meta["mass_bin_boundaries"]
    out = {}
    for i in range(1, 5):
        out[f"m{i}"] = DataVector(
            name=f"ksz_qu2026_lrg_m{i}",
            source="Qu et al. 2026, arXiv:2604.19744 (Part I: LRGs)",
            bins=d[f"m{i}_R"],
            bin_type="R_arcmin",
            bin_units="arcmin",
            values=d[f"m{i}_T"],
            value_units="muK arcmin^2 (same estimator/normalization as the fiducial vector; audit 2026-07-16)",
            covariance=d[f"m{i}_cov"],
            mass_definition=(
                f"log10(M*/Msun) in ({mstar_edges[i-1]:.2f}, {mstar_edges[i]:.2f}); "
                f"approx log10(M200c/Msun) tick = {m200c_ticks[i-1]:.2f} "
                "(abundance-matching estimate from fig02, not a per-galaxy mass)"
            ),
            aperture="CAP filter (ACT DR6 hILC dr6.01, beam FWHM 1.6')",
            redshift_range=(0.4, 1.1),
            sample=f"DESI DR2 LRGs x ACT DR6, stellar-mass quartile {i}/4",
            h_convention="h-free (M200c tick in Msun, SHMR + c-M relation at z=0.7 per Fig. 2 caption)",
            notes="log10(M200c) tick = %.2f -> %s the M>=1e13 floor (R2)."
            % (m200c_ticks[i - 1], "clears" if m200c_ticks[i - 1] >= 13.0 else "BELOW"),
            provenance=_provenance(_QU_SUBDIR, "fig15_mass_dependence.npz"),
        )
    return out


# ---------------------------------------------------------------------------
# 4. kSZ — Hadzhiyska et al. 2026, arXiv:2604.19745 (Part II: BGS + ELG)
# ---------------------------------------------------------------------------

_HADZ2026_SUBDIR = "ksz_hadzhiyska_2604.19745"
_HADZ2026_ZIPROOT = "zenodo"


def load_ksz_hadzhiyska2026_bgs_elg(
    sample: str = "BGS_BRIGHT-20.2",
    log10_mstar: float = 11.25,
    space: str = "config",
) -> DataVector:
    """CAP (config-space) or harmonic-space kSZ profile for one BGS/ELG stellar-mass bin.

    Parameters
    ----------
    sample : "BGS_BRIGHT-20.2" or "ELG_LOPnotqso".
    log10_mstar : lower edge of the stellar-mass bin, as encoded in the
        filename (e.g. 11.25 -> "logm11.25"); valid values differ per sample
        (BGS: 9.50-11.25 in ~0.5 dex steps typically; ELG: 9.00-9.50) — list
        `data_root()/ksz_hadzhiyska_2604.19745/zenodo/` for the exact set.
    space : "config" (Fig8_*.npz: th, ratio, yerr, prof_kappa_err, cov_ksz) or
        "harmonic" (Fig9_*.npz: ell, cl_meas, cl_err, cov).
    """
    if space == "config":
        fname = f"Fig8_{sample}_logm{log10_mstar:.2f}.npz"
        path = data_root() / _HADZ2026_SUBDIR / _HADZ2026_ZIPROOT / fname
        d = np.load(path)
        return DataVector(
            name=f"ksz_hadzhiyska2026_{sample}_logm{log10_mstar:.2f}_config",
            source="Hadzhiyska et al. 2026, arXiv:2604.19745 (Part II: BGS+ELG)",
            bins=d["th"],
            bin_type="theta_arcmin",
            bin_units="arcmin",
            values=d["ratio"],
            value_units=(
                "dimensionless f~gas(theta) = T_kSZ^CAP / kappa^CAP, normalized so "
                "gas-traces-matter -> 1 (audit 2026-07-16, Fig. 8 caption + Sec. 'Gas fractions'). "
                "CAUTION: hybrid data/model quantity — the kappa^CAP denominator is a "
                "*prediction* from best-fit HOD-emulator parameters, not a measurement; "
                "r_fid = 0.64 (BGS) / 0.55 (ELG)"
            ),
            covariance=d["cov_ksz"],
            mass_definition=f"log10(M_stellar/Msun) >= {log10_mstar:.2f} (BGS/ELG selection)",
            aperture="CAP filter",
            sample=f"DESI DR2 {sample} x ACT DR6 (config space)",
            notes="Also carries 'yerr' and 'prof_kappa_err' fields not exposed here — see raw npz.",
            provenance=_provenance(_HADZ2026_SUBDIR, f"{_HADZ2026_ZIPROOT}/{fname}"),
        )
    elif space == "harmonic":
        fname = f"Fig9_{sample}_logm{log10_mstar:.2f}.npz"
        path = data_root() / _HADZ2026_SUBDIR / _HADZ2026_ZIPROOT / fname
        d = np.load(path)
        return DataVector(
            name=f"ksz_hadzhiyska2026_{sample}_logm{log10_mstar:.2f}_harmonic",
            source="Hadzhiyska et al. 2026, arXiv:2604.19745 (Part II: BGS+ELG)",
            bins=d["ell"],
            bin_type="ell",
            bin_units="dimensionless (multipole)",
            values=d["cl_meas"],
            value_units="unverified — C_ell-like kSZ power, see paper Fig. 9 caption",
            covariance=d["cov"],
            mass_definition=f"log10(M_stellar/Msun) >= {log10_mstar:.2f} (BGS/ELG selection)",
            sample=f"DESI DR2 {sample} x ACT DR6 (harmonic space)",
            notes="Also carries 'cl_model' (best-fit model curve) not exposed here — see raw npz.",
            provenance=_provenance(_HADZ2026_SUBDIR, f"{_HADZ2026_ZIPROOT}/{fname}"),
        )
    raise ValueError("space must be 'config' or 'harmonic'")


# ---------------------------------------------------------------------------
# 5. eROSITA gas fractions — eRASS1 DR1 public catalog (Bulbul et al. 2024)
#    as cited/compiled in Siegel et al. 2025, arXiv:2509.10455
# ---------------------------------------------------------------------------

_ERASS1_SUBDIR = "egas_erass1_bulbul"


def load_egas_erass1(catalog: str = "primary") -> GasFractionCatalog:
    """eRASS1 eROSITA DR1 cluster/group catalog with X-ray-derived M500 + f_gas500.

    Parameters
    ----------
    catalog : "primary" (erass1cl_primary_v3.2.fits, 12247 objects, HAS
        M500/FGAS500 columns — the X-ray scaling-relation mass, NOT the
        GGL-recalibrated mass Siegel et al. use) or "cosmology"
        (erass1cl_cosmology_v1.1.fits, 5259-object purity-selected
        subsample used for cosmology, which carries MGAS500 but NOT an
        M500 column in this release — see notes).

    Caveats (see DATA_VECTOR_FREEZE.md for the full writeup)
    ----------------------------------------------------------------
    - M500/FGAS500 in the "primary" catalog are the eROSITA-internal X-ray
      scaling-relation mass (Bulbul et al. 2024), not the weak-lensing
      GGL-calibrated mass that Siegel et al. 2509.10455 use to place these
      clusters on their fgas-M500 comparison (their Table 1-3). The two
      mass definitions can differ systematically — do not treat this
      catalog's M500 as a drop-in replacement for Siegel's GGL masses
      without checking Bulbul et al. 2024's calibration section.
    - Siegel et al.'s own compiled fgas(M) relation (their Fig. 5) is NOT
      machine-readable in their paper (no data-availability statement,
      no table) — it is a figure-only compilation of this catalog plus
      Popesso et al. 2024a and Akino et al. 2022. Treat any value read off
      that figure as `*_provisional` only (task 3 of the WP1 plan).
    """
    if catalog == "primary":
        fname = "erass1cl_primary_v3.2.fits"
    elif catalog == "cosmology":
        fname = "erass1cl_cosmology_v1.1.fits"
    else:
        raise ValueError("catalog must be 'primary' or 'cosmology'")
    path = data_root() / _ERASS1_SUBDIR / fname

    from astropy.io import fits  # local import: keep astropy optional for the rest of the module

    with fits.open(path) as f:
        t = f[1].data
        n = len(t)
        if catalog == "primary":
            # IMPORTANT: M500/M500_L/M500_H are in units of 1e13 Msun per the
            # FITS TUNIT header (verified against erass1cl_primary_v3.2.fits),
            # NOT Msun directly — missing/unconstrained masses are flagged
            # with non-positive sentinel values (~14.7% of rows) and become
            # NaN here rather than a silently wrong log10.
            m500_1e13 = np.asarray(t["M500"], dtype=float)
            m500_1e13_l = np.asarray(t["M500_L"], dtype=float)
            m500_1e13_h = np.asarray(t["M500_H"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                log10_m500 = np.where(m500_1e13 > 0, 13.0 + np.log10(m500_1e13), np.nan)
                log10_m500_err = np.where(
                    (m500_1e13_l > 0) & (m500_1e13_h > 0),
                    0.5 * (np.log10(m500_1e13_h) - np.log10(m500_1e13_l)),
                    np.nan,
                )
            f_gas = np.asarray(t["FGAS500"], dtype=float)
            f_gas = np.where(f_gas > 0, f_gas, np.nan)
            f_gas_err = 0.5 * (
                np.asarray(t["FGAS500_H"], dtype=float) - np.asarray(t["FGAS500_L"], dtype=float)
            )
            z = np.asarray(t["BEST_Z"], dtype=float)
        else:
            log10_m500 = None
            log10_m500_err = None
            f_gas = None
            f_gas_err = None
            z = np.asarray(t["Z_LAMBDA"], dtype=float)

    return GasFractionCatalog(
        name=f"egas_erass1_{catalog}",
        source="Bulbul et al. 2024 (eRASS1 DR1); compiled context in Siegel et al. 2025, arXiv:2509.10455",
        log10_m500_msun=log10_m500,
        log10_m500_err=log10_m500_err,
        f_gas500=f_gas,
        f_gas500_err=f_gas_err,
        redshift=z,
        mass_definition=(
            "M500c in h-free Msun at the eROSITA best-fit cosmology (Ghirardini et al. 2024), "
            "from the count-rate--mass scaling relation whose calibration IS weak-lensing based "
            "(Grandis et al.; audit 2026-07-16) — so 'X-ray scaling mass vs Siegel GGL mass' is a "
            "calibration-vintage difference, not an X-ray-vs-lensing dichotomy. CAVEAT (catalog "
            "paper Sec. on mass inference): median/mean point-estimate masses are biased HIGH for "
            "M500 <~ 1e14 Msun and biased LOW for M500 >~ 7e14 Msun (selection/Eddington-type); "
            "per-cluster mass PDFs ship in the catalog and should be used for any f_gas(M) "
            "likelihood in the group regime"
            if catalog == "primary"
            else "no M500 column in this DR1 vintage of the cosmology subsample"
        ),
        n_objects=n,
        sample=f"eRASS1 {catalog} catalog, western Galactic hemisphere, 12791 deg^2",
        notes=(
            "log10_m500_err is a crude symmetric half-range from the catalog's asymmetric "
            "M500_L/M500_H bounds, not a proper 1-sigma; refine before use in a likelihood."
            if catalog == "primary"
            else "Use catalog='primary' for M500/FGAS500; this subsample only has MGAS500, KT, L500, YX500."
        ),
        provenance=_provenance(_ERASS1_SUBDIR, fname),
    )


# ---------------------------------------------------------------------------
# 6. Shear x tSZ — Pandey et al. 2025, arXiv:2506.07432 (DES Y3 x ACT DR6, 21 sigma)
#    via the public GODMAX analysis repo (github.com/shivampcosmo/GODMAX, DESxACT branch)
# ---------------------------------------------------------------------------

_PANDEY_SUBDIR = "kappa_y_pandey_2506.07432"
_PANDEY_FITS = "DES_ACT_full_data_theorycov_2.5.fits"


def load_kappa_y_pandey2025(component: str = "compton_shear") -> DataVector:
    """DES Y3 shear x ACT DR6 tSZ (or 3x2pt) data vector, with the FULL joint covariance.

    Parameters
    ----------
    component : "compton_shear" (the gamma_t x y cross-correlation, 80
        points — the A6 posterior-predictive target), "xip", or "xim"
        (the accompanying DES Y3 cosmic-shear 3x2pt components, 200 points
        each). The on-disk COVMAT is the full 480x480 joint covariance
        (200 xip + 200 xim + 80 compton_shear, in that concatenation
        order); this loader slices out the requested block's diagonal
        sub-covariance, which is *not* the same as the marginal
        covariance unless the off-block correlations are negligible —
        for a joint fit, load the full COVMAT directly (see
        `load_kappa_y_pandey2025_full_covariance`).

    Caveats
    -------
    Sourced from the public analysis repo the paper's corresponding-author
    footnote points to (github.com/shivampcosmo/GODMAX, DESxACT branch),
    NOT from a formal "Data Availability" statement in the paper text (none
    was found) or a Zenodo/journal deposit. The repo carries no LICENSE
    file — treat as research-use pending explicit permission; do not
    redistribute beyond this project without confirming with the author
    (see REPORT.md author-contact draft).
    """
    order = {"xip": (0, 200), "xim": (200, 400), "compton_shear": (400, 480)}
    if component not in order:
        raise ValueError("component must be one of 'xip', 'xim', 'compton_shear'")
    lo, hi = order[component]

    from astropy.io import fits

    path = data_root() / _PANDEY_SUBDIR / _PANDEY_FITS
    with fits.open(path) as f:
        full_cov = np.asarray(f["COVMAT"].data, dtype=float)
        t = f[component].data
        bin1 = np.asarray(t["BIN1"])
        bin2 = np.asarray(t["BIN2"])
        angbin = np.asarray(t["ANGBIN"])
        ang = np.asarray(t["ANG"])
        value = np.asarray(t["VALUE"])

    sub_cov = full_cov[lo:hi, lo:hi]
    value_units = {
        "compton_shear": (
            "dimensionless xi^{gamma_t y}(theta) cross-correlation (paper figures plot "
            "values x 1e-9; audit 2026-07-16). Paper's quoted masses are in Msun/h."
        ),
        "xip": "DES Y3 cosmic shear xi_+, dimensionless (audit 2026-07-16)",
        "xim": "DES Y3 cosmic shear xi_-, dimensionless (audit 2026-07-16)",
    }[component]

    return DataVector(
        name=f"kappay_pandey2025_{component}",
        source="Pandey et al. 2025, arXiv:2506.07432 (DES Y3 x ACT DR6)",
        bins=ang,
        bin_type="theta_arcmin",
        bin_units="arcmin",
        values=value,
        value_units=value_units,
        covariance=sub_cov,
        mass_definition="tomographic lens/source bin (BIN1xBIN2), not a single halo-mass axis",
        aperture=None,
        redshift_range=None,  # see nz_source/nz_lens extensions in the same FITS file
        sample="DES Y3 metacal shear x ACT DR6+Planck Compton-y, 4x4 tomographic bins" if component != "compton_shear"
        else "DES Y3 metacal shear (source) x ACT DR6+Planck Compton-y (lens proxy)",
        h_convention="Msun/h for halo masses quoted in the paper (abstract + Sec. 1, audit 2026-07-16); the angular data vector itself is h-free",
        notes=(
            f"BIN1/BIN2/ANGBIN columns identify the tomographic sub-vector each point belongs "
            f"to (not exposed as separate arrays here — read the raw FITS table for the full "
            f"structure). sub_cov is the diagonal {component} block of the full 480x480 COVMAT; "
            "off-diagonal cross-covariance with the other two components is dropped by this "
            "slice — use load_kappa_y_pandey2025_full_covariance for a joint fit. Repo has no "
            "LICENSE file (see docstring)."
        ),
        provenance=_provenance(_PANDEY_SUBDIR, _PANDEY_FITS, extra={"bin1": bin1.tolist(), "bin2": bin2.tolist(), "angbin": angbin.tolist()} if len(bin1) < 0 else {}),
    )


def load_kappa_y_pandey2025_full_covariance() -> np.ndarray:
    """The full 480x480 joint covariance (200 xip + 200 xim + 80 compton_shear)."""
    from astropy.io import fits

    path = data_root() / _PANDEY_SUBDIR / _PANDEY_FITS
    with fits.open(path) as f:
        return np.asarray(f["COVMAT"].data, dtype=float)


def load_kappa_y_pandey2025_nz(which: str = "lens") -> dict:
    """Redshift distributions n(z) bundled with the data vector (nz_source or nz_lens extensions).

    Returns a dict with 'z_low', 'z_mid', 'z_high', and 'bin1'..'bin4' arrays.
    """
    if which not in ("lens", "source"):
        raise ValueError("which must be 'lens' or 'source'")

    from astropy.io import fits

    path = data_root() / _PANDEY_SUBDIR / _PANDEY_FITS
    with fits.open(path) as f:
        t = f[f"nz_{which}"].data
        return {name: np.asarray(t[name], dtype=float) for name in t.columns.names}
