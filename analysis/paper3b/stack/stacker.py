"""WP-B2 core stacking pipeline: (y map, peak catalog) -> the measurement bundle.

Operates identically on data and B4 mocks (MEASUREMENT_SPEC: no data-only
branches): the map is any pixell enmap, the catalog any (ra, dec, nu) triple.
Per peak: a native-CAR thumbnail (15', 0.5' — the B1 anchor convention) and
CAP Y at the frozen radius set; per stacking-ν-bin: mean thumbnail, radial
profile, jackknife mean±cov of Y; plus abundances per bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..maps.stack import extract_thumbnails, radial_profile
from .cap import cap_filter_multi
from .covariance import jackknife_mean_cov, patch_ids

CAP_RADII_ARCMIN = (2.0, 3.0, 4.0, 6.0, 8.0)
FIDUCIAL_RADIUS_INDEX = 2          # theta_d = 4' — the headline <Y(nu)> radius
NU_STACK_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 12.0])
THUMB_R_ARCMIN = 15.0
THUMB_RES_ARCMIN = 0.5
NSIDE_JK_DEFAULT = 8
NSIDE_JK_STABILITY = (4, 16)


@dataclass
class PeakStackResult:
    """One (map, catalog) stack per MEASUREMENT_SPEC — see run_peak_stack."""

    label: str
    nu_edges: np.ndarray
    cap_radii_arcmin: np.ndarray
    n_per_bin: np.ndarray            # (nbin,) peaks stacked per nu bin
    y_mean: np.ndarray               # (nbin, nrad) jackknife mean CAP Y [y arcmin^2]
    y_cov: np.ndarray                # (nbin, nrad, nrad) jackknife cov (nside_jk=8)
    y_sigma_stability: dict          # nside_jk -> (nbin, nrad) jackknife sigmas
    n_patches: np.ndarray            # (nbin,) occupied jackknife patches
    profiles_r_arcmin: np.ndarray    # (nprof,)
    profiles: np.ndarray             # (nbin, nprof) azimuthal mean of the bin stack
    stacked_thumbs: np.ndarray       # (nbin, ny, nx) float32 mean thumbnails
    per_peak_y: np.ndarray           # (npeak, nrad) for re-binning downstream
    per_peak_nu: np.ndarray
    per_peak_patch8: np.ndarray
    extras: dict = field(default_factory=dict)

    @property
    def y_err(self) -> np.ndarray:
        return np.sqrt(np.array([np.diag(c) for c in self.y_cov]))

    def significance(self) -> np.ndarray:
        """(nbin,) detection S/N of the headline-radius <Y> per nu bin."""
        j = FIDUCIAL_RADIUS_INDEX
        return self.y_mean[:, j] / self.y_err[:, j]


def run_peak_stack(ymap, ra_deg, dec_deg, nu, label: str,
                   nu_edges: np.ndarray = NU_STACK_EDGES,
                   cap_radii=CAP_RADII_ARCMIN, comm=None) -> PeakStackResult | None:
    """The frozen chain: thumbnails -> per-peak CAP -> nu-binned jackknife stats.

    ``comm``: an MPI communicator makes the heavy part (thumbnail extraction +
    per-peak CAP) COLLECTIVE — ranks stride the peak list, the per-peak table
    is gathered and the per-bin thumbnail sums reduced to rank 0, which alone
    computes the (fast) jackknife/profiles and returns the result; other ranks
    return None. ``comm=None`` (the default, and what B4's mock loop uses) is
    the identical single-process chain — same code path, same numbers.
    """
    ra_deg, dec_deg, nu = (np.asarray(a, dtype=np.float64) for a in (ra_deg, dec_deg, nu))
    sel = (nu >= nu_edges[0]) & (nu < nu_edges[-1])
    ra_deg, dec_deg, nu = ra_deg[sel], dec_deg[sel], nu[sel]

    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    my = np.arange(len(nu))[rank::size]

    thumbs_local = np.asarray(extract_thumbnails(ymap, ra_deg[my], dec_deg[my],
                                                 THUMB_R_ARCMIN, THUMB_RES_ARCMIN),
                              dtype=np.float64)
    y_local = cap_filter_multi(thumbs_local, cap_radii, THUMB_RES_ARCMIN)

    nbin, nrad = len(nu_edges) - 1, len(cap_radii)
    binof = np.digitize(nu, nu_edges) - 1
    ny, nx = thumbs_local.shape[-2:] if thumbs_local.size else (61, 61)
    thumb_sums = np.zeros((nbin, ny, nx))
    for b in range(nbin):
        m = binof[my] == b
        if m.any():
            thumb_sums[b] = thumbs_local[m].sum(axis=0)

    if comm is not None and size > 1:
        from mpi4py import MPI
        all_idx = comm.gather(my, root=0)
        all_y = comm.gather(y_local, root=0)
        thumb_sums = comm.reduce(thumb_sums, op=MPI.SUM, root=0)
        if rank != 0:
            return None
        per_peak_y = np.empty((len(nu), nrad))
        for idx, yv in zip(all_idx, all_y):
            per_peak_y[idx] = yv
    else:
        per_peak_y = y_local

    ids8 = patch_ids(ra_deg, dec_deg, NSIDE_JK_DEFAULT)
    n_per_bin = np.bincount(binof, minlength=nbin).astype(int)
    y_mean = np.full((nbin, nrad), np.nan)
    y_cov = np.full((nbin, nrad, nrad), np.nan)
    n_patches = np.zeros(nbin, dtype=int)
    stab = {ns: np.full((nbin, nrad), np.nan) for ns in NSIDE_JK_STABILITY}
    stacked = []
    profiles = []
    r_prof = None
    for b in range(nbin):
        m = binof == b
        if n_per_bin[b] == 0:
            stacked.append(np.zeros((ny, nx), dtype=np.float32))
            profiles.append(np.full(int(THUMB_R_ARCMIN), np.nan))
            continue
        st = thumb_sums[b] / n_per_bin[b]
        stacked.append(st.astype(np.float32))
        r_prof, prof = radial_profile(st, THUMB_RES_ARCMIN, 1.0, THUMB_R_ARCMIN)
        profiles.append(prof)
        y_mean[b], y_cov[b], n_patches[b] = jackknife_mean_cov(per_peak_y[m], ids8[m])
        for ns in NSIDE_JK_STABILITY:
            try:
                _, cov_s, _ = jackknife_mean_cov(per_peak_y[m], patch_ids(ra_deg[m], dec_deg[m], ns))
                stab[ns][b] = np.sqrt(np.diag(cov_s))
            except ValueError:      # too few patches at this nside for this bin
                pass

    return PeakStackResult(
        label=label, nu_edges=np.asarray(nu_edges, dtype=float),
        cap_radii_arcmin=np.asarray(cap_radii, dtype=float),
        n_per_bin=n_per_bin, y_mean=y_mean, y_cov=y_cov,
        y_sigma_stability={str(k): v for k, v in stab.items()},
        n_patches=n_patches,
        profiles_r_arcmin=np.asarray(r_prof, dtype=float),
        profiles=np.asarray(profiles, dtype=float),
        stacked_thumbs=np.asarray(stacked, dtype=np.float32),
        per_peak_y=per_peak_y, per_peak_nu=nu, per_peak_patch8=ids8,
    )
