"""WP-B3 §7/§8 driver: star/PSF hotspot + dust-region covariate tests.

Pure re-analysis of the FROZEN per-peak tables (`wp2_measurement/
stack_{variant}_sm2am_fid.npz`) — no new stacking, no change to the frozen
central vector (blinding discipline preserved; NULL_CRITERIA.md §7/§8).
Peak RA/Dec come from `wp1_maps/peaks_{variant}_sm2am.npz`, re-selected with
the IDENTICAL nu-range cut `run_peak_stack` applies, so row order matches
`per_peak_y` (verified: nu arrays match exactly).

Covariates:
  star_count / psf_dresid  — from `build_star_psf_map.py`'s HEALPix map
                              (nearest-pixel lookup at each peak position).
  ebv                       — SFD E(B-V) via `dustmaps.sfd.SFDQuery`.

Outputs: wp3_nulls/star_dust_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WP1 = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")
WP2 = Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement")
WP3 = Path("/mnt/home/mlee1/ceph/paper3/B/wp3_nulls")
DUSTMAPS_DATA = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/dustmaps_data")
VARIANTS = ("wiener", "glimpse")


def _peak_positions(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RA/Dec/nu for the frozen per-peak rows, same order as per_peak_y."""
    c = np.load(WP1 / f"peaks_{variant}_sm2am.npz")
    sel = (c["nu"] >= 0.0) & (c["nu"] < 12.0)
    return c["ra_deg"][sel], c["dec_deg"][sel], c["nu"][sel]


def _ebv_at(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from dustmaps.config import config
    config["data_dir"] = str(DUSTMAPS_DATA)
    from dustmaps.sfd import SFDQuery

    sfd = SFDQuery()
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return np.asarray(sfd(c), dtype=np.float64)


def _star_psf_at(ra_deg: np.ndarray, dec_deg: np.ndarray, star_map: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    import healpy as hp

    nside = int(star_map["nside"])
    pix = hp.ang2pix(nside, np.radians(90.0 - dec_deg), np.radians(ra_deg % 360.0))
    star_count = star_map["star_count"][pix].astype(np.float64)
    psf_dresid = star_map["psf_dresid"][pix].astype(np.float64)
    # a handful of peaks may land on an unoccupied (0-star) pixel; fall back
    # to the nside=1024 pixel's neighbours' median for those (rare at the
    # DES-footprint scale — reserved-star sampling is dense).
    bad = ~np.isfinite(psf_dresid)
    if bad.any():
        neigh = hp.get_all_neighbours(nside, pix[bad])
        for j, p in enumerate(pix[bad].tolist()):
            vals = star_map["psf_dresid"][neigh[:, j]]
            vals = vals[np.isfinite(vals)]
            psf_dresid[np.flatnonzero(bad)[j]] = float(vals.mean()) if len(vals) else 0.0
    return star_count, psf_dresid


def main() -> None:
    from analysis.paper3b.stack.battery import covariate_excision_test, proximity_tercile_test
    from analysis.paper3b.stack.covariance import patch_ids
    from analysis.paper3b.stack.stacker import FIDUCIAL_RADIUS_INDEX, NU_STACK_EDGES

    J = FIDUCIAL_RADIUS_INDEX
    star_npz = np.load(WP3 / "star_psf_map_nside1024.npz")
    star_map = {k: star_npz[k] for k in star_npz.files}

    summary = {"nu_edges": NU_STACK_EDGES.tolist(),
              "criteria": "NULL_CRITERIA.md 2026-07-18 sections 7-8",
              "star_psf_map_source": str(star_map["source"])}

    for variant in VARIANTS:
        s = np.load(WP2 / f"stack_{variant}_sm2am_fid.npz")
        per_peak_y, per_peak_nu = s["per_peak_y"], s["per_peak_nu"]
        y_mean_ref, y_err_ref = s["y_mean"][:, J], np.sqrt(s["y_cov"][:, J, J])

        ra, dec, nu = _peak_positions(variant)
        assert np.allclose(nu, per_peak_nu), f"{variant}: position/per_peak_y order mismatch"
        patch8 = patch_ids(ra, dec, 8)

        star_count, psf_dresid = _star_psf_at(ra, dec, star_map)
        ebv = _ebv_at(ra, dec)

        for cov_name, cov in (("star_density", star_count),
                             ("psf_dresid", psf_dresid),
                             ("ebv", ebv)):
            tercile = proximity_tercile_test(per_peak_y, per_peak_nu, cov)
            excision = covariate_excision_test(per_peak_y, per_peak_nu, cov, patch8,
                                               y_mean_ref, y_err_ref,
                                               keep_below_percentile=90.0)
            summary[f"{variant}_{cov_name}_tercile"] = tercile
            summary[f"{variant}_{cov_name}_excision"] = excision
            worst_sigma = max(abs(tercile[b]["sigma"]) for b in tercile
                              if "sigma" in tercile[b])
            all_pass_excision = all(bin_result.get("pass_0p5sig", True)
                                    for bin_result in excision.values()
                                    if isinstance(bin_result, dict))
            print(f"[b3-star-dust] {variant} {cov_name}: worst |tercile sigma| = "
                  f"{worst_sigma:.2f}, excision all-pass = {all_pass_excision}",
                  flush=True)

    WP3.mkdir(parents=True, exist_ok=True)
    (WP3 / "star_dust_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[b3-star-dust] wrote {WP3}/star_dust_summary.json", flush=True)


if __name__ == "__main__":
    main()
