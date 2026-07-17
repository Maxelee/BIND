"""WP-B2 driver: the measurement — ACT y stacked at DES κ-peak positions.

Runs the frozen MEASUREMENT_SPEC chain (`analysis/paper3b/stack/`) on:
  {wiener, glimpse} σ=2′ B1 catalogs × {fiducial, CIB-deprojected} y maps,
plus a random-position null per variant (random positions inside the common
binary footprint, ν values PERMUTED from the real catalog so per-bin counts
match). Per run: ⟨Y(ν)⟩ ± jackknife at 5 CAP radii, per-bin profiles/stamps,
patch-size stability (nside_jk 4/8/16), hemisphere split at median RA,
abundances. Deterministic from B1 products + raw maps.

Outputs → /mnt/home/mlee1/ceph/paper3/B/wp2_measurement/
(`stack_*.npz` via stack.results + `measurement_summary.json`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.maps import load_act_ymap  # noqa: E402
from analysis.paper3b.stack import (  # noqa: E402
    FIDUCIAL_RADIUS_INDEX,
    NU_STACK_EDGES,
    run_peak_stack,
    save_result,
)
from analysis.paper3b.stack.covariance import jackknife_mean_cov, patch_ids  # noqa: E402

WP1 = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")
YMAP_DIR = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/act_dr6_planck_ymap")
YMAPS = {
    "fid": "ilc_actplanck_ymap.fits",
    "deprojcib": "ilc_actplanck_ymap_deproj_cib_1.7_10.7.fits",
}
FOOTPRINT_DEG2 = 4386.3
J = FIDUCIAL_RADIUS_INDEX


def _split_consistency(ra, dec, nu, per_peak_y, edges):
    """Per-bin hemisphere consistency at the fiducial radius [units of sigma].

    Split at the catalog's median RA; each half gets its own nside_jk=16
    jackknife (finer patches so both halves keep >=3 of them).
    """
    med = np.median(ra)
    out = []
    for b in range(len(edges) - 1):
        m = (nu >= edges[b]) & (nu < edges[b + 1])
        cons = np.nan
        try:
            ha, hb = m & (ra < med), m & (ra >= med)
            a, ca, _ = jackknife_mean_cov(per_peak_y[ha][:, [J]], patch_ids(ra[ha], dec[ha], 16))
            bm, cb, _ = jackknife_mean_cov(per_peak_y[hb][:, [J]], patch_ids(ra[hb], dec[hb], 16))
            cons = float((a[0] - bm[0]) / np.sqrt(ca[0, 0] + cb[0, 0]))
        except ValueError:
            pass
        out.append(cons)
    return out


def _random_in_binary(n, seed=0):
    import healpy as hp

    f = np.load(WP1 / "common_footprint_nside1024.npz", allow_pickle=True)
    binary = f["binary"]
    nside = int(f["nside"])
    rng = np.random.default_rng(seed)
    ra, dec = [], []
    got = 0
    while got < n:
        m = 4 * (n - got) + 1024
        r = rng.uniform(0, 360, m)
        d = np.degrees(np.arcsin(rng.uniform(-np.sin(np.radians(70)), np.sin(np.radians(15)), m)))
        ok = binary[hp.ang2pix(nside, np.radians(90 - d), np.radians(r))]
        ra.append(r[ok]); dec.append(d[ok]); got += int(ok.sum())
    return np.concatenate(ra)[:n], np.concatenate(dec)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["wiener", "glimpse"])
    ap.add_argument("--ymaps", nargs="+", default=["fid", "deprojcib"])
    ap.add_argument("--with-null", action="store_true", default=True)
    args = ap.parse_args()

    from pixell import enmap

    summary = {"nu_edges": NU_STACK_EDGES.tolist(), "footprint_deg2": FOOTPRINT_DEG2,
               "fiducial_radius_arcmin": 4.0, "runs": {}}
    for ytag in args.ymaps:
        ymap = enmap.read_map(str(YMAP_DIR / YMAPS[ytag]))
        for variant in args.variants:
            cat = np.load(WP1 / f"peaks_{variant}_sm2am.npz")
            ra, dec, nu = cat["ra_deg"], cat["dec_deg"], cat["nu"]
            label = f"{variant}_sm2am_{ytag}"
            print(f"[b2] stacking {label}: {len(ra)} peaks ...", flush=True)
            res = run_peak_stack(ymap, ra, dec, nu, label=label)
            save_result(res)
            sel = (nu >= NU_STACK_EDGES[0]) & (nu < NU_STACK_EDGES[-1])
            sig8 = res.y_err
            drift = {ns: np.nanmax(np.abs(res.y_sigma_stability[ns] / sig8 - 1.0))
                     for ns in res.y_sigma_stability}
            summary["runs"][label] = {
                "n_per_bin": res.n_per_bin.tolist(),
                "abundance_per_deg2": (res.n_per_bin / FOOTPRINT_DEG2).tolist(),
                "Y_nu_4am": res.y_mean[:, J].tolist(),
                "Y_nu_4am_err": res.y_err[:, J].tolist(),
                "significance": res.significance().tolist(),
                "n_patches": res.n_patches.tolist(),
                "jk_sigma_max_drift": {k: float(v) for k, v in drift.items()},
                "hemisphere_split_sigma": _split_consistency(
                    ra[sel], dec[sel], nu[sel], res.per_peak_y, NU_STACK_EDGES),
            }
            print(f"[b2] {label}: S/N per bin = "
                  f"{np.round(res.significance(), 1).tolist()}", flush=True)

        if ytag == "fid" and args.with_null:
            for variant in args.variants:
                cat = np.load(WP1 / f"peaks_{variant}_sm2am.npz")
                nu = np.random.default_rng(7).permutation(cat["nu"])
                ra, dec = _random_in_binary(len(nu), seed=11)
                label = f"{variant}_sm2am_nullpos"
                print(f"[b2] null {label}: {len(ra)} positions ...", flush=True)
                res = run_peak_stack(ymap, ra, dec, nu, label=label)
                save_result(res)
                summary["runs"][label] = {
                    "n_per_bin": res.n_per_bin.tolist(),
                    "Y_nu_4am": res.y_mean[:, J].tolist(),
                    "Y_nu_4am_err": res.y_err[:, J].tolist(),
                    "significance": res.significance().tolist(),
                }
                print(f"[b2] {label}: S/N per bin = "
                      f"{np.round(res.significance(), 1).tolist()}", flush=True)

    out = Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement")
    out.mkdir(parents=True, exist_ok=True)
    (out / "measurement_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[b2] wrote {out}/measurement_summary.json", flush=True)


if __name__ == "__main__":
    main()
