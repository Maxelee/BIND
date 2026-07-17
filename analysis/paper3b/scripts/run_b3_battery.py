"""WP-B3 battery driver: null ensemble, RA shifts, B-mode stack, CIB band.

Executes NULL_CRITERIA.md §1–§4 as config variants of the frozen B2 chain
(`run_peak_stack`, MPI-collective when launched under mpirun — same contract
as run_peak_ystack.py). §5/§6 are inline re-analyses done separately.

Outputs → /mnt/home/mlee1/ceph/paper3/B/wp3_nulls/:
  ensemble_<variant>.npz          (N,nbin) realization means + jk sigmas
  stack_<variant>_shift{D}.npz    full bundles per surviving RA shift
  peaks_bmode_sm2am.npz + stack_bmode_sm2am_fid.npz
  stack_<variant>_sm2am_<cibtag>.npz   per CIB variant
  battery_summary.json            criteria evaluations (battery.py rules)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(_repo))

from analysis.paper3b.maps import act_mask_to_healpix, build_peak_catalog, load_act_mask  # noqa: E402
from analysis.paper3b.stack import (  # noqa: E402
    FIDUCIAL_RADIUS_INDEX,
    NU_STACK_EDGES,
    load_result,
    run_peak_stack,
    save_result,
)
from analysis.paper3b.stack.battery import (  # noqa: E402
    cib_band,
    null_ensemble_stats,
    shift_catalog_ra,
)
from analysis.paper3b.stack.mpiutil import mpi_comm  # noqa: E402

WP1 = Path("/mnt/home/mlee1/ceph/paper3/B/wp1_maps")
WP2 = Path("/mnt/home/mlee1/ceph/paper3/B/wp2_measurement")
OUT = Path("/mnt/home/mlee1/ceph/paper3/B/wp3_nulls")
YMAP_DIR = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/act_dr6_planck_ymap")
DES_DIR = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/des_y3_massmaps_jeffrey2105.13539")
J = FIDUCIAL_RADIUS_INDEX
VARIANTS = ("wiener", "glimpse")


def _cat(variant):
    c = np.load(WP1 / f"peaks_{variant}_sm2am.npz")
    nu = c["nu"]
    sel = (nu >= NU_STACK_EDGES[0]) & (nu < NU_STACK_EDGES[-1])
    return c["ra_deg"][sel], c["dec_deg"][sel], nu[sel]


def _random_in_binary(n, binary, nside, seed):
    import healpy as hp

    rng = np.random.default_rng(seed)
    ra, dec, got = [], [], 0
    while got < n:
        m = 4 * (n - got) + 1024
        r = rng.uniform(0, 360, m)
        d = np.degrees(np.arcsin(rng.uniform(-np.sin(np.radians(70)), np.sin(np.radians(15)), m)))
        ok = binary[hp.ang2pix(nside, np.radians(90 - d), np.radians(r))]
        ra.append(r[ok]); dec.append(d[ok]); got += int(ok.sum())
    return np.concatenate(ra)[:n], np.concatenate(dec)[:n]


def main() -> None:
    import healpy as hp
    from pixell import enmap

    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=32)
    ap.add_argument("--shifts", nargs="+", type=float, default=[15.0, -15.0, 30.0, -30.0])
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--skip-shifts", action="store_true")
    ap.add_argument("--skip-bmode", action="store_true")
    ap.add_argument("--skip-cib", action="store_true")
    args = ap.parse_args()

    comm = mpi_comm()
    rank = 0 if comm is None else comm.Get_rank()
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)

    f = np.load(WP1 / "common_footprint_nside1024.npz", allow_pickle=True)
    binary, weight, nside = f["binary"], f["weight"].astype(float), int(f["nside"])
    summary = {"nu_edges": NU_STACK_EDGES.tolist(), "criteria": "NULL_CRITERIA.md 2026-07-17"}
    fid = enmap.read_map(str(YMAP_DIR / "ilc_actplanck_ymap.fits"))

    # ---- §1 null ensemble --------------------------------------------------
    if not args.skip_ensemble:
        for vi, variant in enumerate(VARIANTS):
            _, _, nu_real = _cat(variant)
            nu = np.random.default_rng(7).permutation(nu_real)
            means, sigmas = [], []
            for k in range(args.ensemble):
                ra, dec = _random_in_binary(len(nu), binary, nside, seed=1000 + 5000 * vi + k)
                res = run_peak_stack(fid, ra, dec, nu, label=f"null_{variant}_r{k}", comm=comm)
                if rank == 0:
                    means.append(res.y_mean[:, J]); sigmas.append(res.y_err[:, J])
                    if k == 0:
                        save_result(res, OUT)
            if rank == 0:
                means, sigmas = np.array(means), np.array(sigmas)
                st = null_ensemble_stats(means, sigmas)
                np.savez_compressed(OUT / f"ensemble_{variant}.npz",
                                    means=means, jk_sigmas=sigmas, **{
                                        k: v for k, v in st.items() if k != "n_realizations"})
                summary[f"ensemble_{variant}"] = st
                print(f"[b3] ensemble {variant}: mean/2err = "
                      f"{np.round(st['mean_over_2err'], 2)}  jk-ratio = "
                      f"{np.round(st['jk_validation_ratio'], 2)}", flush=True)

    # ---- §2 RA shifts ------------------------------------------------------
    if not args.skip_shifts:
        act_hp = act_mask_to_healpix(load_act_mask(), nside=nside)
        for variant in VARIANTS:
            ra, dec, nu = _cat(variant)
            for dlt in args.shifts:
                ra2, dec2, keep, ret = shift_catalog_ra(ra, dec, dlt, act_hp)
                if ret < 0.60:
                    if rank == 0:
                        summary[f"shift_{variant}_{dlt:+g}"] = {"retention": ret, "verdict": "skipped <60%"}
                    continue
                res = run_peak_stack(fid, ra2, dec2, nu[keep],
                                     label=f"{variant}_shift{dlt:+g}", comm=comm)
                if rank == 0:
                    save_result(res, OUT)
                    summary[f"shift_{variant}_{dlt:+g}"] = {
                        "retention": ret, "n_per_bin": res.n_per_bin.tolist(),
                        "significance": res.significance().tolist(),
                    }
                    print(f"[b3] shift {variant} {dlt:+g}deg (ret {ret:.2f}): S/N = "
                          f"{np.round(res.significance(), 1).tolist()}", flush=True)
        del act_hp  # ~100 MB/rank

    # ---- §3 B-mode peaks ---------------------------------------------------
    if not args.skip_bmode:
        payload = None
        if rank == 0:
            m = hp.read_map(str(DES_DIR / "nullB_full.fits")).astype(np.float64)
            des_mask = hp.read_map(str(DES_DIR / "glimpse_mask.fits")).astype(bool)
            m[~des_mask] = 0.0
            m[m < -1e20] = 0.0
            cat = build_peak_catalog(m, weight, binary, 2.0, "bmode")
            np.savez_compressed(OUT / "peaks_bmode_sm2am.npz", **cat.to_npz_dict())
            payload = (cat.ra_deg, cat.dec_deg, cat.nu)
            print(f"[b3] bmode peaks: {len(cat.ipix)} ({cat.n_excluded} excluded)", flush=True)
        if comm is not None:
            payload = comm.bcast(payload, root=0)
        ra_b, dec_b, nu_b = payload
        res = run_peak_stack(fid, ra_b, dec_b, nu_b, label="bmode_sm2am_fid", comm=comm)
        if rank == 0:
            save_result(res, OUT)
            summary["bmode"] = {"n_per_bin": res.n_per_bin.tolist(),
                                "Y_nu_4am": res.y_mean[:, J].tolist(),
                                "significance": res.significance().tolist()}
            print(f"[b3] bmode stack: S/N = {np.round(res.significance(), 1).tolist()}", flush=True)

    # ---- §4 CIB band -------------------------------------------------------
    if not args.skip_cib:
        # Free the fiducial map before loading variants — holding both was
        # 2 x 1.78 GB/rank x 48 ranks = 170 GB and OOM-killed job 2451392;
        # the CIB section only needs the variant maps (fid Y values come from
        # the B2 archive).
        del fid
        cib_files = sorted(YMAP_DIR.glob("ilc_actplanck_ymap_deproj_*.fits"))
        for variant in VARIANTS:
            ra, dec, nu = _cat(variant)
            y_by = {}
            if rank == 0:
                ref = load_result(f"{variant}_sm2am_fid", WP2)
                y_by["fid"] = ref.y_mean[:, J]
                sigma_stat = ref.y_err[:, J]
            for fpath in cib_files:
                tag = fpath.stem.replace("ilc_actplanck_ymap_deproj_", "")
                if tag == "cib_1.7_10.7":       # already measured in B2 — reuse
                    if rank == 0:
                        y_by[tag] = load_result(f"{variant}_sm2am_deprojcib", WP2).y_mean[:, J]
                    continue
                ymap = enmap.read_map(str(fpath))
                res = run_peak_stack(ymap, ra, dec, nu, label=f"{variant}_sm2am_{tag}", comm=comm)
                del ymap
                if rank == 0:
                    save_result(res, OUT)
                    y_by[tag] = res.y_mean[:, J]
                    print(f"[b3] cib {variant} {tag}: S/N = "
                          f"{np.round(res.significance(), 1).tolist()}", flush=True)
            if rank == 0:
                band = cib_band(y_by, sigma_stat)
                summary[f"cib_band_{variant}"] = band
                print(f"[b3] CIB band {variant}: band/sigma = "
                      f"{np.round(band['band_over_sigma_stat'], 2)} escalate={band['escalate']}",
                      flush=True)

    if rank == 0:
        (OUT / "battery_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        print(f"[b3] wrote {OUT}/battery_summary.json", flush=True)
    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":
    main()
