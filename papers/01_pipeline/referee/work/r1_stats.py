"""Recompute every statistic of the three-rung ladder from the RAW map cubes.

  /mnt/home/mlee1/venvs/BIND_env/bin/python3 r1_stats.py --stage all

Stages (each writes its own npz under ceph/referee_work/r1/, skipped if present):
  pre                 metadata + seed-pairing / cube-identity checks
  kappa:<rung>        per-real Cl^kk (5 planes), mean kappa, nu-domain stats at z_s=1,2
  y:<rung>            per-real Cl^yy (5 planes), mean y, Cl^ky (kappa(z_s) x TOTAL y)
  tau:<rung>          per-real Cl^tautau (5 planes), mean tau, Cl^ktau
  cov                 550-realization pasted-truth per-real Cl^kk + nu stats (LSST cov)

No released Cl_*.npz / nongaussian_stats.npz cache is read anywhere (pre-fix XPk
normalization + stale MFs); see r1_common.py for the estimator conventions.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_common as C  # noqa: E402

RUNGS_K = ("bind", "pasted", "diffuse", "hydro_full", "dmo")
RUNGS_G = ("bind", "pasted", "diffuse", "hydro_full")
N_COV = 550        # pasted-truth realizations used for the LSST-Y10 covariance


def _t(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def stage_pre(t0):
    """Cube metadata + the identity checks the ladder rests on."""
    out = {}
    for rung, d in C.RUNS.items():
        for fld in ("kappa", "y", "tau"):
            p = d / f"{fld}_maps.npz"
            if not p.exists():
                continue
            with np.load(p) as f:
                out[f"{rung}/{fld}"] = {
                    "path": str(p), "n_real": int(np.asarray(f["n_real"]).ravel()[0]),
                    "npix": int(np.asarray(f["npix"]).ravel()[0]),
                    "fov_deg": float(np.asarray(f["fov_deg"]).ravel()[0]),
                    "zs": np.asarray(f["source_redshifts"]).ravel().tolist()}
    _t("metadata read", t0)

    # (1) is the 550-real BIND lightcone's first-50 prefix the same trace as the
    #     50-real bind_science/runs/bind/run_0000 cube the paper's fig 4/5 uses?
    a = C.load_prefix(C.LC / "kappa_maps.npz", "kappa", 3, C.ZI)
    b = C.load_prefix(C.SCI / "runs/bind/run_0000/kappa_maps.npz", "kappa", 3, C.ZI)
    out["check_bind_lc_vs_sci_kappa"] = {
        "max_abs_diff": float(np.abs(a - b).max()), "max_abs": float(np.abs(b).max()),
        "rms_frac": float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))}
    ay = C.load_prefix(C.LC / "y_maps.npz", "y", 3, C.ZI)
    by = C.load_prefix(C.SCI / "runs/bind/run_0000/y_maps.npz", "y", 3, C.ZI)
    out["check_bind_lc_vs_sci_y"] = {
        "max_abs_diff": float(np.abs(ay - by).max()), "max_abs": float(np.abs(by).max()),
        "mean_ratio": float(ay.mean() / by.mean())}
    del a, b, ay, by
    _t("BIND LC-vs-SCI identity checked", t0)

    # (2) diffuse kappa must equal pasted kappa by construction (float32 rounding).
    a = C.load_prefix(C.RUNS["diffuse"] / "kappa_maps.npz", "kappa", 3, C.ZI)
    b = C.load_prefix(C.RUNS["pasted"] / "kappa_maps.npz", "kappa", 3, C.ZI)
    out["check_diffuse_vs_pasted_kappa"] = {
        "max_abs_diff": float(np.abs(a - b).max()), "max_abs": float(np.abs(b).max())}
    # (3) seed pairing across TREES: pasted and hydro_full kappa must be highly
    #     correlated realization by realization (same plane geometry, seed 1992+7r).
    h = C.load_prefix(C.RUNS["hydro_full"] / "kappa_maps.npz", "kappa", 3, C.ZI)
    out["check_pairing_pasted_vs_full"] = {
        "per_real_pixel_corr": [float(np.corrcoef(b[r].ravel(), h[r].ravel())[0, 1])
                                for r in range(3)],
        "cross_real_pixel_corr": float(np.corrcoef(b[0].ravel(), h[1].ravel())[0, 1])}
    del a, b, h
    gc.collect()
    _t("pairing checks done", t0)
    (C.WORK / "pre.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def stage_kappa(rung, t0):
    from bind.inference.stats import power_spectrum
    K = C.load_prefix(C.RUNS[rung] / "kappa_maps.npz", "kappa", C.N_REAL)
    _t(f"{rung}: kappa cube {K.shape} loaded", t0)
    ell, _ = power_spectrum(K[0, 0])
    cl = np.empty((C.N_REAL, C.N_PLANES, len(ell)))
    for zi in range(C.N_PLANES):
        for r in range(C.N_REAL):
            cl[r, zi] = power_spectrum(K[r, zi])[1]
    _t(f"{rung}: {C.N_REAL * C.N_PLANES} kappa spectra", t0)
    store = {"ell": ell, "cl_kk": cl, "mean": K.mean(axis=(2, 3)),
             "sigma0": K.std(axis=(2, 3))}
    for zi in (C.ZI, C.ZI2):
        for k, v in C.nu_stats(np.ascontiguousarray(K[:, zi])).items():
            store[f"{k}_z{zi}"] = v
        _t(f"{rung}: nu-domain stats plane {zi}", t0)
    ng = C.nu_grid()
    store["nu"] = ng.NU
    np.savez_compressed(C.WORK / f"kappa_{rung}.npz", **store)
    del K, cl
    gc.collect()


def stage_gas(rung, fld, t0):
    from bind.inference.stats import power_spectrum
    F = C.load_prefix(C.RUNS[rung] / f"{fld}_maps.npz", fld, C.N_REAL)
    _t(f"{rung}/{fld}: cube {F.shape} loaded", t0)
    ell, _ = power_spectrum(F[0, 0])
    cl = np.empty((C.N_REAL, C.N_PLANES, len(ell)))
    for zi in range(C.N_PLANES):
        for r in range(C.N_REAL):
            cl[r, zi] = power_spectrum(F[r, zi])[1]
    _t(f"{rung}/{fld}: {C.N_REAL * C.N_PLANES} autos", t0)
    store = {"ell": ell, f"cl_{fld}": cl, "mean": F.mean(axis=(2, 3))}
    tot = np.ascontiguousarray(F[:, -1])          # TOTAL column, cross convention
    del F
    gc.collect()
    for zi in (C.ZI, C.ZI2):
        K = C.load_prefix(C.RUNS[rung] / "kappa_maps.npz", "kappa", C.N_REAL, zi)
        x = np.empty((C.N_REAL, len(ell)))
        for r in range(C.N_REAL):
            x[r] = power_spectrum(K[r], tot[r])[1]
        store[f"cl_k{fld}_z{zi}"] = x
        del K
        gc.collect()
        _t(f"{rung}/{fld}: cross at plane {zi}", t0)
    np.savez_compressed(C.WORK / f"{fld}_{rung}.npz", **store)
    del tot
    gc.collect()


def stage_cov(t0):
    """550-realization pasted-truth statistics at z_s=1: the LSST-Y10 covariance."""
    from bind.inference.stats import power_spectrum
    K = C.load_prefix(C.RUNS["pasted"] / "kappa_maps.npz", "kappa", N_COV, C.ZI)
    _t(f"cov: {K.shape} loaded", t0)
    ell, _ = power_spectrum(K[0])
    cl = np.empty((N_COV, len(ell)))
    for r in range(N_COV):
        cl[r] = power_spectrum(K[r])[1]
    _t(f"cov: {N_COV} spectra", t0)
    store = {"ell": ell, "cl_kk": cl}
    for lo in range(0, N_COV, 50):                # chunked: bounds peak memory
        s = C.nu_stats(np.ascontiguousarray(K[lo:lo + 50]))
        for k, v in s.items():
            store.setdefault(k, []).append(v)
        _t(f"cov: nu stats {lo}-{lo + 50}", t0)
    for k in list(store):
        if isinstance(store[k], list):
            store[k] = np.concatenate(store[k], 0)
    store["nu"] = C.nu_grid().NU
    np.savez_compressed(C.WORK / "cov_pasted550.npz", **store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    C.WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = []
    if a.stage in ("all", "pre"):
        todo.append(("pre", None, C.WORK / "pre.json"))
    if a.stage in ("all", "kappa"):
        todo += [("kappa", r, C.WORK / f"kappa_{r}.npz") for r in RUNGS_K]
    if a.stage in ("all", "gas"):
        todo += [(f, r, C.WORK / f"{f}_{r}.npz") for f in ("y", "tau") for r in RUNGS_G]
    if a.stage in ("all", "cov"):
        todo.append(("cov", None, C.WORK / "cov_pasted550.npz"))
    for kind, rung, path in todo:
        if path.exists() and not a.force:
            print(f"skip {path.name} (exists)", flush=True)
            continue
        print(f"=== {kind} {rung or ''} -> {path.name}", flush=True)
        if kind == "pre":
            stage_pre(t0)
        elif kind == "kappa":
            stage_kappa(rung, t0)
        elif kind == "cov":
            stage_cov(t0)
        else:
            stage_gas(rung, kind, t0)
        _t(f"WROTE {path.name}", t0)


if __name__ == "__main__":
    main()
