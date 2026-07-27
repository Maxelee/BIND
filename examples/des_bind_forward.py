#!/usr/bin/env python
"""D3 (docs/tsz_des_data_plan.md §3): BIND-side forward model to DES Y3.

2pt leg (--twopt), per D1's mandate: xi_pm is NOT Hankel-transformed from the
raw 5-deg-FOV Cl (low-ell truncation loses 10-130% across the DES theta
range). Instead:

    xi_pm^node(theta) = Hankel[ Cl_theory^ab(ell) * S_ab^node(ell) ]

with Cl_theory from pyccl (TNG300 cosmology, the OFFICIAL DES nz_source per
bin — exact n(z), no plane discretization) and S_ab^node(ell) the n(z)-plane-
weighted suppression built from the per-run (5,5) Cl_kappa matrices over the
paired DMO trace:

    S_ab(ell) = sum_ij w_a(i) w_b(j) Cl_node[i,j] / (same with Cl_DMO)

(ratio-of-weighted-sums; the coarse 5-plane approximation only touches the
slowly-z-varying S, never the Cl itself; CIC window+aliasing cancel in the
ratio — D1). S is extended by its low-ell mean below ell=87 (S->1 regime,
plan D1) and frozen at its [1.5e4,2e4] mean above 2e4. The n(z) systematic
is bracketed by recomputing theory+weights for a subsample of the 1000
official nz_source realisations (--n_nz_real, default 50).

Map leg: identical bind.inference.stats estimators on both sides:
  --des_map_stats  peaks/minima/moments on the 45 D2 patches
                   (glimpse_full, wiener_full, nullB_full), smoothing 5'/10',
                   nu_norm='map' (per-map sigma normalization — this divides
                   out most of the GLIMPSE/Wiener amplitude suppression).
  --bind_map_stats per node: full-n(z)-weighted kappa (n_eff-weighted DES
                   bins -> plane weights), Gatti+21 shape noise (and a
                   noiseless variant — the noise-model spread joins the
                   filter bracket in the systematic band), same estimator.

Gatti+21 constants (plan decision item 1, default adopted):
  n_eff = [1.476, 1.479, 1.484, 1.461] gal/arcmin^2, sigma_e = [0.243,
  0.262, 0.259, 0.301]; full map: n_eff=5.9, sigma_e=0.268.

Usage
-----
    python examples/des_bind_forward.py --twopt
    python examples/des_bind_forward.py --des_map_stats
    python examples/des_bind_forward.py --bind_map_stats          # ~1 h, background
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"
RUNS = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")
DMO_CL = Path("/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000/Cl_kappa.npz")
FID_CL = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/Cl_kappa.npz")
TWOPT_NPZ = LC / "des_y3_2pt.npz"
PATCH_DIR = LC / "des_patches"

Z_PLANES = np.array([0.5, 1.0, 1.5, 2.0, 2.44])
# TNG300 cosmology (Planck15, the SB35/fiducial painting cosmology)
COSMO = dict(Omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667, Om=0.3089)
N_EFF = np.array([1.476, 1.479, 1.484, 1.461])
SIGMA_E = np.array([0.243, 0.262, 0.259, 0.301])
N_EFF_FULL, SIGMA_E_FULL = 5.9, 0.268

ELL_LOW_MEAN_MAX = 300.0     # S(ell<87) := mean S over [87, 300]
ELL_FREEZE = (1.5e4, 2.0e4)  # S(ell>2e4) := mean over this window
ELL_GRID = np.geomspace(2.0, 3.0e4, 3000)
SMOOTH_SCALES = [5.0, 10.0]
NU_BINS = np.linspace(-4.0, 6.0, 26)

BIN_PAIRS = [(a, b) for a in range(1, 5) for b in range(a, 5)]  # 10 DES pairs


def node_ids_all():
    return np.load(LC / "ksz_consistent_nodes.npz")["node_ids_all"]


# ---------------------------------------------------------------------------
# 2pt leg
# ---------------------------------------------------------------------------

def plane_weights(nz_z, nz_bin):
    """w(plane) ∝ n(z_plane), normalized (desact_sheary_realfit pattern —
    the coarse-plane approximation, flagged in the module docstring)."""
    w = np.interp(Z_PLANES, nz_z, nz_bin, left=0.0, right=0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(len(Z_PLANES), 1.0 / len(Z_PLANES))


def weighted_cl(cl55, wa, wb):
    return np.einsum("i,j,ijl->l", wa, wb, cl55)


def build_S(cl_node, cl_dmo, wa, wb, ell_meas):
    """S_ab on ELL_GRID with low-ell extension + high-ell freeze + light
    5-bin box smoothing of the measured ratio."""
    num = weighted_cl(cl_node, wa, wb)
    den = weighted_cl(cl_dmo, wa, wb)
    r = num / den
    k = np.ones(5) / 5.0
    r_sm = np.convolve(np.pad(r, 2, mode="edge"), k, mode="valid")
    low = float(np.mean(r_sm[ell_meas <= ELL_LOW_MEAN_MAX]))
    hi_m = (ell_meas >= ELL_FREEZE[0]) & (ell_meas <= ELL_FREEZE[1])
    hi = float(np.mean(r_sm[hi_m]))
    S = np.interp(ELL_GRID, ell_meas, r_sm, left=low, right=hi)
    return S


def theory_cls(nz_z, nz_bins):
    """pyccl per-bin-pair Cl on ELL_GRID at TNG300 cosmology, exact n(z)."""
    import pyccl as ccl
    cosmo = ccl.Cosmology(Omega_c=COSMO["Om"] - COSMO["Omega_b"],
                          Omega_b=COSMO["Omega_b"], h=COSMO["h"],
                          sigma8=COSMO["sigma8"], n_s=COSMO["n_s"])
    trs = [ccl.WeakLensingTracer(cosmo, dndz=(nz_z, nz_bins[i])) for i in range(4)]
    out = {}
    for (a, b) in BIN_PAIRS:
        out[(a, b)] = ccl.angular_cl(cosmo, trs[a - 1], trs[b - 1], ELL_GRID)
    return out


def hankel_mats(theta_arcmin):
    """Precompute J0/J4 kernel matrices (n_theta, n_ell) with the ell measure."""
    from scipy.special import jv
    th = np.deg2rad(np.asarray(theta_arcmin) / 60.0)
    x = np.outer(th, ELL_GRID)
    w = ELL_GRID / (2 * np.pi)
    return jv(0, x) * w, jv(4, x) * w


def run_twopt(n_nz_real=50, out_path=LC / "bind_des_xipred.npz"):
    t0 = time.time()
    d2 = np.load(TWOPT_NPZ)
    nz_z, nz_src = d2["nz_z_mid"], d2["nz_source"]
    # per-pair theta grids (the measured mean-pair angles differ per pair)
    th_p = d2["xip_ang_arcmin"].reshape(10, 20)
    th_m = d2["xim_ang_arcmin"].reshape(10, 20)
    J0s = [hankel_mats(th_p[p])[0] for p in range(10)]
    J4s = [hankel_mats(th_m[p])[1] for p in range(10)]

    ids = node_ids_all()
    cl_dmo = np.load(DMO_CL)
    ell_meas = cl_dmo["ell"]
    cl_dmo55 = cl_dmo["cl"]
    w = [plane_weights(nz_z, nz_src[i]) for i in range(4)]
    cth = theory_cls(nz_z, nz_src)

    def xi_for(cl55):
        xp = np.empty((len(BIN_PAIRS), 20))
        xm = np.empty((len(BIN_PAIRS), 20))
        for p, (a, b) in enumerate(BIN_PAIRS):
            S = build_S(cl55, cl_dmo55, w[a - 1], w[b - 1], ell_meas)
            integ = cth[(a, b)] * S
            xp[p] = np.trapezoid(J0s[p] * integ[None, :], ELL_GRID, axis=1)
            xm[p] = np.trapezoid(J4s[p] * integ[None, :], ELL_GRID, axis=1)
        return xp, xm

    # DMO/theory baseline (S=1)
    xp_th = np.empty((len(BIN_PAIRS), 20))
    xm_th = np.empty((len(BIN_PAIRS), 20))
    for p, (a, b) in enumerate(BIN_PAIRS):
        xp_th[p] = np.trapezoid(J0s[p] * cth[(a, b)][None, :], ELL_GRID, axis=1)
        xm_th[p] = np.trapezoid(J4s[p] * cth[(a, b)][None, :], ELL_GRID, axis=1)

    # fiducial + all nodes
    xp_fid, xm_fid = xi_for(np.load(FID_CL)["cl"])
    xp_nodes = np.empty((len(ids), len(BIN_PAIRS), 20))
    xm_nodes = np.empty((len(ids), len(BIN_PAIRS), 20))
    for k, nid in enumerate(ids):
        cl55 = np.load(RUNS / f"run_{nid:04d}" / "Cl_kappa.npz")["cl"]
        xp_nodes[k], xm_nodes[k] = xi_for(cl55)
        if k % 50 == 0:
            print(f"  node {k}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)

    # n(z)-realisation systematic band on the FIDUCIAL (subsampled)
    reals = d2["nz_source_realisations"]
    rng = np.random.default_rng(4)
    sub = rng.choice(len(reals), size=min(n_nz_real, len(reals)), replace=False)
    cl_fid55 = np.load(FID_CL)["cl"]
    xp_nz = np.empty((len(sub), len(BIN_PAIRS), 20))
    xm_nz = np.empty((len(sub), len(BIN_PAIRS), 20))
    for r, ri in enumerate(sub):
        nz_r = reals[ri].astype(np.float64)
        w_r = [plane_weights(nz_z, nz_r[i]) for i in range(4)]
        cth_r = theory_cls(nz_z, nz_r)
        for p, (a, b) in enumerate(BIN_PAIRS):
            S = build_S(cl_fid55, cl_dmo55, w_r[a - 1], w_r[b - 1], ell_meas)
            integ = cth_r[(a, b)] * S
            xp_nz[r, p] = np.trapezoid(J0s[p] * integ[None, :], ELL_GRID, axis=1)
            xm_nz[r, p] = np.trapezoid(J4s[p] * integ[None, :], ELL_GRID, axis=1)
        if r % 10 == 0:
            print(f"  nz real {r}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(out_path,
             bin_pairs=np.array(BIN_PAIRS), theta_p=th_p, theta_m=th_m,
             xip_theory=xp_th, xim_theory=xm_th,
             xip_fid=xp_fid, xim_fid=xm_fid,
             xip_nodes=xp_nodes, xim_nodes=xm_nodes, node_ids=ids,
             xip_nzreal_fid=xp_nz, xim_nzreal_fid=xm_nz,
             cosmo=str(COSMO), z_planes=Z_PLANES,
             plane_weights=np.array(w))
    print(f"[twopt] {len(ids)} nodes + fid + {len(sub)} nz reals in "
          f"{time.time()-t0:.0f}s -> {out_path}")


# ---------------------------------------------------------------------------
# map legs
# ---------------------------------------------------------------------------

def _stats_for_maps(maps4d, shape_noise_ngal, sigma_e, noise_seed=0):
    """peaks/minima (nu_norm='map') + moments for (n_real, 1, npix, npix)."""
    import sys
    sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
    from bind.inference.stats import peak_counts
    res = peak_counts(maps4d, fov_deg=5.0, smoothing_arcmin=SMOOTH_SCALES,
                      nu_bins=NU_BINS, shape_noise_ngal=shape_noise_ngal,
                      sigma_e=sigma_e, noise_seed=noise_seed, nu_norm="map",
                      return_realizations=True)
    return res


def run_des_map_stats(out_path=LC / "des_map_stats.npz"):
    out = {}
    for name in ("glimpse_full", "wiener_full", "nullB_full"):
        p = np.load(PATCH_DIR / f"{name}_patches.npz")["patches"]
        maps4d = p[:, None, :, :].astype(np.float64)     # patches as realizations
        res = _stats_for_maps(maps4d, None, 0.26)
        out[f"{name}_peaks_real"] = res["peak_counts_real"][:, :, 0]   # (scale, n_patch, nu)
        out[f"{name}_minima_real"] = res["minima_counts_real"][:, :, 0]
        # per-patch smoothed sigma (amplitude record, kept for honesty)
        from bind.inference.stats import _gaussian_smooth
        sig = np.array([[float(_gaussian_smooth(m[0], s, 5.0).std()) for m in maps4d]
                        for s in SMOOTH_SCALES])
        out[f"{name}_sigma"] = sig
        print(f"[des_map_stats] {name} done", flush=True)
    out["nu"] = 0.5 * (NU_BINS[1:] + NU_BINS[:-1])
    out["smoothing_arcmin"] = np.array(SMOOTH_SCALES)
    np.savez(out_path, **out)
    print(f"[des_map_stats] -> {out_path}")


def full_nz_plane_weights():
    d2 = np.load(TWOPT_NPZ)
    nz_z, nz_src = d2["nz_z_mid"], d2["nz_source"]
    nfull = (N_EFF[:, None] * nz_src).sum(axis=0)
    return plane_weights(nz_z, nfull)


SHARD_DIR = LC / "map_stats_shards"


def run_bind_map_stats_node(nid, n_real=50):
    """One node's shard (disBatch task; idempotent). The monolithic
    run_bind_map_stats hit a 6h Slurm wall at ~96s/node sequential AND
    ~65GB RSS (float64 einsum + npz decompression churn) — per-task shards
    in float32 keep each task ~3GB / ~2min and embarrassingly parallel."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out = SHARD_DIR / f"node_{nid:04d}.npz"
    if out.exists():
        print(f"[map_stats_node] exists, skipping {out}")
        return
    t0 = time.time()
    wfull = full_nz_plane_weights().astype(np.float32)
    kap = np.load(RUNS / f"run_{nid:04d}" / "kappa_maps.npz")["kappa"][:n_real]
    full = np.einsum("p,rpxy->rxy", wfull, kap.astype(np.float32))[:, None]
    res_n = _stats_for_maps(full, N_EFF_FULL, SIGMA_E_FULL, noise_seed=int(nid))
    res_c = _stats_for_maps(full, None, SIGMA_E_FULL)
    import sys as _sys
    _sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
    from bind.inference.stats import _gaussian_smooth
    sigma = np.array([[_gaussian_smooth(full[r, 0], sc, 5.0).std()
                       for r in range(full.shape[0])] for sc in SMOOTH_SCALES],
                     dtype=np.float32)
    np.savez(out,
             peaks_noisy=res_n["peak_counts_real"][:, :, 0].astype(np.float32),
             minima_noisy=res_n["minima_counts_real"][:, :, 0].astype(np.float32),
             peaks_clean=res_c["peak_counts_real"][:, :, 0].astype(np.float32),
             minima_clean=res_c["minima_counts_real"][:, :, 0].astype(np.float32),
             sigma=sigma, node_id=nid)
    print(f"[map_stats_node] {nid}: {time.time()-t0:.0f}s -> {out}", flush=True)


def run_bind_map_stats_merge(out_path=LC / "bind_map_stats.npz", n_real=50):
    ids = node_ids_all()
    n_nu = len(NU_BINS) - 1
    shapes = (len(ids), len(SMOOTH_SCALES), n_real, n_nu)
    out = {v: np.empty(shapes, dtype=np.float32)
           for v in ("peaks_noisy", "minima_noisy", "peaks_clean", "minima_clean")}
    sigma = np.empty((len(ids), len(SMOOTH_SCALES), n_real), dtype=np.float32)
    missing = []
    for k, nid in enumerate(ids):
        p = SHARD_DIR / f"node_{nid:04d}.npz"
        if not p.exists():
            missing.append(int(nid))
            continue
        d = np.load(p)
        for v in out:
            out[v][k] = d[v]
        sigma[k] = d["sigma"]
    if missing:
        raise SystemExit(f"[map_stats_merge] {len(missing)} shards missing: {missing[:10]}...")
    np.savez(out_path, node_ids=ids, nu=0.5 * (NU_BINS[1:] + NU_BINS[:-1]),
             smoothing_arcmin=np.array(SMOOTH_SCALES), sigma=sigma,
             plane_weights_full=full_nz_plane_weights(), n_eff_full=N_EFF_FULL,
             sigma_e_full=SIGMA_E_FULL, **out)
    print(f"[map_stats_merge] {len(ids)} shards -> {out_path}")


def run_bind_map_stats(n_real=50, out_path=LC / "bind_map_stats.npz"):
    t0 = time.time()
    wfull = full_nz_plane_weights()
    ids = node_ids_all()
    n_nu = len(NU_BINS) - 1
    shapes = (len(ids), len(SMOOTH_SCALES), n_real, n_nu)
    out = {v: np.empty(shapes, dtype=np.float32)
           for v in ("peaks_noisy", "minima_noisy", "peaks_clean", "minima_clean")}
    sigma = np.empty((len(ids), len(SMOOTH_SCALES), n_real), dtype=np.float32)
    for k, nid in enumerate(ids):
        kap = np.load(RUNS / f"run_{nid:04d}" / "kappa_maps.npz")["kappa"][:n_real]
        full = np.einsum("p,rpxy->rxy", wfull, kap.astype(np.float64))[:, None]
        res_n = _stats_for_maps(full, N_EFF_FULL, SIGMA_E_FULL, noise_seed=nid)
        res_c = _stats_for_maps(full, None, SIGMA_E_FULL)
        out["peaks_noisy"][k] = res_n["peak_counts_real"][:, :, 0]
        out["minima_noisy"][k] = res_n["minima_counts_real"][:, :, 0]
        out["peaks_clean"][k] = res_c["peak_counts_real"][:, :, 0]
        out["minima_clean"][k] = res_c["minima_counts_real"][:, :, 0]
        import sys
        sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
        from bind.inference.stats import _gaussian_smooth
        for s, sc in enumerate(SMOOTH_SCALES):
            for r in range(n_real):
                sigma[k, s, r] = _gaussian_smooth(full[r, 0], sc, 5.0).std()
        if k % 20 == 0:
            print(f"  node {k}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(out_path, node_ids=ids, nu=0.5 * (NU_BINS[1:] + NU_BINS[:-1]),
             smoothing_arcmin=np.array(SMOOTH_SCALES), sigma=sigma,
             plane_weights_full=wfull, n_eff_full=N_EFF_FULL,
             sigma_e_full=SIGMA_E_FULL, **out)
    print(f"[bind_map_stats] {len(ids)} nodes in {time.time()-t0:.0f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twopt", action="store_true")
    ap.add_argument("--des_map_stats", action="store_true")
    ap.add_argument("--bind_map_stats", action="store_true")
    ap.add_argument("--bind_map_stats_node", type=int, default=None,
                    help="compute ONE node's shard (disBatch task)")
    ap.add_argument("--bind_map_stats_merge", action="store_true")
    ap.add_argument("--n_nz_real", type=int, default=50)
    ap.add_argument("--n_real", type=int, default=50)
    args = ap.parse_args()
    if args.twopt:
        run_twopt(args.n_nz_real)
    if args.des_map_stats:
        run_des_map_stats()
    if args.bind_map_stats_node is not None:
        run_bind_map_stats_node(args.bind_map_stats_node, args.n_real)
    if args.bind_map_stats_merge:
        run_bind_map_stats_merge(n_real=args.n_real)
    if args.bind_map_stats:
        run_bind_map_stats(args.n_real)
    if not (args.twopt or args.des_map_stats or args.bind_map_stats
            or args.bind_map_stats_node is not None or args.bind_map_stats_merge):
        raise SystemExit("pass --twopt / --des_map_stats / --bind_map_stats[_node/_merge]")


if __name__ == "__main__":
    main()
