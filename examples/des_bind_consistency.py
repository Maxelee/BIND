#!/usr/bin/env python
"""D4 (docs/tsz_des_data_plan.md §3): DES Y3 consistency tests vs the
kSZ-selected SB35 subspace.

2pt leg (--twopt, quantitative): per-node chi^2 of the D3 forward-modelled
xi_pm (theory Cl x S_ab(ell), full 400-point data vector, official COVMAT
xip+xim block) in two mandatory variants:
  (i)  raw fixed-cosmology (TNG300 sigma8=0.8159 — a cosmology test as much
       as a feedback test, kept for honesty);
  (ii) amplitude-marginalized: model -> B * xi_pred with B=A^2 free per node
       (Ckk ∝ A^2, A ~ sigma8*Om^0.5 — the wl_sz_cosmo_anchor pattern),
       analytic profile-likelihood minimum.
Consistency threshold: chi2 < dof + 2*sqrt(2*dof) — the SAME convention as
the kSZ node classification (ksz_consistent_nodes.npz readme), so the
cross-probe table is apples-to-apples.

Model error: a diagonal term from cl_err (std-of-mean over 50 realizations)
propagated through the same Hankel — conservative (ignores the r~0.98
paired-trace deflation); reported but ~negligible vs the DES COVMAT.

Map leg (--maps, consistency-check-grade): nu-space peak/minima counts
(nu_norm='map' — per-map sigma normalization divides out the GLIMPSE/Wiener
amplitude suppression to first order), DES cov from the 45-patch scatter
(x45 to the mean, i.e. cov(mean)=cov_patch/45), BIND per-node realization
scatter; systematic band = (GLIMPSE vs Wiener) ⊕ (noisy vs clean BIND).

Outputs: KS/lightcone/des_consistency_{twopt,maps}.npz + printed summary.
Figures live in the D4/M7 figure script, not here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"

CHI2_SLACK = 2.0  # chi2 < dof + CHI2_SLACK*sqrt(2*dof)


def consistent(chi2, dof):
    return chi2 < dof + CHI2_SLACK * np.sqrt(2.0 * dof)


def load_sets():
    d = np.load(LC / "ksz_consistent_nodes.npz")
    return (d["node_ids_all"], set(int(i) for i in d["node_ids_bgs110"]),
            set(int(i) for i in d["node_ids_bgs1125"]))


def run_twopt():
    d2 = np.load(LC / "des_y3_2pt.npz")
    pred = np.load(LC / "bind_des_xipred.npz")
    ids, ksz110, ksz1125 = load_sets()
    assert np.array_equal(ids, pred["node_ids"])

    data = np.concatenate([d2["xip_value"], d2["xim_value"]])       # (400,)
    cov = d2["cov_full"][0:400, 0:400]

    def model_vec(xp, xm):
        return np.concatenate([xp.ravel(), xm.ravel()])

    m_nodes = np.array([model_vec(pred["xip_nodes"][k], pred["xim_nodes"][k])
                        for k in range(len(ids))])                   # (253, 400)
    m_fid = model_vec(pred["xip_fid"], pred["xim_fid"])
    m_th = model_vec(pred["xip_theory"], pred["xim_theory"])

    # n(z)-realisation systematic (fiducial): rms spread of the prediction
    nz_spread = np.concatenate([
        pred["xip_nzreal_fid"].std(axis=0).ravel(),
        pred["xim_nzreal_fid"].std(axis=0).ravel()])

    cov_use = cov + np.diag(nz_spread ** 2)
    cinv = np.linalg.inv(cov_use)

    dof_raw = len(data)
    dof_marg = len(data) - 1

    def chi2_raw(m):
        r = data - m
        return float(r @ cinv @ r)

    def chi2_marg(m):
        num = float(data @ cinv @ m)
        den = float(m @ cinv @ m)
        B = max(num / den, 0.0)
        r = data - B * m
        return float(r @ cinv @ r), B

    c_raw = np.array([chi2_raw(m) for m in m_nodes])
    marg = [chi2_marg(m) for m in m_nodes]
    c_marg = np.array([x[0] for x in marg])
    B_marg = np.array([x[1] for x in marg])

    res = {
        "node_ids": ids, "chi2_raw": c_raw, "chi2_marg": c_marg, "B_marg": B_marg,
        "chi2_raw_fid": chi2_raw(m_fid), "chi2_marg_fid": chi2_marg(m_fid)[0],
        "B_fid": chi2_marg(m_fid)[1],
        "chi2_raw_theory": chi2_raw(m_th), "chi2_marg_theory": chi2_marg(m_th)[0],
        "dof_raw": dof_raw, "dof_marg": dof_marg,
        "nz_spread_frac_of_err": float(np.median(nz_spread / np.sqrt(np.diag(cov)))),
    }

    summary = {}
    for tag, cval, dof in [("raw", c_raw, dof_raw), ("marg", c_marg, dof_marg)]:
        ok = consistent(cval, dof)
        oksets = {
            "all": int(ok.sum()),
            "of_ksz110": int(sum(ok[k] for k, i in enumerate(ids) if int(i) in ksz110)),
            "of_ksz1125": int(sum(ok[k] for k, i in enumerate(ids) if int(i) in ksz1125)),
        }
        summary[tag] = {
            "chi2_min": float(cval.min()), "chi2_med": float(np.median(cval)),
            "n_consistent": oksets, "threshold": float(dof + CHI2_SLACK * np.sqrt(2 * dof)),
            "P_des_consistent": oksets["all"] / len(ids),
            "P_des_given_ksz110": oksets["of_ksz110"] / max(len(ksz110), 1),
            "P_des_given_ksz1125": oksets["of_ksz1125"] / max(len(ksz1125), 1),
        }
    res["summary"] = json.dumps(summary)
    np.savez(LC / "des_consistency_twopt.npz", **res)
    print(json.dumps(summary, indent=2))
    print(f"fid: chi2_raw/dof={res['chi2_raw_fid']/dof_raw:.2f} "
          f"chi2_marg/dof={res['chi2_marg_fid']/dof_marg:.2f} B_fid={res['B_fid']:.3f}")
    print(f"theory(S=1): raw/dof={res['chi2_raw_theory']/dof_raw:.2f} "
          f"marg/dof={res['chi2_marg_theory']/dof_marg:.2f}")
    print(f"median nz-systematic / DES err = {res['nz_spread_frac_of_err']:.3f}")
    print(f"B_marg range over nodes: [{B_marg.min():.3f}, {B_marg.max():.3f}]")


def run_maps():
    des = np.load(LC / "des_map_stats.npz")
    bind = np.load(LC / "bind_map_stats.npz")
    ids, ksz110, ksz1125 = load_sets()
    assert np.array_equal(ids, bind["node_ids"])
    nu = des["nu"]
    out = {"nu": nu, "node_ids": ids, "smoothing_arcmin": des["smoothing_arcmin"]}
    summary = {}
    for s, sc in enumerate(des["smoothing_arcmin"]):
        # data: mean over patches; cov of the MEAN from patch scatter
        g = des["glimpse_full_peaks_real"][s].astype(np.float64)    # (45, nu)
        w = des["wiener_full_peaks_real"][s].astype(np.float64)
        npatch = g.shape[0]
        # drop empty nu bins (keep where mean count >= 1 in data)
        keep = g.mean(0) >= 1.0
        dmean = g.mean(0)[keep]
        # R0: inverse-Hartlap inflation of the patch-sample covariance
        p_eff = int(keep.sum())
        dcov = ((npatch - 1) / max(npatch - p_eff - 2, 1)) \
            * np.cov(g[:, keep].T) / npatch
        filt_sys = (g.mean(0) - w.mean(0))[keep]                    # GLIMPSE-Wiener
        cov_use = dcov + np.diag(filt_sys ** 2)
        dof = int(keep.sum())

        chi2 = np.empty((2, len(ids)))
        for v, var in enumerate(("peaks_noisy", "peaks_clean")):
            b = bind[var][:, s].astype(np.float64)                  # (253, n_real, nu)
            bmean = b.mean(1)[:, keep]
            bcov_diag = b.std(1)[:, keep] ** 2 / b.shape[1]
            for k in range(len(ids)):
                c = cov_use + np.diag(bcov_diag[k])
                r = dmean - bmean[k]
                chi2[v, k] = float(r @ np.linalg.solve(c, r))
        out[f"chi2_noisy_s{s}"] = chi2[0]
        out[f"chi2_clean_s{s}"] = chi2[1]
        out[f"data_mean_s{s}"] = dmean
        out[f"data_err_s{s}"] = np.sqrt(np.diag(cov_use))
        out[f"keep_s{s}"] = keep
        ok = consistent(chi2[0], dof)
        summary[f"peaks_{sc:g}am"] = {
            "dof": dof,
            "n_consistent_noisy": int(ok.sum()),
            "n_consistent_clean": int(consistent(chi2[1], dof).sum()),
            "of_ksz110": int(sum(ok[k] for k, i in enumerate(ids) if int(i) in ksz110)),
            "of_ksz1125": int(sum(ok[k] for k, i in enumerate(ids) if int(i) in ksz1125)),
            "chi2_med_noisy": float(np.median(chi2[0])),
        }
    out["summary"] = json.dumps(summary)
    np.savez(LC / "des_consistency_maps.npz", **out)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twopt", action="store_true")
    ap.add_argument("--maps", action="store_true")
    a = ap.parse_args()
    if a.twopt:
        run_twopt()
    if a.maps:
        run_maps()


if __name__ == "__main__":
    main()
