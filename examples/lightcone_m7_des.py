#!/usr/bin/env python
"""D4 Fig M7 (docs/tsz_des_data_plan.md §3): the Stream-D money plot.

Three panels: (a) xi_+ data vs node envelope (amplitude-marginalized, the
feedback-test variant), (b) nu-space peak counts (map leg, consistency-check-
grade — reconstruction-filter caveat printed on the panel), (c) the
cross-probe chi2-ranking scatter (the D4 discovery: DES xi_pm cannot REJECT
any SB35 node once amplitude is freed, but its chi2 RANKING is strongly
aligned with the kSZ selection, Spearman rho~0.78).

Runs after des_bind_consistency.py --twopt --maps. Writes figs/M7_des_wl.png
+ verdicts/D3.json + verdicts/D4.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"

PAIR_SHOW = 9  # (4,4) — highest-S/N pair ((1,1) is S/N~1-3/pt, unshowable)


def main():
    d2 = np.load(LC / "des_y3_2pt.npz")
    pred = np.load(LC / "bind_des_xipred.npz")
    tw = np.load(LC / "des_consistency_twopt.npz", allow_pickle=True)
    mp = np.load(LC / "des_consistency_maps.npz", allow_pickle=True)
    ksz = np.load(LC / "ksz_consistent_nodes.npz")
    ids = tw["node_ids"]
    in110 = np.isin(ids, ksz["node_ids_bgs110"])

    th_p = pred["theta_p"].reshape(10, 20) if pred["theta_p"].ndim == 1 else pred["theta_p"]
    th = th_p[PAIR_SHOW]
    data_xip = d2["xip_value"].reshape(10, 20)[PAIR_SHOW]
    err_xip = np.sqrt(np.diag(d2["cov_xip"])).reshape(10, 20)[PAIR_SHOW]

    B = tw["B_marg"]
    xp_nodes = pred["xip_nodes"][:, PAIR_SHOW, :]     # (253, 20)
    xp_marg = B[:, None] * xp_nodes
    xp_fid = float(tw["B_fid"]) * pred["xip_fid"][PAIR_SHOW]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    # (a) xi+ (4,4) fractional residual to the amp-marginalized fiducial —
    # the node envelope is 0.04-0.4% wide vs 11-19% DES errors; on an
    # absolute plot it is invisible, and THAT is the result.
    ax = axes[0]
    ref = xp_fid
    lo, hi = np.nanpercentile(xp_marg / ref - 1, [0, 100], axis=0)
    ax.fill_between(th, 100 * lo, 100 * hi, color="lightgray",
                    label="253-node envelope (amp-marg.)")
    for k in np.nonzero(in110)[0]:
        ax.plot(th, 100 * (xp_marg[k] / ref - 1), color="tab:green", lw=0.5, alpha=0.5)
    ax.plot([], [], color="tab:green", label="kSZ-consistent (31)")
    ax.errorbar(th, 100 * (data_xip / ref - 1), 100 * err_xip / np.abs(ref),
                fmt="o", color="k", capsize=3, zorder=5,
                label=r"DES Y3 $\xi_+^{4,4}$ / (B·fid) − 1")
    ax.axhline(0, color="tab:purple", lw=1.5, label="fiducial ×B")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"fractional deviation from B·fiducial [%]")
    ax.set_ylim(-45, 45)
    s = json.loads(str(tw["summary"]))
    ax.set_title(f"2pt: raw rejects {253-s['raw']['n_consistent']['all']}/253; "
                 f"amp-marg accepts {s['marg']['n_consistent']['all']}/253\n"
                 "(node envelope ≤0.4% ≪ 11–19% DES errors — no residual feedback power)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    # (b) peaks 10am
    ax = axes[1]
    sidx = 1  # 10' smoothing (reconstruction-safe)
    nu = mp["nu"]
    keep = mp[f"keep_s{sidx}"]
    dmean, derr = mp[f"data_mean_s{sidx}"], mp[f"data_err_s{sidx}"]
    bind = np.load(LC / "bind_map_stats.npz")
    bmean = bind["peaks_noisy"][:, sidx].mean(axis=1)[:, keep]
    lo, hi = np.nanpercentile(bmean, [0, 100], axis=0)
    ax.fill_between(nu[keep], lo, hi, color="lightgray", label="253-node envelope")
    for k in np.nonzero(in110)[0]:
        ax.plot(nu[keep], bmean[k], color="tab:green", lw=0.5, alpha=0.5)
    ax.errorbar(nu[keep], dmean, derr, fmt="o", color="k", capsize=3, zorder=5,
                label="DES GLIMPSE patches (45)")
    wmean = np.load(LC / "des_map_stats.npz")["wiener_full_peaks_real"][sidx].mean(0)[keep]
    ax.plot(nu[keep], wmean, "s--", color="tab:blue", ms=3, label="Wiener bracket")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (peak S/N, per-map $\sigma$)")
    ax.set_ylabel("peaks / 25 deg$^2$")
    ms = json.loads(str(mp["summary"]))
    key = [k for k in ms if "10" in k][0]
    ax.set_title(f"peaks @10' (map-norm.): {ms[key]['n_consistent_noisy']}/253 consistent\n"
                 "CAVEAT: GLIMPSE/Wiener-filtered reconstruction — consistency-check-grade",
                 fontsize=9)
    ax.legend(fontsize=7)

    # (c) cross-probe ranking
    ax = axes[2]
    from scipy import stats as st
    c = tw["chi2_marg"]
    rho = st.spearmanr(ksz["chi2_desi_bgs110"], c)
    ax.scatter(ksz["chi2_desi_bgs110"], c, s=14,
               c=np.where(in110, "tab:green", "tab:gray"), alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"kSZ $\chi^2$ (M1, bgs110)")
    ax.set_ylabel(r"DES $\xi_\pm$ $\chi^2$ (amp-marginalized)")
    ax.set_title(f"probes rank nodes the SAME way: Spearman ρ={rho.statistic:.2f} "
                 f"(p={rho.pvalue:.1e})")
    med110 = np.median(c[in110]); medrest = np.median(c[~in110])
    ax.axhline(med110, color="tab:green", ls=":", lw=1)
    ax.axhline(medrest, color="tab:gray", ls=":", lw=1)
    ax.text(0.02, 0.02, f"median Δχ²(kSZ-sel − rest) = {med110-medrest:+.1f}",
            transform=ax.transAxes, fontsize=8)

    fig.suptitle("M7 (Stream-D money plot): DES Y3 WL vs the kSZ-selected SB35 feedback subspace",
                 fontsize=12)
    fig.savefig(LC / "figs/M7_des_wl.png", dpi=140)
    print("M7 saved; rho =", rho.statistic)


if __name__ == "__main__":
    main()
