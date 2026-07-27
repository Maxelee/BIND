#!/usr/bin/env python
"""Capstone X (docs/tsz_des_data_plan.md §4): the multi-probe subspace figure.

Three panels sharing one node coloring {kSZ∩tSZ survivors, kSZ-only, rest}:
  (a) M2R — real ACT DR6 y-CAP at DESI LRGs vs the BIND envelope;
  (b) M7  — DES Y3 xi_+(4,4) amplitude-marginalized fractional residuals;
  (c) M6  — the top-2 kSZ-constrained parameter plane (IMFslope vs
      log10 WindEnergyIn1e51erg) with the multi-probe survivors highlighted.

Headline the caption must state: does the kSZ-selected feedback subspace
survive tSZ and WL simultaneously, and which parameter directions do the
survivors occupy?

Writes figs/X_multiprobe.png + verdicts/X.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"
DESIGN = Path("/mnt/home/mlee1/ceph/bind_sb35/design")

BIND_PIX_AREA = 0.29296875 ** 2
N_XB = 18


def main():
    from bind.params import PARAM_NAMES, PARAM_LOG_FLAG

    ksz = np.load(LC / ("ksz_consistent_nodes_r6.npz" if (LC / "ksz_consistent_nodes_r6.npz").exists() else "ksz_consistent_nodes.npz"))
    tsz = np.load(LC / "tsz_consistent_nodes.npz")
    tw = np.load(LC / "des_consistency_twopt.npz", allow_pickle=True)
    ids = ksz["node_ids_all"]
    in_ksz = np.isin(ids, ksz["node_ids_bgs110"])
    in_tsz = np.isin(ids, tsz["node_ids_tsz_consistent"])
    surv = in_ksz & in_tsz
    kszonly = in_ksz & ~in_tsz

    colors = np.where(surv, "tab:green", np.where(kszonly, "gold", "lightgray"))
    zorder = np.where(surv, 5, np.where(kszonly, 4, 2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    # ---------------- (a) M2R (revised xb-grid convention) ----------------
    ax = axes[0]
    b = np.load(LC / "lrgy_beam_lightcone.npz", allow_pickle=True)
    xb_grid = b["theta_value"][:N_XB]
    sb_mean = b["sb35_mean_lrg"][:, :N_XB] * BIND_PIX_AREA
    fid = b["fid_mean_lrg"][:N_XB] * BIND_PIX_AREA
    dd = np.load(LC / "act_ycap_lrg_real.npz", allow_pickle=True)
    order = np.argsort(zorder)
    for k in order:
        ax.plot(xb_grid, sb_mean[k] * 1e6, color=colors[k], lw=0.6,
                alpha=0.9 if surv[k] or kszonly[k] else 0.35, zorder=zorder[k])
    ax.plot(xb_grid, fid * 1e6, color="tab:purple", lw=2, zorder=6, label="fiducial (TNG300)")
    dv = dd["valid_cols"]
    derr = np.sqrt(dd["err_jk_xb_cib17"] ** 2 + dd["sig_cib_xb"] ** 2)
    ax.errorbar(dd["xb"][dv], dd["mean_xb_cib17"][dv] * 1e6, derr[dv] * 1e6,
                fmt="o", color="k", capsize=3, zorder=7,
                label="ACT DR6 deproj-CIB × DESI LRG (T2f)")
    hi = ~dv & (dd["xb"] > 1.4)
    ax.errorbar(dd["xb"][hi], dd["mean_xb_cib17"][hi] * 1e6,
                dd["err_jk_xb_cib17"][hi] * 1e6, fmt="o", mfc="none", color="k",
                alpha=0.5, capsize=2, label="beyond 1-halo range")
    ax.axvline(1.4, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$x_b=\theta_d/\theta_{200}$")
    ax.set_ylabel(r"CAP $y$-flux [$y\,\mathrm{arcmin}^2\times10^6$]")
    ax.set_title("(a) real tSZ: ACT y-CAP vs SB35 envelope")
    ax.legend(fontsize=7, loc="upper left")

    # ---------------- (b) M7 xi+ residuals ------------------------------
    ax = axes[1]
    pred = np.load(LC / "bind_des_xipred.npz")
    d2 = np.load(LC / "des_y3_2pt.npz")
    th = pred["theta_p"].reshape(10, 20)[9]
    data = d2["xip_value"].reshape(10, 20)[9]
    err = np.sqrt(np.diag(d2["cov_xip"])).reshape(10, 20)[9]
    B = tw["B_marg"]
    ref = float(tw["B_fid"]) * pred["xip_fid"][9]
    xp_marg = B[:, None] * pred["xip_nodes"][:, 9, :]
    for k in order:
        ax.plot(th, 100 * (xp_marg[k] / ref - 1), color=colors[k], lw=0.6,
                alpha=0.9 if surv[k] or kszonly[k] else 0.35, zorder=zorder[k])
    ax.errorbar(th, 100 * (data / ref - 1), 100 * err / np.abs(ref), fmt="o",
                color="k", capsize=3, zorder=7, label=r"DES Y3 $\xi_+^{4,4}$")
    ax.axhline(0, color="tab:purple", lw=1.5, zorder=6)
    ax.set_xscale("log")
    ax.set_ylim(-40, 40)
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"deviation from B·fiducial [%]")
    from scipy import stats as st
    rho = st.spearmanr(ksz["chi2_desi_bgs110"], tw["chi2_marg"])
    ax.set_title(f"(b) DES WL: amp-marg. $\\xi_+$ — envelope ≤0.4%,\n"
                 f"but $\\chi^2$ ranking tracks kSZ (ρ={rho.statistic:.2f})", fontsize=9)
    ax.legend(fontsize=7)

    # ---------------- (c) parameter plane -------------------------------
    ax = axes[2]
    sobol = np.load(DESIGN / "astro_params_sobol.npy")     # (256, 35) full vector
    with open(DESIGN / "design.json") as f:
        dj = json.load(f)
    names, aidx = dj["astro_param_names"], dj["astro_param_indices"]
    iIMF = aidx[names.index("IMFslope")]
    iWE = aidx[names.index("WindEnergyIn1e51erg")]
    x = sobol[ids, iIMF]
    y = np.log10(sobol[ids, iWE])
    for cls, lab, c, s in [(surv, f"kSZ∩tSZ survivors ({surv.sum()})", "tab:green", 42),
                           (kszonly, f"kSZ-only ({kszonly.sum()})", "gold", 30),
                           (~in_ksz, "rest", "lightgray", 12)]:
        ax.scatter(x[cls], y[cls], s=s, c=c, edgecolors="k" if s > 20 else "none",
                   linewidths=0.4, label=lab, zorder=5 if s > 20 else 2)
    ax.set_xlabel("IMFslope")
    ax.set_ylabel(r"$\log_{10}$ WindEnergyIn1e51erg")
    ax.set_title("(c) M6 top-2 kSZ-constrained parameter plane")
    ax.legend(fontsize=7)

    fig.suptitle("Capstone X: does the kSZ-selected feedback subspace survive real tSZ + DES WL simultaneously?",
                 fontsize=12)
    fig.savefig(LC / "figs/X_multiprobe.png", dpi=140)

    v = {
        "phase": "X", "pass": True,
        "metrics": {
            "n_ksz110": int(in_ksz.sum()), "n_tsz_consistent": int(in_tsz.sum()),
            "n_survivors_ksz_and_tsz": int(surv.sum()),
            "P_tsz_given_ksz": float(in_tsz[in_ksz].mean()),
            "P_tsz": float(in_tsz.mean()),
            "des_wl": "no binary cut (amp-marg accepts all 253); ranking rho=%.3f" % rho.statistic,
            "survivor_param_medians": {
                "IMFslope": float(np.median(x[surv])) if surv.any() else None,
                "log10_WindEnergy": float(np.median(y[surv])) if surv.any() else None,
                "rest_IMFslope": float(np.median(x[~in_ksz])),
                "rest_log10_WindEnergy": float(np.median(y[~in_ksz])),
            },
        },
        "figs": ["figs/X_multiprobe.png"],
        "notes": "",
        "next": "paper integration",
    }
    with open(KS / "lightcone/verdicts/X.json", "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v["metrics"], indent=2))


if __name__ == "__main__":
    main()
