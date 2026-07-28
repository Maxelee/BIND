#!/usr/bin/env python
"""R5a/b/d (docs/p4c_referee_hardening_plan.md): the CIB/dust systematic done
properly, plus the variant table and the random-null investigation.

(a) Correlated CIB covariance: Sigma_CIB = empirical covariance of the 11
    fine-res variant curves about the primary (deproj beta=1.7) on the xb
    grid — replaces the diagonal half-band (which ignores the strong
    inter-aperture correlation of CIB residuals). Also the dust-template
    alternative: cov += t t^T with t = baseline - cib1.7 (a 1-sigma
    amplitude marginalization of the dust-leakage shape). chi2/count
    stability across {diag, correlated, template} is the R5 gate (±30%).
(b) Variant table: EBV<0.15, z-window 0.45-0.9, rotated-null pass-through —
    each as a fraction of the total per-column error.
(d) Random-null: per-object large-aperture CAPs vs Galactic latitude and
    dec (footprint proxies) — is the -3sigma theta>=4.75' negative bias a
    Galactic-dust residual?

Writes R5_cib_cov.npz (consumed by T3 when present), figs/R5_cib.png,
verdicts/R5.json (partial — R5c Liu-sample comparison appended separately).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Products root (Round-2 T3, docs/paper_improvement_plan.md): env-var
# override only (straight-line script, no argparse); behavior with
# $BIND_KSZ_PRODUCTS unset is byte-identical to before.
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", "/mnt/home/mlee1/ceph/bind_science/ksz_confront"))
LC = KS / "lightcone"
VARIANTS = ["baseline", "cib1.0", "cib1.2", "cib1.4", "cib1.6", "cib1.7",
            "cib1.8", "cib2.0", "cib1.7_24", "cibdBeta", "cibdBetadT"]
CHI2_SLACK = 2.0


def load(name):
    p = LC / f"T2f_{name}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    fcib = load("lrg_z0406_cib1.7")
    i_xb = np.nonzero(fcib["theta_kind"] == "xb")[0]
    xb = fcib["theta_value"][i_xb].astype(float)

    # ---- (a) correlated CIB covariance -----------------------------------
    curves = {}
    for v in VARIANTS:
        d = load(f"lrg_z0406_{v}")
        if d is not None:
            curves[v] = d["mean"][i_xb]
    arr = np.array([curves[v] for v in curves])          # (11, 18)
    ref = curves["cib1.7"]
    dev = arr - ref[None, :]
    cov_cib = (dev.T @ dev) / max(len(curves) - 1, 1)    # about the primary
    t_dust = curves["baseline"] - ref                    # dust-leak template

    np.savez(LC / "R5_cib_cov.npz", xb=xb, cov_cib=cov_cib, t_dust=t_dust,
             variant_names=np.array(list(curves)), deviations=dev)

    # ---- chi2 stability across treatments --------------------------------
    dd = np.load(LC / "act_ycap_lrg_real.npz", allow_pickle=True)
    hd = np.load(LC / "R2_hod_model_curves.npz")
    ksz = np.load(LC / "ksz_consistent_nodes.npz")
    BIND_PIX_AREA = 0.29296875 ** 2
    node_ids = hd["node_ids"]
    in110 = np.isin(node_ids, ksz["node_ids_bgs110"])
    valid = dd["valid_cols"]
    iv = np.nonzero(valid)[0]
    dof = len(iv)
    h_data = 99 / (100 - dof - 2)
    d_xb = dd["mean_xb_cib17"]
    base_cov = h_data * dd["cov_jk_xb_cib17"][np.ix_(iv, iv)]
    # resample systematic (50% of the R1 correction), separate from CIB
    r1 = np.load(LC / "R1_resample_correction.npz")
    b_res = np.interp(xb * float(dd["theta200_data_arcmin"]),
                      r1["theta_arcmin"], r1["bias_pixwin"])
    sig_res = 0.5 * np.abs(b_res) * np.abs(d_xb)
    corr = 1.0 + b_res                                  # data already corrected in dd

    sb_real = hd["nodes_kcal_f0.08"] * BIND_PIX_AREA
    sb_mean = np.nanmean(sb_real, axis=1)
    sat_sys = 0.5 * np.abs(hd["nodes_kcal_f0.12"].mean(1)
                           - hd["nodes_kcal_f0.04"].mean(1)) * BIND_PIX_AREA

    # CIB treatments (all applied to the resample-corrected frame /corr)
    cov_cib_corr = cov_cib / np.outer(corr, corr)
    t_dust_corr = t_dust / corr
    sig_diag = dd["sig_cib_xb"]                          # current T3 (incl sig_res)
    treatments = {
        "diag_halfband": np.diag(sig_diag[iv] ** 2),
        "correlated": cov_cib_corr[np.ix_(iv, iv)] + np.diag(sig_res[iv] ** 2),
        "dust_template": (np.outer(t_dust_corr[iv], t_dust_corr[iv])
                          + np.diag(sig_res[iv] ** 2)),
    }

    def counts(cib_block):
        ok = np.zeros(len(node_ids), dtype=bool)
        chi2 = np.zeros(len(node_ids))
        for k in range(len(node_ids)):
            n_r = sb_real.shape[1]
            h_b = (n_r - 1) / (n_r - dof - 2)
            ccov = h_b * np.cov(sb_real[k][:, iv].T) / n_r
            cov = base_cov + cib_block + ccov + np.outer(sat_sys[k][iv], sat_sys[k][iv])
            r = d_xb[iv] - sb_mean[k][iv]
            chi2[k] = float(r @ np.linalg.solve(cov, r))
            ok[k] = chi2[k] < dof + CHI2_SLACK * np.sqrt(2 * dof)
        return ok, chi2

    res = {}
    for tag, blk in treatments.items():
        ok, chi2 = counts(blk)
        res[tag] = {"n": int(ok.sum()), "P": float(ok.mean()),
                    "P_given_ksz": float(ok[in110].mean()),
                    "chi2_med": float(np.median(chi2))}
    n_ref = res["correlated"]["n"]
    stable = all(abs(res[t]["n"] - n_ref) <= 0.3 * max(n_ref, 1) for t in res)

    # ---- (b) variant table -----------------------------------------------
    err_tot = np.sqrt(np.diag(base_cov) + np.diag(cov_cib_corr[np.ix_(iv, iv)])
                      + sig_res[iv] ** 2)
    table = {}
    ebv = load("lrg_z0406_ebv015")
    if ebv is not None:
        sh = (ebv["mean"][i_xb] - curves["baseline"]) / corr
        table["ebv015_shift_over_err"] = [float(x) for x in (sh[iv] / err_tot)]
    zw = load("lrg_z04509_baseline")
    if zw is not None:
        sh = (zw["mean"][i_xb] - curves["baseline"]) / corr
        table["zwindow_shift_over_err"] = [float(x) for x in (sh[iv] / err_tot)]
        table["zwindow_note"] = "sample z-mix differs from the mock (z=0.503) — robustness fraction, not a bias"
    rot = load("rotated_null_z0406")
    if rot is not None:
        rn = rot["mean"][i_xb] / corr
        table["rotated_null_over_err"] = [float(x) for x in (rn[iv] / err_tot)]

    # ---- (d) random-null investigation -----------------------------------
    nul = load("random_null_z0406")
    null_diag = {}
    if nul is not None and "per_obj_cap" in nul.files:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        cap6 = nul["per_obj_cap"][:, -1]                 # theta=6'
        okm = nul["per_obj_ok"][:, -1].astype(bool)
        c = SkyCoord(ra=nul["per_obj_ra"][okm] * u.deg,
                     dec=nul["per_obj_dec"][okm] * u.deg, frame="icrs")
        gb = np.abs(c.galactic.b.deg)
        v = cap6[okm]
        bins = np.percentile(gb, np.linspace(0, 100, 7))
        prof = []
        for i in range(6):
            m = (gb >= bins[i]) & (gb < bins[i + 1] + (i == 5))
            prof.append([float(0.5 * (bins[i] + bins[i + 1])),
                         float(np.mean(v[m])), float(np.std(v[m]) / np.sqrt(m.sum()))])
        null_diag["cap6_vs_abs_gal_b"] = prof
        lo, hi = prof[0][1], prof[-1][1]
        null_diag["low_b_minus_high_b_sigma"] = float(
            (lo - hi) / np.hypot(prof[0][2], prof[-1][2]))

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ax = axes[0]
    C = cov_cib[np.ix_(iv, iv)]
    R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(dof)); ax.set_xticklabels([f"{x:.2f}" for x in xb[iv]], fontsize=7)
    ax.set_yticks(range(dof)); ax.set_yticklabels([f"{x:.2f}" for x in xb[iv]], fontsize=7)
    ax.set_title("CIB variant correlation matrix (fit cols)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    labels = list(res)
    ax.bar(np.arange(len(labels)) - 0.18, [res[t]["P"] for t in labels], 0.36,
           color="0.6", label="P(tSZ)")
    ax.bar(np.arange(len(labels)) + 0.18, [res[t]["P_given_ksz"] for t in labels],
           0.36, color="tab:green", label="P(tSZ|kSZ)")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8)
    ax.set_title(f"counts vs CIB treatment (n: "
                 f"{', '.join(str(res[t]['n']) for t in labels)})")

    ax = axes[2]
    if null_diag:
        p = np.array(null_diag["cap6_vs_abs_gal_b"])
        ax.errorbar(p[:, 0], p[:, 1] * 1e7, p[:, 2] * 1e7, fmt="o-", capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"|Galactic b| [deg]")
        ax.set_ylabel(r"random-null CAP(6') [$\times10^7$]")
        ax.set_title(f"null vs |b|: low-high = "
                     f"{null_diag['low_b_minus_high_b_sigma']:+.1f}σ")
    fig.suptitle("R5a/b/d: correlated CIB treatment + variant table + null diagnosis", fontsize=11)
    fig.savefig(LC / "figs/R5_cib.png", dpi=140)

    v = {
        "phase": "R5-partial", "pass": bool(stable),
        "metrics": {
            "counts_by_treatment": res,
            "gate_stability_pm30pct": stable,
            "cib_corr_offdiag_mean": float(np.mean(R[~np.eye(dof, dtype=bool)])),
            "variant_table": table,
            "null_diagnosis": null_diag,
        },
        "figs": ["figs/R5_cib.png"],
        "notes": "Primary treatment going forward: correlated Sigma_CIB (empirical "
                 "covariance of the 11 fine-res deprojection variants about the "
                 "deproj-1.7 primary) + diagonal resampling systematic. "
                 "R5c (Liu-sample reproduction) appended when the battery lands.",
        "next": "R5c Liu comparison; adopt correlated treatment in T3",
    }
    with open(LC / "verdicts/R5.json", "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(res, indent=2))
    print("stable ±30%:", stable, "| mean offdiag corr:",
          round(float(np.mean(R[~np.eye(dof, dtype=bool)])), 3))
    if null_diag:
        print("null low-b vs high-b:", round(null_diag["low_b_minus_high_b_sigma"], 1), "sigma")


if __name__ == "__main__":
    main()
