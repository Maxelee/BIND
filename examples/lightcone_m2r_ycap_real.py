#!/usr/bin/env python
"""T3 (docs/tsz_des_data_plan.md §2), REVISED after the first M2R review: the
corrected map-level tSZ money plot against OUR OWN ACT DR6 y-CAP measurement.

What was wrong with the first M2R (2026-07-28 review, 'normalization issue'):
1. **Aperture discretization**: the data were measured at the map's native
   0.5'/px while BIND shards use 0.29296875'/px. At theta_d <~ 1.6' a CAP
   disk holds <=10 map pixels — the discrete CAP is a DIFFERENT estimator at
   the two scales (the smallest xb columns were empty/garbage). Fix: T2f
   re-measurement with thumbnails resampled at exactly the BIND lux pixel
   (cubic interp of the 1.6'-beam-limited map is lossless), so gates and
   pixel-counting match the shards bit-for-bit in convention.
2. **Grid-convention mix**: data on the fixed-arcmin RAP grid were compared
   to BIND xb·theta200-scaled apertures relabeled via the mean theta200 —
   different measurements. Fix: the primary comparison is now xb-vs-xb
   (both sides per-object-scaled; data theta200 from z & the logM200=13.18
   anchor, BIND from per-halo r200 — anchors agree to 0.15%).
3. **CIB/dust bias**: the baseline ILC is biased low (negative innermost
   points) at LRG positions by unremoved CIB — a known effect. Fix: the
   deproj_cib_1.7 map is the PRIMARY data vector (Liu+2025's own fiducial
   choice); baseline is the labeled secondary; the 11-variant CIB spread
   enters the chi2 covariance as a diagonal systematic (transferred from
   the 0.5' runs via the per-column fine/coarse ratio of cib1.7).
4. **1-halo range**: beyond xb~1.4-2 the real sky contains 2-halo/diffuse y
   from sub-1e13 halos and IGM that BIND does not paint — a physics
   limitation, not a fit failure. Fix: chi2 restricted to xb <= 1.4, the
   SAME 1-halo convention as the kSZ M1 classification, so the cross-probe
   table compares like with like.

Outputs: act_ycap_lrg_real.npz (RAP display products + xb comparison
products), tsz_consistent_nodes.npz, figs/M2R_ycap_real.png, verdicts/T3.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Products root (P4, docs/paper_improvement_plan.md): resolved as
# --products_root > $BIND_KSZ_PRODUCTS > the historical hardcoded path, so
# behavior with neither flag nor env var set is byte-identical to before.
# KS/LC/FIG_DIR are module-level globals (read by load_t2() and others);
# main() re-resolves them from --products_root before anything uses them.
DEFAULT_PRODUCTS_ROOT = "/mnt/home/mlee1/ceph/bind_science/ksz_confront"


def _products_root(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("BIND_KSZ_PRODUCTS", DEFAULT_PRODUCTS_ROOT))


KS = _products_root()
LC = KS / "lightcone"
FIG_DIR = LC / "figs"
REPO = Path("/mnt/home/mlee1/BIND-ksz2")
LIU_NPZ = REPO / "examples/figures_ksz2/tsz_liu2025_official.npz"

BIND_PIX_AREA = 0.29296875 ** 2
N_XB = 18
CHI2_SLACK = 2.0
XB_ONEHALO_MAX = 1.4          # the M1 kSZ 1-halo convention
MIN_USED_FRAC = 0.5           # xb col valid if >50% of galaxies pass gates

ALL_VARIANTS = ["baseline", "cib1.0", "cib1.2", "cib1.4", "cib1.6", "cib1.7",
                "cib1.8", "cib2.0", "cib1.7_24", "cibdBeta", "cibdBetadT"]


def load_t2(name, fine=False):
    p = LC / f"{'T2f' if fine else 'T2'}_{name}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def consistent(chi2, dof):
    return chi2 < dof + CHI2_SLACK * np.sqrt(2.0 * dof)


def cols(d, kind):
    return np.nonzero(d["theta_kind"] == kind)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products_root", default=None,
                    help="root for KS/LC (default: $BIND_KSZ_PRODUCTS or "
                         f"{DEFAULT_PRODUCTS_ROOT})")
    args = ap.parse_args()

    global KS, LC, FIG_DIR
    KS = _products_root(args.products_root)
    LC = KS / "lightcone"
    FIG_DIR = LC / "figs"

    anchors = np.load(LC / "T3_theta200_anchors.npz")
    t200_mock = float(anchors["t200_mock_arcmin"])
    t200_data = float(anchors["t200_data_arcmin"])

    # ---- data: fine-res primary pair + coarse battery --------------------
    fbase = load_t2("lrg_z0406_baseline", fine=True)
    fcib = load_t2("lrg_z0406_cib1.7", fine=True)
    base_c = load_t2("lrg_z0406_baseline")
    cib_c = load_t2("lrg_z0406_cib1.7")
    rand = load_t2("random_null_z0406")
    rot = load_t2("rotated_null_z0406")
    for tag, d in [("T2f baseline", fbase), ("T2f cib1.7", fcib),
                   ("T2 baseline", base_c), ("T2 cib1.7", cib_c)]:
        if d is None:
            raise SystemExit(f"missing: {tag}")

    i_xb_f = cols(fcib, "xb")
    i_rap_c = cols(base_c, "fixed_arcmin")
    xb = fcib["theta_value"][i_xb_f].astype(float)
    th_rap = base_c["theta_value"][i_rap_c].astype(float)

    # ---- discretization correction (finding 1, quantified) ---------------
    i_xb_c = cols(cib_c, "xb")
    with np.errstate(divide="ignore", invalid="ignore"):
        T_disc = fcib["mean"][i_xb_f] / cib_c["mean"][i_xb_c]

    # ---- R1a resampling correction (docs/p4c_referee_hardening_plan.md) --
    # The 0.5'->0.293' cubic-resampling data path biases stacked CAP by
    # -2..-4.5% at theta<1.6' (measured on BIND's own y field at matched
    # halo positions, R1_resample_correction.npz). Divide it out of the
    # fine-res data columns; keep 50% of the correction as a diagonal
    # systematic (added to sig_cib in quadrature below).
    r1_path = LC / "R1_resample_correction.npz"
    theta_cols_arcmin = xb * t200_data
    if r1_path.exists():
        r1 = np.load(r1_path)
        b_resample = np.interp(theta_cols_arcmin, r1["theta_arcmin"],
                               r1["bias_pixwin"])
    else:
        print("[T3] WARNING: R1 correction absent — data uncorrected")
        b_resample = np.zeros(len(xb))

    # ---- CIB systematic on the fine xb grid ------------------------------
    # R1c: if the full fine-res variant suite exists (t2f_fineres disBatch),
    # build the band DIRECTLY from it; else fall back to the coarse-band
    # transfer (pre-R1c behaviour, flagged).
    fine_vars = {v: load_t2(f"lrg_z0406_{v}", fine=True) for v in ALL_VARIANTS}
    fine_vars = {v: d for v, d in fine_vars.items() if d is not None}
    coarse = {v: load_t2(f"lrg_z0406_{v}") for v in ALL_VARIANTS}
    coarse = {v: d for v, d in coarse.items() if d is not None}  # RAP display band
    if len(fine_vars) >= len(ALL_VARIANTS) - 1:
        band_f = np.array([d["mean"][i_xb_f] for d in fine_vars.values()])
        sig_cib = 0.5 * (band_f.max(0) - band_f.min(0))
        cib_band_source = f"fine-res direct ({len(fine_vars)} variants)"
    else:
        band_c = np.array([d["mean"][i_xb_c] for d in coarse.values()])
        with np.errstate(divide="ignore", invalid="ignore"):
            w_frac = (band_c.max(0) - band_c.min(0)) / np.abs(cib_c["mean"][i_xb_c])
        w_frac = np.clip(np.nan_to_num(w_frac, nan=2.0, posinf=2.0), 0.0, 2.0)
        sig_cib = 0.5 * w_frac * np.abs(fcib["mean"][i_xb_f])
        cib_band_source = f"coarse transfer ({len(coarse)} variants) — pre-R1c"
    print(f"[T3] CIB band: {cib_band_source}")

    # ---- BIND side --------------------------------------------------------
    b = np.load(LC / "lrgy_beam_lightcone.npz", allow_pickle=True)
    fid_mean = b["fid_mean_lrg"][:N_XB] * BIND_PIX_AREA
    fid_real = b["fid_real_lrg"][:, :N_XB] * BIND_PIX_AREA
    sb_mean = b["sb35_mean_lrg"][:, :N_XB] * BIND_PIX_AREA
    sb_real = b["sb35_real_lrg"][:, :, :N_XB] * BIND_PIX_AREA
    node_ids = b["node_ids"]

    # R2 final (docs/p4c_referee_hardening_plan.md): when the per-node
    # HOD-population stacks exist, they REPLACE the pure-central/per-halo-rule
    # shards as the model — anchor aperture rule + kappa-calibrated satellite
    # mix at f_eff=0.08, with the f_eff 0.04..0.12 span as a PER-NODE fully-
    # correlated satellite systematic (the fiducial satellite template cannot
    # be shared across nodes: +/-35-39% node spread).
    hod_path = LC / "R2_hod_model_curves.npz"
    sat_sys = fid_sat_sys = None
    model_source = "P5 shards (pure centrals, per-halo theta200 rule)"
    if hod_path.exists():
        hd = np.load(hod_path)
        assert np.array_equal(hd["node_ids"], node_ids)
        sb_real = hd["nodes_kcal_f0.08"] * BIND_PIX_AREA
        sb_mean = np.nanmean(sb_real, axis=1)
        fid_real = hd["fid_kcal_f0.08"] * BIND_PIX_AREA
        fid_mean = np.nanmean(fid_real, axis=0)
        sat_sys = 0.5 * np.abs(hd["nodes_kcal_f0.12"].mean(1)
                               - hd["nodes_kcal_f0.04"].mean(1)) * BIND_PIX_AREA
        fid_sat_sys = 0.5 * np.abs(hd["fid_kcal_f0.12"].mean(0)
                                   - hd["fid_kcal_f0.04"].mean(0)) * BIND_PIX_AREA
        model_source = ("R2 per-node HOD population (anchor rule, "
                        "kappa-calibrated f_eff=0.08 ± 0.04)")
    # R7 (painting-fidelity closure): BIND over-paints CAP-y by +15-18%
    # vs the TNG300-hydro truth lightcone at these exact halos/apertures
    # (mean +16.4%, ~2% uncertainty). Deterministic correction divided out
    # of the model curves; the closure uncertainty joins the budget.
    r7_path = KS / "lightcone/verdicts/R7.json"
    if r7_path.exists() and hod_path.exists():
        import json as _json
        r7 = _json.load(open(r7_path))
        r7cols = r7["metrics"]["fit_columns_xb_0.46_1.25"]
        xb_r7 = np.array([c["xb"] for c in r7cols])
        bias_r7 = np.array([c["frac_bias_pct"] for c in r7cols]) / 100.0
        fid_corr = 1.0 + np.interp(xb, xb_r7, bias_r7)
        sb_real = sb_real / fid_corr[None, None, :]
        sb_mean = sb_mean / fid_corr[None, :]
        fid_real = fid_real / fid_corr[None, :]
        fid_mean = fid_mean / fid_corr
        sat_sys = sat_sys / fid_corr[None, :]
        fid_sat_sys = fid_sat_sys / fid_corr
        model_source += " + R7 fidelity correction (/1.15-1.18)"
    print(f"[T3] model source: {model_source}")

    # ---- chi2 on the shared xb grid, 1-halo range -------------------------
    # R1a correction applied multiplicatively to the fine-res data (raw
    # values retained in the output npz); 50% of the correction joins the
    # diagonal systematic budget.
    corr = 1.0 + b_resample
    d_xb_raw = fcib["mean"][i_xb_f]
    d_xb = d_xb_raw / corr
    d_err = fcib["err_jk"][i_xb_f] / np.abs(corr)
    d_cov = fcib["cov_jk"][np.ix_(i_xb_f, i_xb_f)] / np.outer(corr, corr)
    sig_resample = 0.5 * np.abs(b_resample) * np.abs(d_xb)
    sig_cib = np.sqrt(sig_cib ** 2 + sig_resample ** 2)
    used_frac = fcib["n_used"][i_xb_f] / float(fcib["n_gal"])
    valid = ((xb <= XB_ONEHALO_MAX) & (used_frac > MIN_USED_FRAC)
             & np.isfinite(d_xb) & np.isfinite(fid_mean))
    iv = np.nonzero(valid)[0]
    dof = len(iv)

    # R0 (docs/p4c_referee_hardening_plan.md M-2a): every SAMPLE-estimated
    # covariance block is debiased by its own inverse-Hartlap inflation
    # (n-1)/(n-p-2) BEFORE combination — exact per block, conservative for
    # the sum. n_jk is read from the measurement file (100 cells post-R0
    # engine patch; 30 for pre-patch files).
    n_jk = int(fcib["n_jk_cells"]) if "n_jk_cells" in fcib.files else 30
    h_data = (n_jk - 1) / max(n_jk - dof - 2, 1)
    cov_data = h_data * d_cov[np.ix_(iv, iv)] + np.diag(sig_cib[iv] ** 2)

    # R5/R3 (final budget): when the correlated CIB covariance and the
    # mass-anchor template exist, they REPLACE the diagonal half-band (which
    # R5 showed absorbs non-CIB-shaped residuals: chi2_med 13.5 vs 40) —
    # the primary chi2 becomes the shape-honest TENSION statement.
    budget = "diag half-band CIB (pre-R5)"
    r5_path, r3_path = LC / "R5_cib_cov.npz", LC / "R3_mass_template.npz"
    mass_T = None
    if r5_path.exists() and r3_path.exists():
        r5 = np.load(r5_path)
        r3m = np.load(r3_path)
        cov_cib_c = (r5["cov_cib"] / np.outer(corr, corr))[np.ix_(iv, iv)]
        sig_res_only = 0.5 * np.abs(b_resample) * np.abs(d_xb)
        cov_data = (h_data * d_cov[np.ix_(iv, iv)] + cov_cib_c
                    + np.diag(sig_res_only[iv] ** 2))
        mass_T = (r3m["dmodel_frac_per_sigma"], r3m["ddata_per_sigma"])
        budget = "FULL: jk(Hartlap) + correlated Sigma_CIB + resample + mass-anchor + per-node satellites"
    print(f"[T3] error budget: {budget}")

    def node_chi2(mean_curve, real_curves, sat_vec=None):
        n_r = max(len(real_curves), 1)
        h_bind = (n_r - 1) / max(n_r - dof - 2, 1)
        ccov = h_bind * np.cov(real_curves[:, iv].T) / n_r
        cov = cov_data + ccov
        if sat_vec is not None:
            cov = cov + np.outer(sat_vec[iv], sat_vec[iv])  # fully correlated
        if mass_T is not None:
            Tm = mass_T[0] * mean_curve - mass_T[1]         # per +1sigma_logM
            cov = cov + np.outer(Tm[iv], Tm[iv])
        r = d_xb[iv] - mean_curve[iv]
        return float(r @ np.linalg.solve(cov, r))

    chi2_fid = node_chi2(fid_mean, fid_real, fid_sat_sys)
    chi2_nodes = np.array([node_chi2(sb_mean[k], sb_real[k],
                                     None if sat_sys is None else sat_sys[k])
                           for k in range(len(node_ids))])
    ok = consistent(chi2_nodes, dof)

    # baseline-data variant (secondary, dust-biased — reported not primary)
    db_xb = fbase["mean"][i_xb_f] / corr
    db_cov = h_data * fbase["cov_jk"][np.ix_(i_xb_f, i_xb_f)][np.ix_(iv, iv)] \
        + np.diag(sig_cib[iv] ** 2)

    def node_chi2_base(mean_curve, real_curves, sat_vec=None):
        n_r = max(len(real_curves), 1)
        h_bind = (n_r - 1) / max(n_r - dof - 2, 1)
        ccov = h_bind * np.cov(real_curves[:, iv].T) / n_r
        cov = db_cov + ccov
        if sat_vec is not None:
            cov = cov + np.outer(sat_vec[iv], sat_vec[iv])
        r = db_xb[iv] - mean_curve[iv]
        return float(r @ np.linalg.solve(cov, r))

    chi2_nodes_base = np.array([node_chi2_base(sb_mean[k], sb_real[k],
                                               None if sat_sys is None else sat_sys[k])
                                for k in range(len(node_ids))])
    ok_base = consistent(chi2_nodes_base, dof)

    ksz = np.load(LC / ("ksz_consistent_nodes_r6.npz" if (LC / "ksz_consistent_nodes_r6.npz").exists() else "ksz_consistent_nodes.npz"))
    in110 = np.isin(node_ids, ksz["node_ids_bgs110"])
    in1125 = np.isin(node_ids, ksz["node_ids_bgs1125"])
    from scipy import stats as st
    rho110 = st.spearmanr(ksz["chi2_desi_bgs110"], chi2_nodes)

    counts = {
        "n_tsz_consistent": int(ok.sum()), "P_tsz": float(ok.mean()),
        "P_tsz_given_ksz110": float(ok[in110].mean()),
        "P_tsz_given_ksz1125": float(ok[in1125].mean()),
        "n_tsz_consistent_baselinedata": int(ok_base.sum()),
        "spearman_ksz110_vs_tsz_chi2": [float(rho110.statistic), float(rho110.pvalue)],
    }

    # R0 (m-1): p-values + threshold-sensitivity of the headline enrichment
    pvals = st.chi2.sf(chi2_nodes, dof)
    thresh_scan = {}
    for k_slack in (1.0, 2.0, 3.0):
        okk = chi2_nodes < dof + k_slack * np.sqrt(2.0 * dof)
        thresh_scan[f"slack{k_slack:g}"] = {
            "n": int(okk.sum()),
            "P_tsz": float(okk.mean()),
            "P_tsz_given_ksz110": float(okk[in110].mean()),
        }
    for pcut in (0.05, 0.01):
        okk = pvals > pcut
        thresh_scan[f"p>{pcut:g}"] = {
            "n": int(okk.sum()), "P_tsz": float(okk.mean()),
            "P_tsz_given_ksz110": float(okk[in110].mean()),
        }
    counts["threshold_scan"] = thresh_scan
    counts["hartlap"] = {"n_jk": n_jk, "h_data": float(h_data),
                         "h_bind_n50": float(49 / max(50 - dof - 2, 1))}
    counts["p_value_fid"] = float(st.chi2.sf(chi2_fid, dof))

    # ---- KG-T (RAP grid, coarse baseline vs Liu) --------------------------
    liu = np.load(LIU_NPZ)
    liu_ok = np.isfinite(liu["theta"]) & np.isfinite(liu["pz1_fiducial"])
    liu_th, liu_y = liu["theta"][liu_ok], liu["pz1_fiducial"][liu_ok]
    ratio_liu = np.interp(liu_th, th_rap, base_c["mean"][i_rap_c]) / liu_y
    kgt_range = liu_th <= 3.0 * t200_data + 1e-6
    kgt_pass = bool(np.all((np.abs(ratio_liu[kgt_range]) < 3.0)
                           & (np.abs(ratio_liu[kgt_range]) > 1 / 3.0)))

    fid_data_ratio = fid_mean[iv] / d_xb[iv]

    # ---- deliverable npz ---------------------------------------------------
    band_rap = np.array([d["mean"][i_rap_c] for d in coarse.values()])
    np.savez(
        LC / "act_ycap_lrg_real.npz",
        # display products (RAP grid, native 0.5'/px)
        theta_rap_arcmin=th_rap, mean_rap=base_c["mean"][i_rap_c],
        err_jk_rap=base_c["err_jk"][i_rap_c],
        cib_band_rap_lo=band_rap.min(0), cib_band_rap_hi=band_rap.max(0),
        random_null=rand["mean"], random_null_err=rand["err_jk"],
        rotated_null=rot["mean"][i_rap_c] if rot is not None else None,
        # comparison products (xb grid, 0.29296875'/px — matches BIND)
        xb=xb, mean_xb_cib17=d_xb, err_jk_xb_cib17=d_err,
        cov_jk_xb_cib17=d_cov, mean_xb_baseline=db_xb,
        mean_xb_cib17_raw=d_xb_raw, resample_corr=corr,
        sig_cib_xb=sig_cib, disc_transfer=T_disc, valid_cols=valid,
        theta200_data_arcmin=t200_data, theta200_mock_arcmin=t200_mock,
        n_gal=int(fcib["n_gal"]), res_arcmin_comparison=0.29296875,
        sample="lrg_sgc_z0.4-0.6 WEIGHT-weighted; primary map deproj_cib_1.7",
    )
    np.savez(LC / "tsz_consistent_nodes.npz", node_ids_all=node_ids,
             chi2_tsz=chi2_nodes, chi2_tsz_baselinedata=chi2_nodes_base,
             chi2_fid=chi2_fid, dof=dof,
             node_ids_tsz_consistent=node_ids[ok],
             node_ids_tsz_consistent_baselinedata=node_ids[ok_base])

    # ---- Fig M2R -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.fill_between(th_rap, band_rap.min(0) * 1e6, band_rap.max(0) * 1e6,
                    color="orange", alpha=0.25, label=f"CIB band ({len(coarse)} maps)")
    ax.errorbar(th_rap, base_c["mean"][i_rap_c] * 1e6, base_c["err_jk"][i_rap_c] * 1e6,
                fmt="o-", color="k", capsize=3, label="baseline ILC")
    ax.plot(th_rap, cib_c["mean"][i_rap_c] * 1e6, "s--", color="tab:orange", ms=4,
            label=r"deproj CIB $\beta$=1.7 (primary)")
    ax.errorbar(rand["theta_value"].astype(float), rand["mean"] * 1e6,
                rand["err_jk"] * 1e6, fmt="^-", color="tab:gray", capsize=2,
                label="random null")
    ax.plot(liu_th, liu_y * 1e6, "x-", color="lightsteelblue",
            label="Liu+2025 csv [superseded]")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\theta_d$ [arcmin]")
    ax.set_ylabel(r"CAP $y$-flux [$y\,\mathrm{arcmin}^2\times10^6$]")
    ax.set_title(f"profile view, native 0.5'/px (n={int(base_c['n_gal'])} LRGs)\n"
                 "baseline dips negative at small θ: CIB/dust bias (finding 3)",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    lo16, hi84 = np.nanpercentile(sb_mean, [16, 84], axis=0)
    ax.fill_between(xb, lo16 * 1e6, hi84 * 1e6, color="lightgray", alpha=0.8,
                    label="SB35 253-node 16–84%")
    for k in np.nonzero(in110)[0]:
        ax.plot(xb, sb_mean[k] * 1e6, color="tab:green", lw=0.5, alpha=0.5)
    ax.plot([], [], color="tab:green", lw=1, label="kSZ-consistent (31)")
    ax.plot(xb, fid_mean * 1e6, color="tab:purple", lw=2, label="BIND fiducial")
    ax.errorbar(xb[valid], d_xb[valid] * 1e6,
                np.sqrt(d_err[valid] ** 2 + sig_cib[valid] ** 2) * 1e6,
                fmt="o", color="k", capsize=3, zorder=6,
                label="ACT deproj-CIB (jk ⊕ CIB sys)")
    mask_hi = (~valid) & (xb > XB_ONEHALO_MAX)
    ax.errorbar(xb[mask_hi], d_xb[mask_hi] * 1e6, d_err[mask_hi] * 1e6,
                fmt="o", mfc="none", color="k", alpha=0.5, capsize=2,
                label="beyond 1-halo range (2-halo/unpainted gas)")
    ax.plot(xb, db_xb * 1e6, "d", ms=4, color="tab:brown", alpha=0.7,
            label="baseline-map data (dust-biased)")
    ax.axvline(XB_ONEHALO_MAX, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$x_b = \theta_d/\theta_{200}$  (both sides per-object-scaled, "
                  "0.293'/px)")
    ax.set_ylabel(r"CAP $y$-flux [$y\,\mathrm{arcmin}^2\times10^6$]")
    ax.set_title(f"apples-to-apples comparison;  fid $\\chi^2$/dof = "
                 f"{chi2_fid/dof:.1f}  (dof={dof}, $x_b\\leq$1.4)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[2]
    ax.scatter(ksz["chi2_desi_bgs110"], chi2_nodes, s=14,
               c=np.where(in110, "tab:green", "tab:gray"), alpha=0.85)
    ax.axhline(dof + CHI2_SLACK * np.sqrt(2 * dof), color="tab:red", ls="--",
               lw=1, label="tSZ consistency threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"kSZ $\chi^2$ (bgs110, M1)")
    ax.set_ylabel(r"tSZ $\chi^2$ (this work, deproj data)")
    ax.set_title(f"cross-probe: ρ={rho110.statistic:.2f};  "
                 f"P(tSZ|kSZ)={counts['P_tsz_given_ksz110']:.2f} vs "
                 f"P(tSZ)={counts['P_tsz']:.2f}", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("M2R (REVISED): real ACT DR6 y-CAP at DESI LRGs vs BIND SB35 — "
                 "matched pixel scale, matched aperture convention, CIB systematic in cov",
                 fontsize=11)
    fig.savefig(FIG_DIR / "M2R_ycap_real.png", dpi=140)

    verdict = {
        "phase": "T3",
        "pass": bool(kgt_pass),
        "metrics": {
            "n_lrg": int(fcib["n_gal"]), "dof": dof,
            "xb_range_used": [float(xb[iv[0]]), float(xb[iv[-1]])],
            "chi2_fid_over_dof": chi2_fid / dof,
            "chi2_nodes_min_med": [float(chi2_nodes.min()), float(np.median(chi2_nodes))],
            "fid_over_data_ratio_valid": [float(x) for x in fid_data_ratio],
            "counts": counts,
            "disc_transfer_fine_over_coarse": [None if not np.isfinite(x) else float(x)
                                               for x in T_disc],
            "cib_sys_over_stat": [float(x) for x in (sig_cib[iv] / d_err[iv])],
            "kgt_liu_ratio_range_1halo": [float(np.nanmin(ratio_liu[kgt_range])),
                                          float(np.nanmax(ratio_liu[kgt_range]))],
            "kgt_pass": kgt_pass,
        },
        "figs": ["figs/M2R_ycap_real.png"],
        "notes": "REVISED after user review of the first M2R (see module docstring "
                 "for the 4 findings: pixel-scale discretization, grid-convention "
                 "mix, CIB dust bias, 1-halo range). Primary data vector = "
                 "deproj_cib_1.7 at 0.29296875'/px on the xb grid; chi2 on "
                 "xb<=1.4 (M1 kSZ convention); CIB band as diagonal systematic. "
                 "Baseline-map data variant reported alongside. "
                 f"MODEL SOURCE: {model_source}.",
        "next": "Capstone X (rebuild) + map-leg disBatch",
    }
    with open(KS / "lightcone/verdicts/T3.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(counts, indent=2))
    print("fid/data (valid cols):", np.round(fid_data_ratio, 2))
    print("chi2_fid/dof:", round(chi2_fid / dof, 2), "| dof:", dof,
          "| KG-T:", kgt_pass)
    print("discretization transfer (fine/coarse):", np.round(np.nan_to_num(T_disc), 2))


if __name__ == "__main__":
    main()
