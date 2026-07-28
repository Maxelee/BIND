#!/usr/bin/env python
"""R3a (docs/paper_improvement_plan.md, round-3 polish): covariance-robustness
checks for the two "external covariance" legs of the P4c paper, both of which
rely on a sample covariance whose provenance is outside this repo's control.

TASK 1 -- kSZ external-covariance Hartlap sensitivity bound. The Ried
Guachalla+25 (Zenodo 19160138) release never published the number of
regions/realizations n used to estimate ``cov_ksz`` (R6's own audit,
``examples/_r6_ksz_audit.py``, flags this as "hartlap_missing_data_block":
UNQUANTIFIABLE, not applied). We cannot apply the EXACT Hartlap debias to
that block, but we CAN bound its effect: rerun R6's bgs110 per-node chi2
chain with the (unit-fixed) external covariance additionally multiplied by
h(n) = (n-1)/(n-n_dof-2) for a scan of hypothetical n, and see how much the
consistent-node set and ranking move.

This first reproduces R6's bgs110 chi2 EXACTLY (byte-for-byte against
``ksz_consistent_nodes_r6.npz['chi2_desi_bgs110']``) by importing R6's own
helper functions (``hartlap``, ``fix_cov``, ``shrink``, ``MIN_REAL``, ``ZEN``,
``DCUT``) rather than re-deriving them, then re-runs the same per-node loop
with the scan factor applied only to the external (data-side) covariance
block -- the one R6 left uncorrected.

TASK 2 -- bootstrap-covariance cross-check of the tSZ verdict. T3
(``examples/lightcone_m2r_ycap_real.py``) builds its data-side covariance
from the jackknife block of the fine-res ACT y-CAP measurement
(``T2f_lrg_z0406_cib1.7.npz['cov_jk']``, n_jk=100 cells) plus a correlated
CIB systematic, a resample-bias diagonal, and (BIND side) the realization
covariance + satellite/mass-anchor templates. This rebuilds that EXACT chi2
for the TNG fiducial and the best-fit node only (no plots, no product
writes -- read-only), first as published (jackknife block), then with the
jackknife block replaced by the bootstrap covariance
(``T2f_lrg_z0406_cib1.7.npz['cov_boot']``, n_boot=400) on the same xb
sub-block and fit columns, Hartlap-debiased with the n=400 analog.

Outputs
-------
    LC/covariance_robustness.json

Usage
-----
    python examples/_r3a_covariance_robustness.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import _r6_ksz_audit as r6
from lightcone_m2_ycap_liu import mean_theta200_arcmin, N_XB as N_XB_KSZ

CEPH = Path("/mnt/home/mlee1/ceph")
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", str(CEPH / "bind_science/ksz_confront")))
LC = KS / "lightcone"

SAMPLE = "bgs110"
N_SCAN = [30, 50, 100, 200, 500]
N_BOOT = 400


# =====================================================================
# TASK 1 -- kSZ Hartlap sensitivity scan
# =====================================================================

def task1():
    print("\n" + "=" * 78)
    print("TASK 1 -- kSZ external-covariance Hartlap sensitivity bound (bgs110)")
    print("=" * 78)

    dz = np.load(r6.ZEN / f"Fig8_BGS_BRIGHT-20.2_logm{r6.DCUT[SAMPLE]}.npz")
    th_data, ratio_data, yerr_data, cov_ksz_raw = (
        dz["th"], dz["ratio"], dz["yerr"], dz["cov_ksz"])
    cov_ksz_fixed_full = r6.fix_cov(cov_ksz_raw, yerr_data)

    thr = mean_theta200_arcmin(SAMPLE)
    m1 = th_data <= 1.4 * thr
    ndof = int(m1.sum())
    print(f"[T1] sample={SAMPLE}  theta200_fid={thr:.4f}'  n_dof={ndof} "
          f"(theta<=1.4*theta200 cut)")

    taucap = np.load(LC / "taucap_lightcone.npz", allow_pickle=True)
    capmat = np.load(LC / "capmat_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])
    node_ids = taucap["node_ids"]
    assert np.array_equal(node_ids, capmat["node_ids"])

    xb = taucap["theta_value"][:N_XB_KSZ]
    node_theta = xb * thr

    mat_mean_node = capmat[f"sb35_mean_{SAMPLE}"][:, :N_XB_KSZ].astype(np.float64)
    mat_mean_fid = capmat[f"fid_mean_{SAMPLE}"][:N_XB_KSZ].astype(np.float64)

    fid_tau_mean = taucap[f"fid_mean_{SAMPLE}"][:N_XB_KSZ]
    fid_tau_real = taucap[f"fid_real_{SAMPLE}"][:, :N_XB_KSZ]
    fid_fgas_real = fid_tau_real / mat_mean_fid[None, :] / F_B

    sb35_tau_mean = taucap[f"sb35_mean_{SAMPLE}"][:, :N_XB_KSZ]
    sb35_tau_real = taucap[f"sb35_real_{SAMPLE}"][:, :, :N_XB_KSZ]
    with np.errstate(invalid="ignore", divide="ignore"):
        sb35_fgas_mean = sb35_tau_mean / mat_mean_node / F_B
        sb35_fgas_real = sb35_tau_real / mat_mean_node[:, None, :] / F_B

    fid_real_m1 = np.array([np.interp(th_data[m1], node_theta, fid_fgas_real[r])
                             for r in range(fid_fgas_real.shape[0])])
    cov_fid_fallback = np.cov(fid_real_m1, rowvar=False)
    n_fid_fallback = fid_real_m1.shape[0]

    cov_ksz_m1_fixed = cov_ksz_fixed_full[np.ix_(m1, m1)]
    n_nodes = sb35_fgas_mean.shape[0]

    def compute_chi2(ext_hartlap_factor):
        """R6's exact per-node loop, with the external (data-side) covariance
        block additionally scaled by ``ext_hartlap_factor`` (1.0 = R6
        baseline, no data-side Hartlap -- matches R6 exactly)."""
        cov_ext = cov_ksz_m1_fixed * ext_hartlap_factor
        chi2_arr = np.full(n_nodes, np.nan)
        for k in range(n_nodes):
            if not np.all(np.isfinite(sb35_fgas_mean[k])):
                continue
            pn = np.interp(th_data, node_theta, sb35_fgas_mean[k])
            real_at_data_m1 = np.array([
                np.interp(th_data[m1], node_theta, sb35_fgas_real[k, r])
                for r in range(50)
            ])
            finite = np.all(np.isfinite(real_at_data_m1), axis=1)
            if finite.sum() >= r6.MIN_REAL:
                cov_map = np.cov(real_at_data_m1[finite], rowvar=False)
                n_r = int(finite.sum())
            else:
                cov_map = cov_fid_fallback
                n_r = n_fid_fallback
            resid = pn[m1] - ratio_data[m1]
            h_bind = r6.hartlap(n_r, ndof)
            cov_b = cov_ext + r6.shrink(h_bind * np.atleast_2d(cov_map) / n_r)
            chi2_arr[k] = float(resid @ np.linalg.inv(cov_b) @ resid)
        return chi2_arr

    chi2_baseline = compute_chi2(1.0)

    # ---- validate against the stored R6 product -------------------------
    stored = np.load(LC / "ksz_consistent_nodes_r6.npz", allow_pickle=True)
    assert np.array_equal(stored["node_ids_all"], node_ids)
    stored_chi2 = stored["chi2_desi_bgs110"]
    finite_both = np.isfinite(chi2_baseline) & np.isfinite(stored_chi2)
    max_abs_diff = float(np.nanmax(np.abs(chi2_baseline[finite_both] - stored_chi2[finite_both])))
    n_finite_check = int(finite_both.sum())
    print(f"[T1] validation: reproduced-vs-stored max|d chi2| over {n_finite_check} "
          f"finite nodes = {max_abs_diff:.3e} (expect ~1e-6)")

    thresh = ndof + 2 * np.sqrt(2 * ndof)
    consistent_baseline = np.isfinite(chi2_baseline) & (chi2_baseline < thresh)
    n_consistent_baseline = int(consistent_baseline.sum())
    print(f"[T1] baseline (no data-side Hartlap): n_consistent = {n_consistent_baseline} "
          f"(threshold chi2 < {thresh:.3f}, expect 27)")
    baseline_ids = set(int(x) for x in node_ids[consistent_baseline].tolist())

    scan = {}
    for n in N_SCAN:
        h = r6.hartlap(n, ndof)
        chi2_n = compute_chi2(h)
        consistent_n = np.isfinite(chi2_n) & (chi2_n < thresh)
        n_consistent = int(consistent_n.sum())
        fb = np.isfinite(chi2_baseline) & np.isfinite(chi2_n)
        rho, pval = spearmanr(chi2_baseline[fb], chi2_n[fb])
        ids_n = set(int(x) for x in node_ids[consistent_n].tolist())
        added = sorted(ids_n - baseline_ids)
        removed = sorted(baseline_ids - ids_n)
        set_changed = bool(added or removed)
        scan[str(n)] = {
            "h": float(h),
            "n_consistent": n_consistent,
            "rho_vs_baseline": float(rho),
            "rho_pvalue": float(pval),
            "set_changed_ids": {
                "changed": set_changed,
                "added": added,
                "removed": removed,
            },
        }
        print(f"[T1]   n={n:4d}  h={h:6.3f}  n_consistent={n_consistent:3d}  "
              f"rho={rho:.4f}  set_changed={set_changed}  "
              f"(+{len(added)}/-{len(removed)})")

    return {
        "validation": {
            "max_abs_dchi2_vs_r6_stored": max_abs_diff,
            "n_finite_nodes_checked": n_finite_check,
            "n_dof": ndof,
            "threshold": float(thresh),
        },
        "baseline_n_consistent": n_consistent_baseline,
        "ksz_hartlap_scan": scan,
    }


# =====================================================================
# TASK 2 -- tSZ bootstrap-covariance cross-check (fiducial + best node)
# =====================================================================

XB_ONEHALO_MAX = 1.4
MIN_USED_FRAC = 0.5
N_XB_TSZ = 18
BIND_PIX_AREA = 0.29296875 ** 2
CHI2_SLACK = 2.0


def load_t2(name, fine=False):
    p = LC / f"{'T2f' if fine else 'T2'}_{name}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def cols(d, kind):
    return np.nonzero(d["theta_kind"] == kind)[0]


def task2():
    print("\n" + "=" * 78)
    print("TASK 2 -- tSZ bootstrap-covariance cross-check (fiducial + best node)")
    print("=" * 78)

    anchors = np.load(LC / "T3_theta200_anchors.npz")
    t200_data = float(anchors["t200_data_arcmin"])

    fcib = load_t2("lrg_z0406_cib1.7", fine=True)
    if fcib is None:
        raise SystemExit("missing T2f_lrg_z0406_cib1.7.npz")

    i_xb_f = cols(fcib, "xb")
    xb = fcib["theta_value"][i_xb_f].astype(float)

    r1_path = LC / "R1_resample_correction.npz"
    theta_cols_arcmin = xb * t200_data
    if r1_path.exists():
        r1 = np.load(r1_path)
        b_resample = np.interp(theta_cols_arcmin, r1["theta_arcmin"], r1["bias_pixwin"])
    else:
        print("[T2] WARNING: R1 correction absent — data uncorrected")
        b_resample = np.zeros(len(xb))
    corr = 1.0 + b_resample

    # ---- BIND side (R2 HOD population + R7 fidelity correction) ---------
    b = np.load(LC / "lrgy_beam_lightcone.npz", allow_pickle=True)
    fid_mean = b["fid_mean_lrg"][:N_XB_TSZ] * BIND_PIX_AREA
    fid_real = b["fid_real_lrg"][:, :N_XB_TSZ] * BIND_PIX_AREA
    sb_mean = b["sb35_mean_lrg"][:, :N_XB_TSZ] * BIND_PIX_AREA
    sb_real = b["sb35_real_lrg"][:, :, :N_XB_TSZ] * BIND_PIX_AREA
    node_ids = b["node_ids"]

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
                        "kappa-calibrated f_eff=0.08 +/- 0.04)")

    r7_path = KS / "lightcone/verdicts/R7.json"
    if r7_path.exists() and hod_path.exists():
        r7 = json.load(open(r7_path))
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
    print(f"[T2] model source: {model_source}")

    # ---- fit range / dof --------------------------------------------------
    d_xb_raw = fcib["mean"][i_xb_f]
    d_xb = d_xb_raw / corr
    used_frac = fcib["n_used"][i_xb_f] / float(fcib["n_gal"])
    valid = ((xb <= XB_ONEHALO_MAX) & (used_frac > MIN_USED_FRAC)
             & np.isfinite(d_xb) & np.isfinite(fid_mean))
    iv = np.nonzero(valid)[0]
    dof = len(iv)
    print(f"[T2] dof={dof}  xb range used={xb[iv[0]]:.3f}-{xb[iv[-1]]:.3f}")

    n_jk = int(fcib["n_jk_cells"]) if "n_jk_cells" in fcib.files else 30
    h_data = (n_jk - 1) / max(n_jk - dof - 2, 1)
    d_cov_jk = fcib["cov_jk"][np.ix_(i_xb_f, i_xb_f)] / np.outer(corr, corr)

    # ---- FULL error budget (R5 correlated CIB cov + R3 mass-anchor) -----
    r5_path, r3_path = LC / "R5_cib_cov.npz", LC / "R3_mass_template.npz"
    if not (r5_path.exists() and r3_path.exists()):
        raise SystemExit("missing R5_cib_cov.npz / R3_mass_template.npz — "
                          "cannot reproduce T3's FULL error budget")
    r5 = np.load(r5_path)
    r3m = np.load(r3_path)
    cov_cib_c = (r5["cov_cib"] / np.outer(corr, corr))[np.ix_(iv, iv)]
    sig_res_only = 0.5 * np.abs(b_resample) * np.abs(d_xb)
    mass_T = (r3m["dmodel_frac_per_sigma"], r3m["ddata_per_sigma"])

    cov_data_jk = (h_data * d_cov_jk[np.ix_(iv, iv)] + cov_cib_c
                   + np.diag(sig_res_only[iv] ** 2))

    # ---- bootstrap alternative --------------------------------------------
    d_cov_boot = fcib["cov_boot"][np.ix_(i_xb_f, i_xb_f)] / np.outer(corr, corr)
    h_boot = (N_BOOT - 1) / max(N_BOOT - dof - 2, 1)
    cov_data_boot = (h_boot * d_cov_boot[np.ix_(iv, iv)] + cov_cib_c
                     + np.diag(sig_res_only[iv] ** 2))
    print(f"[T2] h_data(jk, n_jk={n_jk})={h_data:.4f}   "
          f"h_boot(n_boot={N_BOOT})={h_boot:.4f}")

    # informational: err_boot/err_jk on the fit columns (spatial-correlation
    # underestimate check, mirrors Phase N2's printed ratio)
    err_jk_xb = fcib["err_jk"][i_xb_f]
    err_boot_xb = fcib["err_boot"][i_xb_f]
    boot_over_jk_fit = (err_boot_xb / err_jk_xb)[iv]
    print(f"[T2] err_boot/err_jk on fit columns: "
          f"{np.round(boot_over_jk_fit, 3).tolist()}")

    def node_chi2(mean_curve, real_curves, cov_data, sat_vec=None):
        n_r = max(len(real_curves), 1)
        h_bind = (n_r - 1) / max(n_r - dof - 2, 1)
        ccov = h_bind * np.cov(real_curves[:, iv].T) / n_r
        cov = cov_data + ccov
        if sat_vec is not None:
            cov = cov + np.outer(sat_vec[iv], sat_vec[iv])
        if mass_T is not None:
            Tm = mass_T[0] * mean_curve - mass_T[1]
            cov = cov + np.outer(Tm[iv], Tm[iv])
        r = d_xb[iv] - mean_curve[iv]
        return float(r @ np.linalg.solve(cov, r))

    # ---- (a) published jackknife block, all 253 nodes + fiducial --------
    chi2_fid_jk = node_chi2(fid_mean, fid_real, cov_data_jk, fid_sat_sys)
    chi2_nodes_jk = np.array([
        node_chi2(sb_mean[k], sb_real[k], cov_data_jk,
                  None if sat_sys is None else sat_sys[k])
        for k in range(len(node_ids))
    ])
    best_idx = int(np.nanargmin(chi2_nodes_jk))
    chi2_best_jk = float(chi2_nodes_jk[best_idx])
    best_node_id = int(node_ids[best_idx])

    # ---- validate against the stored T3.json verdict ---------------------
    t3 = json.load(open(KS / "lightcone/verdicts/T3.json"))
    t3m = t3["metrics"]
    stored_chi2_fid_over_dof = t3m["chi2_fid_over_dof"]
    stored_chi2_best = t3m["chi2_nodes_min_med"][0]
    print(f"[T2] validation: chi2_fid/dof reproduced={chi2_fid_jk/dof:.6f}  "
          f"stored={stored_chi2_fid_over_dof:.6f}  "
          f"(delta={abs(chi2_fid_jk/dof - stored_chi2_fid_over_dof):.2e})")
    print(f"[T2] validation: chi2_best reproduced={chi2_best_jk:.6f}  "
          f"stored={stored_chi2_best:.6f}  "
          f"(delta={abs(chi2_best_jk - stored_chi2_best):.2e})  "
          f"best node id={best_node_id}")

    # ---- (b) bootstrap block, fiducial + best node ONLY ------------------
    chi2_fid_boot = node_chi2(fid_mean, fid_real, cov_data_boot, fid_sat_sys)
    chi2_best_boot = node_chi2(sb_mean[best_idx], sb_real[best_idx], cov_data_boot,
                               None if sat_sys is None else sat_sys[best_idx])

    thresh = dof + CHI2_SLACK * np.sqrt(2.0 * dof)
    consistent_jk = (chi2_fid_jk < thresh, chi2_best_jk < thresh)
    consistent_boot = (chi2_fid_boot < thresh, chi2_best_boot < thresh)
    zero_consistent_jk = not any(consistent_jk)
    zero_consistent_boot = not any(consistent_boot)
    verdict_unchanged = bool(zero_consistent_jk == zero_consistent_boot and zero_consistent_boot)
    # also require chi2 did not DECREASE below the threshold (the point of
    # the check, per task spec) -- i.e. neither stat dropped into the
    # consistent region under the boot covariance.
    dropped_below_threshold = bool((chi2_fid_boot < thresh) or (chi2_best_boot < thresh))

    print(f"[T2] threshold (dof + 2*sqrt(2*dof)) = {thresh:.3f}")
    print(f"[T2] chi2_fid:  jk={chi2_fid_jk:.3f}  boot={chi2_fid_boot:.3f}")
    print(f"[T2] chi2_best: jk={chi2_best_jk:.3f}  boot={chi2_best_boot:.3f}  (node {best_node_id})")
    print(f"[T2] verdict (zero of {{fid,best}} consistent): "
          f"jk={zero_consistent_jk}  boot={zero_consistent_boot}  "
          f"unchanged={verdict_unchanged}")

    boot_underestimate_note = (
        "cov_boot is a per-object bootstrap; prior N2 (Phase N) surfacing of "
        "err_boot/err_jk on these fit columns found ratios in the ~0.73-1.0 "
        "range (bootstrap underestimating the true spatially-correlated "
        "error), so a priori the boot-covariance chi2 could be EITHER larger "
        "(if the boot covariance is simply smaller/tighter, chi2 inflates) "
        "or smaller depending on how the eigenstructure interacts with the "
        "residual vector -- reported empirically below, not assumed.")

    return {
        "validation": {
            "chi2_fid_over_dof_reproduced": chi2_fid_jk / dof,
            "chi2_fid_over_dof_stored_T3": stored_chi2_fid_over_dof,
            "chi2_best_reproduced": chi2_best_jk,
            "chi2_best_stored_T3": stored_chi2_best,
            "best_node_id": best_node_id,
            "dof": dof,
            "threshold": float(thresh),
        },
        "tsz_boot_check": {
            "chi2_fid_jk": chi2_fid_jk,
            "chi2_fid_boot": chi2_fid_boot,
            "chi2_best_jk": chi2_best_jk,
            "chi2_best_boot": chi2_best_boot,
            "hartlap_boot": float(h_boot),
            "hartlap_jk": float(h_data),
            "n_jk": n_jk,
            "n_boot": N_BOOT,
            "dof": dof,
            "threshold": float(thresh),
            "best_node_id": best_node_id,
            "err_boot_over_err_jk_fit_columns": [float(x) for x in boot_over_jk_fit],
            "dropped_below_threshold_under_boot": dropped_below_threshold,
            "verdict_unchanged": verdict_unchanged,
            "note": boot_underestimate_note,
        },
    }


def main():
    r1 = task1()
    r2 = task2()

    out = {
        "ksz_hartlap_scan": r1["ksz_hartlap_scan"],
        "baseline_n_consistent": r1["baseline_n_consistent"],
        "tsz_boot_check": r2["tsz_boot_check"],
        "method_notes": {
            "task1": (
                "Reproduces examples/_r6_ksz_audit.py's bgs110 per-node chi2 "
                "chain exactly (unit-fixed cov_ksz + BIND-side Hartlap h_bind, "
                "consistency rule chi2 < ndof + 2*sqrt(2*ndof), ndof from the "
                "theta<=1.4*theta200 cut) by importing its own hartlap/fix_cov/"
                "shrink helpers, then additionally multiplies the EXTERNAL "
                "(data-side, Ried Guachalla+25) covariance block by "
                "h(n)=(n-1)/(n-n_dof-2) for a scan of hypothetical sample "
                "counts n, since the true n is not published. This bounds "
                "(does not fix) the referee gap R6 itself flagged as "
                "unquantifiable."
            ),
            "task2": (
                "Reproduces examples/lightcone_m2r_ycap_real.py's FULL error "
                "budget (jk(Hartlap) + correlated Sigma_CIB [R5] + resample "
                "diagonal + mass-anchor template [R3] + per-node satellite "
                "systematic [R2]) for the TNG fiducial and the min-chi2 "
                "('best') node only, then reruns it with the jackknife block "
                "(cov_jk, n_jk=100) replaced by the bootstrap block (cov_boot, "
                "n_boot=400) on the same xb sub-block and fit columns, "
                "Hartlap-debiased with h_boot=(400-1)/(400-dof-2). Read-only: "
                "no plots or product files are (re)written by this script; "
                "act_ycap_lrg_real.npz / tsz_consistent_nodes.npz / T3.json "
                "are untouched."
            ),
            "validation": {
                "task1": r1["validation"],
                "task2": r2["validation"],
            },
        },
    }

    out_path = LC / "covariance_robustness.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[R3a] wrote {out_path}")
    print(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()
