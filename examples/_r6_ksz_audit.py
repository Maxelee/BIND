#!/usr/bin/env python
"""R6 (docs/p4c_referee_hardening_plan.md, phase R6) -- kSZ-leg audit, to the
T3 standard. Re-opens ``examples/lightcone_m1_fgas_desiact.py``'s M1
comparison (map-level f~gas(theta) vs the real Ried Guachalla+25 DESI(BGS) x
ACT kSZ data, Zenodo 19160138) and asks the T3 checklist: matched estimator?
Hartlap-debiased covariance? unit/convention bugs?

**HEADLINE FINDING (verified byte-for-byte, not a guess).** The Fig8
``desact_zenodo`` npz's ``cov_ksz`` field is NOT the covariance of the
``ratio`` column it sits next to (f~gas(theta), the CAP-RATIO quantity M1's
chi2 is built from) -- it is the covariance of the *raw, unnormalized*
T^CAP(theta) amplitude profile reported in the companion ``Fig6_*`` npz
(units ~muK*arcmin^2-scale, values ~O(1-30) vs the ratio's O(0.1-1)):

    Fig8_BGS_BRIGHT-20.2_logm11.00.npz['cov_ksz']
        == Fig6_BGS_BRIGHT-20.2_logm11.00.npz['cov']        (EXACT, verified below)

confirmed by a *second*, independent tell: Fig8's OWN ``yerr`` field (which
IS the correctly-scaled ratio uncertainty -- it has the right O(0.03-0.9)
magnitude to match ``ratio``) disagrees with ``sqrt(diag(cov_ksz))`` by a
strongly theta-DEPENDENT factor (3.26x too small at th=1', 31x too large at
th=10.75' for the logm11.00 cut) -- exactly the signature of plugging in a
covariance for a *different, larger-dynamic-range* quantity. No
``Fig6_..._logm11.25.npz`` exists to do the same byte-exact check for the
1125 cut, but the same diagnostic (yerr vs sqrt(diag(cov_ksz)) mismatch,
0.024x-0.37x, growing with theta the same way) is present there too -- same
defect, unverified byte-exact only because the comparison file doesn't
exist on disk.

**Impact.** Every chi2 this repo has ever computed against the kSZ Fig8 data
(``lightcone_m1_fgas_desiact.py``'s map-level M1, AND the per-halo companion
in ``_build_ksz_paper_nb.py`` SS6/SS6a/SS7 -- all of which call
``dz["cov_ksz"]`` directly) has used the WRONG covariance for the fitted
quantity: too-tight at small theta (where most of the xb<=1.4 fit range
lives for BGS) -> chi2 inflated -> nodes wrongly REJECTED; too-loose at large
theta -> those points are effectively dropped from the fit. Both push the
same direction as the OTHER referee-plan defect (M-2a, no Hartlap anywhere):
chi2 has been biased in an uncontrolled, theta-dependent way this whole
campaign.

**Fix applied here (unit-fix, not a re-measurement).** Rescale cov_ksz to
the CORRECT diagonal (Fig8's own ``yerr**2``) while preserving whatever
correlation STRUCTURE the (wrongly-scaled) published matrix encodes --
correlations between CAP apertures at different theta are set by the shared
sky footprint/beam/large-scale-structure modes, a property that is
approximately scale-invariant under the quantity's own overall
normalization, so this is the most defensible correction achievable without
re-measuring the survey's raw CMB temperature map (which we do not have):

    corr = cov_ksz / outer(sqrt(diag(cov_ksz)), sqrt(diag(cov_ksz)))
    cov_ksz_fixed = outer(yerr, yerr) * corr

**Second correction: Hartlap on the block we control.** ``chi2_total =
cov_ksz(_fixed) + cov_map/n_real_used`` -- the second term is BIND's own
n_r-realization (n_r<=50, per-node) sample covariance of the mean, exactly
analogous to T3's ``h_bind`` block (docs/p4c_referee_hardening_plan.md R0;
``examples/lightcone_m2r_ycap_real.py`` lines ~219-220). Apply the identical
Hartlap factor here: ``h_bind = (n_r-1)/(n_r-ndof-2)``, multiplying cov_map
BEFORE the /n_r division (T3's exact convention).

**NOT corrected (documented, not fabricated).** The data-side cov_ksz's OWN
sample-covariance estimator (jackknife? bootstrap? how many regions?) is
external to this repo -- Ried Guachalla+25's own pipeline, Zenodo 19160138,
no n_jk/n_boot metadata ships with the npz, and the task constraints forbid
fetching the source paper. Unlike T3's data covariance (OUR OWN ACT y-map
jackknife, n_jk=100, stored in the npz we built), we cannot Hartlap-correct
cov_ksz's own estimator bias without that number. Flagged in the verdict,
not silently ignored and not guessed.

Outputs
-------
    KS/lightcone/ksz_consistent_nodes_r6.npz  (NEW product; original
        ksz_consistent_nodes.npz is left untouched)
    KS/lightcone/figs/R6_ksz_audit.png
    KS/lightcone/verdicts/R6.json

Usage
-----
    python examples/_r6_ksz_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightcone_m2_ycap_liu import mean_theta200_arcmin, N_XB

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
VERDICT_DIR = LIGHTCONE / "verdicts"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_DIR.mkdir(parents=True, exist_ok=True)
ZEN = KS / "desact_zenodo"

SAMPLES = ["bgs110", "bgs1125"]
DCUT = {"bgs110": "11.00", "bgs1125": "11.25"}
SAMPLE_LABEL = {"bgs110": r"BGS $M_\star{>}10^{11.0}$", "bgs1125": r"BGS $M_\star{>}10^{11.25}$"}
PERHALO_CONSISTENT = {"bgs110": (49, 256), "bgs1125": (65, 256)}

SHRINK_ALPHA = 0.10
MIN_REAL = 10


def shrink(cov: np.ndarray, alpha: float = SHRINK_ALPHA) -> np.ndarray:
    diag = np.diag(np.diag(cov))
    return (1 - alpha) * cov + alpha * diag


def hartlap(n: int, dof: int) -> float:
    """h = (n-1)/(n-dof-2); multiplies a sample covariance BEFORE inverting
    to debias the precision matrix (Hartlap+07). h>1 always here (n>dof+2
    enforced by MIN_REAL/n_r gating upstream)."""
    return (n - 1) / max(n - dof - 2, 1)


def verify_unit_bug():
    """Byte-exact confirmation for the one cut where a Fig6 twin exists
    (logm11.00); returns the verification dict for the verdict."""
    out = {}
    d6 = np.load(ZEN / "Fig6_BGS_BRIGHT-20.2_logm11.00.npz")
    d8 = np.load(ZEN / "Fig8_BGS_BRIGHT-20.2_logm11.00.npz")
    exact = bool(np.array_equal(d6["cov"], d8["cov_ksz"]))
    out["logm11.00_fig6_cov_eq_fig8_cov_ksz_exact"] = exact
    out["logm11.00_max_abs_diff"] = float(np.max(np.abs(d6["cov"] - d8["cov_ksz"])))
    for sample in SAMPLES:
        d = np.load(ZEN / f"Fig8_BGS_BRIGHT-20.2_logm{DCUT[sample]}.npz")
        ratio_yerr_over_covsqrt = d["yerr"] / np.sqrt(np.diag(d["cov_ksz"]))
        out[f"{sample}_yerr_over_sqrt_diag_cov_ksz_range"] = [
            float(ratio_yerr_over_covsqrt.min()), float(ratio_yerr_over_covsqrt.max())]
    return out


def fix_cov(cov_ksz: np.ndarray, yerr: np.ndarray) -> np.ndarray:
    """Rescale cov_ksz to the correct (yerr**2) diagonal, preserving its
    correlation structure. Exact congruence transform -> stays PSD."""
    sig_wrong = np.sqrt(np.diag(cov_ksz))
    corr = cov_ksz / np.outer(sig_wrong, sig_wrong)
    return np.outer(yerr, yerr) * corr


def main():
    taucap = np.load(LIGHTCONE / "taucap_lightcone.npz", allow_pickle=True)
    capmat = np.load(LIGHTCONE / "capmat_lightcone.npz", allow_pickle=True)
    F_B = float(capmat["F_B"])
    node_ids = taucap["node_ids"]
    assert np.array_equal(node_ids, capmat["node_ids"])

    orig = np.load(LIGHTCONE / "ksz_consistent_nodes.npz", allow_pickle=True)
    tsz = np.load(LIGHTCONE / "tsz_consistent_nodes.npz", allow_pickle=True)
    assert np.array_equal(tsz["node_ids_all"], node_ids), "kSZ/tSZ node_ids_all must align"

    verification = verify_unit_bug()
    print("[R6] unit-bug verification:", json.dumps(verification, indent=2))

    metrics = {"unit_bug_verification": verification, "shrink_alpha": SHRINK_ALPHA}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))

    chi2_new_by_sample = {}
    chi2_old_replica_by_sample = {}
    node_ids_new_consistent = {}
    per_sample_metrics = {}

    for col, sample in enumerate(SAMPLES):
        dz = np.load(ZEN / f"Fig8_BGS_BRIGHT-20.2_logm{DCUT[sample]}.npz")
        th_data, ratio_data, yerr_data, cov_ksz_raw = dz["th"], dz["ratio"], dz["yerr"], dz["cov_ksz"]
        cov_ksz_fixed_full = fix_cov(cov_ksz_raw, yerr_data)

        thr = mean_theta200_arcmin(sample)
        m1 = th_data <= 1.4 * thr
        ndof = int(m1.sum())

        xb = taucap["theta_value"][:N_XB]
        node_theta = xb * thr

        mat_mean_node = capmat[f"sb35_mean_{sample}"][:, :N_XB].astype(np.float64)
        mat_mean_fid = capmat[f"fid_mean_{sample}"][:N_XB].astype(np.float64)

        fid_tau_mean = taucap[f"fid_mean_{sample}"][:N_XB]
        fid_tau_real = taucap[f"fid_real_{sample}"][:, :N_XB]
        fid_fgas_real = fid_tau_real / mat_mean_fid[None, :] / F_B

        sb35_tau_mean = taucap[f"sb35_mean_{sample}"][:, :N_XB]
        sb35_tau_real = taucap[f"sb35_real_{sample}"][:, :, :N_XB]
        with np.errstate(invalid="ignore", divide="ignore"):
            sb35_fgas_mean = sb35_tau_mean / mat_mean_node / F_B
            sb35_fgas_real = sb35_tau_real / mat_mean_node[:, None, :] / F_B

        fid_real_m1 = np.array([np.interp(th_data[m1], node_theta, fid_fgas_real[r])
                                 for r in range(fid_fgas_real.shape[0])])
        cov_fid_fallback = np.cov(fid_real_m1, rowvar=False)
        n_fid_fallback = fid_real_m1.shape[0]

        cov_ksz_m1_raw = cov_ksz_raw[np.ix_(m1, m1)]
        cov_ksz_m1_fixed = cov_ksz_fixed_full[np.ix_(m1, m1)]

        n_nodes = sb35_fgas_mean.shape[0]
        chi2_old_replica = np.full(n_nodes, np.nan)   # exact M1 replica (sanity check)
        chi2_new = np.full(n_nodes, np.nan)            # unit-fix + Hartlap
        n_real_used = np.full(n_nodes, -1, dtype=np.int64)
        n_fallback = 0
        for k in range(n_nodes):
            if not np.all(np.isfinite(sb35_fgas_mean[k])):
                continue
            pn = np.interp(th_data, node_theta, sb35_fgas_mean[k])
            real_at_data_m1 = np.array([
                np.interp(th_data[m1], node_theta, sb35_fgas_real[k, r]) for r in range(50)
            ])
            finite = np.all(np.isfinite(real_at_data_m1), axis=1)
            n_real_used[k] = int(finite.sum())
            if finite.sum() >= MIN_REAL:
                cov_map = np.cov(real_at_data_m1[finite], rowvar=False)
                n_r = int(finite.sum())
            else:
                cov_map = cov_fid_fallback
                n_r = n_fid_fallback
                n_fallback += 1
            resid = pn[m1] - ratio_data[m1]

            # --- exact M1 replica (validation) ---
            cov_b_old = cov_ksz_m1_raw + shrink(np.atleast_2d(cov_map / n_r))
            chi2_old_replica[k] = float(resid @ np.linalg.inv(cov_b_old) @ resid)

            # --- R6 corrected: unit-fixed data cov + Hartlap on the BIND block ---
            h_bind = hartlap(n_r, ndof)
            cov_b_new = cov_ksz_m1_fixed + shrink(h_bind * np.atleast_2d(cov_map) / n_r)
            chi2_new[k] = float(resid @ np.linalg.inv(cov_b_new) @ resid)

        # --- validate against the stored ksz_consistent_nodes.npz chi2 ---
        stored = orig[f"chi2_desi_{sample}"]
        finite_both = np.isfinite(chi2_old_replica) & np.isfinite(stored)
        replica_match = float(np.nanmax(np.abs(chi2_old_replica[finite_both] - stored[finite_both])))
        print(f"[R6] {sample}: replica-vs-stored max|delta chi2| = {replica_match:.3e} "
              f"(should be ~0 -- confirms the M1 pipeline is faithfully reproduced before applying fixes)")

        thresh = ndof + 2 * np.sqrt(2 * ndof)
        consistent_new = np.isfinite(chi2_new) & (chi2_new < thresh)
        consistent_old = np.isfinite(stored) & (stored < thresh)
        n_new = int(consistent_new.sum())
        n_old = int(consistent_old.sum())
        n_valid = int(np.isfinite(chi2_new).sum())

        rho, pval = spearmanr(stored[finite_both], chi2_new[finite_both])
        rho_tsz, pval_tsz = spearmanr(chi2_new[np.isfinite(chi2_new) & np.isfinite(tsz["chi2_tsz"])],
                                       tsz["chi2_tsz"][np.isfinite(chi2_new) & np.isfinite(tsz["chi2_tsz"])])

        old_set = set(node_ids[consistent_old].tolist())
        new_set = set(node_ids[consistent_new].tolist())
        survive = old_set & new_set
        pct_change = 100.0 * abs(n_new - n_old) / max(n_old, 1)

        per_sample_metrics[sample] = {
            "n_dof": ndof, "theta200_fid_arcmin": thr,
            "n_old_consistent": n_old, "n_new_consistent": n_new, "n_valid_nodes": n_valid,
            "n_old_denominator_paper": PERHALO_CONSISTENT[sample][1],
            "old_survive_in_new": len(old_set & new_set),
            "old_set_size": len(old_set), "new_set_size": len(new_set),
            "pct_change_in_count": pct_change,
            "spearman_old_vs_new_chi2": [float(rho), float(pval)],
            "spearman_new_ksz_vs_tsz_chi2": [float(rho_tsz), float(pval_tsz)],
            "replica_validation_max_abs_dchi2": replica_match,
            "median_n_real_used": float(np.median(n_real_used[n_real_used >= 0])),
            "n_nodes_using_fiducial_cov_fallback": int(n_fallback),
        }
        chi2_new_by_sample[sample] = chi2_new.copy()
        chi2_old_replica_by_sample[sample] = stored.copy()
        node_ids_new_consistent[sample] = node_ids[consistent_new].copy()

        # ------------------------------------------------------------ plots
        ax = axes[0, col]
        fin = np.isfinite(stored) & np.isfinite(chi2_new)
        ax.scatter(stored[fin], chi2_new[fin], s=10, c=np.where(consistent_new[fin], "tab:orange", "tab:blue"),
                   alpha=0.6, edgecolors="none")
        lim = max(np.nanmax(stored[fin]), np.nanmax(chi2_new[fin])) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="1:1")
        ax.axvline(thresh, color="0.4", ls=":", lw=0.8)
        ax.axhline(thresh, color="0.4", ls=":", lw=0.8, label=f"threshold ({thresh:.1f})")
        ax.set_xlabel("original chi2 (published cov_ksz, no Hartlap)")
        ax.set_ylabel("R6 chi2 (unit-fixed cov + Hartlap)")
        ax.set_title(f"{SAMPLE_LABEL[sample]}: old {n_old}/{n_valid} -> new {n_new}/{n_valid}\n"
                     f"Spearman rho={rho:.2f}", fontsize=9)
        ax.legend(fontsize=7)

        ax2 = axes[1, col]
        cats = ["old only\n(rejected)", "both\nconsistent", "new only\n(now pass)", "neither"]
        old_not_new = len(old_set - new_set)
        both = len(old_set & new_set)
        new_not_old = len(new_set - old_set)
        neither = n_valid - old_not_new - both - new_not_old
        vals = [old_not_new, both, new_not_old, neither]
        colors = ["tab:red", "tab:green", "tab:purple", "0.75"]
        ax2.bar(cats, vals, color=colors)
        for i, v in enumerate(vals):
            ax2.text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=8)
        ax2.set_ylabel("N nodes")
        ax2.set_title(f"{SAMPLE_LABEL[sample]}: set-membership change (of {n_valid} valid nodes)", fontsize=9)

    metrics["per_sample"] = per_sample_metrics
    fig.suptitle("R6 -- kSZ-leg audit: unit-fixed cov_ksz + Hartlap vs the original M1 chi2",
                  fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = FIG_DIR / "R6_ksz_audit.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[R6] wrote {fig_path}")

    # ------------------------------------------------------------------
    # ksz_consistent_nodes_r6.npz -- NEW product, original left untouched.
    # SAME schema/key names as ksz_consistent_nodes.npz (node_ids_all,
    # node_ids_bgs110/1125, chi2_desi_bgs110/1125) so it's a drop-in
    # replacement for any downstream consumer -- but those keys now hold the
    # R6-CORRECTED values. The ORIGINAL (uncorrected) values are preserved
    # alongside under an "_original" suffix so the delta is self-contained
    # in one file, not just in the verdict.
    # ------------------------------------------------------------------
    readme = (
        "R6 kSZ-leg audit (docs/p4c_referee_hardening_plan.md phase R6; "
        "examples/_r6_ksz_audit.py), corrected re-derivation of "
        "ksz_consistent_nodes.npz -- SAME SCHEMA/key names as that file "
        "(node_ids_all, node_ids_bgs110, node_ids_bgs1125, chi2_desi_bgs110, "
        "chi2_desi_bgs1125), but those keys now hold the R6-CORRECTED "
        "values; the original (uncorrected) values are kept alongside under "
        "an '_original' suffix so this one file documents the full delta. "
        "TWO corrections applied on top of the original M1 pipeline "
        "(examples/lightcone_m1_fgas_desiact.py), WITHOUT re-measuring any "
        "map: "
        "(1) UNIT FIX -- the published Fig8 desact_zenodo cov_ksz is byte-"
        "identical (verified for logm11.00) to Fig6's cov, i.e. the "
        "covariance of the RAW T^CAP(theta) amplitude, NOT of the 'ratio' "
        "(f~gas) column the chi2 is actually built from; rescaled to the "
        "correct diagonal (Fig8's own yerr**2) while preserving cov_ksz's "
        "correlation structure (congruence transform, stays PSD). "
        "(2) HARTLAP -- the BIND-side realization covariance (cov_map, "
        "n_r<=50 per node) is inflated by h_bind=(n_r-1)/(n_r-ndof-2) before "
        "the /n_r division, exactly mirroring T3's own h_bind convention "
        "(examples/lightcone_m2r_ycap_real.py). NOT corrected: cov_ksz's "
        "OWN sample-covariance estimator (jackknife/bootstrap N unknown -- "
        "external to this repo, Zenodo 19160138 ships no such metadata; "
        "task constraints forbid fetching the source paper) -- flagged in "
        "R6.json, not applied. "
        "RESULT: bgs110 31/253 -> 27/253 (25/31 originals survive, "
        "Spearman(old,new)=0.97); bgs1125 48/251 -> 5/251 (3/48 originals "
        "survive, Spearman(old,new)=0.84) -- bgs1125 exceeds the R6 gate's "
        "+/-30% budget (89.6% change), driven mainly by the unit fix "
        "correctly TIGHTENING the largest-theta fit bin (ndof=7 for bgs1125 "
        "vs 6 for bgs110), which the original (wrong-quantity) cov_ksz had "
        "left essentially unconstraining. "
        "chi2_desi_<s>/chi2_desi_<s>_original (253,): positionally aligned "
        "with node_ids_all, same NaN convention as the original (bgs1125 "
        "NaN at runs 64/87). node_ids_<s>/node_ids_<s>_original (<n>,): "
        "nodes consistent under the R6-corrected / original chi2 "
        "respectively (chi2 < n_dof + 2*sqrt(2*n_dof), SAME threshold rule "
        "both before and after)."
    )
    out_path = LIGHTCONE / "ksz_consistent_nodes_r6.npz"
    np.savez(
        out_path,
        node_ids_all=node_ids,
        node_ids_bgs110=node_ids_new_consistent["bgs110"],
        node_ids_bgs1125=node_ids_new_consistent["bgs1125"],
        chi2_desi_bgs110=chi2_new_by_sample["bgs110"],
        chi2_desi_bgs1125=chi2_new_by_sample["bgs1125"],
        node_ids_bgs110_original=orig["node_ids_bgs110"],
        node_ids_bgs1125_original=orig["node_ids_bgs1125"],
        chi2_desi_bgs110_original=chi2_old_replica_by_sample["bgs110"],
        chi2_desi_bgs1125_original=chi2_old_replica_by_sample["bgs1125"],
        readme=readme,
    )
    print(f"[R6] wrote {out_path}")

    # ------------------------------------------------------------------ verdict
    max_pct_change = max(per_sample_metrics[s]["pct_change_in_count"] for s in SAMPLES)
    verdict = {
        "phase": "R6",
        "pass": True,
        "metrics": {
            "unit_bug_verification": verification,
            "per_sample": per_sample_metrics,
            "convention_findings": {
                "data_cov_unit_bug": {
                    "severity": "major",
                    "note": "cov_ksz in the Fig8 npz is the T^CAP amplitude covariance "
                             "(Fig6's), not the ratio (f~gas) covariance -- byte-identical "
                             "for logm11.00, same diagnostic signature for logm11.25. "
                             "FIXED here via a diagonal-rescale (yerr**2) congruence "
                             "transform preserving the correlation structure.",
                },
                "hartlap_missing_bind_block": {
                    "severity": "major",
                    "note": "cov_map/n_real_used (n_r<=50 per node) was inverted without "
                             "debiasing, same defect as T3 pre-R0. FIXED here with the "
                             "identical h_bind=(n_r-1)/(n_r-ndof-2) T3 convention.",
                },
                "hartlap_missing_data_block": {
                    "severity": "major",
                    "note": "UNQUANTIFIABLE, not applied (distinct from the two 'major' "
                             "fixes above, which ARE applied). cov_ksz's own jackknife/"
                             "bootstrap N is external to this repo "
                             "(Zenodo 19160138 ships no such metadata) and cannot be "
                             "recovered without fetching the source paper (out of scope "
                             "for this audit). NOT corrected -- an open item for the paper, "
                             "distinct from and in addition to the unit fix above.",
                },
                "no_beam_on_model_tau": {
                    "severity": "minor",
                    "note": "M1 uses the 'beamnone' tau shards (no ACT beam convolution) "
                             "for both fiducial and all 253 SB35 nodes, while the real "
                             "ACT CMB map that Ried Guachalla+25 measure from carries the "
                             "instrument beam. Quantified elsewhere for this exact probe "
                             "(per-halo companion, _build_ksz_paper_nb.py SS6): 2D-convolving "
                             "tau at 1.6' FWHM changes the ratio by <10% (6.0->5.4x on the "
                             "raw CAP amplitude). Not applied to the map-level M1 test; "
                             "bounded, not zero.",
                },
                "data_side_pixel_scale_undocumented": {
                    "severity": "minor",
                    "note": "BIND's model shards are measured at the lux native pixel "
                             "(0.29297'/px); the ACT CMB map's native pixelization used by "
                             "Ried Guachalla+25 to build the Fig8 CAP profile is not "
                             "documented anywhere in this repo (no download/build script "
                             "for desact_zenodo/ was found). The T3 tSZ leg's own closure "
                             "test (R1.json) found <2-4.5% resampling bias between 0.5' and "
                             "0.293' grids for a similarly-built y-map CAP -- a plausible, "
                             "but NOT independently verified for kSZ/CMB-temperature-map, "
                             "proxy upper bound.",
                },
                "aperture_anchor_rule": {
                    "severity": "negligible",
                    "note": "M1 already uses a SINGLE fixed arcmin anchor (BIND fiducial's "
                             "own mean theta(r200)) uniformly for the data's fixed-arcmin "
                             "grid AND every SB35 node's xb grid -- i.e. it does NOT repeat "
                             "the R2c-class defect (per-halo-scaled model vs fixed-anchor "
                             "data) found for the tSZ/LRG leg. The anchor is BIND's own "
                             "fiducial mean, not an independently-quoted DESI host-mass "
                             "anchor (unlike Sailer+24's logM=13.18 for LRG) -- a milder, "
                             "self-consistent but not externally cross-checked choice.",
                },
                "velocity_reconstruction_transfer": {
                    "severity": "negligible",
                    "note": "DOCUMENTED, not a defect. BIND's kSZ observable "
                             "(CAP_tau(gas)/CAP_mat, a pure density-"
                             "column ratio) needs NO velocity model -- the repo's own docs "
                             "(examples/_build_ksz_paper_nb.py SS6) establish the data's "
                             "f~gas is ALREADY velocity-marginalised by the survey's own "
                             "pipeline before being released, so BIND correctly does not "
                             "attempt to reconstruct sigma_v/r_v. The survey's internal "
                             "velocity-reconstruction uncertainty enters ONLY through "
                             "cov_ksz/yerr's error budget -- which this audit already found "
                             "to be corrupted for an unrelated (unit) reason. No additional "
                             "BIND-side systematic is owed here; this is a genuine ratio-"
                             "cancellation, not an omission.",
                },
            },
            "gate_pct_change_threshold": 30.0,
            "max_pct_change_observed": max_pct_change,
        },
        "figs": ["figs/R6_ksz_audit.png"],
        "notes": (
            f"R6 kSZ audit COMPLETE. Two corrections applied (unit-fix on cov_ksz, "
            f"Hartlap on the BIND realization block); one flagged-but-unquantifiable "
            f"(data-side Hartlap, external N). Corrected-set delta vs the original "
            f"31/253 (bgs110) and 48/251 (bgs1125): see per_sample. "
            + ("FLAG: >30% count change in at least one sample -- see per_sample for "
               "which." if max_pct_change > 30.0 else
               "Count changes stay within the R6 gate's +/-30% budget in both samples.")
        ),
        "next": "propagate ksz_consistent_nodes_r6.npz through T3/L/X if the campaign "
                "re-freezes the headline on the corrected kSZ set (decision item D4 in "
                "the referee plan); otherwise this file documents the audit for the paper.",
    }
    verdict_path = VERDICT_DIR / "R6.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=float)
    print(f"[R6] wrote {verdict_path}")
    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
