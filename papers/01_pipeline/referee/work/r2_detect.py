"""R2 deliverables 2+3: the NOISY twin of paper Fig 7 (fig23_s3_opener) and the
per-statistic detectability table under LSST-Y10 shape noise.

Figure: the 5-95% Sobol spread of S(ell) at z_s=1 against LSST-Y10 and
Euclid-like precision, both computed from the covariance MEASURED on the noisy
(shape-noise + 1'-smoothed) fiducial realizations, area-scaled to each survey.

Smoothing bookkeeping (documented choice):
  * every noisy map is Gaussian-smoothed, so its spectrum is W^2(ell)[C_ell + N_ell].
  * S(ell) is a RATIO of two such spectra, so W^2 cancels identically. What does
    NOT cancel is the additive shape-noise floor, so the floor is subtracted
    first, using Nhat(ell) MEASURED by pushing pure shape noise through the same
    smooth+power_spectrum path (r2_common.measure_noise_bias_cl). The measured
    Nhat reproduces exp(-ell^2 sigma_theta^2) x sigma_e^2/(2 n_gal) to <0.03%,
    so the debiasing is exact, not a fit.
  * the precision band is the scatter of the SAME debiased quantity across the
    noisy fiducial realizations, so it carries the same W^2 cancellation and the
    shape-noise-inflated variance. Restricted to ell <= 8000, where W^2 > 0.066.

Writes  imgs/fig23n_s3_opener_noisy.{png,pdf}
        ceph/referee_work/r2/detectability.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import setup, panel_label, COLORS  # noqa: E402

import r2_common as R  # noqa: E402

IMGS = "/mnt/home/mlee1/BIND/imgs"
ELL_LO, ELL_HI = 100.0, 8.0e3
ZI = R.ZI
NU_KEYS = R.NU_KEYS


def save2(fig, stem):
    fig.savefig(f"{IMGS}/{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{IMGS}/{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {IMGS}/{stem}.png + .pdf")


def main() -> None:
    setup()
    out: dict = {}

    nb = R.measure_noise_bias_cl()
    ell = nb["ell"]
    nhat = nb["nhat"]

    fid = R.load_noisy_target("fid")
    dmo = R.load_noisy_target("dmo")
    sob = R.load_noisy_sobol()
    N_NODE = 50                       # realizations behind every Sobol node

    keep = (ell >= ELL_LO) & (ell <= ELL_HI)
    ellk, nhk = ell[keep], nhat[keep]
    cen, masks = R.log_ell_bins(ellk, ELL_LO, ELL_HI)

    # ── debiased signal spectra, z_s = 1 ─────────────────────────────────────
    cl_d = dmo["cl_kappa"][ZI][keep] - nhk                     # fixed reference
    cl_f_r = fid["cl_kappa_real"][:, ZI, :][:, keep] - nhk     # (550, n_ell)
    cl_s = sob["t__cl_kappa__value"][:, ZI, :][:, keep] - nhk  # (256, n_ell)

    S_f_r = R.rebin_rows(cl_f_r / cl_d, masks)                 # (550, nb)
    S_s = R.rebin_rows(cl_s / cl_d, masks)                     # (256, nb)
    S_f = S_f_r.mean(0)
    cl_f_rb = R.rebin_rows(cl_f_r, masks)
    nh_rb = R.rebin_rows(nhk[None], masks)[0]
    cl_f_sig = cl_f_rb.mean(0)

    # ── survey precision on S(ell) ───────────────────────────────────────────
    # measured: relative scatter of the debiased signal across noisy realizations
    rel_meas = S_f_r.std(0) / S_f                              # 25 deg^2, LSST noise
    sig_lsst = rel_meas * np.sqrt(R.area_scale(R.AREA_LSST))
    # Euclid: the mode-counting factor is the same (verified: the measured
    # relative scatter of Cl^tot tracks sqrt(2/N_modes) to <2% for ell>3000), so
    # only the shot-noise floor is swapped.
    tot = cl_f_sig + nh_rb
    nh_euc = nh_rb * (R.shot_noise_cl(R.NGAL_EUCLID, R.SIGE_EUCLID)
                      / R.shot_noise_cl(R.NGAL_LSST, R.SIGE_LSST))
    rel_mode = rel_meas * cl_f_sig / tot                       # relative error on the TOTAL
    sig_euclid = rel_mode * (cl_f_sig + nh_euc) / cl_f_sig * np.sqrt(R.area_scale(R.AREA_EUCLID))
    # The cache is built at the Lee+22 Eq.7 noise level, N = sigma_e^2/(2 n_gal).
    # The LSST-DESC / paper-cl_relerr convention is TWICE that. Carry both.
    sig_lsst_std = rel_mode * (cl_f_sig + 2 * nh_rb) / cl_f_sig * np.sqrt(R.area_scale(R.AREA_LSST))
    sig_euclid_std = (rel_mode * (cl_f_sig + 2 * nh_euc) / cl_f_sig
                      * np.sqrt(R.area_scale(R.AREA_EUCLID)))

    # node-to-node measurement floor on the Sobol spread. All Sobol runs share the
    # ray-seed ladder, so their common cosmic variance does NOT broaden the spread;
    # only their independent shape-noise draws do. For a spectrum the split is
    # exact: Var(Chat) = C^2 + (2CN + N^2), so f_noise = 1 - [C/(C+N)]^2.
    f_noise_cl = 1.0 - (cl_f_sig / tot) ** 2
    # in units of sigma_LSST (dimensionless), then converted to S units
    floor_sigma_u = np.sqrt(f_noise_cl / N_NODE / R.area_scale(R.AREA_LSST))
    floor_span_u = 2 * 1.6449 * floor_sigma_u        # a 5-95% span from pure noise
    floor_span = floor_span_u * sig_lsst             # same units as `span`

    # the converged (noiseless, seed-paired) prior span on the same bins, as the
    # response reference: for a spectrum the shape-noise bias is target-independent
    # so the RESPONSE is noise-free in expectation, only the covariance is inflated.
    d0s = np.load(R.SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    es0 = d0s["a__suppression__ell"]
    ks0 = (es0 >= ELL_LO) & (es0 <= ELL_HI)
    _, ms0 = R.log_ell_bins(es0[ks0], ELL_LO, ELL_HI)
    S0rb = R.rebin_rows(d0s["t__suppression__value"][:, ZI, :][:, ks0], ms0)
    lo0b, hi0b = np.percentile(S0rb, [5, 95], axis=0)
    span_conv = hi0b - lo0b

    # Analytic Knox cross-check. TWO care points: (i) the paper's cl_relerr uses
    # Nl = sigma_e^2 A_pix/n_gal, i.e. TWICE the Lee+22 Eq.7 noise the cache is
    # built with -- both are printed; (ii) it must be fed the UNSMOOTHED C_ell,
    # since W^2 multiplies signal and noise alike and cancels out of N/C.
    A2SR = (np.pi / 180 / 60) ** 2
    w2_rb = R.rebin_rows(nb["w2"][keep][None], masks)[0]
    cl_unsm = cl_f_sig / w2_rb

    def knox(cl, ngal, sige, fsky, dlnl=0.15, ellv=cen, halve=True):
        Nl = sige ** 2 * A2SR / ngal / (2.0 if halve else 1.0)
        dl = np.maximum(ellv * dlnl, 1.0)
        return np.sqrt(2.0 / ((2 * ellv + 1) * dl * fsky)) * (1 + Nl / cl)

    knox_lsst = knox(cl_unsm, 27.0, 0.26, R.FSKY_LSST)
    knox_euc = knox(cl_unsm, 30.0, 0.30, R.FSKY_EUCLID)
    knox_lsst_paper = knox(cl_unsm, 27.0, 0.26, R.FSKY_LSST, halve=False)
    knox_euc_paper = knox(cl_unsm, 30.0, 0.30, R.FSKY_EUCLID, halve=False)

    lo, hi = np.percentile(S_s, [5, 95], axis=0)
    span = hi - lo
    i5k = int(np.argmin(np.abs(cen - 5000)))
    i2k = int(np.argmin(np.abs(cen - 2000)))
    i1k = int(np.argmin(np.abs(cen - 1000)))
    out["sl_band"] = dict(
        ell=[float(x) for x in cen],
        S_fid=[float(x) for x in S_f],
        S_lo=[float(x) for x in lo], S_hi=[float(x) for x in hi],
        sigma_lsst=[float(x) for x in sig_lsst],
        sigma_euclid=[float(x) for x in sig_euclid],
        sigma_lsst_stdconv=[float(x) for x in sig_lsst_std],
        sigma_euclid_stdconv=[float(x) for x in sig_euclid_std],
        knox_lsst=[float(x) for x in knox_lsst],
        knox_euclid=[float(x) for x in knox_euc],
        knox_lsst_paperconv=[float(x) for x in knox_lsst_paper],
        knox_euclid_paperconv=[float(x) for x in knox_euc_paper],
        span_converged=[float(x) for x in span_conv],
        floor_span=[float(x) for x in floor_span],
        floor_span_sigma_units=[float(x) for x in floor_span_u],
        f_noise_cl=[float(x) for x in f_noise_cl],
        span_over_sigma_lsst=[float(x) for x in span / sig_lsst],
        span_conv_over_sigma_lsst=[float(x) for x in span_conv / sig_lsst],
    )
    print("S(ell) noisy detectability, z_s=1  (span = Sobol 5-95%; "
          "'conv' = converged noiseless span, the noise-free response):")
    for lbl, i in (("1000", i1k), ("2000", i2k), ("5000", i5k)):
        print(f"  ell~{cen[i]:6.0f}: span_noisy {100*span[i]:6.2f}% "
              f"span_conv {100*span_conv[i]:6.2f}% floor {100*floor_span[i]:5.2f}% "
              f"| sig_LSST {100*sig_lsst[i]:5.3f}% (std-conv {100*sig_lsst_std[i]:5.3f}%, "
              f"Knox {100*knox_lsst[i]:5.3f}%) -> {span_conv[i]/sig_lsst[i]:5.1f}x "
              f"(std-conv {span_conv[i]/sig_lsst_std[i]:5.1f}x) "
              f"| Euclid -> {span_conv[i]/sig_euclid[i]:5.1f}x")
    j = int(np.nanargmax(span_conv / sig_lsst))
    print(f"  peak detectability at ell~{cen[j]:.0f}: {span_conv[j]/sig_lsst[j]:.0f}x LSST-Y10, "
          f"{span_conv[j]/sig_euclid[j]:.0f}x Euclid "
          f"({span_conv[j]/sig_lsst_std[j]:.0f}x / {span_conv[j]/sig_euclid_std[j]:.0f}x "
          f"under the 2x-noise standard convention)")
    print(f"  ell where the converged span first exceeds 5 sigma_LSST: "
          f"{cen[np.argmax(span_conv/sig_lsst > 5)]:.0f}; "
          f"where the 50-real measurement floor drops below the span: "
          f"{cen[np.argmax(span_conv > floor_span)]:.0f}")
    out["sl_headline"] = dict(
        ell_peak=float(cen[j]), span_peak_pct=float(100 * span[j]),
        x_lsst_peak=float(span[j] / sig_lsst[j]),
        x_euclid_peak=float(span[j] / sig_euclid[j]),
        x_lsst_5000=float(span[i5k] / sig_lsst[i5k]),
        x_euclid_5000=float(span[i5k] / sig_euclid[i5k]),
        knox_over_meas_5000=float(knox_lsst[i5k] / sig_lsst[i5k]),
        knox_paper_conv_5000_pct=float(100 * knox_lsst_paper[i5k]))

    # The paper's own fig-23 number, recomputed both ways at ell = 5000, z_s = 1:
    # noiseless Sobol span over the analytic Knox sigma (what the notebook prints)
    # vs the same span over the MEASURED noisy covariance.
    d0p = np.load(R.SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    e0p = d0p["a__suppression__ell"]
    ip = int(np.argmin(np.abs(e0p - 5000)))
    S0 = d0p["t__suppression__value"][:, ZI, ip]
    lo0, hi0 = np.percentile(S0, [5, 95])
    span0 = float(hi0 - lo0)
    print("\npaper fig-23 style number at ell=5000, z_s=1:")
    print(f"  NOISELESS Sobol 5-95% span of S = {100*span0:.2f}%")
    print(f"    / Knox sigma (paper cl_relerr convention) {100*knox_lsst_paper[i5k]:.3f}% "
          f"-> {span0/knox_lsst_paper[i5k]:.0f}x LSST-Y10")
    print(f"    / Knox sigma (Lee+22 Eq.7 noise)          {100*knox_lsst[i5k]:.3f}% "
          f"-> {span0/knox_lsst[i5k]:.0f}x")
    print(f"  NOISY (measured cov.) span {100*span[i5k]:.2f}% "
          f"/ sigma {100*sig_lsst[i5k]:.3f}% -> {span[i5k]/sig_lsst[i5k]:.0f}x LSST-Y10, "
          f"{span[i5k]/sig_euclid[i5k]:.0f}x Euclid")
    out["fig23_compare_5000"] = dict(
        span_noiseless_pct=100 * span0, span_noisy_pct=float(100 * span[i5k]),
        knox_paper_pct=float(100 * knox_lsst_paper[i5k]),
        knox_l22_pct=float(100 * knox_lsst[i5k]),
        sigma_meas_noisy_pct=float(100 * sig_lsst[i5k]),
        x_noiseless_paperconv=float(span0 / knox_lsst_paper[i5k]),
        x_noisy_measured=float(span[i5k] / sig_lsst[i5k]),
        x_noisy_measured_euclid=float(span[i5k] / sig_euclid[i5k]))

    # ── per-statistic detectability table ────────────────────────────────────
    thr3, thr5 = {}, {}
    table = []

    # measured shape-noise fraction of the noisy covariance, per nu statistic
    # (r2_noisegain.py). Used to make the node-to-node measurement floor correct:
    # the Sobol runs share the ray-seed ladder, so only the independent shape-noise
    # leg broadens the node-to-node spread.
    try:
        with open(R.OUT / "noise_split.json") as f:
            FNOISE = {k: np.array(v["f_noise_bins"], float)
                      for k, v in json.load(f).items() if not k.startswith("_")}
    except FileNotFoundError:
        FNOISE = {}
    # the ell-domain analogue is exact, not measured: Var(Chat) = C^2 + (2CN + N^2)
    FNOISE["cl_kappa"] = f_noise_cl
    FNOISE["suppression"] = f_noise_cl
    # the FULL measured shape-noise covariance for the six nu statistics
    try:
        _cn = np.load(R.OUT / "noise_split_cov.npz")
        CN_MEAS = {k: _cn[f"Cn_{k}"] for k in NU_KEYS if f"Cn_{k}" in _cn.files}
    except FileNotFoundError:
        CN_MEAS = {}

    def detect_row(name, node_vals, fid_real, case):
        """chi^2 of every node against the fiducial, area-scaled noisy covariance."""
        f = fid_real.mean(0)
        C25 = np.cov(fid_real, rowvar=False)
        d = C25.shape[0]
        n = fid_real.shape[0]
        C = C25 * R.area_scale(R.AREA_LSST)
        h = R.hartlap(n, d)
        Cinv = np.linalg.pinv(C) * h
        D = node_vals - f
        x2 = np.einsum("ij,jk,ik->i", D, Cinv, D)
        # measurement floor: a node identical to the fiducial still shows this
        # much chi^2 from its own 50 realizations + the fiducial's N (upper bound;
        # the shared ray-seed ladder cancels the cosmic-variance leg, so the true
        # floor is smaller -- see the memo).
        floor = float(np.trace((C25 * (1.0 / N_NODE + 1.0 / n)) @ Cinv))
        # Seed-paired floor. Only the SHAPE-NOISE leg of the covariance is drawn
        # independently by a Sobol node and by the fiducial -- all 256 runs and the
        # fiducial share the ray-tracing seed ladder, so their cosmic variance is
        # common-mode and cancels out of the node-to-fiducial difference. The floor
        # covariance is therefore C25 scaled bin-wise by the shape-noise variance
        # fraction f: MEASURED for the six nu statistics (r2_noisegain.py, 120
        # realizations, two independent noise streams on the same maps), and exact
        # for the spectra (Var(Chat) = C^2 + [2CN + N^2] => f = 1 - [C/(C+N)]^2).
        # NOISELESS rows have NO independent leg at all -- their nodes and fiducial
        # are literally the same maps -- so f = 0 there and their RAW columns are
        # the fair comparison.
        if case != "noisy":
            Cn25 = np.zeros_like(C25)
            fnz = 0.0
        elif name in CN_MEAS:                       # measured full matrix
            Cn25 = CN_MEAS[name]
            fnz = float(np.median(np.diag(Cn25) / np.diag(C25)))
        elif name in ("cl_kappa", "suppression"):   # analytic, per bin
            rt = np.sqrt(np.clip(f_noise_cl, 0, 1))
            Cn25 = C25 * np.outer(rt, rt)
            fnz = float(np.median(f_noise_cl))
        else:
            Cn25, fnz = C25.copy(), 1.0
        floor_pair = float(np.trace((Cn25 * (1.0 / N_NODE + 1.0 / n)) @ Cinv))
        t3, t5 = R.sigma_thresholds(d)
        thr3[name], thr5[name] = t3, t5
        x2c = np.clip(x2 - floor_pair, 0, None)
        # span/sigma headline at the peak-response bin
        loq, hiq = np.percentile(node_vals, [5, 95], axis=0)
        sd = np.sqrt(np.diag(C))
        ratio = (hiq - loq) / np.where(sd > 0, sd, np.nan)
        jb = int(np.nanargmax(ratio))
        # conservative variant: with C_meas = C25 (1/50 + 1/N) the product
        # C_meas Cinv is exactly (1/50 + 1/N)(A/25) h times the identity, so the
        # null distribution of chi2_raw is (floor/d) x chi2_d and the "our suite
        # can demonstrate it" threshold is (floor/d) x thr.
        return dict(
            stat=name, case=case, dof=d, n_cov=n,
            frac_gt3=float(np.mean(x2 > t3)), frac_gt5=float(np.mean(x2 > t5)),
            frac_gt3_floorcorr=float(np.mean(x2c > t3)),
            frac_gt5_floorcorr=float(np.mean(x2c > t5)),
            frac_gt3_suitedemo=float(np.mean(x2 > (floor_pair / d) * t3)),
            frac_gt5_suitedemo=float(np.mean(x2 > (floor_pair / d) * t5)),
            floor_chi2=floor_pair, floor_chi2_dof=floor_pair / d,
            floor_chi2_unpaired=floor, f_noise_used=fnz,
            chi2_median=float(np.median(x2)), chi2_p5=float(np.percentile(x2, 5)),
            chi2_p95=float(np.percentile(x2, 95)),
            sigma_median=R.chi2_to_sigma(float(np.median(x2)), d),
            sigma_median_floorcorr=R.chi2_to_sigma(float(np.median(x2c)), d),
            span_over_sigma_peak=float(ratio[jb]), peak_bin=jb,
            span_over_sigma_median=float(np.nanmedian(ratio)),
            thr3=t3, thr5=t5)

    for k in NU_KEYS:
        table.append(detect_row(k, sob[f"t__{k}__value"][:, ZI, :],
                                fid[f"{k}_real"][:, ZI, :], "noisy"))
    table.append(detect_row("cl_kappa", R.rebin_rows(cl_s, masks), cl_f_rb, "noisy"))
    table.append(detect_row("suppression", S_s, S_f_r, "noisy"))

    # noiseless counterparts, same machinery, N=50 covariance
    d0 = np.load(R.SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    nb0, _ = R.load_noiseless_nu()
    for k in NU_KEYS:
        table.append(detect_row(k, d0[f"t__{k}__value"][:, ZI, :],
                                nb0[f"{k}_real"][:, ZI, :], "noiseless"))
    fc = R.load_noiseless_fields()
    if fc is not None:
        e0 = fc["ell"]
        k0 = (e0 >= ELL_LO) & (e0 <= ELL_HI)
        _, m0 = R.log_ell_bins(e0[k0], ELL_LO, ELL_HI)
        kb0 = R.rebin_rows(fc["kk_bind"][:, ZI][:, k0], m0)
        e1 = d0["a__cl_kappa__ell"]
        k1 = (e1 >= ELL_LO) & (e1 <= ELL_HI)
        _, m1 = R.log_ell_bins(e1[k1], ELL_LO, ELL_HI)
        cls0 = R.rebin_rows(d0["t__cl_kappa__value"][:, ZI, ZI, :][:, k1], m1)
        table.append(detect_row("cl_kappa", cls0, kb0, "noiseless"))
        # No separate noiseless "suppression" row: S = C/C_dmo with C_dmo a fixed
        # reference divides numerator and covariance by the same number, so its
        # chi2 is identical to cl_kappa's by construction (confirmed on the noisy
        # side, where the two rows agree to 0.2%).

    out["detect_rows"] = table
    print(f"\n{'case':>10s} {'stat':>14s} {'dof':>4s} {'chi2_med':>10s} {'sig_med':>8s} "
          f"{'>3sig':>7s} {'>5sig':>7s} {'>3s(fc)':>8s} {'>5s(fc)':>8s} "
          f"{'>3s(dem)':>9s} {'floor/dof':>10s} {'span/sig':>9s} {'span/sig_med':>13s}")
    for r in table:
        print(f"{r['case']:>10s} {r['stat']:>14s} {r['dof']:>4d} {r['chi2_median']:>10.1f} "
              f"{r['sigma_median']:>8.1f} {100*r['frac_gt3']:>6.1f}% {100*r['frac_gt5']:>6.1f}% "
              f"{100*r['frac_gt3_floorcorr']:>7.1f}% {100*r['frac_gt5_floorcorr']:>7.1f}% "
              f"{100*r['frac_gt3_suitedemo']:>8.1f}% "
              f"{r['floor_chi2_dof']:>10.2f} {r['span_over_sigma_peak']:>9.1f} "
              f"{r['span_over_sigma_median']:>13.1f}")

    # ── the figure ───────────────────────────────────────────────────────────
    cH = COLORS["truth"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.5 * 1.35, 3.9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.08))

    ax.fill_between(cen, lo, hi, color="0.86", lw=0, zorder=0,
                    label="Sobol prior 5–95%, as measured (noisy)")
    # The converged (noiseless, seed-paired) 5-95% envelope. The released
    # noiseless suppression carries a known multiplicative DMO-denominator
    # offset (its 50-realization numerators sit over a 550-realization DMO
    # mean), which shifts the whole envelope but leaves its WIDTH -- the only
    # thing used numerically -- intact to 3%. It is therefore renormalised bin
    # by bin onto the noisy median before drawing.
    ren = np.median(S_s, axis=0) / np.median(S0rb, axis=0)
    ax.plot(cen, lo0b * ren, color="0.30", lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.plot(cen, hi0b * ren, color="0.30", lw=0.7, ls=(0, (4, 2)), zorder=2,
            label="same, converged (noiseless) response")
    ax.fill_between(cen, S_f - sig_lsst, S_f + sig_lsst, color=COLORS["highlight"],
                    alpha=0.55, lw=0, zorder=3, label=r"LSST-Y10 $\pm1\sigma$ (noisy)")
    ax.plot(cen, S_f - sig_euclid, color=COLORS["secondary"], ls="-.", lw=0.9, zorder=3)
    ax.plot(cen, S_f + sig_euclid, color=COLORS["secondary"], ls="-.", lw=0.9, zorder=3)
    ax.plot([], [], color=COLORS["secondary"], ls="-.", lw=0.9,
            label=r"\textit{Euclid} $\pm1\sigma$ (noisy)" if False
            else r"Euclid $\pm1\sigma$ (noisy)")
    ax.plot(cen, S_f, color=cH, lw=1.4, zorder=4, label="fiducial (TNG)")
    ax.axhline(1, color=COLORS["dmo"], ls=":", lw=0.8, zorder=2)
    ax.set_ylabel(r"$S(\ell)=C_\ell^{\kappa\kappa}/C_\ell^{\kappa\kappa,\rm dmo}$")
    ax.set_xscale("log")
    ax.set_xlim(ELL_LO, ELL_HI)
    ax.set_ylim(0.78, 1.10)
    ax.legend(fontsize=5.4, loc="lower left", ncol=1)
    panel_label(ax, "(a)", loc="upper right")

    axr.fill_between(cen, 0, span_conv / sig_lsst, color=COLORS["highlight"], alpha=0.30, lw=0)
    axr.plot(cen, span_conv / sig_lsst, color=COLORS["highlight"], lw=1.2, label="LSST-Y10")
    axr.plot(cen, span_conv / sig_euclid, color=COLORS["secondary"], ls="-.", lw=1.0,
             label="Euclid")
    axr.plot(cen, span_conv / sig_lsst_std, color=COLORS["highlight"], lw=0.7, ls=":",
             label=r"LSST-Y10, $2\times$ noise conv.")
    axr.plot(cen, floor_span_u, color="0.35", lw=0.8, ls=(0, (4, 2)),
             label="50-real. measurement floor")
    axr.axhline(1, color="0.4", lw=0.7, ls=":")
    axr.axhline(5, color="0.7", lw=0.5, ls="--")
    axr.set_yscale("log")
    axr.set_ylim(0.7, 200)
    axr.set_ylabel(r"prior span / $\sigma_{\rm survey}$", fontsize=7)
    axr.set_xlabel(r"$\ell$")
    axr.set_xscale("log")
    axr.legend(fontsize=4.6, loc="upper left", ncol=2, columnspacing=0.8,
               handlelength=1.6, bbox_to_anchor=(0.08, 1.0))
    axr.tick_params(labelsize=6)
    ax.tick_params(labelsize=6)
    panel_label(axr, "(b)", loc="upper left")
    save2(fig, "fig23n_s3_opener_noisy")
    plt.close(fig)

    with open(R.OUT / "detectability.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {R.OUT/'detectability.json'}")


if __name__ == "__main__":
    main()
