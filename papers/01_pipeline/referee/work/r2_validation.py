"""R2 deliverable 1: the NOISY twin of paper Fig 5 (fig05_field_validation).

BIND-fiducial vs hydro-pasted-truth for every statistic in the LSST-Y10 noisy
cache, with the survey band and the chi^2 built from the NOISY truth 550-real
covariance (main.tex Eq. cov), Hartlap-debiased (Eq. inv_cov) and area-scaled to
A = 18000 deg^2 (Eq. area_scaling).

Writes  imgs/fig05n_field_validation_noisy.{png,pdf}
        ceph/referee_work/r2/validation_numbers.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from paper_style import setup, COLORS  # noqa: E402

import r2_common as R  # noqa: E402
import noisy_grid as ng  # noqa: E402

IMGS = "/mnt/home/mlee1/BIND/imgs"
# 1' smoothing drives W^2 below 7% above ell~8000; beyond that the noise-debiased
# signal Chat = Cl^tot - Nhat is a <15% difference of two large numbers and the
# ratio S(ell) goes numerically unstable, worst at z_s=0.5. 8000 is the display
# AND analysis cut for every ell-domain quantity in this figure.
ELL_LO, ELL_HI = 100.0, 8.0e3
ZI = R.ZI
NU = R.NU
ZCOLS = plt.cm.plasma(np.linspace(0.02, 0.82, 5))


def save2(fig, stem):
    fig.savefig(f"{IMGS}/{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{IMGS}/{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {IMGS}/{stem}.png + .pdf")


def main() -> None:
    setup()
    res_json: dict = {}

    nb = R.measure_noise_bias_cl()
    ell = nb["ell"]
    nhat = nb["nhat"]

    fid = R.load_noisy_target("fid")
    tru = R.load_noisy_target("truth")
    dmo = R.load_noisy_target("dmo")
    NR = int(fid["n_real"])
    assert int(tru["n_real"]) == NR == 550
    assert np.array_equal(fid["cl_ell"], ell)

    # ── ell-domain: debias the shape-noise floor, then log-rebin ──────────────
    keep = (ell >= ELL_LO) & (ell <= ELL_HI)
    ellk, nhk = ell[keep], nhat[keep]
    cen, masks = R.log_ell_bins(ellk, ELL_LO, ELL_HI)

    def sig(target, per_real=False):
        """Noise-debiased kappa auto spectrum at z_s=1 (native ell grid)."""
        a = target["cl_kappa_real"][:, ZI, :] if per_real else target["cl_kappa"][ZI][None]
        return a[:, keep] - nhk

    cl_f_r, cl_t_r, cl_d_r = sig(fid, True), sig(tru, True), sig(dmo, True)
    cl_d = cl_d_r.mean(0)

    # S(ell): numerator per-realization, denominator the fixed 550-real DMO mean
    S_f_r = cl_f_r / cl_d
    S_t_r = cl_t_r / cl_d

    # rebin every per-realization array once
    rb = {
        "cl_kappa": (R.rebin_rows(cl_f_r, masks), R.rebin_rows(cl_t_r, masks)),
        "suppression": (R.rebin_rows(S_f_r, masks), R.rebin_rows(S_t_r, masks)),
    }
    nh_rb = R.rebin_rows(nhk[None], masks)[0]

    # ── noiseless counterparts ───────────────────────────────────────────────
    nb0, nt0 = R.load_noiseless_nu()
    fc = R.load_noiseless_fields()

    # ── the chi^2 table ──────────────────────────────────────────────────────
    rows = []

    def closure_row(name, b_real, t_real, label_axis, n_sub=None):
        """chi^2 of (BIND mean - truth mean) against the area-scaled truth cov."""
        n = b_real.shape[0] if n_sub is None else n_sub
        b, t = b_real[:n], t_real[:n]
        r = b.mean(0) - t.mean(0)
        C25 = np.cov(t, rowvar=False)
        x2, x2d, dof = R.chi2_of(r, C25, R.AREA_LSST, n)
        # 25 deg^2 (single-lightcone) chi2, for context
        x2b, x2db, _ = R.chi2_of(r, C25, R.AREA_BOX, n)
        # measurement floor: expected chi2 from the finite realization count alone
        Cpair = np.cov(b - t, rowvar=False) / n
        h = R.hartlap(n, dof)
        floor = float(np.trace(Cpair @ (np.linalg.pinv(C25 * R.area_scale(R.AREA_LSST)) * h)))
        band = 100 * np.sqrt(np.diag(C25) * R.area_scale(R.AREA_LSST)) / np.abs(t.mean(0))
        resid = 100 * (b.mean(0) - t.mean(0)) / np.where(t.mean(0) != 0, t.mean(0), np.nan)
        excess = x2d - floor / dof
        return dict(stat=name, axis=label_axis, n_real=n, dof=dof,
                    chi2=x2, chi2_dof=x2d, chi2_box=x2b, chi2_dof_box=x2db,
                    floor_chi2_dof=floor / dof,
                    chi2_dof_excess=excess,
                    # survey area at which the (floor-corrected) residual reaches
                    # 1 sigma per dof:  chi2 scales linearly with A.
                    area_eq_deg2=float(R.AREA_LSST / x2d) if x2d > 0 else np.inf,
                    area_eq_excess_deg2=(float(R.AREA_LSST / excess) if excess > 0
                                         else np.inf),
                    sigma=R.chi2_to_sigma(x2, dof),
                    med_abs_resid_pct=float(np.nanmedian(np.abs(resid))),
                    med_band_pct=float(np.nanmedian(band)),
                    max_resid_over_band=float(np.nanmax(np.abs(resid) / band)),
                    frac_bins_within_band=float(np.nanmean(np.abs(resid) <= band)))

    # noisy, nu-domain
    for k in R.NU_KEYS:
        rows.append({**closure_row(k, fid[f"{k}_real"][:, ZI, :], tru[f"{k}_real"][:, ZI, :],
                                   "nu"), "case": "noisy550"})
        rows.append({**closure_row(k, fid[f"{k}_real"][:, ZI, :], tru[f"{k}_real"][:, ZI, :],
                                   "nu", n_sub=50), "case": "noisy50"})
    # noisy, ell-domain
    for k, (br, tr) in rb.items():
        rows.append({**closure_row(k, br, tr, "ell"), "case": "noisy550"})
        rows.append({**closure_row(k, br, tr, "ell", n_sub=50), "case": "noisy50"})

    # noiseless, nu-domain (nu05 shards; 50 realizations; DIFFERENT smoothing)
    for k in R.NU_KEYS:
        rows.append({**closure_row(k, nb0[f"{k}_real"][:, ZI, :], nt0[f"{k}_real"][:, ZI, :],
                                   "nu"), "case": "noiseless50"})
    # noiseless, ell-domain from field_cache (50 realizations)
    if fc is not None:
        e0 = fc["ell"]
        k0 = (e0 >= ELL_LO) & (e0 <= ELL_HI)
        cen0, m0 = R.log_ell_bins(e0[k0], ELL_LO, ELL_HI)
        kb = R.rebin_rows(fc["kk_bind"][:, ZI][:, k0], m0)
        kt = R.rebin_rows(fc["kk_truth"][:, ZI][:, k0], m0)
        rows.append({**closure_row("cl_kappa", kb, kt, "ell"), "case": "noiseless50"})
        dmo0 = R.rebin_rows(cl_d[None], masks)[0]     # noisy dmo -- only for shape
        del dmo0
        rows.append({**closure_row("suppression", kb / kt.mean(0), kt / kt.mean(0), "ell"),
                     "case": "noiseless50_ratio_selfnorm"})

    res_json["closure_rows"] = rows
    res_json["noise_recipe"] = dict(
        sigma_e=ng.SIGMA_E, ngal=ng.NGAL_ARCMIN2,
        sigma_pix=float(ng.NOISE_SIGMA_PIX),
        theta_G_arcmin=ng.THETA_G_ARCMIN,
        filter_sigma_arcmin=float(ng.SMOOTH_SIGMA_ARCMIN),
        filter_sigma_pix=float(ng.SMOOTH_SIGMA_PIX),
        measured_flat_Nell=float(nb["flat_level"]),
        analytic_flat_Nell=float(R.shot_noise_cl(R.NGAL_LSST, R.SIGE_LSST)),
        w2_max_frac_dev=float(np.nanmax(np.abs(
            nb["w2"][(ell < 1.2e4)] / nb["w2_analytic"][(ell < 1.2e4)] - 1))),
        kappa_rms=[float(x) for x in fid["kappa_rms"]],
    )

    print(f"\n{'case':>26s} {'stat':>14s} {'dof':>4s} {'x2/dof':>8s} "
          f"{'floor':>7s} {'excess':>8s} {'A_eq[deg2]':>11s} "
          f"{'|res|med%':>10s} {'band%':>9s} {'max|r|/band':>12s} {'in-band':>8s}")
    for r in rows:
        print(f"{r['case']:>26s} {r['stat']:>14s} {r['dof']:>4d} {r['chi2_dof']:>8.2f} "
              f"{r['floor_chi2_dof']:>7.2f} {r['chi2_dof_excess']:>8.2f} "
              f"{r['area_eq_excess_deg2']:>11.0f} "
              f"{r['med_abs_resid_pct']:>10.2f} {r['med_band_pct']:>9.2f} "
              f"{r['max_resid_over_band']:>12.2f} {r['frac_bins_within_band']:>8.2f}")

    # ── the figure ───────────────────────────────────────────────────────────
    cB, cH = COLORS["bind"], COLORS["truth"]

    def prep_nu(k):
        b = fid[f"{k}_real"][:, ZI, :]
        t = tru[f"{k}_real"][:, ZI, :]
        tm = t.mean(0)
        resid = 100 * (b.mean(0) - tm) / np.where(tm != 0, tm, np.nan)
        pt = 100 * np.std(b - t, axis=0) / np.sqrt(NR) / np.where(tm != 0, np.abs(tm), np.nan)
        band = 100 * np.std(t, axis=0) * np.sqrt(R.area_scale(R.AREA_LSST)) / \
            np.where(tm != 0, np.abs(tm), np.nan)
        return resid, pt, band

    panels = []
    f_ell = cen * (cen + 1) / (2 * np.pi)
    cl_f_rb, cl_t_rb = rb["cl_kappa"]
    S_f_rb, S_t_rb = rb["suppression"]
    r_cl = 100 * (cl_f_rb.mean(0) - cl_t_rb.mean(0)) / cl_t_rb.mean(0)
    p_cl = 100 * np.std(cl_f_rb - cl_t_rb, 0) / np.sqrt(NR) / np.abs(cl_t_rb.mean(0))
    b_cl = 100 * np.std(cl_t_rb, 0) * np.sqrt(R.area_scale(R.AREA_LSST)) / np.abs(cl_t_rb.mean(0))
    r_S = 100 * (S_f_rb.mean(0) - S_t_rb.mean(0)) / S_t_rb.mean(0)
    p_S = 100 * np.std(S_f_rb - S_t_rb, 0) / np.sqrt(NR) / np.abs(S_t_rb.mean(0))
    b_S = 100 * np.std(S_t_rb, 0) * np.sqrt(R.area_scale(R.AREA_LSST)) / np.abs(S_t_rb.mean(0))

    # all-plane curves for the top panels. A plane's curve is masked wherever its
    # own debiased signal drops below 10% of the shape-noise floor -- there the
    # ratio of two nearly-equal large numbers carries no information.
    cl_planes, S_planes = [], []
    for z in range(5):
        sf = fid["cl_kappa"][z][keep] - nhk
        sd = dmo["cl_kappa"][z][keep] - nhk
        bad = (sf < 0.10 * nhk) | (sd < 0.10 * nhk)
        sfm = np.where(bad, np.nan, sf)
        cl_planes.append(R.rebin_rows(sfm[None], masks)[0])
        S_planes.append(R.rebin_rows(np.where(bad, np.nan, sf / sd)[None], masks)[0])
    cl_planes = np.stack(cl_planes)
    S_planes = np.stack(S_planes)

    X2 = {r["stat"]: r["chi2_dof"] for r in rows if r["case"] == "noisy550"}
    panels.append(dict(x=cen, yb=f_ell * cl_planes, yt=f_ell * cl_t_rb.mean(0),
                       res=r_cl, pt=p_cl, band=b_cl, tag="(a)", discrete=False,
                       xl=r"$\ell$", yl=r"$\ell(\ell{+}1)\hat C_\ell^{\kappa\kappa}/2\pi$",
                       xs="log", ys="log", xlim=(ELL_LO, ELL_HI), noise=f_ell * nh_rb,
                       x2=X2["cl_kappa"]))
    panels.append(dict(x=cen, yb=S_planes, yt=S_t_rb.mean(0),
                       res=r_S, pt=p_S, band=b_S, tag="(b)", discrete=False,
                       xl=r"$\ell$", yl=r"$S(\ell)=\hat C_\ell^{\rm bind}/\hat C_\ell^{\rm dmo}$",
                       xs="log", ys="linear", xlim=(ELL_LO, ELL_HI), ylim=(0.8, 1.05),
                       x2=X2["suppression"]))

    NU_LIM = (-3, 8)
    _lt2 = max(1e-12, 0.02 * float(np.max(np.abs(tru["mf_v2"][ZI]))))
    meta = [("pdf", r"PDF$(\nu)$", "log", "(c)", 1e-5),
            ("peak_counts", r"$N_{\rm peak}$", "log", "(d)", None),
            ("minima_counts", r"$N_{\rm min}$", "log", "(e)", None),
            ("mf_v0", r"$V_0(\nu)$", "log", "(f)", None),
            ("mf_v1", r"$V_1(\nu)$", "log", "(g)", None),
            ("mf_v2", r"$V_2(\nu)$", "symlog", "(h)", None)]
    for k, yl, ys, tag, ylo in meta:
        r_, p_, b_ = prep_nu(k)
        panels.append(dict(x=NU, yb=fid[k], yt=tru[k][ZI], res=r_, pt=p_, band=b_,
                           tag=tag, discrete=True, xl=r"$\nu$", yl=yl, xs="linear",
                           ys=ys, xlim=NU_LIM, ylo=ylo, linthresh=_lt2, x2=X2[k]))

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(5, 4, height_ratios=[2.2, 1, 0.55, 2.2, 1], hspace=0.18, wspace=0.45)
    AX, AXR = [], []
    for j, p in enumerate(panels):
        r0 = (j // 4) * 3
        a = fig.add_subplot(gs[r0, j % 4])
        ar = fig.add_subplot(gs[r0 + 1, j % 4], sharex=a)
        for zi in range(5):
            a.plot(p["x"], p["yb"][zi], color=ZCOLS[zi], lw=1.0)
        a.plot(p["x"], p["yt"], color=cH, ls="--", lw=0.9)
        if p.get("noise") is not None:
            a.plot(p["x"], p["noise"], color="0.55", ls=":", lw=0.9)
        a.set_xscale(p["xs"])
        if p["ys"] == "symlog":
            a.set_yscale("symlog", linthresh=p["linthresh"])
        else:
            a.set_yscale(p["ys"])
        a.set_ylabel(p["yl"], fontsize=7)
        a.set_xlim(*p["xlim"])
        if p.get("ylim"):
            a.set_ylim(*p["ylim"])
        if p.get("ylo"):
            a.set_ylim(bottom=p["ylo"])
        plt.setp(a.get_xticklabels(), visible=False)
        a.tick_params(labelsize=6)

        ar.axhline(0, color=COLORS["dmo"], lw=0.6)
        ar.fill_between(p["x"], -p["band"], p["band"], color=cB, alpha=0.22, lw=0)
        ar.plot(p["x"], p["band"], color=cB, lw=0.4, alpha=0.8)
        ar.plot(p["x"], -p["band"], color=cB, lw=0.4, alpha=0.8)
        ar.text(0.97, 0.06, "LSST-Y10", transform=ar.transAxes,
                fontsize=4.2, color=cB, va="bottom", ha="right")
        if p["discrete"]:
            ar.errorbar(p["x"], p["res"], yerr=p["pt"], fmt="o", ms=2.0, mew=0,
                        color=cB, lw=0.7, elinewidth=0.7, capsize=1.3)
        else:
            ar.fill_between(p["x"], p["res"] - p["pt"], p["res"] + p["pt"],
                            color=COLORS["dmo"], alpha=0.35, lw=0)
            ar.plot(p["x"], p["res"], color=cB, lw=0.9)
        if p.get("x2") is not None:
            ar.text(0.03, 0.92, rf"$\chi^2/{{\rm dof}}={p['x2']:.1f}$",
                    transform=ar.transAxes, fontsize=4.2, color="0.25", va="top")
        ar.set_ylim(-8, 8)
        ar.set_xscale(p["xs"])
        ar.set_xlabel(p["xl"], fontsize=7)
        ar.set_ylabel("resid. [%]" if j % 4 == 0 else "", fontsize=7)
        ar.tick_params(labelsize=6)
        # not paper_style.panel_label: its x=0.96 anchor clips the ')' on the
        # right-hand column of this 4-wide grid.
        a.text(0.93, 0.95, p["tag"], transform=a.transAxes, ha="right", va="top",
               fontsize=8, fontweight="bold")
        AX.append(a)
        AXR.append(ar)

    AXR[0].text(0.03, 0.06, r"residual: $z_s=1$", transform=AXR[0].transAxes,
                fontsize=4.2, color="0.3", va="bottom")
    AX[0].legend(handles=[plt.Line2D([], [], color=ZCOLS[zi], lw=1.0,
                                     label=rf"$z_s={R.ZS[zi]:.2f}$") for zi in range(5)]
                 + [plt.Line2D([], [], color="0.55", ls=":", lw=0.9,
                               label="shape-noise floor")],
                 loc="lower right", fontsize=4.2, ncol=2, handlelength=1.1,
                 columnspacing=0.7, labelspacing=0.25, borderpad=0.2, handletextpad=0.4)
    AX[1].legend(handles=[plt.Line2D([], [], color="0.35", lw=1.0, label="BIND"),
                          plt.Line2D([], [], color=cH, lw=0.9, ls="--",
                                     label=r"hydro-pasted, $z_s=1$")],
                 loc="lower left", fontsize=5.0, handlelength=1.6)
    save2(fig, "fig05n_field_validation_noisy")
    plt.close(fig)

    with open(R.OUT / "validation_numbers.json", "w") as f:
        json.dump(res_json, f, indent=1)
    print(f"wrote {R.OUT/'validation_numbers.json'}")


if __name__ == "__main__":
    main()
