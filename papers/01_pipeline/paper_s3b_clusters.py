#!/usr/bin/env python
"""paper_s3b_clusters.py -- publication versions of the response-shape
cluster figures for the paper's S3b (2026-08-06 figure plan).

MAIN   (pfig_s3b_cl_clusters):  Delta C_ell^kappakappa shape families C1-C4 +
       singletons with per-cluster mean correlations rbar -- the families
       result; the C3 panel shows IMFslope (black) riding the AGN family,
       the mechanism conclusion in one glance.
APPENDIX (pfig_s3b_pdf_clusters): the same clustering on the kappa-PDF
       responses -- family consistency across statistics.

Machinery is imf_shape_clusters.py's, unchanged (same S/N gate, average-
linkage on 1-r of unit-peak-normalized twobound bound-to-bound response
differences, same family colors, IMFslope in black); this script only adds
the clk statistic (the scratch quick_deltacl_clusters.png leg, now
regenerated at publication grade) and renders through paper_style.save().
Supersedes for the paper: the single-panel IMF-highlight overlay and old
fig 21 (cut -- cluster membership carries both).

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python paper_s3b_clusters.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

ZI = 1
SNR_MIN = 3.0
T_CLUST = 0.15
EDGES = np.geomspace(100.0, 3e4, 27)


def family(name):
    if any(k in name for k in ("BlackHole", "Quasar", "Radio")):
        return "agn"
    if "Wind" in name or "SN" in name:
        return "wind"
    return "other"


FAMCOL = {"wind": COLORS["bind"], "agn": COLORS["highlight"],
          "other": "#999999"}

names35 = list(pd.read_csv(
    "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")["ParamName"])
tbp = np.load(TB / "twobound_params.npy")
pairs = {}
for i in range(30):
    dcol = np.where(tbp[2 * i] != tbp[2 * i + 1])[0]
    assert len(dcol) == 1
    rr = [2 * i, 2 * i + 1]
    vv = [float(tbp[r, int(dcol[0])]) for r in rr]
    o = np.argsort(vv)
    pairs[names35[int(dcol[0])]] = [rr[k] for k in o]

P = {r: np.load(TB / f"run_{r:04d}/paired_stats.npz") for r in range(60)}
NG = {r: np.load(TB / f"run_{r:04d}/nongaussian_stats.npz") for r in range(60)}
ell = P[0]["ell"]
pdf_bins = NG[0]["pdf_bins"]


def logbin(y):
    ib = np.digitize(ell, EDGES) - 1
    L, Y = [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
    return np.array(L), np.array(Y)


def get(nm, stat):
    """(x, Delta, err_or_None) for one parameter and statistic."""
    rlo, rhi = pairs[nm]
    if stat == "clk":
        x, d = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
        _, e = logbin(np.sqrt(P[rhi]["clk_err"][ZI] ** 2
                              + P[rlo]["clk_err"][ZI] ** 2))
        nb = np.diff(np.searchsorted(ell, EDGES)).clip(1)[:len(x)]
        return x, d, e / np.sqrt(nb)
    if stat == "pdf":
        d = NG[rhi]["pdf"][ZI] - NG[rlo]["pdf"][ZI]
        return pdf_bins, d, None
    raise KeyError(stat)


def snr_of(nm, stat):
    """peak-bin |Delta|/err; pdf uses the V0 response as its S/N proxy
    (identical to imf_shape_clusters.py -- no paired pdf errors stored)."""
    if stat == "pdf":
        rlo, rhi = pairs[nm]
        d = P[rhi]["V0_resp"][ZI] - P[rlo]["V0_resp"][ZI]
        e = np.sqrt(P[rhi]["V0_err"][ZI] ** 2 + P[rlo]["V0_err"][ZI] ** 2)
    else:
        _, d, e = get(nm, stat)
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / e[k])


clk_clusters = None                    # captured for the bridge figure below
for stat, slab, xlab, logx, outname in [
        ("clk", r"$\Delta C_\ell^{\kappa\kappa}$", r"$\ell$", True,
         "figs_v2/pfig_s3b_cl_clusters"),
        ("pdf", r"$\Delta$ $\kappa$ PDF", r"$\kappa/\sigma_\kappa$", False,
         "figs_v2/pfig_s3b_pdf_clusters")]:
    CUR, excl = {}, []
    for nm in pairs:
        if snr_of(nm, stat) < SNR_MIN:
            excl.append(nm)
            continue
        x, d, _ = get(nm, stat)
        CUR[nm] = d / d[np.nanargmax(np.abs(d))]
    NAMES = list(CUR)
    M = np.corrcoef([CUR[n] for n in NAMES])
    Z = linkage(squareform(np.clip(1 - M, 0, None), checks=False),
                method="average")
    lab = fcluster(Z, t=T_CLUST, criterion="distance")
    clusters = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        mem = [NAMES[i] for i in idx]
        rbar = (M[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                if len(idx) > 1 else np.nan)
        clusters.append((mem, rbar))
    clusters.sort(key=lambda c: -len(c[0]))
    multi = [c for c in clusters if len(c[0]) > 1]
    single = [c[0][0] for c in clusters if len(c[0]) == 1]
    if stat == "clk":
        clk_clusters = (multi, single)

    print(f"\n=== {stat} (S/N>={SNR_MIN:g}: {len(NAMES)}/30 params; "
          f"excluded: {', '.join(short_label(n) for n in excl) or 'none'}) ===")
    for k, (mem, rbar) in enumerate(multi):
        star = "  <-- IMFslope" if "IMFslope" in mem else ""
        print(f"  C{k+1} (n={len(mem)}, rbar={rbar:.2f}): "
              + ", ".join(short_label(m) for m in mem) + star)
    if single:
        print(f"  singletons: {', '.join(short_label(m) for m in single)}")

    npan = len(multi) + (1 if single else 0)
    ncol = 3
    nrow = max(1, int(np.ceil(npan / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWO_COL[0], 2.3 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    panels = [(f"C{k+1}  ($\\bar r$={rbar:.2f})", mem)
              for k, (mem, rbar) in enumerate(multi)]
    if single:
        panels.append(("singletons", single))
    for k, (title, mem) in enumerate(panels):
        ax = axes[k]
        for nm in NAMES:
            ax.plot(x, CUR[nm], color="0.88", lw=0.5, zorder=1)
        for nm in mem:
            lw = 2.0 if nm == "IMFslope" else 1.1
            col = "#111111" if nm == "IMFslope" else FAMCOL[family(nm)]
            ax.plot(x, CUR[nm], color=col, lw=lw,
                    zorder=6 if nm == "IMFslope" else 4)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        if logx:
            ax.set_xscale("log")
        panel_label(ax, title)
        ax.text(0.97, 0.05, "\n".join(short_label(m) for m in mem),
                transform=ax.transAxes, fontsize=5.0, ha="right", va="bottom",
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.0))
    for ax in axes[npan:]:
        ax.set_visible(False)
    for k in range(npan):
        axes[k].tick_params(labelsize=6)
        if k % ncol == 0:
            axes[k].set_ylabel(f"{slab} / peak", fontsize=6.5)
        if k >= npan - ncol:
            axes[k].set_xlabel(xlab, fontsize=7)
    fig.tight_layout()
    save(fig, outname)
    plt.close(fig)
    print(f"  wrote {outname}.pdf")


# ═══ BRIDGE FIGURE: family shapes = kernel mixtures (chain-rule closure) ════
# (2026-08-06, author: "I wish each of the groups was related to the
# coefficients in some obvious way".) They are, exactly, by the chain rule:
#     dS/dtheta_j (ell) = sum_i c_i(ell) * (dlambda_i/dtheta_j)
# so a family's ell-shape is a FIXED MIXTURE of the four kernel functions,
# with mixture weights = the family's latent fingerprint Delta-lambda. This
# is an OUT-OF-DESIGN closure: the kernels c_i come from the Sobol fit
# (latent_model_coeffs.npz, the shipped table), while the fingerprints AND
# the measured Delta-S shapes come from the independent twobound sims (tb_*
# rows of the same atlas cube; paired_stats clk_resp). Panels: per-family
# mean measured shape vs kernel-mixture prediction (shape r annotated) + the
# fingerprint matrix (family-mean, sign-aligned Delta-lambda / sigma_lambda).
# NOTE clk_resp convention: responses are ratios to a shared reference, so
# bound-to-bound DIFFERENCES are reference-free to first order; amplitude
# ratios pred/meas land at ~0.8-1.1 for most members (printed).
from predict_from_latents import CACHE, build_cache  # noqa: E402

if not CACHE.exists():
    build_cache()
coef = np.load(CACHE)
assert int(np.atleast_1d(coef["version"])[0]) == 2, "pre-harmonization cache"
zi_b = int(np.argmin(np.abs(coef["zs"] - 1.0)))
Bk = coef["beta"][zi_b][:, :4]                     # (24, 4) kernels, z_s=1
EDGb = coef["ell_edges"]
ctr_b = coef["ell"]

SB35 = CEPH / "bind_sb35"
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
OB_OM = 0.0486 / 0.3089


def latents_rows(prefix, rows):
    mt = cz[f"{prefix}_m_tot_500c_bg"][rows]
    mg = cz[f"{prefix}_m_gas_500c_bg"][rows]
    g2 = cz[f"{prefix}_m_gas_200c_bg"][rows]
    ms = cz[f"{prefix}_m_star_500c"][rows]
    Tm = cz[f"{prefix}_T_mw_500c"][rows]
    lm = np.log10(np.where(mt > 0, mt, np.nan))
    out = np.full((len(rows), 4), np.nan)
    for q in range(len(rows)):
        s = (lm[q] >= 13.3) & (lm[q] < 13.6)
        sg = s & (g2[q] > 0)
        if s.sum() >= 5:
            out[q, 0] = np.nanmedian(((mg + ms) / mt)[q, s]) / OB_OM
            out[q, 1] = np.nanmedian((ms / mt)[q, s]) / OB_OM
            out[q, 3] = np.log10(np.nanmedian(np.where(Tm[q, s] > 0,
                                                       Tm[q, s], np.nan)))
        if sg.sum() >= 5:
            out[q, 2] = np.nanmedian((mg / g2)[q, sg])
    return out


LATtb = latents_rows("tb", np.arange(60))
tbv = cz["tb_valid"]
dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
LATsob = latents_rows("sobol", dsn["run_ids"])
sig_lam = np.nanstd(LATsob, axis=0)


def bandS(r):
    y = P[r]["clk_resp"][ZI]
    return np.stack([np.nanmean(y[(ell >= EDGb[i]) & (ell < EDGb[i + 1])])
                     for i in range(24)])


from scipy.stats import pearsonr  # noqa: E402

multi_clk, single_clk = clk_clusters
panels_b = [(f"C{k+1}", mem) for k, (mem, _) in enumerate(multi_clk)]
if single_clk:
    panels_b.append(("singleton", single_clk))
LNAM = [r"$\tilde f_{\rm bar}$", r"$\tilde f_\star$", r"$c_{\rm gas}$",
        r"$\log\tilde T$"]

fig, AXb = plt.subplots(2, 3, figsize=(TWO_COL[0], 4.3), sharex=False)
AXb = AXb.ravel()
fingerprints, fam_r = [], []
for k, (fam, members) in enumerate(panels_b):
    ax = AXb[k]
    meas_c, pred_c, fps, drop = [], [], [], []
    for nm in members:
        rlo, rhi = pairs[nm]
        if not (tbv[rlo] and tbv[rhi]):
            drop.append((nm, "tb-atlas invalid"))
            continue
        dlam = LATtb[rhi] - LATtb[rlo]
        if not np.isfinite(dlam).all():
            drop.append((nm, "non-finite dlambda"))
            continue
        meas = bandS(rhi) - bandS(rlo)
        pred = Bk @ dlam
        kpk = int(np.nanargmax(np.abs(meas)))
        meas_c.append(meas / meas[kpk])
        pred_c.append(pred / meas[kpk])
        fps.append(np.sign(meas[kpk]) * dlam / sig_lam)
    for nm, why in drop:
        print(f"  [bridge:{fam}] {short_label(nm)} skipped ({why})")
    mc, pc = np.mean(meas_c, 0), np.mean(pred_c, 0)
    r_fam = float(pearsonr(mc, pc)[0])
    fam_r.append(r_fam)
    fingerprints.append(np.mean(fps, 0))
    for m in meas_c:
        ax.plot(ctr_b, m, color="0.85", lw=0.5, zorder=1)
    ax.plot(ctr_b, mc, color=COLORS["bind"], lw=1.6, zorder=4,
            label="measured (twobound mean)")
    ax.plot(ctr_b, pc, color=COLORS["highlight"], lw=1.3, ls="--", zorder=5,
            label="kernel mixture $\sum_i c_i\,\Delta\lambda_i$")
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(300, 3e4)
    panel_label(ax, f"{fam}  ($r$={r_fam:.2f})")
    ax.tick_params(labelsize=5.5)
    if k >= 3:
        ax.set_xlabel(r"$\ell$")
    if k % 3 == 0:
        ax.set_ylabel(r"$\Delta S$ / peak", fontsize=6.5)
    if k == 0:
        ax.legend(fontsize=5.0, loc="center left", handletextpad=0.5)

axF = AXb[len(panels_b)]
FP = np.array(fingerprints)
vmax_f = float(np.nanmax(np.abs(FP)))
imF = axF.imshow(FP, cmap="coolwarm", vmin=-vmax_f, vmax=vmax_f,
                 aspect="auto")
for i in range(FP.shape[0]):
    for j in range(4):
        axF.text(j, i, f"{FP[i, j]:+.1f}", ha="center", va="center",
                 fontsize=5.6,
                 color="k" if abs(FP[i, j]) < 0.6 * vmax_f else "w")
axF.set_xticks(range(4))
axF.set_xticklabels(LNAM, fontsize=6.5)
axF.set_yticks(range(len(panels_b)))
axF.set_yticklabels([f for f, _ in panels_b], fontsize=6.5)
axF.set_title(r"latent fingerprint  $\Delta\lambda/\sigma_\lambda$"
              " (sign-aligned mean)", fontsize=6.3)
axF.tick_params(length=0)
for ax in AXb[len(panels_b) + 1:]:
    ax.set_visible(False)
fig.tight_layout()
save(fig, "figs_v2/pfig_family_kernel_bridge")
plt.close(fig)
print("\nbridge [caption]: family-mean shape corr measured vs kernel "
      "mixture: "
      + ", ".join(f"{f} r={r:.2f}" for (f, _), r in zip(panels_b, fam_r))
      + " -- OUT-OF-DESIGN closure (Sobol kernels x twobound fingerprints/"
      "shapes): a family is a set of parameters sharing a latent "
      "fingerprint; its ell-shape is that fixed mixture of the four "
      "kernels. The families exist BECAUSE there are only four kernels.")
print("bridge [caption]: fingerprint rows (Delta-lambda/sigma_lambda): "
      + "; ".join(f"{f}: " + ",".join(f"{v:+.1f}" for v in fp)
                  for (f, _), fp in zip(panels_b, fingerprints)))


# ═══ MODEL-CONSTRUCTION FIGURE (2026-08-06 arc revision) ════════════════════
# The tutorial (figs_preview/tutorial_family_kernels.png) formalized into a
# paper figure motivating the methodology, four panels:
#  (a) each universe -> four measured halo numbers (standardized strips);
#  (b) at ONE ell, suppression vs one number: the partial slope IS the
#      coefficient there (joint-fit line drawn);
#  (c) the slope at every ell -> the four kernel functions (from the shipped
#      table; the (b) slope circled);
#  (d) one knob (WindEnergy, twobound pair): its measured Delta-S equals the
#      fingerprint-weighted kernel sum -- the chain rule, zero fitting.
S24s = None
dsn_S = dsn["t__suppression__value"][:, ZI, :]
S24s = np.stack([np.nanmean(dsn_S[:, (dsn["a__suppression__ell"] >= EDGb[i])
                                  & (dsn["a__suppression__ell"] < EDGb[i+1])],
                            1) for i in range(24)], axis=1)
oks = np.isfinite(LATsob).all(1)
A_s = np.c_[LATsob[oks], np.ones(int(oks.sum()))]
beta_s, *_ = np.linalg.lstsq(A_s, S24s[oks], rcond=None)

LNAMF = [r"$\tilde f_{\rm bar}$", r"$\tilde f_\star$",
         r"$c_{\rm gas}$", r"$\log\tilde T$"]
KC = [COLORS["bind"], "#111111", COLORS["highlight"], COLORS["secondary"]]
ib_c = int(np.argmin(np.abs(ctr_b - 4847)))

fig, ((ca, cb), (cc, cd)) = plt.subplots(2, 2, figsize=(TWO_COL[0], 4.8))
# (a)
for j in range(4):
    xz = (LATsob[oks, j] - LATsob[oks, j].mean())/LATsob[oks, j].std()
    ca.plot(xz, np.full(len(xz), 3 - j), "|", ms=9, color="0.6", alpha=0.45)
ca.set_yticks([3, 2, 1, 0])
ca.set_yticklabels(LNAMF, fontsize=7)
ca.set_xlabel("standardized value (cloud mean 0, width 1)", fontsize=6.5)
ca.set_xlim(-3.4, 3.4)
panel_label(ca, "(a)")
ca.set_title(r"each universe $\to$ four halo numbers", fontsize=6.8)
# (b)
xj = LATsob[oks, 0]
yj = S24s[oks, ib_c]
cb.scatter(xj, yj, s=6, color="0.6", alpha=0.55, rasterized=True)
xg = np.linspace(xj.min(), xj.max(), 10)
oth = LATsob[oks].mean(0)
cb.plot(xg, beta_s[4, ib_c] + beta_s[0, ib_c]*xg + beta_s[1, ib_c]*oth[1]
        + beta_s[2, ib_c]*oth[2] + beta_s[3, ib_c]*oth[3],
        color=COLORS["bind"], lw=1.8)
cb.set_xlabel(r"$\tilde f_{\rm bar}$", fontsize=6.5)
cb.set_ylabel(r"$S(\ell\simeq4847)$", fontsize=6.5)
cb.set_title(f"one $\\ell$: the (partial) slope = "
             f"$c_1$ = +{beta_s[0, ib_c]:.2f}", fontsize=6.8)
panel_label(cb, "(b)")
# (c)
for j in range(4):
    cc.plot(ctr_b, Bk[:, j], color=KC[j], lw=1.4, label=LNAMF[j])
cc.plot(ctr_b[ib_c], Bk[ib_c, 0], "o", ms=8, mfc="none",
        mec=COLORS["bind"], mew=1.6)
cc.axhline(0, color="0.8", lw=0.6)
cc.set_xscale("log")
cc.set_xlabel(r"$\ell$", fontsize=6.5)
cc.set_ylabel(r"kernel $c_i(\ell)$", fontsize=6.5)
cc.set_title("the slope at every $\\ell$ $\\to$ the kernels",
             fontsize=6.8)
cc.legend(fontsize=5.2, loc="lower left", ncol=2, handletextpad=0.4)
panel_label(cc, "(c)")
# (d)
rlo_w, rhi_w = pairs["WindEnergyIn1e51erg"]
dlam_w = LATtb[rhi_w] - LATtb[rlo_w]
dS_w = bandS(rhi_w) - bandS(rlo_w)
for j in range(4):
    cd.plot(ctr_b, Bk[:, j]*dlam_w[j], color=KC[j], lw=1.0, ls="--",
            alpha=0.8)
tot_w = Bk @ dlam_w
cd.plot(ctr_b, tot_w, color="k", lw=2.0, label="kernel sum")
cd.plot(ctr_b, dS_w, "o", ms=3.2, color=COLORS["highlight"],
        label="measured (twobound)")
r_w = float(np.corrcoef(tot_w, dS_w)[0, 1])
cd.axhline(0, color="0.8", lw=0.6)
cd.set_xscale("log")
cd.set_xlabel(r"$\ell$", fontsize=6.5)
cd.set_ylabel(r"$\Delta S(\ell)$", fontsize=6.5)
cd.set_title(f"one knob (WindEnergy): chain rule, no fitting "
             f"($r$={r_w:.2f})", fontsize=6.8)
cd.legend(fontsize=5.2, loc="lower left", handletextpad=0.4)
panel_label(cd, "(d)")
fig.tight_layout()
save(fig, "figs_v2/pfig_s4a_model_construction")
plt.close(fig)
print(f"pfig_s4a_model_construction: c1(ell~4847)={beta_s[0, ib_c]:+.3f} "
      f"(panel b), WindEnergy chain-rule r={r_w:.2f} (panel d)")
