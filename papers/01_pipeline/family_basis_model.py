#!/usr/bin/env python
"""family_basis_model.py -- can the cluster-family mean shapes THEMSELVES be
the basis of the S(ell) model?  (2026-08-11 author question: "take the mean
curve for each family, weighted by how much the statistic actually cares
about the parameter, amplitude-wise; normalize to one for each of C1-C5;
use that basis as the model, then find amplitude factors".)

The latent model writes dS(ell) = sum_i c_i(ell) dlambda_i with four kernels
c_i fitted on the Sobol design (latent_model_coeffs.npz).  This script builds
five EMPIRICAL alternatives B_k(ell): the amplitude-weighted mean of each
clk family's member Delta-S curves (families C1-C4 + the SNII singleton
"C5" of pfig_s3b_cl_clusters, identical machinery), unit-peak normalized,
in the same 24-band ell space as the kernels.  It then answers, in order:

 (1) SPAN     -- fraction of the Sobol S(ell) deviation variance the 5-vector
                 family span captures, vs the 4 kernels with free amplitudes,
                 vs rank-4/5 PCA of the Sobol responses (in-sample ceiling).
                 The basis is built from twobound sims ONLY, so the Sobol
                 test is out-of-design by construction (no leakage).
                 NB the nu05 dataset carries 256 finite runs INCLUDING
                 0114/0115/0117, which the 253-node latent-model canon
                 excludes; dropping them was checked to move nothing here
                 at the 1e-4 level (suppression is unaffected by the
                 cross-norm issue that motivated the exclusion).
 (2) MUTUAL   -- cross-projection: family basis onto the kernel span and
                 kernels onto the family span (same space, or not?).
 (3) CONDITION-- Gram correlations + singular values of the 5 vectors: are
                 per-vector amplitudes unique?  (Latent result: the response
                 space is ~rank-4, so 5 vectors MUST be partly degenerate --
                 the question is how badly, and which combination is free.)
 (4) AMPLITUDE-- per-run amplitude factors a_k (ridge-stabilized LSQ) and
                 their model: from the four measured halo latents lambda
                 (is the family basis a rotation of the kernel model?) and
                 directly from theta (linear, 5-fold CV).

Weight / normalization variants (amp = |peak dS| [the author's
"amplitude-wise" reading, default], snr = twobound peak S/N, eq = flat) are
all computed and printed -- the conclusion should not hinge on the choice.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_basis_model.py
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
SB35 = CEPH / "bind_sb35"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

ZI = 1
SNR_MIN = 3.0
T_CLUST = 0.15
EDGES = np.geomspace(100.0, 3e4, 27)

# ── twobound pairs + clk clustering: verbatim paper_s3b_clusters machinery ──
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
ell = P[0]["ell"]


def logbin(y):
    ib = np.digitize(ell, EDGES) - 1
    L, Y = [], []
    for b in range(len(EDGES) - 1):
        s = ib == b
        if s.any():
            L.append(np.exp(np.log(ell[s]).mean()))
            Y.append(np.nanmean(y[s]))
    return np.array(L), np.array(Y)


def get_clk(nm):
    rlo, rhi = pairs[nm]
    x, d = logbin(P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI])
    _, e = logbin(np.sqrt(P[rhi]["clk_err"][ZI] ** 2
                          + P[rlo]["clk_err"][ZI] ** 2))
    nb = np.diff(np.searchsorted(ell, EDGES)).clip(1)[:len(x)]
    return x, d, e / np.sqrt(nb)


def snr_of(nm):
    _, d, e = get_clk(nm)
    k = int(np.nanargmax(np.abs(d)))
    return float(np.abs(d[k]) / e[k])


CUR = {}
for nm in pairs:
    if snr_of(nm) < SNR_MIN:
        continue
    _, d, _ = get_clk(nm)
    CUR[nm] = d / d[np.nanargmax(np.abs(d))]
NAMES = list(CUR)
Mc = np.corrcoef([CUR[n] for n in NAMES])
Z = linkage(squareform(np.clip(1 - Mc, 0, None), checks=False),
            method="average")
lab = fcluster(Z, t=T_CLUST, criterion="distance")
clusters = []
for c in np.unique(lab):
    idx = np.where(lab == c)[0]
    clusters.append([NAMES[i] for i in idx])
clusters.sort(key=len, reverse=True)
FAMS = [(f"C{k+1}", mem) for k, mem in enumerate(c for c in clusters
                                                if len(c) > 1)]
single = [c[0] for c in clusters if len(c) == 1]
if single:
    FAMS.append((f"C{len(FAMS)+1}", single))     # the singleton family "C5"
print("families: " + "; ".join(
    f"{f}({len(m)}): " + ",".join(short_label(x) for x in m)
    for f, m in FAMS))

# ── the 24-band ell space shared with the latent-model kernels ──────────────
from predict_from_latents import CACHE, build_cache  # noqa: E402

if not CACHE.exists():
    build_cache()
coef = np.load(CACHE)
assert int(np.atleast_1d(coef["version"])[0]) == 2, "pre-harmonization cache"
zi_b = int(np.argmin(np.abs(coef["zs"] - 1.0)))
Bk = coef["beta"][zi_b][:, :4]                     # (24, 4) kernels at z_s=1
EDGb = coef["ell_edges"]
ctr_b = coef["ell"]
NB = len(ctr_b)


def bandS(r):
    y = P[r]["clk_resp"][ZI]
    return np.stack([np.nanmean(y[(ell >= EDGb[i]) & (ell < EDGb[i + 1])])
                     for i in range(NB)])


def band_dS(nm):
    rlo, rhi = pairs[nm]
    return bandS(rhi) - bandS(rlo)


# ── the family basis: weighted mean of unit-peak member curves ──────────────
def build_basis(weight="amp"):
    """(K, NB) matrix; each row unit-peak (max |B_k| == 1, peak sign +)."""
    rows = []
    for fam, mem in FAMS:
        curves, ws = [], []
        for m in mem:
            d = band_dS(m)
            pk = d[np.nanargmax(np.abs(d))]        # signed peak: aligns signs
            curves.append(d / pk)
            ws.append({"amp": abs(pk), "snr": snr_of(m), "eq": 1.0}[weight])
        b = np.average(curves, axis=0, weights=ws)
        rows.append(b / b[np.nanargmax(np.abs(b))])
    return np.array(rows)


BAS = {w: build_basis(w) for w in ("amp", "snr", "eq")}
B5 = BAS["amp"]                                    # the default basis (K=5)
K = B5.shape[0]
for w in ("snr", "eq"):
    rs = [float(np.corrcoef(B5[k], BAS[w][k])[0, 1]) for k in range(K)]
    print(f"basis robustness vs weight='{w}': per-family shape r = "
          + ", ".join(f"{r:.3f}" for r in rs))

# ── Sobol responses in the same bands (the out-of-design test set) ──────────
dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
dsn_S = np.asarray(dsn["t__suppression__value"], float)[:, ZI, :]
ELLd = np.asarray(dsn["a__suppression__ell"], float)
S24 = np.stack([np.nanmean(dsn_S[:, (ELLd >= EDGb[i]) & (ELLd < EDGb[i + 1])],
                           1) for i in range(NB)], axis=1)
finS = np.isfinite(S24).all(1)
D = S24[finS] - S24[finS].mean(0)                  # (n, 24) deviations
n_runs = D.shape[0]
print(f"Sobol test set: {n_runs} finite runs; total dev power "
      f"{np.sqrt((D**2).mean()):.4f} RMS")


def span_fit(Dm, Bmat):
    """Free-amplitude LSQ of Dm (n,NB) on rows of Bmat (K,NB)."""
    A = Dm @ np.linalg.pinv(Bmat)
    R = Dm - A @ Bmat
    r2_run = 1 - (R ** 2).sum(1) / (Dm ** 2).sum(1)
    r2_tot = 1 - (R ** 2).sum() / (Dm ** 2).sum()
    return A, r2_run, r2_tot


# (1) SPAN: family basis vs kernels vs PCA ceiling ---------------------------
A5, r2r_fam, r2_fam = span_fit(D, B5)
_, r2r_ker, r2_ker = span_fit(D, Bk.T)
U, sv, Vt = np.linalg.svd(D, full_matrices=False)
pca = {k: float((sv[:k] ** 2).sum() / (sv ** 2).sum()) for k in (1, 2, 3, 4, 5)}
_, r2r_p4, _ = span_fit(D, Vt[:4])
_, r2r_p5, _ = span_fit(D, Vt[:5])
print("\n== (1) SPAN on the Sobol deviations (total R^2 | median per-run) ==")
for tag, r2t, r2r in [("family-5 (twobound-built)", r2_fam, r2r_fam),
                      ("kernels-4 (free amps)", r2_ker, r2r_ker),
                      ("PCA-4 (in-sample ceiling)", pca[4], r2r_p4),
                      ("PCA-5 (in-sample ceiling)", pca[5], r2r_p5)]:
    print(f"  {tag:>28s}: {r2t:.4f} | {np.median(r2r):.4f}")
print(f"  (PCA ceilings 1/2/3: {pca[1]:.4f}/{pca[2]:.4f}/{pca[3]:.4f}; "
      f"residual RMS family-5 {np.sqrt(((D - A5 @ B5)**2).mean()):.4f})")
# how few family vectors already suffice? best subset per size
from itertools import combinations  # noqa: E402
for sz in (2, 3):
    best = max(combinations(range(K), sz),
               key=lambda c: span_fit(D, B5[list(c)])[2])
    print("  best family-%d subset {%s}: total R^2 %.4f"
          % (sz, ",".join(FAMS[k][0] for k in best),
             span_fit(D, B5[list(best)])[2]))

# per-weight-variant span (the conclusion must not hinge on the weighting)
for w in ("snr", "eq"):
    _, _, r2v = span_fit(D, BAS[w])
    print(f"  span sensitivity: weight='{w}' total R^2 {r2v:.4f}")
_, _, r2_fam4 = span_fit(D, B5[:4])
print(f"  family-4 (drop the sub-noise C5 singleton): total R^2 {r2_fam4:.4f}")

# (2) MUTUAL span: do families and kernels span the same space? --------------
print("\n== (2) MUTUAL cross-projection (shape r after projection) ==")
proj_on = lambda V, Bmat: (V @ np.linalg.pinv(Bmat)) @ Bmat  # noqa: E731
pr_fam = proj_on(B5, Bk.T)
for k in range(K):
    r = float(np.corrcoef(B5[k], pr_fam[k])[0, 1])
    print(f"  {FAMS[k][0]} onto kernel span: r = {r:.4f}")
pr_ker = proj_on(Bk.T, B5)
LN = ["f_bar", "f_star", "c_gas", "logT"]
for i in range(4):
    r = float(np.corrcoef(Bk[:, i], pr_ker[i])[0, 1])
    print(f"  kernel c_{i+1} ({LN[i]}) onto family span: r = {r:.4f}")

# (3) CONDITIONING of the family basis ---------------------------------------
G = np.corrcoef(B5)
uB, svB, _ = np.linalg.svd(B5, full_matrices=False)
print("\n== (3) CONDITIONING ==")
print("  Gram |r| max off-diag: "
      f"{np.abs(G[np.triu_indices(K, 1)]).max():.3f}; full matrix:")
for k in range(K):
    print("    " + " ".join(f"{G[k, j]:+.2f}" for j in range(K)))
print("  singular values: " + ", ".join(f"{s:.3f}" for s in svB)
      + f"; condition number {svB[0]/svB[-1]:.1f}")
print("  weakest amplitude combination (right-sv of smallest sv): "
      + ", ".join(f"{FAMS[k][0]}:{uB[k, -1]:+.2f}" for k in range(K)))

# (4) AMPLITUDES: ridge-stabilized factors + their model ---------------------
# ridge on the (K,K) normal matrix: alpha scaled to its trace
def ridge_amps(Dm, Bmat, alpha_rel=1e-2):
    BBt = Bmat @ Bmat.T
    al = alpha_rel * np.trace(BBt) / Bmat.shape[0]
    return Dm @ Bmat.T @ np.linalg.inv(BBt + al * np.eye(Bmat.shape[0]))


Ar = ridge_amps(D, B5)
R_r = D - Ar @ B5
r2_ridge = 1 - (R_r ** 2).sum() / (D ** 2).sum()
print("\n== (4) AMPLITUDE FACTORS ==")
print(f"  ridge (alpha_rel=1e-2) amplitudes: span R^2 {r2_ridge:.4f} "
      f"(vs LSQ {r2_fam:.4f}) -- ridge trades {r2_fam-r2_ridge:.4f} R^2 "
      "for identifiable per-family amplitudes")

# 4a. amplitudes from the four measured halo latents (rotation test)
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


LATsob = latents_rows("sobol", dsn["run_ids"])[finS]
okL = np.isfinite(LATsob).all(1)
Lam = LATsob[okL] - LATsob[okL].mean(0)
Mmix, *_ = np.linalg.lstsq(Lam, Ar[okL], rcond=None)     # (4 latents, K)
# 5-fold CV, protocol-matched to the theta fit below (verifier finding:
# the in-sample version overstates by only ~0.01-0.02, but mixed protocols
# in one panel are misleading)
ArL = Ar[okL]
ordL = np.random.default_rng(1).permutation(int(okL.sum()))
pred_aL = np.full_like(ArL, np.nan)
for f in np.array_split(ordL, 5):
    tr = np.setdiff1d(ordL, f)
    W, *_ = np.linalg.lstsq(np.c_[Lam[tr], np.ones(len(tr))], ArL[tr],
                            rcond=None)
    pred_aL[f] = np.c_[Lam[f], np.ones(len(f))] @ W
r2_a_lam = 1 - ((ArL - pred_aL) ** 2).sum(0) / ((ArL - ArL.mean(0)) ** 2).sum(0)
print("  a_k from the 4 halo latents (linear, 5-fold CV R^2): "
      + ", ".join(f"{FAMS[k][0]}:{r2_a_lam[k]:.2f}" for k in range(K)))
# DISCLOSURE (verifier finding): the per-family amplitude R^2 depends on the
# ridge alpha, because with condition number ~566 the unregularized per-run
# amplitudes are noise-dominated along the degenerate combinations of (3).
# The alpha-robust statements are the SPAN and the end-to-end latent route.
for al in (0.0, 1e-3, 1e-1):
    Aa = span_fit(D, B5)[0] if al == 0 else ridge_amps(D, B5, al)
    Wl, *_ = np.linalg.lstsq(np.c_[Lam, np.ones(len(Lam))], Aa[okL],
                             rcond=None)
    pa = np.c_[Lam, np.ones(len(Lam))] @ Wl
    r2v = 1 - ((Aa[okL] - pa) ** 2).sum(0) / ((Aa[okL] - Aa[okL].mean(0)) ** 2).sum(0)
    print(f"    alpha_rel={al:g}: a_k|lambda in-sample R^2 range "
          f"{r2v.min():.2f}-{r2v.max():.2f}")
print("  mixing matrix M (rows = latents, cols = " +
      ",".join(f for f, _ in FAMS) + "):")
for i in range(4):
    print(f"    {LN[i]:>7s}: " + " ".join(f"{Mmix[i, k]:+8.3f}"
                                          for k in range(K)))

# 4b. amplitudes directly from theta (linear, 5-fold CV)
Xu = np.asarray(dsn["X_unit"], float)[finS]
rng = np.random.default_rng(0)
order = rng.permutation(n_runs)
folds = np.array_split(order, 5)
pred_cv = np.full_like(Ar, np.nan)
for f in folds:
    tr = np.setdiff1d(order, f)
    Xtr = np.c_[Xu[tr], np.ones(len(tr))]
    W, *_ = np.linalg.lstsq(Xtr, Ar[tr], rcond=None)
    pred_cv[f] = np.c_[Xu[f], np.ones(len(f))] @ W
r2_a_th = 1 - ((Ar - pred_cv) ** 2).sum(0) / ((Ar - Ar.mean(0)) ** 2).sum(0)
print("  a_k from theta (30-param linear, 5-fold CV R^2): "
      + ", ".join(f"{FAMS[k][0]}:{r2_a_th[k]:.2f}" for k in range(K)))

# end-to-end: S(ell) via theta -> a (CV) -> basis, vs measured
res_e2e = D - pred_cv @ B5
r2_e2e = 1 - (res_e2e ** 2).sum() / (D ** 2).sum()
rms_e2e = float(np.sqrt((res_e2e ** 2).mean()))
print(f"  END-TO-END theta -> a(CV linear) -> S(ell): total R^2 {r2_e2e:.4f}"
      f", RMS {rms_e2e:.4f}")

# the latent ROUTE through the family basis: theta -> measured lambda ->
# a = lambda M -> S.  This is the family-basis analog of the latent model
# (the nonlinearity theta->lambda is carried by the halo measurement, not
# by a fitted map), and the fair head-to-head.
D_fam_lam = (Lam @ Mmix) @ B5
r2_famlam = 1 - ((D[okL] - D_fam_lam) ** 2).sum() / (D[okL] ** 2).sum()
rms_famlam = float(np.sqrt(((D[okL] - D_fam_lam) ** 2).mean()))
print(f"  LATENT ROUTE lambda -> a = lambda M -> S: total R^2 "
      f"{r2_famlam:.4f}, RMS {rms_famlam:.4f}")

# latent-model reference on the same rows (fixed amplitudes = latents)
D_lam = Lam @ Bk.T
r2_latmod = 1 - ((D[okL] - D_lam) ** 2).sum() / (D[okL] ** 2).sum()
print(f"  reference: latent model (amps = measured lambda): total R^2 "
      f"{r2_latmod:.4f} on the same {int(okL.sum())} rows")

# ── figure ──────────────────────────────────────────────────────────────────
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]
fig, ((fa, fb), (fc_, fd)) = plt.subplots(2, 2, figsize=(TWO_COL[0], 4.6))
for k in range(K):
    fa.plot(ctr_b, B5[k], color=FC[k], lw=1.5, label=FAMS[k][0])
    fa.plot(ctr_b, pr_fam[k], color=FC[k], lw=0.8, ls="--", alpha=0.7)
fa.axhline(0, color="0.8", lw=0.5)
fa.set_xscale("log")
fa.set_xlabel(r"$\ell$", fontsize=6.5)
fa.set_ylabel(r"$B_k(\ell)$ (unit peak)", fontsize=6.5)
fa.legend(fontsize=5.2, ncol=2, loc="upper left",
          bbox_to_anchor=(0.02, 0.88), handletextpad=0.5)
fa.set_title("family basis (solid) + its kernel-span projection (dashed)",
             fontsize=6.4)
panel_label(fa, "(a)")

imG = fb.imshow(G, cmap="coolwarm", vmin=-1, vmax=1)
for i in range(K):
    for j in range(K):
        fb.text(j, i, f"{G[i, j]:+.2f}", ha="center", va="center",
                fontsize=5.4, color="w" if abs(G[i, j]) > 0.6 else "k")
fb.set_xticks(range(K)); fb.set_xticklabels([f for f, _ in FAMS], fontsize=6)
fb.set_yticks(range(K)); fb.set_yticklabels([f for f, _ in FAMS], fontsize=6)
fb.tick_params(length=0)
fb.set_title("basis Gram correlations\n"
             + "sv: " + ", ".join(f"{s:.2f}" for s in svB), fontsize=6.2)
panel_label(fb, "(b)")

bins = np.geomspace(1e-5, 0.2, 28)
for r2r, lab_, col in [(r2r_fam, "family-5", COLORS["bind"]),
                       (r2r_ker, "kernels-4", COLORS["highlight"]),
                       (r2r_p5, "PCA-5 ceiling", "0.35")]:
    fc_.hist(np.clip(1 - r2r, bins[0], bins[-1]), bins=bins, histtype="step",
             lw=1.3, color=col, label=lab_)
fc_.set_xscale("log")
fc_.set_xlabel(r"per-run residual fraction $1-R^2$", fontsize=6.5)
fc_.set_ylabel("runs", fontsize=6.5)
fc_.legend(fontsize=5.2, loc="upper right")
fc_.set_title(f"span of the {n_runs} Sobol deviations "
              f"(totals {r2_fam:.4f} / {r2_ker:.4f} / {pca[5]:.4f})",
              fontsize=6.4)
panel_label(fc_, "(c)")

xk = np.arange(K)
fd.bar(xk - 0.2, r2_a_lam, 0.4, color=COLORS["bind"],
       label=r"$a_k$ from 4 halo latents")
fd.bar(xk + 0.2, r2_a_th, 0.4, color=COLORS["highlight"],
       label=r"$a_k$ from $\theta$ (CV)")
fd.set_xticks(xk); fd.set_xticklabels([f for f, _ in FAMS], fontsize=6.5)
fd.set_ylim(0, 1)
fd.set_ylabel(r"amplitude $R^2$", fontsize=6.5)
fd.legend(fontsize=5.2, loc="lower left")
fd.set_title(f"amplitude factors (ridge); end-to-end $R^2$={r2_e2e:.3f}",
             fontsize=6.4)
panel_label(fd, "(d)")
fig.tight_layout()
save(fig, "figs_v2/pfig_family_basis_model")
plt.close(fig)
np.savez("figs_preview/family_basis_model.npz",
         basis=B5, fam_names=[f for f, _ in FAMS], ell=ctr_b,
         amps_ridge=Ar, r2_span=(r2_fam, r2_ker, pca[4], pca[5]),
         gram=G, sv=svB, Mmix=Mmix, r2_a_lam=r2_a_lam, r2_a_th=r2_a_th,
         r2_e2e=r2_e2e)
print("\nwrote figs_v2/pfig_family_basis_model.pdf + "
      "figs_preview/family_basis_model.npz")
