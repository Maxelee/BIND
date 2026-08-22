#!/usr/bin/env python
"""family_basis_staged.py -- the family-basis model as a three-stage story
(2026-08-11 author direction):

 STAGE 1  Build the model with ALL FIVE families C1-C5 and free per-run
          least-squares amplitudes ("whatever the best parameters are",
          not physical yet) -- show it reproduces S(ell) essentially
          perfectly, against a random-basis null and the PCA ceiling.
 STAGE 2  Show the degeneracies between the C's (geometric Gram/SVD AND
          the fitted-amplitude correlations across runs), then compress.
          Subset ADOPTION RULE: at each size k, shortlist subsets within
          5e-4 of the best total R^2, then adopt the best-CONDITIONED one
          -- raw-R^2 ties hide 6-60x conditioning differences (e.g. at
          k=3, {C1,C2,C3} and {C2,C3,C4} tie at 0.9995 but have cond 119
          vs 20), and identifiable amplitudes are the point of stage 3.
          The adopted model is the k=2 pair (essentially the PCA-2
          ceiling, near-orthogonal); the cond-picked k=3 is also reported.
 STAGE 3  Tie the fitted amplitudes to physics: correlate the adopted
          reduced-basis amplitudes with the four measured halo latents
          (baryon fraction f_bar, stellar fraction f_star, gas
          concentration c_gas, temperature logT) -- "are they baryon
          fractions, stellar fractions, temperatures?"

Imports the verified machinery (basis, Sobol deviations, latents) from
family_basis_model.py -- NB that import executes the module, so its
stdout + figure re-render precede this script's output (deterministic,
identical artifacts; the 2026-08-11 adversarial verification pass covers
those components).

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_basis_staged.py
"""
from itertools import combinations

import numpy as np

# executes family_basis_model (verified 2026-08-11): ~40 s, reprints its log
from family_basis_model import (            # noqa: E402
    B5, Bk, D, FAMS, K, LATsob, S24, band_dS, ctr_b, finS, n_runs, okL,
    span_fit)
from paper_style import COLORS, TWO_COL, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NB = B5.shape[1]
LNAME = [r"$\tilde f_{\rm bar}$", r"$\tilde f_\star$", r"$c_{\rm gas}$",
         r"$\log\tilde T$"]
LWORD = ["baryon fraction", "stellar fraction", "gas concentration",
         "temperature"]
Sfin = S24[finS]
Sbar = Sfin.mean(0)

print("\n" + "=" * 70)
print("STAGED FAMILY-BASIS MODEL")
print("=" * 70)

# ══ STAGE 1: full 5-family basis, free per-run amplitudes ═══════════════════
A5, r2r_5, r2_5 = span_fit(D, B5)
rms_5 = float(np.sqrt(((D - A5 @ B5) ** 2).mean()))
sig_D = float(np.sqrt((D ** 2).mean()))

# two nulls (2026-08-11 verifier finding: the white-noise null alone is too
# weak a comparator and overstates "the families are special"):
#  (i)  5 random orthonormal directions of the 24-band space -- fails (~0.2);
#  (ii) 5 random REAL individual (unclustered) twobound response curves --
#       already spans at ~0.999, because the Sobol deviation space is
#       intrinsically ~rank-2-3 and smooth.  Stage 1's near-perfection is
#       therefore a statement about S(ell)'s LOW DIMENSIONALITY; what
#       distinguishes the clustered families is conditioning (stage 2) and
#       physical identifiability of the amplitudes (stage 3).
rngN = np.random.default_rng(7)
null_r2 = np.array([span_fit(D, np.linalg.qr(
    rngN.standard_normal((NB, 5)))[0].T)[2] for _ in range(200)])
allmem = [m for _, mem in FAMS for m in mem]
mcurv = []
for m in allmem:
    d_ = band_dS(m)
    mcurv.append(d_ / d_[np.argmax(np.abs(d_))])
mcurv = np.array(mcurv)
rngC = np.random.default_rng(11)
real_r2 = np.array([span_fit(D, mcurv[rngC.choice(len(allmem), 5,
                                                  replace=False)])[2]
                    for _ in range(500)])

print(f"\nSTAGE 1 (all five, free amplitudes): total R^2 {r2_5:.4f}, "
      f"median per-run R^2 {np.median(r2r_5):.4f}")
print(f"  residual RMS {rms_5:.4f} vs deviation RMS {sig_D:.4f} "
      f"({100 * rms_5 / sig_D:.1f}% of signal)")
print(f"  nulls: 5 random orthonormal directions median R^2 "
      f"{np.median(null_r2):.3f}; 5 random REAL single-param curves median "
      f"{np.median(real_r2):.4f} -- the deviation space is ~rank-2-3 and "
      "smooth, so ANY real curves span it; the families' value is "
      "conditioning + identifiability (stages 2-3), not span")

# ══ STAGE 2: degeneracies + compression ═════════════════════════════════════
Camp = np.corrcoef(A5.T)                       # fitted-amplitude correlations
svB = np.linalg.svd(B5, compute_uv=False)

print("\nSTAGE 2a (degeneracy): basis sv "
      + ", ".join(f"{s:.3f}" for s in svB)
      + f" (cond {svB[0]/svB[-1]:.0f})")
print("  fitted-amplitude correlation extremes: "
      f"min {Camp[np.triu_indices(K, 1)].min():+.2f}, "
      f"max {Camp[np.triu_indices(K, 1)].max():+.2f} "
      "(LSQ amplitudes trade off along the degenerate combinations)")

TOL = 5e-4
# Families whose S(ell) signal is NOT backed by a fig-05-significant member:
# C5's lone member (SNII_MinMass_Msun) sits below the S(ell) row's
# permutation null (the dotted curve of pfig_s3b_cl_clusters; see the
# "[fig05 null] clk: solid 11/30" print of paper_s3b_clusters.py, which
# lists no C5 member).  The adopted production basis must not stand on a
# sub-noise shape, so these are excluded from ADOPTION whenever a
# noise-backed subset ties within TOL (they still appear in the scored
# table and the full-5 stage-1 model).
SUBNOISE = {"C5"}
# fail loudly if reclustering ever shifts the family labels out from under
# the hardcoded SUBNOISE tag (verifier robustness note)
assert FAMS[-1][0] == "C5" and FAMS[-1][1] == ["SNII_MinMass_Msun"], \
    "family labels shifted -- re-derive SUBNOISE from the fig05 verdict"


def cond_of(c):
    s = np.linalg.svd(B5[list(c)], compute_uv=False)
    return float(s[0] / s[-1])


adopted_by_k = {}
print("\nSTAGE 2b (compression): per size k -- best R^2 | adopted "
      f"(min cond within {TOL:g} of best, noise-backed families only):")
for k in range(1, K + 1):
    scored = [(span_fit(D, B5[list(c)])[2], cond_of(c), c)
              for c in combinations(range(K), k)]
    best_r2 = max(s[0] for s in scored)
    short = [s for s in scored if s[0] >= best_r2 - TOL]
    backed = [s for s in short
              if not any(FAMS[i][0] in SUBNOISE for i in s[2])]
    adopted_by_k[k] = min(backed or short, key=lambda s: s[1])
    r2a, ca, sub = adopted_by_k[k]
    print("  k=%d: best %.4f | adopted {%s} R^2 %.4f cond %.1f"
          % (k, best_r2, ",".join(FAMS[i][0] for i in sub), r2a, ca))

r2_3a, cond_3a, RED3 = adopted_by_k[3]
r2_2a, cond_2a, RED2 = adopted_by_k[2]
RED = list(RED2)                                # ADOPTED MODEL: the k=2 pair
REDNAMES = [FAMS[i][0] for i in RED]
A2, r2r_2, r2_2 = span_fit(D, B5[RED])
A3, r2r_3, r2_3 = span_fit(D, B5[list(RED3)])
U_, sv_, _ = np.linalg.svd(D, full_matrices=False)
pca2 = float((sv_[:2] ** 2).sum() / (sv_ ** 2).sum())
print(f"  ADOPTED model: {{{','.join(REDNAMES)}}} -- R^2 {r2_2:.4f} vs "
      f"PCA-2 ceiling {pca2:.4f}, cond {cond_2a:.1f} (near-orthogonal: "
      "identifiable amplitudes, no ridge)")
print(f"  fallback k=3 {{{','.join(FAMS[i][0] for i in RED3)}}}: R^2 "
      f"{r2_3a:.4f}, cond {cond_3a:.1f} (if the last 0.4% matters)")

# ══ STAGE 3: what ARE the adopted amplitudes? ═══════════════════════════════
LAT = LATsob[okL]                              # (n_ok, 4) measured latents
Aok = A2[okL]
CORR = np.zeros((len(RED), 4))
for j in range(len(RED)):
    for i in range(4):
        CORR[j, i] = np.corrcoef(Aok[:, j], LAT[:, i])[0, 1]

# all-4-latent linear CV R^2 per amplitude (5-fold, protocol as before)
Lc = LAT - LAT.mean(0)
ordL = np.random.default_rng(1).permutation(len(Lc))
pred = np.full_like(Aok, np.nan)
for f in np.array_split(ordL, 5):
    tr = np.setdiff1d(ordL, f)
    W, *_ = np.linalg.lstsq(np.c_[Lc[tr], np.ones(len(tr))], Aok[tr],
                            rcond=None)
    pred[f] = np.c_[Lc[f], np.ones(len(f))] @ W
r2cv = 1 - ((Aok - pred) ** 2).sum(0) / ((Aok - Aok.mean(0)) ** 2).sum(0)
D_pred = pred @ B5[RED]
r2_e2e = 1 - ((D[okL] - D_pred) ** 2).sum() / (D[okL] ** 2).sum()
rms_e2e = float(np.sqrt(((D[okL] - D_pred) ** 2).mean()))
# the shipped 4-kernel latent model under the SAME 5-fold CV protocol
# (kernels refit per fold, prediction = Lam @ beta on the held-out fold)
predK = np.full_like(D[okL], np.nan)
Dok = D[okL]
for f in np.array_split(ordL, 5):
    tr = np.setdiff1d(ordL, f)
    Wk, *_ = np.linalg.lstsq(np.c_[Lc[tr], np.ones(len(tr))], Dok[tr],
                             rcond=None)
    predK[f] = np.c_[Lc[f], np.ones(len(f))] @ Wk
r2_ship_cv = 1 - ((Dok - predK) ** 2).sum() / (Dok ** 2).sum()

print("\nSTAGE 3 (identification of the adopted amplitudes):")
best_i = np.argmax(np.abs(CORR), 1)
for j, nm in enumerate(REDNAMES):
    i = int(best_i[j])
    others = ", ".join(f"{LWORD[q]} {CORR[j, q]:+.2f}"
                       for q in range(4) if q != i)
    print(f"  a_{nm}: strongest correlate = {LWORD[i]} "
          f"(r = {CORR[j, i]:+.2f}); {others}; all-4 linear CV R^2 "
          f"{r2cv[j]:.2f}")
print(f"  END-TO-END reduced model, lambda -> (a) [CV] -> S(ell): total "
      f"R^2 {r2_e2e:.4f}, RMS {rms_e2e:.4f}")
print(f"  same-protocol reference: 4-latent kernel model, CV-refit per "
      f"fold: R^2 {r2_ship_cv:.4f} -- the 2-amplitude physical model "
      f"gives up {r2_ship_cv - r2_e2e:.4f}")

np.savez("figs_preview/family_model_stages.npz",
         basis=B5, fam_names=[f for f, _ in FAMS], ell=ctr_b,
         amps_full5=A5, amps_adopted=A2, adopted_idx=RED,
         amps_k3=A3, k3_idx=list(RED3),
         r2=(r2_5, r2_3, r2_2), amp_corr_latents=CORR, r2cv_latents=r2cv)

# ══ FIGURE 1: stages 1 + 2 ══════════════════════════════════════════════════
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]
i_sup = int(np.argmin(D.mean(1)))
i_enh = int(np.argmax(D.mean(1)))
# worst case of the ADOPTED reduced model (verifier figure-honesty fix:
# argmin(r2r_5) showcased a run the reduced model handles comfortably)
i_bad = int(np.argmin(r2r_2))
EX = [(i_sup, "strongest suppression"), (i_enh, "strongest enhancement"),
      (i_bad, "worst-fit run (reduced model)")]

fig, ((pa, pb), (pc, pd)) = plt.subplots(2, 2, figsize=(TWO_COL[0], 4.8))
for q, (ir, tag) in enumerate(EX):
    pa.plot(ctr_b, Sbar + D[ir], "o", ms=2.6, color=FC[q], alpha=0.85,
            label=tag)
    pa.plot(ctr_b, Sbar + A5[ir] @ B5, color=FC[q], lw=1.3)
    pa.plot(ctr_b, Sbar + A2[ir] @ B5[RED], color=FC[q], lw=1.0, ls="--")
pa.plot([], [], "k-", lw=1.3, label="model, all 5 families")
pa.plot([], [], "k--", lw=1.0,
        label="model, adopted {%s}" % ",".join(REDNAMES))
pa.axhline(1, color="0.85", lw=0.5, zorder=0)
pa.set_xscale("log")
pa.set_xlabel(r"$\ell$", fontsize=6.5)
pa.set_ylabel(r"$S(\ell)$", fontsize=6.5)
pa.legend(fontsize=4.8, loc="lower left", handletextpad=0.5)
pa.set_title("stage 1: measured (points) vs family model (lines)",
             fontsize=6.6)
panel_label(pa, "(a)")

bins = np.geomspace(1e-5, 0.3, 30)
for r2r, lab_, col in [
        (r2r_5, "all 5 families", COLORS["bind"]),
        (r2r_3, "k=3 {%s}" % ",".join(FAMS[i][0] for i in RED3), "0.45"),
        (r2r_2, "adopted {%s}" % ",".join(REDNAMES), COLORS["highlight"])]:
    pb.hist(np.clip(1 - r2r, bins[0], bins[-1]), bins=bins, histtype="step",
            lw=1.3, color=col, label=lab_)
pb.axvline(1 - np.median(null_r2), color="0.7", lw=0.9, ls=":")
pb.text(1 - np.median(null_r2), pb.get_ylim()[1] * 0.55,
        " random 5-dim\n basis (median)", fontsize=4.8, color="0.4",
        ha="right")
pb.axvline(1 - np.median(real_r2), color="0.7", lw=0.9, ls="--")
pb.text(1 - np.median(real_r2), pb.get_ylim()[1] * 0.83,
        " 5 random real\n curves (median)", fontsize=4.8, color="0.4",
        ha="left")
pb.set_xscale("log")
pb.set_xlabel(r"per-run $1-R^2$", fontsize=6.5)
pb.set_ylabel("runs", fontsize=6.5)
pb.legend(fontsize=4.8, loc="upper left")
pb.set_title(f"reconstruction quality ({n_runs} Sobol runs)", fontsize=6.6)
panel_label(pb, "(b)")

pc.imshow(Camp, cmap="coolwarm", vmin=-1, vmax=1)
for i in range(K):
    for j in range(K):
        pc.text(j, i, f"{Camp[i, j]:+.2f}", ha="center", va="center",
                fontsize=5.2, color="w" if abs(Camp[i, j]) > 0.6 else "k")
pc.set_xticks(range(K)); pc.set_xticklabels([f for f, _ in FAMS], fontsize=6)
pc.set_yticks(range(K)); pc.set_yticklabels([f for f, _ in FAMS], fontsize=6)
pc.tick_params(length=0)
pc.set_title("stage 2: fitted-amplitude correlations across runs\n"
             "(basis sv " + ", ".join(f"{s:.2f}" for s in svB) + ")",
             fontsize=6.2)
panel_label(pc, "(c)")

ks = np.arange(1, K + 1)
pd.semilogy(ks, [1 - adopted_by_k[k][0] for k in ks], "o-",
            color=COLORS["bind"], lw=1.3, ms=4, label="adopted subset")
for k in ks:
    r2, _, c = adopted_by_k[k]
    pd.annotate("{" + ",".join(FAMS[i][0] for i in c) + "}",
                (k, 1 - r2), textcoords="offset points", xytext=(4, 4),
                fontsize=4.8)
pd.semilogy(ks, [1 - (sv_[:k] ** 2).sum() / (sv_ ** 2).sum() for k in ks],
            "s--", color="0.5", lw=1.0, ms=3, label="PCA ceiling")
pd.set_xticks(ks)
pd.set_xlabel("number of families kept", fontsize=6.5)
pd.set_ylabel(r"$1-R^2$ (total)", fontsize=6.5)
pd.legend(fontsize=5.2, loc="upper right")
pd.set_title("stage 2: compression -- conditioning-aware subsets",
             fontsize=6.6)
panel_label(pd, "(d)")
fig.tight_layout()
save(fig, "figs_v2/pfig_family_model_stages")
plt.close(fig)

# ══ FIGURE 2: stage 3 -- what are the two adopted amplitudes? ═══════════════
fig, AX = plt.subplots(2, 2, figsize=(TWO_COL[0], 4.6))
AX = AX.ravel()
for j, nm in enumerate(REDNAMES):
    ax = AX[j]
    i = int(best_i[j])
    ax.scatter(LAT[:, i], Aok[:, j], s=6, color=FC[RED[j]], alpha=0.55,
               rasterized=True)
    ax.set_xlabel(LNAME[i] + f"  ({LWORD[i]})", fontsize=6.5)
    ax.set_ylabel(rf"$a_{{\rm {nm}}}$", fontsize=6.5)
    ax.set_title(f"$a_{{\\rm {nm}}}$ vs {LWORD[i]} "
                 f"($r={CORR[j, i]:+.2f}$; all-4 CV $R^2$={r2cv[j]:.2f})",
                 fontsize=6.3)
    panel_label(ax, f"({'ab'[j]})")

axp = AX[2]
sc = axp.scatter(Aok[:, 0], Aok[:, 1], s=7, c=LAT[:, 0], cmap="viridis",
                 alpha=0.8, rasterized=True)
cb = plt.colorbar(sc, ax=axp, fraction=0.046, pad=0.03)
cb.set_label(LNAME[0] + " (baryon fraction)", fontsize=6.0)
cb.ax.tick_params(labelsize=5.5)
axp.set_xlabel(rf"$a_{{\rm {REDNAMES[0]}}}$", fontsize=6.5)
axp.set_ylabel(rf"$a_{{\rm {REDNAMES[1]}}}$", fontsize=6.5)
axp.set_title("the model's amplitude plane, colored by baryon fraction",
              fontsize=6.4)
panel_label(axp, "(c)")

axh = AX[3]
axh.imshow(CORR, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
for j in range(len(RED)):
    for i in range(4):
        axh.text(i, j, f"{CORR[j, i]:+.2f}", ha="center", va="center",
                 fontsize=6.2,
                 color="w" if abs(CORR[j, i]) > 0.6 else "k")
axh.set_xticks(range(4)); axh.set_xticklabels(LNAME, fontsize=6.5)
axh.set_yticks(range(len(RED)))
axh.set_yticklabels([rf"$a_{{\rm {n}}}$" for n in REDNAMES], fontsize=6.5)
axh.tick_params(length=0)
axh.set_title("amplitude--latent correlations", fontsize=6.6)
panel_label(axh, "(d)")
fig.tight_layout()
save(fig, "figs_v2/pfig_family_model_amps")
plt.close(fig)
print("\nwrote pfig_family_model_stages + pfig_family_model_amps "
      "+ figs_preview/family_model_stages.npz")
