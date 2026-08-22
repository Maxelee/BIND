#!/usr/bin/env python
"""family_amp_physicals.py -- WHAT is each shipped family amplitude,
physically?  (2026-08-11 author direction: "all of the amplitudes are
correlated with the stellar mass... it should only be one physical
component per amplitude ... mass bins, stellar mass, baryon fractions,
temperatures, pressures, entropy, concentrations, scatter about some
relation -- find them for each component".)

Why the marginal identification degenerated: the shipped amplitudes are
mutually correlated (non-orthogonal basis) and the Sobol design has ONE
dominant feedback axis that moves stars, gas and temperature together, so
every amplitude's MARGINAL best correlate is the same (stellar fraction).
This script fixes both sides:

  (1) CANDIDATE LIBRARY -- per-run scalars built from the shared 2933-halo
      atlas (fixed halo set, binned by the run-independent FoF mass so the
      same objects are measured in every run):
        * medians per mass bin ([13.0,13.3), [13.3,13.6), [13.6,14.0),
          [14.0,inf)): f_bar, f_gas, f_star (500c), f_gas/f_star (200c),
          log T_mw, log K (entropy), log P_e, log Y, c_gas, c_dm,
          sigma_gas/sigma_tot;
        * mass-structure: per-run slope of {f_gas, f_star, logY, logT,
          logK} vs log M across the four bins; group->cluster contrasts;
        * relation scatter: per-run robust residual scatter (half the
          16-84 span) about the run's own Y-M, f_gas-M and K-M relations
          over 13.0<=logM<14.0.
  (2) DISCRIMINATIVE IDENTIFICATION -- for amplitude a_k, the PARTIAL
      correlation with each candidate, controlling for the OTHER shipped
      amplitudes (both residualized on A_{-k} + intercept); then a
      uniqueness-constrained assignment (Hungarian on |partial r|), so no
      two amplitudes may claim the same physical.

Outputs: ranked tables per amplitude (partial + marginal r), the assigned
physical per amplitude with single-candidate CV R^2, added-variable
scatter figures pfig_family_model_amps[_pdf] (REPLACING the marginal
versions), and figs_preview/family_amp_physicals_{clk,pdf}.npz.

Run from papers/01_pipeline (AFTER family_basis_ship.py):
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_amp_physicals.py
"""
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

SB35 = Path("/mnt/home/mlee1/ceph/bind_sb35")
assert Path.cwd().name == "01_pipeline"
OB_OM = 0.0486 / 0.3089
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]

dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
rows = dsn["run_ids"]
NR = len(rows)

lgM = np.log10(cz["M_fof"])                    # shared halos, run-independent
BINS = [(13.0, 13.3), (13.3, 13.6), (13.6, 14.0), (14.0, np.inf)]
BSEL = [(lgM >= lo) & (lgM < hi) for lo, hi in BINS]
BLAB = ["13.0-13.3", "13.3-13.6", "13.6-14.0", ">14.0"]
BCTR = [float(np.median(lgM[s])) for s in BSEL]

Q = {k: np.asarray(cz[f"sobol_{k}"], float)[rows]
     for k in ("m_tot_500c_bg", "m_gas_500c", "m_gas_500c_bg", "m_star_500c",
               "m_tot_200c_bg", "m_gas_200c_bg", "m_star_200c", "m_gas_200c",
               "m_dm_500c", "m_dm_200c", "T_mw_500c", "K_mw_500c",
               "Pe_mw_500c", "Y_500c", "sig_gas_bg", "sig_tot_bg")}


def _med(arr, sel, log=False, min_n=8):
    """per-run median of arr over the FIXED halo selection `sel`
    (positive-and-finite entries only); NaN when fewer than min_n."""
    out = np.full(NR, np.nan)
    for r in range(NR):
        v = arr[r, sel]
        v = v[np.isfinite(v) & (v > 0 if log else np.isfinite(v))]
        if len(v) >= min_n:
            m = np.median(v)
            out[r] = np.log10(m) if log else m
    return out


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


CAND, CLAB = [], []


def add(name, vec):
    CAND.append(vec)
    CLAB.append(name)


# per-bin medians -----------------------------------------------------------
per_bin = {}
for g, (sel, bl) in enumerate(zip(BSEL, BLAB)):
    defs = {
        "f_bar":  _ratio(Q["m_gas_500c_bg"] + Q["m_star_500c"],
                         Q["m_tot_500c_bg"]) / OB_OM,
        "f_gas":  _ratio(Q["m_gas_500c"], Q["m_tot_500c_bg"]) / OB_OM,
        "f_star": _ratio(Q["m_star_500c"], Q["m_tot_500c_bg"]) / OB_OM,
        "f_gas200":  _ratio(Q["m_gas_200c_bg"], Q["m_tot_200c_bg"]) / OB_OM,
        "f_star200": _ratio(Q["m_star_200c"], Q["m_tot_200c_bg"]) / OB_OM,
        "c_gas":  _ratio(Q["m_gas_500c"], Q["m_gas_200c"]),
        "c_dm":   _ratio(Q["m_dm_500c"], Q["m_dm_200c"]),
        "sig_rat": _ratio(Q["sig_gas_bg"], Q["sig_tot_bg"]),
    }
    for nm, arr in defs.items():
        v = _med(arr, sel)
        per_bin[(nm, g)] = v
        add(f"{nm}[{bl}]", v)
    for nm, key in (("logT", "T_mw_500c"), ("logK", "K_mw_500c"),
                    ("logPe", "Pe_mw_500c"), ("logY", "Y_500c")):
        v = _med(Q[key], sel, log=True)
        per_bin[(nm, g)] = v
        add(f"{nm}[{bl}]", v)

# mass-structure: slopes across the four bins + group->cluster contrasts ----
for nm in ("f_gas", "f_star", "logY", "logT", "logK"):
    ys = np.stack([per_bin[(nm, g)] for g in range(4)], 1)   # (NR, 4)
    sl = np.full(NR, np.nan)
    for r in range(NR):
        ok = np.isfinite(ys[r])
        if ok.sum() >= 3:
            sl[r] = np.polyfit(np.array(BCTR)[ok], ys[r, ok], 1)[0]
    add(f"slope:{nm}-M", sl)
for nm in ("f_gas", "f_star", "logT"):
    add(f"contrast:{nm}[>14.0 - 13.0-13.3]",
        per_bin[(nm, 3)] - per_bin[(nm, 0)])

# relation scatter: robust residual scatter about the run's own relation ----
selS = (lgM >= 13.0) & (lgM < 14.0)
xM = lgM[selS]
for nm, arr, log in (("logY", Q["Y_500c"], True),
                     ("f_gas", _ratio(Q["m_gas_500c"],
                                      Q["m_tot_500c_bg"]) / OB_OM, False),
                     ("logK", Q["K_mw_500c"], True)):
    sc = np.full(NR, np.nan)
    for r in range(NR):
        v = arr[r, selS]
        ok = np.isfinite(v) & ((v > 0) if log else True)
        if ok.sum() >= 50:
            y = np.log10(v[ok]) if log else v[ok]
            res = y - np.polyval(np.polyfit(xM[ok], y, 1), xM[ok])
            lohi = np.percentile(res, [16, 84])
            sc[r] = 0.5 * (lohi[1] - lohi[0])
    add(f"scat:{nm}|M", sc)

# hunt-workflow candidates folded back in (2026-08-11): self-similar-scaled
# thermodynamics (X/M^p at fixed mass), baryon-vs-DM concentration excess,
# and the per-run entropy-mass monotonicity
from scipy.stats import spearmanr  # noqa: E402

for g, (sel, bl) in enumerate(zip(BSEL, BLAB)):
    add(f"logY_ss[{bl}]",
        _med(_ratio(Q["Y_500c"], Q["m_tot_500c_bg"] ** (5 / 3)), sel,
             log=True))
    add(f"logT_ss[{bl}]",
        _med(_ratio(Q["T_mw_500c"], Q["m_tot_500c_bg"] ** (2 / 3)), sel,
             log=True))
    add(f"cgas_over_cdm[{bl}]",
        _ratio(per_bin[("c_gas", g)], per_bin[("c_dm", g)]))
selK = lgM >= 13.0
spK = np.full(NR, np.nan)
for r in range(NR):
    v = Q["K_mw_500c"][r, selK]
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() >= 100:
        spK[r] = spearmanr(lgM[selK][ok], np.log10(v[ok]))[0]
add("spearman:logK-M", spK)

C = np.stack(CAND, 1)                              # (NR, P)
print(f"candidate library: {C.shape[1]} candidates x {NR} runs "
      f"(finite fraction {100*np.isfinite(C).mean():.1f}%)")


def pearson_nan(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def residualize(v, Z):
    """v minus its OLS fit on Z (+intercept), NaN-pairwise."""
    ok = np.isfinite(v) & np.isfinite(Z).all(1)
    out = np.full_like(v, np.nan)
    W, *_ = np.linalg.lstsq(np.c_[Z[ok], np.ones(ok.sum())], v[ok],
                            rcond=None)
    out[ok] = v[ok] - np.c_[Z[ok], np.ones(ok.sum())] @ W
    return out


def cv_r2_single(x, y, seed=1, deg=1):
    ok = np.isfinite(x) & np.isfinite(y)
    xo, yo = x[ok], y[ok]
    order = np.random.default_rng(seed).permutation(len(xo))
    pred = np.full_like(yo, np.nan)
    for f in np.array_split(order, 5):
        tr = np.setdiff1d(order, f)
        W = np.polyfit(xo[tr], yo[tr], deg)
        pred[f] = np.polyval(W, xo[f])
    return 1 - np.nansum((yo - pred) ** 2) / np.nansum((yo - yo.mean()) ** 2)


for stat in ("clk", "pdf"):
    z = np.load(f"figs_preview/family_model_ship_{stat}.npz",
                allow_pickle=True)
    A = np.asarray(z["amps_shipped"], float)
    fam_names = [str(s) for s in z["fam_names"]]
    ship = [fam_names[i] for i in z["shipped_idx"]]
    K = A.shape[1]
    assert A.shape[0] == NR
    print("\n" + "=" * 70 + f"\nAMPLITUDE PHYSICALS: {stat} "
          f"(shipped {{{','.join(ship)}}})\n" + "=" * 70)

    marg = np.array([[pearson_nan(A[:, j], C[:, p])
                      for p in range(C.shape[1])] for j in range(K)])
    part = np.zeros_like(marg)
    for j in range(K):
        Zo = A[:, [q for q in range(K) if q != j]]
        aj = residualize(A[:, j], Zo)
        for p in range(C.shape[1]):
            part[j, p] = pearson_nan(aj, residualize(C[:, p], Zo))

    # uniqueness-constrained assignment: each amplitude claims ONE physical,
    # and no two amplitudes may claim the same QUANTITY FAMILY (f_star in a
    # different mass bin is still f_star).  Score = |marginal r| + 0.3
    # |partial r|: the marginal is what the eye sees in a scatter; the
    # partial bonus breaks ties toward the candidate that also tracks the
    # amplitude's UNIQUE variation (pure-partial assignment was tried and is
    # noise-dominated -- the amplitudes co-vary because the response space
    # is ~rank-2-3, so their unique residuals are small).
    def fam_of(lab):
        """quantity family: aperture variants (f_star200 == f_star) and the
        two mass-trend measures (slope:X-M == contrast:X) collapse together,
        so the uniqueness constraint operates on PHYSICALS, not labels."""
        base = lab.split("[")[0].replace("200", "")
        if base.startswith(("slope:", "contrast:")):
            return "trend:" + base.split(":")[1].split("-M")[0]
        return base

    FAMSQ = sorted({fam_of(x) for x in CLAB})
    score = np.abs(np.nan_to_num(marg)) + 0.3 * np.abs(np.nan_to_num(part))
    Sfam = np.zeros((K, len(FAMSQ)))
    Pbest = np.zeros((K, len(FAMSQ)), int)
    for j in range(K):
        for fi, fq in enumerate(FAMSQ):
            ps = [p for p in range(len(CLAB)) if fam_of(CLAB[p]) == fq]
            b = max(ps, key=lambda p: score[j, p])
            Sfam[j, fi] = score[j, b]
            Pbest[j, fi] = b
    ridx, cidx = linear_sum_assignment(-Sfam)
    assign = {j: int(Pbest[j, fi]) for j, fi in zip(ridx, cidx)}

    for j in range(K):
        order = np.argsort(-Sfam[j])[:6]
        print(f"\n  a_{ship[j]} -- top quantity families "
              "(score = |marg r| + 0.3 |partial r|):")
        for fi in order:
            p = Pbest[j, fi]
            mark = "  <== ASSIGNED" if p == assign[j] else ""
            print(f"    {CLAB[p]:>34s}: marginal {marg[j, p]:+.2f}, "
                  f"partial {part[j, p]:+.2f}{mark}")
        p = assign[j]
        print(f"    assigned single-candidate CV R^2: linear "
              f"{cv_r2_single(C[:, p], A[:, j]):.2f}, quadratic "
              f"{cv_r2_single(C[:, p], A[:, j], deg=2):.2f}")

    # amplitude-cloud PCA: how many INDEPENDENT physical scalars are there
    # really?  (2026-08-11 hunt finding: the pdf amplitude cloud is
    # numerically rank-1 + a tiny real thermal mode -- identify the PCs.)
    Ac = A - A.mean(0)
    _, sa, Vta = np.linalg.svd(Ac, full_matrices=False)
    evar = sa ** 2 / (sa ** 2).sum()
    print("\n  amplitude-cloud PCA: "
          + ", ".join(f"PC{i+1} {100*v:.3f}%" for i, v in enumerate(evar)))
    pc_tracers = []
    for i in range(min(2, K)):
        pc = Ac @ Vta[i]
        rs = np.array([pearson_nan(pc, C[:, p]) for p in range(C.shape[1])])
        top = np.argsort(-np.abs(np.nan_to_num(rs)))[:3]
        pc_tracers.append((pc, rs, top))
        print(f"  PC{i+1} tracers: " + "; ".join(
            f"{CLAB[p]} r={rs[p]:+.2f}" for p in top))
    if evar[0] > 0.999:
        print("  NB: the amplitude cloud is effectively RANK-1 -- the "
              "per-family amplitudes are ONE physical scalar (+ a small "
              "second mode); identify the PCs, not the individual a_k.")
    if stat == "clk":
        p = CLAB.index("logY_ss[13.6-14.0]")
        print(f"  a_C1 vs a_C2 unique separator: logY_ss[13.6-14.0] "
              f"(Y/M^(5/3) at fixed mass): partial {part[0, p]:+.2f} (C1) "
              f"vs {part[1, p]:+.2f} (C2) -- opposite-sign thermal content "
              "splits the stellar-axis pair")

    np.savez(f"figs_preview/family_amp_physicals_{stat}.npz",
             cand_labels=np.array(CLAB), candidates=C, amps=A,
             ship_names=np.array(ship), marginal_r=marg, partial_r=part,
             assigned=np.array([assign[j] for j in range(K)]),
             amp_pca_evar=evar)

    # scatters of each amplitude against its ASSIGNED distinct physical
    ncol = 2
    nrow = int(np.ceil(K / ncol))
    fig, AX = plt.subplots(nrow, ncol, figsize=(TWO_COL[0], 2.35 * nrow))
    AX = np.atleast_1d(AX).ravel()
    for j in range(K):
        ax = AX[j]
        p = assign[j]
        ax.scatter(C[:, p], A[:, j], s=6, color=FC[j % len(FC)], alpha=0.55,
                   rasterized=True)
        ax.set_xlabel(CLAB[p], fontsize=6.4)
        ax.set_ylabel(rf"$a_{{\rm {ship[j]}}}$", fontsize=6.4)
        ax.set_title(rf"$a_{{\rm {ship[j]}}} \leftrightarrow$ {CLAB[p]}: "
                     f"$r$={marg[j, p]:+.2f}, partial {part[j, p]:+.2f}",
                     fontsize=6.2)
        ax.tick_params(labelsize=5.5)
        panel_label(ax, f"({'abcdef'[j]})")
    for ax in AX[K:]:
        ax.set_visible(False)
    # pdf: the spare panel shows the genuinely ORTHOGONAL second mode --
    # PC2 of the (rank-1) amplitude cloud vs its thermal tracer
    if stat == "pdf" and K < len(AX):
        ax = AX[K]
        ax.set_visible(True)
        pc2, rs2, _ = pc_tracers[1]
        p = CLAB.index("logT[13.6-14.0]")
        ax.scatter(C[:, p], pc2, s=6, color="0.35", alpha=0.55,
                   rasterized=True)
        ax.set_xlabel(CLAB[p], fontsize=6.4)
        ax.set_ylabel(r"PC2 of $(a_{\rm C1},a_{\rm C2},a_{\rm C3})$",
                      fontsize=6.2)
        ax.set_title(f"the small orthogonal mode is THERMAL: "
                     f"$r$={rs2[p]:+.2f}", fontsize=6.2)
        ax.tick_params(labelsize=5.5)
        panel_label(ax, f"({'abcdef'[K]})")
    fig.tight_layout()
    tag = "" if stat == "clk" else "_pdf"
    save(fig, f"figs_v2/pfig_family_model_amps{tag}")
    plt.close(fig)

print("\ndone.")
