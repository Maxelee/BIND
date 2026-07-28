#!/usr/bin/env python
"""Phase L (post-P4c pivot, user-directed): kSZ + tSZ constraints in the 2-D
gas latent plane, and the backtrack to the full 30-dim SB35 parameter space.
DES WL is dropped as a constraint (P4c/D4: no residual feedback
discrimination after amplitude marginalization; map leg filter-dominated).

Latent construction is VERBATIM `examples/_build_ksz_paper2_nb.py` §0/§A
(the parent paper's baseline): per-node (tau, y) profiles at snap085 in the
BGS mass bin (logM200 13.4-13.8), clean band 0.3<=x<=1.5, standardized SVD
-> 2 components (~97% of variance), then rotated in-plane so
e1 || the inner-gas gradient  (r ~ +0.97 with f~gas(<R500))
e2 -> the shell gas           (r ~ +0.95 with f~gas(R500->R200)).

Constraints = importance weights over the 253 traced Sobol nodes:
w ∝ exp(-Δχ²/2) with
  kSZ:  chi2_desi_bgs110 (M1 DESI-precision, 1-halo theta<=1.4 theta200)
  tSZ:  chi2_tsz          (T3 revised: ACT deproj-CIB, xb<=1.4, dof=6)
  joint: sum (independent data — DESI kSZ vs ACT y-map; BIND-side
  realization noise partially shared across probes, noted).
This is a PSEUDO-POSTERIOR on the Sobol design under a flat box prior — the
M1/M6 selection-test philosophy upgraded to weights; effective sample sizes
are printed on the figure (the design has only 253 nodes; contours are
importance-sampling-grade, not MCMC-grade).

Backtrack: weighted 16/50/84 percentiles of every prior-normalized SB35
param (log10-space for the 19 LogFlag params, empirical 256-design box —
the M6 conventions) under each weight set; constraint strength = 68% width
ratio vs the prior.

Outputs: figs/L_latent_corner.png, figs/L_param_backtrack.png,
KS/lightcone/latent_constraints.npz, verdicts/L.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEPH = Path("/mnt/home/mlee1/ceph")
# Products/bind_sb35 roots (Round-2 T3, docs/paper_improvement_plan.md):
# env-var override only (straight-line script, no argparse); behavior with
# neither env var set is byte-identical to before.
KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", str(CEPH / "bind_science/ksz_confront")))
LC = KS / "lightcone"
PROF_PATH = KS / "bind_tauy_xprof_snap085.npz"
_SB35_ROOT = Path(os.environ.get("BIND_SB35_RUNS", str(CEPH / "bind_sb35")))
PARQUET = _SB35_ROOT / "analysis_cache/integrated.parquet"
DESIGN = _SB35_ROOT / "design"
F_B = 0.0490 / 0.3089
BGS_BIN = 1
SNAP_BGS = 85

COL = {"prior": "0.65", "ksz": "tab:blue", "tsz": "tab:red", "joint": "k"}
LAB = {"ksz": "kSZ (DESI f̃$_{gas}$, M1)", "tsz": "tSZ (ACT y-CAP, T3)",
       "joint": "joint kSZ+tSZ"}


# ---- latent construction (verbatim _build_ksz_paper2_nb.py) ---------------

def bin_window(prof, MB):
    mb = np.asarray(prof["mbins"], float)
    if mb.size == prof["tau"].shape[1] + 1:
        return float(mb[MB]), float(mb[MB + 1])
    c = float(mb[MB]); return c - 0.2, c + 0.2


def gas_fracs(snap, lmlo, lmhi, nodes):
    fo = pd.read_parquet(PARQUET, columns=["run", "snap", "M200", "M_gas_500",
                                           "M_gas_200", "M_tot_500", "M_tot_200"])
    lo = np.log10(fo.M200.values)
    fo = fo[(fo.snap.values == snap) & (lo > lmlo) & (lo < lmhi)].copy()
    fo["f_in"] = fo.M_gas_500.values / fo.M_tot_500.values / F_B
    fo["f_out"] = (fo.M_gas_200.values - fo.M_gas_500.values) / \
        (fo.M_tot_200.values - fo.M_tot_500.values) / F_B
    gf = fo.groupby("run").median(numeric_only=True).reindex(nodes)
    return gf.f_in.values, gf.f_out.values


def svd_latent(prof, MB, use_y=True):
    x, tau, y, nodes = prof["x"], prof["tau"], prof["y"], prof["nodes"]
    band = (x >= 0.3) & (x <= 1.5)
    blocks = [np.log10(np.clip(tau[:, MB, band], 1e-30, None))]
    if use_y:
        blocks.append(np.log10(np.clip(y[:, MB, band], 1e-30, None)))
    R = np.hstack(blocks)
    good = np.isfinite(R).all(1)
    Rs = (R[good] - R[good].mean(0)) / R[good].std(0)
    U, S, Vt = svd(Rs - Rs.mean(0), full_matrices=False)
    lam = S ** 2 / np.sum(S ** 2)
    return nodes, good, U * S, lam


def rotate_to_gas(Z, f_in, f_out):
    m = np.isfinite(f_in) & np.isfinite(Z[:, 0])
    grad = np.array([np.cov(Z[m, k], f_in[m])[0, 1] for k in range(2)])
    e1 = grad / np.linalg.norm(grad); e2 = np.array([-e1[1], e1[0]])
    Ze = Z[:, :2] @ np.c_[e1, e2]
    mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
    if np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1] < 0:
        e2 = -e2; Ze = Z[:, :2] @ np.c_[e1, e2]
    return e1, e2, Ze


# ---- weighted statistics helpers ------------------------------------------

def wquantile(v, w, qs):
    o = np.argsort(v)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(qs, cw, v[o])


def ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))


def kde_contour_levels(dens, dx, dy, fracs=(0.68, 0.95)):
    s = np.sort(dens.ravel())[::-1]
    c = np.cumsum(s) * dx * dy
    # clip: with broad weights the KDE mass can extend beyond the grid, so
    # the target fraction may never be reached inside it
    return [s[min(np.searchsorted(c, f), len(s) - 1)] for f in fracs]


def main():
    # ---- latent ----------------------------------------------------------
    prof = np.load(PROF_PATH)
    lo, hi = bin_window(prof, BGS_BIN)
    nodes, good, Z, lam = svd_latent(prof, BGS_BIN, use_y=True)
    f_in, f_out = gas_fracs(SNAP_BGS, lo, hi, nodes[good])
    e1, e2, Ze = rotate_to_gas(Z, f_in, f_out)
    ng = nodes[good]
    mi = np.isfinite(f_in) & np.isfinite(Ze[:, 0])
    r_in = np.corrcoef(f_in[mi], Ze[mi, 0])[0, 1]
    mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
    r_out = np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1]
    print(f"latent: lam[:2]={lam[:2].round(2)} cum2={lam[:2].sum():.2f} "
          f"r(e1,f_in)={r_in:+.2f} r(e2,f_out)={r_out:+.2f} "
          f"(expect ~0.51/0.46, 0.97, +0.97, +0.95)")

    # ---- chi2 join -------------------------------------------------------
    kszd = np.load(LC / ("ksz_consistent_nodes_r6.npz" if (LC / "ksz_consistent_nodes_r6.npz").exists() else "ksz_consistent_nodes.npz"))
    tszd = np.load(LC / "tsz_consistent_nodes.npz")
    ids = kszd["node_ids_all"]
    pos = {int(n): i for i, n in enumerate(ng)}
    sel = np.array([pos[int(i)] for i in ids if int(i) in pos])
    ids_used = np.array([int(i) for i in ids if int(i) in pos])
    keep = np.isin(ids, ids_used)
    Zc = Ze[sel]
    chik = kszd["chi2_desi_bgs110"][keep]
    chit = tszd["chi2_tsz"][keep]
    fin_c, fout_c = f_in[sel], f_out[sel]

    W = {}
    for tag, c in [("ksz", chik), ("tsz", chit), ("joint", chik + chit)]:
        w = np.exp(-0.5 * (c - np.nanmin(c)))
        w = np.nan_to_num(w, nan=0.0)
        W[tag] = w / w.sum()
    esses = {t: ess(w) for t, w in W.items()}
    print("ESS:", {t: round(v, 1) for t, v in esses.items()})

    # ---- corner figure ---------------------------------------------------
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9),
                             gridspec_kw=dict(hspace=0.06, wspace=0.06))
    axes[0, 1].axis("off")
    z1lim = np.percentile(Zc[:, 0], [0, 100]) + np.array([-1, 1])
    z2lim = np.percentile(Zc[:, 1], [0, 100]) + np.array([-1, 1])
    g1 = np.linspace(*z1lim, 160)
    g2 = np.linspace(*z2lim, 160)

    ax = axes[0, 0]
    for tag in ("ksz", "tsz", "joint"):
        k = gaussian_kde(Zc[:, 0], weights=W[tag], bw_method=0.35)
        ax.plot(g1, k(g1), color=COL[tag], lw=2 if tag == "joint" else 1.4,
                label=LAB[tag])
    kp = gaussian_kde(Zc[:, 0], bw_method=0.35)
    ax.plot(g1, kp(g1), color=COL["prior"], ls="--", label="Sobol design (prior)")
    ax.set_xlim(*z1lim); ax.set_xticklabels([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    ax.scatter(Zc[:, 0], Zc[:, 1], s=8, c="0.8", zorder=1)
    XX, YY = np.meshgrid(g1, g2)
    for tag in ("ksz", "tsz", "joint"):
        k = gaussian_kde(Zc.T, weights=W[tag], bw_method=0.5)
        D = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        l68, l95 = kde_contour_levels(D, g1[1] - g1[0], g2[1] - g2[0])
        # ascending level order = [95% (outer), 68% (inner)]
        ax.contour(XX, YY, D, levels=[l95, l68], colors=COL[tag],
                   linewidths=(1.0, 2.0 if tag == "joint" else 1.4),
                   linestyles=("--", "-"))
    ax.set_xlim(*z1lim); ax.set_ylim(*z2lim)
    ax.set_xlabel(r"$\hat e_1$ · latent  [$\propto \tilde f_{\rm gas}(<R_{500})$,"
                  f"  r={r_in:+.2f}]")
    ax.set_ylabel(r"$\hat e_2$ · latent  [$\propto \tilde f_{\rm gas}(R_{500}\to R_{200})$,"
                  f"  r={r_out:+.2f}]")

    ax = axes[1, 1]
    for tag in ("ksz", "tsz", "joint"):
        k = gaussian_kde(Zc[:, 1], weights=W[tag], bw_method=0.35)
        ax.plot(k(g2), g2, color=COL[tag], lw=2 if tag == "joint" else 1.4)
    kp = gaussian_kde(Zc[:, 1], bw_method=0.35)
    ax.plot(kp(g2), g2, color=COL["prior"], ls="--")
    ax.set_ylim(*z2lim); ax.set_yticklabels([]); ax.set_xticks([])

    fig.suptitle("L: kSZ + tSZ pseudo-posteriors in the 2-D gas latent plane "
                 f"(68/95%)\nESS: kSZ {esses['ksz']:.0f}, tSZ {esses['tsz']:.0f}, "
                 f"joint {esses['joint']:.0f} of {len(Zc)} nodes — "
                 "importance-sampling grade", fontsize=11)
    fig.savefig(LC / "figs/L_latent_corner.png", dpi=140, bbox_inches="tight")

    # weighted latent stats + gas-fraction translation
    lat_stats = {}
    for tag in ("ksz", "tsz", "joint"):
        lat_stats[tag] = {
            "Z1_165084": [float(x) for x in wquantile(Zc[:, 0], W[tag], [.16, .5, .84])],
            "Z2_165084": [float(x) for x in wquantile(Zc[:, 1], W[tag], [.16, .5, .84])],
            "f_in_165084": [float(x) for x in wquantile(fin_c[np.isfinite(fin_c)],
                            W[tag][np.isfinite(fin_c)], [.16, .5, .84])],
            "f_out_165084": [float(x) for x in wquantile(fout_c[np.isfinite(fout_c)],
                             W[tag][np.isfinite(fout_c)], [.16, .5, .84])],
            "ess": esses[tag],
        }

    # ---- 30-dim backtrack ------------------------------------------------
    from bind.params import PARAM_NAMES, PARAM_LOG_FLAG
    sobol = np.load(DESIGN / "astro_params_sobol.npy")
    with open(DESIGN / "design.json") as f:
        dj = json.load(f)
    names, aidx = dj["astro_param_names"], dj["astro_param_indices"]
    logflag = (PARAM_LOG_FLAG[aidx] == 1)

    V = sobol[ids_used][:, aidx]                      # (n_nodes, 30)
    Vn = np.where(logflag[None, :], np.log10(np.clip(V, 1e-30, None)), V)
    vmin = np.where(logflag, np.log10(np.clip(sobol[:, aidx], 1e-30, None)), sobol[:, aidx]).min(0)
    vmax = np.where(logflag, np.log10(np.clip(sobol[:, aidx], 1e-30, None)), sobol[:, aidx]).max(0)
    U = (Vn - vmin) / (vmax - vmin)                   # prior-normalized [0,1]

    qs = {}
    for tag in ("ksz", "tsz", "joint"):
        qs[tag] = np.array([wquantile(U[:, j], W[tag], [.16, .5, .84])
                            for j in range(U.shape[1])])
    prior_q = np.array([np.percentile(U[:, j], [16, 50, 84]) for j in range(U.shape[1])])
    width_ratio = {t: (qs[t][:, 2] - qs[t][:, 0]) / (prior_q[:, 2] - prior_q[:, 0])
                   for t in qs}
    order = np.argsort(width_ratio["joint"])

    fig, ax = plt.subplots(figsize=(9, 12))
    ypos = np.arange(len(names))
    for row, j in enumerate(order):
        ax.axhspan(row - 0.42, row + 0.42, color="0.95" if row % 2 else "white", zorder=0)
        for off, tag in [(-0.22, "ksz"), (0.0, "tsz"), (0.22, "joint")]:
            q = qs[tag][j]
            ax.plot([q[0], q[2]], [row + off] * 2, color=COL[tag], lw=2.2,
                    solid_capstyle="butt")
            ax.plot(q[1], row + off, "o", color=COL[tag], ms=4)
    ax.axvline(0.5, color="0.7", lw=0.8, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{names[j]}{' (log)' if logflag[j] else ''}   "
                        f"[wr {width_ratio['joint'][j]:.2f}]" for j in order],
                       fontsize=7.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("prior-normalized parameter value (16–50–84%)")
    handles = [plt.Line2D([], [], color=COL[t], lw=2.2, label=LAB[t])
               for t in ("ksz", "tsz", "joint")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title("L: backtrack to the 30-dim SB35 space — weighted 68% intervals vs "
                 "the Sobol prior\n(sorted by joint width ratio; wr<1 = constrained)",
                 fontsize=10)
    fig.savefig(LC / "figs/L_param_backtrack.png", dpi=140, bbox_inches="tight")

    top = [(names[j], float(width_ratio["joint"][j]),
            [float(x) for x in qs["joint"][j]]) for j in order[:8]]

    # ---- ESS-robustness of the backtrack (tempered T=2 + LOO best node) --
    chi_j = chik + chit
    w_t = np.exp(-0.25 * (chi_j - np.nanmin(chi_j))); w_t /= w_t.sum()
    w_l = W["joint"].copy(); w_l[np.argmin(chi_j)] = 0; w_l /= w_l.sum()

    def wr_med(w):
        wr = np.empty(U.shape[1]); md = np.empty(U.shape[1])
        for j in range(U.shape[1]):
            q = wquantile(U[:, j], w, [.16, .5, .84])
            p = np.percentile(U[:, j], [16, 84])
            wr[j] = (q[2] - q[0]) / (p[1] - p[0]); md[j] = q[1]
        return wr, md

    wr_raw, md_raw = wr_med(W["joint"])
    wr_tmp, md_tmp = wr_med(w_t)
    wr_loo, md_loo = wr_med(w_l)
    from scipy import stats as st
    top8 = lambda wr: set(np.argsort(wr)[:8])
    stable_top = sorted(names[j] for j in (top8(wr_raw) & top8(wr_tmp) & top8(wr_loo)))
    big = np.abs(md_raw - 0.5) > 0.15
    dir_stable = [bool(np.sign(md_raw[j] - .5) == np.sign(md_tmp[j] - .5)
                       == np.sign(md_loo[j] - .5)) for j in np.nonzero(big)[0]]
    robustness = {
        "wr_rankcorr_raw_vs_tempered": float(st.spearmanr(wr_raw, wr_tmp).statistic),
        "wr_rankcorr_raw_vs_loo": float(st.spearmanr(wr_raw, wr_loo).statistic),
        "stable_top_width_constrained": stable_top,
        "n_strongly_shifted": int(big.sum()),
        "n_direction_stable": int(sum(dir_stable)),
        "direction_unstable": [str(names[j]) for j, s in
                               zip(np.nonzero(big)[0], dir_stable) if not s],
    }
    np.savez(LC / "latent_constraints.npz",
             node_ids=ids_used, Ze=Zc, f_in=fin_c, f_out=fout_c,
             w_ksz=W["ksz"], w_tsz=W["tsz"], w_joint=W["joint"],
             chi2_ksz=chik, chi2_tsz=chit,
             param_names=np.array(names), logflag=logflag,
             U_prior_normalized=U,
             q_ksz=qs["ksz"], q_tsz=qs["tsz"], q_joint=qs["joint"],
             prior_q=prior_q, lam=lam[:4], r_in=r_in, r_out=r_out,
             e1=e1, e2=e2)

    v = {
        "phase": "L", "pass": True,
        "metrics": {
            "latent": {"lam2": [float(lam[0]), float(lam[1])],
                       "cum2": float(lam[:2].sum()),
                       "r_e1_f_in": float(r_in), "r_e2_f_out": float(r_out)},
            "ess": {t: round(v_, 1) for t, v_ in esses.items()},
            "latent_constraints": lat_stats,
            "top8_joint_constrained": top,
            "robustness": robustness,
            "n_nodes": int(len(ids_used)),
        },
        "figs": ["figs/L_latent_corner.png", "figs/L_param_backtrack.png"],
        "notes": "Pseudo-posterior importance weights exp(-dchi2/2) over the 253 "
                 "traced Sobol nodes (flat box prior) — the M1/M6 selection test "
                 "upgraded to weights; NOT an MCMC posterior (ESS printed). DES WL "
                 "dropped as a constraint per user decision (D4: no residual "
                 "feedback discrimination). kSZ chi2 = bgs110 primary. Joint "
                 "assumes data independence (DESI kSZ vs ACT y-map); BIND-side "
                 "realization noise partially shared across the two chi2s.",
        "next": "paper integration",
    }
    with open(LC / "verdicts/L.json", "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v["metrics"]["latent_constraints"], indent=2))
    print("top joint-constrained:", [(n, round(w_, 2)) for n, w_, _ in top])


if __name__ == "__main__":
    main()
