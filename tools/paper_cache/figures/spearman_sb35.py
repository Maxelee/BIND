#!/usr/bin/env python3
"""Referee revision: SB35-only parameter-response Spearman figures with a paired
bootstrap significance test.

Replaces
    fig3_spearman_param_mass.pdf   (SB35 / Test suite only, N = 101 sims)
    fig3_spearman_profile.pdf      (same, radial-profile grid)
and adds
    fig3_spearman_param_mass_pooled.pdf   (all 267 sims, appendix)

Rationale
---------
The original figures pooled CV + 1P + SB35.  CV has *identical* parameters in
every sim (zero variance -> pure noise contribution) and 1P varies one
parameter at a time (all other columns are constant within a 1P set), so a
pooled Spearman over parameters is not a clean latin-hypercube response test.
SB35 is the only suite where all 30 astrophysical parameters vary jointly and
independently, so the SB35-only version is the defensible parameter-response
measurement.  The pooled version is kept as an appendix cross-check.

Significance
------------
Paired bootstrap over *simulations*: each draw resamples the N sims with
replacement and recomputes BOTH rho_True and rho_BIND on the same resampled
set, so the True/BIND correlation is preserved and sigma applies to the
residual  Delta = rho_True - rho_BIND  directly.  Cells with
|Delta| > 2 sigma_boot are marked.

Everything here is CPU-only and loads only the spine cache.

Env (must be set before import; values select the model cache, e.g. fm_two_head):
    PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_testsuite
    PAPER_MODEL_SUBDIR=fm_two_head
    PAPER_MASS_DIR=mass_threshold_1p000e13
    PAPER_MODEL_TAG=fm_two_head
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))
import paper_config as C  # noqa: E402

# ── style: identical to examples/_build_paper_nbs.py SETUP ──────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook"])
except Exception:
    pass
plt.rcParams["hatch.linewidth"] = 1.2

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)
CACHE = C.CACHE_DIR

N_BOOT = 2000
SEED = 20260814
MASS_CH = C.MASS_CHANNELS                     # DM_hydro, Gas, Stars
CH_DISPLAY = C.CH_DISPLAY
ASTRO = [j for j in range(C.N_PARAMS) if (j + 1) not in C.COSMO_PARAM_IDX]  # 30
XLABELS = [C.PARAM_LABELS[j + 1] for j in ASTRO]
WINDOW = "trained"                            # M200c >= 1e13, the paper cut


def save_fig(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", FIG_DIR / f"{name}.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Spearman + paired bootstrap
# ════════════════════════════════════════════════════════════════════════════
def _zrank(a):
    """Rank along axis 1, then centre+normalise so a dot product is Pearson-r."""
    r = rankdata(a, axis=1).astype(np.float64)
    r -= r.mean(1, keepdims=True)
    s = np.sqrt((r ** 2).sum(1, keepdims=True))
    return np.divide(r, s, out=np.zeros_like(r), where=s > 0)


def spearman_point(P, Y):
    """P (n, np) params, Y (n, nq) quantities -> rho (nq, np) point estimate."""
    nq, npar = Y.shape[1], P.shape[1]
    rho = np.full((nq, npar), np.nan)
    for q in range(nq):
        y = Y[:, q]
        for j in range(npar):
            x = P[:, j]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 5 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
                rho[q, j] = spearmanr(x[m], y[m]).statistic
    return rho


def paired_bootstrap(P, Ytrue, Ygen, n_boot=N_BOOT, seed=SEED, block=200):
    """Paired bootstrap over sims.

    Returns (sigma_delta, sigma_true, sigma_bind), each (nq, npar):
    bootstrap std of Delta = rho_True - rho_BIND, and of each rho separately.
    """
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    dstack, tstack, gstack = [], [], []
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        zP = _zrank(P[idx])                 # (b, n, npar)
        zT = _zrank(Ytrue[idx])             # (b, n, nq)
        zG = _zrank(Ygen[idx])
        rt = np.einsum("bnp,bnq->bqp", zP, zT)
        rg = np.einsum("bnp,bnq->bqp", zP, zG)
        dstack.append(rt - rg)
        tstack.append(rt)
        gstack.append(rg)
        done += b
    D = np.concatenate(dstack, 0)
    T = np.concatenate(tstack, 0)
    G = np.concatenate(gstack, 0)
    return D.std(0, ddof=1), T.std(0, ddof=1), G.std(0, ddof=1)


# ════════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════════
def load_mass(subset):
    """subset in {'Test', 'all'} -> (P, Ytrue, Ygen, channels, n_sims)."""
    mp = pickle.load(open(CACHE / "mass_param.pkl", "rb"))
    st: pd.DataFrame = mp["sim_table"].copy()
    if subset != "all":
        st = st[st.suite == subset]
    st = st.reset_index(drop=True)
    P = st[[f"p{j+1}" for j in range(C.N_PARAMS)]].to_numpy(float)[:, ASTRO]
    T = st[[f"true_logmean_{ch}_{WINDOW}" for ch in MASS_CH]].to_numpy(float)
    G = st[[f"gen_logmean_{ch}_{WINDOW}" for ch in MASS_CH]].to_numpy(float)
    return P, T, G, st


def load_prof(subset):
    d = pickle.load(open(CACHE / "profiles_r200.pkl", "rb"))
    suites = np.asarray(d["suites"])
    sims = np.asarray(d["sims"])
    sel = np.ones(len(suites), bool) if subset == "all" else (suites == subset)
    P = np.asarray(d["params"], float)[sel][:, ASTRO]
    T = np.asarray(d["mean_prof"][WINDOW]["truth"], float)[sel]   # (n, 3, nr)
    G = np.asarray(d["mean_prof"][WINDOW]["gen"], float)[sel]
    return P, T, G, np.asarray(d["r_over_r200"], float), suites[sel], sims[sel]


def check_join(subset):
    """The two cache tables are written by different reducers; assert they
    describe the same sims (join key = suite + sim_id) before comparing."""
    mp = pickle.load(open(CACHE / "mass_param.pkl", "rb"))
    st = mp["sim_table"]
    st = st if subset == "all" else st[st.suite == subset]
    _, _, _, _, psuite, psim = load_prof(subset)
    a = list(zip(st.suite, st.sim_id))
    b = list(zip(psuite, psim))
    assert set(a) == set(b), f"sim sets differ for {subset}: {len(a)} vs {len(b)}"
    return len(a)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1/2 — parameter -> integrated mass Spearman heat-map
# ════════════════════════════════════════════════════════════════════════════
def fig_mass(subset, outname, subtitle):
    P, T, G, st = load_mass(subset)
    n = len(st)
    rho_t = spearman_point(P, T)
    rho_g = spearman_point(P, G)
    delta = rho_t - rho_g
    sig_d, sig_t, sig_g = paired_bootstrap(P, T, G)
    signif = np.abs(delta) > 2.0 * sig_d

    vmax = 0.5
    nch = len(MASS_CH)
    fig, axes = plt.subplots(nch, 1, figsize=(17.2, 7.9), sharex=True,
                             gridspec_kw={"hspace": 0.15})
    ims = []
    for ci, (ch, ax) in enumerate(zip(MASS_CH, axes)):
        data = np.vstack([rho_t[ci], rho_g[ci], delta[ci]])
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ims.append(im)
        for ri in range(3):
            for xj in range(len(ASTRO)):
                v = data[ri, xj]
                if not np.isfinite(v):
                    continue
                mark = (ri == 2) and signif[ci, xj]
                ax.text(xj, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=10.5, color="white" if abs(v) > 0.45 else "black",
                        fontweight="bold" if mark else "normal")
                if mark:
                    ax.add_patch(Rectangle((xj - 0.5, ri - 0.5), 1, 1, fill=False,
                                           edgecolor="k", lw=1.6, zorder=5))
        ax.axhline(1.5, color="k", lw=1.5, ls="--")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["True", "BIND", "Resid."])
        ax.set_ylabel(CH_DISPLAY[ch], labelpad=6)
        ax.tick_params(axis="both", which="both", length=0)

    axes[-1].set_xticks(range(len(ASTRO)))
    axes[-1].set_xticklabels(XLABELS, rotation=45, ha="right")
    axes[0].set_title(
        f"{subtitle}   ($N = {n}$ sims, $M_{{200c}} \\geq 10^{{13}}\\,M_\\odot/h$)   "
        f"boxed/bold: $|\\Delta\\rho_{{\\rm S}}| > 2\\sigma$ "
        f"(paired bootstrap, {N_BOOT} draws)",
        fontsize=15, pad=10)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(ims[0], cax=cax).set_label(r"$\rho_{\rm S}$", fontsize=20)
    save_fig(fig, outname)
    plt.close(fig)

    return dict(n=n, rho_true=rho_t, rho_bind=rho_g, delta=delta,
                sigma=sig_d, signif=signif, sig_true=sig_t, sig_bind=sig_g)


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — parameter -> radial-profile Spearman (BIND | True - BIND)
# ════════════════════════════════════════════════════════════════════════════
def fig_profile(subset, outname, subtitle, make_fig=True):
    P, T, G, rr, _, _ = load_prof(subset)
    n, nch, nr = T.shape
    Tf = T.reshape(n, nch * nr)
    Gf = G.reshape(n, nch * nr)
    rho_t = spearman_point(P, Tf).reshape(nch, nr, len(ASTRO))
    rho_g = spearman_point(P, Gf).reshape(nch, nr, len(ASTRO))
    delta = rho_t - rho_g
    sig_d, _, _ = paired_bootstrap(P, Tf, Gf)
    sig_d = sig_d.reshape(nch, nr, len(ASTRO))
    signif = np.abs(delta) > 2.0 * sig_d

    if not make_fig:
        return dict(n=n, rho_true=rho_t, rho_bind=rho_g, delta=delta,
                    sigma=sig_d, signif=signif, r=rr)

    # display on a log-spaced r/R200 grid, exactly as the current figure does
    R_LOG = np.logspace(np.log10(rr[0]), np.log10(rr[-1]), nr)
    r1_row = int(np.argmin(np.abs(R_LOG - 1.0)))

    def to_log(mat):                      # mat (nr, npar) on the linear rr grid
        out = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            v = np.isfinite(col)
            if v.sum() >= 2:
                out[:, j] = np.interp(R_LOG, rr[v], col[v])
        return out

    def to_log_bool(mat):                 # nearest-row resample for the mask
        rows = [int(np.argmin(np.abs(rr - v))) for v in R_LOG]
        return mat[rows]

    vmax = 0.5
    fig, axes = plt.subplots(nch, 2, figsize=(15, 10.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.15, "wspace": 0.04})
    r_tick_vals = [0.1, 0.2, 0.5, 1.0, 2.0]
    r_tick_idx = [int(np.argmin(np.abs(R_LOG - v))) for v in r_tick_vals]

    ims = []
    for ci in range(nch):
        panels = [("BIND", rho_g[ci], None), ("True $-$ BIND", delta[ci], signif[ci])]
        for si, (lab, data, mask) in enumerate(panels):
            ax = axes[ci, si]
            im = ax.imshow(to_log(data), aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="upper",
                           interpolation="nearest")
            ims.append(im)
            if mask is not None:
                mk = to_log_bool(mask)
                for k in range(nr):
                    for xj in range(len(ASTRO)):
                        if mk[k, xj]:
                            ax.add_patch(Rectangle(
                                (xj - 0.5, k - 0.5), 1, 1, fill=False,
                                edgecolor="k", lw=0.5, hatch="///", zorder=5))
            ax.axhline(r1_row, color="k", lw=1.2, ls="--", alpha=0.7)
            if si == 0:
                ax.set_ylabel(f"$\\Sigma_{{\\rm {CH_DISPLAY[MASS_CH[ci]]}}}$\n"
                              r"$r/R_{200}$")
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_linewidth(1.5)
            if ci == 0:
                ax.set_title(lab, pad=4)

    for ci in range(nch):
        axes[ci, 0].set_yticks(r_tick_idx)
        axes[ci, 0].set_yticklabels([str(v) for v in r_tick_vals], fontsize=10)
        axes[ci, 0].tick_params(axis="both", which="both", length=0)
        axes[ci, 1].tick_params(axis="both", which="both", labelleft=False, length=0)
    for si in (0, 1):
        axes[-1, si].set_xticks(range(len(ASTRO)))
        axes[-1, si].set_xticklabels(XLABELS, rotation=45, ha="right", fontsize=8.5)

    fig.suptitle(
        f"{subtitle}   ($N = {n}$ sims, $M_{{200c}} \\geq 10^{{13}}\\,M_\\odot/h$)   "
        f"hatched: $|\\Delta\\rho_{{\\rm S}}| > 2\\sigma$ "
        f"(paired bootstrap, {N_BOOT} draws)",
        fontsize=15, y=0.945)
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(ims[0], cax=cax).set_label(r"$\rho_{\rm S}$", fontsize=18)
    save_fig(fig, outname)
    plt.close(fig)

    return dict(n=n, rho_true=rho_t, rho_bind=rho_g, delta=delta,
                sigma=sig_d, signif=signif, r=rr)


# ════════════════════════════════════════════════════════════════════════════
def report_mass(tag, res):
    print(f"\n── {tag}: parameter -> integrated mass  (N = {res['n']} sims) ──")
    print(f"{'channel':<12}{'RMS|dRho|':>10}{'max|dRho|':>10}  "
          f"{'at param':<32}{'>2sig':>8}{'of':>6}")
    for ci, ch in enumerate(MASS_CH):
        d = res["delta"][ci]
        k = int(np.nanargmax(np.abs(d)))
        print(f"{CH_DISPLAY[ch]:<12}{np.sqrt(np.nanmean(d**2)):>10.4f}"
              f"{abs(d[k]):>10.4f}  {C.PARAM_LABELS[ASTRO[k]+1]:<32}"
              f"{int(res['signif'][ci].sum()):>8}{len(ASTRO):>6}")
    d = res["delta"]
    print(f"{'ALL':<12}{np.sqrt(np.nanmean(d**2)):>10.4f}"
          f"{np.nanmax(np.abs(d)):>10.4f}  {'':<32}"
          f"{int(res['signif'].sum()):>8}{d.size:>6}")
    print(f"   median sigma_boot(Delta) = {np.nanmedian(res['sigma']):.4f}"
          f"   |  mean |rho_True| = {np.nanmean(np.abs(res['rho_true'])):.4f}"
          f"   |  mean |rho_BIND| = {np.nanmean(np.abs(res['rho_bind'])):.4f}")


def report_prof(tag, res):
    nr = res["delta"].shape[1]
    print(f"\n── {tag}: parameter -> radial profile  (N = {res['n']} sims, "
          f"{nr} radial bins x {len(ASTRO)} params) ──")
    print(f"{'channel':<12}{'RMS|dRho|':>10}{'max|dRho|':>10}  "
          f"{'at param':<32}{'r/R200':>8}{'>2sig':>8}{'of':>6}")
    for ci, ch in enumerate(MASS_CH):
        d = res["delta"][ci]
        k = np.unravel_index(int(np.nanargmax(np.abs(d))), d.shape)
        print(f"{CH_DISPLAY[ch]:<12}{np.sqrt(np.nanmean(d**2)):>10.4f}"
              f"{abs(d[k]):>10.4f}  {C.PARAM_LABELS[ASTRO[k[1]]+1]:<32}"
              f"{res['r'][k[0]]:>8.2f}{int(res['signif'][ci].sum()):>8}{d.size:>6}")
    d = res["delta"]
    print(f"{'ALL':<12}{np.sqrt(np.nanmean(d**2)):>10.4f}"
          f"{np.nanmax(np.abs(d)):>10.4f}  {'':<32}{'':>8}"
          f"{int(res['signif'].sum()):>8}{d.size:>6}")
    print(f"   median sigma_boot(Delta) = {np.nanmedian(res['sigma']):.4f}")


def main():
    print(f"cache {CACHE}")
    n_test = check_join("Test")
    n_all = check_join("all")
    print(f"join check OK: Test {n_test} sims, pooled {n_all} sims")

    mt = pd.read_pickle(CACHE / "mass_table.pkl")
    print(f"mass_table: {len(mt)} halos, min log10 M200c = {mt.log_m200c.min():.4f} "
          f"(cut >= 13 verified: {mt.log_m200c.min() >= 13.0})")

    r_sb = fig_mass("Test", "fig3_spearman_param_mass", "SB35 (Test) suite")
    r_pool = fig_mass("all", "fig3_spearman_param_mass_pooled",
                      "Pooled CV + 1P + SB35")
    p_sb = fig_profile("Test", "fig3_spearman_profile", "SB35 (Test) suite")
    p_pool = fig_profile("all", None, None, make_fig=False)

    report_mass("SB35 only", r_sb)
    report_mass("POOLED", r_pool)
    report_prof("SB35 only", p_sb)
    report_prof("POOLED", p_pool)

    # a couple of extra numbers the caption will want
    print("\n── extra ──")
    for tag, res in (("SB35", r_sb), ("POOLED", r_pool)):
        for ci, ch in enumerate(MASS_CH):
            t = res["rho_true"][ci]
            k = int(np.nanargmax(np.abs(t)))
            print(f"{tag:<7}{CH_DISPLAY[ch]:<12} strongest True rho_S = "
                  f"{t[k]:+.3f} ({C.PARAM_LABELS[ASTRO[k]+1]}), "
                  f"BIND {res['rho_bind'][ci][k]:+.3f}, "
                  f"Delta {res['delta'][ci][k]:+.3f} +- {res['sigma'][ci][k]:.3f}")
    # which cells are flagged
    for tag, res in (("SB35", r_sb), ("POOLED", r_pool)):
        w = np.argwhere(res["signif"])
        print(f"\n{tag} flagged mass cells ({len(w)}):")
        for ci, k in w:
            print(f"   {CH_DISPLAY[MASS_CH[ci]]:<12}{C.PARAM_LABELS[ASTRO[k]+1]:<32}"
                  f"Delta {res['delta'][ci,k]:+.3f}  sigma {res['sigma'][ci,k]:.3f}  "
                  f"({abs(res['delta'][ci,k])/res['sigma'][ci,k]:.1f} sigma)")


if __name__ == "__main__":
    main()
