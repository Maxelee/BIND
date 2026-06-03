"""Regenerate every figure for the Observable->f_b report with scienceplots styling.

Rules (per request): style=['science'] (no manual fontsizes), NO figure/axes titles,
shared axes where panels share a quantity, vector PDF output to report/figs/.
Run with the texlive module loaded so usetex works:
  module load texlive/20240312 && python tools/report_figures.py
"""
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' style)
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

plt.style.use(["science"])

SOBOL = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
CVC = Path("/mnt/home/mlee1/ceph/fm_testsuite_cube/CV")
CV50 = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
OUT = Path("report/figs"); OUT.mkdir(parents=True, exist_ok=True)
FCOS = 0.049 / 0.30
RNG = np.random.default_rng(0)

_spec = importlib.util.spec_from_file_location("spr", "tools/stack_profiles_reduce.py")
spr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(spr)
r = spr.R_CEN; NR = spr.NR
Nr = np.array([np.sum(spr._RBIN == b) for b in range(NR)], float)

B = np.load(SOBOL / "stacked_profiles.npz", allow_pickle=True)
MLBL = [m.replace("[", "$[").replace(")", ")$").replace("+", r"\infty") for m in B["mass_lbl"]]
RAWLBL = list(B["mass_lbl"])
Bp = {k: B[k] for k in ["Y", "SX", "T", "S", "P", "fb"]}
nD = Bp["fb"].shape[0]
des = np.load(SOBOL / "design.npz", allow_pickle=True)
pval = des["design_astro_phys"][:, list(des["astro_names"]).index("WindEnergyIn1e51erg")]
Tr = np.load(SOBOL / "truth_stacked_profiles.npz", allow_pickle=True)

LADD = {r"$Y$": ["Y"], r"$Y,S_X$": ["Y", "SX"], r"$Y,S_X,T,S,P$": ["Y", "SX", "T", "S", "P"]}
LOGF = {"Y", "SX", "P", "T", "S"}


# ---- shared helpers -------------------------------------------------------
def feat(cols, b, arr):
    out = []
    for c in cols:
        x = arr[c][:, b, :] if arr[c].ndim == 3 else arr[c][b][None, :]
        out.append(np.log10(np.clip(x, 1e-30, None)) if c in LOGF else x)
    return np.hstack(out)


def oof_profile(X, Yt, seed=0):
    pred = np.full(Yt.shape, np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        xs = StandardScaler().fit(X[tr]); ys = StandardScaler().fit(Yt[tr])
        reg = Ridge(alpha=10).fit(xs.transform(X[tr]), ys.transform(Yt[tr]))
        pred[te] = ys.inverse_transform(reg.predict(xs.transform(X[te])))
    return pred


def truth_agg(sel):
    K = ["y_sum", "SX_sum", "gas_sum", "baryon_sum", "tot_sum", "Tgas_sum", "Sgas_sum", "Pgas_sum"]
    acc = {k: np.zeros((3, NR)) for k in K}; w = np.zeros(3)
    for s in sel:
        for k in K: acc[k] += np.nan_to_num(Tr[k][s])
        w += Tr["counts"][s]
    return dict(Y=acc["y_sum"] / w[:, None], SX=acc["SX_sum"] / w[:, None],
                T=acc["Tgas_sum"] / acc["gas_sum"], S=acc["Sgas_sum"] / acc["gas_sum"],
                P=acc["Pgas_sum"] / acc["gas_sum"], fb=acc["baryon_sum"] / acc["tot_sum"])


def cap_one(O):
    out = np.full(NR, np.nan)
    for j in range(NR):
        din = r <= r[j]; ann = (r > r[j]) & (r <= np.sqrt(2) * r[j])
        if din.sum() and ann.sum():
            out[j] = np.sum(O[din] * Nr[din]) / Nr[din].sum() - np.sum(O[ann] * Nr[ann]) / Nr[ann].sum()
    return out
VALID = np.isfinite(cap_one(Bp["Y"][0, 1]))


# ---- Fig 1: stacked profiles overview -------------------------------------
def fig_profiles():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.2))
    for j, (nm, yl, lg) in enumerate([("Y", r"$Y(r)\;[\mathrm{Mpc}^2/h^2]$", True),
                                      ("SX", r"$S_X(r)$ [arb.]", True),
                                      ("fb", r"$f_b(r)$", False)]):
        P = Bp[nm][:, 1, :]; med = np.median(P, 0); lo, hi = np.percentile(P, [16, 84], 0)
        ax[j].plot(r, med); ax[j].fill_between(r, lo, hi, alpha=0.3)
        ax[j].set_xscale("log"); ax[j].set_xlabel(r"$r\;[\mathrm{kpc}/h]$"); ax[j].set_ylabel(yl)
        if lg: ax[j].set_yscale("log")
        if nm == "fb": ax[j].axhline(FCOS, ls=":", c="k", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_profiles.pdf"); plt.close(fig)


# ---- Fig 2: joint distribution p(f_b, Y) ----------------------------------
def fig_joint():
    b = 1; ri = 3
    x = np.log10(Bp["Y"][:, b, ri]); y = Bp["fb"][:, b, ri]
    fig = plt.figure(figsize=(3.6, 3.6))
    gs = GridSpec(4, 4, fig, hspace=0.04, wspace=0.04)
    axm = fig.add_subplot(gs[1:, :3]); axt = fig.add_subplot(gs[0, :3], sharex=axm)
    axr = fig.add_subplot(gs[1:, 3], sharey=axm)
    sc = axm.scatter(x, y, c=pval, s=6)
    qe = np.quantile(x, np.linspace(0, 1, 11)); xc = 0.5 * (qe[1:] + qe[:-1])
    mu = [np.median(y[(x >= qe[k]) & (x <= qe[k + 1])]) for k in range(10)]
    axm.plot(xc, mu, "k-")
    axm.set_xlabel(r"$\log_{10} Y(r{=}141\,\mathrm{kpc}/h)$"); axm.set_ylabel(r"$f_b(r)$")
    axt.hist(x, bins=28, color="0.6"); axr.hist(y, bins=28, orientation="horizontal", color="0.6")
    axt.axis("off"); axr.axis("off")
    cb = fig.colorbar(sc, ax=axr, fraction=0.5, pad=0.04)
    cb.set_label(r"$E_{\rm SN}\;[10^{51}\,\mathrm{erg}]$")
    fig.savefig(OUT / "fig_joint.pdf", bbox_inches="tight"); plt.close(fig)


# ---- Fig 3: profile->profile prediction skill -----------------------------
def fig_profile_skill():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for b in range(3):
        Yt = Bp["fb"][:, b, :]; sm = np.nanstd(Yt, 0)
        for L, cols in LADD.items():
            pred = oof_profile(feat(cols, b, Bp), Yt)
            red = 1 - np.sqrt(np.nanmean((Yt - pred) ** 2, 0)) / sm
            ax[b].plot(r, red * 100, marker="o", ms=2.5, label=L)
        ax[b].axhline(0, c="k", lw=0.5); ax[b].set_xscale("log")
        ax[b].set_xlabel(r"$r\;[\mathrm{kpc}/h]$")
        ax[b].text(0.05, 0.06, MLBL[b], transform=ax[b].transAxes)
    ax[0].set_ylabel(r"scatter reduction in $f_b(r)$ [\%]"); ax[0].set_ylim(-15, 100)
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout(); fig.savefig(OUT / "fig_profile_skill.pdf"); plt.close(fig)


# ---- Fig 4: real-data validation ------------------------------------------
def fig_realdata():
    truth = truth_agg(np.arange(Tr["counts"].shape[0]))
    boot = [truth_agg(RNG.integers(0, Tr["counts"].shape[0], Tr["counts"].shape[0])) for _ in range(200)]
    fbb = np.array([bb["fb"] for bb in boot]); flo, fhi = np.percentile(fbb, [16, 84], 0)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for b in range(3):
        ax[b].fill_between(r, flo[b], fhi[b], color="C3", alpha=0.25)
        ax[b].plot(r, truth["fb"][b], c="C3", label="CAMELS truth")
        for L, cols, ls in [(r"$Y$", ["Y"], "--"), (r"$Y,S_X$", ["Y", "SX"], "-")]:
            Xtr = feat(cols, b, Bp); ytr = Bp["fb"][:, b, :]
            xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(ytr)
            reg = Ridge(alpha=10).fit(xs.transform(Xtr), ys.transform(ytr))
            p = ys.inverse_transform(reg.predict(xs.transform(feat(cols, b, truth))))[0]
            ax[b].plot(r, p, ls=ls, c="k", lw=1.0, label="pred " + L)
        ax[b].set_xscale("log"); ax[b].set_xlabel(r"$r\;[\mathrm{kpc}/h]$")
        ax[b].text(0.05, 0.9, MLBL[b], transform=ax[b].transAxes)
    ax[0].set_ylabel(r"$f_b(r)$"); ax[0].set_ylim(0.06, 0.20); ax[0].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_realdata.pdf"); plt.close(fig)


# ---- Fig 5: projection (differential vs enclosed, depth) -------------------
def fig_projection():
    acc = {b: dict(dm=0.0, gas=0.0, star=0.0, n=0, r200=[]) for b in range(3)}
    for sd in sorted(CVC.iterdir()):
        md = sd / "snap_090/mass_threshold_1p000e13"
        if not (md / "truth_halos_cube.npz").exists(): continue
        c = np.load(md / "halo_catalog.npz")
        if "masses" not in c.files or len(c["masses"]) == 0: continue
        th = np.load(md / "truth_halos_cube.npz")["truth_halos"]
        lm = np.log10(np.asarray(c["masses"], float)); r2 = np.asarray(c["r200s"], float) * 1000
        for b, (lo, hi) in enumerate(spr.MASS_BINS):
            idx = np.where((lm >= lo) & (lm < hi))[0]
            if len(idx) == 0: continue
            acc[b]["dm"] += th[idx, 0].sum(0); acc[b]["gas"] += th[idx, 1].sum(0)
            acc[b]["star"] += th[idx, 2].sum(0); acc[b]["n"] += len(idx); acc[b]["r200"] += list(r2[idx])
    yy, xx = np.mgrid[0:128, 0:128]; Rk = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2) * spr.PIX_KPC
    cnt = Tr["counts"].sum(0); bar50 = np.nansum(Tr["baryon_sum"], 0); tot50 = np.nansum(Tr["tot_sum"], 0)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for b in range(3):
        n = acc[b]["n"]; gas = acc[b]["gas"] / n; star = acc[b]["star"] / n; dm = acc[b]["dm"] / n
        bar = gas + star; tot = dm + gas + star
        fb_diff = spr.azim(bar) / spr.azim(tot)
        enc = np.array([(bar[Rk <= rr].sum()) / (tot[Rk <= rr].sum()) for rr in r])
        R200 = np.median(acc[b]["r200"])
        ax[b].plot(r, fb_diff, marker="o", ms=2.5, label="differential, 6.25 Mpc")
        ax[b].plot(r, bar50[b] / tot50[b], marker="s", ms=2.5, ls="--", label="differential, 50 Mpc")
        ax[b].plot(r, enc, marker="^", ms=2.5, label=r"enclosed $f_b(<r)$")
        ax[b].axhline(FCOS, ls=":", c="k", lw=0.8); ax[b].axvline(R200, c="0.6", lw=0.7)
        ax[b].set_xscale("log"); ax[b].set_xlabel(r"$r\;[\mathrm{kpc}/h]$")
        ax[b].text(0.05, 0.9, MLBL[b], transform=ax[b].transAxes)
    ax[0].set_ylabel(r"$f_b$"); ax[0].set_ylim(0.06, 0.21); ax[0].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_projection.pdf"); plt.close(fig)


# ---- Fig 6: CAP validation (pred) + noise robustness ----------------------
def cap(O): return cap_one(O)[VALID]
def raw_feat(Y, SX): return np.hstack([np.log10(np.clip(Y, 1e-30, None)), np.log10(np.clip(SX, 1e-30, None))])
def cap_feat(Y, SX): return np.hstack([cap(Y), cap(SX)])
def fig_cap():
    truth = truth_agg(np.arange(Tr["counts"].shape[0]))
    def fit(featfn, b):
        X = np.array([featfn(Bp["Y"][d, b], Bp["SX"][d, b]) for d in range(nD)])
        ys = StandardScaler().fit(Bp["fb"][:, b, :]); xs = StandardScaler().fit(X)
        reg = Ridge(10).fit(xs.transform(X), ys.transform(Bp["fb"][:, b, :]))
        return lambda fv: ys.inverse_transform(reg.predict(xs.transform(fv[None])))[0]
    # pred raw vs CAP vs truth
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for b in range(3):
        pr = fit(raw_feat, b); pc = fit(cap_feat, b)
        ax[b].plot(r, truth["fb"][b], c="C3", label="CAMELS truth")
        ax[b].plot(r, pr(raw_feat(truth["Y"][b], truth["SX"][b])), ls="--", c="k", lw=1.0, label="pred (raw)")
        ax[b].plot(r, pc(cap_feat(truth["Y"][b], truth["SX"][b])), ls="-", c="C0", lw=1.2, label="pred (CAP)")
        ax[b].set_xscale("log"); ax[b].set_xlabel(r"$r\;[\mathrm{kpc}/h]$")
        ax[b].text(0.05, 0.9, MLBL[b], transform=ax[b].transAxes)
    ax[0].set_ylabel(r"$f_b(r)$"); ax[0].set_ylim(0.06, 0.20); ax[0].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_cap_pred.pdf"); plt.close(fig)
    # noise robustness (median per-measurement error)
    NOISE = [0, 0.05, 0.10, 0.20, 0.40]
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for b in range(3):
        pr = fit(raw_feat, b); pc = fit(cap_feat, b)
        for nm, pmdl, ff, c0, ls in [("raw", pr, raw_feat, "k", "--"), ("CAP", pc, cap_feat, "C0", "-")]:
            med = []
            for f in NOISE:
                e = [np.sqrt(np.nanmean((pmdl(ff(truth["Y"][b] * np.exp(RNG.normal(0, f, NR)),
                                                 truth["SX"][b] * np.exp(RNG.normal(0, f, NR))))
                                         - truth["fb"][b]) ** 2)) for _ in range(200 if f > 0 else 1)]
                med.append(np.median(e) * 1e3)
            ax[b].plot(np.array(NOISE) * 100, med, marker="o", ms=2.5, c=c0, ls=ls, label=nm)
        ax[b].set_xlabel(r"observational noise [\%]")
        ax[b].text(0.05, 0.9, MLBL[b], transform=ax[b].transAxes)
    ax[0].set_ylabel(r"median $\mathrm{RMS}\,|\hat f_b-f_b|\;(\times10^{3})$"); ax[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_cap_noise.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_profiles(); print("fig_profiles")
    fig_joint(); print("fig_joint")
    fig_profile_skill(); print("fig_profile_skill")
    fig_realdata(); print("fig_realdata")
    fig_projection(); print("fig_projection")
    fig_cap(); print("fig_cap")
    print("all figures ->", OUT)
