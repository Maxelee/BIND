"""P4 -> multi-probe: kSZ (tau, gas density) + tSZ (y, pressure) feedback FORECAST.

kSZ alone constrains the gas DENSITY (-> f_gas / ejection); the survey lives in a
~2-3D feedback subspace and is SNR-limited. tSZ adds the PRESSURE (~ thermal energy),
an orthogonal axis that separates HEATING from EJECTION. This module forecasts how
much the joint (tau^CAP, y^CAP) constrains the 30 SB35 feedback parameters beyond
kSZ alone, with a REALISTIC (non-diagonal) covariance and a clean corner.

It is a FORECAST (relevant for SO / CMB-S4 x DESI): mock data = BIND at the Sobol
median (a typical TNG feedback); per-aperture errors at SNR comparable to DESIxACT
(default 35%) with an AR(1) aperture correlation (the CAP apertures overlap, so the
7 points are NOT independent -- diagonal would be optimistic). Swap in the published
bin-bin covariance when available (set COV_PUBLISHED).

GP(30 params -> log[tau^CAP, y^CAP]) trained on the 256 Sobol nodes (the same emulator
that predicts the real kSZ, CV R^2~0.8). Posteriors: kSZ-only, tSZ-only, joint.

    python examples/ksz_posterior_multiprobe.py --sample
    python examples/ksz_posterior_multiprobe.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"


def _imp(n):
    s = importlib.util.spec_from_file_location(n, Path(__file__).resolve().parent / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


P = _imp("ksz_posterior"); CAP = _imp("ksz_cap_compare")
PARAMS, LOG_PARAMS = P.PARAMS, P.LOG_PARAMS

DATA_TH = np.array([2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5])
CI = 1                                   # M*>11.25 (cache column)
Z = 0.26
ERR_FRAC = 0.35                          # per-aperture fractional error (DESIxACT-like)
RHO_AP = 0.4                             # AR(1) aperture correlation (CAP overlap)


def _obs_matrix():
    """(nodes,7) tau^CAP and (nodes,7) y^CAP for the M*>11.25 bin, + params."""
    c = np.load(OUT / "bind_mstar_xprof_snap085.npz")
    x, nodes = c["x"], c["nodes"]
    runs, X = P._param_matrix()
    idx = np.array([np.where(runs == n)[0][0] for n in nodes])
    X = X[idx]; da = CAP._DA(Z)
    TAU = np.full((len(nodes), len(DATA_TH)), np.nan)
    Y = np.full((len(nodes), len(DATA_TH)), np.nan)
    for i in range(len(nodes)):
        lM = c["logM200"][i, CI]
        if not np.isfinite(lM):
            continue
        theta = x * CAP._r200phys(lM, Z) / da * CAP.ARCMIN
        tau, yv = c["tau"][i, CI, :], c["y"][i, CI, :]
        if np.isfinite(tau).all():
            TAU[i] = [CAP.cap(t, theta, tau) for t in DATA_TH]
        if np.isfinite(yv).all():
            Y[i] = [CAP.cap(t, theta, yv) for t in DATA_TH]
    ok = np.isfinite(TAU).all(1) & (TAU > 0).all(1) & np.isfinite(Y).all(1) & (Y > 0).all(1)
    return X[ok], TAU[ok], Y[ok]


def _ar1_cov(vals, frac=ERR_FRAC, rho=RHO_AP):
    """Realistic block covariance: diag (frac err)^2 in log-space + AR(1) aperture corr."""
    n = len(vals); sig = np.full(n, frac)                  # log-space sigma ~ frac err
    R = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.outer(sig, sig) * R


def _fit_one(Xs, yk):
    """Fit one GP; emu_var = fitted WhiteKernel noise (free emulator-error estimate,
    no cross-validation refits)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
    kern = C(1.0) * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1e1))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=1, alpha=1e-6)
    gp.fit(Xs, yk)
    emu_var = gp.kernel_.k2.noise_level * np.var(yk)        # noise on normalized-y -> log-space var
    return gp, emu_var


def _train_gps(Xs, logY):
    """Fit the GPs in PARALLEL across outputs (28 GPs on a 48-core node)."""
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=min(logY.shape[1], 14))(
        delayed(_fit_one)(Xs, logY[:, k]) for k in range(logY.shape[1]))
    return [o[0] for o in out], np.array([o[1] for o in out])


def _run_emcee(gps, d_data, Cinv, lo, hi, ndim, nwalk=96, nstep=4000, burn=1200):
    import emcee
    def lnprob(TH):
        TH = np.atleast_2d(TH); inbox = np.all((TH >= lo) & (TH <= hi), axis=1)
        mp = np.stack([g.predict(TH) for g in gps], axis=1); r = mp - d_data
        lp = -0.5 * np.einsum("ni,ij,nj->n", r, Cinv, r); lp[~inbox] = -np.inf
        return lp
    p0 = np.random.uniform(lo, hi, size=(nwalk, ndim))
    s = emcee.EnsembleSampler(nwalk, ndim, lnprob, vectorize=True)
    s.run_mcmc(p0, nstep, progress=False)
    return s.get_chain(discard=burn, flat=True)


def do_sample(err_frac=ERR_FRAC, tag_suffix=""):
    X, TAU, Y = _obs_matrix()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    lo, hi = Xs.min(0), Xs.max(0)
    lT, lY = np.log(TAU), np.log(Y)
    print(f"[mp] {len(X)} nodes, err_frac={err_frac}; training GPs (tau + y) ...")
    gT, eT = _train_gps(Xs, lT)
    gY, eY = _train_gps(Xs, lY)
    # mock data = Sobol median (a typical TNG feedback)
    dT, dY = np.median(lT, 0), np.median(lY, 0)
    CT = _ar1_cov(dT, frac=err_frac) + np.diag(eT)
    CY = _ar1_cov(dY, frac=err_frac) + np.diag(eY)
    prior_sd = (X.max(0) - X.min(0)) / np.sqrt(12)
    res = {}
    for tag, gps, d, Cb in [("ksz", gT, dT, CT), ("tsz", gY, dY, CY),
                            ("joint", gT + gY, np.concatenate([dT, dY]),
                             np.block([[CT, np.zeros_like(CT)], [np.zeros_like(CY), CY]]))]:
        chain = _run_emcee(gps, d, np.linalg.inv(Cb), lo, hi, Xs.shape[1])
        chain = chain * sd + mu
        ratio = chain.std(0) / prior_sd
        res[tag] = (chain, ratio)
        print(f"[mp] {tag:5s}: #params<0.85 prior = {(ratio<0.85).sum():2d}  best={ratio.min():.2f} ({PARAMS[ratio.argmin()]})")
    np.savez(OUT / f"ksz_posterior_multiprobe{tag_suffix}.npz",
             params=np.array(PARAMS), log_params=np.array(sorted(LOG_PARAMS)),
             prior_lo=X.min(0), prior_hi=X.max(0),
             **{f"chain_{k}": v[0] for k, v in res.items()},
             **{f"ratio_{k}": v[1] for k, v in res.items()})
    print(f"[mp] wrote {OUT}/ksz_posterior_multiprobe.npz")


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    import scienceplots  # noqa: F401  (registers the 'science' style)
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        pass

    f = OUT / "ksz_posterior_multiprobe_future.npz"
    if not f.exists():
        f = OUT / "ksz_posterior_multiprobe.npz"
    d = np.load(f, allow_pickle=True)
    print(f"[mp] plotting {f.name}")
    params = list(d["params"])

    # --- constrained DIRECTIONS, not params: feedback is ~2-3D. Whiten by the
    #     (uniform Sobol-box) prior, eigen-decompose the JOINT posterior cov; the
    #     smallest-variance eigenvectors are the directions the data actually pin. ---
    lo, hi = d["prior_lo"], d["prior_hi"]
    pmean, psd = 0.5 * (lo + hi), (hi - lo) / np.sqrt(12)
    def white(ch): return (ch - pmean) / psd
    Uj = white(d["chain_joint"])
    evals, evecs = np.linalg.eigh(np.cov(Uj.T))           # ascending eigval (= post var; prior=1)
    V = evecs[:, :2]                                       # 2 most-constrained directions
    def topload(v, n=3):
        i = np.argsort(np.abs(v))[::-1][:n]
        return " ".join(f"{'+' if v[j] > 0 else '-'}{params[j][:9]}" for j in i)
    labels = [f"dir{k+1} ({evals[k]:.2f})\n{topload(V[:, k])}" for k in range(2)]

    Ak, Aj = white(d["chain_ksz"]) @ V, Uj @ V
    ck = dict(plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95),
              smooth=1.2, bins=26, hist_kwargs=dict(density=True), range=[(-2.2, 2.2)] * 2)
    fig = corner.corner(Ak, color="tab:blue", labels=labels, label_kwargs=dict(fontsize=7), **ck)
    corner.corner(Aj, color="tab:red", fig=fig, **ck)
    fig.legend(handles=[plt.Line2D([], [], color="tab:blue", label="kSZ only"),
                        plt.Line2D([], [], color="tab:red", label="kSZ + tSZ")],
               loc="upper right", fontsize=9, frameon=False)
    fig.savefig(OUT / "ksz_multiprobe_corner.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ksz_multiprobe_corner.png", dpi=140, bbox_inches="tight")
    print(f"[mp] constrained feedback directions (post var; prior=1):")
    for k in range(2):
        print(f"   dir{k+1} var={evals[k]:.2f}: {topload(V[:, k], 4)}")

    # constraint bars: ksz vs tsz vs joint
    fig2, ax = plt.subplots(figsize=(3.4, 5.2))
    o = np.argsort(rj)
    for tag, col in [("ksz", "tab:blue"), ("tsz", "tab:green"), ("joint", "tab:red")]:
        ax.plot(d[f"ratio_{tag}"][o], range(len(params)), "o-", ms=3, color=col, label=tag)
    ax.axvline(1, color="k", ls=":", lw=.8)
    ax.set_yticks(range(len(params))); ax.set_yticklabels([params[i] for i in o], fontsize=5.5)
    ax.set_xlabel("posterior sd / prior sd"); ax.legend(loc="lower right", fontsize=7)
    fig2.savefig(OUT / "ksz_multiprobe_bars.pdf", bbox_inches="tight")
    fig2.savefig(OUT / "ksz_multiprobe_bars.png", dpi=140, bbox_inches="tight")
    print(f"[mp] wrote {OUT}/ksz_multiprobe_bars.{{pdf,png}}")
    for tag in ("ksz", "tsz", "joint"):
        r = d[f"ratio_{tag}"]
        print(f"  {tag:5s}: #<0.85 = {(r<0.85).sum()}, median ratio = {np.median(r):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="current DESIxACT-like (35%)")
    ap.add_argument("--future", action="store_true", help="SO/CMB-S4xDESI forecast (10%)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.sample:
        do_sample(err_frac=0.35, tag_suffix="")
    if a.future:
        do_sample(err_frac=0.10, tag_suffix="_future")
    if a.plot:
        do_plot()
    if not (a.sample or a.future or a.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
