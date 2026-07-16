"""P4 / Phase 2 (proper observable): posterior on the 30 SB35 feedback parameters
from the DESI x ACT kSZ data IN THE COMPENSATED-APERTURE (CAP) OBSERVABLE.

This supersedes the shape-only posterior (`ksz_posterior.py`): the data vector is the
absolute tau^CAP(theta) the survey actually measures (amplitude + aperture trend, with
errors), so the posterior is conditioned on the real signal -- the legitimate "first
kSZ posterior on astrophysical feedback".

Forward model: GP(30 params -> log tau^CAP(theta)) trained on the 256 Sobol nodes,
where each node's tau^CAP is the CAP filter applied to its M*-matched tau(theta)
(central M*>11.25 bin; each node's own SHMR sets r200 -> theta). Data: digitized
DESI x ACT BGS M*>11.25 CAP points / (T_CMB sigma_v/c). emcee over the Sobol box.

NB normalization: data carry a residual (r/r_fid) display factor and an overall
sigma_v; here (r/r_fid)~1 over the fit range and sigma_v from the Fig-2 calibration
(1176 uK). Task (1) [exact r_fid + per-sample sigma_v] firms up the absolute scale;
this posterior already uses the amplitude (not just shape), flagged.

    python examples/ksz_posterior_cap.py --sample
    python examples/ksz_posterior_cap.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"


def _imp(name):
    s = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


P = _imp("ksz_posterior"); CAP = _imp("ksz_cap_compare")
PARAMS, LOG_PARAMS = P.PARAMS, P.LOG_PARAMS

# DESI x ACT BGS CAP points: theta[arcmin], T_kSZ^CAP[uK arcmin^2] per M* bin
# (r/r_fid)=1 (velocity-fidelity, =1 at fiducial). Both bins are BIND-matchable
# (host ~10^13.4-13.8); ELG is below BIND's 1e13 floor, LRG needs Part I data.
DATA_TH = np.array([2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5])
DATA_UK = {0: np.array([2.5, 4.5, 6.5, 9.0, 11.0, 13.5, 17.0]),     # M*>11.0  (cache CI 0)
           1: np.array([4.5, 8.0, 17.0, 22.0, 30.0, 37.0, 40.0])}  # M*>11.25 (cache CI 1)
CUT_LABEL = {0: "M*>11.0", 1: "M*>11.25"}
DATA_ERR_FRAC = 0.35
T_CMB_SIGV = 2.7255e6 * 300.0 / 299792.458    # uK; sigma_v^true~300 km/s (paper), (r/r_fid)=1
Z = 0.26


def _bind_cap_matrix(cis):
    """(nodes, n_theta*len(cis)) BIND tau^CAP for the given M* bins, aligned to params."""
    c = np.load(OUT / "bind_mstar_xprof_snap085.npz")
    x, nodes = c["x"], c["nodes"]
    runs, X = P._param_matrix()
    idx = np.array([np.where(runs == n)[0][0] for n in nodes])
    X = X[idx]; da = CAP._DA(Z)
    blocks = []
    for ci in cis:
        Y = np.full((len(nodes), len(DATA_TH)), np.nan)
        for i in range(len(nodes)):
            lM = c["logM200"][i, ci]
            if not np.isfinite(lM):
                continue
            theta = x * CAP._r200phys(lM, Z) / da * CAP.ARCMIN
            tau = c["tau"][i, ci, :]
            if np.isfinite(tau).all():
                Y[i] = [CAP.cap(t, theta, tau) for t in DATA_TH]
        blocks.append(Y)
    Y = np.concatenate(blocks, axis=1)
    ok = np.isfinite(Y).all(1) & (Y > 0).all(1)
    return X[ok], Y[ok]


def do_sample(cis=(1,), tag="bgs1125", nwalk=64, nstep=3000, burn=800):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
    from sklearn.model_selection import cross_val_predict
    import emcee

    X, Y = _bind_cap_matrix(cis)
    logY = np.log(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    nb = logY.shape[1]
    labels = [f"{CUT_LABEL[ci]} {t:.1f}'" for ci in cis for t in DATA_TH]
    print(f"[cap-post] {len(X)} nodes, {nb} data points ({'+'.join(CUT_LABEL[c] for c in cis)})")

    gps, emu_var = [], []
    for k in range(nb):
        kern = C(1.0) * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2, alpha=1e-6)
        cvp = cross_val_predict(gp, Xs, logY[:, k], cv=5)
        r2 = 1 - np.sum((logY[:, k] - cvp) ** 2) / np.sum((logY[:, k] - logY[:, k].mean()) ** 2)
        gp.fit(Xs, logY[:, k]); gps.append(gp); emu_var.append(np.var(logY[:, k] - cvp))
        print(f"   {labels[k]:16s}: CV R^2={r2:+.2f}")
    emu_var = np.array(emu_var)

    d_data = np.concatenate([np.log(DATA_UK[ci] / T_CMB_SIGV) for ci in cis])   # log tau^CAP
    var_data = DATA_ERR_FRAC ** 2                          # frac err -> log-space var (diagonal;
    Cinv = np.diag(1.0 / (var_data + emu_var))             # ignores published cross-cov -> optimistic
    lo, hi = Xs.min(0), Xs.max(0)

    def lnprob(TH):
        TH = np.atleast_2d(TH)
        inbox = np.all((TH >= lo) & (TH <= hi), axis=1)
        mp = np.stack([g.predict(TH) for g in gps], axis=1)
        r = mp - d_data
        lp = -0.5 * np.einsum("ni,ij,nj->n", r, Cinv, r)
        lp[~inbox] = -np.inf
        return lp

    p0 = np.random.uniform(lo, hi, size=(nwalk, Xs.shape[1]))
    sampler = emcee.EnsembleSampler(nwalk, Xs.shape[1], lnprob, vectorize=True)
    print(f"[cap-post] emcee {nwalk}x{nstep} ...")
    sampler.run_mcmc(p0, nstep, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True) * sd + mu

    prior_sd = (X.max(0) - X.min(0)) / np.sqrt(12)
    ratio = chain.std(0) / prior_sd
    np.savez(OUT / f"ksz_posterior_cap_{tag}.npz", chain=chain, params=np.array(PARAMS),
             ratio=ratio, theta=DATA_TH, d_data=d_data, emu_var=emu_var,
             log_params=np.array(sorted(LOG_PARAMS)))
    print(f"\n[cap-post] ({tag}) most-constrained (post_sd/prior_sd):")
    for j in np.argsort(ratio)[:10]:
        print(f"   {PARAMS[j]:34s} {ratio[j]:.2f}{'  [log10]' if PARAMS[j] in LOG_PARAMS else ''}")
    print(f"[cap-post] wrote {OUT}/ksz_posterior_cap_{tag}.npz")


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    # overlay constraint bars for every available posterior (single vs joint)
    files = sorted(OUT.glob("ksz_posterior_cap_*.npz"))
    if not files:
        print("[cap-post] no posterior npz"); return
    fig2, ax = plt.subplots(figsize=(7.5, 8))
    params = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        params = list(d["params"]); ratio = d["ratio"]; tag = f.stem.replace("ksz_posterior_cap_", "")
        o = np.argsort(np.load(files[-1], allow_pickle=True)["ratio"])
        ax.plot(ratio[o], range(len(params)), "o-", ms=4, label=tag)
    ax.set_yticks(range(len(params))); ax.set_yticklabels([params[i] for i in o], fontsize=7)
    ax.axvline(1, color="k", ls=":"); ax.set_xlabel("posterior sd / prior sd (<1 = constrained)")
    ax.set_title("kSZ CAP feedback constraints: single vs joint M* bins"); ax.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(OUT / "ksz_posterior_cap_constraints.png", dpi=130)
    # corner for the most complete (last) posterior
    d = np.load(files[-1], allow_pickle=True)
    chain, ratio = d["chain"], d["ratio"]; logp = set(map(str, d["log_params"]))
    order = np.argsort(ratio)[:6]
    labels = [params[j] + (" (log10)" if params[j] in logp else "") for j in order]
    fig = corner.corner(chain[:, order], labels=labels, show_titles=True, title_fmt=".2f",
                        label_kwargs=dict(fontsize=8), title_kwargs=dict(fontsize=8))
    fig.suptitle(f"kSZ CAP posterior ({files[-1].stem.replace('ksz_posterior_cap_','')}) — 6 most-constrained",
                 y=1.0, fontsize=11)
    fig.savefig(OUT / "ksz_posterior_cap_corner.png", dpi=130, bbox_inches="tight")
    print(f"[cap-post] wrote {OUT}/ksz_posterior_cap_corner.png + _constraints.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="single BGS M*>11.25 bin")
    ap.add_argument("--joint", action="store_true", help="joint BGS M*>11.0 + M*>11.25")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.sample:
        do_sample(cis=(1,), tag="bgs1125")
    if a.joint:
        do_sample(cis=(0, 1), tag="bgs_joint")
    if a.plot:
        do_plot()
    if not (a.sample or a.joint or a.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
