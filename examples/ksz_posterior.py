"""P4 / Phase 2: posterior on the 30 SB35 feedback parameters from the DESI x ACT
kSZ tau profile (Hadzhiyska+26), via a Gaussian-process emulator over the Sobol grid.

This is the headline P4 claim: the first continuous-parameter (CAMELS-style)
posterior on astrophysical feedback from a direct kSZ confrontation.

Observable = the *shape* of the projected tau profile in the clean band x in
[0.3,1.5] (log-tau minus its clean-band mean -> removes the absolute amplitude,
which we cannot compare cleanly; keeps the outer-slope / extent info that beta
carries). Forward model = GP(30 params -> shape) trained on the 256 BIND Sobol
nodes. Data = the Hadzhiyska+26 GNFW for a MASS-MATCHED BGS bin (default
BGS_Ms11.25, host ~10^13.5, highest SNR), with the measurement covariance from
Monte-Carlo over its (alpha, beta) errors. Posterior sampled with emcee under a
uniform prior on the Sobol box.

Honest scope: BIND's halo floor (M200>=1e13) means we condition on the massive BGS
bins (matched mass), not BGS_all (beta=7.3, outside BIND's range) or ELG (below the
floor). The constraint is expected to live in the ~2-3 effective feedback
directions (cf [[sobol-feedback-latent]]); most of the 30 params stay prior-wide.

    python examples/ksz_posterior.py --sample BGS_Ms11.25
    python examples/ksz_posterior.py --plot
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
PARQUET = CEPH / "bind_sb35/analysis_cache/integrated.parquet"
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"

# reuse the GNFW projection + table + constants from the D2 module
_spec = importlib.util.spec_from_file_location(
    "ksz_tau_gnfw", Path(__file__).resolve().parent / "ksz_tau_gnfw.py")
K = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(K)

PARAMS = ['WindEnergyIn1e51erg', 'RadioFeedbackFactor', 'VariableWindVelFactor',
          'RadioFeedbackReiorientationFactor', 'MaxSfrTimescale', 'FactorForSofterEQS',
          'IMFslope', 'SNII_MinMass_Msun', 'ThermalWindFraction', 'VariableWindSpecMomentum',
          'WindFreeTravelDensFac', 'MinWindVel', 'WindEnergyReductionFactor',
          'WindEnergyReductionMetallicity', 'WindEnergyReductionExponent', 'WindDumpFactor',
          'SeedBlackHoleMass', 'BlackHoleAccretionFactor', 'BlackHoleEddingtonFactor',
          'BlackHoleFeedbackFactor', 'BlackHoleRadiativeEfficiency', 'QuasarThreshold',
          'QuasarThresholdPower', 'UVBH0beta', 'UVBH0Deltaz', 'UVBHepbeta', 'UVBHepDeltaz',
          'SNIa_Rate_Norm', 'SNIa_Rate_DTD_power', 'SofteningComovingType01']
# log-sampled params (positive, span decades) -> emulate/prior in log10
LOG_PARAMS = {'WindEnergyIn1e51erg', 'RadioFeedbackFactor', 'VariableWindVelFactor',
              'SNII_MinMass_Msun', 'VariableWindSpecMomentum', 'SeedBlackHoleMass',
              'BlackHoleAccretionFactor', 'BlackHoleEddingtonFactor', 'BlackHoleFeedbackFactor',
              'BlackHoleRadiativeEfficiency', 'SNIa_Rate_Norm'}

SNAP_BIN = {"BGS": (85, 1), "ELG": (46, 0)}        # massive BGS / ELG-edge mass bin


def _param_matrix():
    """Per-run 30-d param vector (log10 where appropriate), aligned to run index."""
    df = pd.read_parquet(PARQUET, columns=["run"] + PARAMS).drop_duplicates("run").set_index("run").sort_index()
    X = df[PARAMS].to_numpy(float)
    for j, p in enumerate(PARAMS):
        if p in LOG_PARAMS:
            X[:, j] = np.log10(np.clip(X[:, j], 1e-30, None))
    return df.index.to_numpy(), X


def _shape_vector(prof, x):
    """log tau in the clean band, minus its clean-band mean (amplitude removed)."""
    m = (x >= K.FIT_XMIN) & (x <= K.FIT_XMAX)
    s = np.log(np.clip(prof[..., m], 1e-300, None))
    return s - s.mean(axis=-1, keepdims=True), x[m]


def _data_target(sample, x_clean, n_mc=4000, seed=0):
    """MC the Hadzhiyska GNFW (alpha,beta) errors -> shape mean + covariance."""
    h = K.HADZ[sample]; rng = np.random.default_rng(seed)
    a = rng.normal(*h["alpha"], n_mc).clip(0.05, None)
    b = rng.normal(*h["beta"], n_mc).clip(1.01, None)
    S = np.array([np.log(np.clip(K.project_tau_shape(x_clean, ai, bi), 1e-300, None)) for ai, bi in zip(a, b)])
    S = S - S.mean(axis=1, keepdims=True)
    return S.mean(0), np.cov(S.T)


def _build(tracer):
    snap, mbin = SNAP_BIN[tracer]
    c = np.load(OUT / f"bind_tauy_xprof_snap{snap:03d}.npz")
    runs, X = _param_matrix()
    nodes = c["nodes"]
    idx = np.array([np.where(runs == n)[0][0] for n in nodes])
    X = X[idx]                                              # align params to cached nodes
    Yshape, x_clean = _shape_vector(c["tau"][:, mbin, :], c["x"])
    ok = np.isfinite(Yshape).all(1)
    return X[ok], Yshape[ok], x_clean, c["x"]


def do_sample(tracer, sample, nwalk=64, nstep=3000, burn=800):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
    import emcee

    X, Y, x_clean, _ = _build(tracer)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    nb = Y.shape[1]
    print(f"[post] {tracer}/{sample}: {len(X)} nodes, {nb} clean-band bins")

    # GP per shape-bin (ARD Matern) + leave-out CV R^2
    from sklearn.model_selection import cross_val_predict
    gps, emu_var = [], []
    for k in range(nb):
        kern = C(1.0) * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2, alpha=1e-6)
        yk = Y[:, k]
        cvp = cross_val_predict(gp, Xs, yk, cv=5)
        r2 = 1 - np.sum((yk - cvp) ** 2) / np.sum((yk - yk.mean()) ** 2)
        gp.fit(Xs, yk); gps.append(gp); emu_var.append(np.var(yk - cvp))
        print(f"   bin {k} (x={x_clean[k]:.2f}): CV R^2={r2:+.2f}")
    emu_var = np.array(emu_var)

    d_data, C_data = _data_target(sample, x_clean)
    Cmat = C_data + np.diag(emu_var)
    Cinv = np.linalg.inv(Cmat)

    lo, hi = Xs.min(0), Xs.max(0)                            # Sobol box (standardized)

    def lnprob(TH):                                          # vectorized over walkers
        TH = np.atleast_2d(TH)
        inbox = np.all((TH >= lo) & (TH <= hi), axis=1)
        mu_pred = np.stack([g.predict(TH) for g in gps], axis=1)   # (nw, nb)
        r = mu_pred - d_data
        lp = -0.5 * np.einsum("ni,ij,nj->n", r, Cinv, r)
        lp[~inbox] = -np.inf
        return lp

    p0 = np.random.uniform(lo, hi, size=(nwalk, Xs.shape[1]))
    sampler = emcee.EnsembleSampler(nwalk, Xs.shape[1], lnprob, vectorize=True)
    print(f"[post] emcee {nwalk}x{nstep} (burn {burn}) ...")
    sampler.run_mcmc(p0, nstep, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    chain_phys = chain * sd + mu                            # back to physical/log10 units

    # constraint ranking: posterior std / prior std
    prior_sd = (X.max(0) - X.min(0)) / np.sqrt(12)
    post_sd = chain_phys.std(0)
    ratio = post_sd / prior_sd
    np.savez(OUT / f"ksz_posterior_{tracer}_{sample}.npz",
             chain=chain_phys, params=np.array(PARAMS), ratio=ratio,
             x_clean=x_clean, d_data=d_data, C_data=C_data, emu_var=emu_var,
             log_params=np.array(sorted(LOG_PARAMS)))
    order = np.argsort(ratio)
    print(f"\n[post] most-constrained params (post_sd/prior_sd):")
    for j in order[:10]:
        print(f"   {PARAMS[j]:34s} {ratio[j]:.2f}{'  [log10]' if PARAMS[j] in LOG_PARAMS else ''}")
    print(f"[post] wrote {OUT}/ksz_posterior_{tracer}_{sample}.npz")


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    files = sorted(OUT.glob("ksz_posterior_*.npz"))
    if not files:
        print("[post] no posterior npz -- run --sample first"); return
    d = np.load(files[-1], allow_pickle=True)
    chain, params, ratio = d["chain"], list(d["params"]), d["ratio"]
    tag = files[-1].stem.replace("ksz_posterior_", "")

    # corner of the 6 most-constrained params
    order = np.argsort(ratio)[:6]
    labels = [params[j] + (" (log10)" if params[j] in set(map(str, d["log_params"])) else "") for j in order]
    fig = corner.corner(chain[:, order], labels=labels, show_titles=True,
                        title_fmt=".2f", label_kwargs=dict(fontsize=8), title_kwargs=dict(fontsize=8))
    fig.suptitle(f"kSZ posterior on feedback ({tag}) — 6 most-constrained", y=1.0, fontsize=11)
    fig.savefig(OUT / f"ksz_posterior_corner_{tag}.png", dpi=130, bbox_inches="tight")
    print(f"[post] wrote {OUT}/ksz_posterior_corner_{tag}.png")

    # constraint bar: post_sd/prior_sd per param (sorted)
    fig, ax = plt.subplots(figsize=(7, 8))
    o = np.argsort(ratio)
    ax.barh(range(len(params)), ratio[o], color=["tab:red" if ratio[oi] < 0.85 else "tab:gray" for oi in o])
    ax.set_yticks(range(len(params))); ax.set_yticklabels([params[i] for i in o], fontsize=7)
    ax.axvline(1.0, color="k", ls=":"); ax.set_xlabel("posterior sd / prior sd  (<1 = constrained)")
    ax.set_title(f"kSZ feedback constraints ({tag})")
    fig.tight_layout(); fig.savefig(OUT / f"ksz_posterior_constraints_{tag}.png", dpi=130)
    print(f"[post] wrote {OUT}/ksz_posterior_constraints_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=str, help="Hadzhiyska sample, e.g. BGS_Ms11.25")
    ap.add_argument("--tracer", type=str, default="BGS", choices=["BGS", "ELG"])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.sample:
        do_sample(a.tracer, a.sample)
    if a.plot:
        do_plot()
    if not (a.sample or a.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
