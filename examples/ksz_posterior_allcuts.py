"""§7 (expanded): the kSZ feedback posterior using ALL the DESI x ACT stellar-mass cuts,
not just the two M*>11.0/11.25 BGS bins.

The 2-bin posterior (`ksz_posterior_cap.py`) under-uses the data: DESI x ACT publishes the
CAP-ratio f~gas for 5 BGS M* cuts (snap 85, z=0.18) and 3 ELG cuts (snap 46, z=1.16). This
script builds the posterior from ALL of them, reusing the cached per-node f~gas(M) relations
(`fgas_lowmass_snap085/046.npz`, the §6b reduction) so NO new painting is needed:

  observable per cut = BIND f~gas at that cut's host mass (interp of the per-node f~gas-M
                       relation), matched aperture (r200 for BGS, innermost ~2.8 r200 for ELG);
  data per cut       = the REAL Zenodo Fig8 f~gas at the same aperture, with its error.

Honest caveat (the reason the headline still leads with 2 bins): BIND's painted central M* is
floor-limited (M200 >= 1e13), so the 3 lowest BGS cuts collapse to host logM 13.36-13.43 -- they
probe nearly the SAME mass as the 11.0 bin and add little INDEPENDENT info. The ELG cuts add a
genuinely independent z=1.16 / low-mass lever (reuse-captured, truth-validated to ~10% in §6b).

Produces three posteriors with identical machinery so §7 can compare like-for-like:
  bgs2   : BGS M*>11.0 + 11.25          (2 cuts, current scope)
  bgsall : all 5 BGS cuts               (snap 85)
  all    : 5 BGS + 3 ELG                (8 cuts, snap 85 + 46)

    python examples/ksz_posterior_allcuts.py            # writes ksz_posterior_cap_{bgs2,bgsall,all}.npz
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
KS = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"
PARQUET = CEPH / "bind_sb35/analysis_cache/integrated.parquet"

Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100 * 0.6774, 180 * 60 / np.pi
def Ez(z): return np.sqrt(Om * (1 + z) ** 3 + OL)
def DA(z, n=3000):
    zz = np.linspace(0, z, n); return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz) / (1 + z)
def r200phys(lM, z):
    M = 10 ** lM / h; rho = 2.775e11 * h ** 2 * Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)

PARAMS = ['WindEnergyIn1e51erg','RadioFeedbackFactor','VariableWindVelFactor',
 'RadioFeedbackReiorientationFactor','MaxSfrTimescale','FactorForSofterEQS','IMFslope',
 'SNII_MinMass_Msun','ThermalWindFraction','VariableWindSpecMomentum','WindFreeTravelDensFac',
 'MinWindVel','WindEnergyReductionFactor','WindEnergyReductionMetallicity','WindEnergyReductionExponent',
 'WindDumpFactor','SeedBlackHoleMass','BlackHoleAccretionFactor','BlackHoleEddingtonFactor',
 'BlackHoleFeedbackFactor','BlackHoleRadiativeEfficiency','QuasarThreshold','QuasarThresholdPower',
 'UVBH0beta','UVBH0Deltaz','UVBHepbeta','UVBHepDeltaz','SNIa_Rate_Norm','SNIa_Rate_DTD_power',
 'SofteningComovingType01']
import pyarrow.parquet as pq
PARAMS = [p for p in PARAMS if p in set(pq.read_schema(PARQUET).names)]

# each cut: (snap, z, tracer, M*cut-string, host logM200, aperture mode)
CUTS_BGS = [("085", 0.26, "BGS_BRIGHT-20.2", "9.50", 13.36, "r200"),
            ("085", 0.26, "BGS_BRIGHT-20.2", "10.00", 13.38, "r200"),
            ("085", 0.26, "BGS_BRIGHT-20.2", "10.50", 13.43, "r200"),
            ("085", 0.26, "BGS_BRIGHT-20.2", "11.00", 13.60, "r200"),
            ("085", 0.26, "BGS_BRIGHT-20.2", "11.25", 13.82, "r200")]
CUTS_ELG = [("046", 1.16, "ELG_LOPnotqso", "9.00", 12.33, "outer"),
            ("046", 1.16, "ELG_LOPnotqso", "9.50", 12.34, "outer"),
            ("046", 1.16, "ELG_LOPnotqso", "10.00", 12.38, "outer")]


def _bind_data(cuts):
    """Build BIND per-node prediction Y (nodes, ncuts) + data d, err for the given cuts."""
    Xdf = pd.read_parquet(PARQUET, columns=["run"] + PARAMS).groupby("run").first()
    node_ids = None; cols, dvals, derr = [], [], []
    for snap, zc, tracer, cut, hm, ap in cuts:
        fl = np.load(KS / f"fgas_lowmass_snap{snap}.npz")
        lm, sb, ids = fl["logM"], fl["sb35"], fl["node_ids"]
        node_ids = ids if node_ids is None else node_ids
        # BIND f~gas at the cut's host mass, per node (interp the f~gas-M relation)
        pred = np.array([np.interp(hm, lm[np.isfinite(row)], row[np.isfinite(row)]) if np.isfinite(row).sum() > 1 else np.nan
                         for row in sb])
        dz = np.load(KS / f"desact_zenodo/Fig8_{tracer}_logm{cut}.npz")
        if ap == "r200":
            thr = r200phys(hm, zc) / DA(zc) * ARCMIN
            dvals.append(float(np.interp(thr, dz["th"], dz["ratio"]))); derr.append(float(np.interp(thr, dz["th"], dz["yerr"])))
        else:
            dvals.append(float(dz["ratio"][0])); derr.append(float(dz["yerr"][0]))
        cols.append(pred)
    Y = np.column_stack(cols)                                    # (nodes, ncuts)
    X = Xdf.reindex(node_ids).values
    ok = np.isfinite(Y).all(1) & np.isfinite(X).all(1)
    return X[ok], Y[ok], np.array(dvals), np.array(derr)


def posterior(cuts, tag, nwalk=64, nstep=3000, burn=800, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as Ck, WhiteKernel
    from sklearn.model_selection import cross_val_predict
    import emcee
    rng = np.random.default_rng(seed)
    X, Y, d, derr = _bind_data(cuts)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    nb = Y.shape[1]
    print(f"[{tag}] {len(X)} nodes, {nb} cuts; data f~gas={np.round(d,2)}")
    gps, emu_var = [], []
    for k in range(nb):
        kern = Ck(1.0) * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2, alpha=1e-6)
        cvp = cross_val_predict(gp, Xs, Y[:, k], cv=5)
        r2 = 1 - np.sum((Y[:, k] - cvp) ** 2) / np.sum((Y[:, k] - Y[:, k].mean()) ** 2)
        gp.fit(Xs, Y[:, k]); gps.append(gp); emu_var.append(np.var(Y[:, k] - cvp))
        print(f"   cut {k}: CV R^2={r2:+.2f}")
    emu_var = np.array(emu_var)
    Cinv = np.diag(1.0 / (derr ** 2 + emu_var))                  # diagonal: per-cut data err + emulator err
    lo, hi = Xs.min(0), Xs.max(0)
    def lnprob(TH):
        TH = np.atleast_2d(TH); inbox = np.all((TH >= lo) & (TH <= hi), axis=1)
        mp = np.stack([g.predict(TH) for g in gps], axis=1); r = mp - d
        lp = -0.5 * np.einsum("ni,ij,nj->n", r, Cinv, r); lp[~inbox] = -np.inf
        return lp
    p0 = rng.uniform(lo, hi, size=(nwalk, Xs.shape[1]))
    sampler = emcee.EnsembleSampler(nwalk, Xs.shape[1], lnprob, vectorize=True)
    sampler.run_mcmc(p0, nstep, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True) * sd + mu
    prior_sd = (X.max(0) - X.min(0)) / np.sqrt(12); ratio = chain.std(0) / prior_sd
    np.savez(KS / f"ksz_posterior_cap_{tag}.npz", chain=chain, params=np.array(PARAMS),
             ratio=ratio, n_cuts=nb, cut_R2=np.array(emu_var))
    print(f"[{tag}] wrote; most-constrained: " +
          ", ".join(f"{PARAMS[j]}={ratio[j]:.2f}" for j in np.argsort(ratio)[:4]))


if __name__ == "__main__":
    posterior(CUTS_BGS[3:5], "bgs2")          # the current 2-bin scope (apples-to-apples)
    posterior(CUTS_BGS,      "bgsall")        # all 5 BGS cuts
    posterior(CUTS_BGS + CUTS_ELG, "all")     # 5 BGS + 3 ELG
