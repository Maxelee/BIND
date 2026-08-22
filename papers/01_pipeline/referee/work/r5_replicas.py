"""r5_replicas.py -- the direct measurement the R5 bound needs.

The 60-run twobound suite contains THREE runs (18, 49, 53) whose 35-parameter
vectors are bit-identical: for three of the 30 scanned parameters one of the
two bounds happens to coincide with the fiducial value, so those three runs
are the SAME simulation painted THREE INDEPENDENT TIMES.  They share the
halo catalogue (the same 2933 snap_096 halos) and the ray-tracing seed
(RT_SEED=1992), and the flow-matching sampler is unseeded
(bind/model.py::FlowMatching.sample draws torch.randn with no generator),
so the only thing that differs between them is the generative noise
realization -- exactly the quantity referee comment 5 is about.

That gives, WITHOUT any new painting:
  * sigma_rep(lambda) -- the node-specific latent noise, measured (not
    bootstrapped);
  * sigma_rep(D)      -- the node-specific realization noise of each of the
    7 statistics, measured; Var(D_noise)/Var(D_design) is the hard cap on
    how much explained variance ANY predictor can absorb from this channel;
  * Cov(D_noise, Dhat_noise) -- the referee's mechanism itself: how much the
    model's prediction MOVES WITH the statistic when only the noise moves.

Outputs -> /mnt/home/mlee1/ceph/referee_work/r5/r5_replicas.npz
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from r5_common import (CEPH, FIELD_KEYS, LAT_NAMES, OUT, SB35,  # noqa: E402
                       STATS, cv_amps, load_cube, load_model_pieces,
                       measure_lambda, report_folds, sobol_LAT)

TB = CEPH / "bind_science/runs/twobound"
REP = [18, 49, 53]
ZI = 1

bun, amp, dsn, BAS, MEAN, AMPS, D, Y = load_model_pieces()
coef = np.load("/mnt/home/mlee1/BIND/papers/01_pipeline/latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
NBc = len(coef["ell"])

tbp = np.load(TB / "twobound_params.npy")
assert np.array_equal(tbp[REP[0]], tbp[REP[1]]) and \
    np.array_equal(tbp[REP[0]], tbp[REP[2]]), "runs 18/49/53 are not identical"
print(f"replica runs {REP}: 35-parameter vectors bit-identical  [verified]")

P = {r: np.load(TB / f"run_{r:04d}/paired_stats.npz") for r in REP}
NG = {r: np.load(TB / f"run_{r:04d}/nongaussian_stats.npz") for r in REP}
# NB the MF leg uses the ORIGINAL nongaussian_stats grid + interpolation onto
# the canonical centers, i.e. the convention the SHIPPED bundle was built with
# (its v0/v1/v2 bases are 14 bins wide, the NMF=14 path); the 22-bin nu05
# remeasurement postdates the bundle and is not used here.
mf_nu_tb = NG[REP[0]]["mf_nu"]
ell = P[REP[0]]["ell"]
NUg = np.asarray(dsn["a__pdf__pdf_bins"], float)
nu_tb = P[REP[0]]["nu"]
AGG = [(8 + 2 * j, 9 + 2 * j) for j in range(22)]


def bandS(r):
    y = P[r]["clk_resp"][ZI]
    return np.stack([np.nanmean(y[(ell >= EDGb[i]) & (ell < EDGb[i + 1])])
                     for i in range(NBc)])


def pdf_nu(r):
    kb, p = NG[r]["pdf_bins"], NG[r]["pdf"][ZI]
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * NUg, kb, p, left=0.0, right=0.0)


def counts_canon(r, key):
    c = np.nanmean(P[r][key][:, ZI, :], axis=0)
    return np.array([c[i] + c[j] for i, j in AGG])


def curve(r, st):
    if st == "clk":
        v = bandS(r)
    elif st == "pdf":
        v = pdf_nu(r)
    elif st in ("pk", "mn"):
        v = counts_canon(r, "pk_real" if st == "pk" else "min_real")
    else:
        nb = BAS[st].shape[1]
        return np.interp(NUg[:nb], mf_nu_tb,
                         np.asarray(NG[r][st.upper()], float)[ZI])
    return v[:BAS[st].shape[1]]


# ── replica latents ───────────────────────────────────────────────────────
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
TBF = {k: np.asarray(cz[f"tb_{k}"], float) for k in FIELD_KEYS}
LAM_REP = np.stack([measure_lambda({k: TBF[k][r] for k in FIELD_KEYS})
                    for r in REP])
SOB, FID, rows, _ = load_cube()
LAT = sobol_LAT(SOB)
SIG_DES = LAT.std(0)
boot = np.load(OUT / "r5_bootstrap.npz")
SIG_A = np.median(boot["SIG_A"], 0)
SIG_B = np.median(boot["SIG_B"], 0)
# unbiased sigma from 3 replicas (ddof=1)
SIG_REP = LAM_REP.std(0, ddof=1)
print("\nNODE-SPECIFIC LATENT NOISE, measured on 3 independent paints of the "
      "SAME simulation")
print(f"{'latent':>20s} {'rep1':>11s} {'rep2':>11s} {'rep3':>11s} "
      f"{'sig_rep':>10s} {'sig_boot':>9s} {'sig_des':>9s} {'q_rep':>8s}")
q_rep = (SIG_REP / SIG_DES) ** 2
for i, n in enumerate(LAT_NAMES):
    print(f"{n:>20s} {LAM_REP[0,i]:11.5f} {LAM_REP[1,i]:11.5f} "
          f"{LAM_REP[2,i]:11.5f} {SIG_REP[i]:10.4g} {SIG_A[i]:9.4g} "
          f"{SIG_DES[i]:9.4g} {q_rep[i]:8.4f}")
print(f"  mean q_rep = {q_rep.mean():.4f}   max = {q_rep.max():.4f}   "
      f"(bootstrap qA {boot['qA'].mean():.4f}, qB {boot['qB'].mean():.4f})")
print("  sigma_rep / sigma_boot(A) per latent: "
      + " ".join(f"{v:.2f}" for v in SIG_REP / SIG_A))

# ── replica statistics + the absorbed-variance cap ────────────────────────
print("\nNODE-SPECIFIC STATISTIC NOISE and the absorbed-variance cap")
print(f"{'stat':>5s} {'rms(D_noise)':>13s} {'rms(D_des)':>11s} "
      f"{'f_noise':>10s} {'rms(Dhat_n)':>12s} {'corr':>7s} "
      f"{'2Cov/SStot':>11s} {'cap':>9s}")
res = {}
folds = report_folds(len(LAT), 1)
for st in STATS:
    C = np.stack([curve(r, st) for r in REP])          # (3, nbin)
    Dn = C - C.mean(0)                                  # replica deviations
    # model prediction from each replica's OWN measured lambda, using the
    # SHIPPED full-design map (fit on the 256 Sobol runs -- these three runs
    # are not in it)
    Lc = LAM_REP - bun[f"{st}__lat_ref"]
    Ahat = np.c_[Lc, np.ones(3)] @ bun[f"{st}__lat_M"]
    Chat = bun[f"{st}__mean"] + Ahat @ BAS[st]
    Dhn = Chat - Chat.mean(0)
    ddof = 3.0 / 2.0                                    # ddof=1 on 3 samples
    var_Dn = float((Dn ** 2).sum() * ddof / Dn.size)
    var_Dhn = float((Dhn ** 2).sum() * ddof / Dhn.size)
    var_des = float((D[st] ** 2).mean())
    cov = float((Dn * Dhn).sum() * ddof / Dn.size)
    corr = cov / np.sqrt(var_Dn * var_Dhn) if var_Dhn > 0 else np.nan
    f_noise = var_Dn / var_des
    inflate = 2 * cov / var_des                        # the actual mechanism
    cap = 2 * np.sqrt(var_Dn * var_Dhn) / var_des      # Cauchy-Schwarz cap
    res[st] = dict(var_Dn=var_Dn, var_Dhn=var_Dhn, var_des=var_des,
                   cov=cov, corr=corr, f_noise=f_noise, inflate=inflate,
                   cap=cap)
    print(f"{st:>5s} {np.sqrt(var_Dn):13.5g} {np.sqrt(var_des):11.5g} "
          f"{f_noise:10.5f} {np.sqrt(var_Dhn):12.5g} {corr:7.3f} "
          f"{inflate:+11.5f} {cap:9.5f}")

print("\nreading of the table:")
print("  f_noise      = Var(node-specific realization noise) / Var(design)")
print("               = the HARD cap on absorbable explained variance,")
print("                 attainable only if lambda's noise reproduced the")
print("                 statistic's noise perfectly.")
print("  2Cov/SStot   = the measured R^2 inflation of this channel: how much")
print("                 the CV residual shrinks because the prediction moves")
print("                 WITH the statistic when only the draw changes.")
print("  cap          = |2Cov| ceiling at corr = 1.")

np.savez(OUT / "r5_replicas.npz", REP=np.array(REP), LAM_REP=LAM_REP,
         SIG_REP=SIG_REP, SIG_DES=SIG_DES, q_rep=q_rep,
         lat_names=np.array(LAT_NAMES), stats=np.array(STATS),
         **{f"{st}__{k}": v for st in STATS for k, v in res[st].items()})
print(f"\nwrote {OUT/'r5_replicas.npz'}")
