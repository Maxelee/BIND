"""r5_bootstrap.py -- TASK 2: the lambda noise floor.

Three independent handles on how noisy each measured latent is:

  (A) INDEPENDENT within-bin halo bootstrap.  For every Sobol node (and the
      fiducial), resample halos with replacement WITHIN each latent's own
      mass bin and re-take the median.  sigma_A is an UPPER bound on the
      realization noise the referee worries about: it contains (i) per-halo
      generative draw noise (node-specific -- the thing at issue) AND (ii)
      halo population-sampling noise, which is COMMON-MODE across nodes
      because all 256 runs paint the SAME 2933 halos and therefore cannot be
      absorbed by a regression across the design.

  (B) COMMON-DRAW bootstrap.  One resampled halo multiset per replicate,
      applied to all 256 nodes at once; the node-specific residual
      lambda_b(q) - <lambda_b>_q removes the part of the fluctuation that
      every node shares.  Tighter, still conservative (it retains the
      parameter-dependent modulation of the physical halo-to-halo scatter,
      which is signal, not noise).

  (C) BIND-vs-TRUTH per-halo residual at the fiducial.  The truth atlas
      holds the hydro-pasted TRUTH patches for the SAME 2933 halos, so
      x_h^BIND / x_h^truth is a DIRECT measurement of per-halo generative
      error (draw noise + model bias) with no bootstrap assumption.

Outputs -> /mnt/home/mlee1/ceph/referee_work/r5/r5_bootstrap.npz
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from r5_common import (FIELD_KEYS, LAT_BINS, LAT_NAMES, LAT_SPEC,  # noqa: E402
                       OB_OM, OUT, load_cube, load_fid_atlas, load_truth,
                       measure_lambda, per_halo_quantities, sobol_LAT)

NB = 1000
rng = np.random.default_rng(20260812)
t0 = time.time()

SOB, FID, rows, _ = load_cube()
NRUN = len(rows)
NH = len(SOB["m_tot_500c_bg"][0])
LAT = sobol_LAT(SOB)
print(f"{NRUN} Sobol runs x {NH} halos; lambda shape {LAT.shape}; NB={NB}")

QS, LMS = [], []
for q in range(NRUN):
    Q, lm = per_halo_quantities({k: SOB[k][q] for k in FIELD_KEYS})
    QS.append(Q)
    LMS.append(lm)

BINCNT = np.zeros((NRUN, len(LAT_BINS)), int)
for q in range(NRUN):
    for b, (lo, hi) in enumerate(LAT_BINS.values()):
        BINCNT[q, b] = int(((LMS[q] >= lo) & (LMS[q] < hi)).sum())
print("halos per latent mass bin (min / median / max over the 256 runs):")
for b, k in enumerate(LAT_BINS):
    c = BINCNT[:, b]
    print(f"  {k}: {c.min()} / {int(np.median(c))} / {c.max()}")


def _prep(v, pos, log):
    v = np.asarray(v, float)
    if pos:
        v = np.where(v > 0, v, np.nan)
    return v


def boot_within_bin(Q, lm, nb, gen):
    """(nb, 8) bootstrap replicates, halos resampled WITHIN each bin."""
    out = np.empty((nb, 8))
    for c, (bk, qk, pos, log) in enumerate(LAT_SPEC):
        lo, hi = LAT_BINS[bk]
        v = _prep(Q[qk][(lm >= lo) & (lm < hi)], pos, log)
        n = len(v)
        m = np.nanmedian(v[gen.integers(0, n, (nb, n))], axis=1)
        out[:, c] = np.log10(m) if log else m
    out[:, 0] /= OB_OM
    return out


def boot_common(Q, lm, IDX):
    """(nb, 8) replicates from a SHARED resampled multiset IDX (nb, NH)."""
    nb = IDX.shape[0]
    lmb = lm[IDX]
    out = np.empty((nb, 8))
    for c, (bk, qk, pos, log) in enumerate(LAT_SPEC):
        lo, hi = LAT_BINS[bk]
        v = _prep(Q[qk], pos, log)[IDX]
        v = np.where((lmb >= lo) & (lmb < hi), v, np.nan)
        m = np.nanmedian(v, axis=1)
        out[:, c] = np.log10(m) if log else m
    out[:, 0] /= OB_OM
    return out


# ── (A) independent within-bin bootstrap ──────────────────────────────────
SIG_A = np.empty((NRUN, 8))
COV_A = np.empty((NRUN, 8, 8))
for q in range(NRUN):
    LB = boot_within_bin(QS[q], LMS[q], NB, rng)
    SIG_A[q] = LB.std(0)
    COV_A[q] = np.cov(LB.T)
print(f"(A) independent bootstrap done ({time.time()-t0:.0f}s)")

# ── (B) common-draw bootstrap ─────────────────────────────────────────────
NBC = 400                                    # (B) is ~7x costlier per rep
IDX = rng.integers(0, NH, (NBC, NH))
LB_C = np.empty((NBC, NRUN, 8))
for q in range(NRUN):
    LB_C[:, q] = boot_common(QS[q], LMS[q], IDX)
    if (q + 1) % 64 == 0:
        print(f"  common bootstrap run {q+1}/{NRUN}  ({time.time()-t0:.0f}s)")
RES_C = LB_C - LB_C.mean(1, keepdims=True)
SIG_B = RES_C.std(0)
COV_B = np.stack([np.cov(RES_C[:, q].T) for q in range(NRUN)])
# how much of the common-draw fluctuation is common-mode:
SIG_Ball = LB_C.std(0)
print(f"(B) common-draw bootstrap done ({time.time()-t0:.0f}s)")

SIG_DES = LAT.std(0)
qA = (np.median(SIG_A, 0) / SIG_DES) ** 2
qB = (np.median(SIG_B, 0) / SIG_DES) ** 2
print("\nLATENT NOISE BUDGET (medians over the 256 nodes)")
print(f"{'latent':>20s} {'sig_design':>11s} {'sigA':>10s} {'sigB':>10s} "
      f"{'qA':>9s} {'qB':>9s}")
for i, n in enumerate(LAT_NAMES):
    print(f"{n:>20s} {SIG_DES[i]:11.4g} {np.median(SIG_A[:, i]):10.4g} "
          f"{np.median(SIG_B[:, i]):10.4g} {qA[i]:9.4f} {qB[i]:9.4f}")
print(f"  mean q over the 8 latents: qA {qA.mean():.4f}   qB {qB.mean():.4f}")
print(f"  max  q over the 8 latents: qA {qA.max():.4f}   qB {qB.max():.4f}")

# ── (C) BIND-vs-TRUTH at the fiducial ─────────────────────────────────────
FIDA, TRU = load_fid_atlas(), load_truth()
Qf, lmf = per_halo_quantities(FIDA)
Qt, lmt = per_halo_quantities(TRU)
lam_fid, lam_tru = measure_lambda(FIDA), measure_lambda(TRU)
LBF = boot_within_bin(Qf, lmf, NB, rng)
sigA_fid = LBF.std(0)
dlt = lam_fid - lam_tru
print("\nFIDUCIAL lambda: BIND vs TRUTH atlas (same 2933 halos)")
print(f"{'latent':>20s} {'BIND':>12s} {'TRUTH':>12s} {'Delta':>11s} "
      f"{'/sigA':>7s} {'/sig_des':>9s}")
for i, n in enumerate(LAT_NAMES):
    print(f"{n:>20s} {lam_fid[i]:12.5f} {lam_tru[i]:12.5f} {dlt[i]:11.4g} "
          f"{dlt[i]/sigA_fid[i]:7.2f} {dlt[i]/SIG_DES[i]:9.3f}")

print("\nper-halo BIND/TRUTH ratio at the fiducial (median, 16-84 halfwidth):")
PH = {}
for qk in ("f_star", "T", "Y", "Pe", "c_gas", "Y_ss"):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = Qf[qk] / Qt[qk]
    g = r[np.isfinite(r) & (r > 0)]
    lo, hi = np.percentile(g, [16, 84])
    PH[qk] = (float(np.median(g)), float(0.5 * (hi - lo)), len(g))
    print(f"  {qk:>7s}: median {PH[qk][0]:.4f}  halfwidth {PH[qk][1]:.4f}"
          f"  N={PH[qk][2]}")

OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT / "r5_bootstrap.npz", LAT=LAT, lat_names=np.array(LAT_NAMES),
    SIG_A=SIG_A, SIG_B=SIG_B, SIG_Ball=SIG_Ball, COV_A=COV_A, COV_B=COV_B,
    SIG_DES=SIG_DES, qA=qA, qB=qB, BINCNT=BINCNT,
    bin_names=np.array(list(LAT_BINS)), lam_fid=lam_fid, lam_truth=lam_tru,
    sigA_fid=sigA_fid, NB=NB, NBC=NBC,
    ph_ratio=np.array([PH[k] for k in PH]),
    ph_names=np.array(list(PH)))
print(f"\nwrote {OUT/'r5_bootstrap.npz'}  ({time.time()-t0:.0f}s)")
