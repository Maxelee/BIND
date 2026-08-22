"""r5_inflation.py -- TASK 3: how much CV R^2 could lambda noise explain?

(a) GEOMETRY.  Only the z=0.0337 lens planes -- the ones whose halos lambda
    is measured from -- can share generative draws with the statistic.  Their
    fractional contribution to C_ell^kappa at z_s=1 is computed in the Limber
    approximation on the ACTUAL lux plane geometry read out of
    bind_lightcone_tng/lensplanes/config.dat (80 planes, 51.25 Mpc/h thick,
    plane scale factors stored in the file).

(b) ANALYTIC attenuation / absorption algebra, with the noise share q_i from
    r5_bootstrap.npz and r5_replicas.npz.

(c) MONTE CARLO.  Jitter lambda by its measured noise -- FRESH noise,
    uncorrelated with the statistics by construction -- and re-run the exact
    shipped CV.  The degradation distribution says how much of CV R^2 is
    attributable to lambda precision AT ALL (it bounds the attenuation leg;
    it deliberately cannot see the absorption leg, which (b)+r5_replicas do).

Outputs -> /mnt/home/mlee1/ceph/referee_work/r5/r5_inflation.npz
"""
import struct
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from r5_common import (CEPH, OUT, STATS, cv_e2e, load_cube,  # noqa: E402
                       load_model_pieces, report_folds, sobol_LAT, LAT_NAMES)

# ── (a) lens-plane geometry + Limber weights ──────────────────────────────
raw = open(CEPH / "bind_lightcone_tng/lensplanes/config.dat", "rb").read()
NPL, NSNAP = struct.unpack("<ii", raw[:8])
a = np.frombuffer(raw[8:8 + 8 * 359], dtype="<f8")
a_pl = a[1:NPL + 1]                       # scale factor of each of the 80 planes
chi_pl = a[NPL + 2:2 * NPL + 2]           # comoving distance [Mpc/h] of each
dchi = float(a[2 * NPL + 2] - a[NPL + 2 + NPL - 1] + 0)  # placeholder
dchi = float(np.median(np.diff(chi_pl)))
z_pl = 1.0 / a_pl - 1.0
print(f"lux lens planes: {NPL} planes from {NSNAP} snapshots, "
      f"dchi = {dchi:.3f} Mpc/h")
print(f"  plane 1-4 (the snap_096 slabs): chi = "
      f"{np.round(chi_pl[:4], 2)} Mpc/h, z = {np.round(z_pl[:4], 4)}")
N096 = NSNAP and NPL // NSNAP             # 4 slabs per snapshot
print(f"  slabs per snapshot = {N096}; snap_096 covers chi < "
      f"{chi_pl[N096-1] + dchi/2:.1f} Mpc/h, z < {z_pl[N096-1]:.4f}")

import pyccl as ccl                                                # noqa: E402
H0H = 0.6774
cos = ccl.Cosmology(Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=H0H,
                    n_s=0.9667, sigma8=0.8159,
                    matter_power_spectrum="halofit")
ZS = 1.0
chi_s = ccl.comoving_radial_distance(cos, 1 / (1 + ZS)) * H0H       # Mpc/h
print(f"  chi(z_s=1) = {chi_s:.1f} Mpc/h  ({np.sum(chi_pl < chi_s)} planes "
      "in front of the source)")

ELLB = np.asarray(np.load("/mnt/home/mlee1/BIND/papers/01_pipeline/"
                          "latent_model_coeffs.npz")["ell"], float)
use = chi_pl < chi_s
W = np.where(use, (1 + z_pl) * chi_pl * (chi_s - chi_pl) / chi_s, 0.0)
F096 = np.empty(len(ELLB))
for j, L in enumerate(ELLB):
    k = (L + 0.5) / np.maximum(chi_pl, 1e-3)                        # h/Mpc
    Pk = np.array([ccl.nonlin_matter_power(cos, kk * H0H, aa) * H0H ** 3
                   if u else 0.0 for kk, aa, u in zip(k, a_pl, use)])
    w = np.where(use, (W / np.maximum(chi_pl, 1e-3)) ** 2 * Pk * dchi, 0.0)
    F096[j] = w[:N096].sum() / w.sum()
print("\nfractional contribution of the z=0.034 planes to C_ell^kappa "
      "(z_s=1), per S(ell) band:")
for j in range(0, len(ELLB), 4):
    print("   " + "  ".join(f"l={ELLB[i]:7.0f}: {100*F096[i]:6.3f}%"
                            for i in range(j, min(j + 4, len(ELLB)))))
print(f"  band mean {100*F096.mean():.3f}%, max {100*F096.max():.3f}% "
      f"(at ell={ELLB[np.argmax(F096)]:.0f})")

# ── (b) analytic algebra ──────────────────────────────────────────────────
boot = np.load(OUT / "r5_bootstrap.npz")
rep = np.load(OUT / "r5_replicas.npz")
qA, qB, q_rep = boot["qA"], boot["qB"], rep["q_rep"]
print("\nlatent noise share q_i = Var(noise)/Var(across design):")
for i, n in enumerate(LAT_NAMES):
    print(f"  {n:>20s}  bootstrap(indep) {qA[i]:.4f}  bootstrap(node-spec) "
          f"{qB[i]:.4f}  3-replica {q_rep[i]:.4f}")
print(f"  mean: {qA.mean():.4f} / {qB.mean():.4f} / {q_rep.mean():.4f}")

# ── (c) Monte-Carlo jitter of lambda ──────────────────────────────────────
bun, amp, dsn, BAS, MEAN, AMPS, D, Y = load_model_pieces()
SOB, FID, rows, _ = load_cube()
LAT = sobol_LAT(SOB)
folds = report_folds(len(LAT), 1)
r2_0 = cv_e2e(LAT, AMPS, BAS, D, folds)
print("\nbaseline CV R^2: " + "  ".join(f"{s}:{r2_0[s]:.4f}" for s in STATS))

COV_A = boot["COV_A"]                                 # (256, 8, 8) per run
Lch = np.linalg.cholesky(COV_A + 1e-18 * np.eye(8))
SIG_REP = rep["SIG_REP"]
NMC = 300
rng = np.random.default_rng(5)
MC = {}
for tag, draw in (
        ("boot_cov", lambda g: np.einsum("qij,qj->qi", Lch,
                                         g.standard_normal((len(LAT), 8)))),
        ("boot_2x", lambda g: 2 * np.einsum("qij,qj->qi", Lch,
                                            g.standard_normal((len(LAT), 8)))),
        ("replica", lambda g: g.standard_normal((len(LAT), 8)) * SIG_REP)):
    arr = np.empty((NMC, len(STATS)))
    for m in range(NMC):
        r2m = cv_e2e(LAT + draw(rng), AMPS, BAS, D, folds)
        arr[m] = [r2m[s] for s in STATS]
    MC[tag] = arr
    print(f"\njitter = {tag}:  DEGRADATION Delta R^2 = R^2(lambda) - "
          "R^2(lambda + noise)")
    for i, s in enumerate(STATS):
        d = r2_0[s] - arr[:, i]
        print(f"  {s:>4s}: mean {d.mean():+.5f}  [16,84]% "
              f"[{np.percentile(d,16):+.5f}, {np.percentile(d,84):+.5f}]  "
              f"max {d.max():+.5f}")

np.savez(OUT / "r5_inflation.npz", ell_band=ELLB, F096=F096,
         chi_pl=chi_pl, a_pl=a_pl, z_pl=z_pl, chi_s=chi_s, dchi=dchi,
         r2_baseline=np.array([r2_0[s] for s in STATS]),
         stats=np.array(STATS), qA=qA, qB=qB, q_rep=q_rep,
         **{f"mc_{k}": v for k, v in MC.items()})
print(f"\nwrote {OUT/'r5_inflation.npz'}")
