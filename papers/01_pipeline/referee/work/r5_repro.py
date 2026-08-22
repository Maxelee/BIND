"""r5_repro.py -- TASK 1: reproduce the shipped end-to-end CV R^2.

Recomputes lambda from the per-halo atlas cube with the shipped reduction,
asserts it against nothing but the shipped numbers, and runs the shipped
rng(1)-fold end-to-end CV.  Also reproduces the theta-route baseline (the
"0.6" half of the flagship 0.96-vs-0.6 comparison), both linear and with a
gradient-boosted / GP alternative for the record.

    /mnt/home/mlee1/venvs/BIND_env/bin/python3 referee/work/r5_repro.py
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from r5_common import (STATS, OUT, cv_amps, cv_e2e, load_cube,  # noqa: E402
                       load_model_pieces, report_folds, sobol_LAT, LAT_NAMES)

SHIPPED = {"clk": 0.9632, "pdf": 0.9386, "pk": 0.7872, "mn": 0.8379,
           "v0": 0.9599, "v1": 0.9586, "v2": 0.9643}

bun, amp, dsn, BAS, MEAN, AMPS, D, Y = load_model_pieces()
# (a) bit-faithful path: the shipped script never casts the float32 cube
SOB32, _, _, _ = load_cube(np.float32)
LAT32 = sobol_LAT(SOB32)
d32 = np.max(np.abs(LAT32.mean(0) - bun["clk__lat_ref"]))
print(f"float32-faithful path: max |Delta lat_ref| = {d32:.3e} "
      "(float32 round-off only)")
assert d32 < 1e-6, "float32 path does not reproduce the shipped lat_ref"
# (b) the float64 path used for everything downstream
SOB, FID, rows, _ = load_cube()
LAT = sobol_LAT(SOB)
assert np.isfinite(LAT).all()
print(f"lambda recomputed from {LAT.shape[0]} Sobol rows x {LAT.shape[1]} latents")
print("  bundle lat_ref :", np.round(bun["clk__lat_ref"], 6))
print("  recomputed mean:", np.round(LAT.mean(0), 6))
dref = np.max(np.abs(LAT.mean(0) - bun["clk__lat_ref"]))
print(f"  float64 path: max |Delta lat_ref| = {dref:.3e} (float32 rounding)")
assert dref < 1e-5

# also check the stored bundle map reproduces: full-sample W vs bundle lat_M
for st in STATS:
    Lc = LAT32 - LAT32.mean(0)
    W, *_ = np.linalg.lstsq(np.c_[Lc, np.ones(len(Lc))], AMPS[st], rcond=None)
    M0 = np.asarray(bun[f"{st}__lat_M"], float)
    dm = np.max(np.abs(W - M0)) / np.max(np.abs(M0))
    assert dm < 1e-5, f"{st}: lat_M mismatch {dm:.3e}"
print("  full-sample lambda->a map reproduces bundle lat_M for all 7 stats "
      f"(max rel. dev. {dm:.1e})")

folds = report_folds(len(LAT), 1)
r2_32 = cv_e2e(LAT32, AMPS, BAS, D, folds)
r2 = cv_e2e(LAT, AMPS, BAS, D, folds)
print("  float32-vs-float64 latent path, max |Delta CV R^2| = "
      f"{max(abs(r2[s] - r2_32[s]) for s in STATS):.2e}")
print("\nEND-TO-END CV R^2 (rng(1) folds), recomputed vs shipped:")
worst = 0.0
for st in STATS:
    d = abs(r2[st] - SHIPPED[st])
    worst = max(worst, d)
    print(f"  {st:>4s}: {r2[st]:.4f}   shipped {SHIPPED[st]:.4f}   d={d:.2e}")
pooled = float(np.mean([r2[s] for s in STATS]))
print(f"  pooled: {pooled:.4f}   shipped {np.mean(list(SHIPPED.values())):.4f}")
assert worst < 5e-4, f"reproduction failed, worst |delta| = {worst:.3e}"
print(f"REPRODUCED (worst |delta| = {worst:.2e} < 5e-4)")

# ── the theta route (the "0.6" baseline of the flagship comparison) ───────
# canonical design matrix = the dataset's X_unit, the 30 varying astro
# parameters on the SB35 unit cube (log-flagged parameters therefore enter
# in log space, which is what makes the linear theta route reach ~0.58 --
# the raw physical-unit columns only reach ~0.46, reported for the record).
Xu = np.asarray(dsn["X_unit"], float)
th_names = [str(s) for s in dsn["param_names"]]
THz = (Xu - Xu.mean(0)) / Xu.std(0)
TH = np.asarray(amp["theta"], float)
keep = TH.std(0) > 1e-10
TH30 = TH[:, keep]
r2th_raw = cv_e2e((TH30 - TH30.mean(0)) / TH30.std(0), AMPS, BAS, D, folds)
print(f"\ntheta route: X_unit {Xu.shape}, {len(th_names)} astro parameters")
r2th = cv_e2e(THz, AMPS, BAS, D, folds)
print("theta -> a, LINEAR, same rng(1) folds:")
for st in STATS:
    print(f"  {st:>4s}: {r2th[st]:.4f}  (raw units {r2th_raw[st]:.4f})"
          f"   [lambda: {r2[st]:.4f}]")
print(f"  pooled: {np.mean([r2th[s] for s in STATS]):.4f} "
      f"(raw units {np.mean([r2th_raw[s] for s in STATS]):.4f})")

# nonlinear theta routes, for the record (paper claims 'no further than ~0.6')
r2_nl = {}
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    def cv_nl(make):
        out = {}
        for st in STATS:
            A = AMPS[st]
            pred = np.empty_like(A)
            for tr, te in folds:
                for k in range(A.shape[1]):
                    m = make()
                    m.fit(THz[tr], A[tr, k])
                    pred[te, k] = m.predict(THz[te])
            Dp = pred @ BAS[st]
            out[st] = float(1 - ((D[st] - Dp) ** 2).sum() / (D[st] ** 2).sum())
        return out

    r2_nl["GBT"] = cv_nl(lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.07, max_depth=3, random_state=0))
    # isotropic-length-scale GP (the anisotropic 30-length-scale variant is
    # ~100x slower on this 1-CPU cgroup and was measured to land in the same
    # place by the paper's own scan)
    r2_nl["GP"] = cv_nl(lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(5.0) + WhiteKernel(1e-2),
        normalize_y=True, alpha=1e-8, n_restarts_optimizer=0,
        random_state=0))
    for tag, rr in r2_nl.items():
        print(f"theta -> a, {tag}: " + "  ".join(f"{s}:{rr[s]:.3f}" for s in STATS)
              + f"   pooled {np.mean(list(rr.values())):.4f}")
except Exception as e:                                        # pragma: no cover
    print(f"  [nonlinear theta routes skipped: {e}]")

OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "r5_repro.npz", LAT=LAT, lat_names=np.array(LAT_NAMES),
         r2_lambda=np.array([r2[s] for s in STATS]),
         r2_theta_linear=np.array([r2th[s] for s in STATS]),
         r2_theta_linear_rawunits=np.array([r2th_raw[s] for s in STATS]),
         **{f"r2_theta_{k}": np.array([v[s] for s in STATS])
            for k, v in r2_nl.items()},
         stats=np.array(STATS), theta30=Xu, theta_names=np.array(th_names))
print(f"\nwrote {OUT / 'r5_repro.npz'}")
