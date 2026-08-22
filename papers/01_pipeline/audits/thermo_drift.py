"""
Investigation 1(a),(b): per-snapshot BIND/truth ratio of T_mw_500c and Pe_mw_500c
from the cached halo_atlas npz's, + mass-dependence split.
"""
import json
import numpy as np
from pathlib import Path

OUT = Path("/mnt/home/mlee1/ceph/bind_science/halo_atlas")
LC = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")

PH_SNAPS = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

def redshift(snap):
    mf = LC / f"snap_{snap:03d}" / "stage1" / "stage1_manifest.json"
    if mf.exists():
        return float(json.loads(mf.read_text())["redshift"])
    return np.nan

rows = []
for snap in PH_SNAPS:
    ff = OUT / f"fid_snap{snap:03d}.npz"
    tf = OUT / f"truth_snap{snap:03d}.npz"
    if not (ff.exists() and tf.exists()):
        continue
    fid = np.load(ff)
    tru = np.load(tf)
    if "T_mw_500c" not in fid.files or "T_mw_500c" not in tru.files:
        continue
    z = redshift(snap)
    a = 1.0 / (1.0 + z)

    Mf, Tf, Pf, Yf, gf = fid["M_fof"], fid["T_mw_500c"], fid["Pe_mw_500c"], fid["Y_500c"], fid["m_gas_500c"]
    Mt, Tt, Pt, Yt, gt = tru["M_fof"], tru["T_mw_500c"], tru["Pe_mw_500c"], tru["Y_500c"], tru["m_gas_500c"]
    Kf, Kt = fid["K_mw_500c"], tru["K_mw_500c"]

    # halos are the SAME set / same row index across fid vs truth (per memory note)
    n = min(len(Mf), len(Mt))
    if len(Mf) != len(Mt):
        print(f"  [warn] snap{snap}: n_fid={len(Mf)} n_truth={len(Mt)} (mismatched length!)")

    ok = (Tf[:n] > 0) & (Tt[:n] > 0) & (Pf[:n] > 0) & (Pt[:n] > 0) & (gf[:n] > 0) & (gt[:n] > 0) \
        & (Kf[:n] > 0) & (Kt[:n] > 0)
    if ok.sum() < 5:
        continue

    T_ratio = np.median(Tf[:n][ok] / Tt[:n][ok])
    Pe_ratio = np.median(Pf[:n][ok] / Pt[:n][ok])
    K_ratio = np.median(Kf[:n][ok] / Kt[:n][ok])
    Y_ratio = np.median(Yf[:n][ok] / Yt[:n][ok]) if (Yf[:n][ok] > 0).all() and (Yt[:n][ok] > 0).all() else np.nan
    fgas_ratio = np.median((gf[:n][ok] / Mf[:n][ok]) / (gt[:n][ok] / Mt[:n][ok]))

    # mass split: use FoF DMO mass, same both sides
    Msel = Mf[:n][ok]
    lo_mass = Msel < np.median(Msel)
    hi_mass = ~lo_mass

    T_ratio_lo = np.median((Tf[:n][ok] / Tt[:n][ok])[lo_mass])
    T_ratio_hi = np.median((Tf[:n][ok] / Tt[:n][ok])[hi_mass])
    Pe_ratio_lo = np.median((Pf[:n][ok] / Pt[:n][ok])[lo_mass])
    Pe_ratio_hi = np.median((Pf[:n][ok] / Pt[:n][ok])[hi_mass])

    rows.append(dict(snap=snap, z=z, a=a, n=int(ok.sum()),
                      T_ratio=T_ratio, Pe_ratio=Pe_ratio, K_ratio=K_ratio, Y_ratio=Y_ratio, fgas_ratio=fgas_ratio,
                      T_ratio_lo=T_ratio_lo, T_ratio_hi=T_ratio_hi,
                      Pe_ratio_lo=Pe_ratio_lo, Pe_ratio_hi=Pe_ratio_hi,
                      M_med_lo=np.median(Msel[lo_mass]), M_med_hi=np.median(Msel[hi_mass])))

rows.sort(key=lambda r: r["z"])

print(f"{'snap':>5s} {'z':>6s} {'a':>6s} {'n':>5s} {'fgas_r':>8s} {'Y_r':>8s} {'T_r':>8s} {'Pe_r':>8s} {'K_r':>8s} "
      f"{'T_lo':>7s} {'T_hi':>7s} {'Pe_lo':>7s} {'Pe_hi':>7s}")
for r in rows:
    print(f"{r['snap']:5d} {r['z']:6.3f} {r['a']:6.3f} {r['n']:5d} "
          f"{r['fgas_ratio']:8.4f} {r['Y_ratio']:8.4f} {r['T_ratio']:8.4f} {r['Pe_ratio']:8.4f} {r['K_ratio']:8.4f} "
          f"{r['T_ratio_lo']:7.4f} {r['T_ratio_hi']:7.4f} {r['Pe_ratio_lo']:7.4f} {r['Pe_ratio_hi']:7.4f}")

# power-law fit: ratio ~ a^alpha  =>  log(ratio) = alpha*log(a) + const
a_arr = np.array([r["a"] for r in rows])
loga = np.log(a_arr)

def fit_slope(key):
    y = np.log(np.array([r[key] for r in rows]))
    ok = np.isfinite(y)
    A = np.vstack([loga[ok], np.ones(ok.sum())]).T
    slope, intercept = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return slope, intercept

print("\nPower-law fits: ratio ~ a^alpha (fit over all 20 snaps, log-log)")
for key in ["fgas_ratio", "Y_ratio", "T_ratio", "Pe_ratio", "K_ratio", "T_ratio_lo", "T_ratio_hi", "Pe_ratio_lo", "Pe_ratio_hi"]:
    slope, intercept = fit_slope(key)
    print(f"  {key:14s}: alpha = {slope:+.4f}  (ratio(a=1) = exp(intercept) = {np.exp(intercept):.4f})")

np.savez("/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/investigation1/thermo_drift_rows.npz",
         rows=rows)
import pickle
with open("/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/investigation1/thermo_drift_rows.pkl", "wb") as f:
    pickle.dump(rows, f)
