"""fidswap: the fig-11 (z-closure) NUMBERS that are already rebuildable.

The figure itself is NOT re-rendered: panel (d) reads the STACKED profile cache
bind_science/profiles/fid_snap{029..096}.npz, which exists only for the retired
fiducial (the builder, examples/halo_atlas.py, lives on the analysis/sobol-sb35
branch and only snap 096 has been re-reduced for the replica).  Shipping a
figure with three swapped panels and one retired panel would be exactly the
silent fiducial mixing the audit warns about, so this script prints the three
panels that ARE ready -- (a) Cl_kk per source plane, (b) Cl_ky per source plane,
(c) the 20-snapshot halo-level ladder -- plus the REFITTED debias templates for
Y_500c and f_gas, and leaves the y-profile template to the deferred rebuild.
"""
from __future__ import annotations

import numpy as np

from fsfig_common import FID_ATLAS, FID_RUN, OLD_FID, RT, SCI, ZS, tee

tee("fig11_zclosure_numbers")
FS_ATLAS = FID_ATLAS.parent
REPL = FID_ATLAS.name.split("_")[0]          # 'fidtb49'
SNAPS = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52, 56, 59, 63, 67, 71, 76, 80, 85, 90, 96]

# ── panels (a)/(b): cached mean spectra ratios per source plane ──────────────
print("### panel (a)/(b): median BIND/truth - 1 over ell=300-5000, per source plane")
t_cl = np.load(RT / "Cl_kappa.npz")
t_ky = np.load(RT / "Cl_kappa_y.npz")
ell = np.load(FID_RUN / "Cl_kappa.npz")["ell"]
m = (ell >= 300) & (ell <= 5000)
for lab, run in (("OLD", OLD_FID), ("NEW", FID_RUN)):
    b_cl = np.load(run / "Cl_kappa.npz")
    b_ky = np.load(run / "Cl_kappa_y.npz")
    kk = [100 * (np.nanmedian((b_cl["cl"][zi, zi] / np.where(t_cl["cl"][zi, zi] > 0,
                                                             t_cl["cl"][zi, zi], np.nan))[m]) - 1)
          for zi in range(5)]
    ky = [100 * (np.nanmedian((b_ky["cl_ky"][zi] / np.where(np.abs(t_ky["cl_ky"][zi]) > 0,
                                                            t_ky["cl_ky"][zi], np.nan))[m]) - 1)
          for zi in range(5)]
    print(f"  {lab} Cl_kk  " + "  ".join(f"z_s={ZS[i]:.2f}: {kk[i]:+6.2f}%" for i in range(5)))
    print(f"  {lab} Cl_ky  " + "  ".join(f"z_s={ZS[i]:.2f}: {ky[i]:+6.2f}%" for i in range(5)))


# ── panel (c): the 20-snapshot halo-level ladder ─────────────────────────────
def raw_fgas(a):
    tot = a["m_dm_500c"] + a["m_gas_500c"] + a["m_star_500c"]
    return a["m_gas_500c"] / np.where(tot > 0, tot, np.nan)


def ladder(atlas_of):
    rngz = np.random.default_rng(6)
    NBOOT = 500
    z, yr, fr, ye, fe = [], [], [], [], []
    for sn in SNAPS:
        fz = np.load(atlas_of(sn))
        tz = np.load(SCI / f"halo_atlas/truth_snap{sn:03d}.npz")
        ok = (fz["Y_500c"] > 0) & (tz["Y_500c"] > 0)
        rY = fz["Y_500c"][ok] / tz["Y_500c"][ok]
        rF = raw_fgas(fz) / raw_fgas(tz)
        rF = rF[np.isfinite(rF)]
        z.append(float(fz["z"]))
        yr.append(np.median(rY))
        fr.append(np.nanmedian(rF))
        ye.append(np.median(rY[rngz.integers(0, len(rY), (NBOOT, len(rY)))], 1).std())
        fe.append(np.median(rF[rngz.integers(0, len(rF), (NBOOT, len(rF)))], 1).std())
    return (np.array(z), np.array(yr), np.array(fr), np.array(ye), np.array(fe))


OLDL = ladder(lambda sn: SCI / f"halo_atlas/fid_snap{sn:03d}.npz")
NEWL = ladder(lambda sn: FS_ATLAS / f"{REPL}_snap{sn:03d}.npz")

print("\n### panel (c): median BIND/truth per snapshot -- OLD vs NEW")
print(f"{'snap':>5s} {'z':>6s} | {'Y_500c OLD':>11s} {'NEW':>8s} | {'f_gas OLD':>10s} {'NEW':>8s}")
for i, sn in enumerate(SNAPS):
    print(f"{sn:5d} {NEWL[0][i]:6.3f} | {OLDL[1][i]:11.4f} {NEWL[1][i]:8.4f} | "
          f"{OLDL[2][i]:10.4f} {NEWL[2][i]:8.4f}")


def fit_templates(a, r, sig):
    r = np.asarray(r, float)
    sig = np.maximum(np.asarray(sig, float), 1e-4)
    w = 1.0 / sig
    cands = {}
    for nm, A, formula, keys in (
            ("linear_a", np.vstack([np.ones_like(a), a]).T, "r(a) = c0 + c1*a", ("c0", "c1")),
            ("quadratic_a", np.vstack([np.ones_like(a), a, a ** 2]).T,
             "r(a) = c0 + c1*a + c2*a^2", ("c0", "c1", "c2"))):
        Aw = A * w[:, None]
        c, *_ = np.linalg.lstsq(Aw, r * w, rcond=None)
        cerr = np.sqrt(np.diag(np.linalg.inv(Aw.T @ Aw)))
        cands[nm] = dict(pred=A @ c, k=A.shape[1], formula=formula,
                         coef=dict(zip(keys, map(float, c))),
                         coef_err=dict(zip(keys, map(float, cerr))))
    Al = np.vstack([np.ones_like(a), np.log(a)]).T
    wl = r / sig
    cl, *_ = np.linalg.lstsq(Al * wl[:, None], np.log(r) * wl, rcond=None)
    A0 = float(np.exp(cl[0]))
    cands["powerlaw_a"] = dict(pred=A0 * a ** cl[1], k=2, formula="r(a) = A0 * a**alpha",
                               coef={"A0": A0, "alpha": float(cl[1])}, coef_err={})
    for f in cands.values():
        f["chi2"] = float(np.sum(((r - f["pred"]) / sig) ** 2))
        f["aic"] = f["chi2"] + 2 * f["k"]
        f["rms_pct"] = float(100 * np.sqrt(np.mean(((r - f["pred"]) / r) ** 2)))
        f["max_pct"] = float(100 * np.max(np.abs((r - f["pred"]) / r)))
    return cands, min(cands, key=lambda k_: cands[k_]["aic"])


print("\n### refitted debias templates (apply as BIND_debiased = BIND_raw / r(a), a=1/(1+z))")
for lab, L in (("OLD", OLDL), ("NEW", NEWL)):
    a_arr = 1 / (1 + L[0])
    for qty, rv, ev in (("Y_500c", L[1], L[3]), ("f_gas", L[2], L[4])):
        cands, best = fit_templates(a_arr, rv, ev)
        c = cands[best]
        print(f"  {lab} {qty:8s} BEST={best:12s} rms={c['rms_pct']:.2f}% "
              f"max={c['max_pct']:.2f}% chi2/dof={c['chi2']/(len(a_arr)-c['k']):.1f}  "
              + ", ".join(f"{k}={v:.4f}" for k, v in c["coef"].items()))
    alpha = np.polyfit(np.log(a_arr), np.log(L[1]), 1)[0]
    i96 = SNAPS.index(96)
    print(f"  {lab} a-factor test: Y ratio ~ a^alpha, alpha = {alpha:+.3f} "
          f"(a pure missing a-factor would give -1); Y ratio {L[1][i96]:.3f} at "
          f"z={L[0][i96]:.2f} -> max {L[1].max():.3f} at z={L[0][int(np.argmax(L[1]))]:.2f}; "
          f"f_gas ratio range {L[2].min():.3f}-{L[2].max():.3f}")
np.savez_compressed("/mnt/home/mlee1/ceph/referee_work/fidswap/fig11_numbers.npz",
                    snaps=SNAPS, z=NEWL[0], y_old=OLDL[1], y_new=NEWL[1],
                    f_old=OLDL[2], f_new=NEWL[2], ye_old=OLDL[3], ye_new=NEWL[3],
                    fe_old=OLDL[4], fe_new=NEWL[4])
print("\nDONE fig11 numbers (figure DEFERRED: panel (d) needs the stacked "
      "profiles/fid_snapNNN.npz cache for all 20 snapshots)")
