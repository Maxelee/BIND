#!/usr/bin/env python3
"""fidswap -- field-level validation numbers, old fiducial vs new.

Everything here comes from the per-realization field caches, seed-paired
realization by realization (the BIND, truth and DMO traces share the 1992+7r
RT-seed ladder), so cosmic variance cancels.

  A. S(ell) closure against the full-hydro TNG300 truth, on the model's 24 bands
     -- the science item the swap changes most.
  B. fig-4 headline residuals: median |BIND/truth - 1| over ell in [300, 5000]
     for C_kk, C_ky, C_yy, plus the C_tautau and C_ktau legs.
  C. paired chi2/dof of the BIND-vs-truth residual (diagonal, 50-real paired SE).

    python referee/work/fs8_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
sys.path.insert(0, str(P1))

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
SB35 = CEPH / "bind_sb35"
FS = CEPH / "referee_work/fidswap"
ZI = 1
CANON = 49

coef = np.load(P1 / "latent_model_coeffs.npz")
EDGb, ctr = coef["ell_edges"], coef["ell"]
dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
_eld = np.asarray(dsn["a__suppression__ell"], float)
DMO = np.load(SCI / "runs/dmo/run_0000/Cl_kappa_paired.npz")["cl_real"]


def band_reals(cl):
    S = cl[:, ZI, :] / DMO[:, ZI, :]
    return np.stack([np.nanmean(S[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                     for i in range(len(EDGb) - 1)], axis=1)


def main():
    old = np.load(SCI / "field_cache/field_stats_fid.npz")
    new = np.load(FS / f"field_cache/field_stats_fidtb{CANON:02d}.npz")
    ell = old["ell"]

    So, Sn = band_reals(old["kk_bind"]), band_reals(new["kk_bind"])
    St = band_reals(old["kk_truth"])            # truth leg is common
    mo, mn_, mt = So.mean(0), Sn.mean(0), St.mean(0)

    print("A. S(ell) vs full-hydro TNG300 truth (z_s=1, 24 model bands)")
    print(f"{'band':>4s} {'ell':>8s} {'truth':>8s} {'old':>8s} {'new':>8s} "
          f"{'old-truth':>10s} {'new-truth':>10s} {'new-old':>9s}")
    for i in range(len(ctr)):
        print(f"{i:4d} {ctr[i]:8.0f} {mt[i]:8.4f} {mo[i]:8.4f} {mn_[i]:8.4f} "
              f"{mo[i]-mt[i]:+10.4f} {mn_[i]-mt[i]:+10.4f} {mn_[i]-mo[i]:+9.4f}")
    lo, hi = ctr < 5000, ctr >= 5000
    for nm, sel in (("ell < 5000", lo), ("ell >= 5000", hi), ("all bands", np.ones_like(lo))):
        print(f"  RMS |S - S_truth|  {nm:>11s}:  old {np.sqrt(np.mean((mo-mt)[sel]**2)):.4f}"
              f"   new {np.sqrt(np.mean((mn_-mt)[sel]**2)):.4f}")
    k = int(np.argmin(mt))
    print(f"  truth trough  band {k} (ell~{ctr[k]:.0f}): truth {mt[k]:.4f}, "
          f"old {mo[k]:.4f} ({mo[k]-mt[k]:+.4f}), new {mn_[k]:.4f} ({mn_[k]-mt[k]:+.4f})")

    print("\nB. median |BIND/truth - 1| over ell in [300, 5000], z_s=1")
    m = (ell >= 300) & (ell <= 5000)
    print(f"{'stat':>6s} {'old %':>8s} {'new %':>8s}")
    for st in ("kk", "ky", "yy"):
        a = np.nanmean(old[f"{st}_bind"][:, ZI, :], 0) / np.nanmean(old[f"{st}_truth"][:, ZI, :], 0)
        b = np.nanmean(new[f"{st}_bind"][:, ZI, :], 0) / np.nanmean(new[f"{st}_truth"][:, ZI, :], 0)
        print(f"{st:>6s} {100*np.nanmedian(np.abs(a[m]-1)):8.2f} "
              f"{100*np.nanmedian(np.abs(b[m]-1)):8.2f}")
    for st in ("tt", "kt"):
        a = np.nanmean(old[f"{st}_bind"][:, ZI, :], 0)
        b = np.nanmean(new[f"{st}_bind"][:, ZI, :], 0)
        print(f"{st:>6s}  new/old over the same band: "
              f"{np.nanmedian((b/a)[m]):.4f}  (predicted (Ob/Om)^2 = 0.9279 for tt)")

    print("\nC. paired chi2/dof, BIND - truth over ell in [300, 5000] (diagonal, "
          "50-real paired SE)")
    for st in ("kk", "ky", "yy"):
        for tag, d in (("old", old), ("new", new)):
            r = d[f"{st}_bind"][:, ZI, :] / d[f"{st}_truth"][:, ZI, :] - 1.0
            mu, se = np.nanmean(r, 0), np.nanstd(r, 0, ddof=1) / np.sqrt(len(r))
            z2 = (mu[m] / se[m]) ** 2
            z2 = z2[np.isfinite(z2)]
            print(f"  {st}_{tag}: chi2/dof = {z2.mean():8.1f}  (dof {len(z2)})")

    print("\nD. S(ell) paint-to-paint spread over the three replicas")
    Sr = {r: band_reals(np.load(FS / f"field_cache/shards/kk_bind_tb{r:02d}.npz")["cl"]).mean(0)
          for r in (18, 49, 53)}
    A = np.stack([Sr[r] for r in (18, 49, 53)])
    sd = A.std(0, ddof=1)
    se50 = Sn.std(0, ddof=1) / np.sqrt(len(Sn))
    print(f"  per-band paint sd: min {sd.min():.5f}  median {np.median(sd):.5f}  "
          f"max {sd.max():.5f}   ({100*np.median(sd/A.mean(0)):.3f}% median fractional)")
    print(f"  paint sd / 50-real paired SE: median {np.median(sd/se50):.2f}, "
          f"max {np.max(sd/se50):.2f}")

    np.savez(FS / "validation_fidswap.npz", ell_band=ctr, S_truth=mt, S_old=mo,
             S_new=mn_, S_replicas=A, replica_ids=np.array([18, 49, 53]),
             S_new_real=Sn, S_old_real=So, S_truth_real=St, ell=ell)
    print(f"\nwrote {FS/'validation_fidswap.npz'}")


if __name__ == "__main__":
    main()
