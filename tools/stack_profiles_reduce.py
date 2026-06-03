"""Field-level *stacked radial profiles* per (design, mass bin) for the
Observable -> f_b PoC, profile edition.

For each Sobol design (feedback setting) and mass bin we stack all in-bin halos
at the field level (patches are already halo-centred at pixel (64,64)) and build
radial profiles in **physical** radius (pixel = 48.83 kpc/h):

  Y(r)   azimuthal mean of the stacked tSZ Compton-y map           (linear stack)
  SX(r)  azimuthal mean of the stacked X-ray brightness map
         (per-halo SX = Gas^2 * sqrt(T), then mean over halos -- you stack images)
  T,S,P(r)  gas-mass-weighted stacked thermo maps, azimuthal mean
  fb(r)  projected baryon-fraction profile:
         Sigma_b(r) / Sigma_tot(r) with Sigma_b = mean_halo(Gas+Stars),
         Sigma_tot = mean_halo(DM+Gas+Stars)   (the f_b *target* profile)

This keeps 256 design-level samples per mass bin, so the downstream profile->profile
prediction can still be tested on held-out designs (KFold over designs).

Output -> /mnt/home/mlee1/ceph/sobol_ss_cv/stacked_profiles.npz
  r_kpc       (nr,)               radial-bin centres [kpc/h]
  counts      (3,)                # halos per mass bin (shared across designs)
  <name>      (256, 3, nr)        stacked profile, name in {Y,SX,T,S,P,fb,fgas,fstar}
  mass_lbl    (3,)                bin labels

Run (~10-15 min single thread):  python tools/stack_profiles_reduce.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SOBOL = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
OUT = SOBOL / "stacked_profiles.npz"

PATCH = 128
PIX_KPC = 50.0 / 1024 * 1000.0            # 48.83 kpc/h per pixel
_yy, _xx = np.mgrid[0:PATCH, 0:PATCH]
_Rkpc = np.sqrt((_xx - PATCH / 2.0) ** 2 + (_yy - PATCH / 2.0) ** 2) * PIX_KPC

# radial bins defined in PIXELS (always populated; pixel=48.83 kpc/h), 1->38 px,
# then expressed in physical kpc/h. Inner sub-pixel log bins would be empty.
R_EDGES = np.logspace(np.log10(1.0), np.log10(38.0), 13) * PIX_KPC
R_CEN = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])
NR = len(R_CEN)
# precompute pixel->radial-bin assignment (-1 outside range)
_RBIN = np.digitize(_Rkpc.ravel(), R_EDGES) - 1
_RBIN[(_RBIN < 0) | (_RBIN >= NR)] = -1

MASS_BINS = [(13.0, 13.5), (13.5, 14.0), (14.0, np.inf)]
MASS_LBL = ["[13.0,13.5)", "[13.5,14.0)", "[14.0,+)"]


def azim(map2d):
    """Azimuthal mean of a 2D map into the physical radial bins."""
    v = map2d.ravel()
    out = np.full(NR, np.nan)
    for b in range(NR):
        m = _RBIN == b
        if m.any():
            out[b] = v[m].mean()
    return out


def main():
    cube = np.load(SOBOL / "cube.npz", allow_pickle=True)
    M200 = np.asarray(cube["M200"], float)
    logM = np.log10(M200)
    bin_idx = [np.where((logM >= lo) & (logM < hi))[0] for lo, hi in MASS_BINS]
    counts = np.array([len(i) for i in bin_idx])
    print("halos per bin:", dict(zip(MASS_LBL, counts)))

    map_files = sorted((SOBOL / "maps").glob("gen_design*.npz"))
    nD = len(map_files)
    names = ["Y", "SX", "T", "S", "P", "fb", "fgas", "fstar"]
    prof = {k: np.full((nD, 3, NR), np.nan) for k in names}

    for d, mf in enumerate(map_files):
        t0 = time.time()
        g = np.load(mf)["generated"].astype(np.float32)     # (1111,7,128,128)
        DM, GAS, STAR, Y, T, S, P = (g[:, c] for c in range(7))
        SXpix = np.clip(GAS, 0, None) ** 2 * np.sqrt(np.clip(T, 0, None))
        for b, idx in enumerate(bin_idx):
            if len(idx) == 0:
                continue
            gas = GAS[idx].mean(0); star = STAR[idx].mean(0); dm = DM[idx].mean(0)
            tot = dm + gas + star
            wsum = GAS[idx].sum(0)                            # gas weight for thermo
            prof["Y"][d, b]  = azim(Y[idx].mean(0))
            prof["SX"][d, b] = azim(SXpix[idx].mean(0))
            prof["T"][d, b]  = azim((T[idx] * GAS[idx]).sum(0) / np.where(wsum > 0, wsum, np.nan))
            prof["S"][d, b]  = azim((S[idx] * GAS[idx]).sum(0) / np.where(wsum > 0, wsum, np.nan))
            prof["P"][d, b]  = azim((P[idx] * GAS[idx]).sum(0) / np.where(wsum > 0, wsum, np.nan))
            sb = azim(gas + star); st = azim(tot)
            prof["fb"][d, b]    = sb / st
            prof["fgas"][d, b]  = azim(gas) / st
            prof["fstar"][d, b] = azim(star) / st
        if d % 32 == 0 or d == nD - 1:
            print(f"[{d+1}/{nD}] ({time.time()-t0:.1f}s)", flush=True)

    np.savez_compressed(OUT, r_kpc=R_CEN, counts=counts,
                        mass_lbl=np.array(MASS_LBL), **prof)
    print("wrote", OUT, "NR=", NR)


if __name__ == "__main__":
    main()
