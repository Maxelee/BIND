"""Cache absolute gas-column (and tot-column) profiles for the 256 BIND designs.

`tools/stack_profiles_reduce.py` only stores the *fractions* `fgas, fstar, fb` for
BIND designs; it does not store the absolute Sigma_gas(r) and Sigma_tot(r) profiles
that a kSZ analysis needs (kSZ pairwise stack ∝ optical depth τ ∝ Σ_gas).

This script re-reads the same `gen_design*.npz` maps and writes one extra file:

    /mnt/home/mlee1/ceph/sobol_ss_cv/gas_column_profile.npz
        gas_mean (256,3,nr)   ⟨Σ_gas⟩ in the same code units as the BIND maps
        tot_mean (256,3,nr)   ⟨Σ_DM + Σ_gas + Σ_stars⟩
        r_kpc, mass_lbl, counts (consistent with stacked_profiles.npz)

Same geometry/mass-binning/azimuthal averaging as `stack_profiles_reduce.py`
(re-imported below). Run once (~10-15 min single thread):

    python tools/stack_gas_column.py
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np

SOBOL = Path("/mnt/home/mlee1/ceph/sobol_ss_cv")
OUT = SOBOL / "gas_column_profile.npz"

spec = importlib.util.spec_from_file_location(
    "spr", Path(__file__).with_name("stack_profiles_reduce.py"))
spr = importlib.util.module_from_spec(spec); spec.loader.exec_module(spr)
azim, MASS_BINS, MASS_LBL, NR, R_CEN = spr.azim, spr.MASS_BINS, spr.MASS_LBL, spr.NR, spr.R_CEN


def main():
    cube = np.load(SOBOL / "cube.npz", allow_pickle=True)
    M200 = np.asarray(cube["M200"], float)
    logM = np.log10(M200)
    bin_idx = [np.where((logM >= lo) & (logM < hi))[0] for lo, hi in MASS_BINS]
    counts = np.array([len(i) for i in bin_idx])
    print("halos per bin:", dict(zip(MASS_LBL, counts)))

    map_files = sorted((SOBOL / "maps").glob("gen_design*.npz"))
    nD = len(map_files)
    gas_mean = np.full((nD, 3, NR), np.nan)
    tot_mean = np.full((nD, 3, NR), np.nan)
    for d, mf in enumerate(map_files):
        t0 = time.time()
        g = np.load(mf)["generated"].astype(np.float32)     # (1111,7,128,128)
        DM, GAS, STAR = g[:, 0], g[:, 1], g[:, 2]
        for b, idx in enumerate(bin_idx):
            if len(idx) == 0:
                continue
            gas = GAS[idx].mean(0)
            tot = (DM[idx] + GAS[idx] + STAR[idx]).mean(0)
            gas_mean[d, b] = azim(gas)
            tot_mean[d, b] = azim(tot)
        if d % 32 == 0 or d == nD - 1:
            print(f"[{d+1}/{nD}] ({time.time()-t0:.1f}s)", flush=True)

    np.savez_compressed(OUT, r_kpc=R_CEN, counts=counts,
                        mass_lbl=np.array(MASS_LBL),
                        gas_mean=gas_mean, tot_mean=tot_mean)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
