"""Sanity probes before building the R2 figures.

1. Measure Nhat(ell) (pure shape noise through the exact smooth+Pk path) and
   compare the implied W^2(ell) with exp(-ell^2 sigma_theta^2).
2. Check the measured flat shot-noise level against sigma_e^2/(2 n_gal).
3. Check the noise-debiased noisy S(ell) against the NOISELESS suppression the
   released dataset carries -- if the debiasing is right, they must agree.
4. Check whether fid/truth noisy realizations are seed-paired (paired difference
   scatter < unpaired quadrature sum).
"""
from __future__ import annotations

import numpy as np

import r2_common as R


def main() -> None:
    nb = R.measure_noise_bias_cl()
    ell = nb["ell"]
    print(f"[1] noise-bias spectrum from {int(nb['n_draw'])} pure-noise draws, "
          f"{len(ell)} ell bins, ell {ell[0]:.0f}..{ell[-1]:.0f}")
    print(f"    measured flat (unsmoothed) shot-noise level  {float(nb['flat_level']):.4e}")
    print(f"    analytic sigma_e^2/(2 n_gal) [sr]            {float(nb['analytic_flat']):.4e}"
          f"   ratio {float(nb['flat_level'])/float(nb['analytic_flat']):.4f}")
    for lt in (500, 1000, 2000, 5000, 8000, 12000, 20000):
        i = int(np.argmin(np.abs(ell - lt)))
        print(f"    ell={ell[i]:>7.0f}: W2_meas {nb['w2'][i]:.4e}  "
              f"W2_analytic {nb['w2_analytic'][i]:.4e}  "
              f"ratio {nb['w2'][i]/max(nb['w2_analytic'][i],1e-30):.4f}  "
              f"Nhat {nb['nhat'][i]:.4e}")

    fid = R.load_noisy_target("fid")
    dmo = R.load_noisy_target("dmo")
    tru = R.load_noisy_target("truth")
    assert np.array_equal(fid["cl_ell"], ell), "noise-bias ell grid != shard ell grid"
    print(f"\n[2] noisy shards: fid n_real={int(fid['n_real'])} "
          f"truth={int(tru['n_real'])} dmo={int(dmo['n_real'])}")

    # noise floor vs signal at z_s=1
    zi = R.ZI
    cf = fid["cl_kappa"][zi]
    cd = dmo["cl_kappa"][zi]
    nhat = nb["nhat"]
    print("\n[3] noise floor vs total, z_s=1 (fid):")
    for lt in (300, 1000, 3000, 5000, 8000, 12000):
        i = int(np.argmin(np.abs(ell - lt)))
        print(f"    ell={ell[i]:>7.0f}: Cl_tot {cf[i]:.4e}  Nhat {nhat[i]:.4e}  "
              f"signal {cf[i]-nhat[i]:.4e}  N/S {nhat[i]/max(cf[i]-nhat[i],1e-30):8.2f}")

    # debiased S(ell) vs the released noiseless suppression
    S_noisy = (cf - nhat) / (cd - nhat)
    d0 = np.load(R.SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    ell0 = d0["a__suppression__ell"]
    print("\n[4] noise-debiased noisy S(ell) [fid/dmo] vs noiseless released S(ell) "
          "[Sobol median, for scale]:")
    Ssob = d0["t__suppression__value"][:, zi, :]
    for lt in (300, 1000, 3000, 5000, 8000):
        i = int(np.argmin(np.abs(ell - lt)))
        j = int(np.argmin(np.abs(ell0 - lt)))
        print(f"    ell={ell[i]:>7.0f}: S_noisy_debiased {S_noisy[i]:.4f}   "
              f"S_noiseless(Sobol med) {np.nanmedian(Ssob[:, j]):.4f}  "
              f"(5-95% {np.nanpercentile(Ssob[:, j],5):.4f}-{np.nanpercentile(Ssob[:, j],95):.4f})")

    # raw (undebiased) ratio, to show the dilution
    S_raw = cf / cd
    print("\n[5] undebiased ratio Cl_fid^tot/Cl_dmo^tot (the dilution the noise causes):")
    for lt in (300, 1000, 3000, 5000, 8000):
        i = int(np.argmin(np.abs(ell - lt)))
        print(f"    ell={ell[i]:>7.0f}: raw ratio {S_raw[i]:.4f}  debiased {S_noisy[i]:.4f}")

    # seed pairing check on the nu statistics
    print("\n[6] seed-pairing check (fid vs truth noisy per-realization draws, z_s=1):")
    for k in R.NU_KEYS:
        b = fid[f"{k}_real"][:, zi, :]
        t = tru[f"{k}_real"][:, zi, :]
        n = min(len(b), len(t))
        pair = np.std(b[:n] - t[:n], axis=0)
        unpair = np.sqrt(np.var(b[:n], axis=0) + np.var(t[:n], axis=0))
        good = np.isfinite(pair) & (unpair > 0)
        print(f"    {k:14s} median std(b-t)/sqrt(var_b+var_t) = "
              f"{np.nanmedian(pair[good]/unpair[good]):.3f}   (<1 => seed-paired)")
    b = fid["cl_kappa_real"][:, zi, :]
    t = tru["cl_kappa_real"][:, zi, :]
    n = min(len(b), len(t))
    pair = np.std(b[:n] - t[:n], axis=0)
    unpair = np.sqrt(np.var(b[:n], axis=0) + np.var(t[:n], axis=0))
    m = (ell > 200) & (ell < 8000)
    print(f"    {'cl_kappa':14s} median std(b-t)/sqrt(var_b+var_t) = "
          f"{np.nanmedian(pair[m]/unpair[m]):.3f}   (ell 200-8000)")

    # measured relative scatter vs Gaussian mode counting
    print("\n[7] measured single-realization relative scatter of Cl^tot vs "
          "sqrt(2/N_modes) on the 25 deg^2 box:")
    ell_f0 = 2 * np.pi / np.deg2rad(5.0)
    rel = np.std(b, axis=0) / np.mean(b, axis=0)
    for lt in (300, 1000, 3000, 5000, 8000):
        i = int(np.argmin(np.abs(ell - lt)))
        nmod = 2 * np.pi * ell[i] * ell_f0 / ell_f0 ** 2
        print(f"    ell={ell[i]:>7.0f}: measured {rel[i]:.4f}   "
              f"sqrt(2/Nmodes)={np.sqrt(2/nmod):.4f}  ratio {rel[i]/np.sqrt(2/nmod):.3f}")


if __name__ == "__main__":
    main()
