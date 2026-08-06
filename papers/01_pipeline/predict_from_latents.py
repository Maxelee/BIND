#!/usr/bin/env python
"""predict_from_latents.py -- S(ell) from four halo-population numbers.

The user-facing arithmetic of the analytic latent model (ANALYTIC_LATENT_MODEL.md,
figs 20d-g), in the HARMONIZED convention (2026-08-06): every latent is measured
from ONE snapshot (096, z=0.034), ONE mass bin (hinge, log10 m_tot_500c_bg in
[13.3, 13.6)), ONE aperture/background convention (projected cylinders, 2.5-3.0
Mpc/h transverse annulus, bg-subtracted totals/gas) and ONE reducer (median over
bin halos):

    f~_bar  = med[(m_gas_500c_bg + m_star_500c)/m_tot_500c_bg] / (Ob/Om)
    f~_star = med[m_star_500c/m_tot_500c_bg] / (Ob/Om)
    c_gas   = med[m_gas_500c_bg / m_gas_200c_bg]   (R500c=0.659 r200 vs r200)
    logT~   = log10 med[T_mw_500c]                 (gas-mass-weighted, Kelvin)

(Convert observed 3-D fractions to the cylinder convention first -- see the
doc's caveat 6.) The prediction is then, per band and source plane:

    S(ell_b) = c0 + c1*f~_bar + c2*f~_star + c3*c_gas + c4*logT~

The kernel table c_i(ell_b, z_s) is fit once from the 256-node Sobol suite and
cached in latent_model_coeffs.npz next to this script (gitignored; rebuilt on
demand -- and automatically when a pre-harmonization cache is found). The cache
also stores the per-band 5-fold-CV residual std (the model-error term) and the
prior cloud's latent ranges (inputs outside them are flagged as extrapolation).

    /mnt/home/mlee1/venvs/BIND_env/bin/python predict_from_latents.py \
        --fbar 0.80 --fstar 0.10 --cgas 0.65 --logt 6.9 --zs 1.0

Prints the 24-band table and, with --out, writes it as an npz.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "latent_model_coeffs.npz"
CACHE_VERSION = 2                      # v2 = harmonized 4-latent convention
CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
OB_OM = 0.0486 / 0.3089

COEFS = ("fbar", "fstar", "cgas", "logT", "const")
LATENTS = ("f~_bar", "f~_star", "c_gas", "logT~")


def build_cache() -> None:
    """Fit c_i(ell_b, z_s) on the full Sobol suite and cache the table."""
    d = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    run_ids = d["run_ids"]
    ELL = d["a__suppression__ell"]
    ZS = d["source_redshifts"]
    n = len(run_ids)

    cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")
    mt = cz["sobol_m_tot_500c_bg"][run_ids]
    mg = cz["sobol_m_gas_500c_bg"][run_ids]
    g2 = cz["sobol_m_gas_200c_bg"][run_ids]
    ms = cz["sobol_m_star_500c"][run_ids]
    Tm = cz["sobol_T_mw_500c"][run_ids]
    lm = np.log10(np.where(mt > 0, mt, np.nan))

    def binmed(f, valid=None):
        out = np.full(n, np.nan)
        for q in range(n):
            s = (lm[q] >= 13.3) & (lm[q] < 13.6)
            if valid is not None:
                s &= valid[q]
            if s.sum() >= 5:
                out[q] = np.nanmedian(f[q, s])
        return out

    fbar = binmed(np.where(mt > 0, (mg + ms) / mt, np.nan)) / OB_OM
    fstar = binmed(np.where(mt > 0, ms / mt, np.nan)) / OB_OM
    cgas = binmed(np.where(g2 > 0, mg / g2, np.nan), valid=(g2 > 0))
    logT = np.log10(binmed(np.where(Tm > 0, Tm, np.nan)))
    ok = (np.isfinite(fbar) & np.isfinite(fstar) & np.isfinite(cgas)
          & np.isfinite(logT))

    EDG = np.geomspace(300.0, 3e4, 25)
    ctr = np.sqrt(EDG[:-1] * EDG[1:])
    L = np.c_[fbar[ok], fstar[ok], cgas[ok], logT[ok]]
    A = np.c_[L, np.ones(int(ok.sum()))]

    beta = np.empty((len(ZS), 24, 5))
    sig_model = np.empty((len(ZS), 24))
    fold = np.arange(int(ok.sum())) % 5           # deterministic CV folds
    for zi in range(len(ZS)):
        Sb = np.stack([d["t__suppression__value"][:, zi, :]
                       [:, (ELL >= EDG[i]) & (ELL < EDG[i + 1])].mean(1)
                       for i in range(24)], axis=1)[ok]
        beta[zi] = np.linalg.lstsq(A, Sb, rcond=None)[0].T   # (24, 5)
        pred = np.empty_like(Sb)
        for f in range(5):
            tr = fold != f
            bf, *_ = np.linalg.lstsq(A[tr], Sb[tr], rcond=None)
            pred[fold == f] = A[fold == f] @ bf
        sig_model[zi] = (Sb - pred).std(0)
    np.savez(CACHE, version=CACHE_VERSION, beta=beta, sig_model=sig_model,
             ell=ctr, ell_edges=EDG, zs=ZS, coef_names=np.array(COEFS),
             lat_lo=L.min(0), lat_hi=L.max(0), n_nodes=int(ok.sum()),
             src=str(SB35 / "emulator_dataset_nu05.npz"))
    print(f"kernel table (harmonized v{CACHE_VERSION}) fit on "
          f"{int(ok.sum())} nodes -> {CACHE.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fbar", type=float, required=True,
                    help="group budget f~_bar (cylinder, hinge bin, /(Ob/Om))")
    ap.add_argument("--fstar", type=float, required=True,
                    help="group partition f~_star (same units)")
    ap.add_argument("--cgas", type=float, required=True,
                    help="two-aperture gas concentration "
                         "m_gas(<R500c)/m_gas(<R200c), bg-subtracted")
    ap.add_argument("--logt", type=float, required=True,
                    help="log10 of the bin-median mass-weighted T_500c [K]")
    ap.add_argument("--zs", type=float, default=1.0,
                    help="source plane (one of 0.5, 1.0, 1.5, 2.0, 2.44)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional npz to write (ell, S, sig_model)")
    ap.add_argument("--rebuild", action="store_true",
                    help="refit the kernel table even if cached")
    a = ap.parse_args()

    stale = True
    if CACHE.exists() and not a.rebuild:
        c0 = np.load(CACHE)
        stale = ("version" not in c0.files
                 or int(np.atleast_1d(c0["version"])[0]) != CACHE_VERSION)
        if stale:
            print("cached kernel table predates the harmonized convention "
                  "-- refitting")
    if stale or a.rebuild:
        build_cache()
    c = np.load(CACHE)
    zi = int(np.argmin(np.abs(c["zs"] - a.zs)))
    if abs(float(c["zs"][zi]) - a.zs) > 1e-3:
        print(f"note: nearest tabulated source plane is z_s={c['zs'][zi]:.2f}")

    lat = np.array([a.fbar, a.fstar, a.cgas, a.logt])
    lo, hi = c["lat_lo"], c["lat_hi"]
    for name, v, l, h in zip(LATENTS, lat, lo, hi):
        if not (l <= v <= h):
            print(f"WARNING: {name} = {v:.3f} is OUTSIDE the training cloud "
                  f"[{l:.3f}, {h:.3f}] -- this is an extrapolation; the "
                  f"linear kernels are unvalidated there")

    x = np.r_[lat, 1.0]
    S = c["beta"][zi] @ x                          # (24,)
    sig = c["sig_model"][zi]
    print(f"\nS(ell) at z_s={float(c['zs'][zi]):.2f} for f~_bar={a.fbar:.3f}, "
          f"f~_star={a.fstar:.3f}, c_gas={a.cgas:.3f}, logT={a.logt:.2f} "
          f"(+- = per-band CV model error):")
    for i in range(24):
        print(f"  ell~{c['ell'][i]:7.0f}: S = {S[i]:.4f} +- {sig[i]:.4f}")
    if a.out:
        np.savez(a.out, ell=c["ell"], S=S, sig_model=sig,
                 zs=float(c["zs"][zi]), latents=lat)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
