#!/usr/bin/env python
"""predict_from_latents.py -- S(ell) from three halo-population numbers.

The user-facing arithmetic of the analytic latent model (ANALYTIC_LATENT_MODEL.md,
figs 20d-f): given the group-scale budget f~_bar, partition f~_star and stacked-tau
concentration c_tau -- in the MODEL'S units (projected-cylinder apertures, hinge
mass bin log10 M500c,bg in [13.3, 13.6), tilde = fraction / (Omega_b/Omega_m);
convert observed 3-D fractions first, see the doc's caveat 6) -- predict the WL
suppression S(ell) at any of the five source planes:

    S(ell_b) = c0(ell_b,z_s) + c1*f~_bar + c2*f~_star + c3*c_tau

The kernel table c_i(ell_b, z_s) is fit once from the 256-node Sobol suite and
cached in latent_model_coeffs.npz next to this script (gitignored; rebuilt on
demand from emulator_dataset_nu05.npz + the atlas/tau caches). The cache also
stores the per-band 5-fold-CV residual std (the model-error term) and the prior
cloud's latent ranges (an input outside them = extrapolation, and is flagged).

    /mnt/home/mlee1/venvs/BIND_env/bin/python predict_from_latents.py \
        --fbar 0.80 --fstar 0.10 --ctau 2.15 --zs 1.0

Prints the 24-band table and, with --out, writes it as an npz.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "latent_model_coeffs.npz"
CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"
OB_OM = 0.0486 / 0.3089


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
    ms = cz["sobol_m_star_500c"][run_ids]
    lm = np.log10(np.where(mt > 0, mt, np.nan))

    def binmed(f):
        out = np.full(n, np.nan)
        for q in range(n):
            s = (lm[q] >= 13.3) & (lm[q] < 13.6)
            if s.sum() >= 5:
                out[q] = np.nanmedian(f[q, s]) / OB_OM
        return out

    fbar = binmed(np.where(mt > 0, (mg + ms) / mt, np.nan))
    fstar = binmed(np.where(mt > 0, ms / mt, np.nan))
    xp = np.load(SCI / "ksz_confront/bind_tauy_xprof_snap085.npz")
    tau = xp["tau"][:, 0, :][np.isin(xp["nodes"], run_ids)]
    xr = xp["x"]
    ctau = (np.nanmean(tau[:, xr < 0.3], 1)
            / np.nanmean(tau[:, (xr >= 0.3) & (xr < 1.0)], 1))
    ok = np.isfinite(fbar) & np.isfinite(fstar) & np.isfinite(ctau)

    EDG = np.geomspace(300.0, 3e4, 25)
    ctr = np.sqrt(EDG[:-1] * EDG[1:])
    A = np.c_[fbar[ok], fstar[ok], ctau[ok], np.ones(int(ok.sum()))]

    beta = np.empty((len(ZS), 24, 4))
    sig_model = np.empty((len(ZS), 24))
    fold = np.arange(int(ok.sum())) % 5           # deterministic CV folds
    for zi in range(len(ZS)):
        Sb = np.stack([d["t__suppression__value"][:, zi, :]
                       [:, (ELL >= EDG[i]) & (ELL < EDG[i + 1])].mean(1)
                       for i in range(24)], axis=1)[ok]
        beta[zi] = np.linalg.lstsq(A, Sb, rcond=None)[0].T   # (24, 4)
        pred = np.empty_like(Sb)
        for f in range(5):
            tr = fold != f
            bf, *_ = np.linalg.lstsq(A[tr], Sb[tr], rcond=None)
            pred[fold == f] = A[fold == f] @ bf
        sig_model[zi] = (Sb - pred).std(0)
    np.savez(CACHE, beta=beta, sig_model=sig_model, ell=ctr, ell_edges=EDG,
             zs=ZS, coef_names=np.array(["fbar", "fstar", "ctau", "const"]),
             lat_lo=np.array([fbar[ok].min(), fstar[ok].min(), ctau[ok].min()]),
             lat_hi=np.array([fbar[ok].max(), fstar[ok].max(), ctau[ok].max()]),
             n_nodes=int(ok.sum()),
             src=str(SB35 / "emulator_dataset_nu05.npz"))
    print(f"kernel table fit on {int(ok.sum())} nodes -> {CACHE.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fbar", type=float, required=True,
                    help="group budget f~_bar (cylinder, hinge bin, /(Ob/Om))")
    ap.add_argument("--fstar", type=float, required=True,
                    help="group partition f~_star (same units)")
    ap.add_argument("--ctau", type=float, required=True,
                    help="stacked-tau concentration <tau(x<0.3)>/<tau(0.3-1)>")
    ap.add_argument("--zs", type=float, default=1.0,
                    help="source plane (one of 0.5, 1.0, 1.5, 2.0, 2.44)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional npz to write (ell, S, sig_model)")
    ap.add_argument("--rebuild", action="store_true",
                    help="refit the kernel table even if cached")
    a = ap.parse_args()

    if a.rebuild or not CACHE.exists():
        build_cache()
    c = np.load(CACHE)
    zi = int(np.argmin(np.abs(c["zs"] - a.zs)))
    if abs(float(c["zs"][zi]) - a.zs) > 1e-3:
        print(f"note: nearest tabulated source plane is z_s={c['zs'][zi]:.2f}")

    lat = np.array([a.fbar, a.fstar, a.ctau])
    lo, hi = c["lat_lo"], c["lat_hi"]
    for name, v, l, h in zip(("f~_bar", "f~_star", "c_tau"), lat, lo, hi):
        if not (l <= v <= h):
            print(f"WARNING: {name} = {v:.3f} is OUTSIDE the training cloud "
                  f"[{l:.3f}, {h:.3f}] -- this is an extrapolation; the "
                  f"linear kernels are unvalidated there")

    x = np.r_[lat, 1.0]
    S = c["beta"][zi] @ x                          # (24,)
    sig = c["sig_model"][zi]
    print(f"\nS(ell) at z_s={float(c['zs'][zi]):.2f} for f~_bar={a.fbar:.3f}, "
          f"f~_star={a.fstar:.3f}, c_tau={a.ctau:.3f} "
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
