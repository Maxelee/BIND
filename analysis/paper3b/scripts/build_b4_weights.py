"""WP-B4 prep: build + persist the frozen source-plane weighting bundle.

Serial, ~1 min (loads the bind-fiducial κ planes once for the cross-plane
covariance). Run BEFORE the grid driver; the driver refuses to run without
the bundle (no silently recomputed weights). Output:

    <out>/weights_desy3.npz
        source_z, w_default (amplitude-corrected, 12-shell fit),
        w_variant8 (8-shell fit — grid-sensitivity variant),
        w_plain (midpoint binning — REJECTED default, kept for the
        systematic bracket), q, shells, max_fit_frac_err, plane_cov,
        nz_z, nz (combined DES Y3 SOMPZ), neff_perbin, omega_m.

Conventions: analysis/paper3b/mocks/nz.py module docstring (B/wp4 session
2026-07-17); numbers recorded in projectB/wp4-forward-model/REPORT.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.paper3b.mocks.nz import (
    ATLAS_OMEGA_M,
    ShellAmplitudeModel,
    amplitude_corrected_weights,
    combined_desy3_source_nz,
    source_plane_weights,
)
from analysis.paper3b.mocks.shape_noise import DESY3_NEFF_PERBIN_ARCMIN2

DEFAULT_FID_KAPPA = "/mnt/home/mlee1/ceph/bind_science/runs/bind/run_0000/kappa_maps.npz"
DEFAULT_NZ_FITS = ("/mnt/home/mlee1/ceph/paper3/B/downloads/"
                   "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits")
DEFAULT_OUT = "/mnt/home/mlee1/ceph/paper3/B/wp4_mocks"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fid-kappa", default=DEFAULT_FID_KAPPA)
    ap.add_argument("--nz-fits", default=DEFAULT_NZ_FITS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    k = np.load(args.fid_kappa)
    source_z = np.asarray(k["source_redshifts"], dtype=float)
    kap = k["kappa"].astype(np.float64)
    n_real, K = kap.shape[:2]
    X = kap.reshape(n_real, K, -1)
    X -= X.mean(axis=2, keepdims=True)
    plane_cov = np.einsum("rip,rjp->ij", X, X) / (n_real * X.shape[2])
    del kap, X

    nz_z, nz = combined_desy3_source_nz(args.nz_fits)

    fit12 = ShellAmplitudeModel.fit_plane_cov(plane_cov, source_z)
    fit8 = ShellAmplitudeModel.fit_plane_cov(plane_cov, source_z,
                                             shells=np.linspace(0.05, 1.6, 8))
    for name, f in (("12-shell", fit12), ("8-shell", fit8)):
        print(f"{name}: max fit err {f.max_fit_frac_err * 100:.2f}%")
        if f.max_fit_frac_err > 0.02:
            raise RuntimeError(f"{name} shell fit exceeds the 2% acceptance")

    w_default = amplitude_corrected_weights(source_z, nz_z, nz, fit12)
    w_variant8 = amplitude_corrected_weights(source_z, nz_z, nz, fit8)
    w_plain = source_plane_weights(source_z, nz_z, nz)
    print("w_default :", np.round(w_default, 4), "sum", round(float(w_default.sum()), 4))
    print("w_variant8:", np.round(w_variant8, 4), "sum", round(float(w_variant8.sum()), 4))
    print("w_plain   :", np.round(w_plain, 4), "(rejected default; bracket only)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights_desy3.npz",
             source_z=source_z, w_default=w_default, w_variant8=w_variant8,
             w_plain=w_plain, q=fit12.q, shells=fit12.shells,
             q8=fit8.q, shells8=fit8.shells,
             max_fit_frac_err=fit12.max_fit_frac_err,
             max_fit_frac_err8=fit8.max_fit_frac_err,
             plane_cov=plane_cov, nz_z=nz_z, nz=nz,
             neff_perbin=np.asarray(DESY3_NEFF_PERBIN_ARCMIN2, dtype=float),
             omega_m=ATLAS_OMEGA_M,
             fid_kappa_path=str(args.fid_kappa), nz_fits_path=str(args.nz_fits))
    print("wrote", out / "weights_desy3.npz")


if __name__ == "__main__":
    main()
