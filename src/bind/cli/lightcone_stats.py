"""``bind-lightcone-stats`` (stage 3): summary statistics from a run's maps.

Reads ``kappa_maps.npz`` (+ optional ``y_maps.npz``) written by
``bind-lightcone-maps`` and writes the emulator-input statistics into the same
run directory (see :mod:`bind.inference.stats`):

* ``Cl_kappa.npz``          — WL auto/cross C_ell (+ suppression if kappa_dmo present)
* ``Cl_kappa_y.npz``        — WL x tSZ cross + tSZ auto (needs y_maps.npz)
* ``peak_counts.npz``       — peak counts vs nu
* ``nongaussian_stats.npz`` — PDF, moments per scale, Betti curves
* ``halo_scaling.npz``      — per-halo Y_500c/f_gas/f_star/T_mw (needs --snap_root
                              composites with saved patches)

    bind-lightcone-stats --run_dir /ceph/bind_science/runs/fiducial/run_0000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference import stats as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Dir holding kappa_maps.npz (and y_maps.npz); outputs go here")
    p.add_argument("--snap_root", type=Path, default=None,
                   help="Dir with snap_<NNN>/composite_slab*.npz for halo_scaling "
                        "(needs saved patches). Defaults to --run_dir.")
    p.add_argument("--snapshots", type=int, nargs="+", default=None)
    p.add_argument("--smoothing_arcmin", type=float, default=2.0)
    p.add_argument("--no_halo_scaling", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rd = args.run_dir
    km = np.load(rd / "kappa_maps.npz")
    kappa = km["kappa"]                                  # (n_real, n_src, npix, npix)
    fov = float(km["fov_deg"])
    kdmo = km["kappa_dmo"] if "kappa_dmo" in km.files else None

    ck = S.cl_kappa(kappa, fov_deg=fov, kappa_dmo=kdmo)
    np.savez(rd / "Cl_kappa.npz", **ck)
    print(f"[stats] Cl_kappa.npz  cl{ck['cl'].shape}"
          + (" +suppression" if "suppression" in ck else ""))

    ypath = rd / "y_maps.npz"
    if ypath.exists():
        y = np.load(ypath)["y"]
        # tomographic y -> use the total (last source plane) for the cross power
        y_total = y[:, -1] if y.ndim == kappa.ndim else y
        cky = S.cl_kappa_y(kappa, y_total, fov_deg=fov)
        np.savez(rd / "Cl_kappa_y.npz", **cky)
        print(f"[stats] Cl_kappa_y.npz  cl_ky{cky['cl_ky'].shape}")
        # tSZ stacked at WL peaks -> R(nu) emulator target (needs per-source-bin y)
        if y.ndim == kappa.ndim:           # (n_real, n_src, npix, npix)
            pc = S.peak_cross_stats(kappa, y, fov_deg=fov,
                                    smoothing_arcmin=args.smoothing_arcmin)
            np.savez(rd / "peak_cross.npz", **pc)
            print(f"[stats] peak_cross.npz  R{pc['R'].shape} + profile{pc['profile'].shape}")
        else:
            print("[stats] y_maps not per-source-bin — skipping peak_cross (R(nu))")
    else:
        print("[stats] no y_maps.npz — skipping Cl_kappa_y / peak_cross")

    pk = S.peak_counts(kappa, fov_deg=fov, smoothing_arcmin=args.smoothing_arcmin)
    np.savez(rd / "peak_counts.npz", **pk)
    print(f"[stats] peak_counts.npz  peaks{pk['peak_counts'].shape} + minima")

    ng = S.nongaussian_stats(kappa, fov_deg=fov)
    np.savez(rd / "nongaussian_stats.npz", **ng)
    print(f"[stats] nongaussian_stats.npz  pdf{ng['pdf'].shape}")

    if not args.no_halo_scaling:
        snap_root = args.snap_root or rd
        snaps = args.snapshots
        if snaps is None:
            snaps = sorted(int(p.name.split("_")[1]) for p in snap_root.glob("snap_*")
                           if (p / "composite_slab00.npz").exists())
        paths = [snap_root / f"snap_{s:03d}" / f"composite_slab{sl:02d}.npz"
                 for s in snaps for sl in range(8)
                 if (snap_root / f"snap_{s:03d}" / f"composite_slab{sl:02d}.npz").exists()]
        sample = np.load(paths[0]) if paths else None
        if paths and "generated_patches" in sample.files:
            hs = S.halo_scaling(paths)
            np.savez(rd / "halo_scaling.npz", **hs)
            print(f"[stats] halo_scaling.npz  {len(hs['halo_mass'])} halos")
        else:
            print("[stats] no composites with saved patches — skipping halo_scaling")


if __name__ == "__main__":
    main()
