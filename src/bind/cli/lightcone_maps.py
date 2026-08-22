"""``bind-lightcone-maps`` (stage 2): assemble kappa + tSZ-y lightcone maps.

Flat-sky Born assembly of the per-snapshot BIND composites into angular maps on
a common grid (see :mod:`bind.inference.lightcone_maps`).  Writes, into
``--output_dir``:

* ``kappa_maps.npz`` — ``kappa`` ``(n_real, n_source_bins, npix, npix)`` for the
  BIND total-matter field; ``kappa_dmo`` (same shape) when ``--with_dmo`` so the
  same-pipeline suppression ratio cancels the projection MAS artifact.
* ``y_maps.npz``     — ``y`` ``(n_real, n_source_bins, npix, npix)`` Compton-y (tSZ).
* ``tau_maps.npz``   — ``tau`` ``(n_real, n_source_bins, npix, npix)`` kSZ optical
  depth from the gas electron column; FRB ``DM = tau / TAU_PER_DM`` (pc/cm^3), the
  same map in FRB units.  Pixel-aligned with kappa/y for cross-correlation.

``--n_real`` realizations differ by the random per-snapshot transverse shift
(the fiducial run typically uses more realizations than the Sobol runs).

    bind-lightcone-maps --snap_root /ceph/bind_science/runs/fiducial/run_0000 \\
        --output_dir /ceph/bind_science/runs/fiducial/run_0000 \\
        --source_redshifts 0.5 1.0 1.5 2.0 --n_real 8 --npix 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference.lightcone_maps import assemble_lightcone

DEFAULT_SNAPSHOTS = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56,
                     52, 49, 46, 43, 41, 38, 35, 33, 31, 29]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap_root", type=Path, required=True,
                   help="Dir holding snap_<NNN>/composite_slab*.npz")
    p.add_argument("--manifest_root", type=Path, default=None,
                   help="Where snap_<NNN>/stage1/stage1_manifest.json lives, if "
                        "separate from --snap_root (e.g. the shared stage-1 tree "
                        "for per-run science dirs). Defaults to --snap_root.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--snapshots", type=int, nargs="+", default=DEFAULT_SNAPSHOTS)
    p.add_argument("--source_redshifts", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--fov_deg", type=float, default=5.0)
    p.add_argument("--npix", type=int, default=1024)
    p.add_argument("--n_real", type=int, default=8)
    p.add_argument("--omega_m", type=float, default=0.3089)
    p.add_argument("--r200_factor", type=float, default=4.0)
    p.add_argument("--with_dmo", action="store_true",
                   help="Also assemble the same-pipeline DMO kappa (for the "
                        "artifact-cancelling suppression ratio).")
    p.add_argument("--no_y", action="store_true", help="Skip the tSZ y-map.")
    p.add_argument("--no_tau", action="store_true",
                   help="Skip the kSZ-tau / FRB-DM electron-column map.")
    p.add_argument("--seed0", type=int, default=0, help="First realization seed.")
    p.add_argument("--mass_key", default="composite",
                   help="Slab array summed for the total-matter map. The spherical-BCM "
                        "trees carry three keys: 'composite' (radial-warp BCM), "
                        "'composite_mono' (circular monopole), 'composite_bind' (BIND).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(snapshots=args.snapshots, source_redshifts=tuple(args.source_redshifts),
                  fov_deg=args.fov_deg, npix=args.npix, Omega_m=args.omega_m,
                  r200_factor=args.r200_factor, manifest_root=args.manifest_root,
                  mass_key=args.mass_key, verbose=False)

    kappa, ymaps, taumaps, kappa_dmo = [], [], [], []
    for r in range(args.n_real):
        seed = args.seed0 + r
        res = assemble_lightcone(args.snap_root, field="bind",
                                 want_y=not args.no_y, want_tau=not args.no_tau,
                                 seed=seed, **common)
        kappa.append(res["kappa"])
        if not args.no_y:
            ymaps.append(res["y"])
        if not args.no_tau:
            taumaps.append(res["tau"])
        if args.with_dmo:
            rd = assemble_lightcone(args.snap_root, field="dmo",
                                    want_y=False, seed=seed, **common)
            kappa_dmo.append(rd["kappa"])
        print(f"[maps] realization {r} (seed {seed}) done; planes={res['n_planes']}")

    kappa = np.asarray(kappa, dtype=np.float32)            # (n_real, n_src, npix, npix)
    kw = dict(kappa=kappa, source_redshifts=np.asarray(args.source_redshifts),
              fov_deg=args.fov_deg, npix=args.npix, n_real=args.n_real)
    if args.with_dmo:
        kw["kappa_dmo"] = np.asarray(kappa_dmo, dtype=np.float32)
    np.savez_compressed(args.output_dir / "kappa_maps.npz", **kw)
    print(f"[maps] wrote kappa_maps.npz  kappa{kappa.shape}")

    if not args.no_y:
        y = np.asarray(ymaps, dtype=np.float32)            # (n_real, n_src, npix, npix)
        np.savez_compressed(args.output_dir / "y_maps.npz", y=y,
                            source_redshifts=np.asarray(args.source_redshifts),
                            fov_deg=args.fov_deg, npix=args.npix, n_real=args.n_real)
        print(f"[maps] wrote y_maps.npz  y{y.shape}")

    if not args.no_tau:
        from bind.inference.lightcone_maps import TAU_PER_DM
        tau = np.asarray(taumaps, dtype=np.float32)        # (n_real, n_src, npix, npix)
        np.savez_compressed(args.output_dir / "tau_maps.npz", tau=tau,
                            tau_per_dm=TAU_PER_DM,
                            source_redshifts=np.asarray(args.source_redshifts),
                            fov_deg=args.fov_deg, npix=args.npix, n_real=args.n_real)
        print(f"[maps] wrote tau_maps.npz  tau{tau.shape}  (DM[pc/cm^3]=tau/{TAU_PER_DM:.3e})")


if __name__ == "__main__":
    main()
