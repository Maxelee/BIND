"""``bind-lux-collect``: stack lux ray-tracing output into kappa/y npz.

Reads the per-realization ``run<NNN>/kappa{plane}.dat`` (and ``y{plane}.dat`` /
``tau{plane}.dat`` if present) under a lux ``RT_output_dir`` and writes compact
``kappa_maps.npz`` / ``y_maps.npz`` / ``tau_maps.npz`` (shape ``(n_real, n_src,
npix, npix)``) into ``--output_dir`` for the stats stage.  Optionally deletes the
raw ``.dat`` afterwards (the npz is the kept data product).

    bind-lux-collect --rt_root run_0000/rt --output_dir run_0000
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from bind.inference.lux_io import (
    load_kappa_realizations, load_y_realizations, load_tau_realizations, PLANE_TO_ZS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rt_root", type=Path, required=True,
                   help="lux RT_output_dir holding run<NNN>/ subdirs")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--planes", type=int, nargs="+", default=[26, 45, 59, 70, 78])
    p.add_argument("--n_real", type=int, default=None,
                   help="Cap the number of realizations loaded (memory; default all).")
    p.add_argument("--fov_deg", type=float, default=5.0)
    p.add_argument("--delete_raw", action="store_true",
                   help="Delete the raw run<NNN>/ .dat dirs after collecting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    planes = tuple(args.planes)

    kappa, zs = load_kappa_realizations(args.rt_root, planes=planes, n_real=args.n_real)
    np.savez_compressed(args.output_dir / "kappa_maps.npz",
                        kappa=kappa.astype(np.float32),
                        source_redshifts=zs, fov_deg=args.fov_deg,
                        npix=kappa.shape[-1], n_real=kappa.shape[0])
    print(f"[collect] kappa_maps.npz  {kappa.shape}")
    del kappa   # free ~GBs before loading y (avoid OOM)

    try:
        y, _ = load_y_realizations(args.rt_root, planes=planes, n_real=args.n_real)
        np.savez_compressed(args.output_dir / "y_maps.npz",
                            y=y.astype(np.float32), source_redshifts=zs,
                            fov_deg=args.fov_deg, npix=y.shape[-1], n_real=y.shape[0])
        print(f"[collect] y_maps.npz      {y.shape}")
        del y
    except FileNotFoundError:
        print("[collect] no complete y planes — skipping y_maps.npz")

    try:
        tau, _ = load_tau_realizations(args.rt_root, planes=planes, n_real=args.n_real)
        np.savez_compressed(args.output_dir / "tau_maps.npz",
                            tau=tau.astype(np.float32), source_redshifts=zs,
                            fov_deg=args.fov_deg, npix=tau.shape[-1], n_real=tau.shape[0])
        print(f"[collect] tau_maps.npz    {tau.shape}")
        del tau
    except FileNotFoundError:
        print("[collect] no complete tau planes — skipping tau_maps.npz")

    if args.delete_raw:
        for rd in sorted(args.rt_root.glob("run*")):
            if rd.is_dir():
                shutil.rmtree(rd)
        print(f"[collect] deleted raw run*/ dirs under {args.rt_root}")


if __name__ == "__main__":
    main()
