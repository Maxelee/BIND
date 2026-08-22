"""``bind-paint-yplane`` (stage 3, tSZ): composite Compton-y -> lux y-planes.

Writes ``yplane{PP:02d}.dat`` for one snapshot into the lensplane directory,
using the **same plane numbering** as ``bind-paint-lensplane`` so the modified
lux build integrates them along the rays with the identical per-realization
randomization as the lensing planes.  This yields tSZ y-maps that are
pixel-consistent with the ray-traced kappa maps (for the 1a kappa x y cross).

The per-slab Compton-y is taken from ``composite_thermo`` when present, else
composited on the fly from the saved ``thermo_patches`` (``r200_factor`` aperture),
then center-cropped to ``--lp_grid`` exactly like the lensplanes.

    bind-paint-yplane --generate_dir /ceph/.../snap_096 \\
        --output_dir /ceph/.../lensplanes --lc_snap_idx 0 --lc_n_snaps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.lensplane import write_yplane
from bind.inference.lightcone_maps import _thermo_y_from_patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate_dir", type=Path, required=True,
                   help="Directory with composite_slab*.npz (stage 2)")
    p.add_argument("--stage1_dir", type=Path, default=None,
                   help="Stage-1 dir for the manifest (defaults to generate_dir/stage1)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Lensplane directory (yplane*.dat written alongside lenspot*.dat)")
    p.add_argument("--lc_snap_idx", type=int, required=True)
    p.add_argument("--lc_n_snaps", type=int, required=True)
    p.add_argument("--planes_per_snapshot", type=int, default=None)
    p.add_argument("--lp_grid", type=int, default=4096)
    p.add_argument("--r200_factor", type=float, default=4.0)
    p.add_argument("--taper_frac", type=float, default=0.15)
    return p.parse_args()


def _yslab(npz_path: Path, r200_factor: float, taper_frac: float) -> np.ndarray:
    d = np.load(npz_path)
    if "composite_thermo" in d.files:
        return d["composite_thermo"][0].astype(np.float64)
    return _thermo_y_from_patches(d, r200_factor, taper_frac)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = args.stage1_dir or (args.generate_dir / "stage1")
    manifest = json.loads((stage1_dir / "stage1_manifest.json").read_text())
    n_slabs = int(manifest["n_slabs"])
    pps = args.planes_per_snapshot or n_slabs
    if pps != n_slabs:
        raise SystemExit(f"--planes_per_snapshot={pps} must equal n_slabs={n_slabs}")

    plane_offset = args.lc_snap_idx * pps + 1  # lux numbers planes from 1
    for si in range(n_slabs):
        plane_idx = plane_offset + si
        comp = args.generate_dir / f"composite_slab{si:02d}.npz"
        if not comp.exists():
            print(f"[yplane] WARNING: {comp} missing — skipping slab {si}")
            continue
        y = _yslab(comp, args.r200_factor, args.taper_frac)   # (N, N)

        # center-crop to lp_grid (must match bind-paint-lensplane exactly)
        n = y.shape[0]
        if n != args.lp_grid:
            if args.lp_grid > n:
                raise SystemExit(f"--lp_grid {args.lp_grid} > map size {n}")
            off = (n - args.lp_grid) // 2
            y = y[off:off + args.lp_grid, off:off + args.lp_grid]

        out = args.output_dir / f"yplane{plane_idx:02d}.dat"
        write_yplane(y, out)
        print(f"[yplane] plane {plane_idx:02d} (snap={args.lc_snap_idx} slab={si}) "
              f"y_max={y.max():.3e} -> {out.name}")

    print(f"[yplane] snapshot {args.lc_snap_idx} done.")


if __name__ == "__main__":
    main()
