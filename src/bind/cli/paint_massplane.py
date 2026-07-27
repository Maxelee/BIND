"""``bind-paint-massplane`` (stage 3, kSZ-paper Task D companion): composite
total matter (DM+Gas+Star) -> lux tau-format planes.

Writes ``tauplane{PP:02d}.dat`` for one snapshot into a **dedicated** plane
directory (never the run's real ``lensplanes/``!), using the exact same binary
format and plane numbering as ``bind-paint-tauplane``. lux's ``tau_input_dir``
pathway is content-agnostic and, for kSZ/FRB tau, is a straight (unweighted,
un-deflected) LOS sum -- "Born projection ... exact to first order" per
`bind.inference.lightcone_maps`'s own docstring. Since the *same* per-snapshot
random shift geometry is shared by every field traced in one lux invocation, a
"total mass" field fed through this same pathway comes out on the identical
pixel grid as the real kSZ tau it is meant to accompany -- giving a physically
correct, pixel-aligned ``CAP_mat`` denominator for f~gas without needing any
change to lux itself.

The recomposited ``composite_slab{NN}.npz`` (produced by
``bind.inference.paint_stages.recomposite_from_saved``, since the SB35 Sobol
runs' on-disk composites only carry per-halo ``generated_patches``, not the
full grid) is still required as the input -- this script only changes *which*
channel(s) get summed relative to ``bind-paint-tauplane``.

    bind-paint-massplane --generate_dir /ceph/.../run_0000/snap_096 \\
        --output_dir /ceph/.../mass_lensplanes --lc_snap_idx 0 --lc_n_snaps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.lensplane import write_tauplane
from bind.inference.lightcone_maps import _tau_per_gas_pixel, SIGMA_T, X_E_PER_MASS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate_dir", type=Path, required=True,
                   help="Directory with (recomposited) composite_slab*.npz")
    p.add_argument("--stage1_dir", type=Path, default=None,
                   help="Stage-1 dir for the manifest (defaults to generate_dir/stage1)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="DEDICATED mass-plane directory (tauplane*.dat, NOT the real lensplanes/)")
    p.add_argument("--lc_snap_idx", type=int, required=True)
    p.add_argument("--lc_n_snaps", type=int, required=True)
    p.add_argument("--planes_per_snapshot", type=int, default=None)
    p.add_argument("--lp_grid", type=int, default=4096)
    return p.parse_args()


def _mass_col_slab(npz_path: Path, box_size: float, a_l: float) -> np.ndarray:
    """RAW total-matter (DM+Gas+Star) column, in the SAME physical units and
    per-plane area-conversion as `paint_tauplane._tauslab`'s gas column, so that
    CAP_gas / CAP_mat is a dimensionless ratio with the SIGMA_T*X_E_PER_MASS
    constant cancelling exactly (dividing the traced "mass-as-tau" output by
    that same constant recovers the physical column; the ratio needs no such
    division at all since it is common to both fields)."""
    d = np.load(npz_path)
    tot = d["composite"].sum(0).astype(np.float64)          # DM+Gas+Star [Msun/h] / pixel
    return _tau_per_gas_pixel(box_size, tot.shape[0], a_l) * tot


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = args.stage1_dir or (args.generate_dir / "stage1")
    manifest = json.loads((stage1_dir / "stage1_manifest.json").read_text())
    n_slabs = int(manifest["n_slabs"])
    box_size = float(manifest["box_size"])
    a_l = float(manifest["scale_factor"])
    pps = args.planes_per_snapshot or n_slabs
    if pps != n_slabs:
        raise SystemExit(f"--planes_per_snapshot={pps} must equal n_slabs={n_slabs}")

    plane_offset = args.lc_snap_idx * pps + 1  # lux numbers planes from 1 (matches paint_tauplane)
    for si in range(n_slabs):
        plane_idx = plane_offset + si
        comp = args.generate_dir / f"composite_slab{si:02d}.npz"
        if not comp.exists():
            print(f"[massplane] WARNING: {comp} missing — skipping slab {si}")
            continue
        mass = _mass_col_slab(comp, box_size, a_l)   # (N, N)

        n = mass.shape[0]
        if n != args.lp_grid:
            if args.lp_grid > n:
                raise SystemExit(f"--lp_grid {args.lp_grid} > map size {n}")
            off = (n - args.lp_grid) // 2
            mass = mass[off:off + args.lp_grid, off:off + args.lp_grid]

        out = args.output_dir / f"tauplane{plane_idx:02d}.dat"
        write_tauplane(mass, out)
        print(f"[massplane] plane {plane_idx:02d} (snap={args.lc_snap_idx} slab={si}) "
              f"mass_max={mass.max():.3e} -> {out.name}")

    print(f"[massplane] snapshot {args.lc_snap_idx} done.")


if __name__ == "__main__":
    main()
