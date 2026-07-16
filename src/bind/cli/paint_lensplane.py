"""``bind-paint-lensplane`` (stage 3): BIND composite maps → lux lensplane files.

Reads the ``composite_slab*.npz`` files written by ``bind-paint-generate`` (stage 2)
for a **single snapshot** and writes lux-compatible ``lenspot{PP:02d}.dat`` files
plus a global ``config.dat``.  The chain for the full lightcone is:

  1. (stage 1) ``bind-paint-project``   — project particles + extract halo cutouts
  2. (stage 2) ``bind-paint-generate``  — run BIND flow-matching model on halos
  3. **(stage 3) ``bind-paint-lensplane``** — convert composite mass maps to lensplanes
  4. ``lux``                             — raytrace through the stacked lensplanes

The ``config.dat`` is written only by the **first snapshot in the lightcone**
(``--lc_snap_idx 0``) so all snapshots share one global geometry file.  All
other snapshots only write their ``lenspot{PP}.dat`` files.

Baryonic mass used for lensing: total projected mass = DM_hydro + Gas + Stars.
The thermo channels (compton_y, T, entropy, P_e) are *not* used here.

Example (for a 20-snapshot lightcone, 4 planes per snapshot = 80 planes total)::

    bind-paint-lensplane \\
        --generate_dir  /ceph/bind_lightcone_tng/snap_096 \\
        --stage1_dir    /ceph/bind_lightcone_tng/snap_096/stage1 \\
        --output_dir    /ceph/bind_lightcone_tng/lensplanes \\
        --transforms    /ceph/bind_lightcone_tng/lightcone_transforms.json \\
        --lc_snap_idx   0 \\
        --lc_n_snaps    20 \\
        --all_snap_dirs /ceph/bind_lightcone_tng/snap_096 \\
                        /ceph/bind_lightcone_tng/snap_090 ... (all 20)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.lensplane import (
    mass_map_to_delta_scaled,
    density_to_lensplane,
    write_lensplane,
    write_lux_config,
    build_lightcone_geometry,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate_dir", type=Path, default=None,
                   help="Directory containing composite_slab*.npz from stage 2 "
                        "(not needed with --dmo)")
    p.add_argument("--dmo", action="store_true",
                   help="Build DMO-only lensplanes from the stage-1 DMO maps "
                        "(no baryons) — for the ray-traced baryonic-suppression baseline.")
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="Stage-1 intermediate directory (for the manifest + box metadata)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for lenspot*.dat (and config.dat if snap_idx==0)")
    p.add_argument("--transforms", type=Path, required=True,
                   help="Path to lightcone_transforms.json (all snapshots)")
    p.add_argument("--lc_snap_idx", type=int, required=True,
                   help="0-based index of this snapshot in the lightcone "
                        "(0 = lowest-z / closest snapshot)")
    p.add_argument("--lc_n_snaps", type=int, required=True,
                   help="Total number of snapshots in the lightcone")
    p.add_argument("--planes_per_snapshot", type=int, default=None,
                   help="Number of lensplanes per snapshot.  Defaults to the "
                        "number of slabs in the stage-1 manifest (n_slabs).")
    p.add_argument("--all_snap_dirs", type=Path, nargs="+", default=None,
                   help="All snapshot generate_dirs in low-z order "
                        "(needed when writing config.dat, i.e. --lc_snap_idx 0).  "
                        "Each entry is the generate_dir for that snapshot.")
    p.add_argument("--snap_stack", type=str, default=None,
                   help="Comma-separated booleans (true/false) indicating whether "
                        "each snapshot's box is stacked (2x transverse). "
                        "Defaults to 'false' for all snapshots.")
    p.add_argument("--lp_grid", type=int, default=4096,
                   help="Lensplane grid size written to disk.  The mass map is "
                        "center-cropped to lp_grid×lp_grid before the Poisson FFT "
                        "so lux sees a power-of-2 grid (default: 4096).  Must be "
                        "<= the native map pixel count.")
    return p.parse_args()


def _read_manifest(stage1_dir: Path) -> dict:
    return json.loads((stage1_dir / "stage1_manifest.json").read_text())


def _total_mass_map(composite_npz: Path) -> np.ndarray:
    """Sum DM_hydro + Gas + Stars channels from a composite_slab*.npz.

    The BIND composite array has shape (3, N, N) or is split into per-channel
    maps.  We return the sum (total projected mass in Msun/h).
    """
    d = np.load(composite_npz)
    composite = d["composite"]   # (3, N, N): [DM_hydro, Gas, Stars]
    return composite.sum(axis=0).astype(np.float64)  # (N, N)


def _dmo_mass_map(stage1_dir: Path, si: int) -> np.ndarray:
    """DMO (no-baryon) total-matter map for slab ``si`` from stage 1."""
    d = np.load(stage1_dir / f"stage1_slab{si:02d}.npz")
    key = "dmo_aa" if "dmo_aa" in d.files else "dmo"
    return d[key].astype(np.float64)


def main() -> None:
    args = parse_args()
    if not args.dmo and args.generate_dir is None:
        raise SystemExit("--generate_dir is required (unless --dmo)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest for this snapshot ───────────────────────────────────────
    manifest = _read_manifest(args.stage1_dir)
    box_size = float(manifest["box_size"])
    n_slabs = int(manifest["n_slabs"])
    slab_depth = float(manifest["slab_depth"])
    Omega_m = float(manifest["Omega_m"])
    scale_factor = float(manifest["scale_factor"])
    pps = args.planes_per_snapshot if args.planes_per_snapshot is not None else n_slabs

    if pps != n_slabs:
        raise SystemExit(
            f"--planes_per_snapshot={pps} must equal n_slabs={n_slabs} from the "
            "stage-1 manifest (each BIND slab becomes one lensplane)."
        )

    # ── Load transforms ────────────────────────────────────────────────────────
    transforms = LightconeTransforms.load(args.transforms)

    # ── Write lensplanes for this snapshot ────────────────────────────────────
    # Lensplane plane index: first plane of snapshot s is at global index s*pps+1
    plane_offset = args.lc_snap_idx * pps + 1  # lux numbers planes from 1

    for si in range(n_slabs):
        plane_idx = plane_offset + si
        if args.dmo:
            mass_map = _dmo_mass_map(args.stage1_dir, si)        # DMO-only lightcone
        else:
            composite_path = args.generate_dir / f"composite_slab{si:02d}.npz"
            if not composite_path.exists():
                print(f"[lensplane] WARNING: {composite_path} not found — skipping slab {si}")
                continue
            mass_map = _total_mass_map(composite_path)   # (N, N) Msun/h

        # Center-crop to lp_grid × lp_grid so lux sees a power-of-2 FFT grid.
        # box_size scales proportionally to keep pixel_size exact.
        lp_grid = args.lp_grid
        if mass_map.shape[0] != lp_grid:
            n = mass_map.shape[0]
            if lp_grid > n:
                raise SystemExit(f"--lp_grid {lp_grid} > map size {n}")
            off = (n - lp_grid) // 2
            mass_map = mass_map[off:off + lp_grid, off:off + lp_grid]
            box_size_eff = box_size * lp_grid / n
        else:
            box_size_eff = box_size

        delta_scaled = mass_map_to_delta_scaled(
            mass_map, box_size=box_size_eff, slab_depth=slab_depth, Omega_m=Omega_m
        )
        phi = density_to_lensplane(delta_scaled, box_size=box_size_eff, Omega_m=Omega_m)

        out_path = args.output_dir / f"lenspot{plane_idx:02d}.dat"
        write_lensplane(phi, out_path)
        print(f"[lensplane] plane {plane_idx:02d} (snap={args.lc_snap_idx} slab={si}) "
              f"→ {out_path.name}  "
              f"(Omega_m={Omega_m:.4f} slab_depth={slab_depth:.1f} Mpc/h)")

    # ── Write config.dat (only for the first / lowest-z snapshot) ─────────────
    if args.lc_snap_idx == 0:
        if args.all_snap_dirs is None:
            raise SystemExit(
                "--all_snap_dirs is required when --lc_snap_idx 0 (to build config.dat)"
            )
        if len(args.all_snap_dirs) != args.lc_n_snaps:
            raise SystemExit(
                f"Expected {args.lc_n_snaps} entries in --all_snap_dirs, "
                f"got {len(args.all_snap_dirs)}"
            )

        # Gather per-snapshot metadata (scale_factor, Ll, Lt, snap_stack)
        Ll_arr = []
        Lt_arr = []
        a_snaps = []
        for snap_dir in args.all_snap_dirs:
            m = _read_manifest(snap_dir / "stage1")
            Ll_arr.append(float(m["box_size"]))
            # Transverse box size: check if stacking is requested
            snap_stack_flags = []
            if args.snap_stack is not None:
                snap_stack_flags = [
                    s.strip().lower() == "true" for s in args.snap_stack.split(",")
                ]
            stack = (snap_stack_flags[len(a_snaps)]
                     if snap_stack_flags else False)
            Lt_arr.append(float(m["box_size"]) * (2.0 if stack else 1.0))
            a_snaps.append(float(m["scale_factor"]))

        geom = build_lightcone_geometry(
            snapshot_scale_factors=a_snaps,
            Ll=Ll_arr,
            Lt=Lt_arr,
            planes_per_snapshot=pps,
            Omega_m=Omega_m,
        )

        # snap_stack booleans (needed for Lt in config.dat — already applied above)
        # transforms: (Ns, 3) disp and flip
        snap_stack_arr = []
        if args.snap_stack is not None:
            snap_stack_arr = [s.strip().lower() == "true"
                              for s in args.snap_stack.split(",")]
        else:
            snap_stack_arr = [False] * args.lc_n_snaps

        config_path = args.output_dir / "config.dat"
        write_lux_config(
            config_path,
            chi=geom["chi"],
            a=geom["a"],
            chi_out=geom["chi_out"],
            Ll=np.array(Ll_arr),
            Lt=np.array(Lt_arr),
            proj_dirs=transforms.proj_dirs,
            disp=transforms.disp,
            flip=transforms.flip,
        )
        print(f"[lensplane] wrote config.dat → {config_path}")
        print(f"[lensplane] lightcone: {args.lc_n_snaps} snapshots × {pps} planes "
              f"= {args.lc_n_snaps * pps} total planes")

    print(f"[lensplane] snapshot {args.lc_snap_idx} done.")


if __name__ == "__main__":
    main()
