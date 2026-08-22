"""``bind-truth-halos``: per-snapshot TNG300 hydro truth patches (MPI, CPU).

Projects the actual hydro fields (DM_hydro, Gas, Stars + compton_y/T/entropy/P_e)
at the same M>=10^13 DMO-FoF halos and lightcone transform BIND uses, writing the
BIND ``composite_slab{NN}.npz`` patch format.  Drop-in for the recomposite ->
lensplane -> lux -> stats pipeline (run with DESIGN=truth) to get a truth
lightcone on BIND's exact geometry.

    srun python -m bind.cli.truth_halos \\
        --hydro_snapshot /.../L205n2500TNG/output --snapshot_index 96 \\
        --dmo_group_catalog /.../L205n2500TNG_DM/output \\
        --transforms /ceph/bind_lightcone_tng/lightcone_transforms.json \\
        --transforms_snap_idx 0 --output_dir /ceph/bind_truth/snap_096
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bind
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.truth_lightcone import extract_truth_halos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hydro_snapshot", type=Path, required=True,
                   help="IllustrisTNG hydro output dir (has snapdir_<NNN>/)")
    p.add_argument("--dmo_group_catalog", type=Path, required=True,
                   help="DMO FoF/SUBFIND output dir (has groups_<NNN>/) — same halos BIND uses")
    p.add_argument("--snapshot_index", type=int, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--transforms", type=Path, default=None,
                   help="lightcone_transforms.json (same as stage 1)")
    p.add_argument("--transforms_snap_idx", type=int, default=None)
    p.add_argument("--halo_mass_min", type=float, default=1e13)
    p.add_argument("--halo_mass_field", type=str, default="Group_M_Crit200")
    p.add_argument("--pixel_size", type=float, default=bind.NATIVE_PIXEL_SIZE_MPCH)
    p.add_argument("--slab_depth", type=float, default=bind.NATIVE_SLAB_DEPTH_MPCH)
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def _get_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
            raise SystemExit("SLURM_NTASKS>1 but mpi4py not importable; module load openmpi python-mpi")
        return None


def main() -> None:
    args = parse_args()
    comm = _get_comm()
    transforms = None
    if args.transforms is not None:
        if args.transforms_snap_idx is None:
            raise SystemExit("--transforms_snap_idx is required with --transforms")
        transforms = LightconeTransforms.load(args.transforms)

    out = extract_truth_halos(
        args.hydro_snapshot, args.dmo_group_catalog,
        output_dir=args.output_dir, snapshot_index=args.snapshot_index,
        transforms=transforms, transforms_snap_idx=args.transforms_snap_idx,
        halo_mass_min=args.halo_mass_min, halo_mass_field=args.halo_mass_field,
        pixel_size=args.pixel_size, slab_depth=args.slab_depth,
        comm=comm, progress=not args.no_progress,
    )
    if out is not None:
        print(f"=== truth halos -> {out} ===")


if __name__ == "__main__":
    main()
