"""``bind-paint-project`` (stage 1): MPI projection + halo cutouts on CPU.

The memory-heavy half of painting a large box, split out so it runs across many
CPU nodes via MPI instead of OOMing one node.  Each rank reads only its subset
of snapshot chunks, accumulates them into the small z-slab maps, and the partial
maps are reduced onto rank 0, which then extracts per-halo DMO cutouts and writes
the ``stage1_*`` intermediate consumed by ``bind-paint-generate`` (stage 2).

Launch under SLURM/MPI::

    srun python -m bind.cli.paint_project \\
        --snapshot snapdir_099 --group_catalog groups_099 --snapshot_index 99 \\
        --params fiducial_params.npy --output_dir stage1_out

Runs serially (and still streams chunk-by-chunk, so still avoids the OOM) when
launched as a single task / without mpi4py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import bind
from bind.cli.paint import _load_params
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint_stages import project_and_extract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True,
                   help="Gadget/Arepo HDF5 snapshot file, directory, or glob")
    p.add_argument("--group_catalog", type=Path, required=True,
                   help="FOF/SUBFIND group catalog file, directory, or glob")
    p.add_argument("--snapshot_index", type=int, default=None,
                   help="Snapshot index (required when paths are directories)")
    p.add_argument("--params", type=Path, required=True,
                   help="Path to .npy/.npz/.txt holding the 35-dim parameter vector")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Stage-1 intermediate directory")

    p.add_argument("--halo_mass_min", type=float, default=1e13,
                   help="Minimum M200c (Msun/h) for halo selection")
    p.add_argument("--halo_mass_field", type=str, default="Group_M_Crit200")
    p.add_argument("--pixel_size", type=float, default=bind.NATIVE_PIXEL_SIZE_MPCH)
    p.add_argument("--slab_depth", type=float, default=bind.NATIVE_SLAB_DEPTH_MPCH)
    p.add_argument("--patch_pix", type=int, default=bind.PATCH_PIX)
    p.add_argument("--mas_correct", action="store_true",
                   help="Also store an anti-aliased (interlaced + CIC-deconvolved) "
                        "DMO map 'dmo_aa' per slab, used as the composite/lensplane "
                        "background. Model condition cutouts stay raw-CIC. Use when "
                        "matching an anti-aliased reference (e.g. kappaTNG); doubles "
                        "stage-1 projection cost.")
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--allow_cosmology_mismatch", action="store_true",
                   help="Permit a params vector whose Omega_m disagrees with the "
                        "snapshot header's Omega0.  By default stage 1 refuses: "
                        "conditioning a substrate on another suite's cosmology "
                        "silently mis-paints the baryons (use bind.tng300_params() "
                        "for IllustrisTNG, bind.fiducial_params() for CAMELS).")

    # Lightcone transform support
    p.add_argument("--transforms", type=Path, default=None,
                   help="Path to lightcone_transforms.json written by "
                        "bind-lightcone-transforms.  When provided, particles "
                        "are rotated/translated before projection.")
    p.add_argument("--transforms_snap_idx", type=int, default=None,
                   help="Index into the transforms file for this snapshot "
                        "(0 = lowest-z snapshot).  Required with --transforms.")
    return p.parse_args()


def _get_comm() -> object | None:
    """Return the MPI communicator, or None for a serial run.

    Guards the footgun where SLURM launches many tasks but mpi4py is missing:
    that would run N independent full projections all writing the same files.
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        if ntasks > 1:
            raise SystemExit(
                f"Launched with SLURM_NTASKS={ntasks} but mpi4py is not importable.\n"
                "Install it into the venv built against the cluster MPI:\n"
                "    module load openmpi && pip install --no-binary :all: mpi4py\n"
                "or run stage 1 as a single task (still streams chunk-by-chunk)."
            )
        return None


def main() -> None:
    args = parse_args()
    comm = _get_comm()
    rank = comm.rank if comm is not None else 0

    params = _load_params(args.params)

    transforms = None
    if args.transforms is not None:
        if args.transforms_snap_idx is None:
            raise SystemExit("--transforms_snap_idx is required with --transforms")
        transforms = LightconeTransforms.load(args.transforms)

    if rank == 0:
        size = comm.size if comm is not None else 1
        print(f"[bind-paint-project] {size} rank(s); output -> {args.output_dir}")

    out = project_and_extract(
        snapshot=args.snapshot,
        group_catalog=args.group_catalog,
        params=params,
        output_dir=args.output_dir,
        snapshot_index=args.snapshot_index,
        halo_mass_min=args.halo_mass_min,
        halo_mass_field=args.halo_mass_field,
        pixel_size=args.pixel_size,
        slab_depth=args.slab_depth,
        patch_pix=args.patch_pix,
        transforms=transforms,
        transforms_snap_idx=args.transforms_snap_idx,
        mas_correct=args.mas_correct,
        allow_cosmology_mismatch=args.allow_cosmology_mismatch,
        comm=comm,
        progress=not args.no_progress,
    )

    if rank == 0:
        print("=" * 80)
        print(f"stage 1 complete -> {out}")
        print("Next: bind-paint-generate --stage1_dir "
              f"{args.output_dir} --run_dir weights/fm_two_head --output_dir <final>")
        print("=" * 80)


if __name__ == "__main__":
    main()
