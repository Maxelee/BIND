"""``bind-paint`` CLI: generic DMO snapshot + halo catalog -> baryonified maps.

This is the primary user-facing entry point.  For running across the full
CAMELS test suites (CV / 1P / SB35) instead, use ``bind-camels-suite``.

Example::

    bind-paint \\
        --snapshot path/to/snap_090.hdf5 \\
        --group_catalog path/to/fof_subhalo_tab_090.hdf5 \\
        --params my_params.npy \\
        --run_dir weights/fm_two_head \\
        --output_dir bind_output/run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bind


def _load_params(path: Path) -> np.ndarray:
    """Load a 35-dim parameter vector from .npy/.npz/.txt."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        d = np.load(path)
        keys = list(d.keys())
        if "params" in keys:
            arr = d["params"]
        elif len(keys) == 1:
            arr = d[keys[0]]
        else:
            raise ValueError(f"{path} contains {keys}; expected key 'params' or single array")
    elif suffix in (".txt", ".csv"):
        arr = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported params file extension: {path.suffix}")
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[-1] != 35:
        raise ValueError(f"params must be 35-dim, got shape {arr.shape}")
    return arr.astype(np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--snapshot", type=Path, required=True,
                   help="Gadget/Arepo HDF5 snapshot file or directory")
    p.add_argument("--group_catalog", type=Path, required=True,
                   help="FOF/SUBFIND group catalog file or directory")
    p.add_argument("--snapshot_index", type=int, default=None,
                   help="Snapshot index (required when paths are directories)")

    p.add_argument("--params", type=Path, required=True,
                   help="Path to .npy/.npz/.txt holding the 35-dim parameter vector")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--run_dir", type=Path, default=None,
                     help="Directory containing last.ckpt + norm_stats.npz")
    grp.add_argument("--checkpoint", type=Path, default=None,
                    help="Explicit checkpoint path (must accompany --norm_stats)")
    p.add_argument("--norm_stats", type=Path, default=None,
                   help="Explicit norm_stats path (used with --checkpoint)")

    p.add_argument("--output_dir", type=Path, default=Path("bind_output"))

    p.add_argument("--halo_mass_min", type=float, default=1e13,
                   help="Minimum M200c (Msun/h) for halo selection")
    p.add_argument("--pixel_size", type=float, default=bind.NATIVE_PIXEL_SIZE_MPCH,
                   help=f"Pixel size in Mpc/h (default native: {bind.NATIVE_PIXEL_SIZE_MPCH:.4f})")
    p.add_argument("--slab_depth", type=float, default=bind.NATIVE_SLAB_DEPTH_MPCH,
                   help=f"z-slab depth in Mpc/h (default native: {bind.NATIVE_SLAB_DEPTH_MPCH})")

    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--no_patch_mass_match", action="store_true")
    p.add_argument("--taper_frac", type=float, default=0.15)
    p.add_argument("--r200_factor", type=float, default=0.0,
                   help="Circular paste radius as multiple of R200c (0 = square taper)")
    p.add_argument("--no_save_patches", action="store_true",
                   help="Skip saving per-halo generated patches in the output npz")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir is None and args.checkpoint is None:
        args.run_dir = Path("weights/fm_two_head")
    if args.checkpoint is not None and args.norm_stats is None:
        raise SystemExit("--norm_stats is required with --checkpoint")

    params = _load_params(args.params)

    print(f"[bind-paint] loading simulation: {args.snapshot}")
    sim = bind.Simulation.from_paths(
        snapshot=args.snapshot,
        group_catalog=args.group_catalog,
        snapshot_index=args.snapshot_index,
        halo_mass_min=args.halo_mass_min,
    )
    print(f"[bind-paint] {sim!r}")

    print(f"[bind-paint] loading model on device={args.device}")
    if args.run_dir is not None:
        model = bind.Model.from_local(args.run_dir, device=args.device)
    else:
        model = bind.Model.from_files(args.checkpoint, args.norm_stats, device=args.device)
    print(f"[bind-paint] {model!r}")

    result = bind.paint(
        sim, model, params=params,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        slab_depth=args.slab_depth,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        patch_mass_match=not args.no_patch_mass_match,
        taper_frac=args.taper_frac,
        r200_factor=args.r200_factor,
        save_per_halo_patches=not args.no_save_patches,
    )

    print("=" * 80)
    print(f"bind-paint complete: {result.n_halos} halos across {result.n_slabs} slab(s)")
    print(f"Output: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
