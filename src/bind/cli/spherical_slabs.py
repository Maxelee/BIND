"""``bind-spherical-slabs``: build the spherical-BCM control tree for one snapshot.

Azimuthally symmetrizes every per-halo patch in a run's
``snap_<NNN>/composite_slab*.npz`` and writes a parallel single-channel
total-matter slab tree (see :mod:`bind.inference.spherical`).  Run as a SLURM
array over snapshots; then ``bind-lightcone-maps`` on the output tree (with
``--manifest_root`` pointed at the shared stage-1 manifests) produces ``kappa_sph``
on the *same* realization seeds as the BIND ``kappa`` from the original tree.

    bind-spherical-slabs --src_snap_root /ceph/bind_lightcone_tng \\
        --out_root /ceph/bind_science/wl_anisotropy/fid/sph --snapshot 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bind.inference.spherical import write_spherical_tree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_snap_root", type=Path, required=True,
                   help="Run tree holding snap_<NNN>/composite_slab*.npz to sphericalize.")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Output tree for the spherical slabs (snap_<NNN>/ mirror).")
    p.add_argument("--snapshot", type=int, required=True,
                   help="Snapshot number to process (one array task = one snapshot).")
    p.add_argument("--dmo_root", type=Path, default=None,
                   help="Shared DMO tree (fid bind_lightcone_tng) for runs whose "
                        "composites do not store 'dmo' (twobound/truth).")
    p.add_argument("--r200_factor", type=float, default=4.0,
                   help="Circular-taper aperture in R200c (must match the original paste).")
    p.add_argument("--taper_frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    diags = write_spherical_tree(
        args.src_snap_root, args.out_root, args.snapshot,
        dmo_root=args.dmo_root,
        r200_factor=args.r200_factor, taper_frac=args.taper_frac, verbose=True,
    )
    n = sum(int(x["n_halos"]) for x in diags)
    print(f"[spherical-slabs] snap {args.snapshot:03d}: {len(diags)} slabs, "
          f"{n} halos -> {args.out_root}/snap_{args.snapshot:03d}")


if __name__ == "__main__":
    main()
