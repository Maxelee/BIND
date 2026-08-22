"""``bind-extract-halos``: slim per-halo arrays from composite slabs.

Writes ``halos.npz`` (the *generated halos* — per-halo patches + metadata) for a
snapshot, dropping the large 4096²-class maps (``composite``, ``dmo``, ``alpha``,
``composite_thermo``).  Used by the per-run science driver so the heavy composite
slabs can be deleted after the lensplanes are built, while the regenerable-but-
expensive halo patches are kept.

    bind-extract-halos --generate_dir run_0000/snap_096 --out run_0000/snap_096/halos.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# per-halo arrays worth keeping (everything that is NOT a full-box map)
_HALO_KEYS = ("generated_patches", "thermo_patches", "halo_centers",
              "halo_masses", "halo_r200", "condition_sums", "patch_scales",
              "n_halos", "slab_idx", "n_slabs", "box_size")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate_dir", type=Path, required=True,
                   help="Directory with composite_slab*.npz")
    p.add_argument("--out", type=Path, default=None,
                   help="Output halos.npz (default: <generate_dir>/halos.npz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    slabs = sorted(args.generate_dir.glob("composite_slab*.npz"))
    if not slabs:
        raise SystemExit(f"no composite_slab*.npz in {args.generate_dir}")
    out = args.out or (args.generate_dir / "halos.npz")
    bundle: dict[str, np.ndarray] = {}
    for sp in slabs:
        si = int(sp.stem.split("slab")[1])
        d = np.load(sp)
        for k in _HALO_KEYS:
            if k in d.files:
                bundle[f"slab{si:02d}_{k}"] = d[k]
    np.savez_compressed(out, **bundle)
    n = sum(int(v) for k, v in bundle.items() if k.endswith("n_halos"))
    print(f"[extract-halos] {len(slabs)} slabs, {n} halos -> {out}")


if __name__ == "__main__":
    main()
