"""``bind-paint-recomposite`` (stage 3): re-composite saved patches, no GPU.

The flow-matching sampler (stage 2) is the expensive part and its per-halo output
is already saved in each ``composite_slab{NN}.npz`` (``generated_patches``). This
re-runs only the compositing step with new blend settings — ``--taper_frac``,
``--r200_factor`` (circular R200 aperture), ``--paste_mode``, ``--no_patch_mass_match`` — reusing
those patches plus the stage-1 cutouts. No model, no GPU.

Example (square-taper originals -> circular R200 paste)::

    bind-paint-recomposite \\
        --stage1_dir   /…/snap_099/stage1 \\
        --generated_dir /…/snap_099 \\
        --output_dir    /…/snap_099_r200paste \\
        --r200_factor 4.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bind.inference.paint_stages import recomposite_from_saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="Stage-1 intermediate directory (holds stage1_manifest.json + cutouts)")
    p.add_argument("--generated_dir", type=Path, default=None,
                   help="Directory with composite_slab*.npz holding generated_patches "
                        "(default: parent of --stage1_dir, i.e. the stage-2 output dir)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Destination for the re-composited maps (use a NEW dir)")

    p.add_argument("--no_patch_mass_match", action="store_true")
    p.add_argument("--taper_frac", type=float, default=0.15)
    p.add_argument("--r200_factor", type=float, default=4.0,
                   help="Circular paste radius as a multiple of R200c. The default 4.0 "
                        "is the standard; 0 selects the legacy square taper, which loses "
                        "high-k power wherever apertures overlap "
                        "(see docs/circular_aperture.md).")
    p.add_argument("--paste_mode", choices=["shared", "average"], default="shared",
                   help="Overlap handling. 'shared' (the standard) makes overlapping "
                        "halos agree on one realization of the shared region; 'average' "
                        "is the legacy independent-patch blend, measured at -10.6%% "
                        "total-matter P(k) at k=40-70. Requires --r200_factor > 0.")
    p.add_argument("--no_save_patches", action="store_true",
                   help="Don't carry generated patches into the new output "
                        "(makes it non-re-compositable)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed of the generation being re-composited, recorded in the new "
                        "output's provenance. Re-compositing itself draws no noise, so this "
                        "does not change the result; it is only needed when the source "
                        "patches predate provenance stamping (otherwise the source run's "
                        "seed is inherited automatically).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir or args.stage1_dir.parent

    result = recomposite_from_saved(
        args.stage1_dir, generated_dir, args.output_dir,
        patch_mass_match=not args.no_patch_mass_match,
        taper_frac=args.taper_frac,
        r200_factor=args.r200_factor,
        paste_mode=args.paste_mode,
        save_per_halo_patches=not args.no_save_patches,
        seed=args.seed,
    )

    print("=" * 80)
    print(f"recomposite complete: {result.n_halos} halos across {result.n_slabs} slab(s)")
    print(f"Output: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
