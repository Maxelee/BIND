"""``bind-paint-generate`` (stage 2): GPU generation + compositing.

Consumes the intermediate written by ``bind-paint-project`` (stage 1): loads the
saved per-halo DMO cutouts, runs the flow-matching sampler on the GPU, composites
the painted patches back into per-slab maps, and writes the same
``composite_slab{NN}.npz`` / ``summary.json`` as ``bind-paint``.

Example::

    bind-paint-generate \\
        --stage1_dir stage1_out \\
        --run_dir weights/fm_two_head \\
        --output_dir bind_output/run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bind
from bind.cli.paint import _load_params
from bind.inference.paint_stages import generate_from_stage1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="Stage-1 intermediate directory (holds stage1_manifest.json)")
    p.add_argument("--params", type=Path, default=None,
                   help="Override the 35-dim parameter vector (.npy/.npz/.txt). "
                        "Lets one shared stage-1 dir feed many parameter runs "
                        "(the science orchestration path); defaults to the "
                        "stage-1 params.npy when omitted.")
    p.add_argument("--output_dir", type=Path, default=Path("bind_output"))

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--run_dir", type=Path, default=None,
                     help="Directory containing last.ckpt + norm_stats.npz")
    grp.add_argument("--checkpoint", type=Path, default=None,
                     help="Explicit checkpoint path (must accompany --norm_stats)")
    p.add_argument("--norm_stats", type=Path, default=None,
                   help="Explicit norm_stats path (used with --checkpoint)")

    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no_amp", action="store_true")

    rzgrp = p.add_mutually_exclusive_group()
    rzgrp.add_argument("--redshift", type=float, default=None,
                       help="Redshift to condition on (overrides manifest; for redshift-conditioned models)")
    rzgrp.add_argument("--scale_factor", type=float, default=None,
                       help="Scale factor a=1/(1+z) to condition on (overrides manifest)")

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
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for the sampler's initial noise, making the run "
                        "reproducible; omit for the historical behaviour: unseeded, "
                        "a different realization every run.")
    p.add_argument("--no_save_patches", action="store_true",
                   help="Skip saving per-halo generated patches in the output npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir is None and args.checkpoint is None:
        args.run_dir = Path("weights/fm_two_head")
    if args.checkpoint is not None and args.norm_stats is None:
        raise SystemExit("--norm_stats is required with --checkpoint")

    print(f"[bind-paint-generate] loading model on device={args.device}")
    if args.run_dir is not None:
        model = bind.Model.from_local(args.run_dir, device=args.device)
    else:
        model = bind.Model.from_files(args.checkpoint, args.norm_stats, device=args.device)

    params = _load_params(args.params) if args.params is not None else None

    result = generate_from_stage1(
        args.stage1_dir, model,
        output_dir=args.output_dir,
        params=params,
        redshift=args.redshift,
        scale_factor=args.scale_factor,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        patch_mass_match=not args.no_patch_mass_match,
        taper_frac=args.taper_frac,
        r200_factor=args.r200_factor,
        paste_mode=args.paste_mode,
        seed=args.seed,
        save_per_halo_patches=not args.no_save_patches,
    )

    print("=" * 80)
    print(f"bind-paint-generate complete: {result.n_halos} halos across {result.n_slabs} slab(s)")
    print(f"Output: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
