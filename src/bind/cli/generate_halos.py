"""``bind-generate-halos``: GPU generation only (portable, no compositing).

Runs the flow-matching sampler on stage-1 cutouts and saves only the per-halo
patches (``composite_slab{NN}.npz`` with ``generated_patches`` + ``thermo_patches``
+ halo metadata).  Needs only the cutouts (``condition`` / ``large_scale``), so a
stripped stage-1 can be shipped to a GPU-rich machine, generated there, and the
halos shipped back; compositing (DMO background) runs CPU-side with
``bind-paint-recomposite``.

    bind-generate-halos --stage1_dir snap_096/stage1 --params run_0000/params.npy \\
        --run_dir weights/fm_redshift_thermo --output_dir run_0000/snap_096
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bind
from bind.cli.paint import _load_params
from bind.inference.paint_stages import generate_halos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="Stage-1 dir (cutouts + manifest); DMO maps not required")
    p.add_argument("--params", type=Path, default=None,
                   help="35-dim parameter vector (.npy/.npz/.txt); else stage-1 params.npy")
    p.add_argument("--output_dir", type=Path, required=True)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--run_dir", type=Path, default=None)
    grp.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--norm_stats", type=Path, default=None)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no_amp", action="store_true")
    rz = p.add_mutually_exclusive_group()
    rz.add_argument("--redshift", type=float, default=None)
    rz.add_argument("--scale_factor", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None and args.checkpoint is None:
        args.run_dir = Path("weights/fm_redshift_thermo")
    if args.checkpoint is not None and args.norm_stats is None:
        raise SystemExit("--norm_stats is required with --checkpoint")
    if args.run_dir is not None:
        model = bind.Model.from_local(args.run_dir, device=args.device)
    else:
        model = bind.Model.from_files(args.checkpoint, args.norm_stats, device=args.device)

    params = _load_params(args.params) if args.params is not None else None
    out = generate_halos(
        args.stage1_dir, model, output_dir=args.output_dir, params=params,
        redshift=args.redshift, scale_factor=args.scale_factor,
        n_steps=args.n_steps, batch_size=args.batch_size, use_amp=not args.no_amp,
    )
    print(f"[bind-generate-halos] halos -> {out}")


if __name__ == "__main__":
    main()
