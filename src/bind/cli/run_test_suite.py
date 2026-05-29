"""CLI entrypoint for multi-suite DMO->hydro generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.test_suite.artifacts import to_jsonable
from bind.test_suite.config import build_suite_specs, parse_sim_ids
from bind.test_suite.runner import run_suite
from bind.test_suite.schemas import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CV/1P/Test halo generation with notebook-consistent procedure."
    )

    parser.add_argument("--suite", choices=["cv", "1p", "test", "sb35", "all"], default="cv")
    parser.add_argument("--sim_ids", type=str, default=None, help="Comma-separated ids for a single suite")

    parser.add_argument("--n_chunks", type=int, default=1,
                        help="Split the spec list into N chunks and process only one (for SLURM array jobs)")
    parser.add_argument("--chunk_id", type=int, default=0,
                        help="Which chunk to process (0-indexed, used with --n_chunks)")

    parser.add_argument("--snapshot", type=int, default=90)
    parser.add_argument("--box_size", type=float, default=50.0)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--patch_pix", type=int, default=128)
    parser.add_argument("--proj_frac", type=float, default=1.0)
    parser.add_argument("--halo_mass_min", type=float, default=1e13)

    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Run directory containing norm_stats.npz (and checkpoints/last.ckpt unless --checkpoint_path is given)")
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, required=True,
                        help="Where evaluation outputs are written")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Subfolder name under output_root for this run's outputs")

    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", help="auto, cuda, cpu")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--channel_correction",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-channel multiplicative correction factors "
            "(truth/gen), e.g. '1.00,1.02,1.38'. Applied via target_mean shift before inference."
        ),
    )

    parser.add_argument("--prep_only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--regenerate_all", action="store_true")
    parser.add_argument("--repaste", action="store_true")
    parser.add_argument("--no_patch_mass_match", action="store_true")
    parser.add_argument("--taper_frac", type=float, default=0.15)
    parser.add_argument(
        "--r200_factor",
        type=float,
        default=0.0,
        help=(
            "Radius of circular paste region as a multiple of R200c. "
            "0 (default) uses the legacy square taper; 2.0 pastes within 2×R200c."
        ),
    )
    parser.add_argument("--skip_truth", action="store_true", help="Skip hydro truth map projection")

    parser.add_argument("--max_workers", type=int, default=1)

    parser.add_argument("--cv_param_file", type=Path, default=None,
                        help="CAMELS CV parameter file (CosmoAstroSeed_*_CV.txt)")
    parser.add_argument("--cv_nbody_root", type=Path, default=None,
                        help="CAMELS CV DM-only sims root")
    parser.add_argument("--cv_hydro_root", type=Path, default=None,
                        help="CAMELS CV hydro sims root")
    parser.add_argument("--cv_fof_root", type=Path, default=None,
                        help="CAMELS CV FOF/Subfind root")

    parser.add_argument("--onep_param_file", type=Path, default=None)
    parser.add_argument("--onep_nbody_root", type=Path, default=None)
    parser.add_argument("--onep_hydro_root", type=Path, default=None)
    parser.add_argument("--onep_fof_root", type=Path, default=None)

    parser.add_argument(
        "--test_manifest",
        type=Path,
        default=None,
        help="JSON manifest path for suite=test entries",
    )

    parser.add_argument("--sb35_param_file", type=Path, default=None)
    parser.add_argument("--sb35_nbody_root", type=Path, default=None)
    parser.add_argument("--sb35_hydro_root", type=Path, default=None)
    parser.add_argument("--sb35_fof_root", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_path or (args.run_dir / "checkpoints" / "last.ckpt")

    correction = None
    if args.channel_correction:
        parts = [p.strip() for p in args.channel_correction.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                "--channel_correction must provide exactly 3 comma-separated values "
                "for gas, dm, stars channels"
            )
        correction = np.asarray([float(p) for p in parts], dtype=np.float32)
        if not np.all(np.isfinite(correction)) or np.any(correction <= 0):
            raise ValueError("--channel_correction values must be finite and > 0")

    sim_ids = parse_sim_ids(args.sim_ids)
    specs = build_suite_specs(
        suite=args.suite,
        sim_ids=sim_ids,
        snapshot=args.snapshot,
        box_size=args.box_size,
        npix=args.npix,
        patch_pix=args.patch_pix,
        proj_frac=args.proj_frac,
        halo_mass_min=args.halo_mass_min,
        cv_param_file=args.cv_param_file,
        cv_nbody_root=args.cv_nbody_root,
        cv_hydro_root=args.cv_hydro_root,
        cv_fof_root=args.cv_fof_root,
        onep_param_file=args.onep_param_file,
        onep_nbody_root=args.onep_nbody_root,
        onep_hydro_root=args.onep_hydro_root,
        onep_fof_root=args.onep_fof_root,
        test_manifest=args.test_manifest,
        sb35_param_file=args.sb35_param_file,
        sb35_nbody_root=args.sb35_nbody_root,
        sb35_hydro_root=args.sb35_hydro_root,
        sb35_fof_root=args.sb35_fof_root,
    )

    if args.n_chunks > 1:
        if not (0 <= args.chunk_id < args.n_chunks):
            raise ValueError(f"--chunk_id {args.chunk_id} out of range for --n_chunks {args.n_chunks}")
        chunk_size = (len(specs) + args.n_chunks - 1) // args.n_chunks
        specs = specs[args.chunk_id * chunk_size : (args.chunk_id + 1) * chunk_size]

    run_cfg = RunConfig(
        run_dir=args.run_dir,
        checkpoint_path=checkpoint,
        output_root=args.output_root,
        model_name=args.model_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        patch_mass_match=not args.no_patch_mass_match,
        taper_frac=args.taper_frac,
        r200_factor=args.r200_factor,
        use_amp=not args.no_amp,
        device=args.device,
        prep_only=args.prep_only,
        regenerate=args.regenerate,
        regenerate_all=args.regenerate_all,
        repaste=args.repaste,
        channel_correction=correction,
    )

    print("=" * 80)
    print("TEST SUITE RUNNER")
    print("=" * 80)
    print(f"Suite: {args.suite}")
    if args.n_chunks > 1:
        print(f"Chunk: {args.chunk_id + 1}/{args.n_chunks}  ({len(specs)} sims)")
    else:
        print(f"Simulations: {len(specs)}")
    print(f"Model run dir: {run_cfg.run_dir}")
    print(f"Checkpoint: {run_cfg.checkpoint_path}")
    print(f"Output root: {run_cfg.output_root}")
    print(f"Prep only: {run_cfg.prep_only}")
    print(f"Regenerate: {run_cfg.regenerate}")
    print(f"Regenerate all: {run_cfg.regenerate_all}")
    print(f"Repaste: {run_cfg.repaste}")
    if run_cfg.channel_correction is not None:
        print(f"Channel correction (truth/gen): {run_cfg.channel_correction.tolist()}")
    if args.suite in {"sb35", "all"}:
        print(f"SB35 hydro root: {args.sb35_hydro_root}")
        print(f"SB35 FoF root:   {args.sb35_fof_root}")
    print("=" * 80)

    summaries = run_suite(
        specs,
        run_cfg,
        load_truth=not args.skip_truth,
        max_workers=max(1, args.max_workers),
    )

    out_summary = {
        "suite": args.suite,
        "n_simulations_requested": len(specs),
        "n_simulations_completed": len(summaries),
        "prep_only": args.prep_only,
        "simulations": summaries,
    }

    summary_path = args.output_root / f"run_summary_{args.suite}.json"
    summary_path.write_text(json.dumps(to_jsonable(out_summary), indent=2, sort_keys=True))
    print(f"Saved run summary to {summary_path}")


if __name__ == "__main__":
    main()
