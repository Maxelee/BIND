"""CLI entrypoint for multi-suite DMO->hydro generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from test_suite.config import build_suite_specs, parse_sim_ids
from test_suite.runner import run_suite
from test_suite.schemas import RunConfig


def _to_jsonable(value):
    """Recursively convert Path/numpy objects to JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CV/1P/Test halo generation with notebook-consistent procedure."
    )

    parser.add_argument("--suite", choices=["cv", "1p", "test", "all"], default="cv")
    parser.add_argument("--sim_ids", type=str, default=None, help="Comma-separated ids for a single suite")

    parser.add_argument("--snapshot", type=int, default=90)
    parser.add_argument("--box_size", type=float, default=50.0)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--patch_pix", type=int, default=128)
    parser.add_argument("--proj_frac", type=float, default=1.0)
    parser.add_argument("--halo_mass_min", type=float, default=1e13)

    parser.add_argument("--run_dir", type=Path, default=Path("/mnt/home/mlee1/ceph/fm_runs/fm_base"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=Path("/mnt/home/mlee1/ceph/fm_eval"))
    parser.add_argument("--model_name", type=str, default="fm_base")

    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--prep_only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--regenerate_all", action="store_true")
    parser.add_argument("--repaste", action="store_true")
    parser.add_argument("--no_patch_mass_match", action="store_true")
    parser.add_argument("--taper_frac", type=float, default=0.15)
    parser.add_argument("--skip_truth", action="store_true", help="Skip hydro truth map projection")

    parser.add_argument("--max_workers", type=int, default=1)

    parser.add_argument(
        "--cv_param_file",
        type=Path,
        default=Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/CV/CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt"),
    )
    parser.add_argument(
        "--cv_nbody_root",
        type=Path,
        default=Path("/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/CV"),
    )
    parser.add_argument(
        "--cv_hydro_root",
        type=Path,
        default=Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/CV"),
    )

    parser.add_argument(
        "--onep_param_file",
        type=Path,
        default=Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt"),
    )
    parser.add_argument(
        "--onep_nbody_root",
        type=Path,
        default=Path("/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/1P"),
    )
    parser.add_argument(
        "--onep_hydro_root",
        type=Path,
        default=Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P"),
    )

    parser.add_argument(
        "--test_manifest",
        type=Path,
        default=None,
        help="JSON manifest path for suite=test entries",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_path or (args.run_dir / "checkpoints" / "last.ckpt")

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
        onep_param_file=args.onep_param_file,
        onep_nbody_root=args.onep_nbody_root,
        onep_hydro_root=args.onep_hydro_root,
        test_manifest=args.test_manifest,
    )

    run_cfg = RunConfig(
        run_dir=args.run_dir,
        checkpoint_path=checkpoint,
        output_root=args.output_root,
        model_name=args.model_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        patch_mass_match=not args.no_patch_mass_match,
        taper_frac=args.taper_frac,
        use_amp=not args.no_amp,
        device=args.device,
        prep_only=args.prep_only,
        regenerate=args.regenerate,
        regenerate_all=args.regenerate_all,
        repaste=args.repaste,
    )

    print("=" * 80)
    print("TEST SUITE RUNNER")
    print("=" * 80)
    print(f"Suite: {args.suite}")
    print(f"Simulations: {len(specs)}")
    print(f"Model run dir: {run_cfg.run_dir}")
    print(f"Checkpoint: {run_cfg.checkpoint_path}")
    print(f"Output root: {run_cfg.output_root}")
    print(f"Prep only: {run_cfg.prep_only}")
    print(f"Regenerate: {run_cfg.regenerate}")
    print(f"Regenerate all: {run_cfg.regenerate_all}")
    print(f"Repaste: {run_cfg.repaste}")
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
    summary_path.write_text(json.dumps(_to_jsonable(out_summary), indent=2, sort_keys=True))
    print(f"Saved run summary to {summary_path}")


if __name__ == "__main__":
    main()
