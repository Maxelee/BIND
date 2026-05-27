"""CLI entrypoint for 3D DMO->hydro generation runs (pre-extracted halo volumes).

Mirrors run_test_suite.py but for the 3D model (FlowMatching3dLit / UNet3d).

Key differences from the 2D runner:
  - Works with pre-extracted .npz files (condition/target/params per halo).
    No raw snapshot projection, halo catalog loading, or 2D compositing.
  - Uses NormStats3d and FlowMatching3dLit instead of NormStats / FlowMatchingLit.
  - No large_scale context (3D model does not use it).

Output structure::

    <output_root>/<model_name>/
        sim_<N>/halo_<M>.npz        # generated (3, 128, 128, 128) volumes
        run_summary_3d[_chunk<K>].json
"""

from __future__ import annotations

import argparse
import json
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_3d import AstroDataset3d, NormStats3d, load_file_list_3d
from train_3d import FlowMatching3dLit
from test_suite.artifacts import to_jsonable
from test_suite.pipeline import _denormalize_to_physical


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_model_bundle_3d(
    run_dir: Path, checkpoint_path: Path, device: torch.device
) -> tuple[NormStats3d, object]:
    """Load NormStats3d and the FlowMatching3d sampler from a checkpoint."""
    ns = NormStats3d.load(run_dir / "norm_stats_3d.npz")
    lit = FlowMatching3dLit.load_from_checkpoint(str(checkpoint_path), map_location=device)
    lit.eval().to(device)
    return ns, lit.fm


def _collect_file_list(data_root: Path, split: str) -> list[str]:
    """Return the file list for *split* from *data_root*."""
    if split == "all":
        train_files = load_file_list_3d(data_root, "train")
        test_files = load_file_list_3d(data_root, "test")
        seen: set[str] = set()
        merged: list[str] = []
        for p in train_files + test_files:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        return merged
    return load_file_list_3d(data_root, split)


# ── Core generation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_generation(
    file_list: list[str],
    ns: NormStats3d,
    fm,
    device: torch.device,
    *,
    n_steps: int,
    batch_size: int,
    use_amp: bool,
    output_dir: Path,
    regenerate: bool,
) -> list[dict]:
    """Run 3D inference on all halo files and save generated volumes.

    For each input file ``<data_root>/sim_N/halo_M.npz`` the generated
    volume is written to ``<output_dir>/sim_N/halo_M.npz`` (key: ``generated``,
    shape ``(3, D, H, W)`` in physical units).

    Returns a list of per-halo dicts with mass statistics.
    """
    ds = AstroDataset3d(file_list, ns)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    summaries: list[dict] = []
    file_idx = 0

    for batch in tqdm(loader, desc="Generating 3D volumes"):
        bs = batch["condition"].shape[0]
        cond = batch["condition"].to(device)
        params = batch["params"].to(device)

        with amp_ctx:
            gen_raw = fm.sample(cond, params, n_steps=n_steps)

        # Denormalize to physical space — works for both (B,C,H,W) and (B,C,D,H,W)
        gen_np = _denormalize_to_physical(gen_raw.float().cpu().numpy(), ns)
        truth_np = _denormalize_to_physical(batch["target"].numpy().copy(), ns)

        for i in range(bs):
            src = Path(file_list[file_idx])
            # Preserve relative path structure: sim_N/halo_M.npz
            rel = Path(src.parent.name) / src.name
            out_path = output_dir / rel

            if not out_path.exists() or regenerate:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out_path, generated=gen_np[i].astype(np.float32))

            # Per-channel total mass (sum over all spatial dims)
            gen_mass = gen_np[i].sum(axis=(-3, -2, -1)).astype(np.float64)
            truth_mass = truth_np[i].sum(axis=(-3, -2, -1)).astype(np.float64)
            rel_err = (gen_mass - truth_mass) / (truth_mass + 1e-10)

            summaries.append(
                {
                    "file": str(rel),
                    "gen_mass": gen_mass.tolist(),
                    "truth_mass": truth_mass.tolist(),
                    "mass_rel_err": rel_err.tolist(),
                }
            )
            file_idx += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summaries


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 3D DMO->hydro generation on pre-extracted halo volumes."
    )

    # ── Data
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/mnt/home/mlee1/ceph/train_data_1024/train_3d"),
        help="Root of the 3D dataset (contains file_list_cache_3d_*.txt and sim_N/ dirs)",
    )
    p.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to process (default: test)",
    )
    p.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Explicit newline-separated file list (overrides --data_root / --split)",
    )

    # ── Model
    p.add_argument(
        "--run_dir",
        type=Path,
        default=Path("/mnt/home/mlee1/ceph/fm_runs_3d/fm3d_two_head_v2"),
        help="3D model run directory (must contain norm_stats_3d.npz and checkpoints/)",
    )
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Checkpoint to load (default: <run_dir>/checkpoints/last.ckpt)",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="fm3d_two_head_v2",
        help="Subdirectory name under output_root for this model's outputs",
    )

    # ── Output
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/mnt/home/mlee1/ceph/fm3d_testsuite"),
    )

    # ── Inference
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu")
    p.add_argument("--no_amp", action="store_true")

    # ── Parallelism
    p.add_argument(
        "--n_chunks",
        type=int,
        default=1,
        help="Split the file list into N chunks for SLURM array jobs",
    )
    p.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="Which chunk to process (0-indexed, used with --n_chunks)",
    )

    # ── Cache control
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run inference even if the output .npz already exists",
    )

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── File list ─────────────────────────────────────────────────────────────
    if args.file_list is not None:
        with open(args.file_list) as fh:
            all_files = [line.strip() for line in fh if line.strip()]
    else:
        all_files = _collect_file_list(args.data_root, args.split)

    # ── Chunking ──────────────────────────────────────────────────────────────
    if args.n_chunks > 1:
        if not (0 <= args.chunk_id < args.n_chunks):
            raise ValueError(
                f"--chunk_id {args.chunk_id} out of range for --n_chunks {args.n_chunks}"
            )
        chunk_size = (len(all_files) + args.n_chunks - 1) // args.n_chunks
        all_files = all_files[
            args.chunk_id * chunk_size : (args.chunk_id + 1) * chunk_size
        ]

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint = args.checkpoint_path or (args.run_dir / "checkpoints" / "last.ckpt")
    output_dir = args.output_root / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("3D TEST SUITE RUNNER")
    print("=" * 80)
    print(f"Data root:   {args.data_root}")
    print(f"Split:       {args.split}")
    print(f"Files:       {len(all_files)}")
    if args.n_chunks > 1:
        print(f"Chunk:       {args.chunk_id + 1}/{args.n_chunks}  ({len(all_files)} files)")
    print(f"Model dir:   {args.run_dir}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Output dir:  {output_dir}")
    print(f"N steps:     {args.n_steps}")
    print(f"Batch size:  {args.batch_size}")
    print(f"AMP:         {not args.no_amp}")
    print(f"Regenerate:  {args.regenerate}")
    print("=" * 80)

    # ── Load model ────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    ns, fm = load_model_bundle_3d(args.run_dir, checkpoint, device)

    # ── Generate ──────────────────────────────────────────────────────────────
    summaries: list[dict] = []
    try:
        summaries = run_generation(
            all_files,
            ns,
            fm,
            device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
            output_dir=output_dir,
            regenerate=args.regenerate,
        )
    except Exception:
        traceback.print_exc()

    # ── Aggregate per-channel mass stats ──────────────────────────────────────
    channel_names = ["dm_hydro", "gas", "stars"]
    if summaries:
        print("\nPer-channel mass relative error summary:")
        for ch_idx, ch in enumerate(channel_names):
            errs = np.array([s["mass_rel_err"][ch_idx] for s in summaries])
            print(
                f"  {ch:12s}: mean={errs.mean() * 100:+.2f}%  "
                f"std={errs.std() * 100:.2f}%  "
                f"median={np.median(errs) * 100:+.2f}%"
            )

    # ── Save summary ──────────────────────────────────────────────────────────
    chunk_tag = f"_chunk{args.chunk_id:04d}" if args.n_chunks > 1 else ""
    summary_path = output_dir / f"run_summary_3d{chunk_tag}.json"
    out_summary = {
        "split": args.split,
        "n_files_requested": len(all_files),
        "n_files_completed": len(summaries),
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "n_steps": args.n_steps,
        "halos": summaries,
    }
    summary_path.write_text(json.dumps(to_jsonable(out_summary), indent=2, sort_keys=True))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
