"""Download pretrained BIND2 checkpoints from Hugging Face Hub.

Default repo: ``Maxelee/BIND2`` (override with ``--hf_repo``).

Layout on the Hub mirrors the local ``weights/<run>/`` layout:
    <run>/last.ckpt
    <run>/norm_stats.npz

Usage:
    python -m tools.download_weights                # downloads all known runs
    python -m tools.download_weights fm_two_head    # one run
    python -m tools.download_weights fm_thermo --dest weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "Maxelee/BIND2"
KNOWN_RUNS = ("fm_two_head", "fm_thermo")
FILES_PER_RUN = ("last.ckpt", "norm_stats.npz")


def download_run(run: str, dest: Path, hf_repo: str, revision: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - import-time hint
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    run_dir = dest / run
    run_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES_PER_RUN:
        path = hf_hub_download(
            repo_id=hf_repo,
            filename=f"{run}/{fname}",
            revision=revision,
            local_dir=str(dest),
        )
        print(f"  -> {path}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*", default=list(KNOWN_RUNS),
                   help=f"Run names to fetch (default: {' '.join(KNOWN_RUNS)})")
    p.add_argument("--dest", type=Path, default=Path("weights"),
                   help="Local destination directory (default: weights/)")
    p.add_argument("--hf_repo", default=DEFAULT_REPO,
                   help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    p.add_argument("--revision", default=None,
                   help="Optional branch/tag/commit on the HF repo")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        print(f"Downloading {run} from {args.hf_repo}...")
        download_run(run, args.dest, args.hf_repo, args.revision)
    print(f"\nDone. Pass --run_dir {args.dest}/<run> to run_test_suite.py")


if __name__ == "__main__":
    main()
