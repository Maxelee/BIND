"""Slim a Lightning training checkpoint for inference-only release.

Drops `optimizer_states`, `lr_schedulers`, `loops`, and `callbacks` (the bulk of
the file size). Keeps `state_dict`, `hyper_parameters`, `hparams_name`, EMA
shadow params, and the small bookkeeping fields needed by
`FlowMatchingLit.load_from_checkpoint`.

Usage:
    python -m tools.slim_checkpoint \
        /path/to/run_dir/checkpoints/last.ckpt \
        /path/to/release/last.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


_KEEP_KEYS = (
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "state_dict",
    "hparams_name",
    "hyper_parameters",
    "ema_state_dict",  # only present in newer training runs
)


def slim(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: ckpt[k] for k in _KEEP_KEYS if k in ckpt}

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    kept = ", ".join(out.keys())
    print(f"{src} ({src_mb:.0f} MB) -> {dst} ({dst_mb:.0f} MB)")
    print(f"  kept: {kept}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="Full Lightning checkpoint")
    p.add_argument("dst", type=Path, help="Output slim checkpoint path")
    args = p.parse_args()
    slim(args.src, args.dst)


if __name__ == "__main__":
    main()
