"""Self-contained inference demo for BIND2.

Loads a packaged DMO sample (``examples/data/dmo_sample.npz``), runs the
``fm_two_head`` checkpoint, and writes a ``examples/demo_output.png`` figure
that compares input DMO, generated [DM_hydro, Gas, Stars], and ground truth.

Quick start::

    pip install -e .
    bind-download-weights --run fm_two_head
    python examples/generate_demo.py

Override paths via flags or env vars:

    --weights / $BIND_WEIGHTS_DIR  (default: ./weights)
    --sample  / $BIND_DEMO_SAMPLE  (default: examples/data/dmo_sample.npz)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from bind.data import NormStats, log_transform
from bind.train import FlowMatchingLit
from bind.test_suite.pipeline import _denormalize_to_physical


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path,
                   default=Path(os.environ.get("BIND_WEIGHTS_DIR", "weights")),
                   help="Directory holding <run>/last.ckpt and <run>/norm_stats.npz")
    p.add_argument("--run", type=str, default="fm_two_head",
                   choices=["fm_two_head", "fm_thermo"],
                   help="Which checkpoint to use")
    p.add_argument("--sample", type=Path,
                   default=Path(os.environ.get("BIND_DEMO_SAMPLE",
                                               "examples/data/dmo_sample.npz")),
                   help="Packaged DMO sample npz")
    p.add_argument("--n_steps", type=int, default=50,
                   help="Number of flow-matching sampler steps")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("examples/demo_output.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.weights / args.run / "last.ckpt"
    norm_path = args.weights / args.run / "norm_stats.npz"
    if not ckpt_path.exists() or not norm_path.exists():
        raise SystemExit(
            f"Missing weights at {ckpt_path} / {norm_path}.\n"
            f"Run: bind-download-weights --run {args.run}"
        )
    if not args.sample.exists():
        raise SystemExit(f"Missing demo sample at {args.sample}")

    device = torch.device(args.device)
    print(f"[demo] loading {ckpt_path} on {device}")
    model = FlowMatchingLit.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    ns = NormStats.load(str(norm_path))

    d = np.load(args.sample)
    cond_phys = d["condition"].astype(np.float32)              # (128,128)
    ls_phys = d["large_scale"].astype(np.float32)              # (3,128,128)
    raw_params = d["params"].astype(np.float64)                # (35,)
    target = d["target"].astype(np.float32)                    # (3,128,128)

    cond = log_transform(cond_phys)[None]
    cond = (cond - ns.cond_mean) / (ns.cond_std + 1e-8)
    ls = log_transform(ls_phys)
    ls = (ls - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)
    p = np.where(ns.param_log_flag == 1, np.log10(np.maximum(raw_params, 1e-30)), raw_params)
    params = ((p - ns.param_min) / (ns.param_max - ns.param_min + 1e-8)).astype(np.float32)

    # Optional 35→subset selection (for models trained with --exclude_cosmo_params).
    n_params = int(model.hparams.get("n_params", 35))
    if n_params != 35:
        cosmo_idx = [0, 1, 7, 8]
        keep = [i for i in range(35) if i not in cosmo_idx]
        params = params[keep]

    no_large_scale = bool(model.hparams.get("no_large_scale", False))

    cond_t = torch.from_numpy(cond[None]).to(device)
    ls_t = None if no_large_scale else torch.from_numpy(ls[None]).to(device)
    params_t = torch.from_numpy(params[None]).to(device)

    print(f"[demo] sampling with n_steps={args.n_steps}")
    with torch.no_grad():
        gen = model.fm.sample(cond_t, ls_t, params_t, n_steps=args.n_steps)
    gen_np = gen.float().cpu().numpy().astype(np.float32)
    pred = _denormalize_to_physical(gen_np, ns)[0]             # (3,128,128)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        np.savez(args.out.with_suffix(".npz"), prediction=pred, truth=target, dmo=cond_phys)
        print(f"[demo] matplotlib not available; wrote arrays to {args.out.with_suffix('.npz')}")
        return

    field_names = ["DM_hydro", "Gas", "Stars"]
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))
    for j, name in enumerate(field_names):
        ax[0, j].imshow(np.log10(1 + cond_phys), origin="lower")
        ax[0, j].set_title(f"DMO (input) -> {name}" if j == 1 else "DMO (input)")
        ax[1, j].imshow(np.log10(1 + pred[j]), origin="lower")
        ax[1, j].set_title(f"BIND2 {name}")
        ax[2, j].imshow(np.log10(1 + target[j]), origin="lower")
        ax[2, j].set_title(f"truth {name}")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[demo] wrote {args.out}")


if __name__ == "__main__":
    main()
