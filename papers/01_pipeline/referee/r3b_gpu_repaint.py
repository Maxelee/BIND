#!/usr/bin/env python3
"""R3b GPU driver: N fresh-seed re-paints of snap_096 at one parameter vector.

This is the *direct* version of the multi-sample test.  It could not be run by the
agent because the assigned node has no GPU and one CPU core (>130 s/halo; see the
memo).  On any GPU box it is a single command.

It also supplies the **seed control the production path lacks**.  ``bind`` never
seeds the flow sampler (``bind/model.py`` calls ``torch.randn`` off the global
RNG), so re-paints are irreproducible.  Here each sample ``s`` is drawn under

    torch.manual_seed(BASE_SEED + s * SEED_STRIDE)          # SEED_STRIDE = 10**6

with ``BASE_SEED`` recorded in the output manifest, so the ensemble is
reproducible and collides with nothing in the campaign (the only other seed ladder
in the project is the lux realization ladder ``1992 + 7 r``, r < 1000, which lives
in a different RNG entirely and never reaches the sampler).

Usage
-----
    # fiducial location, 8 fresh draws, all four slabs
    python r3b_gpu_repaint.py --params /…/snap_096/stage1/params.npy \\
        --n_samples 8 --out /…/referee_work/texture/gpu_fid --device cuda

    # strong-feedback node (max VariableWindVelFactor = twobound run_0005)
    python r3b_gpu_repaint.py \\
        --params /mnt/home/mlee1/ceph/bind_science/runs/twobound/run_0005/params.npy \\
        --n_samples 8 --out /…/referee_work/texture/gpu_vwv_max --device cuda

Each sample lands in ``<out>/sample_<s>/composite_slab{NN}.npz`` with
``generated_patches`` + ``thermo_patches``, i.e. exactly the layout
``r3b_texture_analysis.py`` already consumes (point ``ENSEMBLE_DIRS`` at them).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import bind
from bind.inference.paint_stages import generate_halos

STAGE1 = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1")
REDSHIFT = 0.0337243718735154
BASE_SEED = 770_000_000
SEED_STRIDE = 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1_dir", type=Path, default=STAGE1)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, default=Path("/mnt/home/mlee1/BIND/weights/fm_redshift_thermo"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--base_seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = bind.Model.from_local(args.run_dir, device=args.device)
    params = np.load(args.params)

    manifest = {
        "stage1_dir": str(args.stage1_dir),
        "params_file": str(args.params),
        "run_dir": str(args.run_dir),
        "n_samples": args.n_samples,
        "base_seed": args.base_seed,
        "seed_stride": SEED_STRIDE,
        "n_steps": args.n_steps,
        "redshift": REDSHIFT,
        "samples": [],
    }

    for s in range(args.n_samples):
        seed = args.base_seed + s * SEED_STRIDE
        torch.manual_seed(seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        dest = args.out / f"sample_{s:02d}"
        t0 = time.time()
        generate_halos(
            args.stage1_dir, model, output_dir=dest, params=params,
            redshift=REDSHIFT, n_steps=args.n_steps, batch_size=args.batch_size,
            use_amp=True,
        )
        dt = time.time() - t0
        manifest["samples"].append({"sample": s, "seed": seed, "dir": str(dest), "seconds": dt})
        print(f"[r3b] sample {s} seed={seed} -> {dest}  ({dt/60:.1f} min)", flush=True)

    (args.out / "r3b_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", args.out / "r3b_manifest.json")


if __name__ == "__main__":
    main()
