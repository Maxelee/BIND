"""``bind-make-portable``: build a self-contained bundle for off-site generation.

Assembles everything the GPU-rich machine needs to generate BIND halos for a
design (default: the 30-param × {min,max} prior-bound set = 60 lightcones) into
one Globus-transferable directory::

    <bundle>/
      design/astro_params_<design>.npy   design.json (+ meta: run -> param,bound)
      runs/run_<i>/params.npy            per-run 35-dim vectors
      conditions/snap_<NNN>/             stripped stage-1 (cutouts; NO DMO maps)
          stage1_manifest.json  stage1_slab*.npz
      weights/<model>/                   last.ckpt + norm_stats.npz
      generate_halos_portable.sh         run on the GPU machine
      README.md                          the few terminal lines

Workflow: build here → Globus to the GPU machine → run the generate script →
Globus ``runs/`` back → run the CPU side (paste → lux → stats) here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bind.inference.design import twobound_design, oneparam_design, sobol_design
from bind.params import PARAM_NAMES

LIGHTCONE_SNAPSHOTS = (96, 90, 85, 80, 76, 71, 67, 63, 59, 56,
                       52, 49, 46, 43, 41, 38, 35, 33, 31, 29)
# stage-1 keys to drop (big full-box maps not needed for generation)
_DROP = {"dmo", "dmo_aa"}

_GEN_SCRIPT = r"""#!/bin/bash
# Generate BIND halos for one run of the portable bundle (GPU machine).
# Requires: BIND installed (pip install -e <bind repo>) + a GPU.
# Single run:  IDX=0 bash generate_halos_portable.sh
# SLURM array: sbatch --array=0-{NMAX} generate_halos_portable.sh
set -euo pipefail
BUNDLE="${{BUNDLE:-$(cd "$(dirname "$0")" && pwd)}}"
IDX="${{IDX:-${{SLURM_ARRAY_TASK_ID:?set IDX or SLURM_ARRAY_TASK_ID}}}}"
SNAPSHOTS=({SNAPS})
RUN_DIR="$BUNDLE/runs/$(printf 'run_%04d' "$IDX")"
WEIGHTS="$BUNDLE/weights/{MODEL}"
for SNAP in "${{SNAPSHOTS[@]}}"; do
  S3=$(printf '%03d' "$SNAP")
  python -u -m bind.cli.generate_halos \
      --stage1_dir "$BUNDLE/conditions/snap_${{S3}}" \
      --params "$RUN_DIR/params.npy" \
      --run_dir "$WEIGHTS" \
      --output_dir "$RUN_DIR/snap_${{S3}}" \
      --device auto
done
echo "=== halos done: $RUN_DIR ==="
"""

_README = """# BIND portable generation bundle — `{design}` ({n_runs} runs)

Self-contained inputs for generating BIND halos on a GPU machine.

## On the GPU machine
1. Install BIND once:  `git clone <BIND repo> && cd BIND && pip install -e .`
2. Generate all {n_runs} runs (each ~4 GPU-hr, 20 snapshots):
   ```
   cd {bundle_name}
   sbatch --array=0-{nmax} generate_halos_portable.sh     # or loop IDX=0..{nmax}
   ```
   Outputs land in `runs/run_<i>/snap_<NNN>/composite_slab*.npz` (halo patches).

## Back home (Flatiron)
3. Globus `runs/` back into `{output_root}/runs/{design}/`.
4. Composite → lensplanes → lux ray-trace → stats:
   ```
   DESIGN={design} sbatch --array=0-{nmax} run_sobol_paste.sh
   DESIGN={design} sbatch --array=0-{nmax} run_sobol_lux.sh
   DESIGN={design} sbatch --array=0-{nmax} run_sobol_stats.sh
   ```
   Kept per run: halos, kappa_maps.npz, y_maps.npz, statistics.

Design map: see `design/design.json` (run index -> param, bound, value).
"""


def _strip_stage1(stage1_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(stage1_dir / "stage1_manifest.json", out_dir / "stage1_manifest.json")
    for sp in sorted(stage1_dir.glob("stage1_slab*.npz")):
        d = np.load(sp)
        np.savez_compressed(out_dir / sp.name,
                            **{k: d[k] for k in d.files if k not in _DROP})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True, help="Output bundle directory")
    p.add_argument("--design", choices=["twobound", "1P", "sobol"], default="twobound")
    p.add_argument("--n_sobol", type=int, default=100)
    p.add_argument("--stage1_root", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"))
    p.add_argument("--weights", type=Path,
                   default=Path("/mnt/home/mlee1/BIND/weights/fm_redshift_thermo"))
    p.add_argument("--output_root", type=str,
                   default="/mnt/home/mlee1/ceph/bind_science",
                   help="Where halos are transferred back to (for the README).")
    p.add_argument("--snapshots", type=int, nargs="+", default=list(LIGHTCONE_SNAPSHOTS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = args.bundle
    (bundle / "design").mkdir(parents=True, exist_ok=True)

    if args.design == "twobound":
        params, meta = twobound_design()
    elif args.design == "1P":
        params, meta = oneparam_design()
    else:
        params, meta = sobol_design(args.n_sobol), []
    n_runs = params.shape[0]

    np.save(bundle / "design" / f"astro_params_{args.design}.npy", params)
    (bundle / "design" / "design.json").write_text(json.dumps(
        {"design": args.design, "n_runs": int(n_runs),
         "astro_param_names": [PARAM_NAMES[m["index"]] for m in meta] if meta else None,
         "runs": meta}, indent=1))

    for i in range(n_runs):
        rd = bundle / "runs" / f"run_{i:04d}"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "params.npy", params[i])

    print(f"[portable] design={args.design}  n_runs={n_runs}")
    for s in args.snapshots:
        s3 = f"{s:03d}"
        _strip_stage1(args.stage1_root / f"snap_{s3}" / "stage1",
                      bundle / "conditions" / f"snap_{s3}")
        print(f"[portable] stripped conditions snap_{s3}")

    wdst = bundle / "weights" / args.weights.name
    wdst.mkdir(parents=True, exist_ok=True)
    for f in ("last.ckpt", "norm_stats.npz"):
        if (args.weights / f).exists():
            shutil.copy(args.weights / f, wdst / f)
    print(f"[portable] copied weights -> {wdst}")

    snaps_str = " ".join(str(s) for s in args.snapshots)
    (bundle / "generate_halos_portable.sh").write_text(
        _GEN_SCRIPT.format(SNAPS=snaps_str, MODEL=args.weights.name, NMAX=n_runs - 1))
    (bundle / "generate_halos_portable.sh").chmod(0o755)
    (bundle / "README.md").write_text(_README.format(
        design=args.design, n_runs=n_runs, nmax=n_runs - 1,
        bundle_name=bundle.name, output_root=args.output_root))

    print(f"[portable] bundle ready -> {bundle}")
    print(f"[portable] Globus this dir; on the GPU box: cd {bundle.name} && "
          f"sbatch --array=0-{n_runs-1} generate_halos_portable.sh")


if __name__ == "__main__":
    main()
