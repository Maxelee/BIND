"""``bind-science-plan``: build the stage-2 disBatch task files for 1a/1b.

Reads the design matrices written by ``bind-design`` and, for each requested
design, lays out the run tree and emits a disBatch task file of
``bind-paint-generate`` commands (one per run x snapshot) that all reuse the
shared, parameter-independent stage-1 dirs.  See :mod:`bind.inference.science`.

    bind-science-plan \\
        --output_root      /ceph/bind_science \\
        --design_dir       /ceph/bind_science/design \\
        --shared_stage1_root /ceph/bind_lightcone_tng \\
        --designs fiducial 1P sobol

Then submit each task file inside a GPU allocation with disBatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference.science import plan_runs

# design name -> npy file
_DESIGN_FILES = {
    "fiducial": "astro_params_fiducial.npy",
    "1P": "astro_params_1P.npy",
    "sobol": "astro_params_sobol.npy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--design_dir", type=Path, required=True,
                   help="Directory of bind-design outputs (astro_params_*.npy)")
    p.add_argument("--shared_stage1_root", type=Path,
                   default=Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"),
                   help="Root holding snap_<NNN>/stage1 (reused across all runs)")
    p.add_argument("--designs", nargs="+", default=["fiducial", "1P", "sobol"],
                   choices=list(_DESIGN_FILES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for design in args.designs:
        params = np.load(args.design_dir / _DESIGN_FILES[design])
        plan = plan_runs(output_root=args.output_root, design=design,
                         params_matrix=params)
        print(f"[plan] {design:9s} {plan['n_runs']:4d} runs ({plan['n_todo']} to do)")
        print(f"        submit: {plan['submit']}")
    print("[plan] each array task runs the full canonical pipeline "
          "(run_science_run.sh) and keeps only halos/kappa/y/stats.")


if __name__ == "__main__":
    main()
