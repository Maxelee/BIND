"""``bind-design`` (stage 0): write the astro-parameter design matrices.

Writes ``astro_params_sobol.npy``, ``astro_params_1P.npy`` (+ ``..._1P_meta.json``)
and ``astro_params_fiducial.npy`` into ``--output_dir``, plus a ``design.json``
recording the convention (cosmology fixed at TNG300, 30 astro variables).

Each row is a full 35-dim BIND parameter vector ready for the lightcone
pipeline.  See :mod:`bind.inference.design`.

    bind-design --output_dir /ceph/bind_science/design --n_sobol 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.design import (
    ASTRO_PARAM_INDICES,
    COSMO_PARAM_INDICES,
    TNG300_COSMO,
    sobol_design,
    oneparam_design,
    fiducial_vector,
)
from bind.params import PARAM_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_sobol", type=int, default=256,
                   help="Sobol design size (power of 2 recommended; 256-512).")
    p.add_argument("--sobol_seed", type=int, default=0)
    p.add_argument("--oneparam_levels", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sobol = sobol_design(args.n_sobol, seed=args.sobol_seed)
    onep, onep_meta = oneparam_design(args.oneparam_levels)
    fid = fiducial_vector()

    np.save(args.output_dir / "astro_params_sobol.npy", sobol)
    np.save(args.output_dir / "astro_params_1P.npy", onep)
    np.save(args.output_dir / "astro_params_fiducial.npy", fid)
    (args.output_dir / "astro_params_1P_meta.json").write_text(json.dumps(onep_meta, indent=1))

    design = {
        "convention": "cosmology fixed at TNG300; 30 astro params varied in SB35 space",
        "tng300_cosmo": TNG300_COSMO,
        "cosmo_param_indices": list(COSMO_PARAM_INDICES),
        "astro_param_indices": list(ASTRO_PARAM_INDICES),
        "astro_param_names": [PARAM_NAMES[i] for i in ASTRO_PARAM_INDICES],
        "n_sobol": int(args.n_sobol),
        "sobol_seed": int(args.sobol_seed),
        "oneparam_levels": int(args.oneparam_levels),
        "n_oneparam": int(onep.shape[0]),
        "files": {
            "sobol": "astro_params_sobol.npy",
            "oneparam": "astro_params_1P.npy",
            "oneparam_meta": "astro_params_1P_meta.json",
            "fiducial": "astro_params_fiducial.npy",
        },
    }
    (args.output_dir / "design.json").write_text(json.dumps(design, indent=1))

    print(f"[design] sobol     {sobol.shape} -> astro_params_sobol.npy")
    print(f"[design] 1P sweep  {onep.shape} -> astro_params_1P.npy "
          f"({args.oneparam_levels} levels x 30 params)")
    print(f"[design] fiducial  {fid.shape} -> astro_params_fiducial.npy")
    print(f"[design] wrote design.json -> {args.output_dir}")


if __name__ == "__main__":
    main()
