"""Build the nu05 per-realization shard for the FULL-HYDRO TNG300 lightcone.

Same engine (nu_grid.compute) and same grid as the BIND and hydro-pasted shards,
so the three are directly comparable on paper Fig 5.  Writes only under
referee_work/fidswap/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
import nu_grid as ng  # noqa: E402

FS = Path("/mnt/home/mlee1/ceph/referee_work/fidswap")
OUT = FS / "nu05_shards"; OUT.mkdir(parents=True, exist_ok=True)
KAPPA = Path("/mnt/home/mlee1/ceph/tng_full_validation/runs/hydro_full/run_0000/kappa_maps.npz")
assert KAPPA.exists(), KAPPA
dst = OUT / "sci_hydrofull.npz"
if dst.exists():
    print(f"{dst} exists -- skipping")
else:
    t0 = time.time()
    res = ng.compute(KAPPA, per_real=True)
    np.savez_compressed(dst, **res, src=str(KAPPA))
    print(f"wrote {dst} in {time.time()-t0:.0f}s")
    print(", ".join(f"{k}{np.asarray(v).shape}" for k, v in res.items() if k != "nu"))
