"""Repackage n1000_nu05_canon.npz into the per-arm shard layout fig05 expects.

fig05 reads two npz "shards" with keys ``pdf``/``peak_counts``/``minima_counts``/
``mf_v0..2`` (means, ``(5, 22)``) plus ``*_real`` per-realization cubes
``(n_real, 5, 22)``.  The canonical builder stores them arm-prefixed instead, and
the truth arm was computed for the z_s=1 plane only (that is the only plane fig05
indexes).  Planes that were not computed are written as NaN so that any future
code touching them fails visibly rather than using a wrong plane.

    python3 n1000_nu05_to_shards.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "n1000_nu05_canon.npz"
OUT = Path("/mnt/home/mlee1/ceph/bind_n1000/nu05_shards")
ZI, N_PL = 1, 5
MAP = {"pdf": "pdf", "peak_counts": "peaks", "minima_counts": "minima",
       "mf_v0": "v0", "mf_v1": "v1", "mf_v2": "v2"}


def main() -> None:
    d = np.load(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, name in (("bind", f"sci_bind_{str(d['replica'])}_n1000.npz"),
                      ("truth", "sci_truth_n1000.npz")):
        store = {"nu": d["nu"]}
        for out_key, in_key in MAP.items():
            a = np.asarray(d[f"{arm}_{in_key}"], float)
            if a.ndim == 2:                    # (n_real, 22): single plane -> pad
                full = np.full((a.shape[0], N_PL, a.shape[1]), np.nan)
                full[:, ZI, :] = a
                a = full
            store[out_key] = np.nanmean(a, axis=0)
            store[f"{out_key}_real"] = a
        np.savez_compressed(OUT / name, **store)
        print(f"wrote {OUT / name}  pdf{store['pdf'].shape} "
              f"pdf_real{store['pdf_real'].shape}")


if __name__ == "__main__":
    main()
