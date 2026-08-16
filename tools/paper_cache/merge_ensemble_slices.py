#!/usr/bin/env python3
"""Merge per-slice outputs of build_posterior_ensemble.py (--halo_slice) into one npz.

    python merge_ensemble_slices.py out.npz slice0.npz slice1.npz [...]

Per-halo arrays are concatenated along axis 0; run metadata is taken from the
first slice (all slices share seed/config by construction).
"""
import sys
import numpy as np

out, parts = sys.argv[1], sys.argv[2:]
ds = [np.load(p, allow_pickle=True) for p in parts]
merged = {}
for k in ds[0].files:
    a = ds[0][k]
    if a.ndim >= 1 and all(k in d.files and d[k].ndim == a.ndim for d in ds) \
            and k not in ("names", "config"):
        merged[k] = np.concatenate([d[k] for d in ds], axis=0)
    else:
        merged[k] = a
np.savez_compressed(out, **merged)
n = merged[[k for k in ("truth", "draws") if k in merged][0]].shape[0]
print(f"wrote {out}  ({n} halos from {len(parts)} slices)")
