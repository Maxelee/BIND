#!/usr/bin/env python
"""Per-pixel mutual information between parameters and stacked halo patches.

For every astrophysical parameter and every channel we compute, at each of the
128x128 pixels, the mutual information ``I(theta_j; X_pixel)`` across the
independent simulations -- separately for the TRUTH stacked patch and the BIND
stacked patch. The stacked patches are the per-simulation mean over that sim's
halos (so this is an over-all-halos result, aggregated per sim, not a single
halo). Output is a compact ``.npz`` the notebook loads and plots.

Heavy but embarrassingly parallel: tasks are (parameter, channel, source) and
fan out over a process pool.

Example
-------
    python tools/compute_pixel_mi.py --workers 32
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Cosmological parameter indices (0-based): Omega_m, sigma_8, Omega_b, h, n_s.
# These are excluded; the remaining 30 are the astrophysical parameters.
N_PARAMS = 35
COSMO_IDX = [0, 1, 6, 7, 8]
ASTRO_IDX = [i for i in range(N_PARAMS) if i not in COSMO_IDX]

# Globals populated in each worker via the pool initializer (avoids re-pickling
# the large stacked-patch arrays for every task).
_STACK = {}


def estimate_mi_bits(x, y, n_neighbors=5, random_state=0):
    """KSG kNN mutual information in bits; nan if too little data / no variation."""
    from sklearn.feature_selection import mutual_info_regression

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    k = min(n_neighbors, x.size - 1)
    mi = mutual_info_regression(
        x.reshape(-1, 1), y, discrete_features=False,
        n_neighbors=k, random_state=random_state)[0]
    return float(mi / np.log(2.0))


def _init_worker(stack_t, stack_b, stack_par, n_neighbors):
    _STACK['t'] = stack_t
    _STACK['b'] = stack_b
    _STACK['par'] = stack_par
    _STACK['k'] = n_neighbors


def _pixel_map(task):
    """Compute one HxW MI map for a (param, channel, source) task."""
    j, c, src = task
    field = _STACK[src][:, c]                       # (n_sim, H, W)
    field = np.log10(1.0 + np.clip(field, 0.0, None))
    x = _STACK['par'][:, j]
    k = _STACK['k']
    H, W = field.shape[1:]
    out = np.zeros((H, W), dtype=np.float32)
    for a in range(H):
        for b in range(W):
            out[a, b] = estimate_mi_bits(x, field[:, a, b], n_neighbors=k)
    return j, c, src, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_name', default='fm_two_head_no_pmm')
    ap.add_argument('--cache_dir',
                    default='examples/paper_figures/mi_cache',
                    help='directory holding sim_stacked_patches_<model>.npz')
    ap.add_argument('--out', default=None,
                    help='output npz (default: <cache_dir>/pixel_mi_<model>.npz)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--neighbors', type=int, default=5)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    patches = cache_dir / f'sim_stacked_patches_{args.model_name}.npz'
    if not patches.exists():
        raise SystemExit(
            f'stacked-patch cache not found: {patches}\n'
            'Run the notebook through section 5/11 first to build it.')
    out_path = Path(args.out) if args.out else \
        cache_dir / f'pixel_mi_{args.model_name}.npz'

    z = np.load(patches, allow_pickle=True)
    stack_t = np.asarray(z['stack_t'], np.float64)
    stack_b = np.asarray(z['stack_b'], np.float64)
    stack_par = np.asarray(z['stack_par'], np.float64)
    n_sim, n_ch, H, W = stack_t.shape
    print(f'loaded {patches.name}: {n_sim} sims, {n_ch} channels, {H}x{W} pixels')

    tasks = [(j, c, src) for j in ASTRO_IDX for c in range(n_ch) for src in ('t', 'b')]
    print(f'{len(tasks)} tasks ({len(ASTRO_IDX)} astro params x {n_ch} ch x 2 sources) '
          f'on {args.workers} workers')

    # astro-index -> row position in the output arrays
    pos = {j: r for r, j in enumerate(ASTRO_IDX)}
    mi_t = np.full((len(ASTRO_IDX), n_ch, H, W), np.nan, np.float32)
    mi_b = np.full((len(ASTRO_IDX), n_ch, H, W), np.nan, np.float32)

    t0 = time.time()
    import multiprocessing as mp

    if args.workers <= 1:
        _init_worker(stack_t, stack_b, stack_par, args.neighbors)
        results = (_pixel_map(t) for t in tasks)
        for n_done, (j, c, src, m) in enumerate(results, 1):
            (mi_t if src == 't' else mi_b)[pos[j], c] = m
            if n_done % 20 == 0:
                print(f'  {n_done}/{len(tasks)} tasks, {time.time()-t0:.0f}s')
    else:
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(stack_t, stack_b, stack_par, args.neighbors)) as pool:
            for n_done, (j, c, src, m) in enumerate(
                    pool.imap_unordered(_pixel_map, tasks), 1):
                (mi_t if src == 't' else mi_b)[pos[j], c] = m
                if n_done % 20 == 0:
                    print(f'  {n_done}/{len(tasks)} tasks, {time.time()-t0:.0f}s')

    np.savez_compressed(
        out_path, mi_t=mi_t, mi_b=mi_b,
        astro_idx=np.asarray(ASTRO_IDX), n_sim=n_sim)
    print(f'wrote {out_path}  ({mi_t.shape})  in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
