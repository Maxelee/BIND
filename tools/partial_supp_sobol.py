"""Parameter-resolved, field-level halo-MASKING decomposition of the matter-power
suppression across the 256-point Sobol feedback cube.

For each design and each halo subset (mass decade x gas-fraction-at-fixed-mass
tercile) we paste back into the full 50 Mpc/h DMO box and measure the total-matter
suppression S(k)=P/P_DMO, sim-averaged, two ways:
  S_sub : ONLY the subset painted   ("first-in" marginal; over-states |dS|)
  S_loo : ALL BUT the subset painted ("last-in"  marginal; under-states |dS|)
plus the full S(k) (all halos).  The 2-bracket Shapley contribution
c_s = 0.5*[(S_sub-1) + (S_full - S_loo)] is ~additive (sum_s c_s ~ S_full-1), so
downstream it licenses an absolute *variance attribution* across the 256 designs:
which halo population subset drives Var_theta[S(k)] at the weak-lensing scales.

Why this is tractable: every design's per-halo BIND patches already exist on ceph
(`sobol_ss_cv/maps/gen_design{d}.npz`, (1111,7,128,128)); we only re-composite
subsets, never re-run the emulator or a hydro sim.

Subsets (9 = 3 mass x 3 gas):
  mass decades (fixed)        : [13,13.5) [13.5,14) [14,15)
  gas terciles (per design)   : within each mass decade, by f_gas at that theta
A subset's membership in the gas axis is design-dependent (a halo is "gas-poor"
*at a given feedback point*); the mass axis is fixed.

Reuses the composite/paste primitives in tools/box_supp_sobol.py verbatim.

Run:
  # thin-slice prototype (every 8th design -> 32 designs), single process
  python tools/partial_supp_sobol.py --designs stride:8 --out partial_supp_proto.npz
  # full campaign as a SLURM array (user submits run_partial_supp.sh), then reduce
  python tools/partial_supp_sobol.py --designs all --n_chunks 16 --chunk_id ${SLURM_ARRAY_TASK_ID}
  python tools/partial_supp_sobol.py --reduce
"""
from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path

import numpy as np
import Pk_library as PKL

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_supp_sobol as B  # composite/paste primitives + load_sim_static

CACHE = Path('analysis_physics_cache')
SHARD_DIR = CACHE / 'partial_supp_shards'
OUT = CACHE / 'partial_supp_sobol.npz'

MASS_EDGES = np.array([13.0, 13.5, 14.0, 15.0])   # -> 3 mass decades
N_MASS = 3
N_GAS = 3
N_SUB = N_MASS * N_GAS
SUBSET_LABELS = [f'M{m}_G{g}' for m in range(N_MASS) for g in range(N_GAS)]
MASS_LABELS = ['[13,13.5)', '[13.5,14)', '[14,15)']
GAS_LABELS = ['gas-poor', 'gas-mid', 'gas-rich']


def pk2d(field):
    """Total-matter 2D P(k); quiet (no PKL banner)."""
    delta = np.asarray(field, dtype=np.float32) / np.mean(field)
    P = PKL.Pk_plane(delta, B.BOX, 'CIC', 1, verbose=False)
    return P.k, P.Pk


def full_pk(sim, patches3):
    bundle = B.build_bind_composite(sim['dmo'], sim['halos'], patches3, sim['cuts'],
                                    box_size=B.BOX, npix=B.NPIX, patch_pix=B.PATCH,
                                    patch_mass_match=True, taper_frac=0.15)
    return pk2d(B.total_matter(bundle['composite']))[1]


def subset_pk(sim, patches3, mask):
    """Paste ONLY the masked halos (rest of the box stays DMO)."""
    if not mask.any():
        return sim['pk_dmo']            # nothing painted -> S = 1
    halos = [h for h, m in zip(sim['halos'], mask) if m]
    cuts = [c for c, m in zip(sim['cuts'], mask) if m]
    bundle = B.build_bind_composite(sim['dmo'], halos, patches3[mask], cuts,
                                    box_size=B.BOX, npix=B.NPIX, patch_pix=B.PATCH,
                                    patch_mass_match=True, taper_frac=0.15)
    return pk2d(B.total_matter(bundle['composite']))[1]


def assign_subsets(logM, fgas):
    """Per-design subset id (0..8) for each halo; -1 if outside [13,15)."""
    mb = np.digitize(logM, MASS_EDGES)              # 1,2,3 inside; 0/4 outside
    sid = np.full(len(logM), -1, dtype=np.int64)
    for m in range(1, N_MASS + 1):
        sel = mb == m
        if not sel.any():
            continue
        fg = fgas[sel]
        t = np.nanpercentile(fg[np.isfinite(fg)], [100 / 3, 200 / 3])
        gl = np.clip(np.digitize(fg, t), 0, N_GAS - 1)   # 0,1,2 within the decade
        sid[sel] = (m - 1) * N_GAS + gl
    return sid


def parse_designs(spec, n):
    if spec == 'all':
        return list(range(n))
    if spec.startswith('stride:'):
        return list(range(0, n, int(spec.split(':')[1])))
    return [int(x) for x in spec.split(',')]


def reduce_shards():
    shards = sorted(SHARD_DIR.glob('chunk*.npz'))
    if not shards:
        raise SystemExit(f'no shards in {SHARD_DIR}')
    k_box = None
    rows = {}
    for s in shards:
        z = np.load(s, allow_pickle=True)
        k_box = z['k_box']
        for j, d in enumerate(z['design_ids']):
            rows[int(d)] = (z['S_full'][j], z['S_sub'][j], z['S_loo'][j], z['n_sub'][j])
    dids = np.array(sorted(rows))
    S_full = np.stack([rows[d][0] for d in dids])
    S_sub = np.stack([rows[d][1] for d in dids])
    S_loo = np.stack([rows[d][2] for d in dids])
    n_sub = np.stack([rows[d][3] for d in dids])
    np.savez_compressed(OUT, design_ids=dids, k_box=k_box, S_full=S_full,
                        S_sub=S_sub, S_loo=S_loo, n_sub=n_sub, mass_edges=MASS_EDGES,
                        subset_labels=np.array(SUBSET_LABELS),
                        mass_labels=np.array(MASS_LABELS),
                        gas_labels=np.array(GAS_LABELS))
    print(f'reduced {len(shards)} shards -> {OUT}  ({len(dids)} designs)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', default='all')
    ap.add_argument('--n_chunks', type=int, default=1)
    ap.add_argument('--chunk_id', type=int, default=0)
    ap.add_argument('--out', default=None, help='single-file output (skips shards)')
    ap.add_argument('--reduce', action='store_true')
    args = ap.parse_args()

    if args.reduce:
        reduce_shards()
        return

    cube = np.load(B.SOBOL_ROOT / 'cube.npz', allow_pickle=True)
    obs = [str(s) for s in cube['obs_names']]
    fgi = obs.index('f_gas')
    logM_all = np.log10(np.asarray(cube['M200'], float))
    n_design = cube['obs'].shape[0]

    dids = parse_designs(args.designs, n_design)
    dids = dids[args.chunk_id::args.n_chunks]      # round-robin chunking
    map_files = sorted((B.SOBOL_ROOT / 'maps').glob('gen_design*.npz'))

    print(f'designs this task: {len(dids)}  {dids[:6]}{"..." if len(dids) > 6 else ""}',
          flush=True)
    print('loading sim static (DMO boxes + truth Pk) ...', flush=True)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sims, _ = B.load_sim_static()
    k_box = sims[0]['k']

    S_full = np.full((len(dids), len(k_box)), np.nan)
    S_sub = np.full((len(dids), N_SUB, len(k_box)), np.nan)   # subset-only paste
    S_loo = np.full((len(dids), N_SUB, len(k_box)), np.nan)   # leave-one-subset-out
    n_sub = np.zeros((len(dids), N_SUB), dtype=np.int64)

    for j, d in enumerate(dids):
        t0 = time.time()
        gen = np.load(map_files[d])['generated'][:, :3].astype(np.float32)
        sid = assign_subsets(logM_all, cube['obs'][d, :, fgi].astype(float))
        S_full[j] = np.mean([full_pk(s, gen[s['block']]) / s['pk_dmo'] for s in sims], 0)
        for sub in range(N_SUB):
            rt, rl = [], []
            for s in sims:
                blk = gen[s['block']]
                m = sid[s['block']] == sub
                n_sub[j, sub] += int(m.sum())
                rt.append(subset_pk(s, blk, m) / s['pk_dmo'])          # only subset
                rl.append(subset_pk(s, blk, ~m) / s['pk_dmo'])         # all but subset
            S_sub[j, sub] = np.mean(rt, 0)
            S_loo[j, sub] = np.mean(rl, 0)
        ik10 = int(np.argmin(np.abs(k_box - 10.0)))
        print(f'[{j + 1}/{len(dids)}] design {d:4d}  S_full(k10)={S_full[j][ik10]:.3f}  '
              f'({time.time() - t0:.1f}s)', flush=True)

    if args.out:
        outp = CACHE / args.out
        np.savez_compressed(outp, design_ids=np.array(dids), k_box=k_box,
                            S_full=S_full, S_sub=S_sub, S_loo=S_loo, n_sub=n_sub,
                            mass_edges=MASS_EDGES,
                            subset_labels=np.array(SUBSET_LABELS),
                            mass_labels=np.array(MASS_LABELS),
                            gas_labels=np.array(GAS_LABELS))
        print(f'wrote {outp}')
    else:
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        shard = SHARD_DIR / f'chunk{args.chunk_id:03d}.npz'
        np.savez_compressed(shard, design_ids=np.array(dids), k_box=k_box,
                            S_full=S_full, S_sub=S_sub, S_loo=S_loo, n_sub=n_sub)
        print(f'wrote {shard}')


if __name__ == '__main__':
    main()
