"""
inject_conditional_params_test.py

For each sim_i directory under TEST_ROOT, look up the 35 SB35 parameters from
row i of CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt and write them as the
'conditional_params' key into every .npz file in that directory.

Files are updated atomically: the new version is written to a temp file first,
then os.replace() is called so partial writes never corrupt existing data.

Usage:
    python inject_conditional_params_test.py [--dry-run] [--workers N]
"""

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SB35_TXT = (
    '/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/'
    'CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt'
)
TEST_ROOT = Path('/mnt/home/mlee1/ceph/diffusion_new/test')


def load_sb35_table(path):
    """Return ndarray shape (N, 35): raw param values indexed by sim number."""
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f):
            if lineno == 0:
                continue          # skip header
            cols = line.split()
            # cols[0]=SB35_i  cols[1:-1]=35 params  cols[-1]=seed
            rows.append([float(v) for v in cols[1:-1]])
    arr = np.array(rows, dtype=np.float32)
    assert arr.shape[1] == 35, f'Expected 35 param cols, got {arr.shape[1]}'
    return arr


def inject_file(npz_path, params):
    """Add/overwrite conditional_params in a single .npz file (atomic write)."""
    d = np.load(npz_path)
    data = {k: d[k] for k in d.files}
    data['conditional_params'] = params
    dir_ = os.path.dirname(npz_path)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.npz')
    try:
        os.close(fd)
        np.savez(tmp, **data)
        os.replace(tmp, npz_path)
    except Exception:
        os.unlink(tmp)
        raise


def process_sim_dir(args):
    sim_dir, sim_idx, params = args
    n_ok = n_err = 0
    for npz_path in sorted(Path(sim_dir).glob('*.npz')):
        try:
            inject_file(str(npz_path), params)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f'  ERROR {npz_path}: {e}', file=sys.stderr)
    return sim_dir, n_ok, n_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    print(f'Loading SB35 param table from {SB35_TXT} ...')
    table = load_sb35_table(SB35_TXT)
    print(f'  Loaded {len(table)} rows × 35 params')

    sim_dirs = sorted(TEST_ROOT.iterdir())
    print(f'Found {len(sim_dirs)} sim directories under {TEST_ROOT}')

    tasks = []
    for sim_dir in sim_dirs:
        m = re.fullmatch(r'sim_(\d+)', sim_dir.name)
        if m is None:
            print(f'  Skipping non-sim dir: {sim_dir.name}')
            continue
        sim_idx = int(m.group(1))
        if sim_idx >= len(table):
            print(f'  WARNING: sim_{sim_idx} out of table range ({len(table)} rows), skipping')
            continue
        tasks.append((str(sim_dir), sim_idx, table[sim_idx]))

    print(f'Injecting conditional_params into {len(tasks)} sim dirs with {args.workers} workers ...')

    total_ok = total_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_sim_dir, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            sim_dir, n_ok, n_err = fut.result()
            total_ok  += n_ok
            total_err += n_err
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f'  [{done}/{len(tasks)}] files ok={total_ok} err={total_err}')

    print(f'\nDone. files written={total_ok}  errors={total_err}')
    if total_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
