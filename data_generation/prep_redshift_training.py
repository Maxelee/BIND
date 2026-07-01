"""
prep_redshift_training.py
=========================
One-time, single-process prep for a --condition_redshift training run on the
multi-redshift dataset. Doing this once up front avoids having all 8 DDP ranks
redundantly (a) recursively rglob the dataset to build the file cache and
(b) compute norm stats over 10k samples — both of which would otherwise race on
their output files at startup.

Builds:
  1. <data_root>/train/file_list_cache_multiz.txt   (recursive enumeration)
  2. <data_root>/test/file_list_cache_multiz.txt
  3. <run_dir>/norm_stats.npz   (stars_two_head + predict_thermo, all redshifts)

After this, `sbatch run_train.sh` (REDSHIFT=1) just reads all three.
"""
import argparse
from pathlib import Path

from bind.data import load_file_list, compute_norm_stats, NormStats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/mnt/home/mlee1/ceph/train_data_multiz_128_cpu')
    ap.add_argument('--run_dir', default='/mnt/home/mlee1/ceph/fm_runs/fm_redshift')
    # 4000 keeps the single-process stack of float64 maps under ~6 GB so this
    # runs inside a modest interactive Slurm cgroup (the training node, with
    # --mem=1000G, can afford the 10000 default; over ~65M pixels/channel the
    # means/stds and the 0.1-percentile thermo floor are already converged).
    ap.add_argument('--n_stats_samples', type=int, default=4000)
    args = ap.parse_args()

    print(f'[prep] building train cache under {args.data_root}/train ...', flush=True)
    train_files = load_file_list(args.data_root, 'train', recursive=True)
    print(f'[prep] train files: {len(train_files)}', flush=True)

    print(f'[prep] building test cache under {args.data_root}/test ...', flush=True)
    test_files = load_file_list(args.data_root, 'test', recursive=True)
    print(f'[prep] test files: {len(test_files)}', flush=True)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / 'norm_stats.npz'
    if stats_path.exists():
        print(f'[prep] norm stats already exist at {stats_path}; skipping', flush=True)
    else:
        print(f'[prep] computing norm stats from {args.n_stats_samples} samples '
              f'(stars_two_head=True, predict_thermo=True) ...', flush=True)
        ns = compute_norm_stats(
            train_files, n_samples=args.n_stats_samples,
            stars_two_head=True, predict_thermo=True,
        )
        ns.save(stats_path)
        print(f'[prep] saved norm stats -> {stats_path}', flush=True)

    # Sanity: reload and report.
    ns = NormStats.load(stats_path)
    print(f'[prep] norm stats: stars_two_head={ns.stars_two_head} '
          f'predict_thermo={ns.predict_thermo}', flush=True)
    print('[prep] done.', flush=True)


if __name__ == '__main__':
    main()
