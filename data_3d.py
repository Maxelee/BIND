"""3D dataset and normalization for cosmological baryonic field painting.

Each .npz holds:
  condition: (D, H, W)        — DMO mass density volume
  target:    (3, D, H, W)     — (DM_hydro, Gas, Stars) volumes
  params:    (36,)            — last column is a constant placeholder; we
                                slice to the first 35 to match SB35 csv.
There is no `large_scale` field in the 3D dataset.

Two new features over the 2D pipeline:
  - Stars two-head split (occupancy + conditional density), as in data.py.
  - Optional zero-voxel interpolation on DM_hydro / Gas targets and on
    the DMO condition, via a mask-aware Gaussian smoothing of the
    log-transformed field. ~95% of Stars voxels are empty, so Stars is
    left raw (or two-headed).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter

from data import (
    SB35_CSV, PARAM_LOG_FLAG, PARAM_MIN_NORM, PARAM_MAX_NORM,
    log_transform, _default_param_log_flag,
)


@dataclass
class NormStats3d:
    """Per-channel normalization statistics for log10(1+x) volumes.

    The Stars channel can be split into (occupancy, conditional density)
    when ``stars_two_head=True``. In that mode AstroDataset3d emits a
    4-channel target [DM_hydro, Gas, occupancy_norm, density_norm].
    """
    target_mean: np.ndarray  # (3,)
    target_std: np.ndarray   # (3,)
    cond_mean: float = 0.0
    cond_std: float = 1.0
    param_min: np.ndarray = field(default_factory=lambda: PARAM_MIN_NORM.copy())
    param_max: np.ndarray = field(default_factory=lambda: PARAM_MAX_NORM.copy())
    param_log_flag: np.ndarray = field(default_factory=_default_param_log_flag)
    # Stars two-head fields
    stars_two_head: bool = False
    stars_occ_mean: float = 0.0
    stars_occ_std: float = 1.0
    stars_cond_mean: float = 0.0
    stars_cond_std: float = 1.0
    # Zero-voxel interpolation settings (recorded so inference matches training)
    interp_empty: bool = False
    interp_sigma: float = 1.5

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        kwargs = {}
        for k, fld in cls.__dataclass_fields__.items():
            if k in d.files:
                v = d[k]
                if fld.type is bool or fld.type == 'bool':
                    v = bool(v)
                elif fld.type is float or fld.type == 'float':
                    v = float(v)
                kwargs[k] = v
            elif k == 'param_log_flag':
                kwargs[k] = PARAM_LOG_FLAG.astype(np.int32)
        return cls(**kwargs)


def fill_zeros_smooth(x, sigma=1.5, eps=1e-8):
    """Fill zero entries of ``x`` by mask-aware Gaussian smoothing.

    Computes a smoothed estimate
        smoothed(p) = sum_q G(p-q; sigma) * mask(q) * x(q)
                      / sum_q G(p-q; sigma) * mask(q)
    where mask = (x > 0). This is the local mean of non-zero neighbours,
    so empty voxels get an interpolated value drawn from nearby occupied
    ones. Voxels that are already non-zero are left untouched.
    """
    mask = (x > 0).astype(np.float32)
    if mask.all():
        return x.astype(np.float32, copy=False)
    if not mask.any():
        return x.astype(np.float32, copy=False)
    num = gaussian_filter(x * mask, sigma=sigma, mode='nearest')
    den = gaussian_filter(mask, sigma=sigma, mode='nearest')
    smoothed = num / (den + eps)
    return np.where(mask > 0, x, smoothed).astype(np.float32, copy=False)


def load_file_list_3d(data_root, split='train',
                      filename_pattern='file_list_cache_3d_{split}_frac0p900_seed42.txt'):
    """Load file paths from the 3D file_list_cache."""
    fname = filename_pattern.format(split=split)
    cache = Path(data_root) / fname
    if not cache.exists():
        candidates = sorted(Path(data_root).glob(f'file_list_cache_3d_{split}_*.txt'))
        if not candidates:
            raise FileNotFoundError(
                f'No 3D file list cache found in {data_root} for split {split}'
            )
        cache = candidates[0]
    with open(cache) as f:
        return [line.strip() for line in f if line.strip()]


def compute_norm_stats_3d(file_list, n_samples=2000, seed=42,
                          stars_two_head=False, interp_empty=False,
                          interp_sigma=1.5):
    """Compute normalization stats from a random subset of 3D volumes.

    Streams sums to avoid stacking large volumes in RAM. Stats are computed
    on the same log-transformed (and, when interp_empty, zero-filled) fields
    that the dataset will produce at training time, so the per-channel
    normalization is tight.
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(file_list))
    indices = rng.choice(len(file_list), n, replace=False)

    sum_t = np.zeros(3, dtype=np.float64)
    sumsq_t = np.zeros(3, dtype=np.float64)
    sum_c = 0.0
    sumsq_c = 0.0
    n_voxels_target_per = 0
    n_voxels_cond_per = 0

    # Stars two-head accumulators (over occupied voxels for conditional density)
    occ_sum = 0.0
    n_occ_voxels = 0
    star_cond_sum = 0.0
    star_cond_sumsq = 0.0

    for idx in indices:
        d = np.load(file_list[idx])
        raw_target = d['target']                # (3, D, H, W)
        raw_cond = d['condition']               # (D, H, W)

        log_t = log_transform(raw_target).astype(np.float64)
        log_c = log_transform(raw_cond).astype(np.float64)
        if interp_empty:
            # Match what the dataset will see at training time:
            # fill DM_hydro, Gas (target ch 0,1) and the DMO condition.
            log_t[0] = fill_zeros_smooth(log_t[0].astype(np.float32),
                                         sigma=interp_sigma).astype(np.float64)
            log_t[1] = fill_zeros_smooth(log_t[1].astype(np.float32),
                                         sigma=interp_sigma).astype(np.float64)
            log_c = fill_zeros_smooth(log_c.astype(np.float32),
                                      sigma=interp_sigma).astype(np.float64)

        n_voxels_target_per = log_t[0].size
        n_voxels_cond_per = log_c.size
        sum_t += log_t.sum(axis=(1, 2, 3))
        sumsq_t += (log_t ** 2).sum(axis=(1, 2, 3))
        sum_c += log_c.sum()
        sumsq_c += (log_c ** 2).sum()

        if stars_two_head:
            raw_stars = raw_target[2]
            occ = (raw_stars > 0)
            occ_sum += float(occ.sum())
            n_occ_voxels += int(occ.size)
            if occ.any():
                logd = np.log10(1.0 + raw_stars[occ]).astype(np.float64)
                star_cond_sum += logd.sum()
                star_cond_sumsq += (logd ** 2).sum()

    Nt = n * n_voxels_target_per
    Nc = n * n_voxels_cond_per
    target_mean = (sum_t / Nt).astype(np.float32)
    target_std = np.sqrt(np.maximum(sumsq_t / Nt - target_mean.astype(np.float64) ** 2,
                                    1e-12)).astype(np.float32)
    cond_mean = float(sum_c / Nc)
    cond_std = float(np.sqrt(max(sumsq_c / Nc - cond_mean ** 2, 1e-12)))

    extra = {}
    if stars_two_head:
        p = float(occ_sum / max(n_occ_voxels, 1))
        occ_std = float(np.sqrt(max(p * (1 - p), 1e-8)))
        n_occ = max(int(occ_sum), 1)
        cmean = float(star_cond_sum / n_occ) if n_occ > 0 else 0.0
        cvar = max(star_cond_sumsq / n_occ - cmean ** 2, 1e-12) if n_occ > 0 else 1.0
        cstd = float(np.sqrt(cvar))
        extra = {
            'stars_two_head': True,
            'stars_occ_mean': p,
            'stars_occ_std': occ_std,
            'stars_cond_mean': cmean,
            'stars_cond_std': max(cstd, 1e-6),
        }

    return NormStats3d(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        param_min=PARAM_MIN_NORM.copy(),
        param_max=PARAM_MAX_NORM.copy(),
        param_log_flag=PARAM_LOG_FLAG.astype(np.int32),
        interp_empty=interp_empty,
        interp_sigma=float(interp_sigma),
        **extra,
    )


class AstroDataset3d(Dataset):
    """3D dataset.

    Returns (single-head, default):
      target:    (3, D, H, W)  normalized
      condition: (1, D, H, W)  normalized
      params:    (35,)         normalized to [0, 1]

    With ``norm_stats.stars_two_head=True``:
      target:    (4, D, H, W) = [DM_hydro, Gas, occupancy_norm, density_norm]

    If ``norm_stats.interp_empty`` is set, channels 0 (DM_hydro) and 1
    (Gas) of the log-transformed target, plus the DMO condition, are
    filled at empty voxels via a mask-aware Gaussian blur before
    normalization. Stars is never filled (it's ~95% empty; the two-head
    split is the right tool there).
    """

    def __init__(self, file_list, norm_stats):
        self.file_list = file_list
        self.ns = norm_stats

    def __len__(self):
        return len(self.file_list)

    def _normalize(self, x, mean, std):
        return (x - mean) / (std + 1e-8)

    def _normalize_params(self, p):
        # Slice to first 35: the 36th column in this dataset is a constant
        # placeholder (50000.0 across all sims/halos) — drop it.
        p = p[:35]
        p = np.where(self.ns.param_log_flag == 1,
                     np.log10(np.maximum(p, 1e-30)),
                     p)
        rang = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (rang + 1e-8)

    def _build_target(self, raw_target):
        """Return either (3, D, H, W) standard or (4, D, H, W) two-head.

        Two-head layout: [DM_hydro, Gas, occupancy_norm, density_norm].
        """
        log_dm = log_transform(raw_target[0]).astype(np.float32)
        log_gas = log_transform(raw_target[1]).astype(np.float32)
        if self.ns.interp_empty:
            log_dm = fill_zeros_smooth(log_dm, sigma=self.ns.interp_sigma)
            log_gas = fill_zeros_smooth(log_gas, sigma=self.ns.interp_sigma)

        dm = (log_dm - self.ns.target_mean[0]) / (self.ns.target_std[0] + 1e-8)
        gas = (log_gas - self.ns.target_mean[1]) / (self.ns.target_std[1] + 1e-8)

        if not self.ns.stars_two_head:
            log_stars = log_transform(raw_target[2]).astype(np.float32)
            stars = (log_stars - self.ns.target_mean[2]) / (self.ns.target_std[2] + 1e-8)
            return np.stack([dm, gas, stars]).astype(np.float32)

        raw_stars = raw_target[2]
        occ = (raw_stars > 0).astype(np.float32)
        log_density = log_transform(raw_stars).astype(np.float32)
        occ_norm = (occ - self.ns.stars_occ_mean) / (self.ns.stars_occ_std + 1e-8)
        density_norm = (log_density - self.ns.stars_cond_mean) / (
            self.ns.stars_cond_std + 1e-8
        )
        density_norm = np.where(occ > 0, density_norm, 0.0).astype(np.float32)
        return np.stack([dm, gas, occ_norm, density_norm]).astype(np.float32)

    def __getitem__(self, idx):
        d = np.load(self.file_list[idx])

        target = self._build_target(d['target'])

        log_c = log_transform(d['condition']).astype(np.float32)  # (D, H, W)
        if self.ns.interp_empty:
            log_c = fill_zeros_smooth(log_c, sigma=self.ns.interp_sigma)
        cond = log_c[None]  # (1, D, H, W)
        cond = self._normalize(cond, self.ns.cond_mean, self.ns.cond_std).astype(np.float32)

        params = self._normalize_params(d['params'].astype(np.float32))

        return {
            'target': torch.from_numpy(target),
            'condition': torch.from_numpy(cond),
            'params': torch.from_numpy(params.astype(np.float32)),
        }


def get_loaders_3d(data_root, norm_stats, batch_size=1, num_workers=4,
                   max_train=None, max_test=None):
    train_files = load_file_list_3d(data_root, 'train')
    test_files = load_file_list_3d(data_root, 'test')
    if max_train:
        train_files = train_files[:max_train]
    if max_test:
        test_files = test_files[:max_test]

    train_ds = AstroDataset3d(train_files, norm_stats)
    test_ds = AstroDataset3d(test_files, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
