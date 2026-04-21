"""Dataset and normalization for cosmological baryonic field painting."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class NormStats:
    """Per-channel normalization statistics for log10(1+x) transformed fields."""
    target_mean: np.ndarray   # (3,)
    target_std: np.ndarray    # (3,)
    cond_mean: float = 0.0
    cond_std: float = 1.0
    ls_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    ls_std: np.ndarray = field(default_factory=lambda: np.ones(3))    # (3,)
    param_min: np.ndarray = field(default_factory=lambda: np.zeros(35))
    param_max: np.ndarray = field(default_factory=lambda: np.ones(35))

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def log_transform(x):
    """log10(1 + x) safe for zero values."""
    return np.log10(1.0 + x)


def compute_norm_stats(file_list, n_samples=5000, seed=42):
    """Compute normalization statistics from a random subset of files."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(file_list), min(n_samples, len(file_list)), replace=False)

    targets, conds, large_scales, params = [], [], [], []
    for idx in indices:
        d = np.load(file_list[idx])
        targets.append(log_transform(d['target']))        # (3,128,128)
        conds.append(log_transform(d['condition']))        # (128,128)
        large_scales.append(log_transform(d['large_scale']))  # (3,128,128)
        params.append(d['params'])

    targets = np.stack(targets)       # (N,3,128,128)
    conds = np.stack(conds)           # (N,128,128)
    large_scales = np.stack(large_scales)  # (N,3,128,128)
    params = np.stack(params)         # (N,35)

    target_mean = targets.mean(axis=(0, 2, 3)).astype(np.float32)
    target_std = targets.std(axis=(0, 2, 3)).astype(np.float32)

    return NormStats(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=float(conds.mean()),
        cond_std=float(conds.std()),
        ls_mean=large_scales.mean(axis=(0, 2, 3)).astype(np.float32),
        ls_std=large_scales.std(axis=(0, 2, 3)).astype(np.float32),
        param_min=params.min(axis=0).astype(np.float32),
        param_max=params.max(axis=0).astype(np.float32),
    )


def load_file_list(data_root, split='train'):
    """Load file paths from the precomputed no-lowmass cache."""
    cache = Path(data_root) / split / 'file_list_cache_no_lowmass.txt'
    with open(cache) as f:
        return [line.strip() for line in f if line.strip()]


class AstroDataset(Dataset):
    """Dataset for cosmological baryonic field painting.
    
    Returns dict with keys: target (3,128,128), condition (1,128,128),
    large_scale (3,128,128), params (35,) — all normalized.
    """

    def __init__(self, file_list, norm_stats):
        self.file_list = file_list
        self.ns = norm_stats

    def __len__(self):
        return len(self.file_list)

    def _normalize(self, x, mean, std):
        return (x - mean) / (std + 1e-8)

    def _normalize_params(self, p):
        rang = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (rang + 1e-8)

    def __getitem__(self, idx):
        d = np.load(self.file_list[idx])

        # Log-transform and standardize
        target = log_transform(d['target'])  # (3,128,128)
        target = self._normalize(target, self.ns.target_mean[:, None, None],
                                 self.ns.target_std[:, None, None])

        cond = log_transform(d['condition'])[None]  # (1,128,128)
        cond = self._normalize(cond, self.ns.cond_mean, self.ns.cond_std)

        ls = log_transform(d['large_scale'])  # (3,128,128)
        ls = self._normalize(ls, self.ns.ls_mean[:, None, None],
                             self.ns.ls_std[:, None, None])

        params = self._normalize_params(d['params'].astype(np.float32))

        return {
            'target': torch.from_numpy(target.astype(np.float32)),
            'condition': torch.from_numpy(cond.astype(np.float32)),
            'large_scale': torch.from_numpy(ls.astype(np.float32)),
            'params': torch.from_numpy(params),
        }


def get_loaders(data_root, norm_stats, batch_size=64, num_workers=8,
                max_train=None, max_test=None):
    """Create train and test dataloaders."""
    train_files = load_file_list(data_root, 'train')
    test_files = load_file_list(data_root, 'test')
    if max_train:
        train_files = train_files[:max_train]
    if max_test:
        test_files = test_files[:max_test]

    train_ds = AstroDataset(train_files, norm_stats)
    test_ds = AstroDataset(test_files, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
