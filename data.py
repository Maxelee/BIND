"""Dataset and normalization for cosmological baryonic field painting."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field


SB35_CSV = '/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv'
_sb_meta = pd.read_csv(SB35_CSV)
PARAM_LOG_FLAG = _sb_meta['LogFlag'].to_numpy().astype(np.int32)        # (35,)
PARAM_MIN_RAW  = _sb_meta['MinVal'].to_numpy().astype(np.float64)        # (35,)
PARAM_MAX_RAW  = _sb_meta['MaxVal'].to_numpy().astype(np.float64)        # (35,)
# Log-space bounds for log-flagged params; linear bounds otherwise.
# Use the linear value as a placeholder for non-log-flagged params before
# np.where selects, to avoid log10(negative) warnings on irrelevant entries.
_log_safe_min = np.where(PARAM_LOG_FLAG == 1, PARAM_MIN_RAW, 1.0)
_log_safe_max = np.where(PARAM_LOG_FLAG == 1, PARAM_MAX_RAW, 1.0)
PARAM_MIN_NORM = np.where(PARAM_LOG_FLAG == 1,
                          np.log10(_log_safe_min),
                          PARAM_MIN_RAW).astype(np.float32)
PARAM_MAX_NORM = np.where(PARAM_LOG_FLAG == 1,
                          np.log10(_log_safe_max),
                          PARAM_MAX_RAW).astype(np.float32)


def _default_param_log_flag():
    return PARAM_LOG_FLAG.astype(np.int32)


@dataclass
class NormStats:
    """Per-channel normalization statistics for log10(1+x) transformed fields.

    The Stars channel can optionally be split into two heads
    (occupancy + conditional density) when ``stars_two_head=True``. In that
    mode the AstroDataset emits a 4-channel target [DM_hydro, Gas,
    occupancy_norm, density_norm] and the model has out_ch=4.
    Inference (test_suite/pipeline.py) recombines them via a soft multiplier
    before writing the standard 3-channel artifact to disk.
    """
    target_mean: np.ndarray   # (3,)
    target_std: np.ndarray    # (3,)
    cond_mean: float = 0.0
    cond_std: float = 1.0
    ls_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    ls_std: np.ndarray = field(default_factory=lambda: np.ones(3))    # (3,)
    # param bounds: in log10 space for params with PARAM_LOG_FLAG[j]==1.
    param_min: np.ndarray = field(default_factory=lambda: PARAM_MIN_NORM.copy())
    param_max: np.ndarray = field(default_factory=lambda: PARAM_MAX_NORM.copy())
    param_log_flag: np.ndarray = field(default_factory=_default_param_log_flag)
    # Stars two-head fields (used iff stars_two_head=True). Defaults are safe
    # passthroughs; old norm_stats.npz files load with stars_two_head=False
    # and the rest of the pipeline behaves identically to single-head.
    stars_two_head: bool = False
    stars_occ_mean: float = 0.0     # E[occ]; standardize occ → (occ-μ)/σ
    stars_occ_std: float = 1.0      # sqrt(p(1-p)) when computed from data
    stars_cond_mean: float = 0.0    # mean of log10(1+stars) on occupied pixels
    stars_cond_std: float = 1.0     # std  of log10(1+stars) on occupied pixels

    def save(self, path):
        # np.savez stores bools as 0-d arrays — cast on load.
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        # Backward compatibility: older norm_stats.npz lacks fields added later.
        # We default any missing key to the dataclass default.
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
            # other missing keys fall back to the dataclass default
        return cls(**kwargs)


def log_transform(x):
    """log10(1 + x) safe for zero values."""
    return np.log10(1.0 + x)


def compute_norm_stats(file_list, n_samples=5000, seed=42, stars_two_head=False):
    """Compute normalization statistics from a random subset of files.

    When ``stars_two_head=True`` we additionally compute statistics for the
    Stars two-head training:
      - stars_occ_mean / stars_occ_std: standardization for the binary
        occupancy mask (mean = occupied fraction, std = sqrt(p*(1-p)))
      - stars_cond_mean / stars_cond_std: mean & std of log10(1+stars)
        computed only over occupied pixels (avoids the zero-pixel
        domination that biases the standard target_mean[2] / target_std[2]).
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(file_list), min(n_samples, len(file_list)), replace=False)

    targets, conds, large_scales, params = [], [], [], []
    raw_stars_chunks = []  # only populated when stars_two_head
    for idx in indices:
        d = np.load(file_list[idx])
        targets.append(log_transform(d['target']))        # (3,128,128)
        conds.append(log_transform(d['condition']))        # (128,128)
        large_scales.append(log_transform(d['large_scale']))  # (3,128,128)
        params.append(d['params'])
        if stars_two_head:
            raw_stars_chunks.append(d['target'][2])        # raw mass density

    targets = np.stack(targets)       # (N,3,128,128)
    conds = np.stack(conds)           # (N,128,128)
    large_scales = np.stack(large_scales)  # (N,3,128,128)
    params = np.stack(params)         # (N,35)

    target_mean = targets.mean(axis=(0, 2, 3)).astype(np.float32)
    target_std = targets.std(axis=(0, 2, 3)).astype(np.float32)

    extra = {}
    if stars_two_head:
        raw_stars = np.stack(raw_stars_chunks)              # (N,128,128)
        occ = (raw_stars > 0).astype(np.float32)
        p = float(occ.mean())                                # occupied fraction
        # std of a Bernoulli; clip to avoid std=0 when the sample is degenerate
        occ_std = float(np.sqrt(max(p * (1 - p), 1e-8)))
        log_density = np.log10(1.0 + raw_stars)
        occ_mask = occ > 0
        if occ_mask.any():
            cond_mean = float(log_density[occ_mask].mean())
            cond_std  = float(log_density[occ_mask].std())
        else:
            cond_mean, cond_std = 0.0, 1.0
        extra = {
            'stars_two_head': True,
            'stars_occ_mean': p,
            'stars_occ_std':  occ_std,
            'stars_cond_mean': cond_mean,
            'stars_cond_std':  max(cond_std, 1e-6),
        }

    return NormStats(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=float(conds.mean()),
        cond_std=float(conds.std()),
        ls_mean=large_scales.mean(axis=(0, 2, 3)).astype(np.float32),
        ls_std=large_scales.std(axis=(0, 2, 3)).astype(np.float32),
        # Bounds come from the SB35 csv (log10 where flagged) so the
        # normalization range is well-defined for any sim, not just the
        # training subset. See note in NormStats docstring above.
        param_min=PARAM_MIN_NORM.copy(),
        param_max=PARAM_MAX_NORM.copy(),
        param_log_flag=PARAM_LOG_FLAG.astype(np.int32),
        **extra,
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
        # log10 first for log-flagged params, then min/max scale to [0, 1].
        # Bounds in self.ns.param_min/max are already in log space for the
        # log-flagged entries.
        p = np.where(self.ns.param_log_flag == 1,
                     np.log10(np.maximum(p, 1e-30)),
                     p)
        rang = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (rang + 1e-8)

    def _build_target(self, raw_target):
        """Return either (3, H, W) standard or (4, H, W) two-head target.

        Two-head layout: [DM_hydro, Gas, occupancy_norm, density_norm].
        For unoccupied pixels the conditional density_norm is set to 0
        (the model is free to predict anything there because it gets
        masked at inference; using 0 keeps gradients well-behaved).
        """
        if not self.ns.stars_two_head:
            target = log_transform(raw_target)              # (3,H,W)
            return self._normalize(
                target, self.ns.target_mean[:, None, None],
                self.ns.target_std[:, None, None],
            )

        # Standard standardization for DM_hydro and Gas
        log_dm = log_transform(raw_target[0])
        log_gas = log_transform(raw_target[1])
        dm = (log_dm - self.ns.target_mean[0]) / (self.ns.target_std[0] + 1e-8)
        gas = (log_gas - self.ns.target_mean[1]) / (self.ns.target_std[1] + 1e-8)

        # Stars split into (occupancy, conditional log-density)
        raw_stars = raw_target[2]
        occ = (raw_stars > 0).astype(np.float32)
        log_density = log_transform(raw_stars)
        occ_norm = (occ - self.ns.stars_occ_mean) / (self.ns.stars_occ_std + 1e-8)
        density_norm = (log_density - self.ns.stars_cond_mean) / (
            self.ns.stars_cond_std + 1e-8
        )
        # Mask conditional density at unoccupied pixels; the soft multiplier
        # at inference will multiply through by ~0 there anyway.
        density_norm = np.where(occ > 0, density_norm, 0.0)

        return np.stack([dm, gas, occ_norm, density_norm]).astype(np.float32)

    def __getitem__(self, idx):
        d = np.load(self.file_list[idx])

        target = self._build_target(d['target'])

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
