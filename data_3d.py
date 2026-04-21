"""3D dataset and normalization for volumetric baryonic field painting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormStats3D:
    """Per-channel normalization statistics for log10(1+x) transformed 3D fields."""

    target_mean: np.ndarray
    target_std: np.ndarray
    cond_mean: float = 0.0
    cond_std: float = 1.0
    ls_mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ls_std: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    param_min: np.ndarray = field(default_factory=lambda: np.zeros(36, dtype=np.float32))
    param_max: np.ndarray = field(default_factory=lambda: np.ones(36, dtype=np.float32))

    def save(self, path: str | Path) -> None:
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "NormStats3D":
        data = np.load(path)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def log_transform(x: np.ndarray) -> np.ndarray:
    """Apply log10(1+x), safe for zero-valued voxels."""

    return np.log10(1.0 + x)


def _sim_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    try:
        return int(name.split("_", 1)[1]), name
    except Exception:
        return 10**9, name


def _halo_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1]), stem
    except Exception:
        return 10**9, stem


def _cache_path(data_root: str | Path, split: str, train_fraction: float, split_seed: int) -> Path:
    frac = f"{train_fraction:.3f}".replace(".", "p")
    return Path(data_root) / f"file_list_cache_3d_{split}_frac{frac}_seed{split_seed}.txt"


def _discover_sim_dirs(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    sims = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("sim_")]
    sims.sort(key=_sim_sort_key)
    return sims


def _split_sims(sim_dirs: list[Path], train_fraction: float, split_seed: int) -> tuple[list[Path], list[Path]]:
    if not sim_dirs:
        return [], []
    idx = np.arange(len(sim_dirs))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(idx)
    n_train = max(1, int(round(len(sim_dirs) * train_fraction)))
    train_ids = set(idx[:n_train].tolist())
    train_sims = [sim_dirs[i] for i in range(len(sim_dirs)) if i in train_ids]
    test_sims = [sim_dirs[i] for i in range(len(sim_dirs)) if i not in train_ids]
    if not test_sims:
        # Keep at least one validation sim when possible.
        train_sims, test_sims = train_sims[:-1], train_sims[-1:]
    return train_sims, test_sims


def _collect_halo_files(sim_dirs: list[Path]) -> list[str]:
    files: list[str] = []
    for sim_dir in sim_dirs:
        halos = [p for p in sim_dir.glob("halo_*.npz") if p.is_file()]
        halos.sort(key=_halo_sort_key)
        files.extend(str(p) for p in halos)
    return files


def load_file_list_3d(
    data_root: str | Path,
    split: str = "train",
    train_fraction: float = 0.9,
    split_seed: int = 42,
    refresh_cache: bool = False,
) -> list[str]:
    """Load 3D halo-file paths for train/test split from sim_* directories."""

    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    cache = _cache_path(data_root, split, train_fraction, split_seed)
    if cache.exists() and not refresh_cache:
        with open(cache, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    sim_dirs = _discover_sim_dirs(data_root)
    train_sims, test_sims = _split_sims(sim_dirs, train_fraction, split_seed)
    chosen = train_sims if split == "train" else test_sims
    files = _collect_halo_files(chosen)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for p in files:
            f.write(p + "\n")

    return files


def _coarse_channel(volume: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample then repeat back to original resolution."""

    d, h, w = volume.shape
    if (d % factor) or (h % factor) or (w % factor):
        raise ValueError(f"Volume shape {volume.shape} is not divisible by factor={factor}")

    ds = volume.reshape(d // factor, factor, h // factor, factor, w // factor, factor)
    ds = ds.mean(axis=(1, 3, 5), dtype=np.float64)
    up = np.repeat(np.repeat(np.repeat(ds, factor, axis=0), factor, axis=1), factor, axis=2)
    return up.astype(np.float32)


def build_large_scale_from_condition(condition_log: np.ndarray) -> np.ndarray:
    """Create 3 pseudo large-scale channels from the DMO condition volume."""

    c2 = _coarse_channel(condition_log, 2)
    c4 = _coarse_channel(condition_log, 4)
    c8 = _coarse_channel(condition_log, 8)
    return np.stack([c2, c4, c8], axis=0)


def compute_norm_stats_3d(
    file_list: list[str],
    n_samples: int = 256,
    seed: int = 42,
) -> NormStats3D:
    """Compute normalization stats using streaming moments to avoid OOM."""

    if not file_list:
        raise ValueError("file_list is empty")

    rng = np.random.RandomState(seed)
    n_pick = min(n_samples, len(file_list))
    indices = rng.choice(len(file_list), n_pick, replace=False)

    sum_target = np.zeros(3, dtype=np.float64)
    sqsum_target = np.zeros(3, dtype=np.float64)
    sum_ls = np.zeros(3, dtype=np.float64)
    sqsum_ls = np.zeros(3, dtype=np.float64)
    sum_cond = 0.0
    sqsum_cond = 0.0
    count_vox = 0

    param_min = None
    param_max = None

    for idx in indices:
        data = np.load(file_list[idx])

        target = log_transform(data["target"].astype(np.float32))
        cond = log_transform(data["condition"].astype(np.float32))
        ls = build_large_scale_from_condition(cond)

        sum_target += target.sum(axis=(1, 2, 3), dtype=np.float64)
        sqsum_target += np.square(target, dtype=np.float64).sum(axis=(1, 2, 3), dtype=np.float64)

        sum_cond += float(cond.sum(dtype=np.float64))
        sqsum_cond += float(np.square(cond, dtype=np.float64).sum(dtype=np.float64))

        sum_ls += ls.sum(axis=(1, 2, 3), dtype=np.float64)
        sqsum_ls += np.square(ls, dtype=np.float64).sum(axis=(1, 2, 3), dtype=np.float64)

        count_vox += int(cond.size)

        params = data["params"].astype(np.float32)
        if param_min is None:
            param_min = params.copy()
            param_max = params.copy()
        else:
            param_min = np.minimum(param_min, params)
            param_max = np.maximum(param_max, params)

    if param_min is None or param_max is None:
        raise RuntimeError("Failed to compute parameter min/max from samples")

    target_mean = (sum_target / count_vox).astype(np.float32)
    target_var = np.maximum(sqsum_target / count_vox - np.square(target_mean, dtype=np.float64), 1e-12)
    target_std = np.sqrt(target_var, dtype=np.float64).astype(np.float32)

    cond_mean = float(sum_cond / count_vox)
    cond_var = max(sqsum_cond / count_vox - cond_mean * cond_mean, 1e-12)
    cond_std = float(np.sqrt(cond_var))

    ls_mean = (sum_ls / count_vox).astype(np.float32)
    ls_var = np.maximum(sqsum_ls / count_vox - np.square(ls_mean, dtype=np.float64), 1e-12)
    ls_std = np.sqrt(ls_var, dtype=np.float64).astype(np.float32)

    return NormStats3D(
        target_mean=target_mean,
        target_std=target_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        ls_mean=ls_mean,
        ls_std=ls_std,
        param_min=param_min.astype(np.float32),
        param_max=param_max.astype(np.float32),
    )


class AstroDataset3D(Dataset):
    """3D volumetric dataset with optional random crops and flips."""

    def __init__(
        self,
        file_list: list[str],
        norm_stats: NormStats3D,
        crop_size: int = 64,
        random_crop: bool = True,
        augment_flip: bool = True,
    ):
        self.file_list = file_list
        self.ns = norm_stats
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.augment_flip = augment_flip

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
        return (x - mean) / (std + 1e-8)

    def _normalize_params(self, p: np.ndarray) -> np.ndarray:
        span = self.ns.param_max - self.ns.param_min
        return (p - self.ns.param_min) / (span + 1e-8)

    def _select_crop(self, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        d, h, w = shape
        c = self.crop_size
        if c <= 0 or c >= min(d, h, w):
            return slice(0, d), slice(0, h), slice(0, w)

        if self.random_crop:
            z0 = np.random.randint(0, d - c + 1)
            y0 = np.random.randint(0, h - c + 1)
            x0 = np.random.randint(0, w - c + 1)
        else:
            z0 = (d - c) // 2
            y0 = (h - c) // 2
            x0 = (w - c) // 2
        return slice(z0, z0 + c), slice(y0, y0 + c), slice(x0, x0 + c)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.file_list[idx])

        cond = log_transform(data["condition"].astype(np.float32))
        target = log_transform(data["target"].astype(np.float32))

        sz, sy, sx = self._select_crop(cond.shape)
        cond = cond[sz, sy, sx]
        target = target[:, sz, sy, sx]

        if self.augment_flip and self.random_crop:
            # Apply the same spatial flips to condition and target.
            if np.random.rand() < 0.5:
                cond = np.flip(cond, axis=0)
                target = np.flip(target, axis=1)
            if np.random.rand() < 0.5:
                cond = np.flip(cond, axis=1)
                target = np.flip(target, axis=2)
            if np.random.rand() < 0.5:
                cond = np.flip(cond, axis=2)
                target = np.flip(target, axis=3)

        cond = np.ascontiguousarray(cond)
        target = np.ascontiguousarray(target)

        ls = build_large_scale_from_condition(cond)

        target = self._normalize(target, self.ns.target_mean[:, None, None, None], self.ns.target_std[:, None, None, None])
        cond = self._normalize(cond[None], self.ns.cond_mean, self.ns.cond_std)
        ls = self._normalize(ls, self.ns.ls_mean[:, None, None, None], self.ns.ls_std[:, None, None, None])

        params = self._normalize_params(data["params"].astype(np.float32))

        return {
            "target": torch.from_numpy(target.astype(np.float32)),
            "condition": torch.from_numpy(cond.astype(np.float32)),
            "large_scale": torch.from_numpy(ls.astype(np.float32)),
            "params": torch.from_numpy(params.astype(np.float32)),
        }


def get_loaders_3d(
    data_root: str | Path,
    norm_stats: NormStats3D,
    batch_size: int = 2,
    num_workers: int = 8,
    crop_size: int = 64,
    train_fraction: float = 0.9,
    split_seed: int = 42,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for 3D data."""

    train_files = load_file_list_3d(data_root, "train", train_fraction, split_seed)
    test_files = load_file_list_3d(data_root, "test", train_fraction, split_seed)

    if max_train:
        train_files = train_files[:max_train]
    if max_test:
        test_files = test_files[:max_test]

    train_ds = AstroDataset3D(train_files, norm_stats, crop_size=crop_size, random_crop=True, augment_flip=True)
    test_ds = AstroDataset3D(test_files, norm_stats, crop_size=crop_size, random_crop=False, augment_flip=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
