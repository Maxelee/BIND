"""Persisted Gaussian-process emulator fits.

WHY. bind.emulator.Emulator already has save()/load() (a torch bundle of the PCA
compressors and per-statistic backend state_dicts), and a round trip is bit-exact —
but nothing used them, so every notebook run refit all ~12 heads from scratch
(~30 s/head on CPU, minutes on GPU). Worse, for Paper I the *fitted* emulator is the
release artifact: "theta -> any released statistic in milliseconds" is only a real
claim if the trained object ships.

WHAT MAKES THIS SAFE. A saved fit is only valid for the split it was trained on.
Paper I's fig 9 reports HELD-OUT errors, so reusing a bundle trained on a different
train/test partition would silently turn a held-out test into a training-set test —
the exact failure that makes emulator papers wrong. The sidecar therefore records the
training indices, the statistic list, the backend configuration and the dataset
identity, and `load_matching()` refuses the bundle unless all of them agree.

BUILD:
    sbatch papers/01_pipeline/run_emulator_fit.sbatch      # GPU
USE:
    from emulator_cache import load_matching
    em = load_matching(ds_path, STATS_EMU, train_idx, backend="gpgpu",
                       n_components=12, backend_kwargs={"epochs": 400, "lr": 0.1})
    if em is None:
        em = Emulator(...); em.fit(...)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
EMU_DIR = CEPH / "bind_sb35" / "emulator_fits"
BUNDLE = EMU_DIR / "paper1_gp.pt"
META = EMU_DIR / "paper1_gp.json"


def _emulator_config_fingerprint() -> dict:
    """Hashes the *code-level* C-head config that isn't a build_emulator.py
    call-site argument -- bind.emulator.core's ell-masking/transform-override/
    composed-head defaults (papers/01_pipeline/audits/spectrum_head_experiment.md).
    Without this, changing those module-level defaults (as the 2026-08 C-head
    improvement set did) would leave `_fingerprint`'s other fields identical to a
    pre-change bundle -- same dataset, same split, same stats/backend/n_components
    -- and `load_matching` would happily hand back a STALE bundle fit under the
    old raw/unmasked/direct-cl_kappa convention.  Bump `_CONFIG_VERSION` on any
    future change to the compression/composition logic that isn't already
    captured by an explicit kwarg here.
    """
    from bind.emulator.core import (
        DEFAULT_ELL_MASK_HEADS,
        DEFAULT_TRANSFORM_OVERRIDES,
        ELL_MASK_ABOVE,
    )

    return {
        "config_version": _CONFIG_VERSION,
        "ell_mask_above": ELL_MASK_ABOVE,
        "ell_mask_heads": sorted(DEFAULT_ELL_MASK_HEADS),
        "transform_overrides": {k: DEFAULT_TRANSFORM_OVERRIDES[k]
                                for k in sorted(DEFAULT_TRANSFORM_OVERRIDES)},
    }


# Bump whenever bind.emulator.core/transforms changes how a statistic is
# compressed/masked/composed in a way the fields above don't already capture
# (e.g. a different mask_err_inflate default, a new composed head). Bundles
# fit under a different version are always refit, never silently reused.
_CONFIG_VERSION = 2   # 2026-08: ell-masking + asinh_std crosses + composed cl_kappa


def _fingerprint(ds_path, stats, train_idx, backend, n_components, backend_kwargs) -> dict:
    """Everything that must match for a saved fit to be reusable."""
    return {
        "dataset": Path(ds_path).name,
        "dataset_sha256_16": hashlib.sha256(
            Path(ds_path).read_bytes() if Path(ds_path).stat().st_size < 5_000_000
            else Path(ds_path).name.encode()).hexdigest()[:16],
        "n_train": int(len(train_idx)),
        "train_idx_sha256_16": hashlib.sha256(
            np.asarray(sorted(int(i) for i in train_idx)).tobytes()).hexdigest()[:16],
        "stats": sorted(map(str, stats)),
        "backend": str(backend),
        "n_components": int(n_components),
        "backend_kwargs": {k: backend_kwargs[k] for k in sorted(backend_kwargs or {})},
        "emulator_config": _emulator_config_fingerprint(),
    }


def save(em, ds_path, stats, train_idx, backend, n_components, backend_kwargs) -> Path:
    """Write the bundle ATOMICALLY, and only publish the sidecar once it lands.

    run_emulator_fit.sbatch and run_figures.sbatch can be in flight together (they
    were, the first time this ran). A plain torch.save leaves a window where the
    figure job could torch.load a half-written file and die; os.replace is atomic
    within a filesystem, so a reader sees either the old bundle or the new one.
    The sidecar is written last, so a bundle is never advertised before it exists.
    """
    import os

    EMU_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BUNDLE.with_suffix(BUNDLE.suffix + f".tmp{os.getpid()}")
    em.save(tmp)
    os.replace(tmp, BUNDLE)
    mtmp = META.with_suffix(META.suffix + f".tmp{os.getpid()}")
    mtmp.write_text(json.dumps(
        _fingerprint(ds_path, stats, train_idx, backend, n_components, backend_kwargs),
        indent=1, sort_keys=True))
    os.replace(mtmp, META)
    return BUNDLE


def load_matching(ds_path, stats, train_idx, backend, n_components, backend_kwargs):
    """Return the saved Emulator, or None if it does not match this exact setup.

    Returning None is always safe: the caller refits. Returning a mismatched bundle
    would not be, so every disagreement is a refusal.
    """
    if not (BUNDLE.exists() and META.exists()):
        return None
    want = _fingerprint(ds_path, stats, train_idx, backend, n_components, backend_kwargs)
    try:
        have = json.loads(META.read_text())
    except json.JSONDecodeError:
        return None
    if have != want:
        diff = [k for k in want if have.get(k) != want[k]]
        print(f"[emulator_cache] saved fit does not match this setup "
              f"(differs in: {', '.join(diff)}) -> refitting")
        return None
    from bind.emulator import Emulator

    try:
        em = Emulator.load(BUNDLE)
    except Exception as e:                    # unreadable/partial bundle -> refit
        print(f"[emulator_cache] could not load {BUNDLE.name} ({type(e).__name__}: {e}) "
              f"-> refitting")
        return None
    missing = set(want["stats"]) - set(em.statistics)
    if missing:
        print(f"[emulator_cache] bundle is missing heads {sorted(missing)} -> refitting")
        return None
    return em
