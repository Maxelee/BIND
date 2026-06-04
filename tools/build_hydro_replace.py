#!/usr/bin/env python
"""Build ``hydro_replace.npy`` for every model in the BIND test suite.

For each ``.../mass_threshold_*/<model>/composite.npz`` this reproduces the
exact BIND compositing blend but pastes the *true* hydro halo patches (cut from
``snap_*/full_maps.npz`` -> ``truth_maps``) instead of the model-generated ones.

Differences from ``build_bind_composite``:
  * NO per-halo ``patch_mass_match`` -- true patches are pasted unscaled.
  * The blend reuses the ``hydro_weights`` stored in ``composite.npz`` (the
    patch-independent accumulated taper) for both the canvas normalization and
    the ``alpha`` map, so ``hydro_replace`` and ``composite`` share identical
    geometry and differ only in patch content.

Result (shape ``(3, npix, npix)`` float32) is saved next to ``composite.npz``.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BOX_SIZE = 50.0      # Mpc/h
PATCH_PIX = 128
TAPER_FRAC = 0.15


def square_taper_weight(patch_size: int, taper_frac: float = TAPER_FRAC) -> np.ndarray:
    """2D separable Hann taper (identical to bind.inference.pipeline)."""
    t = max(1, int(patch_size * taper_frac))
    hann = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t)).astype(np.float32)
    w1d = np.ones(patch_size, dtype=np.float32)
    w1d[:t] = hann
    w1d[-t:] = hann[::-1]
    return np.outer(w1d, w1d)


def extract_periodic_cutout(field: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    n = field.shape[0]
    half = size // 2
    ix = (cx - half + np.arange(size)) % n
    iy = (cy - half + np.arange(size)) % n
    return field[np.ix_(ix, iy)]


def build_hydro_replace(mt_dir: str, model_dir: str) -> np.ndarray | None:
    """Build the true-halo composite for one model directory."""
    snap_dir = os.path.dirname(mt_dir)
    full_path = os.path.join(snap_dir, "full_maps.npz")
    cat_path = os.path.join(mt_dir, "halo_catalog.npz")
    comp_path = os.path.join(model_dir, "composite.npz")
    if not (os.path.exists(full_path) and os.path.exists(cat_path) and os.path.exists(comp_path)):
        return None

    full = np.load(full_path)
    dmo = full["dmo_fullbox"]
    truth = full["truth_maps"]               # (3, npix, npix)
    centers = np.load(cat_path)["centers"]   # (n_halos, 2) in Mpc/h
    comp = np.load(comp_path)
    hydro_weights = comp["hydro_weights"].astype(np.float32)

    npix = dmo.shape[0]
    ppm = npix / BOX_SIZE
    weight = square_taper_weight(PATCH_PIX, TAPER_FRAC)
    w_half = weight.shape[0] // 2

    # Accumulate true patches into a canvas numerator (no mass-match).
    numerator = np.zeros((3, npix, npix), dtype=np.float32)
    for c in centers:
        cx = int(c[0] * ppm) % npix
        cy = int(c[1] * ppm) % npix
        ix = (cx - w_half + np.arange(weight.shape[0])) % npix
        iy = (cy - w_half + np.arange(weight.shape[0])) % npix
        for ch in range(3):
            patch = extract_periodic_cutout(truth[ch], cx, cy, PATCH_PIX)
            numerator[ch][np.ix_(ix, iy)] += patch * weight

    # Normalize by the stored weights; blend with the stored alpha.
    safe_w = np.where(hydro_weights > 0, hydro_weights, 1.0)
    canvas = numerator / safe_w[None]
    alpha = np.clip(hydro_weights, 0.0, 1.0)

    hydro_replace = np.zeros((3, npix, npix), dtype=np.float32)
    hydro_replace[0] = (1 - alpha) * dmo + alpha * canvas[0]
    hydro_replace[1] = alpha * canvas[1]
    hydro_replace[2] = alpha * canvas[2]

    scale_global = float(dmo.sum() / (hydro_replace.sum() + 1e-30))
    hydro_replace *= scale_global
    return hydro_replace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/mnt/home/mlee1/ceph/fm_testsuite",
        help="Test suite root containing <suite>/sim_*/snap_*/mass_threshold_*/<model>/.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Restrict to a single model name (e.g. fm_two_head). Default: all models found.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if hydro_replace.npy exists.")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "*", "*", "snap_*", "mass_threshold_*", "*", "composite.npz")
    comps = sorted(glob.glob(pattern))
    if not comps:
        print(f"No composite.npz found under {args.root}")
        return

    n_done = n_skip = n_fail = 0
    for comp_path in comps:
        model_dir = os.path.dirname(comp_path)
        model_name = os.path.basename(model_dir)
        if args.model is not None and model_name != args.model:
            continue
        mt_dir = os.path.dirname(model_dir)
        out_path = os.path.join(model_dir, "hydro_replace.npy")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        try:
            hr = build_hydro_replace(mt_dir, model_dir)
            if hr is None:
                n_fail += 1
                print(f"SKIP (missing inputs): {model_dir}")
                continue
            np.save(out_path, hr)
            n_done += 1
            print(f"OK  {out_path}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL {model_dir}: {exc}")

    print(f"\nDone. wrote={n_done} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
