"""Fast CV sim_0 × SB35 parameter injection.

Key optimization: instead of 102 sequential inference calls (one per SB35
param vector), tile the N_halos cutouts × N_sb35 param vectors into a single
(N_halos * N_sb35, ...) batch. This reduces the number of forward passes from
102 × ceil(N_halos/batch_size) to ceil(N_halos * N_sb35 / batch_size), and
eliminates all Python loop overhead between calls.

Output: cv_sim0_sb35_injection_fast.npz
  gen_pk       (n_sb35, n_ch, n_k)  – mean gas P(k) over halos per injection
  fiducial_pk  (n_ch, n_k)          – injection with CV fiducial params
  truth_pk     (n_ch, n_k)          – from truth_maps
  sb35_params  (n_sb35, 35)
  sb35_sim_ids (n_sb35,)  str labels
  k_bins       (n_k,)
  channels     (n_ch,)
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data import NormStats, log_transform
from metrics import power_spectrum_2d
from test_suite.pipeline import _denormalize_to_physical, normalize_cutout

TESTSUITE_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite")
RUN_DIR        = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
MANIFEST       = TESTSUITE_ROOT / "manifests" / "sb35_test_manifest.json"
OUTPUT         = TESTSUITE_ROOT / "cv_sim0_sb35_injection_fast.npz"
SNAP           = "snap_090"
MASS_TAG       = "mass_threshold_1p000e13"
PATCH_BOX      = 6.25   # Mpc/h


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv_sim_id",  type=int,  default=0)
    p.add_argument("--run_dir",    type=Path, default=RUN_DIR)
    p.add_argument("--manifest",   type=Path, default=MANIFEST)
    p.add_argument("--output",     type=Path, default=OUTPUT)
    p.add_argument("--n_steps",    type=int,  default=50)
    p.add_argument("--batch_size", type=int,  default=512,
                   help="Total samples per forward pass (halos × param_vectors flattened)")
    p.add_argument("--device",     type=str,  default="auto")
    p.add_argument("--max_sb35",   type=int,  default=None)
    p.add_argument("--channels",   type=str,  default="0,1,2")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, using CPU"); return torch.device("cpu")
    return torch.device(name)


def generate_vectorized(
    halo_cutouts: list[dict],
    all_params: np.ndarray,      # (n_vectors, 35)
    norm_stats: NormStats,
    fm,
    device: torch.device,
    n_steps: int,
    batch_size: int,
    use_amp: bool = True,
) -> np.ndarray:
    """Run inference for all (halo, param_vector) pairs in one big batch.

    Tiles the N_halos cutouts × N_vectors param vectors → N_halos*N_vectors
    samples, runs them through the model in mini-batches of `batch_size`,
    then returns (N_vectors, N_halos, 3, H, W) physical-space output.
    """
    n_halos   = len(halo_cutouts)
    n_vectors = len(all_params)
    N         = n_halos * n_vectors

    print(f"  Vectorized batch: {n_halos} halos × {n_vectors} param vectors = {N} samples", flush=True)

    # ── Pre-normalize all cutouts (shared across param vectors) ─────────────
    # We'll normalize the cutouts once with a dummy param vector, then
    # re-apply just the param normalization per vector in the loop.
    # Actually: normalize_cutout is cheap, just do it for every (halo, vector) pair.
    # The condition/large_scale normalization is independent of params.
    cond_norm  = np.zeros((n_halos, 1, 128, 128), dtype=np.float32)
    ls_norm    = np.zeros((n_halos, 3, 128, 128), dtype=np.float32)
    dummy_p    = all_params[0]
    for hi, hc in enumerate(halo_cutouts):
        c, l, _ = normalize_cutout(hc, norm_stats, dummy_p)
        cond_norm[hi] = c
        ls_norm[hi]   = l

    # ── Normalize all param vectors ──────────────────────────────────────────
    params_norm = np.zeros((n_vectors, 35), dtype=np.float32)
    for vi, p_raw in enumerate(all_params):
        _, _, params_norm[vi] = normalize_cutout(halo_cutouts[0], norm_stats, p_raw)

    # ── Build flat index arrays ──────────────────────────────────────────────
    # Sample i maps to halo hi = i % n_halos, vector vi = i // n_halos
    halo_idx   = np.tile(np.arange(n_halos), n_vectors)   # (N,)
    vector_idx = np.repeat(np.arange(n_vectors), n_halos)  # (N,)

    outputs = []
    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="  Generating"):
            idx   = slice(start, start + batch_size)
            hi_b  = halo_idx[idx]
            vi_b  = vector_idx[idx]

            cond_t   = torch.from_numpy(cond_norm[hi_b]).to(device)
            ls_t     = torch.from_numpy(ls_norm[hi_b]).to(device)
            params_t = torch.from_numpy(params_norm[vi_b]).to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda" else nullcontext()
            )
            with amp_ctx:
                gen = fm.sample(cond_t, ls_t, params_t, n_steps=n_steps)

            outputs.append(gen.float().cpu().numpy())

    flat = np.concatenate(outputs, axis=0)                   # (N, C_model, 128, 128)
    flat_phys = _denormalize_to_physical(flat, norm_stats)   # (N, 3, 128, 128)

    # ── Reshape to (n_vectors, n_halos, 3, 128, 128) ─────────────────────────
    result = np.zeros((n_vectors, n_halos, 3, 128, 128), dtype=np.float32)
    result[vector_idx, halo_idx] = flat_phys
    return result


def compute_pk(patches, channels, patch_box):
    """patches: (N_halos, 3, 128, 128) → dict ch: (N_PK_BINS,) mean over halos."""
    result, k_bins = {}, None
    for ch in channels:
        pks = []
        for h in range(len(patches)):
            k, pk = power_spectrum_2d(patches[h, ch], box_size=patch_box)
            pks.append(pk)
            if k_bins is None:
                k_bins = k
        result[ch] = np.nanmean(pks, axis=0)
    return result, k_bins


def main():
    args = parse_args()
    device   = resolve_device(args.device)
    channels = [int(x) for x in args.channels.split(",")]

    # ── Load model ────────────────────────────────────────────────────────────
    from train import FlowMatchingLit
    ckpt = args.run_dir / "checkpoints" / "last.ckpt"
    print(f"Loading model from {ckpt}", flush=True)
    norm_stats = NormStats.load(args.run_dir / "norm_stats.npz")
    lit        = FlowMatchingLit.load_from_checkpoint(ckpt, map_location=device)
    lit.eval(); lit.to(device)
    fm = lit.fm

    # ── Load SB35 param vectors ───────────────────────────────────────────────
    with open(args.manifest) as f:
        entries = json.load(f)["simulations"]
    if args.max_sb35:
        entries = entries[:args.max_sb35]
    sb35_sim_ids = np.array([e["sim_id"] for e in entries])
    sb35_params  = np.array([e["params"] for e in entries], dtype=np.float32)  # (n_sb35, 35)
    print(f"SB35 param vectors: {len(sb35_params)}", flush=True)

    # ── Load CV sim cutouts ───────────────────────────────────────────────────
    cv_id    = args.cv_sim_id
    mass_dir = TESTSUITE_ROOT / "CV" / f"sim_{cv_id}" / SNAP / MASS_TAG
    cuts     = np.load(mass_dir / "halo_cutouts.npz")
    cat      = np.load(mass_dir / "halo_catalog.npz")
    maps     = np.load(TESTSUITE_ROOT / "CV" / f"sim_{cv_id}" / SNAP / "full_maps.npz")

    halo_cutouts = [
        {"condition": cuts["condition"][i], "large_scale": cuts["large_scale"][i]}
        for i in range(cuts["condition"].shape[0])
    ]
    fiducial_params = cat["params"][0]   # (35,) CV fiducial
    n_halos = len(halo_cutouts)
    print(f"CV sim_{cv_id}: {n_halos} halos", flush=True)
    print(f"Fiducial params: Ωm={fiducial_params[0]:.3f}  σ8={fiducial_params[1]:.3f}", flush=True)

    # ── Truth P(k) ────────────────────────────────────────────────────────────
    truth_maps  = maps["truth_maps"]   # (3, 1024, 1024)
    centers_mpc = cat["centers"]       # (N, 2) Mpc/h
    npix        = truth_maps.shape[-1]
    pix_per_mpc = npix / 50.0
    centers_pix = (centers_mpc * pix_per_mpc).astype(int) % npix

    truth_pk, k_bins = {}, None
    for ch in channels:
        pks = []
        for cx, cy in centers_pix:
            half = 64
            ix = (cx - half + np.arange(128)) % npix
            iy = (cy - half + np.arange(128)) % npix
            patch = truth_maps[ch][np.ix_(ix, iy)]
            k, pk = power_spectrum_2d(patch, box_size=PATCH_BOX)
            pks.append(pk)
            if k_bins is None:
                k_bins = k
        truth_pk[ch] = np.nanmean(pks, axis=0)

    # ── Fiducial injection (just one pass, fast) ──────────────────────────────
    print("\nRunning fiducial injection...", flush=True)
    fid_gen = generate_vectorized(
        halo_cutouts, fiducial_params[None], norm_stats, fm, device,
        args.n_steps, args.batch_size,
    )   # (1, n_halos, 3, 128, 128)
    fid_pk, _ = compute_pk(fid_gen[0], channels, PATCH_BOX)

    # ── SB35 vectorized injection ─────────────────────────────────────────────
    print(f"\nRunning SB35 injection ({len(sb35_params)} vectors × {n_halos} halos)...", flush=True)
    sb35_gen = generate_vectorized(
        halo_cutouts, sb35_params, norm_stats, fm, device,
        args.n_steps, args.batch_size,
    )   # (n_sb35, n_halos, 3, 128, 128)

    # P(k) for each SB35 injection
    n_sb35 = len(sb35_params)
    n_k    = len(k_bins)
    n_ch   = len(channels)
    gen_pk_arr = np.full((n_sb35, n_ch, n_k), np.nan, dtype=np.float32)
    for si in range(n_sb35):
        pk_s, _ = compute_pk(sb35_gen[si], channels, PATCH_BOX)
        for chi, ch in enumerate(channels):
            gen_pk_arr[si, chi] = pk_s[ch]

    fiducial_pk_arr = np.array([fid_pk[ch] for ch in channels])  # (n_ch, n_k)
    truth_pk_arr    = np.array([truth_pk[ch] for ch in channels]) # (n_ch, n_k)

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        gen_pk       = gen_pk_arr,         # (n_sb35, n_ch, n_k)
        fiducial_pk  = fiducial_pk_arr,    # (n_ch, n_k)
        truth_pk     = truth_pk_arr,        # (n_ch, n_k)
        sb35_params  = sb35_params,         # (n_sb35, 35)
        sb35_sim_ids = sb35_sim_ids,
        k_bins       = k_bins,
        channels     = np.array(channels),
        cv_sim_id    = np.array([cv_id]),
    )
    print(f"\nSaved to {args.output}", flush=True)
    print(f"  gen_pk shape: {gen_pk_arr.shape}", flush=True)


if __name__ == "__main__":
    main()
