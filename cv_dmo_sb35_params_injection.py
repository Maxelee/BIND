"""Cross-injection experiment: CV DMO halos × SB35 parameter vectors.

For each CV simulation (pre-existing halo_cutouts.npz on disk), run BIND
inference with every SB35 parameter vector instead of the CV fiducial.

Output: cv_dmo_sb35_params_injection.npz
  gen_pk[cv_sim_id][sb35_sim_id] = (N_PK_BINS,) mean gas P(k) over halos
  true_pk[cv_sim_id]             = (N_PK_BINS,) mean truth gas P(k)
  k_bins                         = (N_PK_BINS,)
  fiducial_pk[cv_sim_id]         = (N_PK_BINS,) mean gas P(k) with fiducial CV params
  sb35_params                    = dict: sim_id -> (35,) param array
  cv_sim_ids                     = list of CV sim ids used
  sb35_sim_ids                   = list of SB35 sim ids (param labels)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data import NormStats
from metrics import power_spectrum_2d
from test_suite.pipeline import generate_halo_patches, normalize_cutout
from train import FlowMatchingLit


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TESTSUITE_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite")
DEFAULT_RUN_DIR        = Path("/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
DEFAULT_MANIFEST       = DEFAULT_TESTSUITE_ROOT / "manifests" / "sb35_test_manifest.json"
DEFAULT_OUTPUT         = Path("/mnt/home/mlee1/ceph/fm_testsuite/cv_dmo_sb35_params_injection.npz")
MASS_TAG               = "mass_threshold_1p000e13"
SNAP                   = "snap_090"
PATCH_BOX              = 6.25   # Mpc/h  (50 * 128/1024)
GAS_CH                 = 1      # index in (DM_hydro, Gas, Stars)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--testsuite_root", type=Path, default=DEFAULT_TESTSUITE_ROOT)
    p.add_argument("--run_dir",        type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--manifest",       type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n_steps",        type=int,  default=50)
    p.add_argument("--batch_size",     type=int,  default=32)
    p.add_argument("--device",         type=str,  default="auto")
    p.add_argument("--cv_sim_ids",     type=str,  default=None,
                   help="Comma-separated subset of CV sim integers to use (default: all available)")
    p.add_argument("--max_sb35",       type=int,  default=None,
                   help="Max number of SB35 param vectors to test (default: all in manifest)")
    p.add_argument("--channels",       type=str,  default="0,1,2",
                   help="Comma-separated channel indices to compute P(k) for (0=DM,1=Gas,2=Stars)")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def load_cv_cutouts(testsuite_root: Path, sim_int: int):
    """Load pre-extracted halo cutouts and truth data for a CV sim."""
    sim_dir = testsuite_root / "CV" / f"sim_{sim_int}" / SNAP
    mass_dir = sim_dir / MASS_TAG

    cuts_path = mass_dir / "halo_cutouts.npz"
    cat_path  = mass_dir / "halo_catalog.npz"
    maps_path = sim_dir / "full_maps.npz"

    if not cuts_path.exists():
        return None

    c = np.load(cuts_path)
    # Reconstruct list-of-dict format expected by normalize_cutout / generate_halo_patches
    halo_cutouts = [
        {"condition": c["condition"][i], "large_scale": c["large_scale"][i]}
        for i in range(c["condition"].shape[0])
    ]

    truth_maps = None
    if maps_path.exists():
        m = np.load(maps_path)
        if "truth_maps" in m.files:
            truth_maps = m["truth_maps"]   # (3, 1024, 1024)

    fiducial_params = None
    if cat_path.exists():
        cat = np.load(cat_path)
        if "params" in cat.files and len(cat["params"]) > 0:
            fiducial_params = cat["params"][0]   # (35,)

    return halo_cutouts, truth_maps, fiducial_params


def mean_pk_gas(gen_patches: np.ndarray, channels: list[int], patch_box: float):
    """Compute per-channel mean P(k) over halos.

    gen_patches: (N, 3, H, W) physical-space output from generate_halo_patches
    Returns dict ch -> (N_PK_BINS,)
    """
    result = {}
    k_bins = None
    for ch in channels:
        pks = []
        for h in range(len(gen_patches)):
            k, pk = power_spectrum_2d(gen_patches[h, ch], box_size=patch_box)
            pks.append(pk)
            if k_bins is None:
                k_bins = k
        result[ch] = np.nanmean(pks, axis=0)
    return result, k_bins


def extract_truth_pk(truth_maps: np.ndarray, centers_pix, patch_pix: int,
                     channels: list[int], patch_box: float):
    """Extract halo patches from truth full-box maps and compute P(k)."""
    npix = truth_maps.shape[-1]
    result = {ch: [] for ch in channels}
    k_bins = None
    for cx, cy in centers_pix:
        half = patch_pix // 2
        ix = (cx - half + np.arange(patch_pix)) % npix
        iy = (cy - half + np.arange(patch_pix)) % npix
        for ch in channels:
            patch = truth_maps[ch][np.ix_(ix, iy)]
            k, pk = power_spectrum_2d(patch, box_size=patch_box)
            result[ch].append(pk)
            if k_bins is None:
                k_bins = k
    return {ch: np.nanmean(result[ch], axis=0) for ch in channels}, k_bins


def main():
    args = parse_args()
    device = resolve_device(args.device)
    channels = [int(x) for x in args.channels.split(",")]

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt  = args.run_dir / "checkpoints" / "last.ckpt"
    ns_path = args.run_dir / "norm_stats.npz"
    print(f"Loading model from {ckpt}", flush=True)
    norm_stats = NormStats.load(ns_path)
    model_lit  = FlowMatchingLit.load_from_checkpoint(ckpt, map_location=device)
    model_lit.eval()
    model_lit.to(device)
    fm = model_lit.fm

    # ── Load SB35 param vectors from manifest ────────────────────────────────
    with open(args.manifest) as f:
        manifest = json.load(f)
    sb35_entries = manifest["simulations"]
    if args.max_sb35 is not None:
        sb35_entries = sb35_entries[: args.max_sb35]
    sb35_sim_ids = [e["sim_id"] for e in sb35_entries]
    sb35_params  = {e["sim_id"]: np.array(e["params"], dtype=np.float32) for e in sb35_entries}
    print(f"SB35 parameter vectors: {len(sb35_sim_ids)}", flush=True)

    # ── Discover available CV sims ───────────────────────────────────────────
    if args.cv_sim_ids is not None:
        cv_ints = [int(x) for x in args.cv_sim_ids.split(",")]
    else:
        cv_ints = sorted(
            int(d.name.split("_")[1])
            for d in (args.testsuite_root / "CV").iterdir()
            if d.is_dir() and d.name.startswith("sim_")
            and (d / SNAP / MASS_TAG / "halo_cutouts.npz").exists()
        )
    print(f"CV sims with cutouts: {cv_ints}", flush=True)

    # ── Results containers ───────────────────────────────────────────────────
    # gen_pk_all[cv_id][sb35_id][ch] = (N_PK_BINS,)
    gen_pk_all    = {}
    fiducial_pk   = {}
    truth_pk_all  = {}
    k_bins_global = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for cv_int in cv_ints:
        cv_id = f"sim_{cv_int}"
        print(f"\n{'='*60}", flush=True)
        print(f"CV sim: {cv_id}", flush=True)

        loaded = load_cv_cutouts(args.testsuite_root, cv_int)
        if loaded is None:
            print(f"  [skip] halo_cutouts.npz not found", flush=True)
            continue
        halo_cutouts, truth_maps, fiducial_params = loaded
        n_halos = len(halo_cutouts)
        print(f"  halos: {n_halos}", flush=True)

        # Truth P(k) — extract from full-box truth maps
        if truth_maps is not None:
            cat = np.load(args.testsuite_root / "CV" / cv_id / SNAP / MASS_TAG / "halo_catalog.npz")
            centers_mpc = cat["centers"]   # (N, 2) in Mpc/h
            npix = truth_maps.shape[-1]
            box_size = 50.0
            pix_per_mpc = npix / box_size
            centers_pix = (centers_mpc * pix_per_mpc).astype(int) % npix
            tpk, k_bins = extract_truth_pk(truth_maps, centers_pix, 128, channels, PATCH_BOX)
            truth_pk_all[cv_id] = tpk
            if k_bins_global is None:
                k_bins_global = k_bins
            print(f"  truth P(k) computed", flush=True)

        gen_pk_all[cv_id] = {}

        # ── Fiducial params run ────────────────────────────────────────────
        if fiducial_params is not None:
            print(f"  Fiducial params (Ωm={fiducial_params[0]:.3f}, σ8={fiducial_params[1]:.3f})", flush=True)
            gen = generate_halo_patches(
                halo_cutouts, norm_stats, fiducial_params, fm, device,
                n_steps=args.n_steps, batch_size=args.batch_size, use_amp=True,
            )
            fpk, k_bins = mean_pk_gas(gen, channels, PATCH_BOX)
            fiducial_pk[cv_id] = fpk
            if k_bins_global is None:
                k_bins_global = k_bins

        # ── SB35 param injection ───────────────────────────────────────────
        for sb35_id in tqdm(sb35_sim_ids, desc=f"  {cv_id} × SB35 params"):
            params = sb35_params[sb35_id]
            gen = generate_halo_patches(
                halo_cutouts, norm_stats, params, fm, device,
                n_steps=args.n_steps, batch_size=args.batch_size, use_amp=True,
            )
            gpk, k_bins = mean_pk_gas(gen, channels, PATCH_BOX)
            gen_pk_all[cv_id][sb35_id] = gpk
            if k_bins_global is None:
                k_bins_global = k_bins

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "k_bins":      k_bins_global,
        "cv_sim_ids":  np.array(cv_ints),
        "sb35_sim_ids": np.array(sb35_sim_ids),
        "channels":    np.array(channels),
    }
    # Flatten nested dicts into arrays
    # gen_pk: (n_cv, n_sb35, n_ch, n_k)
    n_cv   = len(cv_ints)
    n_sb35 = len(sb35_sim_ids)
    n_ch   = len(channels)
    n_k    = len(k_bins_global)

    gen_pk_arr      = np.full((n_cv, n_sb35, n_ch, n_k), np.nan, dtype=np.float32)
    fiducial_pk_arr = np.full((n_cv, n_ch, n_k),         np.nan, dtype=np.float32)
    truth_pk_arr    = np.full((n_cv, n_ch, n_k),         np.nan, dtype=np.float32)
    sb35_params_arr = np.stack([sb35_params[s] for s in sb35_sim_ids])  # (n_sb35, 35)

    for ci, cv_int in enumerate(cv_ints):
        cv_id = f"sim_{cv_int}"
        if cv_id in fiducial_pk:
            for chi, ch in enumerate(channels):
                fiducial_pk_arr[ci, chi] = fiducial_pk[cv_id][ch]
        if cv_id in truth_pk_all:
            for chi, ch in enumerate(channels):
                truth_pk_arr[ci, chi] = truth_pk_all[cv_id][ch]
        if cv_id in gen_pk_all:
            for si, sb35_id in enumerate(sb35_sim_ids):
                if sb35_id in gen_pk_all[cv_id]:
                    for chi, ch in enumerate(channels):
                        gen_pk_arr[ci, si, chi] = gen_pk_all[cv_id][sb35_id][ch]

    save_dict["gen_pk"]       = gen_pk_arr       # (n_cv, n_sb35, n_ch, n_k)
    save_dict["fiducial_pk"]  = fiducial_pk_arr  # (n_cv, n_ch, n_k)
    save_dict["truth_pk"]     = truth_pk_arr      # (n_cv, n_ch, n_k)
    save_dict["sb35_params"]  = sb35_params_arr  # (n_sb35, 35)

    np.savez(args.output, **save_dict)
    print(f"\nSaved to {args.output}", flush=True)
    print(f"  gen_pk shape: {gen_pk_arr.shape}", flush=True)


if __name__ == "__main__":
    main()
