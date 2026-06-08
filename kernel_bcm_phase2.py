#!/usr/bin/env python
"""Phase-2 BCM outpainting: full composite = patches inside r200 + kernel outside.

Phase 1 (kernel_bcm_build.py) showed a smooth mean DoG kernel reproduces the
large-scale total-matter suppression S(k) to ~1% (k<5) but cannot make the
high-k (>30 h/Mpc) stellar-condensation excess -- that lives in the actual
painted patches.  Phase 2 closes that gap: paste the real per-halo patches
inside r200 (which carry the high-k cores) and use the kernel only to outpaint
the inter-halo region, blended by the paste weight alpha:

    delta_b_model = alpha * delta_b_patch  +  (1 - alpha) * delta_b_kernel
    total_model   = DMO + delta_b_model

Two patch sources (``--mode``):
  * ``truth_ceiling`` (CPU, no GPU/model) -- delta_b_patch from the TRUE
    total-matter cutouts.  Isolates whether the composite ARCHITECTURE
    (patch-inside + kernel-outside) recovers S(k), independent of BIND's patch
    fidelity.  This is the achievable ceiling and de-risks the GPU run.
  * ``flow`` (GPU) -- delta_b_patch from BIND flow patches (the deployable
    version).  Mirrors bind.inference.runner: loads the checkpoint, runs
    generate_halo_patches on the cached cutouts.  Falls short of the ceiling
    by BIND's known ~0.8 high-k patch fidelity.

Kernel params are refit from training sims at startup (single source of truth
with Phase 1).  Runs from cached full_maps.npz + halo_cutouts.npz + halo_catalog.npz.

CPU ceiling:  python kernel_bcm_phase2.py --mode truth_ceiling --n_test 8
GPU flow:     sbatch run_kernel_phase2.sh     (writes per-sim S(k) npz; user submits)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, MPC_PER_PIX, NPIX, _discover, _load_sim
from kernel_bcm_build import (
    MASS_EDGES, collect_profiles, fit_kernel, reconstruct, sk,
)
from bind.inference.pipeline import extract_periodic_cutout, square_taper_weight

PATCH_PIX = 128  # 6.25 Mpc/h window at native 1024 res (matches training cutouts)


def paste_scalar(centers, patches, taper):
    """Paste 1-channel patches into the full box with overlap-weighted blending.
    Returns (canvas, alpha) where alpha = clip(accumulated weight, 0, 1)."""
    canvas = np.zeros((NPIX, NPIX), dtype=np.float64)
    w_acc = np.zeros((NPIX, NPIX), dtype=np.float64)
    pix_per_mpc = NPIX / BOX
    h = taper.shape[0] // 2
    for c, p in zip(centers, patches):
        cx = int(c[0] * pix_per_mpc) % NPIX
        cy = int(c[1] * pix_per_mpc) % NPIX
        ix = (cx - h + np.arange(taper.shape[0])) % NPIX
        iy = (cy - h + np.arange(taper.shape[1])) % NPIX
        canvas[np.ix_(ix, iy)] += p * taper
        w_acc[np.ix_(ix, iy)] += taper
    safe = np.where(w_acc > 0, w_acc, 1.0)
    return canvas / safe, np.clip(w_acc, 0.0, 1.0)


def truth_patches(truth_total, dmo, centers):
    """delta_b patches from the TRUE total-matter field (the Phase-2 ceiling)."""
    pix_per_mpc = NPIX / BOX
    out = []
    for c in centers:
        cx = int(c[0] * pix_per_mpc) % NPIX
        cy = int(c[1] * pix_per_mpc) % NPIX
        t = extract_periodic_cutout(truth_total, cx, cy, PATCH_PIX)
        d = extract_periodic_cutout(dmo, cx, cy, PATCH_PIX)
        out.append((t - d).astype(np.float64))
    return out


def flow_patches(snap_dir, dmo, centers, params, bundle):
    """delta_b patches from BIND flow output (deployable). Needs GPU + model."""
    from bind.inference.artifacts import load_halo_cutouts
    from bind.inference.pipeline import generate_halo_patches
    norm_stats, fm, device, param_indices, no_large_scale = bundle
    cutouts = load_halo_cutouts(snap_dir / "mass_threshold_1p000e13" / "halo_cutouts.npz")
    gen = generate_halo_patches(
        cutouts, norm_stats, params, fm, device,
        n_steps=50, batch_size=64, use_amp=True,
        param_indices=param_indices, no_large_scale=no_large_scale)
    gen_total = gen[:, :3].sum(1)  # DM_hydro + Gas + Stars (mass channels only)
    pix_per_mpc = NPIX / BOX
    out = []
    for c, gt in zip(centers, gen_total):
        cx = int(c[0] * pix_per_mpc) % NPIX
        cy = int(c[1] * pix_per_mpc) % NPIX
        d = extract_periodic_cutout(dmo, cx, cy, PATCH_PIX)
        out.append((gt.astype(np.float64) - d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--mode", choices=["truth_ceiling", "flow"], default="truth_ceiling")
    ap.add_argument("--n_cv", type=int, default=27)
    ap.add_argument("--n_train_sb35", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--rmax_mpc", type=float, default=8.0)
    ap.add_argument("--taper_frac", type=float, default=0.15)
    ap.add_argument("--run_dir", default="/mnt/home/mlee1/ceph/fm_runs/fm_thermo")
    ap.add_argument("--checkpoint",
                    default="/mnt/home/mlee1/ceph/fm_runs/fm_thermo/checkpoints/kept/keep_epoch064_ema.ckpt")
    ap.add_argument("--out", default="ceph/fm_diag/kernel_phase2.png")
    ap.add_argument("--npz_out", default="ceph/fm_diag/kernel_phase2_sk.npz")
    args = ap.parse_args()

    root = Path(args.suite_root)
    sb35 = _discover(root, "Test", None)
    train = _discover(root, "CV", args.n_cv) + sb35[:args.n_train_sb35]
    test = sb35[args.n_train_sb35:args.n_train_sb35 + args.n_test]
    print(f"[phase2:{args.mode}] train {len(train)} | test {len(test)}")

    # --- refit kernel (single source of truth with Phase 1) ---
    r_edges = np.geomspace(2 * MPC_PER_PIX, 10.0, 24)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    cut = int(round(2 * 12.0 / MPC_PER_PIX)); cut += cut % 2
    profs, logms, s_ins, area_w = collect_profiles(train, r_edges, cut, aperture_r200=2.0)
    kernel_params, *_ = fit_kernel(profs, logms, r_cent, area_w)
    sin_ref = {}
    for lo, hi in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
        sel = (logms >= lo) & (logms < hi)
        if sel.sum() >= 30:
            sin_ref[0.5 * (lo + hi)] = float(s_ins[sel].mean())
    print(f"[phase2] kernel fit from {len(logms)} halos")

    # --- model bundle (flow mode only) ---
    bundle = None
    if args.mode == "flow":
        import torch
        from bind.data import NormStats
        from bind.train import FlowMatchingLit
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norm_stats = NormStats.load(Path(args.run_dir) / "norm_stats.npz")
        model = FlowMatchingLit.load_from_checkpoint(args.checkpoint, map_location=device)
        model.eval().to(device)
        _COSMO = [0, 1, 7, 8]
        n_params = getattr(model.hparams, "n_params", 35)
        param_indices = (np.array([i for i in range(35) if i not in _COSMO])
                         if n_params < 35 else None)
        no_large_scale = bool(getattr(model.hparams, "no_large_scale", False))
        bundle = (norm_stats, model.fm, device, param_indices, no_large_scale)
        print(f"[phase2] loaded model on {device}")

    taper = square_taper_weight(PATCH_PIX, taper_frac=args.taper_frac)

    rows = []  # (sim, S arrays)
    k_ref = None
    for snap_dir in test:
        # kernel field + maps (reuses Phase-1 reconstruct; amplitude=patch)
        delta_kernel, dmo, truth_total, _ = reconstruct(
            snap_dir, kernel_params, args.rmax_mpc, "patch", sin_ref)
        cat = np.load(snap_dir / "mass_threshold_1p000e13" / "halo_catalog.npz")
        centers = cat["centers"][:, :2].astype(np.float64)
        params = cat["params"][0].astype(np.float32) if "params" in cat else None

        if args.mode == "truth_ceiling":
            dbp = truth_patches(truth_total, dmo, centers)
        else:
            dbp = flow_patches(snap_dir, dmo, centers, params, bundle)

        patch_canvas, alpha = paste_scalar(centers, dbp, taper)

        total_patch = dmo + alpha * patch_canvas                       # patch only (~BIND)
        total_kernel = dmo + delta_kernel                              # Phase 1
        total_phase2 = dmo + alpha * patch_canvas + (1 - alpha) * delta_kernel

        k, s_truth = sk(truth_total, dmo)
        _, s_patch = sk(total_patch, dmo)
        _, s_kernel = sk(total_kernel, dmo)
        _, s_phase2 = sk(total_phase2, dmo)
        k_ref = k
        rows.append((snap_dir.parent.name, s_truth, s_patch, s_kernel, s_phase2,
                     float(alpha.mean())))
        print(f"  {snap_dir.parent.name}: coverage(alpha)={alpha.mean()*100:.0f}%")

    names = [r[0] for r in rows]
    S = {key: np.array([r[i] for r in rows]) for i, key in
         enumerate(["", "truth", "patch", "kernel", "phase2"]) if key}

    def band(k, arr, ref):
        r = np.abs(arr - ref)
        return (float(np.median(r[:, k < 5])), float(np.median(r[:, (k >= 5) & (k <= 20)])),
                float(np.median(r[:, k > 20])))

    print(f"\n=== S(k) closure |S - S_truth|  (mode={args.mode}) ===")
    print(f"{'model':>8} | {'k<5':>8} {'5-20':>8} {'k>20':>8}")
    for key in ["patch", "kernel", "phase2"]:
        b = band(k_ref, S[key], S["truth"])
        print(f"{key:>8} | {b[0]:8.4f} {b[1]:8.4f} {b[2]:8.4f}")

    np.savez(args.npz_out, k=k_ref, names=names, **{f"S_{key}": S[key] for key in S})

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, col, lab in [("truth", "k", "truth"), ("patch", "C0", "patch only (~BIND)"),
                          ("kernel", "C2", "kernel only (Phase 1)"),
                          ("phase2", "C3", "Phase 2 (patch+kernel)")]:
        m = S[key].mean(0)
        ax[0].plot(k_ref, m, color=col, label=lab, lw=1.8)
    ax[0].axhline(1, color="b", ls=":", lw=0.7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("P/P_DMO")
    ax[0].set_title(f"total-matter S(k) (mean over {len(names)} sims, {args.mode})")
    ax[0].legend()
    for key, col in [("patch", "C0"), ("kernel", "C2"), ("phase2", "C3")]:
        ax[1].plot(k_ref, (S[key] / S["truth"]).mean(0), color=col)
    ax[1].axhline(1, color="k", lw=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("k [h/Mpc]"); ax[1].set_ylabel("S/S_truth")
    ax[1].set_title("closure ratio (C0 patch, C2 kernel, C3 phase2)")
    ax[1].set_ylim(0.5, 1.5)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[phase2] wrote {out} and {args.npz_out}")


if __name__ == "__main__":
    main()
