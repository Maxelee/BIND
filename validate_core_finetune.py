#!/usr/bin/env python
"""Quick GPU check: is the core-weighted fine-tune sharpening the cores yet?

Generates patches with the BASELINE (fm_two_head epoch047) and the in-progress
core fine-tune (fm_two_head_core/last.ckpt) on held-out test patches, and reports
the metrics that matter, all vs truth:
  * per-channel patch transfer T(k)=sqrt(P_gen/P_truth) at k=30,50 (Stars+Total are
    the high-k drivers) -- same convention as bind_vs_truth_patches, so the numbers
    are comparable to the baseline diagnostic (T_stars~0.54, T_total~0.85).
  * stellar central concentration (r<2px / total) vs truth.
  * goal-1 mass ratios and goal-3 KS (did core-weighting break the bulk?).

Safe to run repeatedly while training runs (reads the latest last.ckpt).
Prints a scorecard + saves a figure. Submit with run_validate_core.sh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

from bind.data import load_file_list, AstroDataset, NormStats
from bind.train import FlowMatchingLit
from bind.inference.pipeline import _denormalize_to_physical
from bind_vs_truth_patches import radial_bins, PATCH_BOX

CH = ["DM_hydro", "Gas", "Stars", "Total"]


def load_ckpt(run_dir, ckpt, device):
    p = Path(run_dir) / "checkpoints" / ckpt
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found (training not far enough?): {p}")
    model = FlowMatchingLit.load_from_checkpoint(str(p), map_location=device).eval().to(device)
    ns = NormStats.load(Path(run_dir) / "norm_stats.npz")
    return model, ns


@torch.no_grad()
def generate(model, ns, files, device, n_steps, batch, workers):
    loader = DataLoader(AstroDataset(files, ns), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
    real, gen = [], []
    for b in loader:
        cond, ls, pr = b["condition"].to(device), b["large_scale"].to(device), b["params"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            g = model.fm.sample(cond, ls, pr, n_steps=n_steps)
        real.append(_denormalize_to_physical(b["target"].numpy().copy(), ns))
        gen.append(_denormalize_to_physical(g.float().cpu().numpy(), ns))
    return np.concatenate(real), np.concatenate(gen)


def chan(arr, ci):
    return arr[:, ci] if ci < 3 else arr[:, :3].sum(1)


def stacked_power(arr, ci, idx, nb):
    S = np.zeros(nb)
    for f in chan(arr, ci):
        F = np.fft.fft2(f - f.mean())
        S += np.bincount(idx, weights=(np.abs(F) ** 2).ravel(), minlength=nb)
    return S


def central_conc(arr):
    s = arr[:, 2]
    return s[:, 62:66, 62:66].sum((1, 2)) / (s.sum((1, 2)) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu")
    ap.add_argument("--runs_dir", default="/mnt/home/mlee1/ceph/fm_runs")
    ap.add_argument("--baseline_run", default="fm_two_head")
    ap.add_argument("--baseline_ckpt", default="epoch047-val_loss0.2138.ckpt")
    ap.add_argument("--core_run", default="fm_two_head_core")
    ap.add_argument("--core_ckpt", default="last.ckpt")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ceph/fm_diag/core_finetune_progress.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    files_all = load_file_list(args.data_root, "test")
    files = [files_all[i] for i in rng.choice(len(files_all),
                                              min(args.n_test, len(files_all)), replace=False)]
    idx, kcent, nb = radial_bins(128, PATCH_BOX)
    print(f"[validate] {len(files)} test patches, n_steps={args.n_steps}, device={device}")

    runs = {"baseline": (args.baseline_run, args.baseline_ckpt),
            "core": (args.core_run, args.core_ckpt)}
    gens, truth = {}, None
    for name, (rd, ck) in runs.items():
        model, ns = load_ckpt(Path(args.runs_dir) / rd, ck, device)
        real, gen = generate(model, ns, files, device, args.n_steps, args.batch, args.workers)
        gens[name] = gen; truth = real
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  generated {name}: {gen.shape}")

    Ptruth = {ci: stacked_power(truth, ci, idx, nb) for ci in range(4)}
    T = {nm: {ci: np.sqrt(stacked_power(g, ci, idx, nb) / np.where(Ptruth[ci] > 0, Ptruth[ci], 1))
              for ci in range(4)} for nm, g in gens.items()}

    def at(arr, q):
        return arr[np.argmin(np.abs(kcent - q))]

    print("\n=== high-k transfer T(k) = sqrt(P_gen/P_truth)  (target = 1) ===")
    print(f"{'channel':>9} | {'base(30)':>9} {'core(30)':>9} | {'base(50)':>9} {'core(50)':>9}")
    for ci, cn in enumerate(CH):
        print(f"{cn:>9} | {at(T['baseline'][ci],30):9.3f} {at(T['core'][ci],30):9.3f} | "
              f"{at(T['baseline'][ci],50):9.3f} {at(T['core'][ci],50):9.3f}")

    print("\n=== goal checks ===")
    cc_t = np.median(central_conc(truth))
    print(f"  stellar central conc (r<2px/total):  truth={cc_t:.3f}  "
          f"baseline={np.median(central_conc(gens['baseline'])):.3f}  "
          f"core={np.median(central_conc(gens['core'])):.3f}")
    for ci, cn in enumerate(CH[:3]):
        mt = truth[:, ci].sum((1, 2))
        rb = np.median(gens['baseline'][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))
        rc = np.median(gens['core'][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))
        lt = np.log10(1 + np.clip(truth[:, ci].ravel(), 0, None))[::17]
        ksb = stats.ks_2samp(np.log10(1 + np.clip(gens['baseline'][:, ci].ravel(), 0, None))[::17], lt).statistic
        ksc = stats.ks_2samp(np.log10(1 + np.clip(gens['core'][:, ci].ravel(), 0, None))[::17], lt).statistic
        print(f"  {cn:>8}: mass ratio base={rb:.3f} core={rc:.3f} (goal 1, ~1) | "
              f"KS base={ksb:.3f} core={ksc:.3f} (goal 3, lower=better)")

    db_s = at(T['core'][2], 50) - at(T['baseline'][2], 50)
    db_t = at(T['core'][3], 50) - at(T['baseline'][3], 50)
    print(f"\n  VERDICT: T(k=50) change core-baseline  Stars {db_s:+.3f}  Total {db_t:+.3f}  "
          f"{'-> sharpening!' if db_t > 0.01 else '-> not yet / tune CORE_WEIGHT'}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for ci, cn in [(2, "Stars"), (3, "Total")]:
        ls = "-" if cn == "Total" else "--"
        ax[0].plot(kcent, T['baseline'][ci], color="tab:orange", ls=ls, label=f"{cn} baseline")
        ax[0].plot(kcent, T['core'][ci], color="tab:blue", ls=ls, label=f"{cn} core")
    ax[0].axhline(1, color="k", ls=":"); ax[0].set_xscale("log"); ax[0].set_ylim(0.4, 1.2)
    ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("T(k)"); ax[0].set_title("high-k transfer"); ax[0].legend(fontsize=8)
    bins = np.linspace(0, 1, 40)
    for nm, c in [("truth", truth), ("baseline", gens['baseline']), ("core", gens['core'])]:
        col = {"truth": "k", "baseline": "tab:orange", "core": "tab:blue"}[nm]
        ax[1].hist(central_conc(c), bins=bins, histtype="step", color=col, density=True, label=nm)
    ax[1].set_xlabel("stellar central conc"); ax[1].set_title("core concentration"); ax[1].legend()
    for ci, cn in enumerate(CH[:3]):
        mt = truth[:, ci].sum((1, 2))
        ax[2].scatter([ci - 0.1], [np.median(gens['baseline'][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))],
                      color="tab:orange"); ax[2].scatter([ci + 0.1],
                      [np.median(gens['core'][:, ci].sum((1, 2)) / np.clip(mt, 1e-30, None))], color="tab:blue")
    ax[2].axhline(1, color="k", ls=":"); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(CH[:3])
    ax[2].set_ylabel("median gen/truth mass"); ax[2].set_title("goal 1: mass (o=base, b=core)")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[validate] wrote {out}")


if __name__ == "__main__":
    main()
