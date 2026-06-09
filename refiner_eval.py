#!/usr/bin/env python
"""Evaluate the trained refiner on HELD-OUT-theta val sims (the honest test).

Applies refiner.pt to BIND patches on the theta-disjoint val split, and reports:
  * per-channel transfer T(k) = sqrt(P/P_truth) BEFORE (BIND) vs AFTER (refined),
  * recomposited total-matter S(k) = P/P_DMO vs truth and vs uncorrected BIND.
The refiner saw none of these parameters and has no theta input, so any gain here
is a genuine generalization improvement.  CPU (small CNN).

Run:  python refiner_eval.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from refiner_train import Refiner, Norm, radial_index, power_per_bin
from kernel_bcm_feasibility import BOX, NPIX, _periodic_cutout
from bind_vs_truth_patches import PATCH_PIX, PATCH_BOX, radial_bins
from bind.inference.pipeline import build_bind_composite
from bind.inference.artifacts import load_halo_cutouts
from bind.metrics import power_spectrum_pylians_2d

CHANNELS = ["DM_hydro", "Gas", "Stars", "Total"]


def val_sims(suite_root, val_frac, seed):
    sims = sorted(glob.glob(f"{suite_root}/Test/sim_SB35_*/snap_090"))
    order = np.random.default_rng(seed).permutation(len(sims))
    return [sims[i] for i in order[: int(len(sims) * val_frac)]]


def sk(total, dmo):
    k, p, _ = power_spectrum_pylians_2d(total, box_size=BOX, MAS="None")
    _, pd, _ = power_spectrum_pylians_2d(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--ckpt", default="/mnt/home/mlee1/ceph/refiner_two_head/refiner.pt")
    ap.add_argument("--model", default="fm_two_head")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ceph/fm_diag/refiner_eval.png")
    args = ap.parse_args()

    device = torch.device("cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = Refiner(w=int(ck["width"])).to(device).eval()
    model.load_state_dict(ck["model"])
    norm = Norm(ck["mean"], ck["std"], device)

    @torch.no_grad()
    def refine(gen3):  # (N,3,128,128) physical -> refined physical
        x = torch.from_numpy(gen3.astype(np.float32))
        return norm.inv(model(norm.fwd(x))).numpy()

    sims = val_sims(args.suite_root, args.val_frac, args.seed)
    idx, kcent, nb = radial_bins(PATCH_PIX, PATCH_BOX)
    SA_b = np.zeros((4, nb)); SA_r = np.zeros((4, nb)); SB = np.zeros((4, nb))
    s_truth, s_bind, s_ref, s_ceil = [], [], [], []
    kref = None
    ppm = NPIX / BOX

    for s in sims:
        s = Path(s); mt = s / "mass_threshold_1p000e13"
        gen = np.load(mt / args.model / "generated_halos.npz")["generated"][:, :3].astype(np.float64)
        ref = refine(gen).astype(np.float64)
        fm = np.load(s / "full_maps.npz")
        dmo = fm["dmo_fullbox"].astype(np.float64)
        truth = fm["truth_maps"].astype(np.float64)
        cat = np.load(mt / "halo_catalog.npz")

        for i, c in enumerate(cat["centers"]):
            cx = int(c[0] * ppm) % NPIX; cy = int(c[1] * ppm) % NPIX
            gtot = np.zeros((PATCH_PIX,) * 2); rtot = np.zeros((PATCH_PIX,) * 2); ttot = np.zeros((PATCH_PIX,) * 2)
            for ch in range(3):
                b = _periodic_cutout(truth[ch], cx, cy, PATCH_PIX)
                for a, S in [(gen[i, ch], SA_b), (ref[i, ch], SA_r), (b, SB)]:
                    F = np.fft.fft2(a - a.mean())
                    S[ch] += np.bincount(idx, weights=(np.abs(F) ** 2).ravel(), minlength=nb)
                gtot += gen[i, ch]; rtot += ref[i, ch]; ttot += b
            for tot, S in [(gtot, SA_b), (rtot, SA_r), (ttot, SB)]:
                F = np.fft.fft2(tot - tot.mean())
                S[3] += np.bincount(idx, weights=(np.abs(F) ** 2).ravel(), minlength=nb)

        halos = [{"halo_center": cat["centers"][i, :2], "halo_mass": float(cat["masses"][i]),
                  "r200": float(cat["radii"][i]) / 1e3, "params": cat["params"][i]}
                 for i in range(len(cat["masses"]))]
        cutouts = load_halo_cutouts(mt / "halo_cutouts.npz")
        # truth-patch composite = the ACHIEVABLE ceiling (perfect patches, same pasting)
        truthp = np.stack([np.stack([_periodic_cutout(truth[ch], int(c[0] * ppm) % NPIX,
                                                       int(c[1] * ppm) % NPIX, PATCH_PIX)
                                     for ch in range(3)]) for c in cat["centers"]])
        kw = dict(box_size=BOX, npix=NPIX, patch_pix=PATCH_PIX, patch_mass_match=True,
                  taper_frac=0.15, r200_factor=0.0)
        comp_b = build_bind_composite(dmo, halos, gen, cutouts, **kw)["composite"]
        comp_r = build_bind_composite(dmo, halos, ref, cutouts, **kw)["composite"]
        comp_c = build_bind_composite(dmo, halos, truthp, cutouts, **kw)["composite"]
        k, st = sk(truth.sum(0), dmo); _, sb = sk(comp_b.sum(0), dmo)
        _, sr = sk(comp_r.sum(0), dmo); _, sc = sk(comp_c.sum(0), dmo)
        kref = k; s_truth.append(st); s_bind.append(sb); s_ref.append(sr); s_ceil.append(sc)
        print(f"  {s.parent.name}: done")

    Tb = np.sqrt(SA_b / np.where(SB > 0, SB, 1)); Tr = np.sqrt(SA_r / np.where(SB > 0, SB, 1))
    s_truth, s_bind, s_ref, s_ceil = map(lambda a: np.array(a), (s_truth, s_bind, s_ref, s_ceil))

    def at(a, q): return a[np.argmin(np.abs(kcent - q))]
    print(f"\n=== per-channel patch T(k): BIND -> refined (held-out theta) ===")
    print(f"{'chan':>9} | {'T_BIND(30)':>10} {'T_ref(30)':>10} | {'T_BIND(50)':>10} {'T_ref(50)':>10}")
    for ci, cn in enumerate(CHANNELS):
        print(f"{cn:>9} | {at(Tb[ci],30):10.3f} {at(Tr[ci],30):10.3f} | "
              f"{at(Tb[ci],50):10.3f} {at(Tr[ci],50):10.3f}")

    def atk(a, q): return a.mean(0)[np.argmin(np.abs(kref - q))]
    print(f"\n=== total-matter S(k) closure vs truth ===")
    print(f"{'':>12} {'k=10':>7} {'k=30':>7} {'k=50':>7}")
    for lab, a in [("truth_full", s_truth), ("truth-patch*", s_ceil),
                   ("BIND", s_bind), ("BIND+ref", s_ref)]:
        print(f"{lab:>12} {atk(a,10):7.3f} {atk(a,30):7.3f} {atk(a,50):7.3f}")
    print("  (* truth-patch = achievable ceiling: perfect patches, same pasting)")
    # fraction of the recoverable gap (BIND -> ceiling) that the refiner closed
    gap = atk(s_ceil, 50) - atk(s_bind, 50)
    closed = atk(s_ref, 50) - atk(s_bind, 50)
    if abs(gap) > 1e-3:
        print(f"  recoverable-gap closed at k=50: {100*closed/gap:+.0f}%  "
              f"(BIND {atk(s_bind,50):.3f} -> ref {atk(s_ref,50):.3f} -> ceiling {atk(s_ceil,50):.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for ci, cn in enumerate(["Stars", "Total"]):
        c = 2 if cn == "Stars" else 3
        ax[0].plot(kcent, Tb[c], "--", label=f"{cn} BIND")
        ax[0].plot(kcent, Tr[c], "-", label=f"{cn} refined")
    ax[0].axhline(1, color="gray", ls=":"); ax[0].set_xscale("log")
    ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("T(k)"); ax[0].legend()
    ax[0].set_title("patch transfer (held-out theta): BIND vs refined")
    for a, col, lab in [(s_truth, "k", "truth"), (s_ceil, "0.5", "truth-patch ceiling"),
                        (s_bind, "C1", "BIND"), (s_ref, "C0", "BIND+refiner")]:
        ax[1].plot(kref, a.mean(0), col, label=lab,
                   lw=1.8, ls="--" if lab == "truth-patch ceiling" else "-")
    ax[1].axhline(1, color="gray", ls=":"); ax[1].set_xscale("log")
    ax[1].set_xlabel("k [h/Mpc]"); ax[1].set_ylabel("P/P_DMO"); ax[1].legend()
    ax[1].set_title("total-matter suppression")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[refiner-eval] wrote {out}")


if __name__ == "__main__":
    main()
