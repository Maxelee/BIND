#!/usr/bin/env python
"""Core-library transplant: restore co-located sharp joint cores in BIND patches.

The composite high-k deficit is the loss of co-located, sharp DM+Gas+Stars cores
(cross-channel coherence; see project_highk_deficit_fix). Regression smooths it,
sharpening overshoots. This restores it by transplanting REAL truth joint-cores:

  build_codebook()  -- extract central core stamps (r<R, all 3 channels together)
                       from TRAINING truth patches, indexed by halo mass (+theta),
                       saved as a small npz artifact.
  CoreCodebook.transplant(gen, masses, ...)  -- for each BIND patch, pick a
                       mass-matched real core (sampled among k_near neighbours for
                       natural core-to-core scatter) and blend its SHAPE into the
                       patch centre, scaled to BIND's per-channel central mass
                       (strict per-patch mass conservation). No truth at inference;
                       selection is by mass only -> deployable, generalizes like BIND.

Module + a `--build` CLI. Used by core_transplant_validate.py and (later) the
bind.inference composite path.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, NPIX
from bind.inference.pipeline import extract_periodic_cutout

PPM = NPIX / BOX
_YY, _XX = np.mgrid[:128, :128] - 64
RGRID = np.sqrt(_XX ** 2 + _YY ** 2)


def truth_patches(truth, centers):
    return np.stack([np.stack([extract_periodic_cutout(truth[ch], int(c[0] * PPM) % NPIX,
                                                       int(c[1] * PPM) % NPIX, 128)
                               for ch in range(3)]) for c in centers])


def blend_mask(R, taper=0.3):
    """1 inside R*(1-taper), cosine taper to 0 at R, 0 outside (128x128)."""
    r_in = R * (1 - taper)
    w = np.zeros((128, 128))
    w[RGRID <= r_in] = 1.0
    z = (RGRID > r_in) & (RGRID <= R)
    w[z] = 0.5 * (1 + np.cos(np.pi * (RGRID[z] - r_in) / max(R - r_in, 1e-6)))
    return w


# ----------------------------------------------------------------------------
def build_codebook(train_sims, model, R_store, out_path):
    """Save central core stamps (3, SS, SS) + logM + theta from training truth."""
    half = int(np.ceil(R_store))
    cores, logm, theta = [], [], []
    for s in train_sims:
        s = Path(s); mt = s / "mass_threshold_1p000e13"
        cat = np.load(mt / "halo_catalog.npz")
        if len(cat["masses"]) == 0:
            continue
        truth = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float32)
        tp = truth_patches(truth, cat["centers"])
        crop = tp[:, :, 64 - half:64 + half + 1, 64 - half:64 + half + 1]
        cores.append(crop.astype(np.float32))
        logm.extend(np.log10(cat["masses"]).tolist())
        theta.append(cat["params"].astype(np.float32))
    cores = np.concatenate(cores); logm = np.array(logm, dtype=np.float32)
    theta = np.concatenate(theta)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cores=cores, logm=logm, theta=theta, half=half)
    print(f"[codebook] {len(logm)} cores, stamp {cores.shape[1:]}, -> {out_path} "
          f"({cores.nbytes/1e6:.0f} MB)")


class CoreCodebook:
    def __init__(self, path):
        d = np.load(path)
        self.cores = d["cores"]            # (N, 3, SS, SS)
        self.logm = d["logm"]
        self.half = int(d["half"])

    def _stamp128(self, core):
        s = np.zeros((3, 128, 128), dtype=np.float64)
        h = self.half
        s[:, 64 - h:64 + h + 1, 64 - h:64 + h + 1] = core
        return s

    def transplant(self, gen, masses, R=12.0, k_near=5, rng=None, mass_mode="bind"):
        """Return refined copy of gen (N,3,128,128) with mass-matched cores blended in."""
        rng = rng or np.random.default_rng(0)
        w = blend_mask(R)
        out = gen.copy()
        for i in range(len(gen)):
            near = np.argsort(np.abs(self.logm - np.log10(masses[i])))[:k_near]
            core = self._stamp128(self.cores[rng.choice(near)].astype(np.float64))
            for ch in range(3):
                libw = core[ch] * w
                if mass_mode == "bind":
                    m = float((gen[i, ch] * w).sum()); sden = float(libw.sum())
                    if sden > 0:
                        libw = libw * (m / sden)
                out[i, ch] = gen[i, ch] * (1 - w) + libw
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_two_head")
    ap.add_argument("--R_store", type=float, default=15.0)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/home/mlee1/ceph/refiner_two_head/core_codebook.npz")
    args = ap.parse_args()

    if args.build:
        sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))
        order = np.random.default_rng(args.seed).permutation(len(sims))
        n_val = int(len(sims) * args.val_frac)
        train_sims = [sims[i] for i in order[n_val:]]   # theta-disjoint from val
        print(f"[codebook] building from {len(train_sims)} train sims "
              f"(val held out: {n_val})")
        build_codebook(train_sims, args.model, args.R_store, args.out)


if __name__ == "__main__":
    main()
