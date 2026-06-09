#!/usr/bin/env python
"""Core-library transplant test (path B): restore co-located sharp joint cores.

The composite high-k comes from co-located, sharp DM+Gas+Stars cores (cross-channel
coherence; the per-channel swap showed the whole is ~4x the sum of parts). Post-hoc
sharpening overshoots wildly; regression undershoots. So instead we transplant REAL
truth joint-cores: build a codebook of central core stamps from TRAINING halos
(theta-disjoint from val), and for each held-out val halo pick a mass-matched core,
imposing truth's sharp CO-LOCATED shape while keeping BIND's per-channel central mass
(so integrated mass is preserved). Deployable: selection is by (M[,theta]) only, no
truth at inference. This tests whether that reaches truth S(k) on held-out theta.

Run:  python core_transplant_test.py --n_lib_sims 40 --R 5
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from kernel_bcm_feasibility import BOX, NPIX
from bind.inference.pipeline import build_bind_composite, extract_periodic_cutout
from bind.inference.artifacts import load_halo_cutouts
from bind.metrics import power_spectrum_pylians_2d as PK

PPM = NPIX / BOX
YY, XX = np.mgrid[:128, :128] - 64
RGRID = np.sqrt(XX ** 2 + YY ** 2)


def sk(f, dmo):
    k, p, _ = PK(f, box_size=BOX, MAS="None"); _, pd, _ = PK(dmo, box_size=BOX, MAS="None")
    return k, p / pd


def at(k, a, q):
    return a[np.argmin(np.abs(k - q))]


def truth_patches(truth, centers):
    return np.stack([np.stack([extract_periodic_cutout(truth[ch], int(c[0] * PPM) % NPIX,
                                                       int(c[1] * PPM) % NPIX, 128)
                               for ch in range(3)]) for c in centers])


def blend_mask(R, taper=0.3):
    """1 inside R*(1-taper), cosine taper to 0 at R, 0 outside."""
    r_in = R * (1 - taper)
    w = np.zeros((128, 128))
    w[RGRID <= r_in] = 1.0
    z = (RGRID > r_in) & (RGRID <= R)
    w[z] = 0.5 * (1 + np.cos(np.pi * (RGRID[z] - r_in) / max(R - r_in, 1e-6)))
    return w


def transplant(bind_patch, lib_core, w, mass_mode="bind"):
    """Blend lib core into the center.
    mass_mode 'bind': scale lib SHAPE to BIND's per-channel central mass (strict per-patch
        conservation). 'lib': inject the lib core at its own truth-calibrated amplitude
        (global conservation is handled by build_bind_composite's patch_mass_match)."""
    out = bind_patch.copy()
    for ch in range(3):
        libw = lib_core[ch] * w
        if mass_mode == "bind":
            mass = float((bind_patch[ch] * w).sum()); s = float(libw.sum())
            if s > 0:
                libw = libw * (mass / s)
        out[ch] = bind_patch[ch] * (1 - w) + libw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", default="/mnt/home/mlee1/ceph/fm_testsuite")
    ap.add_argument("--model", default="fm_two_head")
    ap.add_argument("--n_lib_sims", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--R", type=float, default=5.0, help="core radius (pixels)")
    ap.add_argument("--k_near", type=int, default=5, help="sample among k nearest cores")
    ap.add_argument("--mass_mode", choices=["bind", "lib"], default="bind")
    ap.add_argument("--theta_weight", type=float, default=0.0,
                    help="weight on z-scored theta in the codebook match (0 = mass-only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = sorted(glob.glob(f"{args.suite_root}/Test/sim_SB35_*/snap_090"))
    order = np.random.default_rng(args.seed).permutation(len(sims))
    n_val = int(len(sims) * 0.3)
    val_sims = [sims[i] for i in order[:n_val]][:args.n_val]
    train_sims = [sims[i] for i in order[n_val:n_val + args.n_lib_sims]]

    # --- build core library from TRAINING truth (theta-disjoint) ---
    lib_logm, lib_core, lib_theta = [], [], []
    for s in train_sims:
        s = Path(s); mt = s / "mass_threshold_1p000e13"
        truth = np.load(s / "full_maps.npz")["truth_maps"].astype(np.float64)
        cat = np.load(mt / "halo_catalog.npz")
        if len(cat["masses"]) == 0:
            continue
        tp = truth_patches(truth, cat["centers"])
        for i in range(len(cat["masses"])):
            lib_logm.append(np.log10(cat["masses"][i]))
            lib_theta.append(cat["params"][i].astype(np.float64))
            lib_core.append((tp[i] * (RGRID <= args.R + 3)).astype(np.float32))  # store core region
    lib_logm = np.array(lib_logm); lib_core = np.stack(lib_core); lib_theta = np.stack(lib_theta)
    # z-scoring for the codebook distance (logM + theta)
    m_mu, m_sd = lib_logm.mean(), lib_logm.std() + 1e-9
    t_mu, t_sd = lib_theta.mean(0), lib_theta.std(0) + 1e-9
    lib_feat = np.concatenate([((lib_logm - m_mu) / m_sd)[:, None],
                               args.theta_weight * (lib_theta - t_mu) / t_sd], axis=1)
    print(f"[transplant] library: {len(lib_logm)} cores from {len(train_sims)} sims; "
          f"val: {len(val_sims)} sims (theta-disjoint); theta_weight={args.theta_weight}")
    rng = np.random.default_rng(args.seed + 1)
    w = blend_mask(args.R)

    res = {k: [] for k in ["BIND", "transplant", "truth"]}
    mass_err = []
    kref = None
    for s in val_sims:
        s = Path(s); mt = s / "mass_threshold_1p000e13"
        gen = np.load(mt / args.model / "generated_halos.npz")["generated"][:, :3].astype(np.float64)
        fm = np.load(s / "full_maps.npz"); dmo = fm["dmo_fullbox"].astype(np.float64)
        truth = fm["truth_maps"].astype(np.float64)
        cat = np.load(mt / "halo_catalog.npz"); cen = cat["centers"]
        if len(cat["masses"]) == 0:
            continue
        tp = truth_patches(truth, cen)
        halos = [{"halo_center": cen[i, :2], "halo_mass": float(cat["masses"][i]),
                  "r200": float(cat["radii"][i]) / 1e3, "params": cat["params"][i]}
                 for i in range(len(cat["masses"]))]
        cut = load_halo_cutouts(mt / "halo_cutouts.npz")
        kw = dict(box_size=BOX, npix=NPIX, patch_pix=128, patch_mass_match=True,
                  taper_frac=0.15, r200_factor=0.0)

        trans = gen.copy()
        for i in range(len(gen)):
            feat = np.concatenate([[(np.log10(cat["masses"][i]) - m_mu) / m_sd],
                                   args.theta_weight * (cat["params"][i] - t_mu) / t_sd])
            d2 = ((lib_feat - feat) ** 2).sum(1)
            near = np.argsort(d2)[:args.k_near]
            pick = lib_core[rng.choice(near)]
            trans[i] = transplant(gen[i], pick, w, args.mass_mode)
        mass_err.append(abs(trans.sum() - gen.sum()) / gen.sum())

        def comp(g):
            return build_bind_composite(dmo, halos, g, cut, **kw)["composite"].sum(0)
        for name, g in [("BIND", gen), ("transplant", trans), ("truth", tp)]:
            k, v = sk(comp(g), dmo); res[name].append(v); kref = k

    print(f"\n  patch-mass change from transplant: {np.mean(mass_err)*100:.3f}% (should be ~0)")
    print(f"\n{'':>11} {'S(k20)':>8} {'S(k30)':>8} {'S(k50)':>8}")
    for name in ["BIND", "transplant", "truth"]:
        a = np.array(res[name]).mean(0)
        print(f"{name:>11} {at(kref,a,20):8.3f} {at(kref,a,30):8.3f} {at(kref,a,50):8.3f}")
    a_b = np.array(res["BIND"]).mean(0); a_t = np.array(res["transplant"]).mean(0)
    a_tr = np.array(res["truth"]).mean(0)
    for q in [30, 50]:
        gap = at(kref, a_tr, q) - at(kref, a_b, q); cl = at(kref, a_t, q) - at(kref, a_b, q)
        if abs(gap) > 1e-3:
            print(f"  k={q}: closed {100*cl/gap:+.0f}% of BIND->truth gap")


if __name__ == "__main__":
    main()
