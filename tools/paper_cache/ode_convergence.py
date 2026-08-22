#!/usr/bin/env python3
"""ODE (Euler) convergence test for the BIND flow-matching sampler — referee I-16.

Answers: is the small-scale (high-k) deficit an artifact of first-order Euler
discretisation at n_steps=20?

Design
------
* PAIRED SEEDS. ``FlowMatching.sample`` draws x0 = randn(B, C, H, W) internally.
  Because that draw has the *same shape* for every n_steps, seeding the global
  RNG immediately before each call gives *bit-identical* initial conditions
  across n_steps, provided batch_size is held fixed.  Differences between
  variants are then pure discretisation error, with zero realisation noise —
  which is what the earlier 4-sim test (pk_diagnostics/nsteps_test.py) lacked.
* REFERENCE SOLUTION n_steps=400 (fp32).  Convergence is measured *against the
  reference*, not against n=20, so a monotone trend is visible.
* THREE LEVELS of metric, cheapest first:
    L1 per-patch field error   ||x(N) - x(ref)|| / ||x(ref)||   (physical + log space)
    L2 per-patch P(k) ratio    P_N(k) / P_ref(k) per channel and for the total
    L3 full composite S(k)     the referee's actual request (CPU paste, ~5 s/sim)
* AMP CONTROL. n=20 is also run in fp32 (--fp32-control) so bf16 rounding is
  separated from discretisation error.

Runs off the *cached* suite artifacts (halo_cutouts.npz / halo_catalog.npz /
full_maps.npz) — no particle re-projection, no SLURM needed.

Usage
-----
    python ode_convergence.py --suite CV --n-sims 27 --steps 20,50,100,200,400 \
        --out ode_conv_cv.npz
    python ode_convergence.py --suite Test --n-sims 30 --out ode_conv_sb35.npz
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "paper_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import paper_config as C
from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts
from bind.inference.paint import Model
from bind.inference.pipeline import build_bind_composite
from bind.metrics import power_spectrum_pylians_2d

RUNS = "/mnt/home/mlee1/ceph/fm_runs"
DEF_CKPT = f"{RUNS}/fm_thermo/checkpoints/kept/keep_epoch064_ema.ckpt"
DEF_NORM = f"{RUNS}/fm_thermo/norm_stats.npz"
PASTE = dict(box_size=C.BOX_SIZE, npix=C.N_PIX_FULL, patch_pix=C.PATCH_PIX,
             patch_mass_match=True, taper_frac=0.15, r200_factor=4.0,
             paste_mode="shared")
BANDS = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 70)]


def pk2d(f):
    return power_spectrum_pylians_2d(f, box_size=C.BOX_SIZE, MAS="None", threads=2)[:2]


def patch_pk(patch, box=C.PATCH_BOX):
    """2D P(k) of one 128^2 patch (6.25 Mpc/h)."""
    return power_spectrum_pylians_2d(patch, box_size=box, MAS="None", threads=1)[:2]


def gen(model, cutouts, params, n_steps, batch_size, amp, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return model.generate(cutouts, params, n_steps=n_steps, batch_size=batch_size,
                          use_amp=amp, progress=False)[:, :C.N_MASS_CH].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="CV")
    ap.add_argument("--n-sims", type=int, default=27)
    ap.add_argument("--steps", default="20,50,100,200,400")
    ap.add_argument("--ref-steps", type=int, default=400)
    ap.add_argument("--mass-min", type=float, default=1e13)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--fp32-control", action="store_true", default=True)
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--norm", default=DEF_NORM)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]

    model = Model.from_files(a.ckpt, a.norm, device="cuda")
    recs = [r for r in C.discover_sims(suites=(a.suite,))][: a.n_sims]
    res, t0 = {}, time.time()

    for rec in recs:
        key = rec["key"]
        fm = C.load_full_maps(rec); dmo = fm["dmo_fullbox"]
        halos, _, _, _ = load_halo_catalog(rec["catalog"])
        cutouts = load_halo_cutouts(rec["cutouts"])
        masses = np.array([h["halo_mass"] for h in halos])
        idx = np.where(masses >= a.mass_min)[0]
        if idx.size == 0:
            continue
        h_s = [halos[i] for i in idx]; c_s = [cutouts[i] for i in idx]
        params = np.asarray(halos[0]["params"], np.float32)

        k, res[f"{key}/truth"] = pk2d(fm["truth_maps"][: C.N_MASS_CH].sum(0))
        _, res[f"{key}/dmo"] = pk2d(dmo)
        res[f"{key}/k"] = k; res[f"{key}/n"] = np.array(idx.size)

        # reference (fp32, high N)
        ref = gen(model, c_s, params, a.ref_steps, a.batch_size, False, a.seed)
        variants = [(f"n{n}", n, True) for n in steps]
        if a.fp32_control:
            variants += [(f"n{n}_fp32", n, False) for n in (20, 50)]
        for tag, n, amp in variants:
            g = gen(model, c_s, params, n, a.batch_size, amp, a.seed)
            # L1: paired field error vs reference, per channel
            num = np.sqrt(((g - ref) ** 2).sum(axis=(2, 3)))
            den = np.sqrt((ref ** 2).sum(axis=(2, 3))) + 1e-30
            res[f"{key}/{tag}/L2rel"] = (num / den).astype(np.float32)   # (N, 3)
            res[f"{key}/{tag}/mass"] = g.sum(axis=(2, 3)).astype(np.float32)
            # L2: per-patch total-matter P(k), stacked
            pp = np.stack([patch_pk(p.sum(0))[1] for p in g])
            res[f"{key}/{tag}/patch_pk"] = pp.astype(np.float32)
            # L3: composite S(k)
            b = build_bind_composite(dmo, h_s, g, c_s, **PASTE)
            _, res[f"{key}/{tag}/pk"] = pk2d(b["composite"].sum(0))
        pp = np.stack([patch_pk(p.sum(0))[1] for p in ref])
        res[f"{key}/ref/patch_pk"] = pp.astype(np.float32)
        res[f"{key}/ref/kpatch"] = patch_pk(ref[0].sum(0))[0]
        b = build_bind_composite(dmo, h_s, ref, c_s, **PASTE)
        _, res[f"{key}/ref/pk"] = pk2d(b["composite"].sum(0))
        print(f"{key}: {idx.size} halos, {time.time()-t0:.0f}s elapsed", flush=True)

    np.savez_compressed(a.out, **res)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
