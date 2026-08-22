#!/usr/bin/env python3
"""fidswap step 1 -- recomposite the new fiducial's snap_096 from its stored patches.

The twobound replicas (run_0018/0049/0053) store per-halo patches only; the
per-snapshot composite canvases were never retained.  This rebuilds them with the
EXACT production compositing (bind.inference.pipeline.build_bind_composite --
per-patch mass match, circular R200 x square taper paste, alpha blend, DMO
background on the DM channel, scale_global mass conservation, thermo blended with
the same weights but NO scale_global) using the SHARED fiducial stage-1 geometry.
The stage-1 tree is the DMO projection + halo catalogue: cosmology-independent,
byte-identical for the released fiducial and every twobound run.

Unlike referee/r3b_texture_analysis.py::build_planes (which cropped to gas + y and
skipped the DMO background / scale_global) this keeps all 3 mass + 4 thermo
channels on the native npix grid, so the output is a drop-in for the released
bind_lightcone_tng/snap_096/composite_slabNN.npz canvases.

INTEGRITY GATE (--gate): run the identical reduction on the RELEASED fiducial's
own patches and compare against its stored composite/alpha/patch_scales/
scale_global.  If that reproduces, the rebuild path IS the shipped path.

    python referee/work/fs1_recomposite.py --gate            # verify first
    python referee/work/fs1_recomposite.py --run 49          # then rebuild
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/src")
from bind.inference.pipeline import (  # noqa: E402
    circular_taper_weight,
    paste_halos_2d,
    square_taper_weight,
)

LC = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")
STAGE1 = LC / "snap_096/stage1"
TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")
OUT = Path("/mnt/home/mlee1/ceph/referee_work/fidswap/composites")

# verbatim from the canonical summary.json
TAPER_FRAC = 0.15
R200_FACTOR = 4.0
PATCH_MASS_MATCH = True

MASS_CH = ("dm", "gas", "star")
THERMO_CH = ("y", "T", "entropy", "Pe")


def composite_slab(patch_npz: Path, stage1_npz: Path):
    """bind.inference.pipeline.build_bind_composite, driven off saved patches."""
    s = np.load(stage1_npz)
    g = np.load(patch_npz)
    box = float(s["box_size"])
    npix = int(s["npix"])
    dmo = s["dmo"]
    gen = g["generated_patches"]
    thermo = g["thermo_patches"]
    n, _, pp, _ = gen.shape

    chk = {}
    for k in ("halo_centers", "halo_r200", "halo_masses", "n_halos"):
        a, b = np.asarray(s[k]), np.asarray(g[k])
        chk[f"eq_{k}"] = bool(a.shape == b.shape and np.array_equal(a, b))
    cond_sums = np.asarray(g["condition_sums"], np.float64)
    chk["cond_sums_reldiff"] = float(
        np.max(np.abs(cond_sums / s["condition"].sum(axis=(1, 2)) - 1.0)))

    centers, r200 = s["halo_centers"], s["halo_r200"]
    gen = gen.astype(np.float32).copy()
    scales = np.empty(n)
    for i in range(n):                       # float32 patch sum, as production
        scales[i] = float(cond_sums[i]) / (float(gen[i].sum()) + 1e-30)
    if PATCH_MASS_MATCH:
        gen *= scales[:, None, None, None].astype(np.float32)

    halos = [{"halo_center": centers[i], "r200": float(r200[i])} for i in range(n)]
    sq = square_taper_weight(pp, taper_frac=TAPER_FRAC)
    ppm = npix / box
    wl = [circular_taper_weight(pp, r_pix=float(r200[i]) * ppm * R200_FACTOR,
                                taper_frac=TAPER_FRAC) for i in range(n)]

    canvas, wacc = paste_halos_2d(npix, box, halos, gen, sq, weights_list=wl)
    alpha = np.clip(wacc, 0.0, 1.0)
    comp = np.zeros((3, npix, npix), np.float32)
    comp[0] = (1 - alpha) * dmo + alpha * canvas[0]
    comp[1] = alpha * canvas[1]
    comp[2] = alpha * canvas[2]
    scale_global = float(dmo.sum() / (comp.sum() + 1e-30))
    comp *= scale_global

    tcanvas, _ = paste_halos_2d(npix, box, halos, thermo.astype(np.float32), sq,
                                weights_list=wl)
    ther = (alpha[None] * tcanvas).astype(np.float32)

    chk.update(n_halos=int(n), alpha_cov=float((alpha > 0.01).mean()),
               scale_global=scale_global, scale_med=float(np.median(scales)),
               scale_p01=float(np.percentile(scales, 1)),
               scale_p99=float(np.percentile(scales, 99)))
    return comp, ther, alpha.astype(np.float32), scales, chk


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmax(np.abs(np.where(a != 0, b / a - 1.0, b - a))))


def gate(slabs):
    """reproduce the RELEASED fiducial's own stored canvases from its patches."""
    ok = True
    for sl in slabs:
        p = LC / f"snap_096/composite_slab{sl:02d}.npz"
        d = np.load(p)
        comp, ther, alpha, scales, chk = composite_slab(p, STAGE1 / f"stage1_slab{sl:02d}.npz")
        r = dict(composite=_rel(d["composite"], comp), alpha=_rel(d["alpha"], alpha),
                 patch_scales=_rel(d["patch_scales"], scales),
                 scale_global=_rel(d["scale_global"], chk["scale_global"]),
                 coverage=_rel(d["coverage_pct"], 100 * chk["alpha_cov"]))
        bad = {k: v for k, v in r.items() if not (v == 0.0 or v < 1e-6)}
        ok &= not bad
        print(f"GATE slab{sl:02d}: " + "  ".join(f"{k}={v:.3e}" for k, v in r.items())
              + ("  PASS" if not bad else f"  FAIL {bad}"), flush=True)
    print("GATE OVERALL:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    ap.add_argument("--snap", type=int, default=96)
    ap.add_argument("--slabs", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()

    if a.gate:
        gate(a.slabs)
        return

    src = TWOBOUND / f"run_{a.run:04d}" / f"snap_{a.snap:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    tot = 0
    for sl in a.slabs:
        t0 = time.time()
        comp, ther, alpha, scales, chk = composite_slab(
            src / f"composite_slab{sl:02d}.npz", STAGE1 / f"stage1_slab{sl:02d}.npz")
        out = OUT / f"tb{a.run:02d}_snap{a.snap:03d}_slab{sl:02d}.npz"
        np.savez(out, composite=comp,
                 **{f"t_{c}": ther[i] for i, c in enumerate(THERMO_CH)},
                 alpha=alpha, patch_scales=scales,
                 scale_global=chk["scale_global"], coverage_pct=100 * chk["alpha_cov"],
                 channels=np.array(MASS_CH), thermo_channels=np.array(THERMO_CH),
                 n_halos=chk["n_halos"], slab_idx=sl)
        tot += chk["n_halos"]
        print(f"slab{sl:02d}  {time.time()-t0:6.1f}s  n={chk['n_halos']:4d}  "
              f"geom_ok={all(v for k, v in chk.items() if k.startswith('eq_'))}  "
              f"cond_reldiff={chk['cond_sums_reldiff']:.1e}  "
              f"cov={100*chk['alpha_cov']:.2f}%  sg={chk['scale_global']:.6f}  "
              f"scale med/1/99={chk['scale_med']:.4f}/{chk['scale_p01']:.4f}/{chk['scale_p99']:.4f}"
              f"  -> {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    print("TOTAL halos:", tot)


if __name__ == "__main__":
    main()
