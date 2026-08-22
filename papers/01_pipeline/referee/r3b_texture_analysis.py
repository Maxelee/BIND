#!/usr/bin/env python3
"""R3b: multi-sample-averaging test of the BIND "map-scale texture systematic".

Referee claim under test
------------------------
The paper attributes 5-10% residuals in the gas spectra (Cl_yy, Cl_tautau) to a
"map-scale texture systematic" and *asserts* it is parameter-independent and
therefore harmless for response science.  The referee counters that it looks like
single-sample over-smoothing, which should worsen as feedback makes gas diffuse,
and asks for a multi-sample-averaging test: does the texture shrink as 1/N?

Experiment
----------
The flow-matching sampler draws fresh noise per halo (``bind.model`` line ~454,
``torch.randn``, off the *global* unseeded RNG).  So two paints of the same halos
at the same parameters differ only by sampler noise.

The twobound campaign contains a duplicate-parameter group -- runs 0018, 0049,
0053 have byte-identical 35-dim parameter vectors -- painted independently on the
*shared fiducial stage-1* halos with the same checkpoint.  That is a free N=3
ensemble of independent draws at one parameter location, with no GPU cost.

Decomposition
-------------
Draw i gives map m_i = s + n_i, with s the deterministic (parameters, halos,
weights) signal and n_i zero-mean sampler noise, independent across draws.

    auto      A(k) = <|F_i|^2>_i               = |S|^2 + P_n
    cross     X(k) = <Re F_i F_j^*>_{i<j}      = |S|^2          (unbiased)
    noise     P_n  = A - X
    mean-of-N        P_mean(N) = |S|^2 + P_n / N

so the *excess* power of an N-averaged map over the noise-free signal falls as
1/N.  We verify that directly at N = 1, 2, 3.

Reference maps
--------------
Texture is isolated by comparing against the **pasted-truth** composite: identical
halos, identical paste apertures, identical compositing code -- only the patch
content differs (BIND-generated vs TNG-hydro-measured).  The full-hydro TNG slabs
are NOT used for this, because they additionally contain all the diffuse gas
outside the paste apertures, which is a different systematic entirely.

Outputs
-------
npz of binned spectra + per-halo diagnostics under $WORK, consumed by
``r3b_texture_figs.py``.
"""

from __future__ import annotations

import argparse
import json
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

# ── paths (all campaign trees are READ-ONLY) ─────────────────────────────────
STAGE1 = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1")
CANON = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096")
TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")
TRUTH = Path("/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/snap_096")
WORK = Path("/mnt/home/mlee1/ceph/referee_work/texture")

# the independent-draw ensemble: identical 35-dim parameter vectors
ENSEMBLE = [18, 49, 53]

# compositing settings, taken verbatim from the canonical summary.json
TAPER_FRAC = 0.15
R200_FACTOR = 4.0
PATCH_MASS_MATCH = True
LP_GRID = 4096  # center-crop, the paper's own lensplane convention (paint_yplane)

GAS_CH = 1  # generated_patches = [DM_hydro, Gas, Stars]
Y_CH = 0  # thermo_patches     = [compton_y, T, entropy, P_e]


def _center_crop(a: np.ndarray, n: int) -> np.ndarray:
    m = a.shape[-1]
    if m == n:
        return a
    o = (m - n) // 2
    return a[..., o : o + n, o : o + n]


def build_planes(patch_npz: Path, stage1_npz: Path, *, mass_match: bool = PATCH_MASS_MATCH):
    """Composite one slab's saved patches into (gas, y) planes.

    Mirrors ``bind.inference.pipeline.build_bind_composite``: per-patch mass
    match against the DMO condition (all 3 mass channels scaled by one scalar),
    circular R200 aperture x square taper paste, alpha-weighted blend.  Thermo
    channels are pasted with the same weights but are NOT mass-matched and get no
    scale_global, exactly as in the production code.
    """
    s = np.load(stage1_npz)
    g = np.load(patch_npz)
    box = float(s["box_size"])
    npix = int(s["npix"])
    gen = g["generated_patches"]  # (n, 3, pp, pp)
    thermo = g["thermo_patches"]  # (n, 4, pp, pp)
    n, _, pp, _ = gen.shape
    centers = s["halo_centers"]
    r200 = s["halo_r200"]
    cond_sums = (
        g["condition_sums"] if "condition_sums" in g.files else s["condition"].sum(axis=(1, 2))
    )

    # per-patch mass match (scalar per halo, varies draw to draw)
    gen = gen.astype(np.float32).copy()
    if mass_match:
        m_pred = gen.reshape(n, -1).sum(axis=1)
        scales = np.asarray(cond_sums, np.float64) / (m_pred + 1e-30)
        gen *= scales[:, None, None, None].astype(np.float32)
    else:
        scales = np.ones(n)

    halos = [{"halo_center": centers[i], "r200": float(r200[i])} for i in range(n)]
    sq = square_taper_weight(pp, taper_frac=TAPER_FRAC)
    ppm = npix / box
    wl = [
        circular_taper_weight(pp, r_pix=float(r200[i]) * ppm * R200_FACTOR, taper_frac=TAPER_FRAC)
        for i in range(n)
    ]

    canvas, wacc = paste_halos_2d(npix, box, halos, gen, sq, weights_list=wl)
    alpha = np.clip(wacc, 0.0, 1.0)
    gas = (alpha * canvas[GAS_CH]).astype(np.float32)

    tcanvas, _ = paste_halos_2d(npix, box, halos, thermo.astype(np.float32), sq, weights_list=wl)
    y = (alpha * tcanvas[Y_CH]).astype(np.float32)

    return (
        _center_crop(gas, LP_GRID),
        _center_crop(y, LP_GRID),
        {"patch_scales": scales, "alpha_cov": float((alpha > 0.01).mean())},
    )


def kgrid(n: int, L: float):
    kf = 2.0 * np.pi / L
    kx = np.fft.fftfreq(n, d=1.0 / n) * kf
    ky = np.fft.rfftfreq(n, d=1.0 / n) * kf
    return np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)


def make_bins(kk: np.ndarray, n_bin: int = 26):
    kf = kk[0, 1]
    kmax = kk.max() / np.sqrt(2.0)  # stay inside the Nyquist circle
    edges = np.geomspace(2 * kf, kmax, n_bin + 1)
    idx = np.digitize(kk.ravel(), edges) - 1
    good = (idx >= 0) & (idx < n_bin)
    idx = idx[good]
    cnt = np.bincount(idx, minlength=n_bin).astype(float)
    kcen = np.bincount(idx, weights=kk.ravel()[good], minlength=n_bin) / np.maximum(cnt, 1)
    return edges, idx, good, cnt, kcen


def binned(field2d: np.ndarray, idx, good, cnt, n_bin: int):
    v = field2d.ravel()[good]
    return np.bincount(idx, weights=v, minlength=n_bin) / np.maximum(cnt, 1)


def ensemble_paths(slab: int, root: Path | None):
    """Ensemble member npz paths for one slab.

    Default: the free duplicate-parameter twobound trio.  With ``--ensemble_root``
    (output of ``r3b_gpu_repaint.py``): ``<root>/sample_NN/composite_slabNN.npz``.
    """
    if root is None:
        return {
            f"draw{r}": TWOBOUND / f"run_{r:04d}" / f"snap_096/composite_slab{slab:02d}.npz"
            for r in ENSEMBLE
        }
    samples = sorted(p for p in root.glob("sample_*") if p.is_dir())
    if not samples:
        raise FileNotFoundError(f"no sample_* dirs under {root}")
    return {p.name: p / f"composite_slab{slab:02d}.npz" for p in samples}


def analyse_slab(slab: int, out: dict, verbose=True, ens_root: Path | None = None):
    t0 = time.time()
    s1 = STAGE1 / f"stage1_slab{slab:02d}.npz"

    sources = dict(ensemble_paths(slab, ens_root))
    draw_keys = list(sources)
    sources["canonical"] = CANON / f"composite_slab{slab:02d}.npz"
    sources["truth"] = TRUTH / f"composite_slab{slab:02d}.npz"

    L = 205.0 * LP_GRID / 4198.0
    kk = kgrid(LP_GRID, L)
    n_bin = 26
    edges, idx, good, cnt, kcen = make_bins(kk, n_bin)

    F = {"gas": {}, "y": {}}
    meta = {}
    for name, p in sources.items():
        gas, y, m = build_planes(p, s1)
        meta[name] = m
        F["gas"][name] = np.fft.rfft2(gas.astype(np.float64))
        F["y"][name] = np.fft.rfft2(y.astype(np.float64))
        if verbose:
            print(
                f"  [{slab}] {name:>10s}  gas.mean={gas.mean():.4g} y.mean={y.mean():.4g}"
                f"  ({time.time()-t0:.0f}s)",
                flush=True,
            )
        del gas, y

    npix2 = float(LP_GRID) ** 2
    norm = (L**2) / npix2**2  # P(k) in (Mpc/h)^2 x field^2

    res = {"k": kcen, "n_modes": cnt, "L": L, "slab": slab}
    for fld in ("gas", "y"):
        Fd = [F[fld][kk] for kk in draw_keys]
        N = len(Fd)
        # single-draw autos
        autos = [binned(np.abs(f) ** 2 * norm, idx, good, cnt, n_bin) for f in Fd]
        res[f"{fld}_auto_each"] = np.array(autos)
        res[f"{fld}_auto"] = np.mean(autos, axis=0)
        # cross-spectra between distinct draws -> unbiased signal power
        crosses = []
        for i in range(N):
            for j in range(i + 1, N):
                crosses.append(
                    binned(np.real(Fd[i] * np.conj(Fd[j])) * norm, idx, good, cnt, n_bin)
                )
        res[f"{fld}_cross_each"] = np.array(crosses)
        res[f"{fld}_cross"] = np.mean(crosses, axis=0)
        # mean-of-N maps (FFT is linear -> average the transforms)
        res[f"{fld}_meanN1"] = res[f"{fld}_auto"]
        pair_means = []
        for i in range(N):
            for j in range(i + 1, N):
                fm = 0.5 * (Fd[i] + Fd[j])
                pair_means.append(binned(np.abs(fm) ** 2 * norm, idx, good, cnt, n_bin))
        res[f"{fld}_meanN2"] = np.mean(pair_means, axis=0)
        fm3 = sum(Fd) / N
        res[f"{fld}_meanN3"] = binned(np.abs(fm3) ** 2 * norm, idx, good, cnt, n_bin)
        # references
        for ref in ("canonical", "truth"):
            fr = F[fld][ref]
            res[f"{fld}_{ref}"] = binned(np.abs(fr) ** 2 * norm, idx, good, cnt, n_bin)
        # truth x mean-of-3 (coherence of the averaged map with truth)
        res[f"{fld}_truth_x_mean3"] = binned(
            np.real(fm3 * np.conj(F[fld]["truth"])) * norm, idx, good, cnt, n_bin
        )
        res[f"{fld}_truth_x_canon"] = binned(
            np.real(F[fld]["canonical"] * np.conj(F[fld]["truth"])) * norm,
            idx, good, cnt, n_bin,
        )
    out[f"slab{slab:02d}"] = res
    out.setdefault("meta", {})[f"slab{slab:02d}"] = {
        k: {"alpha_cov": v["alpha_cov"]} for k, v in meta.items()
    }
    print(f"  slab {slab} done in {time.time()-t0:.0f}s", flush=True)


def perhalo_diagnostics(slabs):
    """Per-halo stochastic scatter vs halo mass and vs gas diffuseness.

    This is the *mechanism* test.  The referee's hypothesis is that the texture
    grows as gas becomes more diffuse.  We cannot repaint at a strong-feedback
    parameter node without a GPU, but the halo population at one parameter point
    already spans a wide range of gas diffuseness, so we can ask directly whether
    the sampler's stochastic amplitude tracks diffuseness.
    """
    rows = []
    for slab in slabs:
        gs, ts, cs = [], [], []
        for r in ENSEMBLE:
            d = np.load(TWOBOUND / f"run_{r:04d}" / f"snap_096/composite_slab{slab:02d}.npz")
            gs.append(d["generated_patches"][:, GAS_CH].astype(np.float32))
            ts.append(d["thermo_patches"][:, Y_CH].astype(np.float32))
            cs.append(d["condition_sums"] if "condition_sums" in d.files else None)
        s1 = np.load(STAGE1 / f"stage1_slab{slab:02d}.npz")
        mass = s1["halo_masses"]
        r200 = s1["halo_r200"]
        G = np.stack(gs)  # (N, n, pp, pp)
        T = np.stack(ts)
        n = G.shape[1]
        gm = G.mean(0)
        tm = T.mean(0)
        # unbiased per-halo noise std / signal rms
        gvar = ((G - gm) ** 2).sum(0) / (len(ENSEMBLE) - 1)
        tvar = ((T - tm) ** 2).sum(0) / (len(ENSEMBLE) - 1)
        g_noise = np.sqrt(gvar.reshape(n, -1).mean(1))
        t_noise = np.sqrt(tvar.reshape(n, -1).mean(1))
        g_sig = np.sqrt((gm**2).reshape(n, -1).mean(1))
        t_sig = np.sqrt((tm**2).reshape(n, -1).mean(1))
        # diffuseness proxy: fraction of patch gas mass outside R200 (in-patch)
        pp = G.shape[-1]
        yy, xx = np.mgrid[:pp, :pp]
        rr = np.sqrt((xx - pp / 2 + 0.5) ** 2 + (yy - pp / 2 + 0.5) ** 2)
        # patch spans 6.25 Mpc/h across pp pixels
        pix_mpc = 6.25 / pp
        diff = np.zeros(n)
        conc = np.zeros(n)
        for i in range(n):
            r200_pix = float(r200[i]) / pix_mpc
            m_in = gm[i][rr <= r200_pix].sum()
            m_tot = gm[i].sum()
            diff[i] = 1.0 - m_in / (m_tot + 1e-30)
            conc[i] = m_in / (m_tot + 1e-30)
        rows.append(
            dict(
                slab=np.full(n, slab), mass=mass, r200=r200,
                g_noise=g_noise, g_sig=g_sig, t_noise=t_noise, t_sig=t_sig,
                diffuse=diff, conc=conc,
            )
        )
    out = {}
    for k in rows[0]:
        out[k] = np.concatenate([r[k] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slabs", type=str, default="0,1,2,3")
    ap.add_argument("--out", type=str, default=str(WORK / "r3b_spectra.npz"))
    ap.add_argument("--perhalo_only", action="store_true")
    ap.add_argument("--ensemble_root", type=Path, default=None,
                    help="Directory of sample_*/ dirs from r3b_gpu_repaint.py "
                         "(default: the free duplicate-parameter twobound trio)")
    args = ap.parse_args()
    slabs = [int(x) for x in args.slabs.split(",")]
    WORK.mkdir(parents=True, exist_ok=True)

    if not args.perhalo_only:
        out: dict = {}
        for s in slabs:
            analyse_slab(s, out, ens_root=args.ensemble_root)
        flat = {}
        for slabkey, res in out.items():
            if slabkey == "meta":
                continue
            for k, v in res.items():
                flat[f"{slabkey}/{k}"] = np.asarray(v)
        flat["meta_json"] = np.array(json.dumps(out.get("meta", {})))
        np.savez_compressed(args.out, **flat)
        print("wrote", args.out, flush=True)

    ph = perhalo_diagnostics(slabs)
    np.savez_compressed(str(WORK / "r3b_perhalo.npz"), **ph)
    print("wrote", WORK / "r3b_perhalo.npz", flush=True)


if __name__ == "__main__":
    main()
