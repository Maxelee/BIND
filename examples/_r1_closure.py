#!/usr/bin/env python
"""R1a/R1b (docs/p4c_referee_hardening_plan.md): estimator closure tests.

R1a — resampling closure. The T2f data path samples the natively-0.5'/px ACT
map onto a 0.29296875'/px thumbnail grid by cubic interpolation. Does that
operator bias the CAP? Test on BIND's own fiducial y map (native 0.293'/px,
1.6' beam applied), at REAL mock-LRG halo positions (peaked profiles — the
demanding case): native CAP  vs  [native -> 0.5' grid -> cubic re-upsample
-> CAP]. Two degrade variants bracket the instrument pixel window: (i) point
sampling (no window), (ii) 0.5'-pixel-window pre-smoothing (sigma =
0.5'/sqrt(12) Gaussian approx). The (native - resampled) stacked-CAP bias
curve IS the resampling systematic.

R1b — exact-beam estimate. The analysis treats the ACT ILC beam as a 1.6'
Gaussian (b_ell verified <0.1% on the table). CAP is linear in the map, so
the beam-shape effect on the STACK is the ell-space ratio
b_exact/b_gauss applied to the stacked profile's spectrum: computed here
from the BIND stacked thumbnail, no re-measurement needed.

Writes figs/R1_closure.png + verdicts/R1.json.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter

KS = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront")
LC = KS / "lightcone"
FID_Y = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/y_maps.npz")
BEAM_TXT = Path("/mnt/home/mlee1/ceph/paper3/B/downloads/act_dr6_planck_ymap/ilc_beam.txt")

PIX_FINE = 0.29296875
PIX_COARSE = 0.5
BEAM_FWHM = 1.6
REALIZATION, SRC_IDX = 1, 4
THETA_ARCMIN = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0])
EDGE_PX = 80


def load_npz_slice(path, key, prefix_index, shape, dtype=np.float32):
    """Stream one [i, j] slab out of a (deflated) npz cube without loading
    the 10GB array (the session cgroup is ~10GB)."""
    zf = zipfile.ZipFile(path)
    f = zf.open(f"{key}.npy")
    version = np.lib.format.read_magic(f)
    np.lib.format._check_version(version)
    hdr_shape, fortran, hdr_dtype = np.lib.format._read_array_header(f, version)
    assert hdr_shape == shape and not fortran, (hdr_shape, shape)
    slab_elems = int(np.prod(shape[2:]))
    itemsize = np.dtype(hdr_dtype).itemsize
    offset = (prefix_index[0] * shape[1] + prefix_index[1]) * slab_elems * itemsize
    f.seek(f.tell() + offset)
    buf = f.read(slab_elems * itemsize)
    return np.frombuffer(buf, dtype=hdr_dtype).reshape(shape[2:]).astype(dtype)


def cap_stack(map2d, ic, jc, theta_px):
    from lightcone_cap_stack import cap_batch
    caps = cap_batch(map2d, ic, jc, np.broadcast_to(theta_px, (len(ic), len(theta_px))).copy())
    return np.nanmean(caps, axis=0)


def main():
    from lightcone_cap_stack import GEOM_PATH, select_sample, apply_beam
    from bind.inference.lux_geometry import load_geometry, plane_to_map, RT_GRID

    # --- native beam-applied fiducial y map, one realization --------------
    ymap = load_npz_slice(FID_Y, "y", (REALIZATION - 1, SRC_IDX),
                          (50, 5, 1024, 1024))
    ymap = apply_beam(ymap[None], BEAM_FWHM)[0].astype(np.float64)

    # --- mock LRG halo positions on this realization ----------------------
    geom = load_geometry(GEOM_PATH)
    sel = select_sample("lrg", "fid")
    copies = plane_to_map(sel["pixel_i"], sel["pixel_j"], sel["plane_p"],
                          REALIZATION, geom)
    ic, jc = [], []
    for rows in copies:
        if rows.shape[0]:
            ic.append(rows[0, 0]); jc.append(rows[0, 1])
    ic = np.clip(np.round(ic), 0, RT_GRID - 1).astype(np.int64)
    jc = np.clip(np.round(jc), 0, RT_GRID - 1).astype(np.int64)
    inb = (ic > EDGE_PX) & (ic < RT_GRID - EDGE_PX) & (jc > EDGE_PX) & (jc < RT_GRID - EDGE_PX)
    ic, jc = ic[inb], jc[inb]
    print(f"[R1a] {len(ic)} halo centers")

    theta_px_fine = THETA_ARCMIN / PIX_FINE
    cap_native = cap_stack(ymap, ic, jc, theta_px_fine)

    # --- degrade -> 0.5' grid -> cubic re-upsample -> CAP ----------------
    n_c = int(round(1024 * PIX_FINE / PIX_COARSE))          # 600
    gc = (np.arange(n_c) + 0.5) * PIX_COARSE / PIX_FINE - 0.5
    gf = (np.arange(1024) + 0.5) * PIX_FINE / PIX_COARSE - 0.5

    results = {}
    for tag, presmooth in [("point-sample", 0.0),
                           ("pixel-window", PIX_COARSE / np.sqrt(12.0) / PIX_FINE)]:
        src = gaussian_filter(ymap, presmooth) if presmooth > 0 else ymap
        CI, CJ = np.meshgrid(gc, gc, indexing="ij")
        coarse = map_coordinates(src, [CI.ravel(), CJ.ravel()], order=3,
                                 mode="nearest").reshape(n_c, n_c)
        cf = spline_filter(coarse, order=3, mode="nearest")
        FI, FJ = np.meshgrid(gf, gf, indexing="ij")
        fine = map_coordinates(cf, [FI.ravel(), FJ.ravel()], order=3,
                               prefilter=False, mode="nearest").reshape(1024, 1024)
        results[tag] = cap_stack(fine, ic, jc, theta_px_fine)

    bias = {t: results[t] / cap_native - 1.0 for t in results}

    # --- R1b: exact-b_ell vs Gaussian on the stacked profile --------------
    bl = np.loadtxt(BEAM_TXT)
    ell_b, b_exact = bl[:, 0], bl[:, 1]
    sig_b = np.deg2rad(BEAM_FWHM / 60.0) / (2 * np.sqrt(2 * np.log(2)))
    b_gauss = np.exp(-0.5 * ell_b * (ell_b + 1) * sig_b ** 2)
    b_gauss /= b_gauss[0]
    # stacked thumbnail around the halos -> its ring-averaged 2D spectrum
    w = 40
    stamps = np.array([ymap[i - w:i + w, j - w:j + w] for i, j in zip(ic, jc)
                       if i > w and j > w and i < 1024 - w and j < 1024 - w])
    stack_img = stamps.mean(0)
    F = np.fft.fftshift(np.fft.fft2(stack_img - stack_img.mean()))
    kfreq = np.fft.fftshift(np.fft.fftfreq(2 * w, d=np.deg2rad(PIX_FINE / 60)))
    KX, KY = np.meshgrid(2 * np.pi * kfreq, 2 * np.pi * kfreq)
    ELL = np.hypot(KX, KY)
    ratio = np.interp(ELL.ravel(), ell_b, b_exact / np.maximum(b_gauss, 1e-12),
                      left=1.0, right=(b_exact[-1] / b_gauss[-1])).reshape(ELL.shape)
    stack_exact = np.real(np.fft.ifft2(np.fft.ifftshift(F * ratio))) + stack_img.mean()
    cw = w
    caps0 = cap_stack(stack_img, np.array([cw]), np.array([cw]), theta_px_fine)
    caps1 = cap_stack(stack_exact, np.array([cw]), np.array([cw]), theta_px_fine)
    beam_bias = caps1 / caps0 - 1.0

    # --- figure + verdict -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for t, c in [("point-sample", "tab:blue"), ("pixel-window", "tab:orange")]:
        ax.plot(THETA_ARCMIN, 100 * bias[t], "o-", color=c,
                label=f"resampling bias ({t} degrade)")
    ax.plot(THETA_ARCMIN, 100 * beam_bias, "s--", color="tab:green",
            label=r"exact $b_\ell$ vs 1.6' Gaussian (R1b)")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhspan(-2, 2, color="0.92", zorder=0, label="±2% gate")
    ax.set_xlabel(r"$\theta_d$ [arcmin]")
    ax.set_ylabel("stacked-CAP bias [%]")
    ax.set_title("R1: closure of the 0.5'→0.293' resampling + beam-shape effect\n"
                 f"(BIND fiducial y, {len(ic)} mock-LRG halo centers, 1.6' beam)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LC / "figs/R1_closure.png", dpi=140)

    # correction template for T3: the data path's measured bias, to be
    # divided out of the T2f data columns (+50% of itself as a budget line)
    np.savez(LC / "R1_resample_correction.npz",
             theta_arcmin=THETA_ARCMIN,
             bias_pixwin=bias["pixel-window"],
             bias_point=bias["point-sample"],
             beam_bias=beam_bias, n_halos=len(ic),
             realization=REALIZATION)

    gate = bool(np.all(np.abs(bias["pixel-window"][THETA_ARCMIN >= 0.6]) < 0.02)
                and np.all(np.abs(beam_bias) < 0.01))
    v = {
        "phase": "R1", "pass": gate,
        "metrics": {
            "resample_bias_pct_point": [float(x * 100) for x in bias["point-sample"]],
            "resample_bias_pct_pixwin": [float(x * 100) for x in bias["pixel-window"]],
            "beam_shape_bias_pct": [float(x * 100) for x in beam_bias],
            "theta_arcmin": [float(x) for x in THETA_ARCMIN],
            "n_halos": int(len(ic)),
        },
        "figs": ["figs/R1_closure.png"],
        "notes": "R1a: native-vs-(0.5' degrade -> cubic re-upsample) stacked-CAP "
                 "closure on BIND's own beam-applied fiducial y at mock-LRG halo "
                 "centers; two degrade variants bracket the ACT pixel window. "
                 "GATE FAILED at theta<1.6' (-2 to -4.5% pixwin variant, i.e. the "
                 "data path slightly mimics feedback) -> remediation: the measured "
                 "curve is saved as R1_resample_correction.npz and DIVIDED OUT of "
                 "the T2f data columns in T3, with 50% of the correction kept as a "
                 "diagonal systematic. The +6% point at 4' is outside the fit "
                 "range (xb<=1.4 -> theta<=1.8') and flagged for investigation. "
                 "R1b: exact ilc_beam.txt b_ell vs the 1.6' Gaussian, applied in "
                 "ell-space to the stacked thumbnail (CAP linear in map) — "
                 "<=0.014%, retired. R1c (all-variant fine-res re-measure) runs "
                 "in the t2f_fineres disBatch job.",
        "next": "R2 HOD stacking",
    }
    with open(LC / "verdicts/R1.json", "w") as f:
        json.dump(v, f, indent=2)
    print("R1 pass:", gate)
    print("resample bias pixwin %:", np.round(100 * bias["pixel-window"], 2))
    print("beam-shape bias %:", np.round(100 * beam_bias, 3))


if __name__ == "__main__":
    main()
