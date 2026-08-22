"""Fig-5(a,b) with the texture-template debias applied.

The replica trio (twobound 0018/0049/0053: identical parameters, identical
ray-tracing seeds, independent paint draws) gives the map-level texture
template directly:

    Cl_auto(l)  = <|kappa_a|^2>                (any single draw; what fig 5 shows)
    Cl_cross(l) = <Re kappa_a kappa_b^*>       (the coherent, draw-independent field)
    f_tex(l)    = 1 - Cl_cross/Cl_auto         (single-draw texture fraction)

The debiased spectrum is Cl_debias = (1 - f_tex) * Cl_auto = Cl_cross, i.e.
the coherent field's power, and S_debias = (1 - f_tex) * S. Because f_tex is a
ratio measured on one ell grid, it applies to the shipped fig-5 legs
(fig05_numbers.npz, same 724-bin grid) without any estimator mismatch.

Caveat printed with the figure: the truth is ONE hydro realization, whose own
halo-to-halo scatter is coherent structure in its maps; a perfectly calibrated
sampler would carry texture matching that scatter, so subtracting ALL draw
variance legitimately under-shoots at the smallest scales. The debias is a
two-point diagnostic, not a replacement for the maps.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python make_fig_texture_debias.py
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
HERE = Path(__file__).resolve().parent
# BIND_CAMPAIGN=n1000 swaps the run roots, the fig05 numbers file and the output
# dir over to the 1000-realization campaign.  The texture TEMPLATE itself is
# campaign-independent: it is a property of the paint (kappa is untouched by the
# campaign's diffuse-gas correction, which only moves mass between channels), and
# realizations 1..50 are the same seeds in both trees.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
_N1K = CEPH / "bind_n1000"
IMGS = HERE.parent / ("imgs_1000" if CAMPAIGN == "n1000" else "imgs")
# The template VALUE is campaign-independent (shared seeds), but its PRECISION
# is not: the n1000 rebuild shrinks the template's own noise ~sqrt(20).  Tag
# the cache per campaign so an n1000 render never silently reuses the N=50
# template (and vice versa).
CACHE = HERE / ("texture_template_maps_n1000.npz" if CAMPAIGN == "n1000"
                else "texture_template_maps.npz")
RUNS = (18, 49, 53)
NPIX, FOV_DEG = 1024, 5.0
ZI = 1
ELL_MAX = 3.0e4

if CACHE.exists():
    z = np.load(CACHE)
    ELLB, F_TEX, N_PLANES = z["ellb"], z["f_tex"], int(z["n_planes"])
    print(f"loaded {CACHE.name}")
else:
    # STREAMED build (2026-08-21): the old version loaded three full cubes
    # (3 x 20 GB at n_real=1000) -- an OOM under any per-job memory cap.  This
    # walks the three compressed npz members in lockstep, one realization
    # (3 x 5 x 1024^2 float32 = 63 MB) at a time, accumulating the same AUTO/
    # CROSS sums; a partial-progress checkpoint makes it resumable.
    import zipfile
    import numpy.lib.format as _fmt

    def _stream(path):
        zf = zipfile.ZipFile(path)
        fh = zf.open("kappa.npy")
        version = _fmt.read_magic(fh)
        shape, _, dtype = (_fmt.read_array_header_1_0(fh) if version == (1, 0)
                          else _fmt.read_array_header_2_0(fh))
        n_real, n_plane, ny, nx = shape
        block = n_plane * ny * nx * dtype.itemsize

        def blocks():
            for _ in range(n_real):
                buf = bytearray()
                while len(buf) < block:
                    chunk = fh.read(block - len(buf))
                    assert chunk, f"truncated kappa.npy in {path}"
                    buf += chunk
                yield np.frombuffer(bytes(buf), dtype=dtype).reshape(n_plane, ny, nx)
        return (n_real, n_plane), blocks()

    t0 = time.time()
    CKPT = CACHE.with_suffix(".partial.npz")
    streams, shapes = {}, {}
    for r in RUNS:
        _rd = (_N1K / f"twobound/run_{r:04d}" if CAMPAIGN == "n1000"
               else CEPH / f"bind_science/runs/twobound/run_{r:04d}")
        shapes[r], streams[r] = _stream(_rd / "kappa_maps.npz")
    assert len(set(shapes.values())) == 1, f"replica cube shapes differ: {shapes}"
    n_real, N_PLANES = shapes[RUNS[0]]

    lx = np.fft.fftfreq(NPIX) * NPIX * 360.0 / FOV_DEG    # ell grid of the box
    ll = np.sqrt(lx[:, None] ** 2 + lx[None, :NPIX // 2 + 1] ** 2)
    EDGES = np.exp(np.linspace(np.log(80), np.log(4.0e4), 41))
    ELLB = np.sqrt(EDGES[1:] * EDGES[:-1])
    IDX = np.digitize(ll.ravel(), EDGES) - 1
    NB = len(ELLB)
    _IN = (IDX >= 0) & (IDX < NB)
    _IDXc = IDX[_IN]
    _CNT = np.bincount(_IDXc, minlength=NB).astype(float)
    _CNT[_CNT == 0] = np.nan

    def bin_p(p2d):
        # bincount-accumulated bin means: identical to the per-bin masked mean
        # (same IDX, same pixels) but ~40x faster, which matters at n_real=1000
        return np.bincount(_IDXc, weights=p2d.ravel()[_IN], minlength=NB) / _CNT

    AUTO = np.zeros((N_PLANES, NB))
    CROSS = np.zeros((N_PLANES, NB))
    i0 = 0
    if CKPT.exists():
        try:
            with np.load(CKPT) as c:
                if int(c["n_real"]) == n_real and int(c["n_planes"]) == N_PLANES:
                    AUTO, CROSS, i0 = c["auto_sum"].copy(), c["cross_sum"].copy(), int(c["i_done"])
                    print(f"resuming template build at realization {i0}/{n_real}")
        except Exception as e:
            print(f"{CKPT.name}: unreadable ({e!r}) -- rebuilding from scratch")
    for i in range(n_real):
        maps = {r: next(streams[r]) for r in RUNS}     # streams always advance
        if i < i0:
            continue
        if i == 0:
            # sanity: paint replicas must share the ray-tracing geometry
            c0 = np.corrcoef(maps[18][ZI].ravel(), maps[49][ZI].ravel())[0, 1]
            print(f"pixel corr(run18, run49) real 0, z_s=1: {c0:.4f} (seed pairing check)")
            assert c0 > 0.98, "replicas are not seed-paired -- cross-spectra invalid"
        for zi in range(N_PLANES):
            F = {r: np.fft.rfft2(maps[r][zi].astype(np.float64)) for r in RUNS}
            for r in RUNS:
                AUTO[zi] += bin_p((F[r] * np.conj(F[r])).real)
            for a in RUNS:
                for b in RUNS:
                    if a < b:
                        CROSS[zi] += bin_p((F[a] * np.conj(F[b])).real)
        if (i + 1) % 50 == 0 or i + 1 == n_real:
            # atomic checkpoint: np.savez truncates its target on open
            _tmp = CKPT.with_suffix(".tmp.npz")
            np.savez_compressed(_tmp, auto_sum=AUTO, cross_sum=CROSS,
                                i_done=i + 1, n_real=n_real, n_planes=N_PLANES)
            os.replace(_tmp, CKPT)
            print(f"  real {i + 1}/{n_real} ({time.time()-t0:.0f}s)", flush=True)
    AUTO /= n_real * len(RUNS)
    CROSS /= n_real * 3
    F_TEX = 1.0 - CROSS / AUTO
    np.savez_compressed(CACHE, ellb=ELLB, f_tex=F_TEX, auto=AUTO, cross=CROSS,
                        n_planes=N_PLANES, n_real=n_real)
    CKPT.unlink(missing_ok=True)
    print(f"template built in {time.time()-t0:.0f}s -> {CACHE.name}")

# ── apply to the shipped fig-5 legs ─────────────────────────────────────────
d5 = np.load(_N1K / "analysis/fig05_numbers_n1000.npz" if CAMPAIGN == "n1000"
             else CEPH / "referee_work/fidswap/fig05_numbers.npz")
ELL, S_b, S_t, band = (np.asarray(d5[k], float) for k in ("ell", "S_new", "S_truth", "band_kk"))
f_z1 = np.interp(ELL, ELLB, F_TEX[ZI])
S_deb = (1.0 - f_z1) * S_b
if CAMPAIGN == "n1000":
    cl49 = np.load(_N1K / "twobound/run_0049/Cl_kappa.npz")["cl"][ZI, ZI]
    # both traces are the full 1000 on one seed ladder -> the plain DMO mean IS
    # the paired denominator (no 50-real prefix cache needed)
    cl_dmo = np.load(_N1K / "dmo/run_0000/Cl_kappa.npz")["cl"][ZI, ZI]
else:
    cl49 = np.load(CEPH / "bind_science/runs/twobound/run_0049/Cl_kappa.npz")["cl"][ZI, ZI]
    cl_dmo = np.load(CEPH / "bind_science/runs/dmo/run_0000/Cl_kappa_paired.npz")["cl_real"].mean(0)[ZI]
cl_tru = S_t * cl_dmo

m = (ELL >= 100) & (ELL <= ELL_MAX)
CB, CT, CG = COLORS["bind"], COLORS["truth"], COLORS["secondary"]
fig, axs = plt.subplots(2, 2, figsize=(TWO_COL[0] * 0.72, 3.6), sharex="col",
                        gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.12,
                                         wspace=0.34))
pref = ELL * (ELL + 1) / (2 * np.pi)
ax, rx = axs[0, 0], axs[1, 0]
ax.plot(ELL[m], (pref * cl_tru)[m], color=CT, ls="--", lw=1.0, label="hydro-pasted")
ax.plot(ELL[m], (pref * cl49)[m], color=CB, lw=1.0, label="BIND (single draw)")
ax.plot(ELL[m], (pref * (1 - f_z1) * cl49)[m], color=CG, lw=1.0, ls="-.",
        label="BIND, texture-debiased")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\kappa\kappa}/2\pi$", fontsize=8)
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.tick_params(labelsize=7)
panel_label(ax, "(a)")
for curve, c, ls in ((S_b / S_t, CB, "-"), (S_deb / S_t, CG, "-.")):
    rx.plot(ELL[m], 100 * (curve - 1)[m], color=c, lw=1.0, ls=ls)
rx.fill_between(ELL[m], -band[m], band[m], color=CB, alpha=0.12, lw=0,
                label="LSST-Y10")
rx.axhline(0, color="0.7", lw=0.6)
rx.set_xscale("log")
rx.set_ylim(-4, 4)
rx.set_xlabel(r"$\ell$", fontsize=8)
rx.set_ylabel("resid. [%]", fontsize=7)
rx.legend(fontsize=6, frameon=False, loc="lower left")
rx.tick_params(labelsize=7)

ax, rx = axs[0, 1], axs[1, 1]
ax.plot(ELL[m], S_t[m], color=CT, ls="--", lw=1.0)
ax.plot(ELL[m], S_b[m], color=CB, lw=1.0)
ax.plot(ELL[m], S_deb[m], color=CG, lw=1.0, ls="-.")
ax.axhline(1, color="0.8", lw=0.6)
ax.set_xscale("log")
ax.set_ylabel(r"$S(\ell) = C_\ell^{\rm bind}/C_\ell^{\rm dmo}$", fontsize=8)
ax.tick_params(labelsize=7)
panel_label(ax, "(b)")
for curve, c, ls in ((S_b / S_t, CB, "-"), (S_deb / S_t, CG, "-.")):
    rx.plot(ELL[m], 100 * (curve - 1)[m], color=c, lw=1.0, ls=ls)
rx.fill_between(ELL[m], -band[m], band[m], color=CB, alpha=0.12, lw=0)
rx.axhline(0, color="0.7", lw=0.6)
rx.set_xscale("log")
rx.set_ylim(-4, 4)
rx.set_xlabel(r"$\ell$", fontsize=8)
rx.set_ylabel("resid. [%]", fontsize=7)
rx.tick_params(labelsize=7)
save(fig, str(IMGS / "figA3_texture_debias"))
plt.close(fig)

# the numbers
it = int(np.nanargmin(S_t))
print(f"\nz_s=1 trough (ell = {ELL[it]:.0f}):")
print(f"  S truth      {S_t[it]:.4f}")
print(f"  S raw        {S_b[it]:.4f}   ({100*(S_b[it]/S_t[it]-1):+.2f}%)")
print(f"  S debiased   {S_deb[it]:.4f}   ({100*(S_deb[it]/S_t[it]-1):+.2f}%)")
for lo, hi in ((300, 1000), (1000, 5000), (5000, 15000), (15000, 30000)):
    mm = (ELL >= lo) & (ELL < hi)
    print(f"  ell {lo:>5d}-{hi:<5d}: raw {100*np.nanmedian(np.abs(S_b/S_t-1)[mm]):5.2f}%"
          f"  debiased {100*np.nanmedian(np.abs(S_deb/S_t-1)[mm]):5.2f}%"
          f"  f_tex {100*np.nanmedian(f_z1[mm]):5.2f}%")
print("texture fraction f_tex(z_s=1) at ell 2e3/6e3/1.3e4/3e4: "
      + "  ".join(f"{100*np.interp(l, ELLB, F_TEX[ZI]):.2f}%" for l in (2e3, 6e3, 1.3e4, 3e4)))
