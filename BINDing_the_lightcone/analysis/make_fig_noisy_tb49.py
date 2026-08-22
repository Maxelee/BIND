"""LSST-Y10 shape-noise validation twin, rebuilt on the corrected fiducial.

The released fig05n was computed on the retired CAMELS-conditioned fiducial
paint. This rebuilds it on twobound/run_0049 (correct TNG300 cosmology):
the 50 fiducial realizations get an independent per-(target,real,plane)
shape-noise draw through noisy_grid's own machinery (sigma_e=0.26, n_gal=27,
noise added BEFORE the 1' Gaussian filter), and are compared against the
hydro-pasted truth's noisy twin (nu05n_shards/truth.npz, 550 realizations),
whose per-realization draws supply the covariance.

Writes imgs/fig05n_field_validation_noisy.{png,pdf} (overwrites the retired
render in THIS paper folder only; nothing under BIND/imgs is touched).

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python make_fig_noisy_tb49.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
sys.path.insert(0, str(P1))
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import os
os.chdir(P1)
import noisy_grid as ng  # noqa: E402
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
HERE = Path(__file__).resolve().parent
# BIND_CAMPAIGN=n1000: the bind arm streams the campaign's 1000-real tb49 cube
# (campaign-tagged local cache so a 50-real result is never silently reused).
# The truth reference stays the 550-real sci50 noisy shard -- no n1000 noisy
# truth twin exists, and the reference covariance is set by the TRUTH arm's own
# per-realization scatter, which does not depend on the bind arm's N.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
IMGS = HERE.parent / ("imgs_1000" if CAMPAIGN == "n1000" else "imgs")
CACHE = HERE / ("noisy_tb49_stats_n1000.npz" if CAMPAIGN == "n1000"
                else "noisy_tb49_stats.npz")
ZI = 1
AREA_SCALE = 25.0 / 18000.0                    # covariance area factor

# the ONE addition: the corrected fiducial as a noisy-grid target
ng.TARGETS["fidtb49"] = (
    CEPH / "bind_n1000/twobound/run_0049/kappa_maps.npz" if CAMPAIGN == "n1000"
    else CEPH / "bind_science/runs/twobound/run_0049/kappa_maps.npz")

if CACHE.exists():
    res = dict(np.load(CACHE))
    print(f"loaded cache {CACHE.name}")
else:
    # noisy_grid.compute loads the WHOLE kappa cube (no streaming): the n1000
    # tb49 cube is ~21 GB uncompressed, so refuse under a small memory cgroup
    # (e.g. the 17.5 GB jupyter session) instead of getting OOM-killed hours in.
    _need = np.prod(np.array([1000, 5, 1024, 1024])) * 4 * 1.4  # cube + headroom
    try:
        import re as _re
        _cg = Path("/proc/self/cgroup").read_text()
        _m = _re.search(r"memory:(\S+)", _cg)
        _lim = int(Path(f"/sys/fs/cgroup/memory{_m.group(1)}/memory.limit_in_bytes")
                   .read_text()) if _m else None
    except Exception:
        _lim = None
    if CAMPAIGN == "n1000" and _lim is not None and _lim < _need:
        raise SystemExit(
            f"memory cgroup limit {_lim/1e9:.1f} GB < ~{_need/1e9:.0f} GB needed to hold "
            "the 1000-real kappa cube (noisy_grid loads it whole). Run this on a "
            "node/allocation with more memory, e.g. your workstation or an salloc.")
    t0 = time.time()
    res = ng.compute("fidtb49", per_real=True)
    np.savez_compressed(CACHE, **{k: v for k, v in res.items() if v is not None})
    print(f"computed fidtb49 noisy stats in {time.time() - t0:.0f}s")

TR = np.load(CEPH / "bind_sb35/nu05n_shards/truth.npz")
NU = np.asarray(TR["nu"], float)
ELL = np.asarray(TR["cl_ell"], float)

KEYS = [("cl_kappa", r"$\ell(\ell+1)C_\ell^{\kappa\kappa,{\rm noisy}}/2\pi$", ELL, True),
        ("pdf", r"PDF$(\nu)$", NU, False),
        ("peak_counts", r"$N_{\rm pk}(\nu)$", NU, False),
        ("minima_counts", r"$N_{\rm min}(\nu)$", NU, False),
        ("mf_v0", r"$V_0(\nu)$", NU, False),
        ("mf_v1", r"$V_1(\nu)$", NU, False),
        ("mf_v2", r"$V_2(\nu)$", NU, False)]

CB, CT = COLORS["bind"], COLORS["truth"]
fig, axs = plt.subplots(4, 4, figsize=(TWO_COL[0], 5.6),
                        gridspec_kw=dict(height_ratios=[2.0, 1.0, 2.0, 1.0],
                                         hspace=0.5, wspace=0.45))
for i, (k, slab, xg, isl) in enumerate(KEYS):
    r0, c0 = 2 * (i // 4), i % 4
    ax, rx = axs[r0, c0], axs[r0 + 1, c0]
    b = np.asarray(res[f"{k}_real"], float)[:, ZI]              # (50, nbin)
    t_all = np.asarray(TR[f"{k}_real"], float)[:, ZI]           # (550, nbin)
    if k == "cl_kappa":                     # beyond ~2 filter scales the 1'-
        keep = xg <= 4000.0                 # smoothed spectrum is pure cutoff
        xg, b, t_all = xg[keep], b[:, keep], t_all[:, keep]
    # Noise draws are independent between targets BY DESIGN (noisy_grid), so
    # seed pairing cannot cancel them: use ALL 550 truth realizations for the
    # reference mean, and carry the finite-N noise floor of the two means as
    # its own band -- at N=(50, 550) that floor is ~5x the LSST-scaled sigma,
    # so without it the residual would misread our own sampling noise as a
    # model offset.
    bm, tm = np.nanmean(b, 0), np.nanmean(t_all, 0)
    sig_real = np.nanstd(t_all, 0)
    sig = sig_real * np.sqrt(AREA_SCALE)                        # LSST-scaled truth cov
    floor = sig_real * np.sqrt(1.0 / len(b) + 1.0 / len(t_all))  # measurement floor
    pref = xg * (xg + 1) / (2 * np.pi) if k == "cl_kappa" else 1.0
    ax.plot(xg, pref * tm, color=CT, lw=1.0, ls="--")
    ax.plot(xg, pref * bm, color=CB, lw=1.0)
    if isl:
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif k in ("pdf", "peak_counts", "minima_counts", "mf_v1", "mf_v2"):
        ok = np.abs(tm) > 0
        if k != "mf_v2":
            ax.set_yscale("log")
    ax.set_ylabel(slab, fontsize=7)
    ax.tick_params(labelsize=6, labelbottom=False)
    panel_label(ax, f"({'abcdefg'[i]})")
    good = np.abs(tm) > (0.005 * np.nanmax(np.abs(tm)))
    dd = np.where(good, 100 * (bm - tm) / np.where(good, tm, 1), np.nan)
    bandpct = np.where(good, 100 * sig / np.abs(np.where(good, tm, 1)), np.nan)
    floorpct = np.where(good, 100 * floor / np.abs(np.where(good, tm, 1)), np.nan)
    rx.fill_between(xg, -floorpct, floorpct, color="0.75", alpha=0.45, lw=0)
    rx.fill_between(xg, -bandpct, bandpct, color=CB, alpha=0.25, lw=0)
    rx.plot(xg, dd, color=CB, lw=0.8, marker="." if not isl else None, ms=2.5)
    rx.axhline(0, color="0.7", lw=0.6)
    if isl:
        rx.set_xscale("log")
    lim = max(2.0, 1.3 * np.nanpercentile(np.abs(np.r_[dd, floorpct]), 98))
    rx.set_ylim(-lim, lim)
    rx.set_xlabel(r"$\ell$" if isl else r"$\nu$", fontsize=7)
    rx.set_ylabel(r"resid. [%]", fontsize=6)
    rx.tick_params(labelsize=6)

axl = axs[2, 3]
axs[3, 3].axis("off")
axl.axis("off")
from matplotlib.lines import Line2D  # noqa: E402
axl.legend(handles=[
    Line2D([], [], color=CB, lw=1.0, label="BIND (noisy)"),
    Line2D([], [], color=CT, lw=1.0, ls="--", label="hydro-pasted (noisy)"),
    Line2D([], [], color=CB, lw=6, alpha=0.25,
           label=r"truth cov. $\pm1\sigma$, LSST-Y10 area"),
    Line2D([], [], color="0.75", lw=6, alpha=0.45,
           label=r"finite-$N$ noise floor of the two means")],
    loc="center", fontsize=6.5, frameon=False)
save(fig, str(IMGS / "fig05n_field_validation_noisy"))
plt.close(fig)

# caption numbers: fraction of bins inside the band, per statistic
print("wrote fig05n_field_validation_noisy (fidtb49)")
for k, *_ in KEYS:
    b = np.asarray(res[f"{k}_real"], float)[:, ZI]
    t_all = np.asarray(TR[f"{k}_real"], float)[:, ZI]
    bm, tm = np.nanmean(b, 0), np.nanmean(t_all, 0)
    sig_real = np.nanstd(t_all, 0)
    tot = np.sqrt((sig_real ** 2) * AREA_SCALE
                  + (sig_real ** 2) * (1.0 / len(b) + 1.0 / len(t_all)))
    good = np.abs(tm) > (0.005 * np.nanmax(np.abs(tm)))
    z = np.abs(bm - tm)[good] / tot[good]
    print(f"  {k:14s} median |resid|/(LSST (+) floor) = {np.nanmedian(z):5.2f}  "
          f"frac<1: {np.mean(z < 1):.2f}  frac<3: {np.mean(z < 3):.2f}")
