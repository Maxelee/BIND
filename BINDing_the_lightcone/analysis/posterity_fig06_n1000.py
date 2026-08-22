"""Posterity: the fig-6 spectra validation at N = 1000 realizations.

Same six-panel layout as the paper's Fig. 6 (kk, yy, tautau on top; ky, ktau,
ytau below), built from the bind_n1000 campaign: the BIND arm is
twobound/run_0018 -- an independent paint replica at the fiducial parameters
with the CORRECT TNG300 conditioning, traced to 1000 realizations -- and the
truth arm is the hydro-pasted control, both on the shared 1992+7r seed ladder.
Matches the current paper render: no error contours in the residual panels and
no CIC/pixel annotation (the dotted 0.8 ell_Nyq trust line stays).

Data:
  * kk / ky / ktau come from the per-run Cl caches (means over 1000 reals,
    xpkfix cross-norm already correct in the n1000 campaign).
  * per-plane yy / tautau / ytau are NOT cached per z_s, so they are streamed
    from the y_maps/tau_maps cubes (~75 GB, chunked) and memoized in
    n1000_gas_spectra.npz -- delete that file to force a re-stream.

Writes figs_preview/fig06_spectra_validation_n1000.{png,pdf}.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python posterity_fig06_n1000.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npf

CEPH = Path("/mnt/home/mlee1/ceph")
N1K = CEPH / "bind_n1000"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs_preview"
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation, LogLocator  # noqa: E402

import contextlib  # noqa: E402
import io  # noqa: E402

import Pk_library as PKL  # noqa: E402

from bind.inference.stats import power_spectrum  # noqa: E402

FOV = np.deg2rad(5.0)
_XNORM = None


def spectra_pair(y, t):
    """(ell, C^yy, C^tt, C^yt) from ONE XPk_plane call (2 FFTs, not 4).

    Pylians XPk_plane and Pk_plane use different power conventions; the per-bin
    ratio is a map-independent constant for a fixed grid (the same xpkfix used
    in bind.inference.stats.power_spectrum), so it is measured once and reused.
    """
    global _XNORM
    a = (y - y.mean()).astype(np.float32)
    b = (t - t.mean()).astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        x = PKL.XPk_plane(a, b, FOV, MAS1="None", MAS2="None", threads=1)
        if _XNORM is None:
            pa = PKL.Pk_plane(a, FOV, MAS="None", threads=1, verbose=False)
            _XNORM = np.asarray(pa.Pk) / np.asarray(x.Pk)[:, 0]
    return (np.asarray(x.k), np.asarray(x.Pk)[:, 0] * _XNORM,
            np.asarray(x.Pk)[:, 1] * _XNORM, np.asarray(x.XPk) * _XNORM)


ZI = 1
ZS = [0.5, 1.0, 1.5, 2.0, 2.44]
ELL_LO, ELL_MAX_PLOT, ELL_TRUST = 100.0, 36864.0, 3.0e4
cB, cH = COLORS["bind"], COLORS["truth"]
ZCOLS = plt.cm.plasma(np.linspace(0.02, 0.82, 5))
NREAL, NPLANE, CHUNK = 1000, 5, 25

ARM_DIR = {"bind": "twobound/run_0018", "truth": "truth/run_0000"}
CK = {a: np.load(N1K / ARM_DIR[a] / "Cl_kappa.npz") for a in ("bind", "truth")}
KY = {a: np.load(N1K / ARM_DIR[a] / "Cl_kappa_y.npz") for a in ("bind", "truth")}
KT = {a: np.load(N1K / ARM_DIR[a] / "Cl_tau.npz") for a in ("bind", "truth")}
ELL = np.asarray(CK["bind"]["ell"], float)
NELL = len(ELL)
f_ell = ELL * (ELL + 1) / (2 * np.pi)


def stream_gas_spectra(arm):
    """Per-plane mean C^yy, C^tautau, C^ytau over all 1000 reals (chunked)."""
    rd = N1K / ARM_DIR[arm]
    zy = zipfile.ZipFile(rd / "y_maps.npz")
    zt = zipfile.ZipFile(rd / "tau_maps.npz")
    fy, ft = zy.open("y.npy"), zt.open("tau.npy")
    shapes = []
    for fh in (fy, ft):
        version = npf.read_magic(fh)
        shape, _, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0)
                           else npf.read_array_header_2_0(fh))
        assert shape[:2] == (NREAL, NPLANE) and dtype == np.float32, (shape, dtype)
        shapes.append(shape)
    ny, nx = shapes[0][2:]
    per_map = NPLANE * ny * nx
    acc = {k: np.zeros((NPLANE, NELL)) for k in ("yy", "tt", "yt")}
    for r0 in range(0, NREAL, CHUNK):
        n = min(CHUNK, NREAL - r0)
        Y = np.frombuffer(fy.read(n * per_map * 4), np.float32).reshape(n, NPLANE, ny, nx)
        T = np.frombuffer(ft.read(n * per_map * 4), np.float32).reshape(n, NPLANE, ny, nx)
        for r in range(n):
            for zi in range(NPLANE):
                lg, cyy, ctt, cyt = spectra_pair(Y[r, zi], T[r, zi])
                if r0 == 0 and r == 0 and zi <= 1:
                    assert np.allclose(lg, ELL), "ell grid mismatch vs Cl caches"
                    for ref, got, nm in ((power_spectrum(Y[r, zi])[1], cyy, "yy"),
                                         (power_spectrum(T[r, zi])[1], ctt, "tt")):
                        assert np.allclose(ref, got, rtol=1e-3), \
                            f"xpk norm reuse broke {nm} (real {r}, plane {zi})"
                acc["yy"][zi] += cyy
                acc["tt"][zi] += ctt
                acc["yt"][zi] += cyt
        del Y, T
        print(f"  {arm}: {r0 + n}/{NREAL} reals", flush=True)
    for fh in (fy, ft, zy, zt):
        fh.close()
    return {k: v / NREAL for k, v in acc.items()}


# per-arm caches: truth reuses the original stream, the tb18 bind arm its own
GC = {"bind": HERE / "n1000_gas_spectra_tb18.npz",
      "truth": HERE / "n1000_gas_spectra.npz"}
GAS = {}
for arm in ("bind", "truth"):
    if GC[arm].exists():
        _g = np.load(GC[arm])
        GAS[arm] = {k: _g[f"{k}_{arm}"] for k in ("yy", "tt", "yt")}
        print(f"loaded {GC[arm].name} [{arm}]")
    else:
        print(f"streaming {arm} ({ARM_DIR[arm]}) y/tau cubes ...", flush=True)
        GAS[arm] = stream_gas_spectra(arm)
        np.savez_compressed(GC[arm], **{f"{k}_{arm}": GAS[arm][k] for k in GAS[arm]})
        print(f"wrote {GC[arm].name}")

# self-check: the streamed LAST cumulative column against the cached total yy
for arm in ("bind", "truth"):
    m = (ELL >= 300) & (ELL <= 5000)
    r = GAS[arm]["yy"][-1][m] / KY[arm]["cl_yy"][m]
    print(f"{arm}: streamed plane-4 C^yy vs cached total-column: "
          f"median |ratio-1| = {100 * np.nanmedian(np.abs(r - 1)):.3f}%")


def resid(b, t):
    return 100 * (b / np.where(t > 0, t, np.nan) - 1)


KK5b = np.array([CK["bind"]["cl"][zi, zi] for zi in range(5)])
KK5t = np.array([CK["truth"]["cl"][zi, zi] for zi in range(5)])
panels = [
    dict(yb=f_ell * KK5b, yt=f_ell * KK5t[ZI], res=resid(KK5b[ZI], KK5t[ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\kappa}/2\pi$", tag="(a)", rlim=25),
    dict(yb=f_ell * GAS["bind"]["yy"], yt=f_ell * GAS["truth"]["yy"][ZI],
         res=resid(GAS["bind"]["yy"][ZI], GAS["truth"]["yy"][ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{yy}/2\pi$", tag="(b)", rlim=100),
    dict(yb=f_ell * GAS["bind"]["tt"], yt=f_ell * GAS["truth"]["tt"][ZI],
         res=resid(GAS["bind"]["tt"][ZI], GAS["truth"]["tt"][ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{\tau\tau}/2\pi$", tag="(c)", rlim=100),
    dict(yb=f_ell * KY["bind"]["cl_ky"], yt=f_ell * KY["truth"]["cl_ky"][ZI],
         res=resid(KY["bind"]["cl_ky"][ZI], KY["truth"]["cl_ky"][ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa y}/2\pi$", tag="(d)", rlim=50),
    dict(yb=f_ell * KT["bind"]["cl_kt"], yt=f_ell * KT["truth"]["cl_kt"][ZI],
         res=resid(KT["bind"]["cl_kt"][ZI], KT["truth"]["cl_kt"][ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{\kappa\tau}/2\pi$", tag="(e)", rlim=50),
    dict(yb=f_ell * GAS["bind"]["yt"], yt=f_ell * GAS["truth"]["yt"][ZI],
         res=resid(GAS["bind"]["yt"][ZI], GAS["truth"]["yt"][ZI]),
         yl=r"$\ell(\ell{+}1)C_\ell^{y\tau}/2\pi$", tag="(f)", rlim=100),
]

fig = plt.figure(figsize=(TWO_COL[0], 6.9))
gs = fig.add_gridspec(6, 3, height_ratios=[2.2, 1, 0.55, 2.2, 1, 0.9],
                      hspace=0.18, wspace=0.42)
for j, p in enumerate(panels):
    r0 = (j // 3) * 3
    a = fig.add_subplot(gs[r0, j % 3])
    ar = fig.add_subplot(gs[r0 + 1, j % 3], sharex=a)
    for zi in range(5):
        a.plot(ELL, p["yb"][zi], color=ZCOLS[zi], lw=1.0)
    a.plot(ELL, p["yt"], color=cH, ls="--", lw=0.9)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylabel(p["yl"], fontsize=7)
    a.set_xlim(ELL_LO, ELL_MAX_PLOT)
    a.axvline(ELL_TRUST, color="0.55", lw=0.5, ls=":", zorder=1)
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
    a.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.4)))
    plt.setp(a.get_xticklabels(), visible=False)
    a.tick_params(labelsize=6)

    ar.axhline(0, color=COLORS["dmo"], lw=0.6)
    ar.plot(ELL, p["res"], color=cB, lw=0.9)
    ar.set_ylabel("resid. [%]" if j % 3 == 0 else "", fontsize=6)
    ar.set_ylim(-50, 50)   # shared across all six residual panels
    ar.set_xscale("log")
    ar.set_xlabel(r"$\ell$", fontsize=7)
    ar.tick_params(labelsize=6)
    panel_label(a, p["tag"], loc="upper right")

alg = fig.add_subplot(gs[5, :])
alg.axis("off")
_leg = [plt.Line2D([], [], color=ZCOLS[zi], lw=1.0, label=rf"$z_s={ZS[zi]:.2f}$")
        for zi in range(5)]
_leg += [plt.Line2D([], [], color=cH, lw=0.9, ls="--", label="hydro-pasted")]
alg.legend(handles=_leg, loc="center", ncol=4, fontsize=5.4, handlelength=1.6,
           labelspacing=0.5, columnspacing=1.4)
save(fig, str(OUT / "fig06_spectra_validation_n1000"))
plt.close(fig)

mb = (ELL >= 300) & (ELL <= 5000)
m310 = (ELL >= 300) & (ELL <= 1000)
print("\nN=1000 median |resid| over ell=300-5000 (z_s=1):")
for p, lab in zip(panels, ("kk", "yy", "tt", "ky", "kt", "yt")):
    print(f"  {lab:3s} {np.nanmedian(np.abs(p['res'][mb])):6.2f}%")
print("SIGNED BIND/truth mean over ell=300-1000 / 300-5000:")
for p, lab in zip(panels, ("kk", "yy", "tt", "ky", "kt", "yt")):
    r = 1 + p["res"] / 100
    print(f"  {lab:3s} {np.nanmean(r[m310]):.4f} / {np.nanmean(r[mb]):.4f}")
print("\nDONE fig06 n1000")
