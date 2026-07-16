"""P4 / D4 capstone: the SIGMA_V-INDEPENDENT gas-fraction confrontation.

The DESI x ACT papers report f~gas == f_gas/(Omega_b/Omega_m) vs aperture theta --
the gas fraction relative to cosmic, derived from the kSZ tau with their full
velocity model (so it -> ~1 at large theta as cosmic baryons are recovered). Because
it is physically normalized, f~gas bypasses the sigma_v calibration entirely -- the
cleanest comparison for "is BIND too gas-rich, and is its gas too centrally
concentrated?".

Two BIND views, both vs the DESI x ACT BGS points (Hadzhiyska+26, M*-split):
  * 3D ANCHORS (clean): BIND's spherical f_gas within r500/r200 from the per-halo
    catalogue (integrated.parquet), f~gas = f_gas/(Omega_b/Omega_m). No projection,
    no sigma_v. These are the trustworthy numbers.
  * PROJECTED SHAPE: BIND's cumulative bg-subtracted gas within projected aperture
    (mstar reduce, mgas_cum) -> shows whether BIND's gas RISES faster with aperture
    (more centrally concentrated). NB it overshoots cosmic at large theta because the
    2-halo term is not modelled (the data fit it out via A_k2h) -- shape only.

    python examples/ksz_fgas_profile.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

CEPH = Path(os.environ.get("CEPH", "/mnt/home/mlee1/ceph"))
OUT = Path(os.environ.get("OUTPUT_ROOT", CEPH / "bind_science")) / "ksz_confront"
PARQUET = CEPH / "bind_sb35/analysis_cache/integrated.parquet"
F_B = 0.0490 / 0.3089
Z, R500_FAC = 0.26, 0.659


def _imp(n):
    s = importlib.util.spec_from_file_location(n, Path(__file__).resolve().parent / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


CAP = _imp("ksz_cap_compare")

# DESI x ACT BGS f~gas(theta) (Hadzhiyska+26 Fig.): theta[arcmin], f~gas, ~err
DATA = {
    "M*>11.0":  ([1.0, 1.7, 2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5, 8.3, 9.1, 9.9],
                 [0.14, 0.23, 0.35, 0.55, 0.56, 0.56, 0.60, 0.76, 0.76, 0.83, 0.76, 0.88]),
    "M*>11.25": ([1.0, 1.7, 2.5, 3.4, 4.2, 5.0, 5.9, 6.7, 7.5, 8.3, 9.1, 9.9],
                 [0.15, 0.22, 0.37, 0.58, 0.70, 0.63, 0.82, 0.95, 1.06, 1.15, 1.17, 1.20]),
}
CI = {"M*>11.0": 0, "M*>11.25": 1}


def _theta(x_or_r_over_r200, logM200):
    r2 = CAP._r200phys(logM200, Z)
    return x_or_r_over_r200 * r2 / CAP._DA(Z) * CAP.ARCMIN


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.load(OUT / "bind_mstar_xprof_snap085.npz")
    x = c["x"]
    # 3D anchors from parquet at matched mass (snap 85, logM500 > 13.3)
    df = pd.read_parquet(PARQUET, columns=["run", "snap", "M_tot_500", "f_gas_200", "f_gas_500"])
    s = df[(df.snap == 85) & (np.log10(df.M_tot_500) > 13.3)]
    fg200 = (s.groupby("run").f_gas_200.median() / F_B)
    fg500 = (s.groupby("run").f_gas_500.median() / F_B)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    for lbl, col in [("M*>11.0", "tab:green"), ("M*>11.25", "tab:orange")]:
        ci = CI[lbl]
        ax.errorbar(DATA[lbl][0], DATA[lbl][1], yerr=0.4 * np.array(DATA[lbl][1]),
                    fmt="o", color=col, capsize=2, alpha=0.85, label=f"DESIxACT {lbl}")
        lM = np.nanmedian(c["logM200"][:, ci])
        # projected shape (inner range only; overshoots at large theta -> 2-halo)
        mg = c["mgas_cum"][:, ci, :]
        fproj = mg / (F_B * 10 ** c["logM200"][:, ci][:, None])
        med = np.nanmedian(fproj, axis=0)
        th = _theta(x, lM); inner = th < _theta(1.6, lM)          # <~1.6 r200
        ax.plot(th[inner], med[inner], "--", color=col, lw=1.6, alpha=0.8)
    # 3D anchors (clean) at theta(r500), theta(r200) for the massive sample
    lM = np.nanmedian(c["logM200"][:, 1])
    for fg, rr, mk in [(fg500, R500_FAC, "P"), (fg200, 1.0, "*")]:
        th = _theta(rr, lM)
        ax.errorbar([th], [fg.median()], yerr=[[fg.median() - fg.quantile(.16)], [fg.quantile(.84) - fg.median()]],
                    fmt=mk, ms=14, color="navy", capsize=4, zorder=5,
                    label=f"BIND 3D f~gas(<r{'500' if rr<1 else '200'}) = {fg.median():.2f}")
    ax.axhline(1, color="k", ls=":", lw=1, label=r"cosmic ($f_{\rm gas}=\Omega_b/\Omega_m$)")
    ax.set_xlabel(r"$\theta$ [arcmin]"); ax.set_ylabel(r"$\tilde f_{\rm gas}\,(\Omega_m/\Omega_b)$")
    ax.set_ylim(0, 1.6); ax.set_xlim(0, 10.5)
    ax.set_title("Gas fraction vs aperture: BIND vs DESI$\\times$ACT (sigma_v-free)\n"
                 "navy = BIND 3D (clean); dashed = BIND projected shape (inner; 2-halo not removed)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "ksz_fgas_profile.png", dpi=150, bbox_inches="tight")
    print(f"[fgas] wrote {OUT}/ksz_fgas_profile.png")
    th200, th500 = _theta(1.0, lM), _theta(R500_FAC, lM)
    di = np.interp(th200, DATA["M*>11.25"][0], DATA["M*>11.25"][1])
    print(f"[fgas] BIND 3D f~gas(<r200)={fg200.median():.2f} vs data@theta200({th200:.1f}')~{di:.2f}"
          f"  -> BIND {fg200.median()/di:.1f}x; data reaches ~1 only at ~4 r200 (gas pushed out).")


if __name__ == "__main__":
    main()
