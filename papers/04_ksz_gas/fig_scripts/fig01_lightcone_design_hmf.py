"""fig01_lightcone_design_hmf — the continuous, painted feedback lightcone (§1 DATA).

3 panels:
  (a) halo counts per lightcone snapshot vs redshift (20 slices, z=0.03-2.44),
      with the BGS (z=0.18) and ELG (z=1.16) confronted slices circled.
  (b) 256-node Sobol feedback design in PHYSICAL (log) units: wind energy
      (ASN1) vs wind speed (ASN2), with the fiducial-TNG point marked.
  (c) halo mass function at the BGS and ELG slices, with the BIND painting
      floor (1e13), the four BGS stacking-bin edges, the BGS science bin,
      the patch-reuse band (1e12-1e13), and the DESI BGS/LRG/ELG host-mass
      ranges (literature, hardcoded).

Source: examples/paper_ksz_desi_act.ipynb (branch analysis/ksz_project,
worktree .../wt/ksz-desi-act), cell 4 (heading "§1 DATA"), built by
examples/_build_ksz_paper_nb.py lines ~260-310. Constants (PARAMS/PMETA/
DESI/cosmology) ported from the notebook's cell 2 setup cell.

Data (read-only, cached on disk, no recomputation):
  - /mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet
    columns: snap, z (panel a); run, VariableWindVelFactor,
    WindEnergyIn1e51erg (panel b); snap, M200 (panel c)
  - src/bind/assets/SB35_param_minmax.csv (in-repo asset, PMETA fiducial
    values for panel b)
  - DESI host-mass ranges: hardcoded literature dict (Hahn+23 BGS,
    Zhou+23 LRG, Raichoor+23 ELG), matching the notebook's `DESI` dict.
"""
import sys
from importlib.resources import files as _ir_files
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")

BGS_BIN = 1
SNAP_BGS, Z_BGS = 85, 0.18
SNAP_ELG, Z_ELG = 46, 1.16

PX, PY = "VariableWindVelFactor", "WindEnergyIn1e51erg"
_PM = pd.read_csv(_ir_files("bind.assets") / "SB35_param_minmax.csv").set_index("ParamName")
PMETA = {p: dict(fid=float(_PM.loc[p, "FiducialVal"])) for p in [PX, PY] if p in _PM.index}

DESI = {
    "BGS": dict(logM=(13.0, 13.9), c=COLORS["highlight"]),
    "LRG": dict(logM=(13.2, 13.9), c="tab:purple"),
    "ELG": dict(logM=(11.7, 12.6), c=COLORS["secondary"]),
}

setup()

fig, ax = plt.subplots(1, 3, figsize=(TWO_COL[0], 2.9), constrained_layout=True)

# (a) snapshot <-> redshift coverage + halo counts, mark BGS & ELG slices
cov = (
    pd.read_parquet(PARQUET, columns=["snap", "z"])
    .groupby("snap")
    .agg(z=("z", "median"), n=("z", "size"))
    .reset_index()
)
ax[0].plot(cov.z, cov.n, "o-", color=COLORS["bind"], ms=4, lw=1)
for snp, zz, nm, col in [(SNAP_BGS, Z_BGS, "BGS", COLORS["highlight"]), (SNAP_ELG, Z_ELG, "ELG", COLORS["secondary"])]:
    nn = cov.loc[cov.snap == snp, "n"]
    if len(nn):
        ax[0].scatter(
            [zz], [nn.iloc[0]], s=75, facecolor="none", edgecolor=col, lw=1.5, zorder=5,
            label=f"{nm} snap ($z$={zz})",
        )
ax[0].set_yscale("log")
ax[0].set_xlabel(r"redshift $z$")
ax[0].set_ylabel(r"halos / snapshot ($M_{200}{\geq}10^{13}$)")
ax[0].legend(loc="upper right")
ax[0].text(0.05, 0.05, "20 slices\n$z$=0.03–2.44", transform=ax[0].transAxes)
panel_label(ax[0], "(a)", loc="lower right")

# (b) Sobol design in physical units (both knobs log-spaced; star = fiducial)
X = pd.read_parquet(PARQUET, columns=["run", PX, PY]).groupby("run").first()
ax[1].scatter(X[PX], X[PY], s=8, color="tab:purple", alpha=0.7)
ax[1].plot(
    PMETA[PX]["fid"], PMETA[PY]["fid"], "*", color="k", ms=12, mec="w", mew=0.6, zorder=5,
    label="fiducial TNG",
)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"ASN2 = wind speed [$V_{\rm wind}$ factor]")
ax[1].set_ylabel(r"ASN1 = wind energy [$10^{51}$ erg]")
ax[1].legend(loc="lower right")
panel_label(ax[1], "(b)")

# (c) halo mass function at both confronted slices + bins + floor + reuse band + DESI hosts
Md = pd.read_parquet(PARQUET, columns=["snap", "M200"])
edges = np.array([13.0, 13.4, 13.8, 14.2, 15.0])
for snp, sty, lab in [(SNAP_BGS, "-", f"BGS slice $z$={Z_BGS}"), (SNAP_ELG, "--", f"ELG slice $z$={Z_ELG}")]:
    lm = np.log10(Md.query("snap==@snp").M200.values)
    hc, be = np.histogram(lm, bins=np.linspace(11.8, 14.8, 34))
    ax[2].step(0.5 * (be[1:] + be[:-1]), np.maximum(hc, 0.5), where="mid", lw=1.2, ls=sty, color="0.35", label=lab)
ax[2].axvspan(12.0, 13.0, color="0.88", zorder=0)
ax[2].text(12.5, 1.5, "patch\nreuse\n(§6b)", fontsize=7, ha="center", va="bottom", color="0.45")
for e in edges:
    ax[2].axvline(e, color="navy", ls=":", lw=0.6)
ax[2].axvspan(edges[BGS_BIN], edges[BGS_BIN + 1], color=COLORS["bind"], alpha=0.20, label="BGS science bin")
ax[2].axvline(13.0, color="k", lw=1.2)
trans = ax[2].get_xaxis_transform()
ax[2].text(13.05, 0.55, "BIND floor", rotation=90, fontsize=7, va="top", transform=trans)
for nm, yl in [("BGS", 0.96), ("LRG", 0.89), ("ELG", 0.82)]:
    d = DESI[nm]
    ax[2].plot(d["logM"], [yl, yl], color=d["c"], lw=5, alpha=0.65, transform=trans, solid_capstyle="butt")
    ax[2].text(d["logM"][1] + 0.05, yl, nm, color=d["c"], fontsize=7, va="center", transform=trans)
ax[2].set_yscale("log")
ax[2].set_xlabel(r"$\log_{10} M_{200}\,[M_\odot/h]$")
ax[2].set_ylabel("halo count")
ax[2].set_xlim(11.8, 14.8)
ax[2].set_ylim(0.5, None)
ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), framealpha=0.85, facecolor="white", edgecolor="0.7")
panel_label(ax[2], "(c)")

save(fig, "figs/fig01_lightcone_design_hmf")
