"""fig11_fgas_saturation.pdf — sorted group-scale gas fraction across the 1P feedback
suite vs. the X-ray-standard band and the eROSITA-low tension value.

Single-panel bar chart: group-scale (M200c 1-2e13 Msun/h) background-subtracted f_gas
for each of the 57 one-parameter (1P) CAMELS feedback runs, sorted ascending, colored by
feedback family (AGN / SN-wind / other). Overlaid: the BIND fiducial value, the X-ray
"standard" band (0.06-0.10), and the eROSITA-low value (Popesso+24, 0.026).

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
    (keys: M_fof, m_gas_500c_bg, m_tot_500c_bg)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json

Original source: examples/paper_decomp_figs.ipynb (worktree analysis/wl-tsz-bridge)
cell 19, section 6.1 "Calibration to the Siegel et al. regime". Placeholder
figs/fig11_fgas_saturation.pdf is byte-identical (md5) to examples/figures_lightcone/fig_siegel.pdf.
Distinct provenance from the retracted "41% of cosmic" number discussed in the same
main.tex paragraph -- main.tex already flags that itself; not reproduced here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, ONE_COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SCI = Path("/mnt/home/mlee1/ceph/bind_science")
CACHE = SCI / "dashboard_cache"
ATLAS = SCI / "halo_atlas"
SNAP = 96


def _family(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "supernov", "sfr", "eqs")):
        return "SN / wind"
    return "other"


FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def fgas_group(run, mlo=1e13, mhi=3e13, snap=SNAP):
    """Median background-subtracted f_gas (R500c) in a group M200c bin."""
    f = ATLAS / f"{run}_snap{snap:03d}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    if "m_gas_500c_bg" not in d.files:
        return np.nan
    s = (d["M_fof"] >= mlo) & (d["M_fof"] < mhi) & (d["m_tot_500c_bg"] > 0)
    return np.nanmedian((d["m_gas_500c_bg"] / d["m_tot_500c_bg"])[s]) if s.sum() >= 5 else np.nan


def main():
    setup()

    design = [
        dict(run=r, name=name)
        for r, _i, name, _val, _fid in json.load(open(CACHE / "g1_design.json"))
    ]
    for d in design:
        d["fam"] = _family(d["name"])

    # group f_gas (M200c 1-2e13, i.e. the Siegel-regime bin) per run at z~0, sorted
    vals = []
    for d in design:
        f = fgas_group(d["run"], 1e13, 2e13)
        if np.isfinite(f):
            vals.append((d, f))
    vals.sort(key=lambda t: t[1])
    fg = np.array([v for _, v in vals])
    fam = [d["fam"] for d, _ in vals]
    fid_fg = fgas_group("fid", 1e13, 2e13)

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.bar(range(len(fg)), fg, color=[FAMC[f] for f in fam], edgecolor="k", lw=0.2, width=0.9)
    ax.axhline(fid_fg, color=COLORS["bind"], lw=1.4, label=f"BIND fiducial ({fid_fg:.3f})")
    ax.axhspan(0.06, 0.10, color=COLORS["secondary"], alpha=0.18, lw=0, label="X-ray standard")
    ax.axhline(0.026, color=COLORS["highlight"], ls="--", lw=1.3, label="eROSITA-low")
    ax.set_xlabel("1P feedback run (sorted)")
    ax.set_ylabel(r"$f_{\rm gas}$ ($M_{200c}\,1$--$2\times10^{13}\,M_\odot/h$)")
    ax.set_xlim(-1, len(fg))
    ax.legend(loc="upper left", fontsize=6.0)
    for fam_ in FAMC:
        ax.bar([], [], color=FAMC[fam_], label=fam_)

    fig.tight_layout()
    save(fig, "figs/fig11_fgas_saturation")

    print(f"BIND fiducial group f_gas: {fid_fg:.3f}")
    print(f"min group f_gas reached by a single knob: {fg.min():.3f}  ({vals[0][0]['name']})")


if __name__ == "__main__":
    main()
