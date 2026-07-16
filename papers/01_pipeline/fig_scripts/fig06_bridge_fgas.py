#!/usr/bin/env python
"""fig06_bridge_fgas -- the per-halo gas fraction sets the field-level WL suppression
(the van Daalen+2020-style bridge).

Panel (a): S(ell) = C_ell^bind / C_ell^DMO for the fiducial run plus the
lowest- and highest-group-f_gas 1P feedback runs -- shows the CIC-aliasing
upturn at very high ell (a projection artefact, not a physical effect; see
[[lightcone-kappa-upturn-aliasing]]).
Panel (b): S(ell~1500) vs. background-subtracted group f_gas (M200c in
[1,3]x10^13 Msun/h, R500c) across the 57-run 1P feedback suite, coloured by
feedback family (AGN / SN-wind / other), r=0.91.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz (ell, per-run "{run}_clk")
  - /mnt/home/mlee1/ceph/bind_science/runs/{bind,dmo}/run_0000/Cl_kappa.npz (fiducial absolute C_ell)
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz
      (keys: M_fof, m_gas_500c_bg, m_tot_500c_bg)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
  - src/bind/assets/SB35_param_minmax.csv (param Description, for family tagging)

Source: examples/bind_bridge.py::fig_fgas_sofl() (worktree analysis/wl-tsz-bridge).
Placeholder figs/fig06_bridge_fgas.png is byte-identical (md5) to the on-disk
figs/bind_bridge_fgas_sofl.png produced by that function.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path("/mnt/home/mlee1/BIND")
SCI = Path("/mnt/home/mlee1/ceph/bind_science")
CACHE = SCI / "dashboard_cache"
ATLAS = SCI / "halo_atlas"
RUNS = SCI / "runs"
ACSV = REPO / "src/bind/assets/SB35_param_minmax.csv"
ZIDX = 1  # z_s = 1.0 reference plane


def _design():
    dj = json.load(open(CACHE / "g1_design.json"))
    meta = {r["ParamName"]: r for r in csv.DictReader(open(ACSV))}
    out = []
    for run, _i, name, val, fid in dj:
        m = meta.get(name, {})
        out.append((run, name, float(val), float(fid), int(m.get("LogFlag", 0)),
                    m.get("Description", name)))
    return out


def _family(name, desc):
    """Feedback family for colouring: AGN / SN-wind / other (matches bind_bridge.py)."""
    s = (name + " " + desc).lower()
    if any(k in s for k in ("blackhole", "quasar", "radio", "agn", "aagn")):
        return "AGN"
    if any(k in s for k in ("wind", "snii", "snia", "imf", "asn", "supernov")):
        return "SN / wind"
    return "other"


FAMC = {"AGN": COLORS["highlight"], "SN / wind": COLORS["bind"], "other": COLORS["dmo"]}


def _fgas_run(run, mlo=1e13, mhi=3e13):
    """bg-subtracted f_gas in the group M200c bin (1-3e13) at z~0."""
    f = ATLAS / f"{run}_snap096.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    if "m_gas_500c_bg" not in d.files:
        return np.nan
    s = (d["M_fof"] >= mlo) & (d["M_fof"] < mhi) & (d["m_tot_500c_bg"] > 0)
    return np.median((d["m_gas_500c_bg"] / d["m_tot_500c_bg"])[s]) if s.sum() >= 5 else np.nan


def main():
    setup()

    gs = np.load(CACHE / "g1_stats.npz")
    ell = gs["ell"]
    clf = np.load(RUNS / "bind/run_0000/Cl_kappa.npz")["cl"][ZIDX, ZIDX]
    cld = np.load(RUNS / "dmo/run_0000/Cl_kappa.npz")["cl"][ZIDX, ZIDX]
    S_fid = clf / cld
    band = (ell >= 1000) & (ell <= 2000)
    design = _design()

    fS, ff, fam = [], [], []
    for run, name, val, fid, lf, desc in design:
        if f"{run}_clk" not in gs.files:
            continue
        fg = _fgas_run(run)
        if not np.isfinite(fg):
            continue
        S_run = (1.0 + gs[f"{run}_clk"]) * S_fid
        fS.append(S_run[band].mean())
        ff.append(fg)
        fam.append(_family(name, desc))
    fS, ff, fam = np.array(fS), np.array(ff), np.array(fam)
    fg_fid = _fgas_run("fid")
    print(f"fgas->S(l): {len(fS)} runs; fid group f_gas={fg_fid:.3f}, "
          f"S(l~1500)={S_fid[band].mean():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    # (a) S(ell) curves: fiducial + lowest/highest f_gas runs
    ax[0].plot(ell, S_fid, color=COLORS["bind"], lw=1.8, zorder=5,
              label=f"fiducial ({fg_fid:.3f})")
    order = np.argsort(ff)
    okruns = [d[0] for d in design if f"{d[0]}_clk" in gs.files and np.isfinite(_fgas_run(d[0]))]
    for j, lab, col in ((order[0], "lowest", COLORS["highlight"]),
                        (order[-1], "highest", COLORS["secondary"])):
        run = okruns[j]
        ax[0].plot(ell, (1 + gs[f"{run}_clk"]) * S_fid, color=col, lw=1.2,
                   label=f"{lab} ({ff[j]:.3f})")
    ax[0].axhline(1, c="0.7", lw=0.6)
    ax[0].axvspan(1000, 2000, color=COLORS["dmo"], alpha=0.15, lw=0)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$S(\ell)=C_\ell^{\rm bind}/C_\ell^{\rm DMO}$")
    ax[0].legend(fontsize=6.2, title="group $f_{\\rm gas}$", title_fontsize=6.2)
    panel_label(ax[0], "(a)")

    # (b) S(ell~1500) vs group f_gas, van Daalen-style bridge
    r = np.corrcoef(ff, fS)[0, 1]
    z = np.polyfit(ff, fS, 1)
    xg = np.linspace(ff.min(), ff.max(), 40)
    ax[1].plot(xg, np.polyval(z, xg), color=COLORS["truth"], ls="--", lw=1.2, zorder=2,
              label=f"fit, r={r:.2f}")
    for fk, col in FAMC.items():
        sel = fam == fk
        if sel.any():
            ax[1].scatter(ff[sel], fS[sel], c=col, s=22, edgecolor="k", lw=0.3,
                          zorder=3, label=f"{fk} ({sel.sum()})")
    ax[1].scatter([fg_fid], [S_fid[band].mean()], marker="*", s=110, color="#f2c14e",
                  edgecolor="k", lw=0.5, zorder=6, label="fiducial")
    ax[1].set_xlabel(r"$f_{\rm gas}$ (group $M_{200c}$ $1$-$3\times10^{13}\,h^{-1}M_\odot$)")
    ax[1].set_ylabel(r"$S(\ell\!\sim\!1500)$")
    ax[1].legend(fontsize=6.2, loc="lower right")
    panel_label(ax[1], "(b)")

    fig.tight_layout(w_pad=1.6)
    save(fig, "figs/fig06_bridge_fgas")
    print(f"S-f_gas r={r:.2f}")


if __name__ == "__main__":
    main()
