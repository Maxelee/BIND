#!/usr/bin/env python
"""fig05_bridge_hero -- the same painted halo, seen by WL and tSZ.

Panel (a): per-peak WL peak-height change (Delta-kappa/kappa) vs. the atlas
gas-mass excursion (Delta-ln M_gas) of the dominant halo behind that peak,
paired across the 57 one-parameter (1P) feedback runs. Grey cloud = individual
peaks (field/projection noise); blue points = the same relation averaged over
the 42 peak-halos per feedback run -- kappa dims in proportion to the ejected
gas, slope ~= the peaks' projected gas fraction.
Panel (b): the ejection-heating plane, per feedback run: Delta-ln M_gas
(what kappa sees) vs. Delta-ln T = Delta-ln Y - Delta-ln M_gas (heating,
what kappa does NOT see), coloured by Delta-ln Y (the tSZ observable) --
tSZ probes the full gas x temperature plane, WL only the gas axis.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_peakhalo_map.npz
      (key: pk_uh -- dominant-halo index per WL peak)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_peakhalo.npz
      (keys: ref_nu, ref_Mg, ref_Y, and per-run "{run}_nu"/"{run}_Mg"/"{run}_Y")
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json (1P design table)
  - src/bind/assets/SB35_param_minmax.csv (param Description, for family tagging
    -- loaded for parity with the source script; family is not used in this figure)

Source: examples/bind_bridge.py::fig_hero() (worktree analysis/wl-tsz-bridge).
Placeholder figs/fig05_bridge_hero.png is byte-identical (md5) to the on-disk
figs/bind_bridge_hero.png produced by that function.
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
ACSV = REPO / "src/bind/assets/SB35_param_minmax.csv"


def _design():
    """[(run, name, val, fid, logflag, desc)] from g1_design.json + SB35 csv."""
    dj = json.load(open(CACHE / "g1_design.json"))
    meta = {r["ParamName"]: r for r in csv.DictReader(open(ACSV))}
    out = []
    for run, _i, name, val, fid in dj:
        m = meta.get(name, {})
        out.append((run, name, float(val), float(fid), int(m.get("LogFlag", 0)),
                    m.get("Description", name)))
    return out


def main():
    setup()

    m = np.load(CACHE / "g1_peakhalo_map.npz")
    ph = dict(np.load(CACHE / "g1_peakhalo.npz"))
    pk_uh = m["pk_uh"]
    design = _design()
    refnu = ph["ref_nu"]

    pxc, pyc = [], []  # per-peak cloud (field noise)
    rx, ry, ej, heat, lnY = [], [], [], [], []  # per-run aggregates
    for run, *_ in design:
        if f"{run}_nu" not in ph or f"{run}_Mg" not in ph:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            dkk = (ph[f"{run}_nu"] - refnu) / refnu
            dmg = np.log(ph[f"{run}_Mg"][pk_uh] / ph["ref_Mg"][pk_uh])
            dy = np.log(ph[f"{run}_Y"][pk_uh] / ph["ref_Y"][pk_uh])
        g = np.isfinite(dkk) & np.isfinite(dmg) & np.isfinite(dy)
        if g.sum() < 5:
            continue
        pxc.append(dmg[g])
        pyc.append(dkk[g])
        rx.append(dmg[g].mean())
        ry.append(dkk[g].mean())
        ej.append(dmg[g].mean())
        heat.append((dy - dmg)[g].mean())
        lnY.append(dy[g].mean())
    rx, ry, ej, heat, lnY = (np.asarray(v) for v in (rx, ry, ej, heat, lnY))
    pxc, pyc = np.concatenate(pxc), np.concatenate(pyc)
    print(f"hero: {len(rx)} feedback runs x {len(set(pk_uh.tolist()))} peak-halos")

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    # (a) WL peak dims proportional to its gas content
    ax[0].scatter(pxc, pyc, s=4, c=COLORS["dmo"], alpha=0.35, lw=0, label="per peak")
    z = np.polyfit(rx, ry, 1)
    r = np.corrcoef(rx, ry)[0, 1]
    xg = np.linspace(rx.min(), rx.max(), 20)
    ax[0].plot(xg, np.polyval(z, xg), color=COLORS["truth"], lw=1.6, zorder=4)
    ax[0].scatter(rx, ry, s=22, c=COLORS["bind"], edgecolor="k", lw=0.3, zorder=5,
                  label="per run (mean)")
    ax[0].text(0.04, 0.06, fr"slope$\,={z[0]:.3f}\approx f_{{\rm gas}}^{{\rm proj}}$" "\n"
               fr"$r={r:.2f}$", transform=ax[0].transAxes, fontsize=6.5, va="bottom")
    ax[0].axhline(0, c="0.7", lw=0.6)
    ax[0].axvline(0, c="0.7", lw=0.6)
    ax[0].set_xlabel(r"$\Delta\ln M_{\rm gas}$ (atlas, per halo)")
    ax[0].set_ylabel(r"$\Delta\kappa/\kappa=\Delta\nu/\nu$ (WL peak)")
    ax[0].legend(fontsize=6.2, loc="upper left")
    panel_label(ax[0], "(a)", loc="lower right")

    # (b) ejection-heating plane; kappa sees only the x-axis, y (tSZ) the diagonal
    lim = 1.05 * max(np.abs(ej).max(), np.abs(heat).max())
    vlim = np.abs(lnY).max()
    s2 = ax[1].scatter(ej, heat, c=lnY, cmap="RdBu_r", s=32, edgecolor="k", lw=0.3,
                       vmin=-vlim, vmax=vlim)
    ax[1].axhline(0, c="0.6", lw=0.6)
    ax[1].axvline(0, c="0.6", lw=0.6)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel(r"$\Delta\ln M_{\rm gas}$ (ejection, $\kappa$-visible)")
    ax[1].set_ylabel(r"$\Delta\ln T=\Delta\ln Y-\Delta\ln M_{\rm gas}$ (heating)")
    cb = fig.colorbar(s2, ax=ax[1])
    cb.set_label(r"$\Delta\ln Y$ (tSZ)")
    panel_label(ax[1], "(b)")

    fig.tight_layout(w_pad=1.6)
    save(fig, "figs/fig05_bridge_hero")
    print(f"per-run r={r:.2f}, slope={z[0]:.3f}")


if __name__ == "__main__":
    main()
