#!/usr/bin/env python
"""fig07_bridge_response -- the per-halo gas fraction predicts the field observables
(multi-probe response matrix across the 57-run 1P feedback suite).

Panel (a): |corr(field observable, halo Delta-f_gas(group))| for 7 field
statistics -- S(ell), R(nu) [tSZ-at-peaks thermal energy], N_min, C_kappa-y,
V1, V2 (Minkowski functionals), N_pk -- bar colour flags |r|>0.7 (green) vs
weaker (orange).
Panel (b): the strongest concrete case, field Delta-R(nu) vs. halo
Delta-ln f_gas (group), coloured by feedback family, r=0.92.

Data (cached, no engine re-run):
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_stats.npz
      (keys: nu, ell, Rnu, per-run "{run}_clk"/"{run}_ky"/"{run}_R"/"{run}_min"/
      "{run}_V1"/"{run}_V2"/"{run}_pk")
  - /mnt/home/mlee1/ceph/bind_science/halo_atlas/{run}_snap096.npz,
    /mnt/home/mlee1/ceph/bind_science/halo_atlas/fid_snap096.npz
      (paired response: keys M_fof, m_gas_500c_bg, m_tot_500c_bg, Y_500c)
  - /mnt/home/mlee1/ceph/bind_science/dashboard_cache/g1_design.json
  - src/bind/assets/SB35_param_minmax.csv (param Description, for family tagging)

Source: examples/bind_bridge.py::fig_response() (worktree analysis/wl-tsz-bridge).
Placeholder figs/fig07_bridge_response.png is byte-identical (md5) to the on-disk
figs/bind_bridge_response.png produced by that function.
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
ACSV = REPO / "src/bind/assets/SB35_param_minmax.csv"


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


def _atlas_resp(run, mlo, mhi, key="fgas"):
    """Delta-ln(quantity) vs fiducial in an M200c bin at z~0 (paired, same halos)."""
    fr, ff = ATLAS / f"{run}_snap096.npz", ATLAS / "fid_snap096.npz"
    if not (fr.exists() and ff.exists()):
        return np.nan
    dr, df = np.load(fr), np.load(ff)
    if "m_gas_500c_bg" not in dr.files:
        return np.nan
    s = (df["M_fof"] >= mlo) & (df["M_fof"] < mhi)

    def q(d):
        if key == "fgas":
            return d["m_gas_500c_bg"][s] / np.where(d["m_tot_500c_bg"][s] > 0,
                                                     d["m_tot_500c_bg"][s], np.nan)
        return d["Y_500c"][s]

    with np.errstate(all="ignore"):
        return float(np.log(np.nanmedian(q(dr)) / np.nanmedian(q(df))))


def main():
    setup()

    gs = np.load(CACHE / "g1_stats.npz")
    nu, ell = gs["nu"], gs["ell"]
    li = int(np.argmin(np.abs(ell - 1500)))
    hi = nu > 3.0
    design = _design()

    probes = ["S(ℓ)", "R(ν)", "C_κy", "N_min", "V1", "V2", "N_pk"]
    fg, P, fam = [], {k: [] for k in probes}, []
    for run, name, val, fid, lf, desc in design:
        if f"{run}_clk" not in gs.files:
            continue
        g = _atlas_resp(run, 1e13, 3e13, "fgas")
        if not np.isfinite(g):
            continue

        def band(key, mask):
            return float(np.nanmean(gs[f"{run}_{key}"][mask])) if f"{run}_{key}" in gs.files else np.nan

        fg.append(g)
        fam.append(_family(name, desc))
        P["S(ℓ)"].append(float(gs[f"{run}_clk"][li]))
        P["R(ν)"].append(band("R", gs["Rnu"] > 2.0) if "Rnu" in gs.files else np.nan)
        P["C_κy"].append(float(gs[f"{run}_ky"][li]) if f"{run}_ky" in gs.files else np.nan)
        P["N_min"].append(band("min", nu < -1.0))
        P["V1"].append(float(np.nanmean(gs[f"{run}_V1"])) if f"{run}_V1" in gs.files else np.nan)
        P["V2"].append(float(np.nanmean(gs[f"{run}_V2"])) if f"{run}_V2" in gs.files else np.nan)
        P["N_pk"].append(band("pk", hi))
    fg = np.array(fg)
    fam = np.array(fam)
    rr = {}
    for k in probes:
        v = np.array(P[k])
        ok = np.isfinite(fg) & np.isfinite(v)
        rr[k] = abs(np.corrcoef(fg[ok], v[ok])[0, 1])
    order = sorted(probes, key=lambda k: rr[k])

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    # (a) correlation of every field observable with the halo gas mechanism
    bcol = [COLORS["secondary"] if rr[k] > 0.7 else "#e08214" for k in order]
    ax[0].barh(range(len(order)), [rr[k] for k in order], color=bcol, edgecolor="k", lw=0.5)
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels(order, fontsize=7.5)
    for i, k in enumerate(order):
        ax[0].text(rr[k] + 0.02, i, f"{rr[k]:.2f}", va="center", fontsize=6.5)
    ax[0].set_xlim(0, 1.08)
    ax[0].set_xlabel(r"$|{\rm corr}|$ with halo $\Delta f_{\rm gas}$(group), 57 runs")
    panel_label(ax[0], "(a)")

    # (b) example: R(nu) vs f_gas
    v = np.array(P["R(ν)"])
    z = np.polyfit(fg, v, 1)
    xg = np.linspace(fg.min(), fg.max(), 30)
    ax[1].plot(xg, np.polyval(z, xg), color=COLORS["truth"], ls="--", lw=1.2, zorder=2,
              label=f"fit, r={rr['R(ν)']:.2f}")
    for fk, col in FAMC.items():
        sel = fam == fk
        if sel.any():
            ax[1].scatter(fg[sel], v[sel], c=col, s=22, edgecolor="k", lw=0.3,
                          zorder=3, label=fk)
    ax[1].axhline(0, c="0.6", lw=0.6)
    ax[1].axvline(0, c="0.6", lw=0.6)
    ax[1].set_xlabel(r"halo $\Delta\ln f_{\rm gas}$ (group, atlas)")
    ax[1].set_ylabel(r"field $\Delta R(\nu)$ (tSZ energy at WL peaks)")
    ax[1].legend(fontsize=6.2, loc="lower right")
    panel_label(ax[1], "(b)")

    fig.tight_layout(w_pad=1.6)
    save(fig, "figs/fig07_bridge_response")
    print(", ".join(f"{k}:{rr[k]:.2f}" for k in order))


if __name__ == "__main__":
    main()
