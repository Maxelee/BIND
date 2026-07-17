"""WP-A4 validation: k-fold + out-of-design results -> acceptance table + figures.

Consumes the fit artifacts under /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/
(gasemu_kfold.npz, gasemu_outdesign.npz, gasemu_gp.npz, gasemu_dataset.npz)
and produces, in the same directory:

- ``validation_summary.json``: per-block k-fold fractional errors per
  snapshot; the plan's acceptance ratios (held-out emulation error vs 1/3 of
  the frozen measurement error, per data-comparable bin); out-of-design
  (twobound + fiducial) errors; physics-sanity verdicts.
- ``figures/wp4_kfold_errors.png``: k-fold error by block and snapshot,
  with the acceptance thresholds where data errors exist.
- ``figures/wp4_outdesign.png``: emulator prediction vs painted truth at the
  60 twobound 1P extremes + fiducial (none in the training design).
- ``figures/wp4_wind_response.png``: emulated group-scale f_gas response
  along WindEnergyIn1e51erg (the deciding axis), with the painted twobound
  truth at the bounds — the plan's monotonicity sanity check.
- ``figures/wp4_ksz_decorrelation.png``: w(R) and the smoke-run raw vs
  decorrelated T_kSZ against the frozen Qu vectors.

Run (torch3 venv, after the fit):
    python analysis/paper3a/scripts/wp4_validate_emulator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.data_vectors.data_vectors import (  # noqa: E402
    load_egas_erass1,
    load_ksz_qu2026_lrg_by_mass,
)
from analysis.paper3a.emulator.forward import ForwardModel, transverse_weight  # noqa: E402
from analysis.paper3a.emulator.gasemu import GasEmulator  # noqa: E402
from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.velocity import LinearVelocity  # noqa: E402
from analysis.paper3a.gate.aggregate import cyltosph_for_snap  # noqa: E402
from analysis.paper3a.observables.constants import T_CMB_UK  # noqa: E402

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
FIGS = WP4 / "figures"

# Okabe-Ito (matches the gate overlay figures)
C_BAND = "#0072B2"
C_DATA = "#D55E00"
C_OK = "#009E73"
C_GRAY = "#999999"

SNAP_FOR_Z0 = "096"
SNAP_FOR_LRG = "063"


def _slices(f):
    sizes = np.asarray(f["block_sizes"], int)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    return {str(b): slice(int(offs[i]), int(offs[i + 1]))
            for i, b in enumerate(f["block_names"])}


def kfold_tables(kf) -> dict:
    """Median |pred-truth|/|truth| per (block, snap) over held-out rows."""
    snaps = [str(s) for s in kf["snaps"]]
    slices = _slices(kf)
    out = {}
    for zi, s in enumerate(snaps):
        row = {}
        for b, sl in slices.items():
            t = kf["truth"][:, zi, sl].astype(float)
            p = kf["pred"][:, zi, sl].astype(float)
            m = np.isfinite(t) & np.isfinite(p) & (np.abs(t) > 0)
            row[b] = float(np.median(np.abs(p[m] - t[m]) / np.abs(t[m]))) if m.any() else np.nan
        out[s] = row
    return out


def acceptance_fgas(kf, emu) -> dict:
    """fgas_med bins vs 1/3 of the eRASS1 stacked-median fractional errors,
    z~0 snapshot. Data medians/errors binned exactly as in gate_overlay
    (per-cluster catalog, h-free masses -> Msun/h, MAD/sqrt(n))."""
    from analysis.paper3a.observables.constants import TNG_H

    egas = load_egas_erass1("primary")
    c2s, _ = cyltosph_for_snap(SNAP_FOR_Z0)
    slices = _slices(kf)
    zi = [str(s) for s in kf["snaps"]].index(SNAP_FOR_Z0)
    sl = slices["fgas_med"]
    t = kf["truth"][:, zi, sl].astype(float)
    p = kf["pred"][:, zi, sl].astype(float)
    emu_frac = np.nanmedian(np.abs(p - t) / np.abs(t), axis=0)     # per gate bin

    known = egas.mass_known_mask()
    logm_h = egas.log10_m500_msun[known] + np.log10(TNG_H)
    fg = egas.f_gas500[known]
    edges = emu.logm500_bin_edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    rows = []
    for i, cen in enumerate(centers):
        sel = (logm_h >= edges[i]) & (logm_h < edges[i + 1]) & np.isfinite(fg)
        if sel.sum() < 10 or not np.isfinite(emu_frac[i]):
            rows.append({"logm500_center": float(cen), "n_data": int(sel.sum()),
                         "pass": None})
            continue
        med = np.median(fg[sel])
        err = 1.4826 * np.median(np.abs(fg[sel] - med)) / np.sqrt(sel.sum())
        data_frac = float(err / med)
        rows.append({
            "logm500_center": float(cen), "n_data": int(sel.sum()),
            "emulation_frac_err": float(emu_frac[i]),
            "data_frac_err": data_frac,
            "ratio_vs_third": float(emu_frac[i] / (data_frac / 3.0)),
            "pass": bool(emu_frac[i] <= data_frac / 3.0),
        })
    return {"cyltosph_applied_downstream": c2s, "bins": rows}


def acceptance_ksz(kf) -> dict:
    """ksz bins vs 1/3 of the Qu fractional errors (per radius), LRG snapshot.

    Model tau errors are fractional; the normalization is a common factor, so
    fractional comparison is exact. bin0 ~ m1, bin1 ~ m4 (nearest ticks).
    """
    dv = load_ksz_qu2026_lrg_by_mass()
    slices = _slices(kf)
    zi = [str(s) for s in kf["snaps"]].index(SNAP_FOR_LRG)
    out = {}
    for blk, key in (("ksz0", "m1"), ("ksz1", "m4")):
        sl = slices[blk]
        t = kf["truth"][:, zi, sl].astype(float)
        p = kf["pred"][:, zi, sl].astype(float)
        emu_frac = np.nanmedian(np.abs(p - t) / np.abs(t), axis=0)
        data_frac = np.abs(dv[key].errors / dv[key].values)
        out[blk] = {
            "data_quartile": key,
            "radii_arcmin": dv[key].bins.tolist(),
            "emulation_frac_err": emu_frac.tolist(),
            "data_frac_err": data_frac.tolist(),
            "pass_per_radius": (emu_frac <= data_frac / 3.0).tolist(),
        }
    return out


def outdesign_summary(od) -> dict:
    slices = _slices(od)
    snaps = [str(s) for s in od["snaps"]]
    res = {}
    for bundle in ("twobound", "fiducial"):
        t = od[f"{bundle}_truth"].astype(float)
        p = od[f"{bundle}_pred"].astype(float)
        per_block = {}
        for b, sl in slices.items():
            tt, pp = t[..., sl], p[..., sl]
            m = np.isfinite(tt) & np.isfinite(pp) & (np.abs(tt) > 0)
            per_block[b] = float(np.median(np.abs(pp[m] - tt[m]) / np.abs(tt[m])))
        res[bundle] = per_block
    res["snaps"] = snaps
    return res


def wind_response_figure(emu, ds) -> dict:
    """Emulated group-bin f_gas along WindEnergyIn1e51erg at z~0 vs painted
    twobound truth; returns the monotonicity verdict."""
    name = "WindEnergyIn1e51erg"
    i = pm.ASTRO_NAMES.index(name)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    sweep = np.linspace(0.0, 1.0, 21)
    U = np.tile(u_fid, (len(sweep), 1))
    U[:, i] = sweep
    pred = emu.predict(U, SNAP_FOR_Z0, return_std=True)
    gbin = 0                                            # 13.0-13.4, the loaded axis
    y, sd = pred["fgas_med"][:, gbin], pred["fgas_med_std"][:, gbin]

    tb_par = [str(x) for x in ds[f"snap{SNAP_FOR_Z0}_twobound_param"]]
    tb_bnd = [str(x) for x in ds[f"snap{SNAP_FOR_Z0}_twobound_bound"]]
    tbY = ds[f"snap{SNAP_FOR_Z0}_Y_twobound"]
    tbX = ds[f"snap{SNAP_FOR_Z0}_X_twobound"]
    pts = [(tbX[j, i], tbY[j, gbin]) for j in range(len(tb_par)) if tb_par[j] == name]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(sweep, y, color=C_BAND, lw=2, label="emulator (fiducial slice)")
    ax.fill_between(sweep, y - sd, y + sd, color=C_BAND, alpha=0.25, lw=0)
    if pts:
        ax.scatter(*zip(*pts), color=C_DATA, zorder=5, s=60, marker="s",
                   label="painted twobound truth")
    fidY = ds[f"snap{SNAP_FOR_Z0}_Y_fiducial"]
    ax.scatter([u_fid[i]], [fidY[0, gbin]], color="k", zorder=6, s=70, marker="*",
               label="painted TNG fiducial")
    ax.set_xlabel(f"{name} (unit cube)")
    ax.set_ylabel(r"median $f_{\rm gas,cyl}(<R_{500})$, $10^{13.0-13.4}\,M_\odot/h$")
    ax.set_title("WP-A4 sanity: group f$_{gas}$ response along the deciding axis (z=0.034)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "wp4_wind_response.png", dpi=150)
    plt.close(fig)
    dy = np.diff(y)
    return {"monotonic_increasing_frac": float(np.mean(dy > 0)),
            "range": [float(y[0]), float(y[-1])]}


def kfold_figure(kf, tables):
    snaps = [str(s) for s in kf["snaps"]]
    blocks = [str(b) for b in kf["block_names"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(blocks))
    for k, s in enumerate(snaps):
        vals = [tables[s][b] * 100 for b in blocks]
        ax.plot(x + (k - 2.5) * 0.06, vals, "o", ms=5,
                color=plt.cm.viridis(k / max(len(snaps) - 1, 1)), label=f"snap {s}")
    ax.set_xticks(x, blocks)
    ax.set_yscale("log")
    ax.set_ylabel("k-fold held-out median |Δ|/truth  [%]")
    ax.set_title("WP-A4 emulator: 8-fold held-out error by block (256 Sobol runs)")
    ax.axhline(1.0, color=C_GRAY, lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "wp4_kfold_errors.png", dpi=150)
    plt.close(fig)


def outdesign_figure(od):
    slices = _slices(od)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, blk, lab in zip(axes, ("fgas_med", "ym", "ksz1"),
                            ("f_gas medians", "Y-M (alpha, beta, scatter)", "kSZ bin1 tau")):
        sl = slices[blk]
        t = od["twobound_truth"][..., sl].astype(float).ravel()
        p = od["twobound_pred"][..., sl].astype(float).ravel()
        m = np.isfinite(t) & np.isfinite(p)
        ax.scatter(t[m], p[m], s=6, alpha=0.4, color=C_BAND, label="twobound (out-of-design)")
        tf = od["fiducial_truth"][..., sl].astype(float).ravel()
        pf = od["fiducial_pred"][..., sl].astype(float).ravel()
        mf = np.isfinite(tf) & np.isfinite(pf)
        ax.scatter(tf[mf], pf[mf], s=25, color="k", marker="*", zorder=5, label="fiducial")
        lo, hi = np.nanmin(t[m]), np.nanmax(t[m])
        ax.plot([lo, hi], [lo, hi], color=C_GRAY, lw=0.8)
        if blk != "ym":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("painted truth")
        ax.set_ylabel("emulator prediction")
        ax.set_title(lab, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("WP-A4 out-of-design test: 1P extremes + fiducial (all snaps)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "wp4_outdesign.png", dpi=150)
    plt.close(fig)


def decorrelation_figure(smoke_path: Path):
    d = dict(np.load(smoke_path))
    lv = LinearVelocity()
    fm = ForwardModel(velocity=lv)
    radii = d["radii_arcmin"]
    z = float(d["z_snap"])
    dv = load_ksz_qu2026_lrg_by_mass()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2))
    R = np.linspace(0.05, 8.5, 200)
    ax0.plot(R, transverse_weight(lv, R), color=C_BAND, lw=2)
    ax0.axhline(lv.slab_mean_rv(51.25), color=C_GRAY, ls="--", lw=1,
                label=f"uniform-slab mean = {lv.slab_mean_rv(51.25):.2f}")
    ax0.set_xlabel("transverse R  [Mpc/h]")
    ax0.set_ylabel(r"velocity coherence weight $w(R)$")
    ax0.set_ylim(0, 1.05)
    ax0.legend(frameon=False, fontsize=9)
    ax0.set_title("LOS decorrelation weight (linear theory)")

    for bi, qk, c in ((0, "m1", C_BAND), (1, "m4", C_DATA)):
        res = fm.ksz_tksz_from_profile(d[f"ksz_bin{bi}_sigma_r"],
                                       d["sigma_r_centers_mpch"], z, radii)
        ax1.plot(radii, res.tksz_raw, color=c, ls=":", lw=1.5)
        ax1.plot(radii, res.tksz, color=c, lw=2,
                 label=f"model bin{bi} (decorr.)")
        ax1.errorbar(dv[qk].bins, dv[qk].values, yerr=dv[qk].errors, fmt="o",
                     ms=4, color=c, alpha=0.6, label=f"Qu {qk}")
    ax1.set_yscale("log")
    ax1.set_xlabel("CAP radius  [arcmin]")
    ax1.set_ylabel(r"$T_{\rm kSZ}$  [$\mu$K arcmin$^2$]")
    ax1.set_title(f"smoke run_0000 snap063 (z={z:.2f}); dotted = co-moving")
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "wp4_ksz_decorrelation.png", dpi=150)
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help='e.g. "_v3" to validate the v3 fit')
    args = ap.parse_args()
    sfx = args.suffix

    FIGS.mkdir(parents=True, exist_ok=True)
    kf = dict(np.load(WP4 / f"gasemu_kfold{sfx}.npz", allow_pickle=False))
    od = dict(np.load(WP4 / f"gasemu_outdesign{sfx}.npz", allow_pickle=False))
    ds = dict(np.load(WP4 / f"gasemu_dataset{sfx}.npz", allow_pickle=False))
    emu = GasEmulator.load(WP4 / f"gasemu_gp{sfx}.npz")

    tables = kfold_tables(kf)
    summary = {
        "kfold_frac_err_by_snap_block": tables,
        "acceptance_fgas_vs_erass1": acceptance_fgas(kf, emu),
        "acceptance_ksz_vs_qu": acceptance_ksz(kf),
        "outdesign": outdesign_summary(od),
        "wind_response": None,
    }
    kfold_figure(kf, tables)
    outdesign_figure(od)
    summary["wind_response"] = wind_response_figure(emu, ds)
    smoke = Path("/tmp/claude-2107/-mnt-home-mlee1-bind-paper3-plans/"
                 "f5bec188-4cfe-4a1c-9062-4919e088c70d/scratchpad/v3b_smoke_run0000_snap063.npz")
    if smoke.exists():
        decorrelation_figure(smoke)
    (WP4 / f"validation_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("kfold_frac_err_by_snap_block", "wind_response")}, indent=2))
    print(f"wrote {WP4}/validation_summary{sfx}.json and figures/")


if __name__ == "__main__":
    main()
