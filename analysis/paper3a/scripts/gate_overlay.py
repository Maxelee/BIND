"""A3 gate overlay driver: envelopes + data-overlay figures + support stats.

Consumes the operator tables (see gate.aggregate) and the A1 frozen vectors,
writes per-snapshot envelope npz, the two gate overlay figures, and the
support-fraction json that the DECISION_MEMO quotes. Run on rusty:

    python analysis/paper3a/scripts/gate_overlay.py

Everything lands under /mnt/ceph/users/mlee1/paper3/A/wp3_gate/{envelopes,figures}.
Colors are the Okabe-Ito colorblind-safe set in fixed assignment (band/median
= blue, data = vermilion, fiducial = black, 1P trajectories = gray).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paper3a.gate.aggregate import (  # noqa: E402
    ANNOTATIONS,
    LOGM500_BIN_EDGES,
    fgas_envelope,
    ksz_envelope,
    load_all,
    save_envelopes,
    support_fraction,
    ym_envelope,
)
from analysis.paper3a.data_vectors.data_vectors import (  # noqa: E402
    load_egas_erass1,
    load_ksz_qu2026_lrg_by_mass,
)
from analysis.paper3a.observables.constants import TNG_H  # noqa: E402

OUT = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate")
GATE_SNAPS = ("096", "071", "067", "063", "056", "049")
KSZ_SNAPS = ("063", "056")  # bracket the Qu mass-binned z_eff ~ 0.65-0.7

C_BAND = "#0072B2"   # Okabe-Ito blue: SB35 envelope
C_DATA = "#D55E00"   # vermilion: frozen data
C_FID = "#000000"    # fiducial curve
C_1P = "#999999"     # twobound trajectories


def plot_band(ax, x, b, label):
    ax.fill_between(x, b["min"], b["max"], color=C_BAND, alpha=0.15, lw=0)
    ax.fill_between(x, b["p16"], b["p84"], color=C_BAND, alpha=0.35, lw=0,
                    label=f"{label} 16-84% (n={b['n']})")
    ax.plot(x, b["p50"], color=C_BAND, lw=2)


def ksz_figure(tables, qu_by_mass) -> dict:
    """2x2: (mass bin 0/1) x (snap 063/056), tau_CAP->T_kSZ nominal, Qu overlay."""
    support = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, snap in enumerate(KSZ_SNAPS):
        z = tables[snap]["sb35"][0].z_snap
        for row, (mbin, qu_key) in enumerate([(0, "m3"), (1, "m4")]):
            ax = axes[row, col]
            env = ksz_envelope(tables, snap, mbin, in_tksz=True)
            x = env["radii_arcmin"]
            plot_band(ax, x, env["sobol_band"], "SB35")
            for curve in env["twobound"].values():
                ax.plot(x, curve, color=C_1P, lw=0.6, alpha=0.5, zorder=1)
            if env["fiducial"] is not None:
                ax.plot(x, env["fiducial"], color=C_FID, lw=2, label="TNG fiducial")
            dv = qu_by_mass[qu_key]
            ax.errorbar(dv.bins, dv.values, yerr=dv.errors, fmt="o", ms=5,
                        color=C_DATA, label=f"Qu+26 {qu_key} (log M200c tick "
                        f"{dv.mass_definition.split('tick = ')[1][:5]})", zorder=5)
            lo, hi = env["mass_bin_range"]
            ax.set_title(f"snap {snap} (z={z:.2f}) · model logM200c [{lo:.1f}, {hi:.1f})", fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$\theta_d$ [arcmin]")
            if col == 0:
                ax.set_ylabel(r"$T_{\rm kSZ}$ [$\mu$K arcmin$^2$] (nominal norm)")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(alpha=0.2, lw=0.5)

            # gate-level support stat, diagonal errors, nominal normalization
            from analysis.paper3a.gate.aggregate import NOMINAL_VRMS_OVER_C
            from analysis.paper3a.observables.constants import T_CMB_UK

            sobol = np.array([t.data[f"ksz_bin{mbin}_stack"] for t in tables[snap]["sb35"]])
            sobol_tksz = sobol * T_CMB_UK * NOMINAL_VRMS_OVER_C
            support[f"ksz_{qu_key}_snap{snap}"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in support_fraction(sobol_tksz, dv.values, dv.errors).items()
                if k != "rms_per_point"
            }
    fig.suptitle("A3 gate: kSZ CAP profiles — SB35 envelope vs Qu et al. 2026 "
                 "(velocity normalization NOMINAL, see annotations)", fontsize=10)
    fig.tight_layout()
    out = OUT / "figures" / "gate_ksz_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")
    return support


def fgas_figure(tables, egas) -> dict:
    """f_gas(M500) at the z~0 snap: model (sphericalized) band vs eRASS1 medians."""
    snap = "096"
    env = fgas_envelope(tables, snap)
    x = env["logm500_centers_msunh"]

    # eROSITA per-cluster medians in the same bins; catalog masses are h-free
    # Msun -> convert to Msun/h for the shared axis.
    known = egas.mass_known_mask()
    logm_h = egas.log10_m500_msun[known] + np.log10(TNG_H)
    fg = egas.f_gas500[known]
    data_med = np.full(len(x), np.nan)
    data_err = np.full(len(x), np.nan)
    for i in range(len(x)):
        sel = (logm_h >= LOGM500_BIN_EDGES[i]) & (logm_h < LOGM500_BIN_EDGES[i + 1]) & np.isfinite(fg)
        if sel.sum() >= 10:
            data_med[i] = np.median(fg[sel])
            data_err[i] = 1.4826 * np.median(np.abs(fg[sel] - data_med[i])) / np.sqrt(sel.sum())

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_band(ax, x, env["sobol_band"], "SB35")
    for curve in env["twobound"].values():
        ax.plot(x, curve, color=C_1P, lw=0.6, alpha=0.5, zorder=1)
    if env["fiducial"] is not None:
        ax.plot(x, env["fiducial"], color=C_FID, lw=2, label="TNG fiducial")
    good = np.isfinite(data_med)
    ax.errorbar(x[good], data_med[good], yerr=data_err[good], fmt="o", ms=6,
                color=C_DATA, label="eRASS1 medians (point-estimate masses)", zorder=5)
    ax.set_xlabel(r"$\log_{10} M_{500c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$f_{\rm gas}(<R_{500c})$, sphericalized "
                  f"(CylToSph = {env['cyltosph_applied']:.3f})")
    ax.set_title(f"A3 gate: f_gas(M) at z≈0 (snap {snap}) — SB35 envelope vs eRASS1", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    out = OUT / "figures" / "gate_fgas_overlay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")

    sobol = np.array([t.fgas_median_by_massbin(env["cyltosph_applied"])
                      for t in tables[snap]["sb35"]])
    return {"fgas_snap096": {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in support_fraction(sobol[:, good], data_med[good], data_err[good]).items()
        if k != "rms_per_point"
    }}


def main() -> None:
    tables = load_all(snaps=GATE_SNAPS)
    for snap in GATE_SNAPS:
        if snap not in tables:
            print(f"snap {snap}: no tables yet, skipping")
            continue
        envs = {
            "ksz_bin0": ksz_envelope(tables, snap, 0),
            "ksz_bin1": ksz_envelope(tables, snap, 1),
            "fgas": fgas_envelope(tables, snap),
            "ym": ym_envelope(tables, snap),
        }
        save_envelopes(OUT / "envelopes", snap, envs)
        nb = {b: len(v) for b, v in tables[snap].items()}
        print(f"snap {snap}: envelopes saved ({nb})")

    support = {}
    support.update(ksz_figure(tables, load_ksz_qu2026_lrg_by_mass()))
    support.update(fgas_figure(tables, load_egas_erass1("primary")))
    support["annotations"] = ANNOTATIONS
    sp = OUT / "envelopes" / "support_fractions.json"
    sp.write_text(json.dumps(support, indent=2))
    print(f"wrote {sp}")
    for k, v in support.items():
        if k != "annotations":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
