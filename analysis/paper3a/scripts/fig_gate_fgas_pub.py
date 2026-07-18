"""Publication-style A3 gate f_gas(M500) overlay (style-kit).

SB35 sphericalized f_gas(<R500) envelope (16-84% + min-max bands, TNG
fiducial) vs eRASS1 per-cluster medians, at the z~0 snapshot (096). Same
data recipe as `gate_overlay.py::fgas_figure` (kept in the original,
untouched); this script only restyles the output through the house kit.

Run: python analysis/paper3a/scripts/fig_gate_fgas_pub.py
Out: /mnt/ceph/users/mlee1/paper3/A/wp3_gate/figures/gate_fgas_overlay_pub.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

import matplotlib

matplotlib.use("Agg")

from analysis.paper3a.data_vectors.data_vectors import load_egas_erass1  # noqa: E402
from analysis.paper3a.gate.aggregate import LOGM500_BIN_EDGES  # noqa: E402
from analysis.paper3a.observables.constants import TNG_H  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, apply, band, condition_tag, data_points, fiducial_line, fig_single,
)

ENV = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/envelopes/envelopes_snap096.npz")
OUT = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/figures")


def _erass1_medians(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eRASS1 per-cluster medians in the same log-M500 bins, exactly as
    gate_overlay.py::fgas_figure (h-free catalog masses -> Msun/h;
    MAD/sqrt(n) errors)."""
    egas = load_egas_erass1("primary")
    known = egas.mass_known_mask()
    logm_h = egas.log10_m500_msun[known] + np.log10(TNG_H)
    fg = egas.f_gas500[known]
    med = np.full(len(x), np.nan)
    err = np.full(len(x), np.nan)
    for i in range(len(x)):
        sel = (logm_h >= LOGM500_BIN_EDGES[i]) & (logm_h < LOGM500_BIN_EDGES[i + 1]) & np.isfinite(fg)
        if sel.sum() >= 10:
            med[i] = np.median(fg[sel])
            err[i] = 1.4826 * np.median(np.abs(fg[sel] - med[i])) / np.sqrt(sel.sum())
    return med, err


def main() -> None:
    apply()
    d = np.load(ENV)
    p = "snap096__fgas__"
    x = d[p + "logm500_centers_msunh"]

    fig, ax = fig_single()

    band(ax, x, d[p + "sobol_band__min"], d[p + "sobol_band__max"],
         color=COL["model"], alpha=0.12, label="SB35 min-max")
    band(ax, x, d[p + "sobol_band__p16"], d[p + "sobol_band__p84"],
         color=COL["model"], alpha=0.30, label="SB35 16-84%")
    fiducial_line(ax, x, d[p + "fiducial"], label="TNG fiducial")

    med, err = _erass1_medians(x)
    good = np.isfinite(med)
    data_points(ax, x[good], med[good], yerr=err[good], label="eRASS1 medians")

    ax.set_xlabel(r"$\log_{10} M_{500}\ [M_\odot/h]$")
    ax.set_ylabel(r"$f_{\rm gas,sph}(<R_{500})$")
    condition_tag(ax, r"$z \simeq 0$ · eRASS1")
    ax.legend(loc="upper left", fontsize=6.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"gate_fgas_overlay_pub.{ext}")
    print(f"wrote {OUT}/gate_fgas_overlay_pub.pdf/.png")


if __name__ == "__main__":
    main()
