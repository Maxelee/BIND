"""Publication-style WP-A4 wind-response sanity figure (style-kit).

Emulated group-bin f_gas response along WindEnergyIn1e51erg (the deciding
axis) at z~0, with the painted twobound truth at the parameter bounds and
the painted TNG fiducial. Same data recipe as
`wp4_validate_emulator.py::wind_response_figure` (kept in the original,
untouched); this script only restyles the output through the house kit.

Run: python analysis/paper3a/scripts/fig_wind_response_pub.py
Out: /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/figures/wp4_wind_response_pub.{pdf,png}
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

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.gasemu import GasEmulator  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, apply, band, condition_tag, data_points, fig_single,
)

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
OUT = WP4 / "figures"
SNAP = "096"
PARAM = "WindEnergyIn1e51erg"
GBIN = 0  # 13.0-13.4, the loaded axis


def main() -> None:
    apply()
    emu = GasEmulator.load()
    ds = dict(np.load(WP4 / "gasemu_dataset_v4.npz", allow_pickle=False))

    i = pm.ASTRO_NAMES.index(PARAM)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    sweep = np.linspace(0.0, 1.0, 21)
    U = np.tile(u_fid, (len(sweep), 1))
    U[:, i] = sweep
    pred = emu.predict(U, SNAP, return_std=True)
    y, sd = pred["fgas_med"][:, GBIN], pred["fgas_med_std"][:, GBIN]

    tb_par = [str(x) for x in ds[f"snap{SNAP}_twobound_param"]]
    tbY = ds[f"snap{SNAP}_Y_twobound"]
    tbX = ds[f"snap{SNAP}_X_twobound"]
    pts = [(tbX[j, i], tbY[j, GBIN]) for j in range(len(tb_par)) if tb_par[j] == PARAM]
    fidY = ds[f"snap{SNAP}_Y_fiducial"]

    fig, ax = fig_single()

    ax.plot(sweep, y, color=COL["model"], lw=2.2, label="emulator (fiducial slice)")
    band(ax, sweep, y - sd, y + sd, color=COL["model"], alpha=0.25)

    if pts:
        px, py = zip(*pts)
        data_points(ax, px, py, color=COL["data"], marker="s", ms=6,
                    label="painted twobound truth")
    data_points(ax, [u_fid[i]], [fidY[0, GBIN]], color="black", marker="*",
                ms=9, label="painted TNG fiducial")

    ax.set_xlabel(f"{PARAM} (unit cube)")
    ax.set_ylabel(r"median $f_{\rm gas,cyl}(<R_{500})$, $10^{13.0-13.4}\,M_\odot/h$")
    condition_tag(ax, "snap 096 · group bin")
    ax.legend(loc="lower right", fontsize=6.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"wp4_wind_response_pub.{ext}")
    print(f"wrote {OUT}/wp4_wind_response_pub.pdf/.png")


if __name__ == "__main__":
    main()
