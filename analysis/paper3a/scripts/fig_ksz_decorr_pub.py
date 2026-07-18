"""Publication-style WP-A4 kSZ decorrelation figure (style-kit).

Two panels: (left) the LOS velocity-coherence weight w(R) with the
uniform-slab mean as a reference line; (right) the theta-continuous
decorrelated T_kSZ(theta) forward-modeled from the emulated bin1 sigma_r
profile at the fiducial (snap 063, z=0.60), co-moving vs decorrelated,
against the frozen Qu et al. 2026 m4 (highest stellar-mass quartile) data.

Data flow follows `wp4_validate_emulator.py::decorrelation_figure()` for the
w(R) panel, but the right panel replaces that function's smoke-run npz
(a scratch artifact from an earlier session, not guaranteed to exist) with
the emulator prediction directly: GasEmulator.load() (v4) -> predict
`ksz1_sr` at the fiducial unit parameters, snap 063 -> ForwardModel
.ksz_tksz_from_profile() evaluated on a fine continuous theta grid (rather
than only the 9 discrete gate radii), using the v4 dataset's
`sigma_r_centers_mpch` r-grid. Model curves are role-locked blue (never the
data color, unlike the original script's bin1-vs-C_DATA choice); data are
vermilion circles.

Run: python analysis/paper3a/scripts/fig_ksz_decorr_pub.py
Out: /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/figures/wp4_ksz_decorrelation_pub.{pdf,png}
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

from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_by_mass  # noqa: E402
from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.forward import (  # noqa: E402
    SLAB_DEPTH_HMPC, ForwardModel, transverse_weight,
)
from analysis.paper3a.emulator.gasemu import GasEmulator  # noqa: E402
from analysis.paper3a.emulator.velocity import LinearVelocity  # noqa: E402
from analysis.paper3a.style import (  # noqa: E402
    COL, apply, condition_tag, data_points, fig_double, refline,
)

WP4 = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
OUT = WP4 / "figures"
SNAP = "063"


def main() -> None:
    apply()
    emu = GasEmulator.load()
    lv = LinearVelocity()
    fm = ForwardModel(velocity=lv)
    z = emu.snap_z[SNAP]

    fig, (ax0, ax1) = fig_double(height=2.8)

    # --- left: LOS velocity-coherence weight w(R) -------------------------
    R = np.linspace(0.05, 8.5, 200)
    w = transverse_weight(lv, R)
    ax0.plot(R, w, color=COL["model"], lw=1.8)
    slab_mean = lv.slab_mean_rv(SLAB_DEPTH_HMPC)
    refline(ax0, y=slab_mean, label=f"uniform-slab mean = {slab_mean:.2f}")
    ax0.set_xlabel(r"transverse $R$ [Mpc/$h$]")
    ax0.set_ylabel(r"velocity coherence weight $w(R)$")
    ax0.set_ylim(0, 1.05)

    # --- right: theta-continuous decorrelated T_kSZ vs Qu m4 --------------
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    pred = emu.predict(u_fid, SNAP, return_std=False)
    sigma_r = pred["ksz1_sr"]
    ds = dict(np.load(WP4 / "gasemu_dataset_v4.npz", allow_pickle=False))
    r_centers = ds["sigma_r_centers_mpch"]

    dv = load_ksz_qu2026_lrg_by_mass()["m4"]
    theta = np.linspace(0.75, 6.25, 60)
    res = fm.ksz_tksz_from_profile(sigma_r, r_centers, z, theta)

    ax1.plot(theta, res.tksz_raw, color=COL["model"], lw=1.1, ls="--",
             label="co-moving (no decorr.)")
    ax1.plot(theta, res.tksz, color=COL["model"], lw=2.2,
             label="decorrelated (bin1)")
    data_points(ax1, dv.bins, dv.values, yerr=dv.errors, color=COL["data"],
                marker="o", ms=5, label="Qu+26 m4")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"CAP radius $\theta$ [arcmin]")
    ax1.set_ylabel(r"$T_{\rm kSZ}$ [$\mu$K arcmin$^2$]")
    condition_tag(ax1, r"$z = 0.60$ · bin1 vs Qu m4")
    ax1.legend(loc="lower right", fontsize=6.5)
    ax1.annotate("sample-selection model\npending (A5)", (0.03, 0.90),
                 xycoords="axes fraction", ha="left", va="top",
                 fontsize=7, color="#666666")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"wp4_ksz_decorrelation_pub.{ext}")
    print(f"wrote {OUT}/wp4_ksz_decorrelation_pub.pdf/.png")


if __name__ == "__main__":
    main()
