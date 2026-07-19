"""Publication figure: the joint A+B posterior in the decomposition
plane (paper centerpiece). Grammar follows Pandey+25 Fig. 7 (one
physical plane, constraint bands + individually-labeled comparison
points) rendered in the house style kit.

Run: python analysis/paper3a/scripts/fig_joint_ab_plane_pub.py
Out: wp6_propagation/figures/joint_ab_plane_pub.{pdf,png}
     (+ png copy into the plans repo figures_AB/)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.scripts.fig_a5_tension import load_flat  # noqa: E402
from analysis.paper3a.scripts.run_ab_gate import (  # noqa: E402
    GRID, bin_weights, delta_coords)
from analysis.paper3a.style import COL, W_DOUBLE, apply, condition_tag  # noqa: E402

N_MAP = 4000
PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_AB")


def contour_levels(h):
    s = np.sort(h.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    return [s[np.searchsorted(c, q)] for q in (0.95, 0.68)]


def kde_contours(ax, pts, color, lw=(0.9, 1.6), fill=False, zorder=3):
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=44)
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, 1.1)
    lv = contour_levels(H)
    x, y = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if fill:
        ax.contourf(x, y, H.T, levels=[lv[0], lv[1], H.max()],
                    colors=[color, color], alpha=0.35, zorder=zorder)
    ax.contour(x, y, H.T, levels=lv, colors=color,
               linewidths=lw, zorder=zorder + 1)


def main() -> None:
    apply()
    gate = json.loads((WP6 / "ab_gate.json").read_text())
    off = np.array([gate["reference_offset"]["dln_mgas"]["value"],
                    gate["reference_offset"]["dln_t"]["value"]])
    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                  allow_pickle=False)
    edges = np.asarray(json.loads(str(raw["manifest"]))["meta"]["mass_bins"],
                       float)
    w = bin_weights(edges)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    rng = np.random.default_rng(23)

    def coords_of(flat):
        sub = flat[rng.choice(len(flat), min(N_MAP, len(flat)),
                              replace=False)][:, :30]
        cc = delta_coords(emu, sub, u_fid, w)
        return np.stack([cc["dln_mgas"], cc["dln_t"]], axis=1)

    jflat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    jpts = coords_of(jflat)
    apts = coords_of(load_flat("a5_subset_joint_seed{k}.npz"))
    kpts = coords_of(load_flat("a5_subset_kszonly_seed{k}.npz"))
    sigma_pos = jflat[rng.choice(len(jflat), N_MAP, replace=False), 31] * 3.6

    fig = plt.figure(figsize=(W_DOUBLE, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    g = np.load(GRID, allow_pickle=True)
    nodes = np.stack([g["delta_ln_mgas"], g["delta_ln_t"]],
                     axis=1) - off[None, :]
    ax.scatter(nodes[:, 0], nodes[:, 1], s=11, facecolor="none",
               edgecolor="#888888", lw=0.55, zorder=2,
               label="$\\kappa$-peak model grid")

    kde_contours(ax, kpts, COL["aux"], lw=(0.8, 1.3))
    kde_contours(ax, apts, COL["variant"], lw=(0.8, 1.3))
    kde_contours(ax, jpts, COL["model"], lw=(1.1, 2.0), fill=True, zorder=5)

    ax.plot(0, 0, "s", color="k", ms=5.5, zorder=7)
    ax.annotate("TNG fiducial", (0, 0), textcoords="offset points",
                xytext=(5, 6), fontsize=7.5)
    ax.annotate("kSZ-only", (-0.66, 0.062), fontsize=7.5,
                color="#2a7fb5")
    ax.annotate("X-ray + kSZ", (-0.205, 0.072), fontsize=7.5,
                color="#a06c00")
    ax.annotate("joint\n(+ $\\kappa$-peaks $\\times$ y)", (-0.47, 0.008),
                fontsize=7.5, color=COL["model"], ha="center")
    ax.axhline(0, color=COL["ref"], lw=0.5, ls=":")
    ax.axvline(0, color=COL["ref"], lw=0.5, ls=":")
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}\;(M_{200c} > 10^{13.5}\,"
                  r"{\rm M_\odot})$")
    ax.set_ylabel(r"$\Delta \ln T_{\rm mw}$")
    ax.set_xlim(-0.82, 0.20)
    ax.set_ylim(-0.055, 0.105)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    condition_tag(ax, "fiducial-$\\theta$ frame · 68 / 95%")

    ax2.hist(sigma_pos, bins=34, color=COL["model"], alpha=0.55,
             density=True)
    ax2.axvline(3.6, color=COL["ref"], lw=0.8, ls=":")
    ax2.annotate("prior\nbound", (3.6, ax2.get_ylim()[1] * 0.82),
                 fontsize=7, ha="right", xytext=(-4, 0),
                 textcoords="offset points")
    ax2.set_xlabel(r"$\sigma_{\rm pos}$ [arcmin]")
    ax2.set_ylabel("posterior density")
    ax2.set_xlim(3.1, 3.65)

    fig.tight_layout()
    out = WP6 / "figures" / "joint_ab_plane_pub"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    if PLANS_FIG.exists():
        shutil.copy(f"{out}.png", PLANS_FIG / "joint_ab_plane_pub.png")
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
