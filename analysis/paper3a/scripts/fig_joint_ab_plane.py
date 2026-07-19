"""JOINT_AB step 4, the result figure: the joint A+B posterior in the
(dln M_gas, dln T) plane, against the A5 single-probe posteriors and
B's grid. Runs only after `run_joint_ab_fit.py --assemble` has written
joint_ab_summary.json (exits cleanly otherwise).

Everything is drawn in the fiducial-theta frame (B nodes shifted by the
run_0018 measured reference offset, as in fig_ab_synthesis).

Run: python analysis/paper3a/scripts/fig_joint_ab_plane.py
Out: wp6_propagation/figures/joint_ab_plane.png
     wp6_propagation/joint_ab_plane.json
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
from analysis.paper3a.inference.jointab import SIGMA_POS_MAX  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.scripts.fig_a5_tension import load_flat  # noqa: E402
from analysis.paper3a.scripts.run_ab_gate import (  # noqa: E402
    GRID, bin_weights, delta_coords)

N_MAP = 4000
PLANS_FIG = Path("/mnt/home/mlee1/bind-paper3-plans/figures_AB")
A5_CHAINS = {
    "fgas": ("a5_final_seed{k}.npz", "#0072B2", "f$_{gas}$-only"),
    "kszonly": ("a5_subset_kszonly_seed{k}.npz", "#56B4E9", "kSZ-only"),
    "joint": ("a5_subset_joint_seed{k}.npz", "#E69F00", "A joint"),
}


def contour_levels(h):
    s = np.sort(h.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    return [s[np.searchsorted(c, q)] for q in (0.95, 0.68)]


def draw_cloud(ax, pts, color, label, lw=(1.2, 2.2), bins=42):
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
    ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T,
               levels=contour_levels(H), colors=color, linewidths=lw)
    ax.plot([], [], color=color, lw=lw[1], label=label)


def main() -> None:
    summ_path = CHAINS / "joint_ab_summary.json"
    if not summ_path.exists():
        raise SystemExit("joint_ab_summary.json not written yet — run "
                         "after stage 4 of job the joint_ab job")
    summ = json.loads(summ_path.read_text())
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

    def coords_of(flat, ncol):
        sub = flat[rng.choice(len(flat), min(N_MAP, len(flat)),
                              replace=False)][:, :30]
        cc = delta_coords(emu, sub, u_fid, w)
        return np.stack([cc["dln_mgas"], cc["dln_t"]], axis=1), sub

    # joint A+B chains
    jflat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    jpts, _ = coords_of(jflat, 32)
    jsub = jflat[rng.choice(len(jflat), N_MAP, replace=False)]
    sigma_pos = jsub[:, 31] * SIGMA_POS_MAX

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.4, 1.0]})

    g = np.load(GRID, allow_pickle=True)
    nodes = np.stack([g["delta_ln_mgas"], g["delta_ln_t"]],
                     axis=1) - off[None, :]
    ax.scatter(nodes[:, 0], nodes[:, 1], s=16, facecolor="none",
               edgecolor="#777777", lw=0.6, label="B5 grid nodes")

    for name, (pattern, col, label) in A5_CHAINS.items():
        flat = load_flat(pattern)
        if flat is None:
            continue
        pts, _ = coords_of(flat, flat.shape[1])
        draw_cloud(ax, pts, col, f"A5 {label}", lw=(0.8, 1.4))
    draw_cloud(ax, jpts, "#D55E00", "JOINT A+B", lw=(1.6, 3.0))

    ax.plot(0, 0, "s", color="k", ms=7, label="fiducial θ", zorder=5)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$ (fiducial-θ frame)")
    ax.set_ylabel(r"$\Delta \ln T$")
    ax.set_title("The joint A+B posterior in the decomposition plane",
                 fontsize=11)
    ax.legend(fontsize=7, loc="lower left")

    ax2.hist(sigma_pos, bins=36, color="#D55E00", alpha=0.8, density=True)
    ax2.set_xlabel(r"$\sigma_{\rm pos}$ [arcmin]")
    ax2.set_title(r"$\sigma_{\rm pos}$ posterior (prior U[0, 3.6])",
                  fontsize=10)
    fig.tight_layout()
    figdir = WP6 / "figures"
    figdir.mkdir(exist_ok=True)
    out_png = figdir / "joint_ab_plane.png"
    fig.savefig(out_png, dpi=160)
    if PLANS_FIG.exists():
        shutil.copy(out_png, PLANS_FIG / "joint_ab_plane.png")

    med = np.median(jpts, axis=0)
    result = {
        "joint_coords_posterior": {
            "dln_mgas": {q: float(np.percentile(jpts[:, 0], p))
                         for q, p in (("p16", 16), ("p50", 50),
                                      ("p84", 84))},
            "dln_t": {q: float(np.percentile(jpts[:, 1], p))
                      for q, p in (("p16", 16), ("p50", 50), ("p84", 84))},
        },
        "sigma_pos_posterior": {q: float(np.percentile(sigma_pos, p))
                                for q, p in (("p16", 16), ("p50", 50),
                                             ("p84", 84))},
        "map_chi2_per_block": summ["map"]["chi2_per_block"],
        "posterior_predictive_p_b": summ["posterior_predictive_p_b"],
        "rhat_max": summ["chain"]["cross_chain_rhat_max"],
        "preregistered_outcome": (
            "joint exclusion" if summ["posterior_predictive_p_b"] < 0.01
            or summ["map"]["chi2_per_block"]["b_kappa_y_peaks"]["chi2"] > 20
            else "compatible corner found — report with equal standing"),
    }
    (WP6 / "joint_ab_plane.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
