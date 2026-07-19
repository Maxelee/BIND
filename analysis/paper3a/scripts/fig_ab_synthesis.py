"""JOINT_AB_PLAN product 1: the A x B synthesis figure — both projects in
the one physical plane (Delta ln M_gas, Delta ln T).

Runs ONLY if the step-2 gate json exists with PASS (pre-registered order).

- B side: the 60 twobound grid nodes at their B4-MEASURED coordinates,
  colored by an approximate B5 chi2 against the frozen Wiener vector
  (the 4 science nu bins at the 4' headline radius; sigma^2 = frozen
  jackknife variance + decomposed Sigma_sys + the node's grid-MC error;
  diagonal). Explicitly labeled approximate — the real B5 fit is the
  Popeye GP chain; this coloring only orients the plane.
- A side: the three A5 posteriors (f_gas-only FINAL, kSZ-only, joint)
  mapped through the GATE-VALIDATED statsemu mirror (same weights), as
  68/95% contours. TNG fiducial at the origin by construction.

The pre-registered questions (plan step 3) are answered in the emitted
json: (a) does the A5-favored region lie in B's preferred (lower-chi2)
direction? (b) does B demand MORE evacuation than even kSZ-only?
(c) same failure axis?

Run: python analysis/paper3a/scripts/fig_ab_synthesis.py
Out: wp6_propagation/ab_synthesis.json, figures/ab_synthesis.png
     (+ copy of the png into the plans repo figures_AB/)
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

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.scripts.fig_a5_tension import load_flat  # noqa: E402
from analysis.paper3a.scripts.run_ab_gate import (  # noqa: E402
    GRID, bin_weights, delta_coords,
)

B = Path("/mnt/ceph/users/mlee1/paper3/B")
N_MAP = 2000
CHAINS = {
    "fgas": ("a5_final_seed{k}.npz", "#0072B2", "f$_{gas}$-only"),
    "kszonly": ("a5_subset_kszonly_seed{k}.npz", "#56B4E9", "kSZ-only"),
    "joint": ("a5_subset_joint_seed{k}.npz", "#E69F00", "joint"),
}


def contour_levels(h):
    s = np.sort(h.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    return [s[np.searchsorted(c, q)] for q in (0.95, 0.68)]


def main() -> None:
    gate = json.loads((WP6 / "ab_gate.json").read_text())
    if not gate.get("PASS"):
        raise SystemExit("step-2 gate did not PASS — product 1 is blocked")

    # ---- B side: nodes + approximate chi2 ------------------------------
    g = np.load(GRID, allow_pickle=True)
    meas = np.stack([g["delta_ln_mgas"], g["delta_ln_t"]], axis=1)
    y_nodes = np.asarray(g["y_mean"], float)[:, 1:5, 2]      # (60, 4) at 4'
    mc = np.asarray(g["y_mc_err"], float)[:, 1:5, 2]
    stack = np.load(B / "wp2_measurement" / "stack_wiener_sm2am_fid.npz",
                    allow_pickle=True)
    data = np.asarray(stack["y_mean"], float)[1:5, 2]
    ycov = np.asarray(stack["y_cov"], float)
    var_stat = (np.diag(ycov.reshape(25, 25)).reshape(5, 5)[1:5, 2]
                if ycov.size == 625 else np.asarray(stack["y_sigma_jk16"],
                                                    float)[1:5, 2] ** 2)
    sysn = np.load(B / "wp3_nulls" / "sigma_sys_wiener_decomposed.npz",
                   allow_pickle=True)
    skey = [k for k in sysn.files if "sigma" in k or "sys" in k]
    sig_sys = np.asarray(sysn[skey[0]], float)
    sig_sys = sig_sys[1:5, 2] if sig_sys.ndim == 2 else sig_sys[1:5]
    var = var_stat + sig_sys**2
    chi2 = np.array([np.sum((y - data) ** 2 / (var + e**2))
                     for y, e in zip(y_nodes, mc)])

    # ---- A side: posteriors through the validated mirror ---------------
    emu = StatsEmulator.load()
    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz", allow_pickle=False)
    edges = np.asarray(json.loads(str(raw["manifest"]))["meta"]["mass_bins"], float)
    w = bin_weights(edges)
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    rng = np.random.default_rng(7)
    clouds = {}
    for name, (pattern, col, label) in CHAINS.items():
        flat = load_flat(pattern)
        if flat is None:
            raise SystemExit(f"{name} chains missing")
        sub = flat[rng.choice(len(flat), N_MAP, replace=False)][:, :30]
        cc = delta_coords(emu, sub, u_fid, w)
        clouds[name] = np.stack([cc["dln_mgas"], cc["dln_t"]], axis=1)

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(meas[:, 0], meas[:, 1], c=np.log10(chi2), s=40,
                    cmap="Greys_r", edgecolor="k", linewidth=0.4, zorder=3,
                    label=None)
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}\chi^2_{\rm B5,approx}$ (4 bins @ 4')")
    imin = int(np.argmin(chi2))
    ax.plot(*meas[imin], marker="*", ms=16, color="#D55E00", zorder=5,
            label=f"B min-$\\chi^2$ node ({chi2[imin]:.0f}/4)")

    for name, (_, col, label) in CHAINS.items():
        pts = clouds[name]
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=40)
        lv = contour_levels(H)
        ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]),
                   H.T, levels=lv, colors=col, linewidths=(1.0, 2.0))
        ax.plot([], [], color=col, lw=2, label=f"A5 {label}")

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.plot(0, 0, "s", color="#009E73", ms=8, label="TNG fiducial", zorder=5)
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$ (halo-matched, $M > 10^{13.5}$)")
    ax.set_ylabel(r"$\Delta \ln T$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("A x B synthesis: gas posteriors vs the kappa-peak y likelihood",
                 fontsize=10)
    fig.tight_layout()
    (WP6 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP6 / "figures" / "ab_synthesis.png", dpi=170)

    # ---- pre-registered answers ---------------------------------------
    dirvec = meas[imin] / np.linalg.norm(meas[imin])
    summary = {"gate": gate["gate"],
               "b_min_chi2": {"value": float(chi2[imin]),
                              "coords": meas[imin].tolist(),
                              "run": str(np.asarray(g["run_names"])[imin])},
               "chi2_at_fiducial_nodeless": "fiducial not a grid node",
               "answers": {}}
    for name, pts in clouds.items():
        med = np.median(pts, axis=0)
        proj = float(np.dot(med, dirvec))
        summary["answers"][name] = {
            "median_coords": med.tolist(),
            "projection_on_B_min_direction": proj,
            "toward_B_direction": bool(proj > 0),
        }
    (WP6 / "ab_synthesis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {WP6}/figures/ab_synthesis.png")


if __name__ == "__main__":
    main()
