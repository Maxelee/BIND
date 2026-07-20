"""A×B cross-check: does the INDEPENDENT 253-run Sobol box reject the
region the joint A+B posterior actually occupies?

Why this is not circular. The joint fit's B block (`bblock.py`) is a GP
trained on the **60-unit twobound** grid; it has never seen the 253-run
SB35 Sobol grid. B5's Sobol verdict (`b5_sobol_verdict.json`) is an
independent, pre-registered per-unit chi2 of the same frozen data
vector against those 253 units. So asking "what chi2 do the Sobol units
nearest the joint posterior carry?" tests the headline exclusion
against a grid that played no part in producing it.

FRAME (the trap this script exists to get right). The A-side mirror
coordinates are deltas against the CAMELS fiducial theta, so fiducial
theta sits at (0,0) by construction. The B grid coordinates are deltas
against the bind reference, in which the fiducial-theta node sits at
(-0.0387, +0.0143) -- the reference offset caused by the bind fiducial
having been painted at CAMELS-CV cosmology. The map is therefore
    c_B = c_mirror + TWOBOUND_REF_OFFSET
and it is VERIFIED here at runtime against the value stored inside
sb35_coords.npz rather than trusted (they agree to 3.7e-5).

Run: python analysis/paper3a/scripts/run_ab_sobol_crosscheck.py
Out: wp6_propagation/ab_sobol_crosscheck.json (+ figures/)
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

from analysis.paper3a.inference.bblock import TWOBOUND_REF_OFFSET  # noqa: E402
from analysis.paper3a.inference.jointab import JointBBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6  # noqa: E402

BROOT = Path("/mnt/home/mlee1/ceph/paper3/B/wp4_mocks/sobol")
GRID = BROOT / "model_grid_tfwiener_sb35.npz"
VERDICT = BROOT / "b5_sobol_verdict.json"
COORDS = BROOT / "sb35_coords.npz"
N_SUB = 8_000
CHI2_OK = 100.0          # B5's pre-registered "not rejected" threshold


def main() -> None:
    for p in (GRID, VERDICT, COORDS):
        if not p.exists():
            raise SystemExit(f"missing input {p} — rsync from Popeye first")

    c = np.load(COORDS, allow_pickle=True)
    off_stored = np.asarray(
        c["fiducial_theta_node_twobound_run_0018_delta"], float)
    if not np.allclose(off_stored, TWOBOUND_REF_OFFSET, atol=5e-5):
        raise SystemExit(
            f"FRAME MISMATCH: sb35_coords says {off_stored}, bblock says "
            f"{TWOBOUND_REF_OFFSET} — refusing to compare across frames")

    v = json.loads(VERDICT.read_text())
    per_unit = v["per_unit_chi2"]
    runs = [str(r) for r in c["run"]]
    cb = np.column_stack([np.asarray(c["delta_ln_mgas"], float),
                          np.asarray(c["delta_ln_t"], float)])
    chi2 = np.array([per_unit.get(f"sb35/{r}", np.nan) for r in runs])
    ok = np.isfinite(chi2)

    # joint posterior -> grid frame
    jb = JointBBlock()
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    rng = np.random.default_rng(5)
    U = flat[rng.choice(len(flat), N_SUB, replace=False)]
    cm, _ = jb.coords(U[:, :30])
    cg = cm + TWOBOUND_REF_OFFSET[None, :]      # mirror -> grid frame

    med = np.median(cg, axis=0)
    lo, hi = np.percentile(cg, [16, 84], axis=0)
    lo95, hi95 = np.percentile(cg, [2.5, 97.5], axis=0)

    # scale distances by the posterior width so "nearby" is meaningful
    w = 0.5 * (hi - lo)
    d = np.sqrt((((cb - med[None, :]) / w[None, :]) ** 2).sum(axis=1))

    inside68 = ok & np.all((cb >= lo) & (cb <= hi), axis=1)
    inside95 = ok & np.all((cb >= lo95) & (cb <= hi95), axis=1)
    order = np.argsort(np.where(ok, d, np.inf))

    nearest = [{"run": runs[i], "coords": cb[i].tolist(),
                "dist_in_posterior_sigmas": float(d[i]),
                "chi2": float(chi2[i])} for i in order[:5]]

    out = {
        "what": "chi2 of the INDEPENDENT 253-run Sobol units nearest the "
                "joint A+B posterior; the joint fit's B block is a GP on "
                "the 60-unit twobound grid and never saw these units",
        "frame": {"map": "c_B = c_mirror + TWOBOUND_REF_OFFSET",
                  "offset": TWOBOUND_REF_OFFSET.tolist(),
                  "verified_against_sb35_coords": True,
                  "max_abs_diff": float(np.abs(off_stored
                                               - TWOBOUND_REF_OFFSET).max())},
        "joint_posterior_grid_frame": {
            "median": med.tolist(), "p16": lo.tolist(), "p84": hi.tolist(),
            "p2p5": lo95.tolist(), "p97p5": hi95.tolist()},
        "sobol_chi2_threshold_not_rejected": CHI2_OK,
        "n_units_inside_68": int(inside68.sum()),
        "n_units_inside_95": int(inside95.sum()),
        "min_chi2_inside_68": (float(chi2[inside68].min())
                               if inside68.any() else None),
        "min_chi2_inside_95": (float(chi2[inside95].min())
                               if inside95.any() else None),
        "nearest_units": nearest,
        "sobol_global": {"chi2_min": v["chi2_min"], "at": v["chi2_min_at"],
                         "coords": v["chi2_min_coords"],
                         "median": v["chi2_median"],
                         "n_below_100": v["n_below_100"]},
    }
    (WP6 / "ab_sobol_crosscheck.json").write_text(json.dumps(out, indent=2))

    print("frame verified: c_B = c_mirror + "
          f"{TWOBOUND_REF_OFFSET} (max diff "
          f"{out['frame']['max_abs_diff']:.1e})")
    print(f"\njoint posterior (GRID frame): dlnMgas "
          f"{med[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]   "
          f"dlnT {med[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]")
    print(f"Sobol global best: chi2 {v['chi2_min']:.1f}/4 at "
          f"{v['chi2_min_at']} coords "
          f"[{v['chi2_min_coords'][0]:+.3f}, {v['chi2_min_coords'][1]:+.3f}]")
    print(f"\nSobol units inside the joint 68% box: {inside68.sum()}"
          f"   inside 95%: {inside95.sum()}")
    if inside95.any():
        print(f"  min chi2 among them: {chi2[inside95].min():.1f}/4")
    print("\nnearest Sobol units to the joint median "
          "(distance in posterior sigmas):")
    for n in nearest:
        print(f"  {n['run']:12s} d={n['dist_in_posterior_sigmas']:5.2f}  "
              f"coords [{n['coords'][0]:+.3f}, {n['coords'][1]:+.3f}]  "
              f"chi2 {n['chi2']:7.1f}/4"
              + ("   <-- NOT REJECTED" if n["chi2"] < CHI2_OK else ""))

    # ---- figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    s = ax.scatter(cb[ok, 0], cb[ok, 1], c=np.log10(chi2[ok]), s=26,
                   cmap="viridis", edgecolor="none")
    plt.colorbar(s, ax=ax, label=r"$\log_{10}\chi^2$ (B5 Sobol, /4)")
    ax.plot(cg[::40, 0], cg[::40, 1], ".", ms=1.2, color="#D55E00",
            alpha=0.25, label="joint A+B posterior")
    ax.plot(*med, "*", ms=15, color="#D55E00", mec="k", label="joint median")
    ax.plot(v["chi2_min_coords"][0], v["chi2_min_coords"][1], "s", ms=8,
            color="w", mec="k", label=f"Sobol best ({v['chi2_min']:.0f}/4)")
    ax.set_xlabel(r"$\Delta \ln M_{\rm gas}$ (grid frame)")
    ax.set_ylabel(r"$\Delta \ln T$ (grid frame)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    (WP6 / "figures").mkdir(exist_ok=True)
    fig.savefig(WP6 / "figures" / "ab_sobol_crosscheck.png", dpi=150)
    print(f"\nwrote {WP6}/ab_sobol_crosscheck.json + figures/")


if __name__ == "__main__":
    main()
