"""A x B cross-check in matter-power units: what suppression does the
A-side calibrated posterior PREDICT, vs what the B-side data DEMANDS?

Both sides are pushed through the SAME external relation (van Daalen+20)
from the SAME underlying quantity, so the comparison is apples to apples
and the shared systematics cancel.

WHY THIS IS DEFENSIBLE (the convention audit that unblocked it).
An earlier pass declared this blocked because `scaling_f_gas`'s aperture
is undocumented and vd20 is calibrated on a SPHERICAL f_bar,500c. The
producer was then traced -- `bind.inference.stats.halo_scaling` -- and
the quantity is:

    f_gas_500c = M_gas / M_tot, both summed inside a CIRCULAR PROJECTED
    disc of radius 0.659 * R200c (the NFW c=5 approximation), on the
    per-halo composite patches.

So it is a projected-disc ratio, not a spherical fraction. That matters
for the ABSOLUTE vd20 numbers -- but NOT for this comparison, because:

  * p5 feeds the same `f_gas_500c` / `f_star_500c` columns to vd20, and
  * statsemu's scaling_f_gas is built from the same per-run
    `halo_scaling.npz` (manifest meta: runs_dir=bind_sb35/runs,
    scaling_snap=96). Verified numerically: statsemu at CAMELS fiducial
    gives (f_gas, f_star) = (0.1156, 0.0152) against the remeasured
    catalogue's global medians (0.1158, 0.0155).

Numerator and denominator are co-projected, so the ratio largely
cancels projection bias -- which is why p5's TNG f_bar ~ 0.82-0.92 is
physically sensible. Whatever residual projection systematic remains is
COMMON to both sides and cancels in the A-vs-B difference. The absolute
ΔP/P values inherit p5's caveats (including that M_eff sits below
vd20's stated 6e13 floor); the COMPARISON does not.

Do NOT apply the A-side CylToSph factor (~0.53) here: it corrects a
different cylindrical quantity (the wp2 operator-table f_gas, a
full-slab line-of-sight cylinder), not this per-halo disc ratio.
Conflating them would inject a spurious factor ~2.

Run: python analysis/paper3a/scripts/run_ab_pk_crosscheck.py
Out: wp6_propagation/ab_pk_crosscheck.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

P5 = Path("/mnt/home/mlee1/ceph/paper3/B/wp5_inference/"
          "p5_pk_feedback_implication.json")

# vd20 Fig.16 caption / footnote 11, k = 0.5 h/Mpc (p5 verified verbatim 3x)
VD20_D, VD20_E = -5.990, -0.5107
OB_OM = 0.156                 # p5's choice, kept for consistency
BIN = 2                       # [13.5, 13.75) M200c Msun/h
N_SUB = 4000


def dP_P_k05(ftilde):
    """vd20 at k=0.5: dP/P = -exp(d*ftilde_bar + e)."""
    return -np.exp(VD20_D * np.asarray(ftilde, float) + VD20_E)


def main() -> None:
    p5 = json.loads(P5.read_text())
    emu = StatsEmulator.load()

    # ---- A side ------------------------------------------------------
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)
    fg_f = float(np.atleast_1d(emu.predict(u_fid, "scaling_f_gas")).ravel()[BIN])
    fs_f = float(np.atleast_1d(emu.predict(u_fid, "scaling_f_star")).ravel()[BIN])
    ft_fid = (fg_f + fs_f) / OB_OM

    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    rng = np.random.default_rng(11)
    U = flat[rng.choice(len(flat), N_SUB, replace=False)][:, :30]
    fg = np.atleast_2d(emu.predict(U, "scaling_f_gas"))[:, BIN]
    fs = np.atleast_2d(emu.predict(U, "scaling_f_star"))[:, BIN]
    ft_post = (fg + fs) / OB_OM

    dp_fid = float(dP_P_k05(ft_fid))
    dp_post = dP_P_k05(ft_post)
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731

    # ---- B side (p5, remeasured catalog = the plan-specified one) -----
    b = p5["absolute_branch_BOTH_CATALOGS"]["remeasured_d1ce1133"][
        "per_bin_mass_matched"]["nu3"]
    b_run0 = p5["absolute_branch_BOTH_CATALOGS"]["run0000_15481cba"][
        "per_bin_mass_matched"]["nu3"]

    out = {
        "what": "A-side calibrated posterior vs B-side data, both through "
                "vd20 at k=0.5 from the same halo_scaling-derived quantity",
        "aperture_resolved": "f_gas_500c = M_gas/M_tot inside a CIRCULAR "
                             "PROJECTED disc of radius 0.659*R200c "
                             "(bind.inference.stats.halo_scaling); "
                             "co-projected ratio, shared by both sides",
        "vd20": {"d": VD20_D, "e": VD20_E, "k_h_per_Mpc": 0.5,
                 "Omega_b_over_Omega_m": OB_OM},
        "mass_bin": {"index": BIN, "log10_M200c_Msunh": [13.5, 13.75],
                     "p5_nu3_M_eff_Msun": b["M_eff_Msun"],
                     "note": "p5's nu3 M_eff = 4.44e13 Msun maps to "
                             "log10 M200c[Msun/h] = 13.62 via the net "
                             "+0.028 dex (h and 200c->500c) conversion, "
                             "i.e. the centre of this bin"},
        "A_side": {
            "ftilde_bar_fiducial": ft_fid,
            "ftilde_bar_posterior_p16_p50_p84":
                [q(ft_post, 16), q(ft_post, 50), q(ft_post, 84)],
            "dP_P_k0p5_fiducial": dp_fid,
            "dP_P_k0p5_posterior_p16_p50_p84":
                [q(dp_post, 16), q(dp_post, 50), q(dp_post, 84)]},
        "B_side_p5": {
            "remeasured": {"ftilde_bar_TNG": b["ftilde_bar_TNG"],
                           "ftilde_bar_data": b["ftilde_bar_data"],
                           "dP_P_k0p5_TNG": b["dP_P_k0p5_TNG"],
                           "dP_P_k0p5_data": b["dP_P_k0p5_data"]},
            "run0000": {"ftilde_bar_data": b_run0["ftilde_bar_data"],
                        "dP_P_k0p5_data": b_run0["dP_P_k0p5_data"]}},
    }
    a_med = q(dp_post, 50)
    b_dat = b["dP_P_k0p5_data"]

    # The GAS-RATIO route to the same shortfall, persisted here because
    # it had been quoted in prose (2.48x) with no artifact behind it --
    # it was computed ad hoc and never written down, which is exactly
    # the untraceable-number failure this project guards against.
    #   A: the calibrated posterior's gas relative to CAMELS fiducial
    #      = exp(dln M_gas) from joint_ab_summary (MIRROR frame, the
    #      frame in which the fiducial sits at 0 by construction)
    #   B: p5's R_fgas headline, data relative to TNG fiducial
    j = json.loads((Path("/mnt/ceph/users/mlee1/paper3/A/wp5_chains")
                    / "joint_ab_summary.json").read_text())
    dm = j["coords_posterior"]["dln_mgas"]
    a_ratio = float(np.exp(dm["p50"]))
    b_ratio = float(p5["R_fgas_headline_nu1_3_mean"])
    out["gas_ratio_route"] = {
        "A_posterior_over_fiducial": a_ratio,
        "A_p16_p84": [float(np.exp(dm["p16"])), float(np.exp(dm["p84"]))],
        "B_data_over_TNG": b_ratio,
        "shortfall_factor": float(a_ratio / b_ratio),
        "note": "A is exp(dln M_gas) in the MIRROR frame (CAMELS fiducial "
                "at 0); B is p5's nu1-3 mean R_fgas. Different reference "
                "points, both ratios, so the comparison is of how far each "
                "sits below its own TNG baseline."}

    out["comparison"] = {
        "A_posterior_predicts_dP_P": a_med,
        "B_data_demands_dP_P": b_dat,
        "ratio_demand_over_prediction": float(b_dat / a_med),
        "gap_in_dP_P": float(b_dat - a_med),
        "shortfall_power_route": float(b_dat / a_med),
        "shortfall_gas_ratio_route": float(a_ratio / b_ratio)}

    (WP6 / "ab_pk_crosscheck.json").write_text(json.dumps(out, indent=2))

    print("vd20 @ k=0.5, mass bin [13.5,13.75) M200c Msun/h "
          f"(p5 nu3 M_eff = {b['M_eff_Msun']:.3g} Msun)\n")
    print(f"  A fiducial (CAMELS theta)  ftilde_bar {ft_fid:6.3f}   "
          f"dP/P {dp_fid:+7.4f}")
    print(f"  A joint posterior          ftilde_bar "
          f"{q(ft_post,50):6.3f}   dP/P {a_med:+7.4f}   "
          f"[{q(dp_post,16):+.4f}, {q(dp_post,84):+.4f}]")
    print(f"  B TNG   (p5, remeasured)   ftilde_bar "
          f"{b['ftilde_bar_TNG']:6.3f}   dP/P {b['dP_P_k0p5_TNG']:+7.4f}")
    print(f"  B DATA  (p5, remeasured)   ftilde_bar "
          f"{b['ftilde_bar_data']:6.3f}   dP/P {b_dat:+7.4f}")
    print(f"  B DATA  (p5, run0000)      ftilde_bar "
          f"{b_run0['ftilde_bar_data']:6.3f}   dP/P "
          f"{b_run0['dP_P_k0p5_data']:+7.4f}")
    print(f"\n  THE GAP: the data demand {b_dat/a_med:.1f}x the power "
          f"suppression the calibrated posterior can deliver")
    print(f"  ({b_dat*100:+.1f}% demanded vs {a_med*100:+.1f}% predicted "
          f"at k=0.5 h/Mpc)")
    print(f"\nwrote {WP6}/ab_pk_crosscheck.json")


if __name__ == "__main__":
    main()
