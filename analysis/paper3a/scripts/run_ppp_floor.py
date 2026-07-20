"""TK #17: what is actually behind the B-block "p_pp = 0.000"?

`run_joint_ab_fit.assemble` estimates the posterior-predictive p by
Monte Carlo with **200 draws** (the count was not stored in
joint_ab_summary.json, which is why the draft could not state a floor).
200 draws puts a hard floor at 1/200 = 5e-3, so "0.000" to three
decimals claims 20x more precision than that estimator can deliver.
The honest MC statement is p_pp < 5e-3.

But the MC step is unnecessary here, and dropping it removes the floor
entirely. In `assemble` the replication is

    sim = y + chol(C) @ z,  z ~ N(0, I_4)
    chi2_sim = (sim - y)^T C^-1 (sim - y) = z^T z

so conditional on theta, chi2_sim is EXACTLY chi2 with 4 dof --
independent of y, C and the data. The posterior-predictive p is
therefore

    p_pp = E_theta[ P(chi2_4 >= chi2_obs(theta)) ]
         = E_theta[ sf(chi2_obs(theta), 4) ]

which is the Rao-Blackwellised version of the same estimator: exact in
the replication dimension, with only the posterior average left to
converge. It cannot floor at 1/N, and it is strictly lower-variance
than the MC form. This script computes it, and reproduces the 200-draw
MC number alongside as a consistency check.

Run: python analysis/paper3a/scripts/run_ppp_floor.py
Out: wp8_robustness/a8_ppp_floor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.bblock import (  # noqa: E402
    OFFSET_SYS, SLOPE_SYS, TWOBOUND_REF_OFFSET)
from analysis.paper3a.inference.jointab import (  # noqa: E402
    SIGMA_POS_MAX, JointBBlock)
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

WP8 = Path("/mnt/ceph/users/mlee1/paper3/A/wp8_robustness")
NDIM = 32
N_POST = 20_000     # posterior draws for the outer average
DOF_B = 4
MC_N = 200          # what assemble() used, reproduced for comparison


def main() -> None:
    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, NDIM) for k in range(4)])
    rng = np.random.default_rng(5)
    U = flat[rng.choice(len(flat), N_POST, replace=False)]

    jb = JointBBlock()
    print(f"coords on {N_POST} posterior draws ...", flush=True)
    c, e = jb.coords(U[:, :30])          # batched; the expensive step

    chi2_obs = np.empty(len(U))
    for i, u in enumerate(U):
        cB = c[i] + TWOBOUND_REF_OFFSET
        cvar = e[i] ** 2 + (SLOPE_SYS * cB) ** 2 + OFFSET_SYS ** 2
        chi2_obs[i] = jb.b.chi2(cB, u[31] * SIGMA_POS_MAX,
                                coord_var=cvar)[0]

    # exact in the replication dimension
    p_cond = stats.chi2.sf(chi2_obs, DOF_B)
    p_pp = float(p_cond.mean())
    # MC error on the outer average only
    se = float(p_cond.std(ddof=1) / np.sqrt(len(p_cond)))

    # the 200-draw MC estimator, reproduced
    z = rng.normal(size=(MC_N, DOF_B))
    mc = float(np.mean((z ** 2).sum(1) >= chi2_obs[:MC_N]))

    out = {
        "what": "B-block posterior-predictive p, computed exactly in the "
                "replication dimension (chi2_sim | theta ~ chi2_4 by "
                "construction), so there is no 1/N Monte-Carlo floor",
        "n_posterior_draws": N_POST,
        "dof_b": DOF_B,
        "p_pp_b_exact": p_pp,
        "p_pp_b_exact_se": se,
        "log10_p_pp_b": float(np.log10(p_pp)) if p_pp > 0 else None,
        "chi2_obs_percentiles": {
            q: float(np.percentile(chi2_obs, q)) for q in (1, 16, 50, 84, 99)},
        "p_cond_percentiles": {
            q: float(np.percentile(p_cond, q)) for q in (1, 16, 50, 84, 99)},
        "mc_reference": {
            "n_draws": MC_N,
            "p_pp_mc": mc,
            "floor": 1.0 / MC_N,
            "note": "this is what assemble() reports as 0.000; its floor is "
                    "1/200 = 5e-3, so the honest MC statement was p_pp < "
                    "5e-3 and the three-decimal 0.000 overstated it",
        },
        "reporting_guidance": "quote the exact value (or a bound like "
                              "p_pp < 1e-4) rather than '0.000'; the "
                              "rejection does not rest on the draw count",
    }
    WP8.mkdir(exist_ok=True)
    (WP8 / "a8_ppp_floor.json").write_text(json.dumps(out, indent=2))

    print(f"chi2_obs median {np.median(chi2_obs):.1f} / {DOF_B} dof")
    print(f"p_pp exact = {p_pp:.3e}  (se {se:.1e})")
    print(f"p_pp MC({MC_N}) = {mc:.3f}   floor {1.0/MC_N:.1e}")
    print(f"wrote {WP8}/a8_ppp_floor.json")


if __name__ == "__main__":
    main()
