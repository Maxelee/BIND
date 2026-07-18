"""WP-A5 recovery battery (plan task 2): inject-and-recover before any data fit.

For each of ``N_INJECT`` prior draws theta*, replace the data vector with
pred(theta*) + noise ~ N(0, diag sigma(theta*)^2), run the same MCMC the
data fit would use, and score:

- the rank of theta* within the marginal posterior of the CONSTRAINED
  directions (the f_gas data constrain ~2 combinations; unconstrained
  dims are prior-uniform and trivially calibrated — documented), summarized
  by the data-space rank of pred(theta*) per bin (rank-uniformity in the
  space the data actually live in);
- central-credible coverage at 68/95% for the derived group-bin f_gas
  summary f_group(theta) = pred(theta)[0] (the paper's reduced summary).

PASS criteria (pre-registered here, before the data fit):
- 68% coverage in [0.58, 0.78] and 95% in [0.88, 0.99] over the battery
  (binomial 1sigma bands for N=16);
- data-space rank distribution consistent with uniform (KS p > 0.01).

Output: /mnt/ceph/users/mlee1/paper3/A/wp5_chains/a5_recovery.json
Run (background or sbatch): python analysis/paper3a/scripts/run_a5_recovery.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, run_mcmc  # noqa: E402

N_INJECT = 16
N_WALKERS = 64
N_STEPS = 1500
N_BURN = 400


def main() -> None:
    rng = np.random.default_rng(42)
    base = FgasBlock()
    ranks, cov68, cov95 = [], [], []
    t0 = time.time()
    for j in range(N_INJECT):
        theta_star = rng.uniform(0.05, 0.95, 30)
        pred_star = base.predict(theta_star)[0]
        sig_star = base.sigma(pred_star[None, :])[0]
        blk = copy.copy(base)
        blk.data = copy.copy(base.data)
        blk.data.values = pred_star + rng.normal(0, sig_star)

        res = run_mcmc([blk], n_walkers=N_WALKERS, n_steps=N_STEPS,
                       n_burn=N_BURN, seed=1000 + j, label=f"recov{j:02d}",
                       archive=False)
        flat = res["flat"][:: max(1, len(res["flat"]) // 4000)]
        f_post = blk.predict(flat)[:, 0]                 # group-bin summary
        f_true = pred_star[0]
        ranks.append(float(np.mean(f_post < f_true)))
        lo68, hi68 = np.percentile(f_post, [16, 84])
        lo95, hi95 = np.percentile(f_post, [2.5, 97.5])
        cov68.append(bool(lo68 <= f_true <= hi68))
        cov95.append(bool(lo95 <= f_true <= hi95))
        print(f"inject {j:02d}: rank={ranks[-1]:.2f} in68={cov68[-1]} "
              f"in95={cov95[-1]} rhat_max={res['rhat'].max():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    ranks = np.array(ranks)
    from scipy.stats import kstest

    ks_p = float(kstest(ranks, "uniform").pvalue)
    out = {
        "n_inject": N_INJECT,
        "ranks_group_fgas": ranks.tolist(),
        "ks_uniform_p": ks_p,
        "coverage68": float(np.mean(cov68)),
        "coverage95": float(np.mean(cov95)),
        "pass_coverage68": bool(0.58 <= np.mean(cov68) <= 0.78),
        "pass_coverage95": bool(0.88 <= np.mean(cov95) <= 0.99),
        "pass_ks": bool(ks_p > 0.01),
        "settings": {"n_walkers": N_WALKERS, "n_steps": N_STEPS, "n_burn": N_BURN},
    }
    out["PASS"] = bool(out["pass_coverage68"] and out["pass_coverage95"] and out["pass_ks"])
    CHAINS.mkdir(parents=True, exist_ok=True)
    (CHAINS / "a5_recovery.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ranks_group_fgas"}, indent=2))


if __name__ == "__main__":
    main()
