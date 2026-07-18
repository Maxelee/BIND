"""WP-A5 recovery batteries for the kSZ-carrying likelihoods (plan task 2
discipline applied to task 5): inject-and-recover BEFORE any data fit.

Same pre-registered design as `run_a5_recovery.py` (16 injections, amended
binomial coverage criteria + KS rank uniformity — the 2026-07-18 amendment
is inherited, not re-litigated), extended to the 31-dim chains (theta + the
f_sat nuisance) and to multivariate kSZ noise draws from the full per-theta
covariance.

Scored summaries (data-space, the convention the first battery set):
- kszonly: T_kSZ at the smallest aperture (1'), `KszBlock.predict[:, 0]`.
- joint:   BOTH the group-bin f_gas summary and T_kSZ(1'); PASS requires
           every criterion to pass for both summaries.

Modes (disBatch-friendly: each injection is an independent single-core task):
  --which kszonly|joint                 run all 16 injections serially
  --which ... --inject-index J          run injection J only -> parts json
  --which ... --assemble                score the 16 parts -> final json

Out: /mnt/ceph/users/mlee1/paper3/A/wp5_chains/a5_recovery_{which}.json
     (parts in a5_recovery_{which}_parts/)
"""

from __future__ import annotations

import argparse
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
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, run_mcmc  # noqa: E402

N_INJECT = 16
N_WALKERS = 64
N_STEPS = 1500
N_BURN = 400
NDIM = 31


def _blocks(which):
    ksz = KszBlock()
    if which == "joint":
        return [FgasBlock(emu=ksz.emu), ksz]
    return [ksz]


def _run_injection(which, blocks, j) -> dict:
    """One inject-and-recover; the rng is seeded per injection so the
    serial and disBatch paths draw identical theta*/noise."""
    rng = np.random.default_rng(4200 + j)
    theta_star = rng.uniform(0.05, 0.95, NDIM)

    ksz_base = blocks[-1]
    fgas_base = blocks[0] if which == "joint" else None

    ksz = copy.copy(ksz_base)
    pred_star = ksz_base.predict(theta_star)[0]
    cov_star = ksz_base.cov(pred_star[None, :])[0]
    ksz.values = pred_star + rng.multivariate_normal(np.zeros(len(pred_star)),
                                                     cov_star)
    fit_blocks = [ksz]
    fpred_star = None
    if which == "joint":
        fg = copy.copy(fgas_base)
        fg.data = copy.copy(fgas_base.data)
        fpred_star = fgas_base.predict(theta_star)[0]
        fsig_star = fgas_base.sigma(fpred_star[None, :])[0]
        fg.data.values = fpred_star + rng.normal(0, fsig_star)
        fit_blocks = [fg, ksz]

    res = run_mcmc(fit_blocks, n_walkers=N_WALKERS, n_steps=N_STEPS,
                   n_burn=N_BURN, seed=1000 + j,
                   label=f"recov_{which}{j:02d}", archive=False, ndim=NDIM)
    flat = res["flat"][:: max(1, len(res["flat"]) // 4000)]

    out = {"inject": j, "rhat_max": float(res["rhat"].max()), "summaries": {}}
    t_post = ksz_base.predict(flat)[:, 0]
    t_true = float(pred_star[0])
    lo68, hi68 = np.percentile(t_post, [16, 84])
    lo95, hi95 = np.percentile(t_post, [2.5, 97.5])
    out["summaries"]["tksz1"] = {
        "rank": float(np.mean(t_post < t_true)),
        "in68": bool(lo68 <= t_true <= hi68),
        "in95": bool(lo95 <= t_true <= hi95),
    }
    if which == "joint":
        f_post = fgas_base.predict(flat)[:, 0]
        f_true = float(fpred_star[0])
        lo68, hi68 = np.percentile(f_post, [16, 84])
        lo95, hi95 = np.percentile(f_post, [2.5, 97.5])
        out["summaries"]["f_group"] = {
            "rank": float(np.mean(f_post < f_true)),
            "in68": bool(lo68 <= f_true <= hi68),
            "in95": bool(lo95 <= f_true <= hi95),
        }
    return out


def _score(name, ranks, cov68, cov95):
    from scipy.stats import binomtest, kstest

    ranks = np.asarray(ranks)
    ks_p = float(kstest(ranks, "uniform").pvalue)
    p68 = float(binomtest(int(np.sum(cov68)), N_INJECT, 0.68).pvalue)
    p95 = float(binomtest(int(np.sum(cov95)), N_INJECT, 0.95).pvalue)
    return {
        "summary": name,
        "ranks": ranks.tolist(),
        "ks_uniform_p": ks_p,
        "coverage68": float(np.mean(cov68)),
        "coverage95": float(np.mean(cov95)),
        "binom_p68": p68, "binom_p95": p95,
        "pass_coverage68": bool(p68 > 0.01),
        "pass_coverage95": bool(p95 > 0.01),
        "pass_ks": bool(ks_p > 0.01),
    }


def _assemble(which, parts_dir: Path) -> None:
    parts = []
    for j in range(N_INJECT):
        p = parts_dir / f"inj{j:02d}.json"
        if not p.exists():
            raise SystemExit(f"missing {p} — battery incomplete, not scoring")
        parts.append(json.loads(p.read_text()))
    names = list(parts[0]["summaries"])
    scored = []
    for name in names:
        s = [p["summaries"][name] for p in parts]
        scored.append(_score(name, [x["rank"] for x in s],
                             [x["in68"] for x in s], [x["in95"] for x in s]))
    out = {
        "which": which,
        "n_inject": N_INJECT,
        "ndim": NDIM,
        "summaries": scored,
        "rhat_max_worst": max(p["rhat_max"] for p in parts),
        "settings": {"n_walkers": N_WALKERS, "n_steps": N_STEPS, "n_burn": N_BURN},
        "criteria": "amended binomial (p>0.01) + KS rank uniformity (p>0.01), "
                    "per run_a5_recovery.py 2026-07-18 amendment",
    }
    out["PASS"] = bool(all(s["pass_coverage68"] and s["pass_coverage95"]
                           and s["pass_ks"] for s in scored))
    path = CHAINS / f"a5_recovery_{which}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "summaries"}, indent=2))
    for s in scored:
        print(json.dumps({k: v for k, v in s.items() if k != "ranks"}, indent=2))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("kszonly", "joint"), required=True)
    ap.add_argument("--inject-index", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    parts_dir = CHAINS / f"a5_recovery_{args.which}_parts"
    if args.assemble:
        _assemble(args.which, parts_dir)
        return

    parts_dir.mkdir(parents=True, exist_ok=True)
    blocks = _blocks(args.which)
    idx = [args.inject_index] if args.inject_index is not None else range(N_INJECT)
    t0 = time.time()
    for j in idx:
        out = _run_injection(args.which, blocks, j)
        (parts_dir / f"inj{j:02d}.json").write_text(json.dumps(out, indent=2))
        print(f"inject {j:02d}: " + " ".join(
            f"{k}(rank={v['rank']:.2f} 68={v['in68']} 95={v['in95']})"
            for k, v in out["summaries"].items())
            + f" rhat_max={out['rhat_max']:.3f} ({time.time()-t0:.0f}s)",
            flush=True)
    if args.inject_index is None:
        _assemble(args.which, parts_dir)


if __name__ == "__main__":
    main()
