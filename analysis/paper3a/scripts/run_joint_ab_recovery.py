"""JOINT_AB_PLAN step 4: the joint A+B recovery battery — inject theta*
through BOTH likelihoods BEFORE the data fit (house rule).

Design conforms to the inherited A5 battery convention
(`run_a5_recovery_subsets.py`: 16 injections theta* ~ U(0.05,0.95)^ndim,
amended binomial coverage + KS rank-uniformity criteria, 2026-07-18
amendment inherited), extended to 32 dims (30 astro + f_sat + sigma_pos)
and to the B block: synthetic B data are drawn from N(model_B(theta*),
C_B(theta*)) with the SAME covariance the likelihood uses (incl. the
coordinate-systematic tier), so the test checks calibration of the whole
pre-registered error model.

Scored summaries: tksz1 + f_group (the A5 pair) + dln_mgas (the mirror
coordinate — the quantity L_B consumes) + sigma_pos (the new nuisance).

Modes (disBatch-friendly):
  --inject-index J   run injection J -> parts json
  --assemble         score the 16 parts -> a5_recovery_jointab.json

Out: /mnt/ceph/users/mlee1/paper3/A/wp5_chains/a5_recovery_jointab.json
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
from analysis.paper3a.inference.jointab import SIGMA_POS_MAX, JointBBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, run_mcmc  # noqa: E402
from analysis.paper3a.scripts.run_a5_recovery_subsets import _score  # noqa: E402

N_INJECT = 16
N_WALKERS = 64
N_STEPS = 1500
N_BURN = 400
NDIM = 32
WHICH = "jointab"


def _blocks():
    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)
    jb = JointBBlock()
    return fgas, ksz, jb


def _run_injection(blocks, j) -> dict:
    rng = np.random.default_rng(5200 + j)
    theta_star = rng.uniform(0.05, 0.95, NDIM)
    fgas_base, ksz_base, jb_base = blocks

    ksz = copy.copy(ksz_base)
    kpred = ksz_base.predict(theta_star)[0]
    kcov = ksz_base.cov(kpred[None, :])[0]
    ksz.values = kpred + rng.multivariate_normal(np.zeros(len(kpred)), kcov)

    fg = copy.copy(fgas_base)
    fg.data = copy.copy(fgas_base.data)
    fpred = fgas_base.predict(theta_star)[0]
    fsig = fgas_base.sigma(fpred[None, :])[0]
    fg.data.values = fpred + rng.normal(0, fsig)

    jb = copy.copy(jb_base)
    jb.b = copy.copy(jb_base.b)
    _, ystar, Cstar = jb_base.chi2(theta_star)
    jb.b.data = ystar + rng.multivariate_normal(np.zeros(4), Cstar)

    res = run_mcmc([fg, ksz, jb], n_walkers=N_WALKERS, n_steps=N_STEPS,
                   n_burn=N_BURN, seed=2000 + j,
                   label=f"recov_{WHICH}{j:02d}", archive=False, ndim=NDIM)
    flat = res["flat"][:: max(1, len(res["flat"]) // 4000)]

    out = {"inject": j, "rhat_max": float(res["rhat"].max()), "summaries": {}}

    def score_one(name, post, true):
        lo68, hi68 = np.percentile(post, [16, 84])
        lo95, hi95 = np.percentile(post, [2.5, 97.5])
        out["summaries"][name] = {
            "rank": float(np.mean(post < true)),
            "in68": bool(lo68 <= true <= hi68),
            "in95": bool(lo95 <= true <= hi95),
        }

    score_one("tksz1", ksz_base.predict(flat)[:, 0],
              float(kpred[0]))
    score_one("f_group", fgas_base.predict(flat)[:, 0], float(fpred[0]))
    c_true = jb_base.coords(theta_star[None, :30])[0][0]
    c_post = jb_base.coords(flat[:, :30])[0]
    score_one("dln_mgas", c_post[:, 0], float(c_true[0]))
    score_one("sigma_pos", flat[:, 31] * SIGMA_POS_MAX,
              float(theta_star[31] * SIGMA_POS_MAX))
    return out


def _assemble(parts_dir: Path) -> None:
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
        "which": WHICH,
        "n_inject": N_INJECT,
        "ndim": NDIM,
        "summaries": scored,
        "rhat_max_worst": max(p["rhat_max"] for p in parts),
        "settings": {"n_walkers": N_WALKERS, "n_steps": N_STEPS,
                     "n_burn": N_BURN},
        "criteria": "amended binomial (p>0.01) + KS rank uniformity "
                    "(p>0.01), per run_a5_recovery.py 2026-07-18 "
                    "amendment; B data drawn through C_B(theta*) incl. "
                    "the coordinate-systematic tier",
    }
    out["PASS"] = bool(all(s["pass_coverage68"] and s["pass_coverage95"]
                           and s["pass_ks"] for s in scored))
    path = CHAINS / f"a5_recovery_{WHICH}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "summaries"},
                     indent=2))
    for s in scored:
        print(json.dumps({k: v for k, v in s.items() if k != "ranks"},
                         indent=2))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-index", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    parts_dir = CHAINS / f"a5_recovery_{WHICH}_parts"
    if args.assemble:
        _assemble(parts_dir)
        return

    parts_dir.mkdir(parents=True, exist_ok=True)
    blocks = _blocks()
    idx = ([args.inject_index] if args.inject_index is not None
           else range(N_INJECT))
    t0 = time.time()
    for j in idx:
        out = _run_injection(blocks, j)
        (parts_dir / f"inj{j:02d}.json").write_text(json.dumps(out, indent=2))
        print(f"inject {j:02d}: " + " ".join(
            f"{k}(rank={v['rank']:.2f} 68={v['in68']} 95={v['in95']})"
            for k, v in out["summaries"].items())
            + f" rhat_max={out['rhat_max']:.3f} ({time.time()-t0:.0f}s)",
            flush=True)
    if args.inject_index is None:
        _assemble(parts_dir)


if __name__ == "__main__":
    main()
