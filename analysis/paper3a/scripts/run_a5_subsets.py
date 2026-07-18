"""WP-A5 task 5: the probe-subset data fits (kSZ-only / f_gas+kSZ joint).

Convergence-grade convention mirrors `run_a5_chains_final.py` (decision-2
settings): DE-move mixture, 128 walkers x 30k steps, 4 independent seeds,
cross-chain R-hat at assembly. 31-dim chains: the SB35 unit cube + the
f_sat satellite-fraction nuisance (Bigwood prior U(0.10, 0.30), see
`inference.ksz.KszBlock`).

Gated on the matching recovery battery
(`run_a5_recovery_subsets.py --which ...` -> a5_recovery_{which}.json PASS).

Usage:    python run_a5_subsets.py --which kszonly --seed K     (K in 0..3)
Assemble: python run_a5_subsets.py --which kszonly --assemble
(and the same for --which joint.)

Outputs: wp5_chains/a5_subset_{which}_seed{K}.npz, then
a5_fit_{which}_summary.json at assembly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402
from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, log_prob_factory  # noqa: E402
from analysis.paper3a.scripts.run_a5_chains_final import cross_chain_rhat  # noqa: E402

NDIM = 31
N_WALKERS = 128
N_STEPS = 30_000
N_BURN = 8_000
THIN = 15
N_SEEDS = 4
EDGE_FRAC = 0.05


def _gate(which: str) -> dict:
    gate = CHAINS / f"a5_recovery_{which}.json"
    if not gate.exists():
        raise SystemExit(f"recovery battery has not run ({gate}) — refusing the data fit")
    rec = json.loads(gate.read_text())
    if not rec.get("PASS", False):
        raise SystemExit(f"recovery battery did NOT pass ({gate}) — fix calibration first")
    return rec


def _blocks(which: str):
    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)
    fit_blocks = [fgas, ksz] if which == "joint" else [ksz]
    return fgas, ksz, fit_blocks


def run_seed(which: str, seed: int) -> None:
    import emcee

    _gate(which)
    _, _, fit_blocks = _blocks(which)
    rng = np.random.default_rng(100 + seed)
    p0 = rng.uniform(0.02, 0.98, size=(N_WALKERS, NDIM))
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM, log_prob_factory(fit_blocks),
                                    vectorize=True, moves=moves)
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    chain = sampler.get_chain(discard=N_BURN, thin=THIN)          # (S, W, 31)
    logp = sampler.get_log_prob(discard=N_BURN, thin=THIN)
    CHAINS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHAINS / f"a5_subset_{which}_seed{seed}.npz",
        chain=chain.astype(np.float32), logp=logp.astype(np.float32),
        acceptance=np.mean(sampler.acceptance_fraction),
        settings=np.array(json.dumps({"which": which, "ndim": NDIM,
                                      "n_walkers": N_WALKERS, "n_steps": N_STEPS,
                                      "n_burn": N_BURN, "thin": THIN,
                                      "moves": "DE 0.8 + DESnooker 0.2",
                                      "seed": seed})))
    print(f"{which} seed {seed}: acceptance "
          f"{np.mean(sampler.acceptance_fraction):.3f}, archived {chain.shape}")


def assemble(which: str) -> None:
    rec = _gate(which)
    fgas, ksz, _ = _blocks(which)
    chains, logps, accs = [], [], []
    for k in range(N_SEEDS):
        f = np.load(CHAINS / f"a5_subset_{which}_seed{k}.npz", allow_pickle=False)
        chains.append(f["chain"].astype(float))
        logps.append(f["logp"].astype(float))
        accs.append(float(f["acceptance"]))
    rhat = cross_chain_rhat(chains)
    flat = np.concatenate([c.reshape(-1, NDIM) for c in chains])
    logp = np.concatenate([lp.reshape(-1) for lp in logps])

    sub = flat[:: max(1, len(flat) // 40_000)]
    f_group = fgas.predict(sub)[:, 0]
    t1 = ksz.predict(sub)[:, 0]
    f_sat = ksz.F_SAT_RANGE[0] + np.diff(ksz.F_SAT_RANGE)[0] * flat[:, 30]

    imap = int(np.argmax(logp))
    u_map = flat[imap]
    per_block = {"ksz": {"chi2": float(ksz.chi2(u_map)[0]), "n": int(len(ksz.values))}}
    if which == "joint":
        per_block["fgas"] = {"chi2": float(fgas.chi2(u_map)[0]),
                             "n": int(len(fgas.data.values))}

    edge = {pm.ASTRO_NAMES[i]: {"low": float(np.mean(flat[:, i] < EDGE_FRAC)),
                                "high": float(np.mean(flat[:, i] > 1 - EDGE_FRAC))}
            for i in range(30)}
    tripwire = {n: v for n, v in edge.items() if max(v.values()) > 3 * EDGE_FRAC}

    # posterior-predictive p for the kSZ vector (full-covariance draws)
    rng = np.random.default_rng(3)
    draws = sub[rng.integers(0, len(sub), 400)]
    pp = ksz.predict(draws)
    cc = ksz.cov(pp)
    chol = np.linalg.cholesky(cc)
    eps = rng.normal(size=(len(draws), len(ksz.values), 1))
    sim = pp + (chol @ eps)[:, :, 0]
    chi2_sim = np.sum((sim - pp) * np.linalg.solve(cc, (sim - pp)[:, :, None])[:, :, 0], axis=1)
    robs = ksz.values[None, :] - pp
    chi2_obs = np.sum(robs * np.linalg.solve(cc, robs[:, :, None])[:, :, 0], axis=1)
    p_pp_ksz = float(np.mean(chi2_sim >= chi2_obs))

    corr = np.array([abs(np.corrcoef(sub[:, i], t1)[0, 1]) for i in range(30)])
    top = np.argsort(-corr)[:6]

    summary = {
        "which": which,
        "recovery_gate": {"PASS": rec["PASS"],
                          "summaries": [{k: s[k] for k in
                                         ("summary", "coverage68", "coverage95",
                                          "ks_uniform_p")}
                                        for s in rec["summaries"]]},
        "chain": {"acceptance_by_seed": accs,
                  "cross_chain_rhat_max": float(rhat.max()),
                  "cross_chain_rhat_top6": {pm.ASTRO_NAMES[i]: float(rhat[i])
                                            for i in top},
                  "cross_chain_rhat_fsat": float(rhat[30]),
                  "n_samples": int(len(flat))},
        "map": {"chi2_per_block": per_block,
                "pred_ksz": ksz.predict(u_map)[0].tolist(),
                "data_ksz": ksz.values.tolist(),
                "theta_map_named": {pm.ASTRO_NAMES[i]: float(u_map[i]) for i in top},
                "f_sat_map": float(ksz.F_SAT_RANGE[0]
                                   + np.diff(ksz.F_SAT_RANGE)[0] * u_map[30])},
        "posterior_predictive_p_ksz": p_pp_ksz,
        "boundary_tripwire_fired": tripwire,
        "f_group_posterior": {"p16": float(np.percentile(f_group, 16)),
                              "p50": float(np.percentile(f_group, 50)),
                              "p84": float(np.percentile(f_group, 84)),
                              "fgas_data_value": float(fgas.data.values[0])},
        "tksz1_posterior": {"p16": float(np.percentile(t1, 16)),
                            "p50": float(np.percentile(t1, 50)),
                            "p84": float(np.percentile(t1, 84)),
                            "data_value": float(ksz.values[0])},
        "f_sat_posterior": {"p16": float(np.percentile(f_sat, 16)),
                            "p50": float(np.percentile(f_sat, 50)),
                            "p84": float(np.percentile(f_sat, 84))},
    }
    out = CHAINS / f"a5_fit_{which}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:3000])
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("kszonly", "joint"), required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble(args.which)
    elif args.seed is not None:
        run_seed(args.which, args.seed)
    else:
        raise SystemExit("pass --seed K or --assemble")


if __name__ == "__main__":
    main()
