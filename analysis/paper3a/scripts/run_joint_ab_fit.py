"""JOINT_AB_PLAN step 4, product 2: THE joint A+B data fit.

L_joint(theta) = L_fgas x L_kSZ x L_B(coords(theta)); 32-dim chains
(30 SB35 astro + f_sat + sigma_pos), decision-2 convergence convention
(DE-move mixture, 128 x 30k, 4 seeds, cross-chain R-hat). Gated on
a5_recovery_jointab.json PASS — refuses to run otherwise.

Usage:    python run_joint_ab_fit.py --seed K        (K in 0..3)
Assemble: python run_joint_ab_fit.py --assemble

Outputs: wp5_chains/joint_ab_seed{K}.npz, then joint_ab_summary.json.
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
from analysis.paper3a.inference.jointab import SIGMA_POS_MAX, JointBBlock  # noqa: E402
from analysis.paper3a.inference.ksz import KszBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS, log_prob_factory  # noqa: E402
from analysis.paper3a.scripts.run_a5_chains_final import cross_chain_rhat  # noqa: E402

NDIM = 32
N_WALKERS = 128
N_STEPS = 30_000
N_BURN = 8_000
THIN = 15
N_SEEDS = 4
EDGE_FRAC = 0.05


def _gate() -> dict:
    gate = CHAINS / "a5_recovery_jointab.json"
    if not gate.exists():
        raise SystemExit(f"recovery battery has not run ({gate}) — "
                         "refusing the data fit")
    rec = json.loads(gate.read_text())
    if not rec.get("PASS", False):
        raise SystemExit(f"recovery battery did NOT pass ({gate}) — "
                         "fix calibration first")
    return rec


def _blocks():
    ksz = KszBlock()
    fgas = FgasBlock(emu=ksz.emu)
    jb = JointBBlock()
    return fgas, ksz, jb


def run_seed(seed: int) -> None:
    import emcee

    _gate()
    fgas, ksz, jb = _blocks()
    rng = np.random.default_rng(300 + seed)
    p0 = rng.uniform(0.02, 0.98, size=(N_WALKERS, NDIM))
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(
        N_WALKERS, NDIM, log_prob_factory([fgas, ksz, jb]),
        vectorize=True, moves=moves)
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    chain = sampler.get_chain(discard=N_BURN, thin=THIN)
    logp = sampler.get_log_prob(discard=N_BURN, thin=THIN)
    CHAINS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHAINS / f"joint_ab_seed{seed}.npz",
        chain=chain.astype(np.float32), logp=logp.astype(np.float32),
        acceptance=np.mean(sampler.acceptance_fraction),
        settings=np.array(json.dumps({
            "which": "joint_ab", "ndim": NDIM, "n_walkers": N_WALKERS,
            "n_steps": N_STEPS, "n_burn": N_BURN, "thin": THIN,
            "moves": "DE 0.8 + DESnooker 0.2", "seed": seed})))
    print(f"joint_ab seed {seed}: acceptance "
          f"{np.mean(sampler.acceptance_fraction):.3f}, archived {chain.shape}")


def assemble() -> None:
    rec = _gate()
    fgas, ksz, jb = _blocks()
    chains, logps, accs = [], [], []
    for k in range(N_SEEDS):
        f = np.load(CHAINS / f"joint_ab_seed{k}.npz", allow_pickle=False)
        chains.append(f["chain"].astype(float))
        logps.append(f["logp"].astype(float))
        accs.append(float(f["acceptance"]))
    rhat = cross_chain_rhat(chains)
    flat = np.concatenate([c.reshape(-1, NDIM) for c in chains])
    logp = np.concatenate([lp.reshape(-1) for lp in logps])
    sub = flat[:: max(1, len(flat) // 40_000)]

    imap = int(np.argmax(logp))
    u_map = flat[imap]
    chi2_b, y_b, C_b = jb.chi2(u_map)
    per_block = {
        "fgas": {"chi2": float(fgas.chi2(u_map)[0]),
                 "n": int(len(fgas.data.values))},
        "ksz": {"chi2": float(ksz.chi2(u_map)[0]),
                "n": int(len(ksz.values))},
        "b_kappa_y_peaks": {"chi2": float(chi2_b), "n": 4,
                            "data_over_model":
                                np.round(jb.b.data / y_b, 3).tolist()},
    }

    coords, cerr = jb.coords(sub[:, :30])
    sigma_pos = flat[:, 31] * SIGMA_POS_MAX
    f_sat = ksz.F_SAT_RANGE[0] + np.diff(ksz.F_SAT_RANGE)[0] * flat[:, 30]
    f_group = fgas.predict(sub)[:, 0]
    t1 = ksz.predict(sub)[:, 0]

    edge = {pm.ASTRO_NAMES[i]: {"low": float(np.mean(flat[:, i] < EDGE_FRAC)),
                                "high": float(np.mean(flat[:, i]
                                                      > 1 - EDGE_FRAC))}
            for i in range(30)}
    tripwire = {n: v for n, v in edge.items()
                if max(v.values()) > 3 * EDGE_FRAC}

    def pct(x):
        return {"p16": float(np.percentile(x, 16)),
                "p50": float(np.percentile(x, 50)),
                "p84": float(np.percentile(x, 84))}

    # B posterior-predictive p (draws through the theta-dependent C)
    rng = np.random.default_rng(5)
    dsub = sub[rng.integers(0, len(sub), 200)]
    chi2_sim, chi2_obs = [], []
    for u in dsub:
        c2, y, C = jb.chi2(u)
        chol = np.linalg.cholesky(C)
        sim = y + chol @ rng.normal(size=4)
        r_s, r_o = sim - y, jb.b.data - y
        chi2_sim.append(float(r_s @ np.linalg.solve(C, r_s)))
        chi2_obs.append(float(r_o @ np.linalg.solve(C, r_o)))
    p_pp_b = float(np.mean(np.asarray(chi2_sim) >= np.asarray(chi2_obs)))

    summary = {
        "which": "joint_ab",
        "recovery_gate": {"PASS": rec["PASS"],
                          "summaries": [{k: s[k] for k in
                                         ("summary", "coverage68",
                                          "coverage95", "ks_uniform_p")}
                                        for s in rec["summaries"]]},
        "chain": {"acceptance_by_seed": accs,
                  "cross_chain_rhat_max": float(rhat.max()),
                  "cross_chain_rhat_fsat": float(rhat[30]),
                  "cross_chain_rhat_sigmapos": float(rhat[31]),
                  "n_samples": int(len(flat))},
        "map": {"chi2_per_block": per_block,
                "logp": float(logp[imap]),
                "f_sat_map": float(ksz.F_SAT_RANGE[0]
                                   + np.diff(ksz.F_SAT_RANGE)[0]
                                   * u_map[30]),
                "sigma_pos_map_arcmin": float(u_map[31] * SIGMA_POS_MAX),
                "coords_map": jb.coords(u_map[None, :30])[0][0].tolist()},
        "posterior_predictive_p_b": p_pp_b,
        "boundary_tripwire_fired": tripwire,
        "coords_posterior": {"dln_mgas": pct(coords[:, 0]),
                             "dln_t": pct(coords[:, 1])},
        "sigma_pos_posterior_arcmin": pct(sigma_pos),
        "f_sat_posterior": pct(f_sat),
        "f_group_posterior": {**pct(f_group),
                              "fgas_data_value": float(fgas.data.values[0])},
        "tksz1_posterior": {**pct(t1), "data_value": float(ksz.values[0])},
    }
    out = CHAINS / "joint_ab_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:3000])
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble()
    elif args.seed is not None:
        run_seed(args.seed)
    else:
        raise SystemExit("pass --seed K or --assemble")


if __name__ == "__main__":
    main()
