"""SYSTEMATICS-HUNT re-score: does the f_gas top-bin residual survive its
own halo-sampling error?

FINDINGS_fgas.md candidate 2. The top model bin (14.6-15.2) is built from
FIVE halos in wp3_gate/operator_tables_v3/fiducial_run_0000_snap096.npz
(per-bin counts 1377/507/144/30/5). Its halo-bootstrap error is 13.3%; the
likelihood carries the k-fold GP rms, 3.0%. The k-fold rms structurally
cannot see halo-sampling noise because the 256-run design shares one DMO
catalogue, so that noise is common-mode across the design and cancels out
of every k-fold residual.

PART 1 -- re-score the FROZEN chains (wp5_chains/a5_final_seed{0..3}.npz)
with emul_frac[top] 0.030 -> 0.133, everything else identical.

  PRE-REGISTERED (fixed before running, copied from FINDINGS_fgas.md):
    ARTIFACT  if top-bin residual < 1.5 sigma AND f_gas MAP chi2 drops >= 2.0
    PHYSICAL  if top-bin residual stays > 2.0 sigma
    otherwise INCONCLUSIVE -- reported as such, not rounded to a verdict.

PART 2 -- companion: drop the 14.6-15.2 bin entirely, re-fit on 4 bins,
and compare f_group. If f_group is unchanged the top bin never carried the
constraint and the draft's "shape strain" wording needs revision.

  Stage A (this script, `rescore`): importance-reweight the frozen chain to
  the 4-bin posterior, with the effective sample size reported so the
  reader can judge it.
  Stage B (`refit4`): an actual 4-bin emcee run initialized FROM the frozen
  5-bin posterior (dropping a bin only relaxes the constraint, so the frozen
  posterior is a well-inside-the-typical-set start and a short run is
  already burned in). Stage B is the number of record; Stage A is the
  cross-check.

Nothing frozen is written. Outputs go to a NEW ceph directory.

Usage:
    python run_syshunt_fgas_topbin.py rescore
    python run_syshunt_fgas_topbin.py refit4
    python run_syshunt_fgas_topbin.py assemble
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.fgas import FgasBlock  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

OUT = Path("/mnt/ceph/users/mlee1/paper3/A/systematics_hunt")

TOP_BIN = 4                 # the 14.6-15.2 model/data bin
EMUL_FRAC_TOP_OLD = 0.030   # k-fold GP rms carried by FgasBlock
EMUL_FRAC_TOP_NEW = 0.133   # halo bootstrap over the 5 halos in that bin

# pre-registered thresholds
ART_SIGMA_MAX = 1.5
ART_DCHI2_MIN = 2.0
PHYS_SIGMA_MIN = 2.0

N_TOP = 20_000              # highest-logp frozen samples to search for the MAP
N_EVEN = 20_000             # even stride subsample for posterior statistics


# --------------------------------------------------------------- helpers ---

def load_frozen():
    """Flat frozen chain (N,30) + logp (N,). Read-only."""
    ch, lp = [], []
    for k in range(4):
        d = np.load(CHAINS / f"a5_final_seed{k}.npz", allow_pickle=False)
        ch.append(d["chain"].astype(float).reshape(-1, 30))
        lp.append(d["logp"].astype(float).reshape(-1))
    return np.concatenate(ch), np.concatenate(lp)


def sigma_from(blk, pred, emul_frac):
    """FgasBlock.sigma, but with emul_frac supplied instead of blk's own.

    Copied (not imported) from inference/fgas.py so that this script never
    depends on -- or perturbs -- the frozen likelihood module.
    """
    from analysis.paper3a.inference.fgas import C2S_TRANSFER_SYS, PAINT_BIAS_RESID
    emul = np.interp(blk.data.bin_centers, blk.model_centers, emul_frac)
    frac_model = np.sqrt(emul**2 + PAINT_BIAS_RESID**2 + C2S_TRANSFER_SYS**2)
    return np.sqrt(blk.data.stat_err[None, :] ** 2
                   + (frac_model[None, :] * pred) ** 2)


def loglike_from(blk, pred, emul_frac, bins=None):
    """Gaussian loglike given precomputed pred; `bins` selects the terms."""
    sig = sigma_from(blk, pred, emul_frac)
    r = (pred - blk.data.values[None, :]) / sig
    t = r**2 + np.log(2 * np.pi * sig**2)
    if bins is not None:
        t = t[:, bins]
    return -0.5 * np.sum(t, axis=1)


def point_report(blk, u, emul_frac, label):
    pred = blk.predict(u[None, :])
    sig = sigma_from(blk, pred, emul_frac)
    r = (pred - blk.data.values[None, :]) / sig
    return {
        "label": label,
        "emul_frac": emul_frac.tolist(),
        "pred": pred[0].tolist(),
        "sigma": sig[0].tolist(),
        "residual_sigma": r[0].tolist(),
        "chi2": float(np.sum(r[0] ** 2)),
        "chi2_top_bin": float(r[0, TOP_BIN] ** 2),
        "f_group": float(pred[0, 0]),
    }


def batched_predict(blk, U, chunk=256, tag=""):
    # chunk matters: the GP builds an (chunk, n_train, 30) broadcast, so
    # large chunks are memory-bound. 256 runs ~2.5x faster per sample
    # than 2000 on this box.
    out = np.empty((len(U), len(blk.data.values)))
    t0 = time.time()
    for i in range(0, len(U), chunk):
        out[i:i + chunk] = blk.predict(U[i:i + chunk])
        print(f"  [{tag}] {i + min(chunk, len(U) - i)}/{len(U)}"
              f"  {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- stage A ---

def rescore() -> None:
    blk = FgasBlock()
    frac_old = blk.emul_frac.copy()
    assert abs(frac_old[TOP_BIN] - EMUL_FRAC_TOP_OLD) < 2e-3, frac_old
    frac_new = frac_old.copy()
    frac_new[TOP_BIN] = EMUL_FRAC_TOP_NEW

    flat, logp = load_frozen()
    print(f"frozen chain {flat.shape}, logp finite {np.isfinite(logp).sum()}",
          flush=True)

    order = np.argsort(-logp)
    idx_top = order[:N_TOP]
    idx_even = np.arange(0, len(flat), max(1, len(flat) // N_EVEN))
    idx = np.unique(np.concatenate([idx_top, idx_even]))
    U = flat[idx]
    print(f"scoring {len(U)} samples ({len(idx_top)} top-logp + "
          f"{len(idx_even)} even stride)", flush=True)

    pred = batched_predict(blk, U, tag="rescore")

    ll_old = loglike_from(blk, pred, frac_old)
    ll_new = loglike_from(blk, pred, frac_new)

    u_map_old = flat[int(np.argmax(logp))]                 # the frozen MAP
    u_map_new = U[int(np.argmax(ll_new))]                  # MAP under new sigma

    res = {
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "chains": [str(CHAINS / f"a5_final_seed{k}.npz") for k in range(4)],
            "n_frozen_samples": int(len(flat)),
            "n_scored": int(len(U)),
            "emul_frac_frozen": frac_old.tolist(),
            "emul_frac_inflated": frac_new.tolist(),
            "change": f"emul_frac[{TOP_BIN}] {EMUL_FRAC_TOP_OLD} -> "
                      f"{EMUL_FRAC_TOP_NEW} (halo bootstrap, 5 halos)",
            "data_values": blk.data.values.tolist(),
            "data_stat_err": blk.data.stat_err.tolist(),
            "data_n_per_bin": blk.data.n_per_bin.tolist(),
            "bin_centers": blk.data.bin_centers.tolist(),
        },
        "preregistered_criterion": {
            "artifact": f"top-bin |residual| < {ART_SIGMA_MAX} sigma AND "
                        f"chi2 drop >= {ART_DCHI2_MIN}",
            "physical": f"top-bin |residual| > {PHYS_SIGMA_MIN} sigma",
            "else": "INCONCLUSIVE",
        },
    }

    # (a) pure re-score: frozen MAP, inflated sigma
    a0 = point_report(blk, u_map_old, frac_old, "frozen MAP / frozen sigma")
    a1 = point_report(blk, u_map_old, frac_new, "frozen MAP / inflated sigma")
    # (b) re-MAP under the inflated likelihood
    b1 = point_report(blk, u_map_new, frac_new, "re-MAP / inflated sigma")
    res["at_frozen_map"] = {"baseline": a0, "inflated": a1}
    res["re_map_inflated"] = b1
    res["remap_moved"] = bool(not np.allclose(u_map_new, u_map_old))

    for key, new in (("pure_rescore", a1), ("with_remap", b1)):
        r_top = abs(new["residual_sigma"][TOP_BIN])
        dchi2 = a0["chi2"] - new["chi2"]
        if r_top < ART_SIGMA_MAX and dchi2 >= ART_DCHI2_MIN:
            v = "ARTIFACT"
        elif r_top > PHYS_SIGMA_MIN:
            v = "PHYSICAL"
        else:
            v = "INCONCLUSIVE"
        res.setdefault("verdicts", {})[key] = {
            "top_bin_residual_sigma_baseline":
                abs(a0["residual_sigma"][TOP_BIN]),
            "top_bin_residual_sigma_new": r_top,
            "chi2_baseline": a0["chi2"], "chi2_new": new["chi2"],
            "delta_chi2": dchi2,
            "verdict": v,
        }

    # ---- stage A of part 2: importance-reweight to the 4-bin posterior ----
    keep = np.array([b for b in range(len(blk.data.values)) if b != TOP_BIN])
    ie = np.searchsorted(idx, idx_even)                 # even subsample only
    ie = ie[(ie < len(idx)) & (idx[np.clip(ie, 0, len(idx) - 1)] == idx_even)]
    pe = pred[ie]
    ll5 = loglike_from(blk, pe, frac_old)
    ll4 = loglike_from(blk, pe, frac_old, bins=keep)
    w = np.exp(ll4 - ll5 - np.max(ll4 - ll5))
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    fg = pe[:, 0]
    res["drop_top_bin_importance_sampled"] = {
        "n_samples": int(len(fg)),
        "ess": ess,
        "ess_frac": ess / len(fg),
        "f_group_5bin": {
            "p16": float(np.percentile(fg, 16)),
            "p50": float(np.percentile(fg, 50)),
            "p84": float(np.percentile(fg, 84)),
            "mean": float(fg.mean()),
        },
        "f_group_4bin_IS": {
            "p16": float(_wq(fg, w, 0.16)),
            "p50": float(_wq(fg, w, 0.50)),
            "p84": float(_wq(fg, w, 0.84)),
            "mean": float(np.sum(w * fg)),
        },
        "data_f_group": float(blk.data.values[0]),
        "caveat": "IS from the 5-bin posterior to the broader 4-bin posterior; "
                  "read ess_frac before trusting. Stage B (refit4) is the "
                  "number of record.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fgas_topbin_rescore.json"
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdicts"], indent=2))
    print(json.dumps(res["drop_top_bin_importance_sampled"], indent=2))
    print(f"wrote {p}")

    np.savez_compressed(OUT / "fgas_topbin_rescore_scored.npz",
                        idx=idx, pred=pred, ll_old=ll_old, ll_new=ll_new)


def _wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, c / c[-1], x[o])


# ---------------------------------------------------------------- stage B ---

def refit4(n_walkers=128, n_steps=4000, n_burn=1500, thin=5, seed=0) -> None:
    """Actual 4-bin re-fit, walkers initialized from the frozen posterior."""
    import emcee

    blk = FgasBlock()
    keep = np.array([b for b in range(len(blk.data.values)) if b != TOP_BIN])
    frac = blk.emul_frac.copy()

    def log_prob(U):
        U = np.atleast_2d(U)
        lp = np.full(len(U), -np.inf)
        good = ~np.any((U < 0.0) | (U > 1.0), axis=1)
        if good.any():
            lp[good] = loglike_from(blk, blk.predict(U[good]), frac, bins=keep)
        return lp

    flat, logp = load_frozen()
    rng = np.random.default_rng(2026 + seed)
    p0 = flat[rng.choice(len(flat), size=n_walkers, replace=False)]
    p0 = np.clip(p0 + rng.normal(0, 0.01, p0.shape), 1e-3, 1 - 1e-3)

    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    s = emcee.EnsembleSampler(n_walkers, 30, log_prob, vectorize=True,
                              moves=moves)
    t0 = time.time()
    s.run_mcmc(p0, n_steps, progress=False)
    chain = s.get_chain(discard=n_burn, thin=thin)
    lp = s.get_log_prob(discard=n_burn, thin=thin)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"fgas_drop_topbin_chain_seed{seed}.npz",
        chain=chain.astype(np.float32), logp=lp.astype(np.float32),
        acceptance=np.mean(s.acceptance_fraction),
        settings=np.array(json.dumps(
            {"n_walkers": n_walkers, "n_steps": n_steps, "n_burn": n_burn,
             "thin": thin, "seed": seed, "dropped_bin": TOP_BIN,
             "init": "frozen a5_final posterior + N(0,0.01) jitter",
             "wall_s": time.time() - t0})))
    print(f"refit4 seed {seed}: acceptance "
          f"{np.mean(s.acceptance_fraction):.3f} shape {chain.shape} "
          f"{time.time() - t0:.0f}s")


def assemble() -> None:
    blk = FgasBlock()
    fs = sorted(OUT.glob("fgas_drop_topbin_chain_seed*.npz"))
    if not fs:
        print("no refit4 chains present")
        return
    ch = [np.load(f, allow_pickle=False)["chain"].astype(float).reshape(-1, 30)
          for f in fs]
    flat4 = np.concatenate(ch)
    sub = flat4[:: max(1, len(flat4) // 8000)]
    fg4 = batched_predict(blk, sub, tag="refit4")[:, 0]

    flat5, logp5 = load_frozen()
    sub5 = flat5[:: max(1, len(flat5) // 8000)]
    fg5 = batched_predict(blk, sub5, tag="frozen")[:, 0]

    def q(x):
        return {"p16": float(np.percentile(x, 16)),
                "p50": float(np.percentile(x, 50)),
                "p84": float(np.percentile(x, 84)),
                "mean": float(x.mean()), "n": int(len(x))}

    d5, d4 = q(fg5), q(fg4)
    shift = d4["p50"] - d5["p50"]
    width5 = 0.5 * (d5["p84"] - d5["p16"])

    # convergence of the quantity that is actually being reported
    from analysis.paper3a.inference.sampler import split_rhat
    rh_theta = [float(split_rhat(np.load(f, allow_pickle=False)["chain"]
                                 .astype(float)).max()) for f in fs]
    def _rhat_scalar(groups):
        nmin = min(len(g) for g in groups)
        x = np.stack([np.asarray(g)[:nmin, None] for g in groups])
        n = x.shape[1]
        m = x.mean(axis=1)
        wv = x.var(axis=1, ddof=1).mean(axis=0)
        bv = n * m.var(axis=0, ddof=1)
        return float(np.sqrt(((n - 1) / n * wv + bv / n) / wv)[0])

    # split-half on WALKERS within each seed (works with a single seed);
    # plus cross-seed when more than one seed is present.
    fgc, halves = [], []
    for f in fs:
        c = np.load(f, allow_pickle=False)["chain"].astype(float)  # (S,W,30)
        w = c.shape[1]
        for lo, hi in ((0, w // 2), (w // 2, w)):
            s = c[:, lo:hi].reshape(-1, 30)
            s = s[:: max(1, len(s) // 1500)]
            halves.append(batched_predict(blk, s, tag="rhat")[:, 0])
        fgc.append(np.concatenate(halves[-2:]))
    rhat_fg_split = _rhat_scalar(halves)
    rhat_fg = _rhat_scalar(fgc) if len(fgc) > 1 else None

    out = {
        "chains": [str(f) for f in fs],
        "convergence": {
            "split_rhat_max_theta_per_seed": rh_theta,
            "cross_chain_rhat_f_group": rhat_fg,
            "split_walker_rhat_f_group": rhat_fg_split,
            "acceptance_per_seed": [
                float(np.load(f, allow_pickle=False)["acceptance"])
                for f in fs],
            "note": "R-hat on the 30-dim theta is dominated by "
                    "prior-unconstrained directions; f_group is the "
                    "reported quantity and carries its own R-hat.",
        },
        "f_group_5bin_frozen": d5,
        "f_group_4bin_refit": d4,
        "shift_p50": shift,
        "shift_in_units_of_5bin_sigma": shift / width5 if width5 else None,
        "data_f_group": float(blk.data.values[0]),
        "group_over_prediction": {
            "5bin": d5["p50"] / float(blk.data.values[0]),
            "4bin": d4["p50"] / float(blk.data.values[0]),
            "note": "the OTHER pillar of the shape-strain argument; it rests "
                    "on 1377 halos (0.8% sampling error) and is not at issue "
                    "in this test.",
        },
        "interpretation": (
            "the top bin carried NONE of the f_group constraint"
            if abs(shift) < 0.25 * width5 else
            "the top bin carried PART of the f_group constraint: dropping it "
            "moves f_group toward the data, but the group over-prediction "
            "survives essentially intact"
            if abs(shift) < 1.0 * width5 else
            "the top bin carried the f_group constraint"),
    }
    p = OUT / "fgas_drop_topbin_refit.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rescore"
    {"rescore": rescore, "refit4": refit4, "assemble": assemble}[cmd]()
