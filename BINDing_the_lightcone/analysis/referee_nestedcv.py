"""Nested cross-validation of the latent search (referee point II-4) and the
matched forward-search-with-eviction on the 30 parameters (II-2 (ii)).

Outer loop: 5 folds.  Inside each outer TRAINING set the entire SFFS search is
re-run from the empty set, scored on its own inner 5-fold split of the training
rows only; the selected set is then fit on the full training set and used to
predict the held-out outer fold.  The reported R^2 therefore contains no
selection information from the rows it is scored on.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python referee_nestedcv.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
CEPH = Path("/mnt/home/mlee1/ceph")
OUT = Path(__file__).resolve().parent
os.chdir(P1)
sys.path.insert(0, str(P1))
from family_model import FamilyModel  # noqa: E402

t0 = time.time()
STATS = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
ZI, EPS, CEIL = 1, 0.002, 14
fm = FamilyModel()
dsn = np.load(CEPH / "bind_sb35/emulator_dataset_nu05.npz", allow_pickle=True)
ELL = np.asarray(dsn["a__suppression__ell"], float)
EDGb = np.load(P1 / "latent_model_coeffs.npz")["ell_edges"]
AMPS = np.load(P1 / "figs_preview/amplitude_sets.npz", allow_pickle=True)
SEARCH = np.load(P1 / "figs_preview/agnostic_lambda_results_obs.npz", allow_pickle=True)
NAMES = [str(x) for x in SEARCH["names"]]
X_ALL = SEARCH["X_ALL"]
THETA = np.asarray(dsn["X_unit"], float)
PNAMES = [str(x) for x in dsn["param_names"]]


def sobol_Y(st):
    dk = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts", "mn": "minima_counts",
          "v0": "mf_v0", "v1": "mf_v1", "v2": "mf_v2"}[st]
    Y = np.asarray(dsn[f"t__{dk}__value"], float)
    Y = Y[:, ZI, :] if Y.ndim == 3 else Y
    if st == "clk":
        Y = np.stack([np.nanmean(Y[:, (ELL >= EDGb[i]) & (ELL < EDGb[i + 1])], 1)
                      for i in range(len(EDGb) - 1)], axis=1)
    return Y[:, :fm.basis(st).shape[1]]


A_ = {st: AMPS[f"{st}__a_sobol"] for st in STATS}
B_ = {st: AMPS[f"{st}__basis"] for st in STATS}
MEAN_ = {st: AMPS[f"{st}__mean"] for st in STATS}
D_ = {st: sobol_Y(st) - MEAN_[st] for st in STATS}
N = len(X_ALL)


def zscore(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def cv_pred(X, rows, folds):
    """CV predictions of the amplitudes, restricted to `rows`."""
    pred = {st: np.empty((len(rows), A_[st].shape[1])) for st in STATS}
    for tr, te in folds:
        Xtr = np.c_[X[rows][tr] - X[rows][tr].mean(0), np.ones(len(tr))]
        Xte = np.c_[X[rows][te] - X[rows][tr].mean(0), np.ones(len(te))]
        for st in STATS:
            W, *_ = np.linalg.lstsq(Xtr, A_[st][rows][tr], rcond=None)
            pred[st][te] = Xte @ W
    return pred


def pooled_score(X, idx, rows, folds):
    Xs = X[:, idx]
    pred = cv_pred(Xs, rows, folds)
    tot = 0.0
    for st in STATS:
        d = D_[st][rows]
        tot += 1 - ((d - pred[st] @ B_[st]) ** 2).sum() / (d ** 2).sum()
    return tot / len(STATS)


def sffs(X, pool, rows, folds, eps=EPS, ceiling=CEIL, verbose=False):
    """the paper's forward search with the floating eviction move."""
    cur, cur_s, best_at_k = [], 0.0, {0: 0.0}
    while len(cur) < ceiling:
        cand = [i for i in pool if i not in cur]
        sc = np.array([pooled_score(X, cur + [i], rows, folds) for i in cand])
        ib = int(np.argmax(sc))
        if sc[ib] - cur_s < eps:
            break
        cur.append(cand[ib]); cur_s = float(sc[ib]); best_at_k[len(cur)] = cur_s
        if verbose:
            print(f"      + {NAMES[cand[ib]] if X is XZ else PNAMES[cand[ib]]:26s} "
                  f"-> {cur_s:.4f}", flush=True)
        improved = True
        while improved and len(cur) > 2:
            improved = False
            for j in list(cur[:-1]):
                trial = [c for c in cur if c != j]
                s = pooled_score(X, trial, rows, folds)
                if s > best_at_k.get(len(trial), -np.inf) + 1e-4:
                    cur, cur_s, best_at_k[len(trial)] = trial, s, s
                    improved = True
                    break
    return cur, cur_s


XZ = zscore(X_ALL, np.arange(N))
TZ = zscore(THETA, np.arange(N))
perm = np.random.default_rng(1).permutation(N)
OUTER = [(np.setdiff1d(perm, te), te) for te in np.array_split(perm, 5)]
R = {}

for tag, X, pool, lab in (("lambda", X_ALL, list(range(X_ALL.shape[1])), NAMES),
                          ("theta", THETA, list(range(THETA.shape[1])), PNAMES)):
    print(f"\n{'=' * 70}\nNESTED CV — {tag} ({len(pool)} candidates)\n{'=' * 70}", flush=True)
    sse = {st: 0.0 for st in STATS}
    sst = {st: 0.0 for st in STATS}
    picks, sizes = [], []
    for k, (tr, te) in enumerate(OUTER):
        Xz = zscore(X, tr)                       # standardization fit on training rows only
        inner = list(KFold(5, shuffle=True, random_state=100 + k).split(tr))
        sel, s_in = sffs(Xz, pool, tr, inner)
        Xtr = np.c_[Xz[tr][:, sel] - Xz[tr][:, sel].mean(0), np.ones(len(tr))]
        Xte = np.c_[Xz[te][:, sel] - Xz[tr][:, sel].mean(0), np.ones(len(te))]
        for st in STATS:
            W, *_ = np.linalg.lstsq(Xtr, A_[st][tr], rcond=None)
            dp = (Xte @ W) @ B_[st]
            mu = D_[st][tr].mean(0)
            sse[st] += ((D_[st][te] - dp) ** 2).sum()
            sst[st] += ((D_[st][te] - mu) ** 2).sum()
        names = [lab[i] for i in sel]
        picks.append(names); sizes.append(len(sel))
        print(f"  outer fold {k}: k={len(sel):2d} inner {s_in:.4f} | {names}", flush=True)
    r2 = {st: 1 - sse[st] / sst[st] for st in STATS}
    pooled = float(np.mean(list(r2.values())))
    print("  NESTED: " + "  ".join(f"{st}:{r2[st]:.4f}" for st in STATS)
          + f"   pooled:{pooled:.4f}", flush=True)
    flat = [n for p in picks for n in p]
    freq = {n: flat.count(n) for n in sorted(set(flat), key=lambda z: -flat.count(z))}
    print(f"  selection stability (of 5 outer folds): "
          f"{ {k: v for k, v in list(freq.items())[:10]} }", flush=True)
    R[tag] = {"nested_cv": {s: round(r2[s], 4) for s in STATS},
              "nested_pooled": round(pooled, 4),
              "sizes": sizes, "picks": picks,
              "stability": freq}

    # flat (non-nested) search on all rows, for the same candidate pool
    Xz = zscore(X, np.arange(N))
    sel_all, s_all = sffs(Xz, pool, np.arange(N),
                          list(KFold(5, shuffle=True, random_state=123).split(np.arange(N))))
    R[tag]["flat_selected"] = [lab[i] for i in sel_all]
    R[tag]["flat_score"] = round(float(s_all), 4)
    print(f"  flat search on all 256: k={len(sel_all)} score {s_all:.4f} "
          f"{[lab[i] for i in sel_all]}", flush=True)

(OUT / "referee_nestedcv.json").write_text(json.dumps(R, indent=1))
print(f"\nwrote referee_nestedcv.json in {time.time() - t0:.0f}s")
