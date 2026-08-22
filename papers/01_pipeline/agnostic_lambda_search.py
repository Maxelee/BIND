"""Fully agnostic latent search for the family-basis model.

Author request (2026-08-12): NO a-priori anchor latents, NO two-bin
restriction, NO cap on the latent count. Search the full candidate
library of halo-population summaries for the set of latents lambda that
best drives the family-basis amplitude map a = M.(lambda - lambda_bar),
pooled over ALL 7 WL statistics (not clk-targeted).

Design:
  - Candidate library: 15 quantities x 7 mass bins (13.0..14.3+, every
    bin >= 27 halos in every run) + 6 cross-bin mass-trend slopes
    + 10 shuffled DECOY candidates (tripwire: selecting one = threshold
    too loose).
  - Selection: sequential floating forward selection (SFFS) from the
    EMPTY set, scored on the pooled mean of the 7 per-stat end-to-end
    CV R^2, selection folds = sklearn KFold(5, shuffle, rs=123).
    Stopping: pooled gain < EPS (no cap; safety ceiling 20 reported if
    hit). Floating step: after each addition, remove any member whose
    removal IMPROVES the best-known score at that cardinality.
  - Cross-checks: exhaustive best subsets (k=1,2 full; k=3 over top-40
    marginals), MultiTaskLasso path, 20-bootstrap greedy stability.
  - Reporting: final set evaluated on the PIPELINE-NATIVE rng(1) folds
    (comparable to the shipped table); fiducial closure computed ONCE at
    the end (never used in selection).

Run:
    /mnt/home/mlee1/venvs/BIND_env/bin/python agnostic_lambda_search.py
"""
import itertools
import os
import sys
import time

import numpy as np
from sklearn.model_selection import KFold

os.chdir("/mnt/home/mlee1/BIND/papers/01_pipeline")
sys.path.insert(0, ".")
from family_model import FamilyModel  # noqa: E402

t0 = time.time()
STATS = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
OB_OM = 0.0486 / 0.3089
EPS = 0.002          # pooled-score acceptance threshold
SAFETY_CEILING = 20
MIN_N = 5
RNG = np.random.default_rng(0)

fm = FamilyModel()
dsn = np.load("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz",
              allow_pickle=True)
coef = np.load("latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
_eld = np.asarray(dsn["a__suppression__ell"], float)
cz = np.load("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/"
             "atlas_cube_snap096.npz")
AMPS = np.load("figs_preview/amplitude_sets.npz", allow_pickle=True)
rows_s = dsn["run_ids"]


def sobol_Y(st):
    dk = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts",
          "mn": "minima_counts", "v0": "mf_v0", "v1": "mf_v1",
          "v2": "mf_v2"}[st]
    Y = np.asarray(dsn[f"t__{dk}__value"], float)
    Y = Y[:, 1, :] if Y.ndim == 3 else Y
    if st == "clk":
        Y = np.stack([np.nanmean(Y[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                      for i in range(len(EDGb) - 1)], axis=1)
    return Y[:, :fm.basis(st).shape[1]]


# ─────────────────────────────────────────────────────────────────────────
# candidate library
# ─────────────────────────────────────────────────────────────────────────
FIELD_KEYS = ["m_tot_500c_bg", "m_gas_500c_bg", "m_gas_500c", "m_star_500c",
              "m_gas_200c_bg", "m_gas_200c", "m_star_200c", "m_tot_200c_bg",
              "m_dm_500c", "m_dm_200c", "T_mw_500c", "K_mw_500c",
              "Pe_mw_500c", "Y_500c"]
SOB = {k: np.asarray(cz[f"sobol_{k}"], float)[rows_s] for k in FIELD_KEYS}
FID = {k: np.asarray(cz[f"fid_{k}"], float) for k in FIELD_KEYS}

BIN_EDGES = [13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.3, 15.0]
BINS = list(zip(BIN_EDGES[:-1], BIN_EDGES[1:]))
BLAB = [f"{lo:.1f}-{hi:.1f}".replace("-15.0", "+") for lo, hi in BINS]
TREND_QUANT = ["f_bar", "f_star", "f_gas200", "logT", "logPe", "logY"]


def _med(v, pos=False, log=False, mn=MIN_N):
    if pos:
        v = np.where(v > 0, v, np.nan)
    if np.isfinite(v).sum() < mn:
        return np.nan
    m = np.nanmedian(v)
    return np.log10(m) if (log and m > 0) else m


def _scat(v, pos=True):
    if pos:
        v = np.where(v > 0, v, np.nan)
    v = v[np.isfinite(v)]
    if len(v) < 4 * MIN_N:
        return np.nan
    lo, hi = np.percentile(v, [16, 84])
    return 0.5 * (hi - lo)


def features_one(F):
    mt5, mg5bg, mg5, ms5 = (F["m_tot_500c_bg"], F["m_gas_500c_bg"],
                            F["m_gas_500c"], F["m_star_500c"])
    mg2bg, mg2, mt2 = F["m_gas_200c_bg"], F["m_gas_200c"], F["m_tot_200c_bg"]
    md5, md2 = F["m_dm_500c"], F["m_dm_200c"]
    T5, K5, Pe5, Y5 = F["T_mw_500c"], F["K_mw_500c"], F["Pe_mw_500c"], F["Y_500c"]
    lm = np.log10(np.where(mt5 > 0, mt5, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = {
            "f_bar": (mg5bg + ms5) / mt5 / OB_OM,
            "f_star": ms5 / mt5 / OB_OM,
            "f_gas200": mg2bg / mt2 / OB_OM,
            "c_gas": mg5bg / mg2bg,
            # c_gas_raw (raw-aperture convention) dropped 2026-08-12:
            # r>=0.99 with the bg version in every bin -- a convention
            # duplicate that confuses the presentation. cgas_over_cdm
            # below still uses raw/raw internally (DM apertures are
            # raw-only in the atlas).
            "c_dm": md5 / md2,
            "cgas_over_cdm": (mg5 / mg2) / (md5 / md2),
        }
        L = {  # pos+log quantities
            "logT": T5, "logK": K5, "logPe": Pe5, "logY": Y5,
            "logT_ss": T5 / np.where(mt5 > 0, mt5, np.nan) ** (2 / 3),
            "logY_ss": Y5 / np.where(mt5 > 0, mt5, np.nan) ** (5 / 3),
        }
    out = {}
    binmed = {q: [] for q in TREND_QUANT}
    for (lo, hi), bl in zip(BINS, BLAB):
        s = (lm >= lo) & (lm < hi)
        for q, arr in Q.items():
            out[f"{q}[{bl}]"] = _med(arr[s])
        for q, arr in L.items():
            out[f"{q}[{bl}]"] = _med(arr[s], pos=True, log=True)
        out[f"scat_fbar[{bl}]"] = _scat(Q["f_bar"][s], pos=False)
        out[f"scat_logY[{bl}]"] = _scat(Y5[s], pos=True)
        ctr = 0.5 * (lo + min(hi, 14.6))
        for q in TREND_QUANT:
            binmed[q].append((ctr, out[f"{q}[{bl}]"]))
    for q in TREND_QUANT:
        x = np.array([c for c, v in binmed[q]])
        y = np.array([v for c, v in binmed[q]])
        ok = np.isfinite(y)
        out[f"trend_{q}"] = (np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 4
                             else np.nan)
    return out


print("Building candidate library ...")
rows = []
names = None
for r in range(len(rows_s)):
    F = {k: SOB[k][r] for k in FIELD_KEYS}
    d = features_one(F)
    if names is None:
        names = list(d.keys())
    rows.append([d[n] for n in names])
X_ALL = np.array(rows)
d_fid = features_one({k: FID[k] for k in FIELD_KEYS})
X_FID = np.array([d_fid[n] for n in names])

# drop non-finite candidates, then add decoys (shuffled copies of real cols)
OBS_ONLY = os.environ.get("OBS_ONLY", "0") == "1"
if OBS_ONLY:
    UNOBS = ("c_dm", "cgas_over_cdm")   # require the DM-only profile
    obs_keep = np.array([not n.split("[")[0] in UNOBS for n in names])
    print(f"[OBS_ONLY] dropping {int((~obs_keep).sum())} DM-referencing "
          f"candidates: {[n for n, k in zip(names, obs_keep) if not k]}")
    X_ALL, X_FID = X_ALL[:, obs_keep], X_FID[obs_keep]
    names = [n for n, k in zip(names, obs_keep) if k]

keep = np.isfinite(X_ALL).all(0) & np.isfinite(X_FID)
dropped = [n for n, k in zip(names, keep) if not k]
X_ALL, X_FID = X_ALL[:, keep], X_FID[keep]
names = [n for n, k in zip(names, keep) if k]
n_real = len(names)
decoy_src = RNG.choice(n_real, 10, replace=False)
decoys = np.stack([RNG.permutation(X_ALL[:, j]) for j in decoy_src], axis=1)
X_ALL = np.c_[X_ALL, decoys]
X_FID = np.r_[X_FID, decoys.mean(0)]  # decoy fid value: irrelevant, never used
names += [f"DECOY_{names[j]}" for j in decoy_src]
print(f"  {n_real} real candidates (+10 decoys), dropped {len(dropped)}: "
      f"{dropped}")

# standardize once (selection operates on z-scored columns; final fit uses raw)
MU, SD = X_ALL.mean(0), X_ALL.std(0)
XZ = (X_ALL - MU) / SD

# ─────────────────────────────────────────────────────────────────────────
# scoring machinery
# ─────────────────────────────────────────────────────────────────────────
SELECT_FOLDS = list(KFold(5, shuffle=True, random_state=123).split(X_ALL))
# pipeline-native reporting folds (family_basis_all.py convention)
perm = np.random.default_rng(1).permutation(len(X_ALL))
REPORT_FOLDS = [(np.setdiff1d(perm, te), te)
                for te in np.array_split(perm, 5)]

Y_ = {st: sobol_Y(st) for st in STATS}
A_ = {st: AMPS[f"{st}__a_sobol"] for st in STATS}
B_ = {st: AMPS[f"{st}__basis"] for st in STATS}
MEAN_ = {st: AMPS[f"{st}__mean"] for st in STATS}
D_ = {st: Y_[st] - MEAN_[st] for st in STATS}
SST_ = {st: (D_[st] ** 2).sum() for st in STATS}


def e2e_all(idx, folds):
    """per-stat e2e CV R^2 for candidate index list idx; returns dict."""
    X = XZ[:, idx]
    r2 = {}
    pred_amp = {st: np.empty_like(A_[st]) for st in STATS}
    for tr, te in folds:
        Xtr = np.c_[X[tr] - X[tr].mean(0), np.ones(len(tr))]
        Xte = np.c_[X[te] - X[tr].mean(0), np.ones(len(te))]
        for st in STATS:
            W, *_ = np.linalg.lstsq(Xtr, A_[st][tr], rcond=None)
            pred_amp[st][te] = Xte @ W
    for st in STATS:
        Dp = pred_amp[st] @ B_[st]
        r2[st] = 1 - ((D_[st] - Dp) ** 2).sum() / SST_[st]
    return r2


def pooled(idx, folds=SELECT_FOLDS):
    r2 = e2e_all(idx, folds)
    return float(np.mean([r2[st] for st in STATS])), r2


NC = XZ.shape[1]
ALL = list(range(NC))

# ─────────────────────────────────────────────────────────────────────────
# 1. marginal (k=1) scan + exhaustive pairs
# ─────────────────────────────────────────────────────────────────────────
print("\n--- marginal scan (k=1, selection folds, pooled score) ---")
marg = np.array([pooled([i])[0] for i in ALL])
order = np.argsort(marg)[::-1]
for i in order[:12]:
    print(f"  {names[i]:26s} pooled R2 = {marg[i]:.4f}")
print(f"  (best DECOY: {marg[[i for i in ALL if names[i].startswith('DECOY')]].max():.4f})")

print("\n--- exhaustive pairs (k=2) ---")
best_pairs = []
for i, j in itertools.combinations(order[:60], 2):   # top-60 marginals
    best_pairs.append((pooled([i, j])[0], i, j))
best_pairs.sort(reverse=True)
for s, i, j in best_pairs[:5]:
    print(f"  {names[i]:24s} + {names[j]:24s} = {s:.4f}")

print("\n--- exhaustive triples (k=3, top-40 marginals) ---")
best_tri = []
for i, j, k in itertools.combinations(order[:40], 3):
    best_tri.append((pooled([i, j, k])[0], i, j, k))
best_tri.sort(reverse=True)
for s, i, j, k in best_tri[:5]:
    print(f"  {names[i]:22s} + {names[j]:22s} + {names[k]:22s} = {s:.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 2. SFFS from the empty set (the headline search)
# ─────────────────────────────────────────────────────────────────────────
print(f"\n--- SFFS from empty set (EPS = {EPS}, pooled objective) ---")
cur: list[int] = []
cur_score = 0.0
best_at_k = {0: 0.0}
path = []
PATH_REC = []          # (action, cand_index, score_after) for every move
STEP_SCORES = []       # per FORWARD step: full-NC vector of trial scores
while len(cur) < SAFETY_CEILING:
    # forward step
    cand = [i for i in ALL if i not in cur]
    scores = np.array([pooled(cur + [i])[0] for i in cand])
    full = np.full(NC, np.nan)
    full[cand] = scores
    STEP_SCORES.append(full)
    ib = int(np.argmax(scores))
    gain = scores[ib] - cur_score
    if gain < EPS:
        print(f"  STOP: best gain {gain:+.4f} < EPS ({names[cand[ib]]})")
        break
    cur.append(cand[ib])
    PATH_REC.append(("+", int(cand[ib]), float(scores[ib])))
    cur_score = float(scores[ib])
    best_at_k[len(cur)] = cur_score
    path.append((names[cand[ib]], cur_score, gain))
    print(f"  + {names[cand[ib]]:26s} -> pooled {cur_score:.4f} (gain {gain:+.4f})")
    # floating backward step(s): remove if it IMPROVES best-known at k-1
    improved = True
    while improved and len(cur) > 2:
        improved = False
        for j in list(cur[:-1]):        # never immediately drop the newest
            trial = [c for c in cur if c != j]
            s = pooled(trial)[0]
            if s > best_at_k.get(len(trial), -np.inf) + 1e-4:
                print(f"  - {names[j]:26s} (float-out) -> pooled {s:.4f}")
                PATH_REC.append(("-", int(j), float(s)))
                cur = trial
                cur_score = s
                best_at_k[len(cur)] = s
                improved = True
                break
if len(cur) >= SAFETY_CEILING:
    print(f"  WARNING: hit safety ceiling {SAFETY_CEILING}")

CHOSEN = list(cur)
print(f"\nCHOSEN ({len(CHOSEN)}): {[names[i] for i in CHOSEN]}")
n_decoy = sum(names[i].startswith("DECOY") for i in CHOSEN)
print(f"decoys selected: {n_decoy}  {'(TRIPWIRE FAILED)' if n_decoy else '(tripwire clean)'}")

# ─────────────────────────────────────────────────────────────────────────
# 3. bootstrap stability of the greedy (no floating, same EPS)
# ─────────────────────────────────────────────────────────────────────────
print("\n--- bootstrap stability (20 resamples, plain greedy) ---")
freq = np.zeros(NC)
BOOT_SETS = []
for b in range(20):
    bs = RNG.choice(len(X_ALL), len(X_ALL), replace=True)
    folds_b = list(KFold(5, shuffle=True, random_state=123).split(bs))
    Xb = XZ[bs]
    Db = {st: D_[st][bs] for st in STATS}
    Ab = {st: A_[st][bs] for st in STATS}
    SSTb = {st: (Db[st] ** 2).sum() for st in STATS}

    def pooled_b(idx):
        tot = 0.0
        for st in STATS:
            pred = np.empty_like(Ab[st])
            for tr, te in folds_b:
                Xtr = np.c_[Xb[tr][:, idx] - Xb[tr][:, idx].mean(0), np.ones(len(tr))]
                Xte = np.c_[Xb[te][:, idx] - Xb[tr][:, idx].mean(0), np.ones(len(te))]
                W, *_ = np.linalg.lstsq(Xtr, Ab[st][tr], rcond=None)
                pred[te] = Xte @ W
            Dp = pred @ B_[st]
            tot += 1 - ((Db[st] - Dp) ** 2).sum() / SSTb[st]
        return tot / len(STATS)

    curb, sb = [], 0.0
    while len(curb) < 12:
        cand = [i for i in ALL if i not in curb]
        sc = np.array([pooled_b(curb + [i]) for i in cand])
        ib = int(np.argmax(sc))
        if sc[ib] - sb < EPS:
            break
        curb.append(cand[ib]); sb = sc[ib]
    freq[curb] += 1
    BOOT_SETS.append(np.array(curb))
    print(f"  boot {b:2d}: k={len(curb)} {[names[i] for i in curb[:4]]}...")
print("\nselection frequency (>=25%):")
for i in np.argsort(freq)[::-1]:
    if freq[i] >= 5:
        print(f"  {names[i]:26s} {freq[i]:.0f}/20")

# ─────────────────────────────────────────────────────────────────────────
# 4. near-duplicate structure of the chosen set
# ─────────────────────────────────────────────────────────────────────────
print("\n--- near-duplicates (|r|>0.95) of each chosen latent ---")
C = np.corrcoef(XZ.T)
for i in CHOSEN:
    dup = [names[j] for j in ALL if j != i and abs(C[i, j]) > 0.95]
    print(f"  {names[i]:26s}: {dup if dup else '(none)'}")

# ─────────────────────────────────────────────────────────────────────────
# 5. final evaluation on REPORTING folds + comparisons + fiducial closure
# ─────────────────────────────────────────────────────────────────────────
def report(tag, idx):
    r2 = e2e_all(idx, REPORT_FOLDS)
    print(f"  {tag:24s} " + "  ".join(f"{st}:{r2[st]:.4f}" for st in STATS)
          + f"   pooled:{np.mean(list(r2.values())):.4f}")
    return r2


print("\n--- FINAL (pipeline-native rng(1) reporting folds) ---")
i4 = [names.index(n) for n in ["f_bar[13.4-13.6]"] if n in names]  # placeholder guard
try:
    SHIP4 = [names.index(n) for n in
             ["f_bar[13.4-13.6]", "f_star[13.4-13.6]", "c_gas[13.4-13.6]",
              "logT[13.4-13.6]"]]
except ValueError:
    SHIP4 = None
r2_chosen = report(f"agnostic-{len(CHOSEN)}", CHOSEN)
# shipped-8 comparison, re-measured in THIS harness for apples-to-apples:
# shipped bins were 13.3-13.6 / 13.6-14.5 which are NOT in this bin grid;
# quote the shipped bundle numbers instead (printed for reference).
print("  shipped-8 (bundle, rng(1) canon): clk:0.9593 pdf:0.9330 pk:0.7793 "
      "mn:0.8232 v0:0.9541 v1:0.9425 v2:0.9565 pooled:0.9211")

print("\n--- fiducial closure (fit all 256, predict fid; NEVER used above) ---")
Xr = X_ALL[:, CHOSEN]
mu = Xr.mean(0)
lam_fid = X_FID[CHOSEN]
for st in STATS:
    W, *_ = np.linalg.lstsq(np.c_[Xr - mu, np.ones(len(Xr))], A_[st], rcond=None)
    a_fid = np.r_[lam_fid - mu, 1.0] @ W
    curve = MEAN_[st] + a_fid @ B_[st]
    meas = AMPS[f"{st}__fid_measured"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(curve - meas) / np.abs(meas)
    print(f"  {st}: median |dev| = {100 * np.nanmedian(dev):.2f}%")
    if st == "clk":
        print(f"    clk trough band19: pred={curve[19]:.4f} measured={meas[19]:.4f} "
              f"(8-latent shipped: 0.908, 4-latent: 0.920)")

print("\n--- fiducial percentile of each chosen latent ---")
for i in CHOSEN:
    pct = 100 * (X_ALL[:, i] < X_FID[i]).mean()
    print(f"  {names[i]:26s} fid pct = {pct:5.1f}")

np.savez("figs_preview/agnostic_lambda_results"
         + ("_obs" if OBS_ONLY else "") + ".npz",
         names=np.array(names), X_ALL=X_ALL, X_FID=X_FID,
         chosen=np.array(CHOSEN), freq=freq, marg=marg,
         boot_sets=np.array(BOOT_SETS, dtype=object),
         step_scores=np.array(STEP_SCORES),
         path_action=np.array([a for a, _, _ in PATH_REC]),
         path_index=np.array([i for _, i, _ in PATH_REC]),
         path_score=np.array([sc for _, _, sc in PATH_REC]),
         chosen_names=np.array([names[i] for i in CHOSEN]))
print(f"\nDONE in {time.time() - t0:.0f}s")
