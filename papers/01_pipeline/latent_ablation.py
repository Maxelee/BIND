"""Systematic latent selection: candidate pool -> forward selection +
exhaustive size-4 search, scored on the multi-statistic CV suite.

Candidates (hinge bin H=[13.3,13.6) unless noted; median reducer; snap096;
c_tau is the snap085 profile leg, flagged non-harmonized):
budget/partition: fbar, fgas, fstar, fbar200
structure:        cgas (500/200), cstar (500/200), fout (R500c-R200c gas)
thermo:           logT, logK, logPe, logY
cluster bin:      fbar_C, logT_C, logY_C
profile leg:      ctau (snap085)

Objective: mean over 9 statistics of the median per-bin 5-fold CV-R2
(deterministic folds). SZ spectra exclude the repaint trio.
"""
from itertools import combinations
from pathlib import Path
import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
SCI = CEPH / "bind_science"
OB_OM = 0.0486 / 0.3089
ZI = 1

d = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
run_ids = d["run_ids"]
ELL = d["a__suppression__ell"]
n = len(run_ids)
EDG = np.geomspace(300.0, 3e4, 25)
cz = np.load(SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz")

mt5 = cz["sobol_m_tot_500c_bg"][run_ids]
mg5 = cz["sobol_m_gas_500c_bg"][run_ids]
ms5 = cz["sobol_m_star_500c"][run_ids]
mt2 = cz["sobol_m_tot_200c_bg"][run_ids]
mg2 = cz["sobol_m_gas_200c_bg"][run_ids]
ms2 = cz["sobol_m_star_200c"][run_ids]
Tm = cz["sobol_T_mw_500c"][run_ids]
Km = cz["sobol_K_mw_500c"][run_ids]
Pm = cz["sobol_Pe_mw_500c"][run_ids]
Ym = cz["sobol_Y_500c"][run_ids]
lm = np.log10(np.where(mt5 > 0, mt5, np.nan))


def binmed(perhalo, lo=13.3, hi=13.6, valid=None):
    out = np.full(n, np.nan)
    for q in range(n):
        s = (lm[q] >= lo) & (lm[q] < hi)
        if valid is not None:
            s &= valid[q]
        if s.sum() >= 5:
            out[q] = np.nanmedian(perhalo[q, s])
    return out


pos = lambda x: np.where(x > 0, x, np.nan)
CAND = {
    "fbar":    binmed(pos((mg5 + ms5) / mt5)) / OB_OM,
    "fgas":    binmed(pos(mg5 / mt5)) / OB_OM,
    "fstar":   binmed(ms5 / mt5) / OB_OM,
    "fbar200": binmed(pos((mg2 + ms2) / mt2)) / OB_OM,
    "cgas":    binmed(pos(mg5 / mg2), valid=(mg2 > 0)),
    "cstar":   binmed(pos(ms5 / ms2), valid=(ms2 > 0)),
    "fout":    binmed((mg2 - mg5) / mt2, valid=(mt2 > 0)) / OB_OM,
    "logT":    np.log10(binmed(pos(Tm))),
    "logK":    np.log10(binmed(pos(Km))),
    "logPe":   np.log10(binmed(pos(Pm))),
    "logY":    np.log10(binmed(pos(Ym))),
    "fbar_C":  binmed(pos((mg5 + ms5) / mt5), 13.6, 15.5) / OB_OM,
    "logT_C":  np.log10(binmed(pos(Tm), 13.6, 15.5)),
    "logY_C":  np.log10(binmed(pos(Ym), 13.6, 15.5)),
}
xp = np.load(SCI / "ksz_confront/bind_tauy_xprof_snap085.npz")
tau = xp["tau"][:, 0, :][np.isin(xp["nodes"], run_ids)]
xr = xp["x"]
CAND["ctau*"] = (np.nanmean(tau[:, xr < 0.3], 1)
                 / np.nanmean(tau[:, (xr >= 0.3) & (xr < 1.0)], 1))

names = list(CAND)
CM = np.c_[tuple(CAND[k] for k in names)]
ok = np.isfinite(CM).all(1) & ~np.isin(run_ids, [114, 115, 117])
CM = CM[ok]
m = int(ok.sum())
print(f"candidates: {len(names)}; nodes: {m}")

# statistic suite
def leg(name):
    a = d[f"t__{name}__value"].astype(float)
    a = a[:, ZI, ZI, :] if a.ndim == 4 else \
        (a[:, ZI, :] if (a.ndim == 3 and a.shape[1] == 5) else a)
    return a[ok]


def clean(Y):
    gd = np.isfinite(Y).all(0) & (np.nanstd(Y, 0) > 0)
    return Y[:, gd]


def bin24(Y):
    return np.stack([np.nanmean(Y[:, (ELL >= EDG[i]) & (ELL < EDG[i + 1])], 1)
                     for i in range(24)], axis=1)


def logbin(name):
    return clean(bin24(np.log10(pos(leg(name)))))


STATS = {
    "S":     clean(bin24(leg("suppression"))),
    "pdf":   clean(leg("pdf")),
    "mfv1":  clean(leg("mf_v1")),
    "mfv2":  clean(leg("mf_v2")),
    "Y-M":   clean(np.log10(pos(leg("scaling_Y")))),
    "fg-M":  clean(leg("scaling_f_gas")),
    "T-M":   clean(np.log10(pos(leg("scaling_T")))),
    "kt":    logbin("cl_kappa_tau"),
    "yy":    logbin("cl_yy"),
}
fold = np.arange(m) % 5


def score(idx):
    """mean over stats of median per-bin CV-R2 for candidate subset idx."""
    X = CM[:, list(idx)]
    A = np.c_[X, np.ones(m)]
    tot = 0.0
    for Y in STATS.values():
        pred = np.empty_like(Y)
        for f in range(5):
            tr = fold != f
            b, *_ = np.linalg.lstsq(A[tr], Y[tr], rcond=None)
            pred[fold == f] = A[fold == f] @ b
        r2 = 1 - ((Y - pred) ** 2).mean(0) / Y.var(0)
        tot += float(np.median(r2))
    return tot / len(STATS)


# forward selection to size 6
print("\nFORWARD SELECTION (objective = mean CV-R2 over 9 stats):")
chosen = []
for step in range(6):
    best = (-9, None)
    for j in range(len(names)):
        if j in chosen:
            continue
        sc = score(chosen + [j])
        if sc > best[0]:
            best = (sc, j)
    chosen.append(best[1])
    print(f"  size {step+1}: + {names[best[1]]:8s} -> score {best[0]:.3f}   "
          f"[{', '.join(names[j] for j in chosen)}]")

# exhaustive size-4
print("\nEXHAUSTIVE size-4 (top 12 of C(15,4)=1365):")
ours = tuple(sorted(names.index(k) for k in ("fbar", "fstar", "cgas", "logT")))
results = []
for combo in combinations(range(len(names)), 4):
    results.append((score(combo), combo))
results.sort(reverse=True)
for sc, combo in results[:12]:
    tag = "   <-- OURS" if tuple(sorted(combo)) == ours else ""
    print(f"  {sc:.3f}: " + ", ".join(names[j] for j in combo) + tag)
rank = [i for i, (_, cmb) in enumerate(results)
        if tuple(sorted(cmb)) == ours][0] + 1
sc_ours = [s for s, cmb in results if tuple(sorted(cmb)) == ours][0]
print(f"\nour quad (fbar, fstar, cgas, logT): score {sc_ours:.3f}, "
      f"rank {rank}/1365 (top {100*rank/1365:.1f}%)")
print(f"best quad advantage over ours: {results[0][0] - sc_ours:+.3f}")
# per-stat comparison: ours vs the winner
def per_stat(idx):
    X = CM[:, list(idx)]
    A = np.c_[X, np.ones(m)]
    out = {}
    for nm, Y in STATS.items():
        pred = np.empty_like(Y)
        for f in range(5):
            tr = fold != f
            b, *_ = np.linalg.lstsq(A[tr], Y[tr], rcond=None)
            pred[fold == f] = A[fold == f] @ b
        out[nm] = float(np.median(1 - ((Y - pred)**2).mean(0)/Y.var(0)))
    return out

ps_o, ps_b = per_stat(ours), per_stat(results[0][1])
print("\nper-stat (ours -> best): "
      + ", ".join(f"{nm} {ps_o[nm]:.2f}->{ps_b[nm]:.2f}" for nm in STATS))
