"""Fig 5b replacement (referee I-3): full-box total-mass P(k) for the paper's
fiducial model (env-selected via PAPER_* vars), M200c >= 1e13, circular 4xR200
shared-content paste.

Three columns (CV, SB35, 1P) x three rows:
  row 1  S(k) = P/P_DMO      : Truth (black), BIND (tab:orange), Hydro-replaced (tab:blue)
  row 2  P/P_Truth           : BIND + Hydro-replaced (16-84 bands), DMO (grey dashed)
  row 3  P_BIND/P_hydro-repl : median with 16-84 and 5-95 bands (NEW - the ratio the
                               "saturates the pasting ceiling" claim needs)

Statistic: per-sim ratio of log-rebinned spectra, median/quantiles over sims
(as in referee_response/pk_agent/make_fig5_replacement.py). Data: the model
cache pk_fixed.npz keys {suite}_fixed_{truth,dmo,ge13,hr_ge13} (built by
tools/paper_cache/build_pk_fixed.py; hr_ge13 = truth patches of the SAME >=1e13
halos, same shared paste). Truncated at the 1024-pixel Nyquist k = 64.3 h/Mpc.

Usage:  export PAPER_SUITE_ROOT/PAPER_MODEL_SUBDIR/PAPER_MASS_DIR/PAPER_MODEL_TAG
        python fig5b_total_field_pk.py
Writes: examples/paper_figures/fig5b_total_field_pk.{pdf,png} + stats to stdout.
"""
import os
import sys
from pathlib import Path

# Model selection comes from the environment (no hardcoded model paths) so the
# script is re-runnable for any model tag; fail fast if the caller forgot.
_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR", "PAPER_MASS_DIR",
                 "PAPER_MODEL_TAG")
_missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
if _missing:
    sys.exit(f"fig5b_total_field_pk: export {', '.join(_missing)} before running")

sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/tools/paper_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paper_config as C

# ── style: identical to examples/_build_paper_nbs.py SETUP ──────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook"])
except Exception:
    pass

FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)

def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        fig.savefig(FIG_DIR / f"{name}.{e}", dpi=300, bbox_inches="tight")
    print("  saved", name)

# ── data: spine pk_fixed.npz only ───────────────────────────────────────────
pkf = dict(np.load(C.CACHE_DIR / "pk_fixed.npz"))
k = pkf["k"]
KNYQ = np.pi * C.N_PIX_FULL / C.BOX_SIZE          # 64.34 h/Mpc
km = k <= KNYQ
kk = k[km]

# log-rebin (56 geometric bins), as in the pk_agent prototype
EDGES = np.geomspace(kk.min() * 0.999, kk.max() * 1.001, 56)
IDX = np.digitize(kk, EDGES) - 1
GRP = [np.where(IDX == i)[0] for i in range(len(EDGES) - 1) if (IDX == i).any()]
KB = np.array([kk[g].mean() for g in GRP])

def rb(a):
    """(n_sim, n_k_native) -> (n_sim, n_KB): mean P in each log-k bin."""
    return np.stack([a[:, g].mean(1) for g in GRP], 1)

SUITES = [("CV", "CV"), ("Test", "SB35"), ("1P", "1P")]
STAT_BANDS = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 64)]

data = {}
for s, disp in SUITES:
    tr, dm = pkf[f"{s}_fixed_truth"][:, km], pkf[f"{s}_fixed_dmo"][:, km]
    bi, hr = pkf[f"{s}_fixed_ge13"][:, km], pkf[f"{s}_fixed_hr_ge13"][:, km]
    fin = np.isfinite(bi).all(1) & np.isfinite(hr).all(1)   # SB35_665: no >=1e13 halos
    data[s] = dict(tr=rb(tr[fin]), dm=rb(dm[fin]), bi=rb(bi[fin]), hr=rb(hr[fin]),
                   n=int(fin.sum()), disp=disp,
                   tr_nat=tr[fin], dm_nat=dm[fin], bi_nat=bi[fin], hr_nat=hr[fin])

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(3, 3, figsize=(15.6, 10.5), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1.25, 1.25],
                                    "hspace": 0, "wspace": 0.07})
med = lambda r: np.median(r, 0)
q = lambda r, p: np.quantile(r, p, 0)

for col, (s, disp) in enumerate(SUITES):
    d = data[s]
    tr, dm, bi, hr = d["tr"], d["dm"], d["bi"], d["hr"]

    # row 1 — suppression S(k) = P/P_DMO
    a = ax[0, col]
    for r_, colr in ((tr / dm, "0.25"), (bi / dm, "tab:orange"), (hr / dm, "tab:blue")):
        a.fill_between(KB, q(r_, .16), q(r_, .84), color=colr, alpha=0.16, lw=0)
    a.plot(KB, med(tr / dm), "k", lw=1.6, label="Truth / DMO")
    a.plot(KB, med(bi / dm), color="tab:orange", lw=1.6, label="BIND / DMO")
    a.plot(KB, med(hr / dm), color="tab:blue", lw=1.4, label="Hydro-replaced / DMO")
    a.axhline(1, color="gray", lw=0.6, ls="--")
    a.set_title(f"{disp}  ($n={d['n']}$)")
    a.set_ylim(0.62, 1.10)
    if col == 0:
        a.set_ylabel(r"$S(k)=P(k)/P_{\rm DMO}(k)$")
        a.legend(fontsize=11, frameon=False, loc="lower left")

    # row 2 — P/P_Truth with 16-84 bands
    a = ax[1, col]
    for num, colr, lab in [(bi, "tab:orange", "BIND / Truth"),
                           (hr, "tab:blue", "Hydro-replaced / Truth")]:
        r = num / tr
        a.fill_between(KB, q(r, .16), q(r, .84), color=colr, alpha=0.20, lw=0)
        a.plot(KB, med(r), color=colr, lw=1.6, label=lab)
    a.plot(KB, med(dm / tr), color="0.45", lw=1.2, ls="--", label="DMO / Truth")
    a.axhline(1, color="k", lw=0.7, ls="--")
    a.set_ylim(0.73, 1.52)
    if col == 0:
        a.set_ylabel(r"$P/P_{\rm Truth}$")
        a.legend(fontsize=11, frameon=False, loc="upper left")

    # row 3 — BIND / hydro-replaced (paste-ceiling ratio) with 16-84 and 5-95
    a = ax[2, col]
    r = bi / hr
    a.fill_between(KB, q(r, .05), q(r, .95), color="tab:orange", alpha=0.12, lw=0)
    a.fill_between(KB, q(r, .16), q(r, .84), color="tab:orange", alpha=0.28, lw=0)
    a.plot(KB, med(r), color="tab:orange", lw=1.7, label="BIND / Hydro-replaced")
    a.axhline(1, color="k", lw=0.7, ls="--")
    a.set_ylim(0.80, 1.27)
    a.set_xlabel(r"$k$ [$h$/Mpc]")
    if col == 0:
        a.set_ylabel(r"$P_{\rm BIND}/P_{\rm hydro-repl}$")
        a.legend(fontsize=11, frameon=False, loc="upper left")

    for row in range(3):
        aa = ax[row, col]
        aa.set_xscale("log")
        aa.set_xlim(KB.min(), KNYQ)
        aa.grid(which="both", alpha=0.25)
        if col:
            aa.tick_params(labelleft=False)

save_fig(fig, "fig5b_total_field_pk")

# ── headline stats ──────────────────────────────────────────────────────────
def band_stats(num, den, lo, hi):
    """Per-sim mean ratio over native k in [lo,hi) -> median [q16,q84] over sims."""
    m = (kk >= lo) & (kk < hi)
    per_sim = (num[:, m] / den[:, m]).mean(1)
    return (np.median(per_sim), np.quantile(per_sim, .16), np.quantile(per_sim, .84))

print("\n== band medians (per-sim band-mean ratio; median [16,84] over sims) ==")
for s, disp in SUITES:
    d = data[s]
    pairs = [("BIND/Truth", d["bi_nat"], d["tr_nat"]),
             ("HydroRepl/Truth", d["hr_nat"], d["tr_nat"]),
             ("BIND/HydroRepl", d["bi_nat"], d["hr_nat"])]
    for lab, num, den in pairs:
        cells = []
        for lo, hi in STAT_BANDS:
            m0, l0, h0 = band_stats(num, den, lo, hi)
            cells.append(f"{m0:.3f} [{l0:.3f},{h0:.3f}]")
        print(f"{disp:5s} {lab:16s} " + " | ".join(cells))

print("\n== peak of each rebinned median ratio ==")
for s, disp in SUITES:
    d = data[s]
    for lab, num, den in [("BIND/Truth", d["bi"], d["tr"]),
                          ("HydroRepl/Truth", d["hr"], d["tr"]),
                          ("BIND/HydroRepl", d["bi"], d["hr"])]:
        m = med(num / den)
        i = int(np.argmax(m))
        print(f"{disp:5s} {lab:16s} peak={m[i]:.3f} at k={KB[i]:.1f}")

print("\n== CV BIND/Truth median first exceeds threshold ==")
mcv = med(data["CV"]["bi"] / data["CV"]["tr"])
for thr in (0.02, 0.05, 0.10):
    above = np.where(np.abs(mcv - 1) > thr)[0]
    print(f"  >{thr*100:.0f}%: k={KB[above[0]]:.2f}" if len(above) else f"  >{thr*100:.0f}%: never")

print("\n== suppression recovery at truth-minimum (rebinned medians) ==")
for s, disp in SUITES:
    d = data[s]
    St, Sb, Sh = med(d["tr"] / d["dm"]), med(d["bi"] / d["dm"]), med(d["hr"] / d["dm"])
    i = int(np.argmin(St))
    fb = (1 - Sb[i]) / (1 - St[i])
    fh = (1 - Sh[i]) / (1 - St[i])
    print(f"{disp:5s} k*={KB[i]:.1f} S_truth={St[i]:.3f} S_BIND={Sb[i]:.3f} "
          f"S_HR={Sh[i]:.3f} frac_BIND={fb:.3f} frac_HR={fh:.3f}")
