#!/usr/bin/env python3
"""fig_p15_sweep: stellar residual vs VariableWindSpecMomentum (p15, 0-based idx 14).

Referee figure explaining the CV/1P vs SB35 stellar-mass offset: the fiducial
(CV, 1P baseline) sits at the PRIOR MINIMUM of VariableWindSpecMomentum
(p15 = 0 of [0, 4000]), while SB35 samples the full range.

Left panel : 1P p15 sweep — per-sim median stellar residual (r <= R200c) vs the
             parameter value {0 = fiducial (1P_p1_0), 1000, 2000, 3000, 4000
             (1P_p15_1..4)}, bootstrap-over-halos error bars; horizontal bands =
             CV and SB35 pooled medians +- 1 sigma from a cluster bootstrap over
             sims.
Right panel: SB35 per-sim median stellar residual vs the sim's normalised p15
             coordinate (value/4000); fiducial edge at 0 marked; Spearman rho.

All numbers from the paper cache selected by the PAPER_* env vars:
mass_table.pkl (per-halo masses, already cut at M200c >= 1e13) +
mass_param.pkl sim_table (per-sim params). The sim-level join is asserted on
a shared mass column (true_logmean_Stars_trained == per-sim mean log10
truth_Stars, full patch).

Run:
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    export PAPER_SUITE_ROOT=... PAPER_MODEL_SUBDIR=... PAPER_MASS_DIR=... PAPER_MODEL_TAG=...
    python fig_p15_sweep.py
"""
import os
import pickle
import sys
from pathlib import Path

# Model selection comes from the environment — no hardcoded model defaults.
_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR", "PAPER_MASS_DIR", "PAPER_MODEL_TAG")
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    sys.exit(f"set env vars before running: {' '.join(_missing)}")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import paper_config as C

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook"])
except Exception:
    pass

SUITE_COLORS = C.SUITE_COLORS
FIG_DIR = Path("/mnt/home/mlee1/vdm_bind2/examples/paper_figures")
FIG_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
NBOOT = 2000
P15_MAX = 4000.0
SWEEP = {"1P_p1_0": 0.0, "1P_p15_1": 1000.0, "1P_p15_2": 2000.0,
         "1P_p15_3": 3000.0, "1P_p15_4": 4000.0}


def save_fig(fig, name, ext=("pdf", "png")):
    for e in ext:
        out = FIG_DIR / f"{name}.{e}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")


def stellar_resid(sub):
    """Per-halo stellar residual (gen/truth - 1) within R200c; finite only."""
    t = sub["truth_Stars_rvir"].to_numpy()
    g = sub["gen_Stars_rvir"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (g - t) / t
    return r[np.isfinite(r)]


def boot_median_halos(r, nboot=NBOOT):
    """sigma of the median under a bootstrap over halos."""
    meds = np.median(RNG.choice(r, size=(nboot, len(r)), replace=True), axis=1)
    return meds.std()


def cluster_boot_median(tbl, suite, nboot=NBOOT):
    """Pooled median residual +- sigma from a cluster bootstrap over sims."""
    sub = tbl[tbl["suite"] == suite]
    sims = sub["sim_id"].unique()
    per_sim = {s: stellar_resid(sub[sub["sim_id"] == s]) for s in sims}
    pooled = np.concatenate(list(per_sim.values()))
    meds = np.empty(nboot)
    for i in range(nboot):
        pick = RNG.choice(sims, size=len(sims), replace=True)
        meds[i] = np.median(np.concatenate([per_sim[s] for s in pick]))
    return np.median(pooled), meds.std(), len(sims), len(pooled)


def main():
    tbl = pickle.load(open(C.CACHE_DIR / "mass_table.pkl", "rb"))
    assert (tbl["log_m200c"] >= 13.0).all(), "spine mass_table has halos below 1e13"
    st = pickle.load(open(C.CACHE_DIR / "mass_param.pkl", "rb"))["sim_table"]

    # ── cross-check the sweep parameter values against mass_param.pkl ──────
    p15 = st.set_index(["suite", "sim_id"])["p15"]
    for sim, val in SWEEP.items():
        assert p15[("1P", sim)] == val, f"{sim}: p15={p15[('1P', sim)]} != {val}"
    assert (st[st.suite == "CV"]["p15"] == 0).all(), "CV p15 not at prior minimum"

    # ── sim-level join assertion on a shared mass column ───────────────────
    # sim_table's true_logmean_Stars_trained is the per-sim mean log10 of the
    # full-patch truth stellar mass over the trained halos; recompute it from
    # mass_table and require exact agreement before trusting the p15 join.
    recomp = (
        tbl.assign(logst=np.log10(tbl["truth_Stars"].where(tbl["truth_Stars"] > 0)))
        .groupby(["suite", "sim_id"])["logst"].mean().rename("recomp").reset_index()
    )
    j = st.merge(recomp, on=["suite", "sim_id"], validate="one_to_one")
    assert np.allclose(j["true_logmean_Stars_trained"], j["recomp"]), \
        "sim-level join failed the shared-mass assertion"
    print(f"join OK: {len(j)} sims, logmean-Stars assertion exact")

    # ── left panel data: 1P sweep ──────────────────────────────────────────
    xs, med, err, nh = [], [], [], []
    for sim, val in sorted(SWEEP.items(), key=lambda kv: kv[1]):
        r = stellar_resid(tbl[(tbl["suite"] == "1P") & (tbl["sim_id"] == sim)])
        xs.append(val)
        med.append(np.median(r))
        err.append(boot_median_halos(r))
        nh.append(len(r))
        print(f"  {sim:9s} p15={val:6.0f}  N={len(r):3d}  "
              f"median = {100 * med[-1]:+6.2f} +- {100 * err[-1]:.2f} %")
    xs, med, err = map(np.asarray, (xs, med, err))

    cv_med, cv_sig, cv_ns, cv_nh = cluster_boot_median(tbl, "CV")
    sb_med, sb_sig, sb_ns, sb_nh = cluster_boot_median(tbl, "Test")
    gap, gap_sig = cv_med - sb_med, np.hypot(cv_sig, sb_sig)
    print(f"  CV   pooled median = {100 * cv_med:+.2f} +- {100 * cv_sig:.2f} %"
          f"  ({cv_ns} sims, {cv_nh} halos)")
    print(f"  SB35 pooled median = {100 * sb_med:+.2f} +- {100 * sb_sig:.2f} %"
          f"  ({sb_ns} sims, {sb_nh} halos)")
    print(f"  CV - SB35 gap      = {100 * gap:+.2f} +- {100 * gap_sig:.2f} %"
          f"  ({gap / gap_sig:.1f} sigma)")

    # ── right panel data: SB35 per-sim medians vs normalised p15 ───────────
    sb = tbl[tbl["suite"] == "Test"]
    rows = []
    for sim in sb["sim_id"].unique():
        r = stellar_resid(sb[sb["sim_id"] == sim])
        rows.append((p15[("Test", sim)] / P15_MAX, np.median(r), len(r)))
    sbx, sby, sbn = map(np.asarray, zip(*rows))
    rho, pval = spearmanr(sbx, sby)
    print(f"  SB35: {len(sbx)} sims, Spearman rho(p15, median stellar resid)"
          f" = {rho:+.2f}  (p = {pval:.1e})")

    # ── figure ─────────────────────────────────────────────────────────────
    # Separate y scales: the left panel spans ~+-15% (sweep + reference bands),
    # the right panel must accommodate the full SB35 per-sim scatter.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: reference bands then sweep points
    for m, s, suite, lbl in [(cv_med, cv_sig, "CV", "CV median $\\pm 1\\sigma$"),
                             (sb_med, sb_sig, "Test", "SB35 median $\\pm 1\\sigma$")]:
        axL.axhspan(m - s, m + s, color=SUITE_COLORS[suite], alpha=0.20, lw=0)
        axL.axhline(m, color=SUITE_COLORS[suite], lw=1.2, label=lbl)
    axL.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axL.plot(xs, med, color=SUITE_COLORS["1P"], lw=1.2, alpha=0.7, zorder=4)
    axL.errorbar(xs[1:], med[1:], yerr=err[1:], fmt="o", ms=7,
                 color=SUITE_COLORS["1P"], mec="k", mew=0.8, ecolor="k",
                 capsize=3, lw=1.2, zorder=5, label="1P $p_{15}$ sweep")
    axL.errorbar(xs[:1], med[:1], yerr=err[:1], fmt="s", ms=8,
                 color=SUITE_COLORS["1P"], mec="k", mew=0.8, ecolor="k",
                 capsize=3, lw=1.2, zorder=6, label="fiducial (1P_p1_0)")
    axL.set_xticks(list(SWEEP.values()))
    axL.set_xlim(-280, 4280)
    axL.set_ylim(-0.155, 0.26)
    axL.set_xlabel(r"VariableWindSpecMomentum  $p_{15}$")
    axL.set_ylabel(r"median $\Delta M_\star / M_{\star,\mathrm{Truth}}$"
                   "\n" r"($r \leq R_{200}$)")
    axL.legend(fontsize=10, loc="upper right", framealpha=0.95)
    axL.set_title(r"1P sweep")

    # right: SB35 scatter vs normalised coordinate
    axR.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    axR.axvline(0.0, color="k", lw=1.0, ls=":", alpha=0.8)
    axR.scatter(sbx, sby, s=22, color=SUITE_COLORS["Test"], alpha=0.75,
                edgecolors="k", linewidths=0.4, label="SB35 sims", zorder=4)
    axR.errorbar([0.0], [cv_med], yerr=[cv_sig], fmt="D", ms=8,
                 color=SUITE_COLORS["CV"], mec="k", mew=0.8, ecolor="k",
                 capsize=3, zorder=6, label="CV (fiducial, $p_{15}=0$)")
    axR.text(0.035, 0.97, "fiducial = prior edge", rotation=90, fontsize=9,
             ha="left", va="top", transform=axR.get_xaxis_transform())
    axR.annotate(rf"Spearman $\rho = {rho:+.2f}$",
                 xy=(0.97, 0.96), xycoords="axes fraction",
                 ha="right", va="top", fontsize=11)
    axR.set_xlim(-0.07, 1.05)
    axR.set_xlabel(r"$p_{15}\,/\,4000$")
    axR.set_ylabel(r"median $\Delta M_\star / M_{\star,\mathrm{Truth}}$"
                   "  ($r \leq R_{200}$)", fontsize=10)
    axR.legend(fontsize=10, loc="lower right", framealpha=0.95)
    axR.set_title("SB35 per-sim medians")

    plt.tight_layout()
    save_fig(fig, "fig_p15_sweep")


if __name__ == "__main__":
    main()
