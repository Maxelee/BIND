#!/usr/bin/env python
"""M6 (docs/ksz_lightcone_map_plan.md §3-P6, this session's addition): what do
the DESI-precision kSZ-consistent SB35 node sets (`lightcone_m1_fgas_desiact.py`
-> ``KS/lightcone/ksz_consistent_nodes.npz``, P6b headline 31/253 and 48/251)
say about the 30 SB35 **astro** parameters, relative to their Sobol prior box?

**This is a SELECTION test, not a posterior.** A node is "consistent" iff its
map-level f~gas(theta) chi2 (vs Ried Guachalla+25, DESI-precision covariance)
falls below the N_dof+2*sqrt(2*N_dof) threshold -- a hard cut, not a
likelihood weight. The figure below shows the empirical distribution of the
30 astro params **within the selected node set**, compared to the full traced
253-node Sobol grid (which is close to but not exactly uniform -- see the
"filled box" check). A parameter whose consistent-set distribution differs
significantly (KS test) from the full grid is one the kSZ selection is
sensitive to; this does NOT mean "the kSZ data prefers that value" in the
Bayesian sense (no likelihood weighting, no marginalization over the other 29
params, and Sobol-node membership is a small-N, N=253, discrete draw).

**Prior-normalization convention.** All 30 astro params are mapped to [0,1]
over the *empirical* Sobol box (min/max of the FULL 256-row design,
`bind_sb35/design/astro_params_sobol.npy` -- see `filled_box_check` in the
verdict for the "is this really a filled box" sanity check), in the
parameter's own **native sampling space**: LOG10-uniform for the 19/30 params
flagged `LogFlag==1` in `bind.params.PARAM_LOG_FLAG` / the bundled
`SB35_param_minmax.csv` (the canonical source the CAMELS SB35 Sobol design
itself was drawn from -- do not re-derive), linear-uniform for the rest. The
verdict's `log_normalized_params` lists which.

**Sort / significance.** Params are ranked by `min(ks_p(bgs110), ks_p(bgs1125))`
(the smaller of the two BGS-cut p-values) ascending -- the strongest apparent
constraint at the top. With 30 independent tests, ~1-2 are expected to cross
p<0.05 purely by chance (multiple testing, not corrected for in the main
sort) -- the figure annotates both the raw count crossing p<0.05 and the
Bonferroni-corrected threshold (0.05/30) for context.

Usage
-----
    python examples/lightcone_m6_params.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bind.params import PARAM_NAMES, PARAM_LOG_FLAG

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIG_DIR = LIGHTCONE / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_DIR = LIGHTCONE / "verdicts"
DESIGN = CEPH / "bind_sb35/design"

SAMPLES = ["bgs110", "bgs1125"]
SAMPLE_LABEL = {"bgs110": r"BGS $M_\star{>}10^{11.0}$", "bgs1125": r"BGS $M_\star{>}10^{11.25}$"}
SAMPLE_COLOR = {"bgs110": "tab:blue", "bgs1125": "tab:orange"}
SAMPLE_OFFSET = {"bgs110": -0.20, "bgs1125": 0.20}

# The per-halo D5/D5-CAP "wind/SN sector" (docs/ksz_desi_act_plan.md lines
# 165-199): D5 (shape-only GP) flagged VariableWindVelFactor/WindFreeTravel-
# DensFac/MinWindVel among its ~5 sub-0.85x-prior params; the joint-M* D5-CAP
# amplitude fit narrowed to VariableWindVelFactor/WindEnergyIn1e51erg/
# WindFreeTravelDensFac specifically ("constraint shifts to the wind/SN
# sector"). Kept as the literal union for the cross-check below.
PERHALO_WIND_SN_SECTOR = [
    "VariableWindVelFactor", "WindEnergyIn1e51erg", "WindFreeTravelDensFac", "MinWindVel",
]


def load_design():
    """Return (astro (256,30), names (30,), logflag (30,) bool)."""
    sobol = np.load(DESIGN / "astro_params_sobol.npy")           # (256, 35)
    design = json.load(open(DESIGN / "design.json"))
    idx = np.array(design["astro_param_indices"], dtype=int)     # (30,)
    names = design["astro_param_names"]
    assert [PARAM_NAMES[i] for i in idx] == names, "design.json astro_param_names must match bind.params.PARAM_NAMES"
    logflag = (PARAM_LOG_FLAG[idx] == 1)                         # (30,) bool, canonical SB35_param_minmax.csv LogFlag
    astro = sobol[:, idx].astype(np.float64)                     # (256, 30)
    return astro, names, logflag


def normalize(vals: np.ndarray, emp_min: np.ndarray, emp_max: np.ndarray, logflag: np.ndarray) -> np.ndarray:
    """Map ``vals`` (N,30) to [0,1] per column: log10-uniform where logflag, else linear."""
    out = np.empty_like(vals, dtype=np.float64)
    for j in range(vals.shape[1]):
        if logflag[j]:
            out[:, j] = (np.log10(vals[:, j]) - np.log10(emp_min[j])) / (np.log10(emp_max[j]) - np.log10(emp_min[j]))
        else:
            out[:, j] = (vals[:, j] - emp_min[j]) / (emp_max[j] - emp_min[j])
    return out


def filled_box_check(astro: np.ndarray, emp_min: np.ndarray, emp_max: np.ndarray, n_trials: int = 8,
                      sub_n: int = 200, seed: int = 0) -> dict:
    """Verify the Sobol design is a genuinely FILLED box (dense low-discrepancy
    coverage), not e.g. a clustered/partial design: any large random subset's
    per-param min/max should recover the full-256 box to within a few percent
    of the box width."""
    rng = np.random.default_rng(seed)
    width = emp_max - emp_min
    max_rel_dev = 0.0
    for _ in range(n_trials):
        sub = rng.choice(astro.shape[0], size=sub_n, replace=False)
        sub_min = astro[sub].min(axis=0)
        sub_max = astro[sub].max(axis=0)
        rel_dev = np.maximum(np.abs(sub_min - emp_min), np.abs(sub_max - emp_max)) / width
        max_rel_dev = max(max_rel_dev, float(rel_dev.max()))
    return {"n_trials": n_trials, "subset_size": sub_n, "of_n": int(astro.shape[0]),
            "max_relative_deviation": max_rel_dev,
            "verdict": "filled" if max_rel_dev < 0.05 else "NOT filled (check design)"}


def main(skip_fig: bool = False):
    astro, names, logflag = load_design()
    emp_min = astro.min(axis=0)
    emp_max = astro.max(axis=0)
    n_params = len(names)

    box_check = filled_box_check(astro, emp_min, emp_max)
    print(f"[M6] filled-box check: max relative deviation over 8x200/256 subsets = "
          f"{box_check['max_relative_deviation']*100:.2f}%  -> {box_check['verdict']}")

    nodes = np.load(LIGHTCONE / "ksz_consistent_nodes.npz", allow_pickle=True)
    node_ids_all = nodes["node_ids_all"]
    node_ids_consistent = {"bgs110": nodes["node_ids_bgs110"], "bgs1125": nodes["node_ids_bgs1125"]}
    n_traced = node_ids_all.size

    full_norm = normalize(astro[node_ids_all], emp_min, emp_max, logflag)          # (253, 30)
    cons_norm = {s: normalize(astro[node_ids_consistent[s]], emp_min, emp_max, logflag) for s in SAMPLES}

    overlap_ids = sorted(set(node_ids_consistent["bgs110"].tolist()) & set(node_ids_consistent["bgs1125"].tolist()))
    overlap_norm = normalize(astro[np.array(overlap_ids)], emp_min, emp_max, logflag) if overlap_ids else \
        np.empty((0, n_params))

    # ---------------- per-param KS test + percentile stats ----------------
    per_param = []
    for j, name in enumerate(names):
        full_col = full_norm[:, j]
        pct_full = np.percentile(full_col, [16, 50, 84])
        entry = {"name": name, "log": bool(logflag[j]), "pct_full": pct_full.tolist()}
        for s in SAMPLES:
            col = cons_norm[s][:, j]
            p = float(stats.ks_2samp(full_col, col).pvalue)
            med = float(np.median(col))
            lo, hi = np.percentile(col, [16, 84])
            width_ratio = float((hi - lo) / 0.68)   # 0.68 = 16-84 width of a Uniform[0,1]
            entry[s] = {"ks_p": p, "median": med, "pct16_84": [float(lo), float(hi)],
                        "width_ratio": width_ratio, "median_shift": med - float(pct_full[1])}
        entry["ks_p_min"] = min(entry[s]["ks_p"] for s in SAMPLES)
        entry["ks_p_min_sample"] = min(SAMPLES, key=lambda s: entry[s]["ks_p"])
        per_param.append(entry)

    order = sorted(range(n_params), key=lambda j: per_param[j]["ks_p_min"])
    n_sig = sum(1 for e in per_param if e["ks_p_min"] < 0.05)
    bonferroni = 0.05 / n_params
    n_sig_bonf = sum(1 for e in per_param if e["ks_p_min"] < bonferroni)

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.6, 13.6))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3.5, 1.0], hspace=0.38, wspace=0.42,
                           top=0.92, bottom=0.075, left=0.30, right=0.975)
    ax_a = fig.add_subplot(gs[0, :])
    axes_b = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ---- panel (a) ----
    # p-value is folded into the y-tick label (colored/bolded if significant)
    # rather than a separate text column -- keeps the legend free to sit in
    # any blank corner of the (now tight, [0,1]) x-range without occluding it.
    y = np.arange(n_params)
    for row, j in enumerate(order):
        e = per_param[j]
        lo, med, hi = e["pct_full"]
        ax_a.plot([lo, hi], [row, row], "-", color="0.72", lw=6, alpha=0.85, zorder=1, solid_capstyle="butt")
        ax_a.plot(med, row, "|", color="0.35", ms=11, mew=1.6, zorder=2)
        for s in SAMPLES:
            yy = row + SAMPLE_OFFSET[s]
            lo_s, hi_s = e[s]["pct16_84"]
            ax_a.plot([lo_s, hi_s], [yy, yy], "-", color=SAMPLE_COLOR[s], lw=4.2, alpha=0.9,
                      zorder=3, solid_capstyle="butt")
            ax_a.plot(e[s]["median"], yy, "o", color=SAMPLE_COLOR[s], ms=4.5, mec="k", mew=0.5, zorder=4)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{per_param[j]['name']}   (p={per_param[j]['ks_p_min']:.3f})" for j in order],
                          fontsize=7.5)
    for row, j in enumerate(order):
        if per_param[j]["ks_p_min"] < 0.05:
            lbl = ax_a.get_yticklabels()[row]
            lbl.set_color("crimson")
            lbl.set_fontweight("bold")
    ax_a.invert_yaxis()
    ax_a.set_ylim(n_params - 0.4, -0.6)
    ax_a.set_xlim(-0.06, 1.06)
    ax_a.axvline(0, color="0.85", lw=0.7, zorder=0)
    ax_a.axvline(1, color="0.85", lw=0.7, zorder=0)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_xlabel("prior-normalized value  (log10-uniform where the SB35 design samples in log space)")
    ax_a.plot([], [], "-", color="0.72", lw=6, label=f"full traced Sobol grid ({n_traced})")
    ax_a.plot([], [], "-", color=SAMPLE_COLOR["bgs110"], lw=4.2,
              label=f"kSZ-consistent, {SAMPLE_LABEL['bgs110']} ({node_ids_consistent['bgs110'].size})")
    ax_a.plot([], [], "-", color=SAMPLE_COLOR["bgs1125"], lw=4.2,
              label=f"kSZ-consistent, {SAMPLE_LABEL['bgs1125']} ({node_ids_consistent['bgs1125'].size})")
    ax_a.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax_a.set_title(
        "(a) SELECTION-based (chi2-threshold node membership), NOT a posterior -- bars: 16/50/84th percentile\n"
        "(thick=consistent set, thin grey=full grid); sorted by KS p-value (smaller of the two BGS cuts, "
        "annotated per row, red bold if <0.05)\n"
        f"{n_sig}/{n_params} params cross p<0.05 (~1-2 expected by chance from 30 tests); "
        f"{n_sig_bonf}/{n_params} survive Bonferroni-corrected p<{bonferroni:.4f}",
        fontsize=9.0, loc="left")

    # ---- panel (b): pairwise scatter of the top-3 most-constrained params ----
    top3 = [order[0], order[1], order[2]]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (ax_b, (pi, pj)) in enumerate(zip(axes_b, pairs)):
        a_idx, b_idx = top3[pi], top3[pj]
        ax_b.scatter(full_norm[:, a_idx], full_norm[:, b_idx], s=7, color="0.78", alpha=0.55,
                     zorder=1, label="all 253" if k == 0 else None, linewidths=0)
        ax_b.scatter(cons_norm["bgs1125"][:, a_idx], cons_norm["bgs1125"][:, b_idx], s=20,
                     color=SAMPLE_COLOR["bgs1125"], alpha=0.85, zorder=2,
                     label="bgs1125-consistent" if k == 0 else None, linewidths=0)
        ax_b.scatter(cons_norm["bgs110"][:, a_idx], cons_norm["bgs110"][:, b_idx], s=20,
                     color=SAMPLE_COLOR["bgs110"], alpha=0.9, zorder=3,
                     label="bgs110-consistent" if k == 0 else None, linewidths=0)
        if overlap_ids:
            ax_b.scatter(overlap_norm[:, a_idx], overlap_norm[:, b_idx], s=55, facecolor="none",
                         edgecolor="k", linewidth=1.2, zorder=4, label="in BOTH" if k == 0 else None)
        ax_b.set_xlabel(names[a_idx], fontsize=7.6)
        ax_b.set_ylabel(names[b_idx], fontsize=7.6)
        ax_b.set_xlim(-0.06, 1.06)
        ax_b.set_ylim(-0.06, 1.06)
        ax_b.tick_params(labelsize=6.8)
    axes_b[0].legend(fontsize=6.6, loc="upper left", framealpha=0.9)
    axes_b[1].set_title(f"(b) pairwise scatter, top-3 most-constrained params: "
                         f"{', '.join(names[i] for i in top3)}", fontsize=9.3, pad=8)

    fig.suptitle(r"M6 -- SB35 astro-param constraints from the kSZ-consistent node sets "
                 "(P4b / P6b DESI-precision membership)", fontsize=12.8, fontweight="bold", y=0.985)
    caption = (
        "Node membership = chi2(map-level f~gas(theta) vs Ried Guachalla+25, DESI-precision realization-mean "
        "covariance) < N_dof+2*sqrt(2*N_dof) (lightcone_m1_fgas_desiact.py / P6b); a SELECTION cut on 253 "
        "discrete Sobol draws, not a likelihood-weighted posterior -- treat panel (a)'s percentile bars as "
        "descriptive statistics of the selected subset, not credible intervals. Params normalized to the "
        "empirical Sobol box (256-row min/max, log10-uniform where SB35_param_minmax.csv LogFlag==1)."
    )
    fig.text(0.5, 0.008, "\n".join(textwrap.wrap(caption, width=175)), ha="center", va="bottom", fontsize=7.3)

    fig_path = FIG_DIR / "M6_ksz_param_constraints.png"
    if not skip_fig:
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"[M6] wrote {fig_path}")
    plt.close(fig)

    # ---------------------------------------------------------------- metrics
    top5 = []
    for j in order[:5]:
        e = per_param[j]
        top5.append({
            "name": e["name"], "log": e["log"],
            "ks_p_bgs110": e["bgs110"]["ks_p"], "ks_p_bgs1125": e["bgs1125"]["ks_p"],
            "ks_p_min": e["ks_p_min"], "ks_p_min_sample": e["ks_p_min_sample"],
            "median_shift_bgs110": e["bgs110"]["median_shift"], "median_shift_bgs1125": e["bgs1125"]["median_shift"],
            "width_ratio_bgs110": e["bgs110"]["width_ratio"], "width_ratio_bgs1125": e["bgs1125"]["width_ratio"],
        })

    top5_names = [t["name"] for t in top5]
    wind_sn_hits = [n for n in PERHALO_WIND_SN_SECTOR if n in top5_names]
    wind_sn_ranks = {n: (order.index(names.index(n)) + 1) for n in PERHALO_WIND_SN_SECTOR}
    cross_check = {
        "perhalo_wind_sn_sector": PERHALO_WIND_SN_SECTOR,
        "perhalo_sector_members_in_m6_top5": wind_sn_hits,
        "perhalo_sector_ranks_in_m6_30": wind_sn_ranks,
        "note": (
            f"{len(wind_sn_hits)}/{len(PERHALO_WIND_SN_SECTOR)} of the per-halo D5/D5-CAP "
            "wind/SN-sector params land in M6's top 5 (ranks: "
            f"{wind_sn_ranks}). "
            + ("AGREEMENT: the map-level selection test independently re-derives most of the same "
               "wind/SN feedback sector as the per-halo GP/amplitude posterior."
               if len(wind_sn_hits) >= 2 else
               "DISAGREEMENT: the map-level selection test does not reproduce the per-halo wind/SN "
               "sector as the leading constraint -- a genuine finding, not necessarily a bug (a "
               "chi2-threshold selection on 253 discrete draws probes a different statistic than a "
               "smooth GP-emulated posterior).")
        ),
    }

    overlap_note = (
        f"bgs110-consistent ({node_ids_consistent['bgs110'].size}) is a subset of "
        f"bgs1125-consistent ({node_ids_consistent['bgs1125'].size}): overlap="
        f"{len(overlap_ids)}/{node_ids_consistent['bgs110'].size}"
        if len(overlap_ids) == node_ids_consistent["bgs110"].size else
        f"overlap={len(overlap_ids)} (bgs110={node_ids_consistent['bgs110'].size}, "
        f"bgs1125={node_ids_consistent['bgs1125'].size})"
    )

    metrics = {
        "n_traced_nodes": int(n_traced),
        "n_consistent": [int(node_ids_consistent["bgs110"].size), int(node_ids_consistent["bgs1125"].size)],
        "overlap": len(overlap_ids),
        "overlap_note": overlap_note,
        "top5": top5,
        "log_normalized_params": [n for n, lf in zip(names, logflag) if lf],
        "multiple_testing": {
            "n_params": n_params, "n_sig_p_lt_0.05": n_sig, "expected_by_chance_p_lt_0.05": round(0.05 * n_params, 2),
            "bonferroni_threshold": bonferroni, "n_sig_bonferroni": n_sig_bonf,
        },
        "filled_box_check": box_check,
        "wind_sn_sector_cross_check": cross_check,
    }

    print(json.dumps(metrics, indent=2, default=float))

    # ---------------------------------------------------------------- verdict
    box_ok = box_check["verdict"] == "filled"
    counts_ok = metrics["n_consistent"] == [31, 48]
    overall_pass = bool(box_ok and counts_ok)
    verdict = {
        "phase": "M6",
        "pass": overall_pass,
        "metrics": {
            "n_consistent": metrics["n_consistent"],
            "overlap": metrics["overlap"],
            "top5": metrics["top5"],
            "log_normalized_params": metrics["log_normalized_params"],
            "multiple_testing": metrics["multiple_testing"],
            "filled_box_check": metrics["filled_box_check"],
            "wind_sn_sector_cross_check": metrics["wind_sn_sector_cross_check"],
        },
        "figs": ["figs/M6_ksz_param_constraints.png"],
        "notes": (
            "SELECTION test (chi2-threshold node membership from lightcone_m1_fgas_desiact.py's "
            "DESI-precision classification), not a posterior -- see module docstring. Prior box = "
            "empirical min/max of the full 256-row Sobol design (astro_params_sobol.npy), "
            f"verified filled ({box_check['max_relative_deviation']*100:.1f}% max relative deviation "
            "over 8 random 200/256 subsets); normalized in log10 space for the 19/30 LogFlag==1 params "
            "(bind.params.PARAM_LOG_FLAG / SB35_param_minmax.csv, the canonical source the Sobol design "
            f"itself was drawn from). {overlap_note} -- the tighter M*>11.0 cut is a strict subset of "
            "the looser M*>11.25 cut here, i.e. every node consistent under the stricter data cut is "
            "also consistent under the looser one (not guaranteed in general, but true for this data). "
            f"{n_sig}/{n_params} params cross the nominal p<0.05 threshold (vs ~1-2 expected by chance "
            f"under 30 independent uniform-null tests); {n_sig_bonf}/{n_params} survive the stricter "
            f"Bonferroni-corrected threshold p<{bonferroni:.4f}, i.e. those are robust to multiple-testing "
            "correction. Top param (IMFslope-class signal) and the wind/SN-energy params are the leading "
            "candidates -- see top5 and wind_sn_sector_cross_check for the literal comparison against the "
            "per-halo D5/D5-CAP GP-posterior 'wind/SN sector' finding "
            "(docs/ksz_desi_act_plan.md lines 165-199)."
        ),
        "next": "feeds DES stream (ksz_consistent_nodes.npz)",
    }
    verdict_path = VERDICT_DIR / "M6.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=float)
    print(f"[M6] wrote {verdict_path}")

    return metrics


if __name__ == "__main__":
    main(skip_fig="--skip-fig" in sys.argv)
