#!/usr/bin/env python3
"""mass_error_table.tex: AASTeX deluxetable of integrated-mass residual stats.

Per component (DM / Gas / Stars / Total) x aperture (r <= R200c, full patch) x
suite (CV / 1P / SB35): median [%] with a cluster-bootstrap-over-sims error, the 16-84
percentile range [%], and f50 = frac(|delta| > 0.5) [%], plus N_halo (N_sim).
delta = gen/truth - 1, UNCLIPPED (the [-1,1] clip in fig_mass_error is
display-only).

Everything is recomputed from the paper-cache mass_table.pkl selected by the
PAPER_* env vars (M200c >= 1e13). Optionally, a previously written table for
the SAME model can be passed via PAPER_PRIOR_TEX; it is parsed and diffed
against the recompute before the adapted table is written — the script fails
loudly if median / 16-84 / f50 disagree. Leave PAPER_PRIOR_TEX unset to skip
the diff (e.g. on a model switch, where the numbers legitimately change).

Run:
    source /mnt/home/mlee1/venvs/torch3/bin/activate
    export PAPER_SUITE_ROOT=... PAPER_MODEL_SUBDIR=... PAPER_MASS_DIR=... PAPER_MODEL_TAG=...
    python mass_error_table.py       # writes ./mass_error_table.tex
"""
import os
import pickle
import re
import sys
from pathlib import Path

# Model selection comes from the environment — no hardcoded model defaults.
_REQUIRED_ENV = ("PAPER_SUITE_ROOT", "PAPER_MODEL_SUBDIR", "PAPER_MASS_DIR", "PAPER_MODEL_TAG")
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    sys.exit(f"set env vars before running: {' '.join(_missing)}")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "paper_cache"))

import numpy as np

import paper_config as C

HERE = Path(__file__).resolve().parent
OUT_TEX = HERE / "mass_error_table.tex"
_prior = os.environ.get("PAPER_PRIOR_TEX", "")
PRIOR_TEX = Path(_prior) if _prior else None

RNG = np.random.default_rng(42)
NBOOT = 2000

COMPONENTS = ["DM_hydro", "Gas", "Stars", "Total"]
COMP_TEX = {"DM_hydro": "DM", "Gas": "Gas", "Stars": "Stars", "Total": "Total"}
APERTURES = [("_rvir", r"$<R_{200c}$"), ("", "Full patch")]
SUITES = list(C.SUITES)
SUITE_TEX = {"CV": "CV", "1P": "1P", "Test": "SB35"}


def residuals(tbl, suite, ch, suffix):
    """(residual array, matching sim_id array) — finite residuals only."""
    sub = tbl[tbl["suite"] == suite]
    if ch == "Total":
        t = sum(sub[f"truth_{c}{suffix}"] for c in C.MASS_CHANNELS).to_numpy()
        g = sum(sub[f"gen_{c}{suffix}"] for c in C.MASS_CHANNELS).to_numpy()
    else:
        t = sub[f"truth_{ch}{suffix}"].to_numpy()
        g = sub[f"gen_{ch}{suffix}"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (g - t) / t
    m = np.isfinite(r)
    return r[m], sub["sim_id"].to_numpy()[m]


def row_stats(r, sims):
    """(median%, med_err%, p16%, p84%, f20%, f50%) — all in percent.

    The median error is a CLUSTER bootstrap over simulations (resample sims
    with replacement, pool their halos, take the median): in 1P/SB35 the
    parameters vary per sim, so halos within a sim are correlated and a plain
    halo bootstrap underestimates the error (matches the prior table).
    """
    med = 100 * np.median(r)
    uniq = np.unique(sims)
    per_sim = {s: r[sims == s] for s in uniq}
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        boot[i] = np.median(np.concatenate([per_sim[s] for s in pick]))
    med_err = 100 * boot.std()
    p16, p84 = 100 * np.percentile(r, [16, 84])
    f20 = 100 * np.mean(np.abs(r) > 0.2)
    f50 = 100 * np.mean(np.abs(r) > 0.5)
    return med, med_err, p16, p84, f20, f50


def parse_prior(path):
    """Prior tex data rows -> dict keyed by (aperture_idx, comp, suite).

    Parses exactly the layout this script writes below, i.e. seven columns:

        aperture & component & suite & N (N_sim) & $med\\pm err$ & $[p16,p84]$ & f50

    The parser used to expect an older eight-column layout that carried an f20
    column and put N last. Every row of the current table failed its cell-count
    check, so `prior table rows parsed: 0` and the advertised "fails loudly if
    the numbers disagree" guard silently never fired. If you change the writer,
    change this too -- the two formats must stay in lockstep.
    """
    rows, ap_i, comp = {}, -1, None
    pat = re.compile(
        r"\$([+-][\d.]+)\\pm([\d.]+)\$\s*&\s*\$\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]\$"
        r"\s*&\s*([\d.]+)"
    )
    for line in path.read_text().splitlines():
        if "&" not in line or "colhead" in line:
            continue
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")]
        if len(cells) < 7:
            continue
        m = pat.search(line)
        if m is None:
            continue
        if cells[0]:  # new aperture block
            ap_i += 1
        if cells[1]:
            comp = cells[1]
        suite = cells[2]
        n_halo = int(cells[3].split()[0])
        rows[(ap_i, comp, suite)] = dict(
            med=float(m.group(1)), err=float(m.group(2)),
            p16=float(m.group(3)), p84=float(m.group(4)),
            f50=float(m.group(5)), n=n_halo,
        )
    return rows


def main():
    tbl = pickle.load(open(C.CACHE_DIR / "mass_table.pkl", "rb"))
    assert (tbl["log_m200c"] >= 13.0).all(), "spine mass_table has halos below 1e13"
    n_sims = {s: tbl[tbl["suite"] == s]["sim_id"].nunique() for s in SUITES}
    n_halo = {s: (tbl["suite"] == s).sum() for s in SUITES}
    print("spine mass_table:", {SUITE_TEX[s]: f"{n_halo[s]} ({n_sims[s]})" for s in SUITES})

    prior = parse_prior(PRIOR_TEX) if (PRIOR_TEX and PRIOR_TEX.exists()) else {}
    print(f"prior table rows parsed: {len(prior)}")

    # ── compute + verify ───────────────────────────────────────────────────
    stats = {}
    n_bad = 0
    for ap_i, (suffix, ap_tex) in enumerate(APERTURES):
        for ch in COMPONENTS:
            for s in SUITES:
                r, r_sims = residuals(tbl, s, ch, suffix)
                med, err, p16, p84, f20, f50 = row_stats(r, r_sims)
                stats[(ap_i, ch, s)] = (med, err, p16, p84, f20, f50, len(r))
                key = (ap_i, COMP_TEX[ch], SUITE_TEX[s])
                if key in prior:
                    p = prior[key]
                    ok = (
                        abs(p["med"] - med) < 0.015
                        and abs(p["p16"] - p16) < 0.06 and abs(p["p84"] - p84) < 0.06
                        and abs(p["f50"] - f50) < 0.06
                        and abs(p["err"] - err) < max(0.05, 0.15 * p["err"])
                        and p["n"] == len(r)
                    )
                    flag = "OK " if ok else "MISMATCH"
                    n_bad += not ok
                    print(f"  {flag} {ap_tex:12s} {ch:9s} {SUITE_TEX[s]:4s} "
                          f"med {med:+6.2f}+-{err:4.2f} "
                          f"(prior {p['med']:+6.2f}+-{p['err']:4.2f}) "
                          f"16-84 [{p16:+5.1f},{p84:+5.1f}] "
                          f"(prior [{p['p16']:+5.1f},{p['p84']:+5.1f}]) "
                          f"f50 {f50:4.1f} (prior {p['f50']:4.1f}) N {len(r)}")
                else:
                    print(f"  ---  {ap_tex:12s} {ch:9s} {SUITE_TEX[s]:4s} "
                          f"med {med:+6.2f}  16-84 [{p16:+5.1f},{p84:+5.1f}] "
                          f"f50 {f50:4.1f}  N {len(r)}  (no prior row)")
    if prior:
        assert n_bad == 0, f"{n_bad} rows disagree with the prior table"
        print("verification vs prior table: all rows agree")

    # ── write the adapted deluxetable ──────────────────────────────────────
    lines = [
        r"\begin{deluxetable*}{llrrccc}",
        r"\tabletypesize{\footnotesize}",
        r"\tablecaption{Integrated-mass residuals $\delta = M_{\rm BIND}/M_{\rm truth}-1$"
        r" per component, aperture, and suite, for all $M_{200c}\ge 10^{13}\,h^{-1}M_\odot$"
        r" halos (the trained regime). Median errors are a cluster bootstrap over"
        r" simulations;"
        r" $f_{50}$ is the fraction of halos with $|\delta|>0.5$."
        r" \label{tab:mass_residuals}}",
        r"\tablehead{\colhead{Aperture} & \colhead{Component} & \colhead{Suite} &"
        r" \colhead{$N_{\rm halo}$ ($N_{\rm sim}$)} & \colhead{median [\%]} &"
        r" \colhead{16--84 [\%]} & \colhead{$f_{50}$ [\%]}}",
        r"\startdata",
    ]
    for ap_i, (suffix, ap_tex) in enumerate(APERTURES):
        for ci, ch in enumerate(COMPONENTS):
            for si, s in enumerate(SUITES):
                med, err, p16, p84, f20, f50, n = stats[(ap_i, ch, s)]
                c0 = ap_tex if (ci == 0 and si == 0) else ""
                c1 = COMP_TEX[ch] if si == 0 else ""
                lines.append(
                    f"{c0} & {c1} & {SUITE_TEX[s]} & {n} ({n_sims[s]}) & "
                    f"${med:+.2f}\\pm{err:.2f}$ & $[{p16:+.1f},{p84:+.1f}]$ & "
                    f"{f50:.1f} \\\\"
                )
            if not (ap_i == len(APERTURES) - 1 and ci == len(COMPONENTS) - 1):
                lines.append(r"\hline")
    lines += [r"\enddata", r"\end{deluxetable*}", ""]
    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
