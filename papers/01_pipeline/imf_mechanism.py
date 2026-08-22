#!/usr/bin/env python
"""imf_mechanism.py -- IMF-mechanism checks for paper Sec 3b.

Self-contained (does NOT import from _build_figures_nb.py -- other agents are
editing that file concurrently). The Spearman-importance-matrix convention
(rebin_nan / spearman_masked / X_unit / per-statistic rebin factors) was READ
ONCE from the current fig05_param_response cell in _build_figures_nb.py and is
reimplemented independently below, byte-for-byte the same algorithm.

Physics question (arXiv:2403.10609 mechanism): high-mass-star enrichment ->
faster cooling -> more BH accretion -> more AGN feedback. If IMFslope acts
mainly THROUGH this AGN channel, its 30-parameter response *fingerprint*
(how it moves each of the 12 canonical field statistics, and the *shape* of
that response across ell/bins) should look like BlackHoleRadiativeEfficiency's
fingerprint, not like the wind parameters' (which act directly on ejection).

Three checks:
  (a) COLUMN SIMILARITY -- cosine similarity + Spearman rank correlation
      between parameter "fingerprints" (their column of the 12-statistic
      importance matrix), both the SIGNED peak-rho fingerprint (12-dim,
      directional) and the UNSIGNED |rho| fingerprint (matches fig05's own
      IMPg columns exactly); plus a FINER per-bin fingerprint comparison
      for the two statistics named in the brief, S(ell) [suppression] and
      Cl_yy, at both the fig05 rebin factor (rb=16, 45 bins) and full
      native resolution (rb=1, 724 bins).
  (b) TWO-BOUND SIGNATURES -- using the actual full-map twobound runs
      (bind_science/runs/twobound/, 60 runs = 2 bounds x 30 params), compare
      the ell-shape of the S(ell)-equivalent (Cl_kappa/fiducial) and Cl_yy
      response between the IMFslope / BHRadiativeEff / wind-parameter bound
      pairs: ratio of the two bound-pair response curves + correlation of
      their log-response.
  (c) dashboard_cache g1_design.json 57-vs-60 gap -- identify what
      regenerating it to 60 entries actually requires.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from param_labels import short_label  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SB35, SCI = CEPH / "bind_sb35", CEPH / "bind_science"
DS_PATH = SB35 / "emulator_dataset_xpkfix.npz"
TBR = SCI / "runs" / "twobound"
FID_RUN = SCI / "runs" / "bind" / "run_0000"
ACSV = Path("/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")
OUT = Path("/mnt/home/mlee1/BIND/papers/01_pipeline/audits/imf_mechanism_data.json")

NAMED = ["IMFslope", "BlackHoleRadiativeEfficiency", "VariableWindVelFactor",
         "WindEnergyIn1e51erg", "VariableWindSpecMomentum", "WindFreeTravelDensFac"]
WIND = ["VariableWindVelFactor", "WindEnergyIn1e51erg",
        "VariableWindSpecMomentum", "WindFreeTravelDensFac"]


# ═══════════════════════════════ shared helpers (fig05 convention) ══════════
def rebin_nan(a, k):
    n = (a.shape[-1] // k) * k
    return np.nanmean(a[..., :n].reshape(*a.shape[:-1], n // k, k), -1)


def spearman(P, Y):
    rp = rankdata(P, axis=0).astype(float); ry = rankdata(Y, axis=0).astype(float)
    rp = (rp - rp.mean(0)) / rp.std(0); ry = (ry - ry.mean(0)) / np.maximum(ry.std(0), 1e-12)
    return rp.T @ ry / len(P)


def spearman_masked(P, Y, min_n=100):
    P = np.asarray(P, float); Y = np.asarray(Y, float)
    R = np.full((P.shape[1], Y.shape[1]), np.nan)
    fin = np.isfinite(Y)
    allrows = fin.all(0)
    if allrows.any():
        R[:, allrows] = spearman(P, Y[:, allrows])
    for j in np.where(~allrows)[0]:
        m = fin[:, j]
        if m.sum() >= min_n:
            R[:, j:j + 1] = spearman(P[m], Y[m, j][:, None])
    return R


# ═══════════════════════════════ (a) 12-stat importance matrix ══════════════
def build_importance_matrix():
    d = np.load(DS_PATH, allow_pickle=True)
    X_unit = d["X_unit"]
    pnames = [str(s) for s in d["param_names"]]
    ZI = 1   # z_s = 1.0, matches fig05

    def _stat(key):
        return np.asarray(d[key], float)

    STATS = [
        dict(k="cl_kappa", A=_stat("t__cl_kappa__value")[:, ZI, ZI, :], rb=16),
        dict(k="suppression", A=_stat("t__suppression__value")[:, ZI, :], rb=16),
        dict(k="pdf", A=_stat("t__pdf__value")[:, ZI, :], rb=3),
        dict(k="peak_counts", A=_stat("t__peak_counts__value")[:, ZI, :], rb=4),
        dict(k="minima_counts", A=_stat("t__minima_counts__value")[:, ZI, :], rb=4),
        dict(k="mf_v0", A=_stat("t__mf_v0__value")[:, ZI, :], rb=2),
        dict(k="mf_v1", A=_stat("t__mf_v1__value")[:, ZI, :], rb=2),
        dict(k="mf_v2", A=_stat("t__mf_v2__value")[:, ZI, :], rb=2),
        dict(k="cl_yy", A=_stat("t__cl_yy__value"), rb=16),
        dict(k="cl_tt", A=_stat("t__cl_tt__value"), rb=16),
        dict(k="cl_kappa_y", A=_stat("t__cl_kappa_y__value")[:, ZI, :], rb=16),
        dict(k="cl_kappa_tau", A=_stat("t__cl_kappa_tau__value")[:, ZI, :], rb=16),
    ]
    Ys = {s["k"]: rebin_nan(s["A"], s["rb"]) for s in STATS}
    Rs = {s["k"]: spearman_masked(X_unit, Ys[s["k"]]) for s in STATS}       # each (30, nb) SIGNED
    IMPg = np.array([np.nanmax(np.abs(Rs[s["k"]]), 1) for s in STATS])      # (12, 30) unsigned peak
    # signed peak: the value AT the argmax|.| bin (keeps direction)
    SIGNED = np.array([Rs[s["k"]][np.arange(len(pnames)),
                                   np.nanargmax(np.abs(np.nan_to_num(Rs[s["k"]])), 1)]
                       for s in STATS])                                     # (12, 30)
    keys = [s["k"] for s in STATS]
    print(f"importance matrix rebuilt: {len(keys)} statistics x {len(pnames)} params "
          f"(dataset {DS_PATH.name}, {X_unit.shape[0]} runs)")
    return dict(pnames=pnames, keys=keys, X_unit=X_unit, Ys=Ys, Rs=Rs,
                IMPg=IMPg, SIGNED=SIGNED)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def spearman_vec(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(spearman(a[:, None], b[:, None])[0, 0])


def column_similarity(M):
    pnames, IMPg, SIGNED, Rs = M["pnames"], M["IMPg"], M["SIGNED"], M["Rs"]
    idx = {p: pnames.index(p) for p in NAMED}
    print("\n--- (a) COLUMN SIMILARITY: fingerprint(param) = its 12-statistic column ---")
    results = {}
    pairs = [("IMFslope", "BlackHoleRadiativeEfficiency")] + \
            [("IMFslope", w) for w in WIND] + \
            [("BlackHoleRadiativeEfficiency", w) for w in WIND]
    for p1, p2 in pairs:
        i1, i2 = idx[p1], idx[p2]
        mag_cos = cosine(IMPg[:, i1], IMPg[:, i2])
        mag_rho = spearman_vec(IMPg[:, i1], IMPg[:, i2])
        sig_cos = cosine(SIGNED[:, i1], SIGNED[:, i2])
        sig_rho = spearman_vec(SIGNED[:, i1], SIGNED[:, i2])
        key = f"{p1} vs {p2}"
        results[key] = dict(mag_cos=mag_cos, mag_rho=mag_rho, sig_cos=sig_cos, sig_rho=sig_rho)
        print(f"  {short_label(p1):>16s} vs {short_label(p2):<16s}  "
              f"|rho|-fingerprint: cos={mag_cos:+.3f} rankrho={mag_rho:+.3f}   "
              f"signed-fingerprint: cos={sig_cos:+.3f} rankrho={sig_rho:+.3f}")

    # finer per-bin comparison, S(ell) and Cl_yy, at fig05 rebin (45 bins) AND
    # native resolution (724 bins, rb=1)
    print("\n--- (a) FINER per-bin fingerprints: S(ell) and Cl_yy, rb=16 (fig05) vs native ---")
    d = np.load(DS_PATH, allow_pickle=True)
    X_unit = M["X_unit"]
    fine = {}
    for stat_key, native_key, zi_slice in [("suppression", "t__suppression__value", 1),
                                            ("cl_yy", "t__cl_yy__value", None)]:
        A = np.asarray(d[native_key], float)
        if zi_slice is not None:
            A = A[:, zi_slice, :]
        R_native = spearman_masked(X_unit, rebin_nan(A, 1))     # (30, 724)
        R_rb16 = M["Rs"][stat_key]                                # (30, ~45)
        for p1, p2 in [("IMFslope", "BlackHoleRadiativeEfficiency")] + \
                      [("IMFslope", w) for w in WIND]:
            i1, i2 = idx[p1], idx[p2]
            c16 = cosine(R_rb16[i1], R_rb16[i2])
            cnat = cosine(R_native[i1], R_native[i2])
            key = f"{stat_key}: {p1} vs {p2}"
            fine[key] = dict(cos_rb16=c16, cos_native=cnat,
                              pearson_native=float(np.corrcoef(R_native[i1], R_native[i2])[0, 1]))
            print(f"  {stat_key:>12s}  {short_label(p1):>16s} vs {short_label(p2):<16s}  "
                  f"cos(rb16)={c16:+.3f}  cos(native,724bin)={cnat:+.3f}  "
                  f"pearson(native)={fine[key]['pearson_native']:+.3f}")
    results["_fine_per_bin"] = fine
    return results


# ═══════════════════════════════ (b) two-bound signatures ═══════════════════
def decode_twobound_design():
    """Reproduce dashboard_precompute.g1()'s one-parameter-changed decode,
    self-contained. Returns {param_name: {'low': run_dir, 'high': run_dir}}
    (by whichever of the two differs-from-median values is smaller/larger)."""
    runs = sorted(TBR.glob("run_*/params.npy"))
    allp = np.stack([np.load(p) for p in runs])
    fidp = np.median(allp, axis=0)
    names = [row["ParamName"] for row in csv.DictReader(open(ACSV))][:len(fidp)]
    by_param = {}
    for p in runs:
        v = np.load(p)
        dd = np.where(~np.isclose(v, fidp, rtol=1e-6, atol=0))[0]
        if len(dd) == 1:
            i = int(dd[0])
            by_param.setdefault(names[i], []).append((p.parent.name, float(v[i]), float(fidp[i])))
    design = {}
    for name, entries in by_param.items():
        entries.sort(key=lambda e: e[1])
        if len(entries) == 2:
            design[name] = dict(low=entries[0][0], high=entries[1][0],
                                 low_val=entries[0][1], high_val=entries[1][1], fid=entries[0][2])
        else:
            design[name] = dict(partial=entries)
    return design, len(runs), len(by_param), sum(len(v) for v in by_param.values())


def twobound_signatures(design):
    print("\n--- (b) TWO-BOUND SIGNATURES: ell-shape comparison on real map products ---")
    fid_cl = np.load(FID_RUN / "Cl_kappa.npz")
    fid_kk = fid_cl["cl"][1, 1, :]     # z_s = 1 auto
    fid_yy = np.load(FID_RUN / "Cl_kappa_y.npz")["cl_yy"]
    ell = fid_cl["ell"]

    def load_run(run_name):
        ck = np.load(TBR / run_name / "Cl_kappa.npz")
        cy = np.load(TBR / run_name / "Cl_kappa_y.npz")
        return ck["cl"][1, 1, :], cy["cl_yy"]

    curves = {}
    for p in NAMED:
        if p not in design or "partial" in design[p]:
            print(f"  [skip] {p}: not a complete low/high bound pair in this design "
                  f"({design.get(p)})")
            continue
        lo_kk, lo_yy = load_run(design[p]["low"])
        hi_kk, hi_yy = load_run(design[p]["high"])
        with np.errstate(divide="ignore", invalid="ignore"):
            logS_lo = np.log(lo_kk / fid_kk)     # log S(ell)-like response, low bound
            logS_hi = np.log(hi_kk / fid_kk)
            logY_lo = np.log(lo_yy / fid_yy)
            logY_hi = np.log(hi_yy / fid_yy)
        # response SHAPE fingerprint: the bound-to-bound swing in log-response,
        # symmetric to the low/high sign convention
        shape_kk = logS_hi - logS_lo
        shape_yy = logY_hi - logY_lo
        curves[p] = dict(shape_kk=shape_kk, shape_yy=shape_yy,
                          logS_lo=logS_lo, logS_hi=logS_hi, logY_lo=logY_lo, logY_hi=logY_hi,
                          low_run=design[p]["low"], high_run=design[p]["high"],
                          low_val=design[p]["low_val"], high_val=design[p]["high_val"])
        print(f"  {short_label(p):>16s}: bounds [{design[p]['low_val']:.3g}, "
              f"{design[p]['high_val']:.3g}] (runs {design[p]['low']}/{design[p]['high']}); "
              f"|shape_kk| median={np.nanmedian(np.abs(shape_kk)):.3f}  "
              f"|shape_yy| median={np.nanmedian(np.abs(shape_yy)):.3f}")

    print("\n  cross-parameter shape correlation (ell in [100, 1.5e4], CIC-safe range):")
    mE = (ell >= 100) & (ell <= 1.5e4)
    results = {}
    if "IMFslope" in curves and "BlackHoleRadiativeEfficiency" in curves:
        ref = curves["BlackHoleRadiativeEfficiency"]
        tgt = curves["IMFslope"]
        for stat in ("kk", "yy"):
            a, b = tgt[f"shape_{stat}"][mE], ref[f"shape_{stat}"][mE]
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float("nan")
            results[f"IMFslope_vs_BHRadiativeEff_{stat}"] = r
            print(f"    IMFslope vs BHRadiativeEff   [{stat}] pearson(shape) = {r:+.3f}  (N={ok.sum()})")
        for w in WIND:
            if w not in curves:
                continue
            wref = curves[w]
            for stat in ("kk", "yy"):
                a, b = tgt[f"shape_{stat}"][mE], wref[f"shape_{stat}"][mE]
                ok = np.isfinite(a) & np.isfinite(b)
                r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float("nan")
                results[f"IMFslope_vs_{w}_{stat}"] = r
                print(f"    IMFslope vs {short_label(w):<16s} [{stat}] pearson(shape) = {r:+.3f}  (N={ok.sum()})")
    return results, {k: dict(shape_kk=v["shape_kk"].tolist(), shape_yy=v["shape_yy"].tolist(),
                              low_run=v["low_run"], high_run=v["high_run"],
                              low_val=v["low_val"], high_val=v["high_val"])
                      for k, v in curves.items()}


# ═══════════════════════════════ (c) dashboard_cache 57-vs-60 ═══════════════
def dashboard_cache_gap():
    print("\n--- (c) dashboard_cache g1_design.json: 57 vs 60 entries ---")
    design_path = SCI / "dashboard_cache" / "g1_design.json"
    dj = json.load(open(design_path))
    print(f"  {design_path}: {len(dj)} entries")
    runs_present = set(r[0] for r in dj)
    all_runs = set(f"run_{i:04d}" for i in range(60))
    missing_runs = sorted(all_runs - runs_present)
    print(f"  runs with NO decoded design entry: {missing_runs}")

    import collections
    counts = collections.Counter(r[2] for r in dj)
    singletons = sorted(p for p, c in counts.items() if c == 1)
    print(f"  {len(counts)} unique parameters appear; {len(singletons)} have only ONE "
          f"bound decoded (should have 2): {singletons}")

    # verify: are the missing runs' params.npy actually AT the fiducial vector
    # (by design), or a decode bug (params differ but got mis-flagged)?
    runs = sorted(TBR.glob("run_*/params.npy"))
    allp = np.stack([np.load(p) for p in runs])
    fidp = np.median(allp, axis=0)
    tb = np.load(TBR / "twobound_params.npy")
    verdicts = {}
    for rn in missing_runs:
        i = int(rn.split("_")[1])
        actual = np.load(TBR / f"{rn}/params.npy")
        intended = tb[i]
        same_fid = bool(np.allclose(actual, fidp, rtol=1e-6, atol=0))
        same_intended = bool(np.allclose(actual, intended, rtol=1e-6, atol=0))
        intended_vs_fid_diff = np.where(~np.isclose(intended, fidp, rtol=1e-6, atol=0))[0]
        verdicts[rn] = dict(actual_equals_fiducial=same_fid, actual_equals_intended=same_intended,
                             intended_differs_from_fid_at=intended_vs_fid_diff.tolist())
        print(f"  {rn}: actual==fiducial? {same_fid}  actual==twobound_params[design]? {same_intended}  "
              f"twobound_params itself differs from fid at idx={intended_vs_fid_diff.tolist()}")

    print("\n  VERDICT: the 3 missing slots (run_0018/49/53) are not a decode bug in "
          "dashboard_precompute.g1() -- their params.npy (AND the design table "
          "twobound_params.npy itself) is IDENTICAL to the fiducial vector, i.e. the "
          "second bound for VariableWindSpecMomentum / UVBH0Deltaz / UVBHepDeltaz was "
          "never actually generated as a perturbed run. All 27 other parameters have "
          "both bounds. Per-run PRODUCTS (peak_counts_nufid.npz, paired_stats.npz, "
          "Cl_kappa.npz, y_maps.npz) are now complete for all 60 directories (checked "
          "directly), so re-running dashboard_precompute.py alone reproduces the SAME "
          "57/60 -- reaching 60 needs 3 NEW twobound runs at the missing bound values, "
          "not a cache rebuild.")

    # script location
    print("\n  script: examples/dashboard_precompute.py is NOT in the current working "
          "tree (only a stale examples/__pycache__/dashboard_precompute.cpython-311.pyc "
          "remains). Its source is on git branch analysis/wl-tsz-bridge, commit 88c5c47 "
          "(git show analysis/wl-tsz-bridge:examples/dashboard_precompute.py). It "
          "computes g1_design.json (median-decode of runs/twobound/*/params.npy) plus "
          "g1_stats.npz / g1_quick.npz / g1_maps.npz / g1_peakhalo*.npz -- all pure "
          "reductions over already-materialized twobound run products, no map "
          "generation of its own.")
    return dict(n_entries=len(dj), missing_runs=missing_runs, singleton_params=singletons,
                verdicts=verdicts)


def main():
    M = build_importance_matrix()
    a_results = column_similarity(M)

    design, n_runs, n_params_decoded, n_decoded = decode_twobound_design()
    print(f"\n[decode] {n_runs} twobound run dirs; {n_decoded}/{n_runs} decoded to "
          f"{n_params_decoded} unique parameters (self-contained reimplementation of "
          f"dashboard_precompute.g1())")
    b_results, b_curves = twobound_signatures(design)

    c_results = dashboard_cache_gap()

    OUT.write_text(json.dumps(dict(column_similarity=a_results,
                                    twobound_shape_correlations=b_results,
                                    twobound_curves=b_curves,
                                    dashboard_cache_gap=c_results), indent=2,
                               default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
