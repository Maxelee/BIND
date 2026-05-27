"""scatter/validate_1p_truth.py — validate the emulator's feedback response against 1P truth.

The scatter decomposition claims feedback (mainly SN) drives the scatter of group-scale
observables. That claim rests on the emulator's response to the feedback parameters. Here we
check it against CAMELS 1P *ground truth*, which exists on disk for each feedback level:

    full_maps.npz['truth_maps']                      (3, 1024, 1024)  hydro truth (varies with theta)
    .../fm_two_head/generated_halos.npz['generated'] (N, 3, 128, 128) model halos (already generated)
    halo_catalog.npz                                 centers / halo_masses / params / radii

For each feedback parameter (A_SN1=p3, A_SN2=p5, A_AGN1=p4, A_AGN2=p6) and each 1P level
(n2, n1, fiducial, 1, 2) we measure, from BOTH truth and generated maps, the scaling-relation
normalization alpha and residual scatter sigma:

    O_h = alpha + beta * (logM_h - PIVOT) + residual,   sigma = std(residual)

then test:
  (1) TRUTH RESPONDS   — does alpha/sigma actually move across the parameter? (kills the earlier
                          false-alarm that 1P truth was identical across levels)
  (2) MODEL TRACKS      — does the emulator's alpha/sigma trend follow truth (sign + correlation)?

This is pure CPU: no model is loaded — generated halos are precomputed and physical-unit.
Per-halo truth vs generated are paired within a level (same sim), so we also report fiducial
per-halo accuracy.

Usage:
    python -m scatter.validate_1p_truth
    python -m scatter.validate_1p_truth --params A_SN1 A_AGN1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scatter.obs_common import (
    observables_from_phys, axis_ratio_q, r200c_pix,
    OMEGA_B_FIXED, PATCH_PIX, BOX_SIZE, N_PIX_FULL,
)

# ─────────────────────────────────────────────────────────────────────────────
TEST_BASE = pathlib.Path("/mnt/home/mlee1/ceph/fm_testsuite/1P")
SUB       = "snap_090/mass_threshold_1p000e13"
MODEL     = "fm_two_head"
FIG_DIR   = pathlib.Path("figures/scatter_diagnostics")
OUT_DIR   = pathlib.Path("outputs/scatter_diagnostics")

# Feedback parameter -> (1P family tag, index in 35-vector)
FEEDBACK = {"A_SN1": ("p3", 2), "A_SN2": ("p5", 4),
            "A_AGN1": ("p4", 3), "A_AGN2": ("p6", 5)}
LEVEL_TAGS = ["n2", "n1", "FID", "1", "2"]   # FID = shared fiducial sim 1P_p1_0
FID_SIM    = "1P_p1_0"

FOCUS_OBS = ["M_gas", "M_star", "f_b", "q_gas"]
LOG_OBS   = {"M_gas", "M_star", "M_dm", "Sigma_gas_c"}   # measured in log10
PIVOT_LOGM = 13.5                                        # fixed pivot for alpha


def _centers_to_pixels(centers_mpc: np.ndarray) -> np.ndarray:
    ppm = N_PIX_FULL / BOX_SIZE
    return (np.asarray(centers_mpc) * ppm).astype(np.int64) % N_PIX_FULL


def _extract_patch(field_2d: np.ndarray, cx: int, cy: int, size: int = PATCH_PIX) -> np.ndarray:
    n = field_2d.shape[0]; half = size // 2
    ix = (cx - half + np.arange(size)) % n
    iy = (cy - half + np.arange(size)) % n
    return field_2d[np.ix_(ix, iy)]


def ingest_sim(sim: str) -> dict | None:
    """Per-halo truth and generated observables for one sim. None if files absent."""
    base = TEST_BASE / sim
    fm_p  = base / "snap_090" / "full_maps.npz"
    cat_p = base / SUB / "halo_catalog.npz"
    cut_p = base / SUB / "halo_cutouts.npz"
    gen_p = base / SUB / MODEL / "generated_halos.npz"
    if not all(p.exists() for p in (fm_p, cat_p, cut_p, gen_p)):
        return None

    fm   = np.load(fm_p, allow_pickle=True)
    cat  = np.load(cat_p, allow_pickle=True)
    cuts = np.load(cut_p, allow_pickle=True)
    gen  = np.load(gen_p, allow_pickle=True)["generated"]   # (N, 3, 128, 128), physical

    centers_pix = _centers_to_pixels(cat["centers"])
    masses = cat["halo_masses"].astype(np.float64)
    params = cat["params"].astype(np.float64)
    truth_maps = fm["truth_maps"]                            # (3, 1024, 1024)
    cond = cuts["condition"]                                 # (N, 128, 128) DMO
    n = len(masses)

    keys = FOCUS_OBS
    truth = {k: np.full(n, np.nan) for k in keys}
    genr  = {k: np.full(n, np.nan) for k in keys}
    for i in range(n):
        mass = float(masses[i])
        omega_m = float(params[i, 0]) if params[i, 0] > 0 else 0.3
        r200p = r200c_pix(mass)
        f_b_cos = OMEGA_B_FIXED / max(omega_m, 1e-10)
        r_aper = max(min(r200p, PATCH_PIX / 2 - 2), 4.0)
        q_dmo = axis_ratio_q(np.maximum(cond[i].astype(np.float64), 0.0), r_aper)

        tp = np.stack([_extract_patch(truth_maps[c], centers_pix[i, 0], centers_pix[i, 1])
                       for c in range(3)]).astype(np.float64)
        t = observables_from_phys(tp, r200p, f_b_cos, q_dmo)
        g = observables_from_phys(gen[i].astype(np.float64), r200p, f_b_cos, q_dmo)
        for k in keys:
            truth[k][i] = t[k]; genr[k][i] = g[k]

    pos = cat["halo_positions"].astype(np.float64) if "halo_positions" in cat.files else None
    return {"masses": masses, "logM": np.log10(masses), "params": params,
            "truth": truth, "gen": genr, "N": n, "pos": pos}


def relation_fit(O: np.ndarray, logM: np.ndarray, is_log: bool) -> tuple[float, float, float]:
    """Fit O (or log10 O) = alpha + beta*(logM - PIVOT); return (alpha, beta, sigma_resid)."""
    y = np.log10(np.clip(O, 1e-30, None)) if is_log else O.astype(float)
    ok = np.isfinite(y) & np.isfinite(logM)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    x = logM[ok] - PIVOT_LOGM
    beta, alpha = np.polyfit(x, y[ok], 1)
    resid = y[ok] - (alpha + beta * x)
    return float(alpha), float(beta), float(np.std(resid, ddof=1))


def fiducial_accuracy() -> dict:
    """Paired per-halo accuracy (gen - truth) at the shared fiducial sim."""
    fid = ingest_sim(FID_SIM)
    acc = {}
    if fid is None:
        return acc
    for o in FOCUS_OBS:
        is_log = o in LOG_OBS
        t, g = fid["truth"][o], fid["gen"][o]
        if is_log:
            t = np.log10(np.clip(t, 1e-30, None)); g = np.log10(np.clip(g, 1e-30, None))
        d = g - t; ok = np.isfinite(d)
        acc[o] = {"bias": float(np.nanmean(d[ok])),
                  "rms": float(np.sqrt(np.nanmean(d[ok] ** 2))),
                  "units": "dex" if is_log else "linear"}
    return acc


def run(params: list[str] | None = None, verbose: bool = False) -> dict:
    """Full validation, returning per-parameter trends + verdicts (no file I/O).

    Returns {"fiducial_accuracy": {...},
             "params": {pname: {"x": [...], "rec": {obs: {alpha_t,sig_t,alpha_g,sig_g}},
                                "verdict": {obs: {...}}, "levels": [...]}}}.
    Importable by the notebook to run the validation inline (pure CPU).
    """
    params = params or list(FEEDBACK)
    acc = fiducial_accuracy()
    if verbose and acc:
        print("=== Fiducial per-halo accuracy (gen - truth) ===")
        for o, a in acc.items():
            print(f"  {o:8s}: bias={a['bias']:+.4f}  rms={a['rms']:.4f}  ({a['units']})")

    out = {}
    for pname in params:
        tag, pidx = FEEDBACK[pname]
        if verbose:
            print(f"\n=== {pname} (1P family {tag}, param idx {pidx}) ===")
        xvals, levels_used = [], []
        rec = {o: {k: [] for k in ("alpha_t", "sig_t", "alpha_g", "sig_g")} for o in FOCUS_OBS}
        for lvl in LEVEL_TAGS:
            sim = FID_SIM if lvl == "FID" else f"1P_{tag}_{lvl}"
            d = ingest_sim(sim)
            if d is None:
                if verbose:
                    print(f"  [skip] {sim} missing")
                continue
            xvals.append(float(np.nanmedian(d["params"][:, pidx]))); levels_used.append(lvl)
            for o in FOCUS_OBS:
                is_log = o in LOG_OBS
                a_t, _, s_t = relation_fit(d["truth"][o], d["logM"], is_log)
                a_g, _, s_g = relation_fit(d["gen"][o],   d["logM"], is_log)
                rec[o]["alpha_t"].append(a_t); rec[o]["sig_t"].append(s_t)
                rec[o]["alpha_g"].append(a_g); rec[o]["sig_g"].append(s_g)
            if verbose:
                print(f"  {sim:12s} {pname}={xvals[-1]:7.3f}  N={d['N']:3d}")
        x = np.array(xvals)
        verdict = {o: _verdict(x, np.array(rec[o]["alpha_t"]), np.array(rec[o]["alpha_g"]),
                               np.array(rec[o]["sig_t"]), np.array(rec[o]["sig_g"]))
                   for o in FOCUS_OBS}
        out[pname] = {"x": xvals, "levels": levels_used, "rec": rec, "verdict": verdict}
        if verbose:
            _print_verdict(pname, {"obs": verdict})
    return {"fiducial_accuracy": acc, "params": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate emulator feedback response vs 1P truth.")
    ap.add_argument("--params", nargs="+", default=list(FEEDBACK), choices=list(FEEDBACK))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    val = run(args.params, verbose=True)
    for pname, pr in val["params"].items():
        _plot_param(pname, np.array(pr["x"]), pr["rec"], FIG_DIR / f"validate_1p_{pname}")

    out = {"fiducial_accuracy": val["fiducial_accuracy"], "pivot_logM": PIVOT_LOGM,
           "focus_obs": FOCUS_OBS,
           "results": {p: {"levels": pr["levels"], "param_values": pr["x"],
                           "obs": pr["verdict"]} for p, pr in val["params"].items()}}
    (OUT_DIR / "validate_1p_truth.json").write_text(json.dumps(out, indent=2))
    print(f"\n[validate] wrote {OUT_DIR / 'validate_1p_truth.json'}")


def _trend_corr(x, yt, yg) -> dict:
    """Trend agreement between truth and gen across parameter levels."""
    ok = np.isfinite(yt) & np.isfinite(yg) & np.isfinite(x)
    if ok.sum() < 3:
        return {"truth_range": np.nan, "sign_agree": None, "pearson": np.nan}
    xt, a, b = x[ok], yt[ok], yg[ok]
    truth_range = float(np.nanmax(a) - np.nanmin(a))
    slope_t = np.polyfit(xt, a, 1)[0]
    slope_g = np.polyfit(xt, b, 1)[0]
    pear = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    return {"truth_range": truth_range,
            "sign_agree": bool(np.sign(slope_t) == np.sign(slope_g)),
            "pearson": pear,
            "slope_truth": float(slope_t), "slope_gen": float(slope_g)}


def _verdict(x, at, ag, st, sg) -> dict:
    alpha = _trend_corr(x, at, ag)
    sigma = _trend_corr(x, st, sg)
    # PASS: truth responds (alpha OR sigma moves) AND model tracks that trend.
    truth_responds = (alpha["truth_range"] or 0) > 0.02 or (sigma["truth_range"] or 0) > 0.01
    tracks = bool(alpha["sign_agree"]) and (np.isnan(alpha["pearson"]) or alpha["pearson"] > 0.5)
    return {"alpha": alpha, "sigma": sigma,
            "truth_responds": bool(truth_responds),
            "model_tracks_alpha": tracks,
            "pass": bool(truth_responds and tracks)}


def _print_verdict(pname, pr) -> None:
    print(f"  --- verdict ({pname}) ---")
    for o, v in pr["obs"].items():
        a = v["alpha"]
        print(f"    {o:8s}: truth_responds={v['truth_responds']!s:5s}  "
              f"alpha sign_agree={a['sign_agree']!s:5s} r={a['pearson']:+.2f}  "
              f"-> {'PASS' if v['pass'] else 'check'}")


def _plot_param(pname, x, rec, stub) -> None:
    fig, axs = plt.subplots(2, len(FOCUS_OBS), figsize=(3.4 * len(FOCUS_OBS), 6.2), squeeze=False)
    for j, o in enumerate(FOCUS_OBS):
        ax = axs[0][j]
        ax.plot(x, rec[o]["alpha_t"], "o-", color="k", label="truth")
        ax.plot(x, rec[o]["alpha_g"], "s--", color="#E65100", label="emulator")
        ax.set_title(o); ax.set_ylabel(r"$\alpha$ (norm.)" if j == 0 else "")
        if j == 0: ax.legend(fontsize=8)
        ax2 = axs[1][j]
        ax2.plot(x, rec[o]["sig_t"], "o-", color="k")
        ax2.plot(x, rec[o]["sig_g"], "s--", color="#6A1B9A")
        ax2.set_xlabel(pname); ax2.set_ylabel(r"$\sigma$ (scatter)" if j == 0 else "")
    fig.suptitle(f"1P truth vs emulator across {pname}\n(top: relation normalization; "
                 f"bottom: residual scatter)", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stub.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
        print(f"  saved {stub.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
