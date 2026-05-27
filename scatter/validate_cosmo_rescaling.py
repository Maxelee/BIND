"""scatter/validate_cosmo_rescaling.py — does vdm_bind rescale baryon content with cosmology?

Cosmology enters the model through TWO channels that were always consistent in training:
  (1) the DMO conditioning field (its structure encodes Omega_m, sigma8), and
  (2) the parameter vector.
A fixed-DMO counterfactual (vary the cosmo number in the parameter vector while keeping a
fiducial-cosmology DMO field) is therefore out-of-distribution. This script tests, empirically,
whether the emulator nonetheless reproduces the cosmology response — i.e. whether it has learned a
"cosmology rescaling" of baryonic content from the parameter vector alone.

For each cosmology parameter (Omega_m=p1, sigma8=p2) and each 1P level we compare the scaling-relation
normalization alpha (and scatter sigma) from THREE sources:

  TRUTH       — measured from full_maps['truth_maps']            (ground truth)
  GEN_correct — emulator fed the level's OWN DMO field + params  (in-distribution; precomputed)
  GEN_fixed   — emulator fed the FIDUCIAL DMO field + the varied cosmo param  (the rescaling test)

Key metric per observable: the "rescaling fraction" = slope(GEN_fixed) / slope(TRUTH) of alpha vs
the cosmo parameter.
  ~1  -> the parameter vector fully encodes the cosmology response: the rescaling works (powerful).
  ~0  -> cosmology must come through the DMO field: fixed-DMO cosmo counterfactuals are invalid.
GEN_correct vs TRUTH is the in-distribution ceiling (should track if the model is good at all).

Generation for GEN_fixed needs the model (GPU); TRUTH and GEN_correct are precomputed (CPU).

Usage:
    python -m scatter.validate_cosmo_rescaling                 # Omega_m + sigma8
    python -m scatter.validate_cosmo_rescaling --params Omega_m
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data import NormStats
from train import FlowMatchingLit
from scatter.measure_scatter import measure_scatter
from scatter.scatter_decomposition import load_halos, normalize_params_fid, RUN_DIR
from scatter.validate_1p_truth import ingest_sim, relation_fit, FOCUS_OBS, LOG_OBS

OUT_DIR = pathlib.Path("outputs/scatter_diagnostics")
FIG_DIR = pathlib.Path("figures/scatter_diagnostics")

# cosmology parameter -> (1P family tag, index in 35-vector)
COSMO = {"Omega_m": ("p1", 0), "sigma8": ("p2", 1)}
LEVEL_TAGS = ["n2", "n1", "FID", "1", "2"]
FID_SIM = "1P_p1_0"
NOISE_SEED = 42


def alpha_sigma(O: np.ndarray, logM: np.ndarray, is_log: bool) -> tuple[float, float]:
    a, _, s = relation_fit(O, logM, is_log)
    return a, s


def main() -> None:
    ap = argparse.ArgumentParser(description="Test emulator cosmology rescaling (fixed-DMO).")
    ap.add_argument("--params", nargs="+", default=list(COSMO), choices=list(COSMO))
    ap.add_argument("--n-halos", type=int, default=49)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cosmo] device={device}")

    # ── Model + fixed fiducial-cosmology halo basis (for GEN_fixed) ─────────────
    ns = NormStats.load(RUN_DIR / "norm_stats.npz")
    lit = FlowMatchingLit.load_from_checkpoint(str(RUN_DIR / "checkpoints/last.ckpt"),
                                               map_location=device)
    lit.eval()
    if hasattr(lit, "ema"):
        del lit.ema
    model_fm = lit.fm; model_fm.model.eval()

    fid = load_halos(FID_SIM, ns, args.n_halos, seed=NOISE_SEED)
    if fid is None:
        print("[cosmo] STOP: fiducial halos not found."); return
    fid_logM = np.log10(fid["masses"])
    fid_theta_raw = fid["params"][0].copy()        # raw 35-vector at fiducial
    print(f"[cosmo] fiducial DMO basis: N={fid['N']} halos")

    results = {}
    for cname in args.params:
        tag, cidx = COSMO[cname]
        print(f"\n=== {cname} (1P family {tag}, idx {cidx}) ===")
        rec = {o: {k: [] for k in ("a_truth", "s_truth", "a_genc", "s_genc",
                                   "a_genf", "s_genf")} for o in FOCUS_OBS}
        xvals, levels_used, n_truth = [], [], []
        for lvl in LEVEL_TAGS:
            sim = FID_SIM if lvl == "FID" else f"1P_{tag}_{lvl}"
            ing = ingest_sim(sim)                       # truth + GEN_correct (precomputed)
            if ing is None:
                print(f"  [skip] {sim} missing"); continue
            cval = float(np.nanmedian(ing["params"][:, cidx]))

            # GEN_fixed: fiducial DMO halos, fiducial params except this cosmo param.
            theta_b_raw = fid_theta_raw.copy(); theta_b_raw[cidx] = cval
            theta_b = normalize_params_fid(theta_b_raw, ns)
            r = measure_scatter(
                model_fm=model_fm, norm_stats=ns, theta_norm=theta_b,
                dmo_conds=fid["cond_norm"], ls_conds=fid["ls_norm"],
                masses=fid["masses"], r200_pix=fid["radii_pix"],
                K=args.k, n_steps=args.n_steps, device=device, batch_size=args.batch_size,
                dmo_raw=fid["cond_raw"], omega_m=np.full(fid["N"], theta_b_raw[0]), seed=NOISE_SEED,
            )
            obs_names_ms = [str(x) for x in r["obs_names"]]
            Ybar = r["Y_bar"]                            # (N_h, N_obs) mass-space per-halo mean

            xvals.append(cval); levels_used.append(lvl); n_truth.append(ing["N"])
            for o in FOCUS_OBS:
                is_log = o in LOG_OBS
                a_t, s_t = alpha_sigma(ing["truth"][o], ing["logM"], is_log)
                a_c, s_c = alpha_sigma(ing["gen"][o],   ing["logM"], is_log)
                yb = Ybar[:, obs_names_ms.index(o)]
                a_f, s_f = alpha_sigma(yb, fid_logM, is_log)
                rec[o]["a_truth"].append(a_t); rec[o]["s_truth"].append(s_t)
                rec[o]["a_genc"].append(a_c);  rec[o]["s_genc"].append(s_c)
                rec[o]["a_genf"].append(a_f);  rec[o]["s_genf"].append(s_f)
            print(f"  {sim:12s} {cname}={cval:6.3f}  N_truth={ing['N']:3d}  "
                  f"M_gas alpha truth/genC/genF = "
                  f"{rec['M_gas']['a_truth'][-1]:.3f}/{rec['M_gas']['a_genc'][-1]:.3f}/"
                  f"{rec['M_gas']['a_genf'][-1]:.3f}")

        x = np.array(xvals)
        obs_res = {}
        for o in FOCUS_OBS:
            obs_res[o] = _verdict(x, np.array(rec[o]["a_truth"]),
                                  np.array(rec[o]["a_genc"]), np.array(rec[o]["a_genf"]))
        results[cname] = {"levels": levels_used, "param_values": x.tolist(),
                          "n_truth": n_truth, "obs": obs_res, "rec": rec}
        _print_verdict(cname, obs_res)
        _plot(cname, x, rec, FIG_DIR / f"validate_cosmo_{cname}")

    out = {"focus_obs": FOCUS_OBS,
           "results": {c: {k: v for k, v in r.items() if k != "rec"} for c, r in results.items()}}
    (OUT_DIR / "validate_cosmo_rescaling.json").write_text(json.dumps(out, indent=2))
    print(f"\n[cosmo] wrote {OUT_DIR / 'validate_cosmo_rescaling.json'}")
    print("[cosmo] interpretation: rescaling_fraction ~1 => cosmology-via-parameter-vector is "
          "trustworthy; ~0 => cosmology needs the DMO field (fixed-DMO cosmo scans invalid).")


def _slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan


def _verdict(x, a_truth, a_genc, a_genf) -> dict:
    s_truth, s_genc, s_genf = _slope(x, a_truth), _slope(x, a_genc), _slope(x, a_genf)
    def corr(u, v):
        ok = np.isfinite(u) & np.isfinite(v)
        return float(np.corrcoef(u[ok], v[ok])[0, 1]) if ok.sum() >= 3 and np.std(u[ok]) > 0 and np.std(v[ok]) > 0 else np.nan
    resc = float(s_genf / s_truth) if (np.isfinite(s_truth) and abs(s_truth) > 1e-6) else np.nan
    return {
        "slope_truth": s_truth, "slope_gen_correct": s_genc, "slope_gen_fixed": s_genf,
        "in_dist_corr": corr(a_truth, a_genc),       # GEN_correct vs truth (ceiling)
        "fixed_corr": corr(a_truth, a_genf),         # GEN_fixed  vs truth (the test)
        "rescaling_fraction": resc,                  # how much of the cosmo response the vector carries
    }


def _print_verdict(cname, obs_res) -> None:
    print(f"  --- verdict ({cname}) ---")
    print(f"    {'obs':8s} {'in_dist_r':>9s} {'fixed_r':>8s} {'rescale_frac':>12s}")
    for o, v in obs_res.items():
        print(f"    {o:8s} {v['in_dist_corr']:+9.2f} {v['fixed_corr']:+8.2f} "
              f"{v['rescaling_fraction']:12.2f}")


def _plot(cname, x, rec, stub) -> None:
    fig, axs = plt.subplots(1, len(FOCUS_OBS), figsize=(3.4 * len(FOCUS_OBS), 4.0), squeeze=False)
    for j, o in enumerate(FOCUS_OBS):
        ax = axs[0][j]
        ax.plot(x, rec[o]["a_truth"], "o-", color="k", label="truth")
        ax.plot(x, rec[o]["a_genc"], "s--", color="#1565C0", label="gen (correct DMO)")
        ax.plot(x, rec[o]["a_genf"], "^:", color="#E65100", label="gen (fixed DMO)")
        ax.set_title(o); ax.set_xlabel(cname)
        if j == 0:
            ax.set_ylabel(r"relation normalization $\alpha$"); ax.legend(fontsize=8)
    fig.suptitle(f"Cosmology rescaling test: {cname}\n"
                 "fixed-DMO tracking truth ⇒ the parameter vector carries the cosmology response",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stub.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
        print(f"  saved {stub.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
