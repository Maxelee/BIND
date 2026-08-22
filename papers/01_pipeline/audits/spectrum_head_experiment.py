"""Controlled target-engineering experiment for the Paper I spectrum heads.

GOAL. The released emulator's WL-field heads (suppression, PDF, MFs, ...) hold out
at 2-3% median |frac err|; the six ell-domain spectrum heads (cl_yy, cl_tt,
cl_kappa_y, cl_kappa_tau, cl_yt, cl_kappa) hold out at 10-21%. This script asks,
with a single fixed hand-rolled PCA+GP pipeline (sklearn, CPU, NOT bind.emulator's
production gpytorch backend), how much of that gap closes under three candidate
target-engineering changes, tested independently per head:

  (a) BASELINE       current convention: log10 for positive spectra, raw for
                      signed crosses; full 724-bin ell grid (incl. the untrusted,
                      CIC-aliased ell > 1.5e4 tail -- see docs memory
                      "Lightcone kappa upturn = CIC aliasing").
  (b) MASKED         same transform, ell <= 1.5e4 only (the trusted range).
  (c) MASKED+SCALED  masked domain; positive spectra unchanged (log10 already
                      tames the dynamic range); signed crosses get
                      asinh(x / s_bin) with s_bin = 1.4826*MAD_bin(TRAIN), which
                      is what actually changes under (c) for the cross heads.
  (d) COMPOSED (cl_kappa only)  predict suppression (variant a) and multiply by
                      the run-independent DMO trace C_dmo = mean_train(cl_kappa /
                      suppression) instead of training cl_kappa directly.

Every variant shares: X standardized (train mean/std) over the 30 unit-cube
params; per-bin center/scale (train stats) on the transformed target; PCA via
SVD, 12 components; one independent sklearn GaussianProcessRegressor per PCA
component (isotropic RBF + WhiteKernel, alpha=1e-8, normalize_y=True,
n_restarts_optimizer=0); invert PCA -> per-bin scale -> transform to get the
physical-space prediction. Metrics are computed in ORIGINAL (physical) units on
the 50 held-out runs of the seed-0 split.

KERNEL CHOICE (read before trusting absolute numbers). The brief's literal spec
is a 30-dim ARD kernel (length_scale=np.ones(30)). Two things ruled that out:
  1. Timing: on this node's SINGLE shared CPU core, one ARD component fit is
     ~4-6s (vs ~0.3-0.5s isotropic) -- ~19 head/variant pipelines x 12
     components would be ~15-20 minutes just for the GP fits.
  2. Worse: with n_restarts_optimizer=0 (as specified) AND the literal
     length_scale=1 initial value, we verified directly (log-marginal-
     likelihood gradient ~0 at the initial theta, confirmed with a manual
     scipy L-BFGS-B run reproducing sklearn's exact optimizer path) that the
     single unrestarted optimization run never leaves length_scale=1 in 30
     standardized dimensions -- pairwise squared distances there are already
     ~2*30=60, so the RBF kernel is saturated ("no correlation") at that
     init and the gradient is flat. The fit collapses to a constant+white-noise
     split that does not use the parameters at all. This is a known pitfall of
     unrestarted ARD-GP optimization in higher dimensions, not a bug.
  Fix used HERE: isotropic RBF (one length scale, not 30), initialized at
  length_scale = sqrt(30) ~ 5.48 (the natural covariate scale for 30
  independent unit-variance dimensions) instead of the literal default 1.0.
  Kernel FORM (Constant*RBF + White), alpha, normalize_y and
  n_restarts_optimizer=0 are exactly as specified; only the kernel's
  dimensionality and initial length scale changed, uniformly across every
  head/variant, so the (a)/(b)/(c)/(d) COMPARISON is still apples-to-apples.
  Absolute numbers should NOT be expected to match the production gpytorch
  backend (see the suppression sanity anchor below) -- they are calibrated
  against each other, not against the release numbers.

RESPONSE R^2 -- reported TWO ways, because they diverge substantially here:
  * "pooled"  : 1 - var(pred-truth)/var(truth-train_mean), a SINGLE variance
    computed over the concatenated (test_run x bin) array (the brief's literal
    formula). This is dominated by whichever bins have the largest absolute
    variance (typically the lowest ell), so a handful of badly-modeled
    high-variance bins can sink it even when most bins fit well.
  * "per-bin" : per-bin r2_j = 1 - var(pred_j-truth_j)/var(truth_j-train_mean_j)
    (over the run axis), aggregated by the MEDIAN over bins, excluding bins
    with zero training variance. This is bind.emulator's / the paper repo's
    own "response R^2" convention (papers/01_pipeline/_build_figures_nb.py,
    fig 9 cell) and is what the 1.00/0.84/... reference numbers in the brief
    actually are, so it is the one comparable to the production anchor.
Both are reported; "per-bin" is the primary number for cross-variant judgment
(closer to what the release cares about), "pooled" is kept because it is the
literal brief formula and because the size of the pooled/per-bin gap is itself
diagnostic of "a few catastrophic bins vs a uniformly mediocre fit".

Run: python spectrum_head_experiment.py   (takes ~2-3 minutes on 1 CPU core)
"""
from __future__ import annotations

import time
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*length_scale is close to.*")

DS_PATH = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_xpkfix.npz")
OUT_DIR = Path(__file__).resolve().parent
ELL_TRUST = 1.5e4          # trusted range; ell > this is the CIC-aliased tail
N_COMPONENTS = 12
SPLIT_SEED = 0
N_TEST = 50
ZI = 1                     # source-redshift plane index (z_s[1] = 1.0)
TINY = 1e-30
INIT_LENGTH_SCALE = np.sqrt(30.0)   # see docstring: avoids the stuck-at-1 optimum
GP_SEED = 0

POS_HEADS = ["cl_yy", "cl_tt", "cl_kappa"]
CROSS_HEADS = ["cl_kappa_y", "cl_kappa_tau", "cl_yt"]
ALL_HEADS = POS_HEADS + CROSS_HEADS

PRODUCTION_REF = {   # from the task brief -- "current held-out performance"
    "cl_yy": (11.7, 1.00), "cl_tt": (9.6, 0.84), "cl_kappa_y": (11.5, 1.00),
    "cl_kappa_tau": (13.5, 0.78), "cl_yt": (13.9, 1.00), "cl_kappa": (21.2, -0.02),
    "suppression": (2.7, 0.72),
}


# ── data plumbing ────────────────────────────────────────────────────────────

def split(n_runs: int, n_test: int = N_TEST, seed: int = SPLIT_SEED):
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(n_runs, n_test, replace=False))
    train = np.setdiff1d(np.arange(n_runs), test)
    return train, test


def load_heads(d) -> dict:
    """head -> dict(raw=(256,724) physical values, ell=(724,), kind='pos'|'cross')."""
    out = {}
    out["cl_yy"] = dict(raw=np.asarray(d["t__cl_yy__value"], float),
                         ell=np.asarray(d["a__cl_yy__ell"], float), kind="pos")
    out["cl_tt"] = dict(raw=np.asarray(d["t__cl_tt__value"], float),
                         ell=np.asarray(d["a__cl_tt__ell"], float), kind="pos")
    out["cl_kappa"] = dict(raw=np.asarray(d["t__cl_kappa__value"], float)[:, ZI, ZI, :],
                            ell=np.asarray(d["a__cl_kappa__ell"], float), kind="pos")
    out["cl_kappa_y"] = dict(raw=np.asarray(d["t__cl_kappa_y__value"], float)[:, ZI, :],
                              ell=np.asarray(d["a__cl_kappa_y__ell"], float), kind="cross")
    out["cl_kappa_tau"] = dict(raw=np.asarray(d["t__cl_kappa_tau__value"], float)[:, ZI, :],
                                ell=np.asarray(d["a__cl_kappa_tau__ell"], float), kind="cross")
    out["cl_yt"] = dict(raw=np.asarray(d["t__cl_yt__value"], float),
                         ell=np.asarray(d["a__cl_yt__ell"], float), kind="cross")
    out["suppression"] = dict(raw=np.asarray(d["t__suppression__value"], float)[:, ZI, :],
                               ell=np.asarray(d["a__suppression__ell"], float),
                               kind="raw_anchor")
    return out


# ── GP + PCA pipeline ────────────────────────────────────────────────────────

def build_gp(seed: int = GP_SEED) -> GaussianProcessRegressor:
    kernel = ConstantKernel() * RBF(length_scale=INIT_LENGTH_SCALE) + WhiteKernel()
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True,
                                     n_restarts_optimizer=0, random_state=seed)


def fit_variant(raw: np.ndarray, ell: np.ndarray, kind: str, variant: str,
                 Xtr: np.ndarray, Xte: np.ndarray, train_idx: np.ndarray,
                 test_idx: np.ndarray, n_components: int = N_COMPONENTS) -> dict:
    """One head/variant: mask -> transform -> per-bin standardize -> PCA -> GP x k
    -> invert. Returns physical-unit prediction + train/test truth on the same mask."""
    mask = np.ones_like(ell, dtype=bool) if variant == "a" else (ell <= ELL_TRUST)
    Ytr_raw = raw[train_idx][:, mask]
    Yte_raw = raw[test_idx][:, mask]

    s_bin = None
    if kind == "cross" and variant == "c":
        med = np.median(Ytr_raw, axis=0)
        mad = np.median(np.abs(Ytr_raw - med), axis=0) * 1.4826
        s_bin = np.maximum(mad, TINY)

    def fwd(Y):
        if kind == "pos":
            return np.log10(np.clip(Y, TINY, None))
        if kind == "cross" and variant == "c":
            return np.arcsinh(Y / s_bin[None, :])
        return Y                                          # raw: cross a/b, suppression anchor

    def inv(X):
        if kind == "pos":
            return 10.0 ** X
        if kind == "cross" and variant == "c":
            return np.sinh(X) * s_bin[None, :]
        return X

    Xf_tr = fwd(Ytr_raw)
    mean_ = Xf_tr.mean(0)
    std_ = Xf_tr.std(0)
    std_ = np.where(std_ > TINY, std_, 1.0)
    Xs_tr = (Xf_tr - mean_) / std_

    k = int(min(n_components, Xs_tr.shape[0] - 1, Xs_tr.shape[1]))
    _, sv, Vt = np.linalg.svd(Xs_tr, full_matrices=False)
    comp = Vt[:k]
    Z_tr = Xs_tr @ comp.T

    Z_pred = np.zeros((len(test_idx), k))
    for j in range(k):
        gp = build_gp()
        gp.fit(Xtr, Z_tr[:, j])
        Z_pred[:, j] = gp.predict(Xte)

    Xs_pred = Z_pred @ comp
    Xf_pred = Xs_pred * std_ + mean_
    Y_pred = inv(Xf_pred)

    return dict(pred=Y_pred, truth_test=Yte_raw, truth_train=Ytr_raw, mask=mask, k=k,
                s_bin=s_bin)


# ── metrics (all in ORIGINAL/physical units) ────────────────────────────────

def median_frac_err(pred: np.ndarray, truth: np.ndarray) -> float:
    """Median |pred/truth - 1| over bins with |truth| above the 5th percentile of
    |truth| (pooled over test runs x bins) -- avoids the 0/0 blowup near the
    zero-crossings that signed cross-spectra have."""
    absT = np.abs(truth)
    thresh = np.percentile(absT, 5)
    m = absT > thresh
    return float(np.median(np.abs(pred[m] / truth[m] - 1.0)))


def response_r2_pooled(pred: np.ndarray, truth: np.ndarray, train_mean: np.ndarray) -> float:
    resid = (pred - truth).ravel()
    base = (truth - train_mean[None, :]).ravel()
    bv = np.var(base)
    return float(1.0 - np.var(resid) / bv) if bv > 0 else float("nan")


def response_r2_perbin_median(pred: np.ndarray, truth: np.ndarray,
                               train_mean: np.ndarray) -> float:
    num = np.sum((pred - truth) ** 2, axis=0)
    den = np.sum((truth - train_mean[None, :]) ** 2, axis=0)
    r2 = np.where(den > 0, 1.0 - num / np.where(den > 0, den, 1.0), np.nan)
    return float(np.nanmedian(r2))


def cross_yardstick(pred: np.ndarray, truth: np.ndarray, train_truth: np.ndarray) -> float:
    """median_bins[ median_runs(|pred-truth|) / per-bin (P84-P16)/2 of TRAIN truth ].
    The error-to-response view for signed crosses: numerator is an absolute (not
    fractional) residual scale per bin, denominator is that bin's physical Sobol
    spread (train), so both are bin-local and the ratio is well-defined even
    where the cross-spectrum straddles zero."""
    p16, p84 = np.percentile(train_truth, [16, 84], axis=0)
    halfw = np.maximum((p84 - p16) / 2.0, TINY)
    err_bin = np.median(np.abs(pred - truth), axis=0)
    return float(np.median(err_bin / halfw))


# ── main experiment ──────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    print(f"loading {DS_PATH}")
    d = np.load(DS_PATH)
    n_runs = d["X_unit"].shape[0]
    train_idx, test_idx = split(n_runs)
    print(f"split: {len(train_idx)} train / {len(test_idx)} test, seed={SPLIT_SEED}, "
          f"n_runs={n_runs}")

    heads = load_heads(d)

    sc = StandardScaler().fit(d["X_unit"][train_idx])
    Xtr = sc.transform(d["X_unit"][train_idx])
    Xte = sc.transform(d["X_unit"][test_idx])
    print(f"kernel: isotropic RBF, init length_scale={INIT_LENGTH_SCALE:.3f}, "
          f"alpha=1e-8, normalize_y=True, n_restarts_optimizer=0 "
          f"(see docstring for why isotropic + this init, not literal ARD)")

    rows = []
    fits = {}   # (head, variant) -> fit_variant() output, kept for cl_kappa composed reuse
    for head in ALL_HEADS:
        info = heads[head]
        for variant in ("a", "b", "c"):
            t0 = time.time()
            out = fit_variant(info["raw"], info["ell"], info["kind"], variant,
                               Xtr, Xte, train_idx, test_idx)
            fits[(head, variant)] = out
            fe = median_frac_err(out["pred"], out["truth_test"])
            r2p = response_r2_pooled(out["pred"], out["truth_test"], out["truth_train"].mean(0))
            r2b = response_r2_perbin_median(out["pred"], out["truth_test"],
                                             out["truth_train"].mean(0))
            row = dict(head=head, variant=variant, n_bins=int(out["mask"].sum()),
                       k=out["k"], frac_err_pct=100 * fe, r2_pooled=r2p, r2_perbin=r2b)
            if info["kind"] == "cross":
                row["yardstick"] = cross_yardstick(out["pred"], out["truth_test"],
                                                    out["truth_train"])
            rows.append(row)
            extra = f", yardstick={row['yardstick']:.3f}" if "yardstick" in row else ""
            print(f"  [{time.time()-t0:5.1f}s] {head:14s} ({variant})  n_bins={row['n_bins']:3d}  "
                  f"frac_err={row['frac_err_pct']:6.2f}%  R2_pooled={r2p:7.3f}  "
                  f"R2_perbin={r2b:6.3f}{extra}")

    # ── suppression sanity anchor (variant a only) ──────────────────────────
    print("\n[suppression sanity anchor -- variant (a), raw, idx z_s=1, full grid]")
    sinfo = heads["suppression"]
    t0 = time.time()
    out_s = fit_variant(sinfo["raw"], sinfo["ell"], sinfo["kind"], "a",
                         Xtr, Xte, train_idx, test_idx)
    fe_s = median_frac_err(out_s["pred"], out_s["truth_test"])
    r2p_s = response_r2_pooled(out_s["pred"], out_s["truth_test"], out_s["truth_train"].mean(0))
    r2b_s = response_r2_perbin_median(out_s["pred"], out_s["truth_test"],
                                       out_s["truth_train"].mean(0))
    ref_fe, ref_r2 = PRODUCTION_REF["suppression"]
    print(f"  [{time.time()-t0:5.1f}s] hand-rolled: frac_err={100*fe_s:.2f}%  "
          f"R2_pooled={r2p_s:.3f}  R2_perbin={r2b_s:.3f}")
    print(f"  production reference: frac_err={ref_fe:.2f}%  R2={ref_r2:.2f}")
    print(f"  ratio (hand-rolled/production): frac_err x{100*fe_s/ref_fe:.2f}, "
          f"R2_perbin {r2b_s - ref_r2:+.2f} (additive gap)")

    # ── cl_kappa proportionality check + composed variant (d) ──────────────
    print("\n[cl_kappa <-> suppression proportionality check]")
    kappa_raw = heads["cl_kappa"]["raw"]           # (256,724), full grid, z_s idx=1
    kappa_ell = heads["cl_kappa"]["ell"]
    supp_raw = sinfo["raw"]
    supp_ell = sinfo["ell"]
    assert np.allclose(kappa_ell, supp_ell), "cl_kappa and suppression ell grids differ!"
    ratio_train = kappa_raw[train_idx] / supp_raw[train_idx]     # (n_train, 724)
    ratio_mean = ratio_train.mean(0)
    rel_dev = np.abs(ratio_train - ratio_mean[None, :]) / np.maximum(
        np.abs(ratio_mean[None, :]), TINY)
    allclose_1pct = bool(np.allclose(ratio_train, ratio_mean[None, :], rtol=1e-2, atol=0))
    print(f"  ratio = cl_kappa_diag(train) / suppression(train) (both z_s idx={ZI})")
    print(f"  max relative deviation across the {len(train_idx)} train runs, per bin: "
          f"median={np.median(rel_dev.max(0)):.2e}, worst-bin={rel_dev.max():.2e}")
    print(f"  np.allclose(ratio, ratio.mean(axis=0), rtol=1e-2): {allclose_1pct}")
    C_dmo = ratio_mean

    composed_pred = out_s["pred"] * C_dmo[None, :]
    kappa_truth_test = kappa_raw[test_idx]
    kappa_truth_train = kappa_raw[train_idx]
    fe_d = median_frac_err(composed_pred, kappa_truth_test)
    r2p_d = response_r2_pooled(composed_pred, kappa_truth_test, kappa_truth_train.mean(0))
    r2b_d = response_r2_perbin_median(composed_pred, kappa_truth_test,
                                       kappa_truth_train.mean(0))
    rows.append(dict(head="cl_kappa", variant="d_composed", n_bins=int(kappa_raw.shape[1]),
                      k=out_s["k"], frac_err_pct=100 * fe_d, r2_pooled=r2p_d, r2_perbin=r2b_d))
    print(f"\n[cl_kappa variant (d) COMPOSED = suppression(a) x C_dmo]")
    print(f"  frac_err={100*fe_d:.2f}%  R2_pooled={r2p_d:.3f}  R2_perbin={r2b_d:.3f}")
    print(f"  vs suppression(a) anchor: frac_err={100*fe_s:.2f}%  R2_pooled={r2p_s:.3f}  "
          f"R2_perbin={r2b_s:.3f}")
    print("  (frac_err and R2_perbin should match suppression almost exactly: multiplying by "
          "the run-independent, positive C_dmo(ell) is a per-bin rescaling, which leaves "
          "fractional error and per-bin R2 invariant; R2_pooled differs because it reweights "
          "bins by their now-different absolute variance.)")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'head':14s} {'variant':12s} {'n_bins':7s} {'frac_err%':10s} "
          f"{'R2_pooled':10s} {'R2_perbin':10s} {'yardstick':10s}")
    for r in rows:
        y = f"{r['yardstick']:.3f}" if "yardstick" in r else "-"
        print(f"{r['head']:14s} {r['variant']:12s} {r['n_bins']:7d} "
              f"{r['frac_err_pct']:10.2f} {r['r2_pooled']:10.3f} {r['r2_perbin']:10.3f} {y:10s}")
    print(f"{'suppression':14s} {'a (anchor)':12s} {int(sinfo['raw'].shape[1]):7d} "
          f"{100*fe_s:10.2f} {r2p_s:10.3f} {r2b_s:10.3f} {'-':10s}")

    out_json = OUT_DIR / "spectrum_head_results.json"
    payload = {
        "rows": rows,
        "suppression_anchor": dict(frac_err_pct=100 * fe_s, r2_pooled=r2p_s, r2_perbin=r2b_s,
                                    production_ref_frac_err_pct=ref_fe,
                                    production_ref_r2=ref_r2),
        "cl_kappa_proportionality": dict(
            worst_bin_rel_dev=float(rel_dev.max()),
            median_of_per_bin_max_rel_dev=float(np.median(rel_dev.max(0))),
            allclose_rtol_1e2=allclose_1pct),
        "config": dict(ell_trust=ELL_TRUST, n_components=N_COMPONENTS, split_seed=SPLIT_SEED,
                        n_test=N_TEST, init_length_scale=INIT_LENGTH_SCALE,
                        kernel="isotropic RBF + WhiteKernel (see docstring)"),
    }
    out_json.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out_json}")
    print(f"total runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
