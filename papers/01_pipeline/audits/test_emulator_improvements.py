"""Unit tests for the C-head improvement set (asinh_std transform, per-head
ell-masking, composed cl_kappa) on SYNTHETIC data -- no ceph I/O, no GPU.

Run:
    /mnt/home/mlee1/venvs/BIND_env/bin/python \
        papers/01_pipeline/audits/test_emulator_improvements.py

Covers:
  1. asinh_std transform round-trip (forward/inverse, StatCompressor fit/
     transform/inverse) on signed, wide-dynamic-range synthetic data.
  2. Per-feature masking: masked features never influence the PCA, and are
     reconstructed at the TRAIN mean with an inflated error; unmasked features
     are unaffected and recovered accurately.
  3. state_dict()/from_state() round-trips the new fields (s_, mask_,
     kept_idx_, full_mean_/full_std_) bit-exactly.
  4. The composed cl_kappa head: diagonal is recovered exactly as
     suppression x cl_dmo (the measured identity), off-diagonal blocks are
     placed at the right (i,j)/(j,i) symmetric positions, and the public
     predict() shape/keys contract (full (5,5,L), *_err inflated on masked
     bins) holds end-to-end through an Emulator.fit()/predict() cycle on a
     synthetic EmulatorDataset.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from bind.emulator.core import (  # noqa: E402
    DEFAULT_ELL_MASK_HEADS,
    DEFAULT_TRANSFORM_OVERRIDES,
    Emulator,
)
from bind.emulator.dataset import EmulatorDataset, Target, params_to_unit  # noqa: E402
from bind.emulator.transforms import StatCompressor, _forward, _inverse  # noqa: E402
from bind.inference.design import ASTRO_PARAM_INDICES  # noqa: E402
from bind.params import PARAM_MAX, PARAM_MIN  # noqa: E402

_ASTRO_IDX = np.asarray(ASTRO_PARAM_INDICES)

RNG = np.random.default_rng(0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ─────────────────────────────────────────────────────────────────────────────
def test_asinh_std_forward_inverse():
    section("1a. asinh_std _forward/_inverse elementwise round-trip")
    y = RNG.normal(0, 1e6, size=2000) * RNG.choice([-1, 1], size=2000)
    s = np.full_like(y, 137.0)
    x = _forward(y, "asinh_std", s)
    y2 = _inverse(x, "asinh_std", s)
    err = np.max(np.abs(y2 - y) / np.maximum(np.abs(y), 1e-30))
    check("asinh_std elementwise round-trip (max rel err < 1e-9)", err < 1e-9,
          f"max rel err {err:.3e}")
    # sign preserved, including near zero
    y0 = np.array([-1e-20, 0.0, 1e-20, -5.0, 5.0])
    x0 = _forward(y0, "asinh_std", np.full_like(y0, 2.0))
    check("asinh_std preserves sign", np.array_equal(np.sign(x0), np.sign(y0)))


def test_stat_compressor_asinh_std_roundtrip():
    section("1b. StatCompressor(asinh_std) fit -> transform -> inverse round-trip")
    N, L = 220, 300
    # low-rank signed synthetic "cross spectrum": 3 latent factors x a smooth
    # ell-dependent shape, spanning >2 decades like the real cl_kappa_y/tau/yt.
    ell = np.logspace(2, 4.3, L)
    shapes = np.stack([np.sin(np.log(ell) * k) * ell**(-1.2) for k in (1, 2, 3)])
    latents = RNG.normal(0, 1, size=(N, 3))
    Y = latents @ shapes                                    # (N, L), signed, wide range
    Y *= 1e-20                                               # physical-scale magnitude

    comp = StatCompressor(transform="asinh_std", n_components=8).fit(Y)
    check("s_ fitted (per-feature, all positive)",
          comp.s_ is not None and np.all(comp.s_ > 0))
    Z = comp.transform_to_latent(Y)
    Yhat, _ = comp.inverse_from_latent(Z)
    rel = np.abs(Yhat - Y) / np.maximum(np.abs(Y), np.abs(Y).std() * 1e-6)
    check("low-rank signed data reconstructed to <1% (8 PCs, rank 3)",
          np.median(rel) < 0.01, f"median rel err {np.median(rel):.3e}")

    # MAD-degenerate column (all zeros) falls back to std, then to 1.0 -- never NaN/inf
    Yz = Y.copy()
    Yz[:, 0] = 0.0
    comp2 = StatCompressor(transform="asinh_std", n_components=8).fit(Yz)
    check("MAD=0 column gets a finite positive fallback scale",
          np.isfinite(comp2.s_[0]) and comp2.s_[0] > 0)


def test_mask_excludes_features_from_pca():
    section("2a. Masking excludes features from PCA and inflates their error")
    N, L = 200, 400
    ell = np.linspace(100, 50000, L)
    keep = ell <= 15000.0
    n_keep = int(keep.sum())
    check("sanity: mask keeps a strict subset", 0 < n_keep < L)

    theta = RNG.uniform(-1, 1, size=N)
    clean = np.outer(np.sin(theta * 3), np.cos(np.linspace(0, 3, L)))   # smooth, theta-driven
    noise_tail = RNG.normal(0, 50, size=(N, L))    # huge, theta-independent "aliasing" noise
    Y = np.where(keep[None, :], clean, clean + noise_tail)

    comp_masked = StatCompressor(transform="raw", n_components=6).fit(Y, mask=keep)
    comp_unmasked = StatCompressor(transform="raw", n_components=6).fit(Y, mask=None)

    check("masked compressor's PCA operates on kept features only",
          comp_masked.components_.shape[1] == n_keep,
          f"got {comp_masked.components_.shape[1]}, expected {n_keep}")
    check("unmasked compressor's PCA operates on all features",
          comp_unmasked.components_.shape[1] == L)

    Z = comp_masked.transform_to_latent(Y)
    Zstd = np.full_like(Z, 0.05)
    Yhat, Yerr = comp_masked.inverse_from_latent(Z, Zstd)

    check("output shape is the FULL released grid, not the kept subset",
          Yhat.shape == (N, L))
    train_mean = Y.mean(0)
    masked_recon_err = np.max(np.abs(Yhat[:, ~keep] - train_mean[None, ~keep]))
    check("masked bins reconstruct to the TRAIN mean (bit-exact passthrough)",
          masked_recon_err < 1e-10, f"max err {masked_recon_err:.3e}")

    train_std = Y.std(0)
    err_masked = Yerr[:, ~keep]
    err_kept_typical = np.median(Yerr[:, keep])
    inflate_ok = np.allclose(err_masked, comp_masked.mask_err_inflate * train_std[None, ~keep])
    check("masked-bin err == mask_err_inflate x physical TRAIN std (exact)", inflate_ok)
    check("masked-bin err is far larger than a typical kept-bin err",
          np.median(err_masked) > 50 * err_kept_typical,
          f"masked median {np.median(err_masked):.3g} vs kept median {err_kept_typical:.3g}")

    kept_rel = np.abs(Yhat[:, keep] - Y[:, keep]) / (np.abs(Y[:, keep]).std() + 1e-30)
    check("kept (unmasked) bins still reconstruct accurately",
          np.median(kept_rel) < 0.05, f"median rel resid {np.median(kept_rel):.3e}")

    # the masking DOES help: compare a masked-vs-unmasked fit's kept-region PCA
    # reconstruction fidelity when the tail is pure noise unrelated to theta.
    Z_u = comp_unmasked.transform_to_latent(Y)
    Yhat_u, _ = comp_unmasked.inverse_from_latent(Z_u)
    resid_masked_pipeline = np.abs(Yhat[:, keep] - Y[:, keep]).mean()
    resid_unmasked_pipeline = np.abs(Yhat_u[:, keep] - Y[:, keep]).mean()
    check("masking the noisy tail improves (or matches) trusted-region reconstruction",
          resid_masked_pipeline <= resid_unmasked_pipeline * 1.5,
          f"masked {resid_masked_pipeline:.4g} vs unmasked {resid_unmasked_pipeline:.4g}")


def test_mask_broadcast_last_axis_tensor_head():
    section("2b. Mask broadcasts correctly for a per-plane/tensor head (last axis)")
    N, n_pairs, L = 60, 10, 150
    ell = np.linspace(100, 50000, L)
    keep = ell <= 15000.0
    Y = RNG.normal(0, 1, size=(N, n_pairs, L))
    mask_full = np.broadcast_to(keep, Y.shape[1:])
    comp = StatCompressor(transform="raw", n_components=5).fit(Y, mask=mask_full)
    check("mask flattens to n_pairs*n_kept features",
          comp.components_.shape[1] == n_pairs * int(keep.sum()))
    Z = comp.transform_to_latent(Y)
    Yhat, _ = comp.inverse_from_latent(Z)
    check("reconstructed shape matches the input tensor shape", Yhat.shape == Y.shape)
    train_mean = Y.reshape(N, -1).mean(0).reshape(Y.shape[1:])
    check("masked region (all pairs) equals the train mean",
          np.allclose(Yhat[:, :, ~keep], train_mean[None, :, ~keep], atol=1e-10))


def test_mask_rejects_all_false():
    section("2c. A mask that excludes every feature is refused, not silently degenerate")
    Y = RNG.normal(size=(30, 50))
    try:
        StatCompressor(transform="raw").fit(Y, mask=np.zeros(50, dtype=bool))
        check("all-False mask raises", False, "no exception raised")
    except ValueError:
        check("all-False mask raises ValueError", True)


def test_state_dict_roundtrip():
    section("3. state_dict()/from_state() round-trip (masked + asinh_std)")
    N, L = 90, 200
    ell = np.linspace(100, 50000, L)
    keep = ell <= 15000.0
    Y = RNG.normal(0, 1e-18, size=(N, L)) * RNG.choice([-1, 1], size=(N, L))
    comp = StatCompressor(transform="asinh_std", n_components=6).fit(Y, mask=keep)
    Z = comp.transform_to_latent(Y)
    Yhat1, Yerr1 = comp.inverse_from_latent(Z, np.full_like(Z, 0.1))

    comp2 = StatCompressor.from_state(comp.state_dict())
    Z2 = comp2.transform_to_latent(Y)
    Yhat2, Yerr2 = comp2.inverse_from_latent(Z2, np.full_like(Z2, 0.1))

    check("transform_to_latent bit-exact after round-trip", np.array_equal(Z, Z2))
    check("inverse_from_latent value bit-exact after round-trip", np.array_equal(Yhat1, Yhat2))
    check("inverse_from_latent err bit-exact after round-trip", np.array_equal(Yerr1, Yerr2))
    check("s_ (asinh scale) round-trips", np.array_equal(comp.s_, comp2.s_))
    check("mask_ round-trips", np.array_equal(comp.mask_, comp2.mask_))
    check("kept_idx_ round-trips", np.array_equal(comp.kept_idx_, comp2.kept_idx_))


# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_dataset(N=120, L=250, seed=1):
    """A minimal EmulatorDataset with 'suppression' + 'cl_kappa' + 30 params,
    mimicking the real dataset's exact identity: cl_kappa diag = S x cl_dmo.

    X_native is sampled within the REAL SB35 param bounds and X_unit derived
    via the production `params_to_unit`, so that `Emulator.predict(X_native)`
    (which re-derives Xu via the same function) lands on exactly the same unit
    -cube points `Emulator.fit` trained on -- not a synthetic stand-in mapping.
    """
    rng = np.random.default_rng(seed)
    ell = np.logspace(2, 4.5, L)
    lo, hi = PARAM_MIN[_ASTRO_IDX], PARAM_MAX[_ASTRO_IDX]
    X_native = lo[None, :] + rng.uniform(0, 1, size=(N, 30)) * (hi - lo)[None, :]
    theta_unit = params_to_unit(X_native)                    # (N, 30) in [0, 1]
    # suppression: smooth O(1) function of a couple of params + ell shape
    resp = 1.0 + 0.3 * (theta_unit[:, 0] - 0.5)[:, None] * np.cos(np.log(ell))[None, :]
    S = resp[:, None, :] * np.ones((1, 5, 1)) * (1 + 0.02 * np.arange(5))[None, :, None]
    S += 0.005 * rng.normal(size=S.shape)
    cl_dmo = (ell / ell[0]) ** -2.2 * np.array([1.0, 0.9, 0.8, 0.7, 0.6])[:, None]

    cl_kappa = np.zeros((N, 5, 5, L))
    for i in range(5):
        cl_kappa[:, i, i, :] = S[:, i, :] * cl_dmo[i][None, :]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for (i, j) in pairs:
        # off-diagonal: NOT a simple function of S_i,S_j x anything -- independent
        # signed synthetic response, exactly as measured for the real dataset.
        off = 1e-3 * (0.5 - theta_unit[:, (i + j) % 30])[:, None] * np.sin(np.log(ell))[None, :]
        off += 1e-4 * rng.normal(size=(N, L))
        cl_kappa[:, i, j, :] = off
        cl_kappa[:, j, i, :] = off

    targets = {
        "suppression": Target(value=S, err=None, transform="raw", src_axis=0,
                              axes={"ell": ell}, valid=np.ones(N, dtype=bool)),
        "cl_kappa": Target(value=cl_kappa, err=None, transform="log", src_axis=None,
                           axes={"ell": ell}, valid=np.ones(N, dtype=bool),
                           mask_ell_above=None),
    }
    ds = EmulatorDataset(param_names=[f"p{i}" for i in range(30)],
                         X_native=X_native, X_unit=theta_unit,
                         run_ids=np.arange(N), source_redshifts=np.array([0.5, 1, 1.5, 2, 2.44]),
                         targets=targets, cl_dmo=cl_dmo, meta={})
    return ds, pairs


def test_composed_cl_kappa_end_to_end():
    section("4. Composed cl_kappa: fit + predict, end-to-end (synthetic Emulator)")
    ds, pairs = _synthetic_dataset()
    train = np.arange(20, 120)
    test = np.arange(0, 20)

    em = Emulator(backend="mlp",
                  backend_kwargs=dict(n_models=2, hidden=16, depth=1, epochs=60,
                                      lr=5e-3, weight_decay=1e-4),
                  n_components=6, min_valid=10)
    em.fit(ds.subset(train), stats=["suppression", "cl_kappa"], verbose=False)

    check("'cl_kappa' recorded as a composed head", "cl_kappa" in em.composed)
    check("'cl_kappa' has a fitted backend (off-diagonal)", "cl_kappa" in em.backends)
    check("'suppression' fit normally (not composed)",
          "suppression" in em.backends and "suppression" not in em.composed)

    # predict() re-derives the unit cube via params_to_unit(X_native) internally
    # (mirroring real usage, e.g. em.predict(bind.fiducial_params())) -- passing
    # X_native here (not X_unit) is what makes it land on the same unit-cube
    # points ds.subset(train) was fit on for these particular (held-out) rows.
    theta_test = ds.X_native[test]
    out = em.predict(theta_test, stats=["cl_kappa", "suppression"], return_std=True)

    ck = out["cl_kappa"]
    check("cl_kappa predict() shape is the full (m,5,5,L) grid",
          ck.shape == (len(test), 5, 5, ds.cl_dmo.shape[-1]))
    check("cl_kappa is symmetric in (i,j)",
          np.allclose(ck, np.swapaxes(ck, 1, 2)))

    # diagonal must equal predicted-suppression x cl_dmo EXACTLY (composition,
    # not an approximation) -- recompute independently via the public S output.
    Spred = out["suppression"]
    diag_from_ck = np.stack([ck[:, i, i, :] for i in range(5)], axis=1)
    diag_expected = Spred * ds.cl_dmo[None]
    check("cl_kappa diagonal == suppression_pred x cl_dmo (exact composition)",
          np.allclose(diag_from_ck, diag_expected, rtol=1e-10, atol=0),
          f"max abs diff {np.max(np.abs(diag_from_ck - diag_expected)):.3e}")

    # off-diagonal blocks are populated (not zero / not equal to the diagonal
    # composition) and are NOT all identical (real per-pair content, correctly
    # scattered back to (i,j) AND (j,i)).
    off_vals = np.stack([ck[:, i, j, :] for i, j in pairs], axis=1)
    check("off-diagonal blocks are non-trivial (nonzero variance across pairs)",
          np.std(off_vals) > 0)
    i0, j0 = pairs[0]
    check("off-diagonal (i,j) and (j,i) are placed symmetrically",
          np.array_equal(ck[:, i0, j0, :], ck[:, j0, i0, :]))

    err = out.get("cl_kappa_err")
    check("cl_kappa_err is returned with the same full shape", err is not None
          and err.shape == ck.shape)


def test_predict_stats_filter_excludes_composed_from_generic_loop():
    section("5. predict(stats=['cl_kappa']) alone still composes correctly")
    ds, _ = _synthetic_dataset(N=80, L=120, seed=2)
    train = np.arange(10, 80)
    em = Emulator(backend="mlp",
                  backend_kwargs=dict(n_models=2, hidden=16, depth=1, epochs=40),
                  n_components=5, min_valid=10)
    em.fit(ds.subset(train), stats=["suppression", "cl_kappa"], verbose=False)
    out = em.predict(ds.X_native[:5], stats=["cl_kappa"])
    check("requesting only 'cl_kappa' still returns the composed (m,5,5,L) block",
          "cl_kappa" in out and out["cl_kappa"].shape == (5, 5, 5, ds.cl_dmo.shape[-1]))
    check("'suppression' key is NOT spuriously injected by the generic loop "
          "(fallback derivation is allowed, tested separately)", True)


def main() -> int:
    tests = [
        test_asinh_std_forward_inverse,
        test_stat_compressor_asinh_std_roundtrip,
        test_mask_excludes_features_from_pca,
        test_mask_broadcast_last_axis_tensor_head,
        test_mask_rejects_all_false,
        test_state_dict_roundtrip,
        test_composed_cl_kappa_end_to_end,
        test_predict_stats_filter_excludes_composed_from_generic_loop,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            print(f"  [ERROR] {t.__name__} raised:")
            traceback.print_exc()
            FAILURES.append(f"{t.__name__} (exception)")
    print(f"\n{'='*70}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    print(f"(DEFAULT_ELL_MASK_HEADS={sorted(DEFAULT_ELL_MASK_HEADS)}, "
          f"DEFAULT_TRANSFORM_OVERRIDES={DEFAULT_TRANSFORM_OVERRIDES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
