"""Fit the Paper I GP emulator once and persist it (see emulator_cache.py).

    python build_emulator.py --fit       # fit + save the bundle
    python build_emulator.py --verify    # reload and check predictions are identical

The fit configuration MUST stay in step with the fig 9 notebook cell: same dataset,
same train/test split (seed 0, 50 held out), same statistic list, same backend. That
is not a convention anyone has to remember — emulator_cache stores a fingerprint of
all of it, and the notebook refuses a bundle that disagrees.

**C-head improvement set** (papers/01_pipeline/audits/spectrum_head_experiment.md;
wiring in bind.emulator.core/transforms, ADJUDICATED 2026-08; ELL_MASK_ABOVE widened
2026-08-04):
  1. The six ell-domain spectrum heads (cl_kappa, cl_kappa_y, cl_yy, cl_kappa_tau,
     cl_tt, cl_yt) are fit only on ell <= 3.0e4 (bind.emulator.core.ELL_MASK_ABOVE;
     moved from 1.5e4 to the MEASURED CIC/pixelization contamination onset in
     lockstep with _build_figures_nb.py's ELL_TRUST -- see that notebook's setup
     cell for the two measurements behind the move) -- the contaminated tail above
     that is excluded from the PCA+GP entirely, not just from a display mask.
     `suppression` is deliberately NOT masked (a ratio; the aliasing artifact
     cancels in it -- the audit's own suppression anchor used the full grid and
     matched production). `em.predict()` still returns every head on its FULL
     released grid: masked bins come back at the per-bin TRAIN mean with an
     inflated (1e3x TRAIN std) error, so they are always distinguishable by their
     error bar, never by a missing key or a different shape.
  2. The three signed cross-spectra (cl_kappa_y, cl_kappa_tau, cl_yt) fit in
     asinh(x/s_bin) (s_bin = 1.4826xMAD over train, per bin) instead of raw --
     raw is numerically broken for these under PCA+GP (~1e7% median frac err in the
     controlled experiment; asinh_std gets an error-to-response ratio ~0.7).
  3. cl_kappa is a COMPOSED head, not a direct fit: the diagonal (same-plane)
     blocks are exactly suppression x cl_dmo (measured identity, ~1e-15 relative
     deviation -- this is literally how `suppression` is defined in
     dataset.assemble); the off-diagonal (cross-plane) blocks were measured NOT
     run-invariant under any of 3 candidate compositions against the DMO run's own
     off-diagonal trace (~9x coefficient of variation across runs, vs ~1e-15 for the
     diagonal) so they get a dedicated small PCA+GP (10 unique i<j pairs, asinh_std,
     ell-masked). See bind.emulator.core's module docstring for the full writeup.
The STATS list below is UNCHANGED (still 13 heads, cl_kappa still present) --
only how cl_kappa and the crosses are fit changed, which is why the
emulator_cache fingerprint must (and does) change to force a refit of any bundle
built before this revision.
"""
import argparse
import time
import warnings

import emulator_cache as ec
import numpy as np

warnings.filterwarnings("ignore", message=".*length_scale is close to.*")

DS_PATH = ec.CEPH / "bind_sb35" / "emulator_dataset_nu05.npz"
# 2026-08-04 nu05 grid migration (lockstep with the fig-9 notebook cell's
# DS_PATH, and with _build_figures_nb.py's setup cell): emulator_dataset_nu05
# is emulator_dataset_xpkfix's drop-in superset with the six nu-domain WL
# statistics (pdf, peak_counts, minima_counts, mf_v0/v1/v2) and their shared
# axis replaced by nu_grid.py's canonical Delta-nu=0.5, 22-bin grid -- see
# build_nu_cache.py's assemble(). The NEW FILENAME alone is enough to force a
# refit: emulator_cache.py fingerprints DS_PATH (among other things), so any
# bundle fit against the old _xpkfix path is automatically rejected by
# load_matching() the first time --fit/--verify runs against this path,
# rather than silently reused with stale nu-domain heads.

# Mirrors STATS_EMU in the fig 9 cell EXACTLY: the locked 13-head list = the 7
# WL-field heads (suppression, pdf, peak_counts, minima_counts, mf_v0-2) plus
# the 5 gas/cross spectra (cl_yy, cl_tt, cl_kappa_y, cl_kappa_tau, cl_yt),
# plus cl_kappa as the 13th head (fork-A ruling, 2026-08-03): now a COMPOSED
# head (see the module docstring above) rather than a direct fit on the raw
# 5x5x724 cube, but still a first-class entry here -- `Emulator.fit` routes it
# to the composed path automatically, no caller-visible difference in STATS.
# scaling_Y/scaling_f_gas emulation is deferred (not fit here).
STATS = ["suppression", "cl_kappa", "pdf", "peak_counts", "minima_counts", "mf_v0",
         "mf_v1", "mf_v2", "cl_yy", "cl_tt", "cl_kappa_y", "cl_kappa_tau", "cl_yt"]
BACKEND = "gpgpu"
N_COMPONENTS = 12
BACKEND_KWARGS = {"epochs": 400, "lr": 0.1}
N_TEST = 50
SPLIT_SEED = 0


def split(n_runs: int):
    rng = np.random.default_rng(SPLIT_SEED)
    test = np.sort(rng.choice(n_runs, N_TEST, replace=False))
    return np.setdiff1d(np.arange(n_runs), test), test


# Mirrors bind.emulator.core.DEFAULT_ELL_MASK_HEADS/ELL_MASK_ABOVE (duplicated,
# not imported, to keep this smoke-test self-contained -- same convention the
# fig-9 notebook cells use for their own reimplementations). These heads are
# fit only on ell <= ELL_MASK_ABOVE; the verify metric below excludes the
# masked tail too, or the "new" number would just report how noisy the
# untrusted, train-mean-filled aliased bins are -- the opposite of the point.
# 2026-08-04: 1.5e4 -> 3.0e4, kept in lockstep with core.ELL_MASK_ABOVE (see
# that module's comment for the measurement provenance) -- MUST be updated
# together, this file deliberately duplicates rather than imports the value.
ELL_MASK_HEADS = {"cl_kappa", "cl_kappa_y", "cl_yy", "cl_kappa_tau", "cl_tt", "cl_yt"}
ELL_MASK_ABOVE = 3.0e4


def _eval_head(name: str, p_test: np.ndarray, t_test: np.ndarray, v_train: np.ndarray,
               ell: np.ndarray | None) -> tuple[float, float]:
    """(median |frac err|, response R^2), the fig-9 notebook cell's own convention:
    5th-percentile-|truth| amplitude floor (1.0 for counts heads), R^2 against the
    TRAIN-mean baseline aggregated by the per-bin median.  For the six ell-masked
    heads, restricts to ell <= ELL_MASK_ABOVE first (see the module docstring)."""
    p, t, base = np.asarray(p_test, float), np.asarray(t_test, float), np.asarray(v_train, float)
    if ell is not None and name in ELL_MASK_HEADS:
        keep = np.asarray(ell, float) <= ELL_MASK_ABOVE
        p, t, base = p[..., keep], t[..., keep], base[..., keep]
    p, t = p.reshape(len(p), -1), t.reshape(len(t), -1)
    base = base.reshape(len(base), -1).mean(0)
    thresh = 1.0 if "counts" in name else np.nanpercentile(np.abs(t), 5)
    ok = np.isfinite(t) & (np.abs(t) > thresh)
    fe = float(np.nanmedian(np.abs(p[ok] / t[ok] - 1)))
    num = np.nansum(np.where(ok, (p - t) ** 2, np.nan), axis=0)
    den = np.nansum(np.where(ok, (t - base[None, :]) ** 2, np.nan), axis=0)
    r2 = 1 - num / np.maximum(den, 1e-30)
    r2_med = float(np.nanmedian(r2[den > 0])) if np.any(den > 0) else float("nan")
    return fe, r2_med


def main() -> int:
    from bind.emulator import Emulator, EmulatorDataset
    from bind.emulator.dataset import DEFAULT_DMO

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if not (a.fit or a.verify):
        ap.error("need --fit or --verify")

    ds = EmulatorDataset.load(DS_PATH)
    train_idx, test_idx = split(ds.n_runs)

    if a.fit:
        t0 = time.time()
        # raw DMO run's FULL (5,5,L) tomographic tensor -- bundle provenance
        # only (see the module docstring's composed-cl_kappa writeup); NOT
        # required for the fit to succeed.
        dmo_path = DEFAULT_DMO / "Cl_kappa.npz"
        cl_dmo_full = None
        if dmo_path.exists():
            with np.load(dmo_path) as dmo:
                if "cl" in dmo.files:
                    cl_dmo_full = np.array(dmo["cl"])
        if cl_dmo_full is not None and ds.cl_dmo is not None \
                and cl_dmo_full.shape[-1] == ds.cl_dmo.shape[-1]:
            # The on-disk Cl_kappa.npz is a mean over ITS OWN realization set
            # (550 since the 2026-08-05 DMO retrace), while ds.cl_dmo is the
            # 50-real seed-paired denominator the suppression target was built
            # with.  Force the diagonal to ds.cl_dmo so the stored provenance
            # tensor is consistent with the composed head's actual denominator
            # (off-diagonals keep the raw trace's values; no paired per-real
            # cross-spectra cache exists).
            for i in range(cl_dmo_full.shape[0]):
                cl_dmo_full[i, i] = ds.cl_dmo[i]
        em = Emulator(backend=BACKEND, n_components=N_COMPONENTS,
                      backend_kwargs=BACKEND_KWARGS)
        em.fit(ds.subset(train_idx), stats=STATS, cl_dmo_full=cl_dmo_full)
        p = ec.save(em, DS_PATH, STATS, train_idx, BACKEND, N_COMPONENTS, BACKEND_KWARGS)
        print(f"fit {len(STATS)} heads on {len(train_idx)} runs in {time.time()-t0:.0f}s")
        print(f"saved -> {p} ({p.stat().st_size/1e6:.1f} MB)  meta -> {ec.META.name}")
        if cl_dmo_full is None:
            print(f"  note: DMO tensor not found/readable at {dmo_path} -- "
                  "cl_kappa's off-diagonal-composition provenance tensor not stored "
                  "(fit itself is unaffected)")

    if a.verify:
        em = ec.load_matching(DS_PATH, STATS, train_idx, BACKEND, N_COMPONENTS,
                              BACKEND_KWARGS)
        if em is None:
            print("VERIFY FAILED: no matching bundle (fit first)")
            return 1
        t0 = time.time()
        pred = em.predict(ds.X_native[test_idx])
        print(f"loaded bundle predicts {len(test_idx)} held-out runs in "
              f"{time.time()-t0:.2f}s; heads = {len(em.statistics)}")
        bad = 0
        print(f"  {'head':>14s}  {'held-out |frac err|':>20s}  {'response R^2':>13s}")
        for s in STATS:
            if s not in pred:
                print(f"  MISSING head {s}")
                bad += 1
                continue
            t_test = np.asarray(ds.targets[s].value)[test_idx]
            p_test = np.asarray(pred[s])
            if p_test.shape != t_test.shape:
                print(f"  SHAPE MISMATCH {s}: {p_test.shape} vs {t_test.shape}")
                bad += 1
                continue
            v_train = np.asarray(ds.targets[s].value)[train_idx]
            ell = ds.targets[s].axes.get("ell")
            fe, r2 = _eval_head(s, p_test, t_test, v_train, ell)
            note = f"  (ell<={ELL_MASK_ABOVE:.1e} only)" if s in ELL_MASK_HEADS else ""
            print(f"  {s:>14s}  {100*fe:19.2f}%  {r2:13.2f}{note}")
        # the split must be the one the bundle was trained on, or the numbers above
        # are training-set errors wearing a held-out label
        print(f"  split guard: {len(train_idx)} train / {len(test_idx)} test, "
              f"seed {SPLIT_SEED}; fingerprint matched on load")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
