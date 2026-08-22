"""r5_common.py -- shared machinery for referee comment 5 (latent noise).

Re-implements, VERBATIM in convention, the two pieces of the shipped
pipeline that the R5 experiments have to exercise:

  * ``measure_lambda(F)``  -- the 8-latent reduction of
    family_basis_all.py lines 408-457 (LAT_BINS / LAT_NAMES / _row_bin_med),
    applied to any per-halo field dict F (a Sobol row, the fid atlas, the
    truth atlas).
  * ``cv_e2e(LAM, ...)``   -- the end-to-end 5-fold CV of
    family_basis_all.py::lam_fit_amps + the D_pred = pred_cv @ Bs
    reduction, on the pipeline-native rng(1) folds.

Everything else (basis, mean, stored amplitudes, measured fiducial curves)
is read from the SHIPPED artefacts figs_preview/family_model_bundle.npz and
figs_preview/amplitude_sets.npz -- this module never re-derives them.

Read-only w.r.t. every campaign tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

P1 = Path("/mnt/home/mlee1/BIND/papers/01_pipeline")
CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
CUBE = SB35 / "analysis_cache/atlas_cubes/atlas_cube_snap096.npz"
ATLAS = CEPH / "bind_science/halo_atlas"
OUT = CEPH / "referee_work/r5"

STATS = ("clk", "pdf", "pk", "mn", "v0", "v1", "v2")

# ── the shipped latent reduction (family_basis_all.py L408-457) ────────────
OB_OM = 0.0486 / 0.3089
MIN_N = 5
LAT_BINS = {"13.0-13.2": (13.0, 13.2), "13.2-13.4": (13.2, 13.4),
            "13.4-13.6": (13.4, 13.6), "14.0-14.3": (14.0, 14.3)}
LAT_NAMES = ["f_star[13.2-13.4]", "logT[13.0-13.2]", "logY[13.0-13.2]",
             "logPe[14.0-14.3]", "c_gas[14.0-14.3]", "logY_ss[13.4-13.6]",
             "c_gas[13.2-13.4]", "logPe[13.4-13.6]"]
# (bin key, per-halo quantity key, pos, log) for each latent column
LAT_SPEC = [("13.2-13.4", "f_star", False, False),
            ("13.0-13.2", "T",      True,  True),
            ("13.0-13.2", "Y",      True,  True),
            ("14.0-14.3", "Pe",     True,  True),
            ("14.0-14.3", "c_gas",  False, False),
            ("13.4-13.6", "Y_ss",   True,  True),
            ("13.2-13.4", "c_gas",  False, False),
            ("13.4-13.6", "Pe",     True,  True)]

FIELD_KEYS = ["m_tot_500c_bg", "m_gas_500c_bg", "m_star_500c",
              "m_gas_200c_bg", "T_mw_500c", "Pe_mw_500c", "Y_500c"]


def _row_bin_med(arr_row, mask, pos=False, log=False, min_n=MIN_N):
    """family_basis_all.py::_row_bin_med, verbatim."""
    v = arr_row[mask]
    if pos:
        v = np.where(v > 0, v, np.nan)
    if np.isfinite(v).sum() < min_n:
        return np.nan
    m = np.nanmedian(v)
    return np.log10(m) if (log and m > 0) else m


def per_halo_quantities(F):
    """the 5 derived per-halo arrays the 8 latents reduce, + log10 M bin field."""
    mt, mg5, ms, mg2 = (F["m_tot_500c_bg"], F["m_gas_500c_bg"],
                        F["m_star_500c"], F["m_gas_200c_bg"])
    # NB f_star is the RAW ratio here; the Ob/Om division is applied to the
    # bin median (family_basis_all.py L449 divides after _row_bin_med)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = {"f_star": ms / mt,
             "T": F["T_mw_500c"],
             "Y": F["Y_500c"],
             "Pe": F["Pe_mw_500c"],
             "c_gas": mg5 / mg2,
             "Y_ss": F["Y_500c"] / np.where(mt > 0, mt, np.nan) ** (5 / 3)}
        lm = np.log10(np.where(mt > 0, mt, np.nan))
    return Q, lm


def measure_lambda(F, sel=None):
    """8-latent lambda from one per-halo field dict.  ``sel`` optionally
    restricts to a halo subset (bootstrap / split-half); the mass bins are
    always evaluated on the run's OWN log10 M_tot,500c,bg (shipped
    convention -- bin membership is per-run)."""
    Q, lm = per_halo_quantities(F)
    if sel is not None:
        Q = {k: v[sel] for k, v in Q.items()}
        lm = lm[sel]
    lam = np.empty(8)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, (bk, qk, pos, log) in enumerate(LAT_SPEC):
            lo, hi = LAT_BINS[bk]
            lam[c] = _row_bin_med(Q[qk], (lm >= lo) & (lm < hi), pos, log)
    lam[0] /= OB_OM                     # family_basis_all.py L449
    return lam


def load_cube(dtype=float):
    """per-halo Sobol / fid rows, aligned to the dataset's run_ids.

    ``dtype=np.float32`` reproduces family_basis_all.py bit-for-bit (that
    script never casts the cube, so its latent arithmetic runs in the cube's
    native float32); ``float`` (default) is the float64 path used for the
    referee experiments.  The two differ by <=1.3e-6 in lat_ref and by
    <2e-5 in every end-to-end CV R^2 -- verified in r5_repro.py."""
    cz = np.load(CUBE)
    dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    rows = np.asarray(dsn["run_ids"])
    SOB = {k: np.asarray(cz[f"sobol_{k}"], dtype)[rows] for k in FIELD_KEYS}
    FID = {k: np.asarray(cz[f"fid_{k}"], dtype) for k in FIELD_KEYS}
    return SOB, FID, rows, dsn


def load_truth():
    z = np.load(ATLAS / "truth_snap096.npz")
    return {k: np.asarray(z[k], float) for k in FIELD_KEYS}


def load_fid_atlas():
    z = np.load(ATLAS / "fid_snap096.npz")
    return {k: np.asarray(z[k], float) for k in FIELD_KEYS}


def sobol_LAT(SOB):
    n = len(SOB["m_tot_500c_bg"])
    return np.stack([measure_lambda({k: SOB[k][q] for k in FIELD_KEYS})
                     for q in range(n)])


# ── the shipped e2e CV (family_basis_all.py::lam_fit_amps + D_pred) ────────
def report_folds(n, seed=1):
    """family_basis_all.py convention: rng(seed).permutation -> 5 splits,
    train = setdiff of the whole permutation."""
    order = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(order, f), f) for f in np.array_split(order, 5)]


def cv_amps(LAM, A, folds, lat_ref=None):
    """5-fold CV predictions of the amplitude matrix A from latents LAM,
    linear map a = M.(lambda - lam_ref) + intercept (lam_ref only shifts the
    intercept; the shipped code uses the full-sample mean)."""
    ref = LAM.mean(0) if lat_ref is None else lat_ref
    Lc = LAM - ref
    pred = np.full_like(A, np.nan)
    for tr, te in folds:
        W, *_ = np.linalg.lstsq(np.c_[Lc[tr], np.ones(len(tr))], A[tr],
                                rcond=None)
        pred[te] = np.c_[Lc[te], np.ones(len(te))] @ W
    return pred


def cv_e2e(LAM, AMP, BAS, D, folds):
    """per-statistic end-to-end CV R^2 dict."""
    r2 = {}
    for st in STATS:
        pred = cv_amps(LAM, AMP[st], folds)
        Dp = pred @ BAS[st]
        r2[st] = float(1 - ((D[st] - Dp) ** 2).sum() / (D[st] ** 2).sum())
    return r2


def load_model_pieces():
    """shipped basis / mean / stored amplitudes + the Sobol statistic matrix
    D = Y - mean, all read straight off the shipped artefacts."""
    bun = np.load(P1 / "figs_preview/family_model_bundle.npz",
                  allow_pickle=True)
    amp = np.load(P1 / "figs_preview/amplitude_sets.npz", allow_pickle=True)
    dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
    coef = np.load(P1 / "latent_model_coeffs.npz")
    EDGb = coef["ell_edges"]
    eld = np.asarray(dsn["a__suppression__ell"], float)
    DK = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts",
          "mn": "minima_counts", "v0": "mf_v0", "v1": "mf_v1", "v2": "mf_v2"}
    BAS, MEAN, AMPS, D, Y = {}, {}, {}, {}, {}
    for st in STATS:
        BAS[st] = np.asarray(bun[f"{st}__basis"], float)
        MEAN[st] = np.asarray(bun[f"{st}__mean"], float)
        AMPS[st] = np.asarray(bun[f"{st}__amps"], float)
        Yv = np.asarray(dsn[f"t__{DK[st]}__value"], float)
        Yv = Yv[:, 1, :] if Yv.ndim == 3 else Yv
        if st == "clk":
            Yv = np.stack([np.nanmean(Yv[:, (eld >= EDGb[i])
                                          & (eld < EDGb[i + 1])], 1)
                           for i in range(len(EDGb) - 1)], axis=1)
        Yv = Yv[:, :BAS[st].shape[1]]
        Y[st] = Yv
        D[st] = Yv - MEAN[st]
    return bun, amp, dsn, BAS, MEAN, AMPS, D, Y
