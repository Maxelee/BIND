"""Fit the BIND WL statistics emulator from a statistics cache.

Training-side counterpart of :mod:`bind.wlemu.emulator` (which is numpy-only).
Requires ``gpytorch`` and ``scikit-learn`` (``pip install bind[wlemu-fit]``).

Input is a ``stats_cache.npz`` of per-(run, realization, source-redshift)
summary statistics measured on raytraced convergence maps (see
:mod:`bind.wlemu.stats` for the estimators), with the run-aligned
``params_unit`` design matrix. The regression target at each (run, z) is the
mean over realizations; the realization scatter sets the noise floor any
emulator is judged against.

Recipe (validated against 8 alternative designs — nearest-neighbor, linear /
quadratic ridge, RBF, gradient-boosted trees, MLP ensembles, joint-z GP — the
per-z PCA+GP wins on every statistic block): per source redshift, standardize
the log/linear-transformed statistics vector, compress with PCA, and fit one
exact ARD Matern-5/2 GP per PCA coefficient (batched, float64). The exported
artifact contains plain numpy arrays only; prediction never needs torch.

Usage
-----
Build the production artifact (fit on all runs) + k-fold validation file::

    python -m bind.wlemu.fit --stats <stats_cache.npz> \
        --artifact-out src/bind/assets/wlemu_gp.npz \
        --kfold 13 --validation-out examples/data/wlemu_validation.npz

``--standard-check`` reproduces the historical fixed-split protocol
(fit on 213 train runs, score on 20 test runs) for comparison against the
design-comparison table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .emulator import BLOCKS, LOG_BLOCKS, WLEmulator, _matern25

# The five SB35 parameters held fixed (IllustrisTNG fiducial cosmology) in the
# lightcone suite; the remaining 30 astro parameters are the emulator inputs.
COSMO_IDX = (0, 1, 6, 7, 8)  # Omega0, sigma8, OmegaBaryon, HubbleParam, n_s
FIXED_COSMOLOGY = {"Omega0": 0.3089, "sigma8": 0.8159, "OmegaBaryon": 0.0486,
                   "HubbleParam": 0.6774, "n_s": 0.9667}


# ------------------------------------------------------------------ data -----

class StatsTable:
    """Assemble (X, Y) from stats_cache.npz with per-run-mean targets.

    Per (run, z) row: ``Y_phys`` is the mean statistics vector over
    realizations, ``sem_phys`` its standard error. ``Y`` is the transformed
    (log10 for strictly positive high-dynamic-range blocks) vector,
    standardized per dimension with **train-run** statistics (no leakage).
    """

    def __init__(self, path: str, train_runs: np.ndarray):
        f = np.load(path)
        self.path = str(path)
        self.f = f
        R, N, Z = f["Cl"].shape[:3]
        self.R, self.N, self.Z = R, N, Z

        means, sems, slices, d0 = [], [], {}, 0
        for k in BLOCKS:
            a = np.asarray(f[k], np.float64)              # (R, N, Z, D)
            mu = a.mean(axis=1)                           # (R, Z, D)
            sem = a.std(axis=1) / np.sqrt(N)
            D = mu.shape[-1]
            slices[k] = slice(d0, d0 + D)
            d0 += D
            means.append(mu.reshape(R * Z, D))
            sems.append(sem.reshape(R * Z, D))
        self.Y_phys = np.concatenate(means, axis=1)       # (R*Z, D_tot)
        self.sem_phys = np.concatenate(sems, axis=1)
        self.slices = slices
        self.D = d0

        self.params = np.asarray(f["params_unit"], np.float64)   # (R, 30)
        self.X = np.repeat(self.params, Z, axis=0)               # (R*Z, 30)
        self.run_of_row = np.repeat(np.arange(R), Z)
        self.z_of_row = np.tile(np.arange(Z), R)

        self.Y_t = self._transform(self.Y_phys)
        tr = np.isin(self.run_of_row, train_runs)
        self.mu = self.Y_t[tr].mean(axis=0)
        self.sd = self.Y_t[tr].std(axis=0) + 1e-12
        self.Y = (self.Y_t - self.mu) / self.sd

    def rows_of(self, runs, z: int | None = None) -> np.ndarray:
        m = np.isin(self.run_of_row, runs)
        if z is not None:
            m &= self.z_of_row == z
        return np.where(m)[0]

    def _transform(self, Yp: np.ndarray) -> np.ndarray:
        Yt = Yp.copy()
        for k in LOG_BLOCKS:
            s = self.slices[k]
            Yt[:, s] = np.log10(np.maximum(Yp[:, s], 1e-300))
        return Yt


# ------------------------------------------------------------------- fit -----

def _fit_gp_batch(X: np.ndarray, C_scaled: np.ndarray, iters: int, lr: float,
                  seed: int, device: str | None):
    """Fit a batch of exact GPs (one per column of ``C_scaled``) with a
    constant mean and ScaleKernel(ARD Matern-5/2); returns hyperparameters as
    plain numpy arrays."""
    import gpytorch
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    B, d = C_scaled.shape[1], X.shape[1]
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    Xb = Xt.unsqueeze(0).expand(B, *Xt.shape)
    Cb = torch.as_tensor(C_scaled.T, dtype=torch.float64, device=device)

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, x, y, lik):
            super().__init__(x, y, lik)
            bs = torch.Size([B])
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=bs)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=bs),
                batch_shape=bs)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    lik = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=torch.Size([B])).to(device, torch.float64)
    model = _GP(Xb, Cb, lik).to(device, torch.float64)
    model.train()
    lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(model(Xb), Cb).sum()
        loss.backward()
        opt.step()

    return {
        "lengthscale": model.covar_module.base_kernel.lengthscale
                            .detach().cpu().numpy().reshape(B, d),
        "outputscale": model.covar_module.outputscale.detach().cpu().numpy().reshape(B),
        "noise": lik.noise.detach().cpu().numpy().reshape(B),
        "mean_const": model.mean_module.constant.detach().cpu().numpy().reshape(B),
    }


def _alpha_from_hypers(X: np.ndarray, C_scaled: np.ndarray, h: dict) -> np.ndarray:
    """alpha = (K + noise I)^-1 (y - m), computed with the same numpy kernel
    the emulator predicts with (guarantees fit/predict consistency)."""
    B, n = C_scaled.shape[1], X.shape[0]
    alpha = np.empty((B, n))
    for b in range(B):
        diff = (X[:, None, :] - X[None, :, :]) / h["lengthscale"][b]
        K = h["outputscale"][b] * _matern25(np.sqrt((diff**2).sum(-1)))
        K[np.diag_indices(n)] += h["noise"][b]
        alpha[b] = np.linalg.solve(K, C_scaled[:, b] - h["mean_const"][b])
    return alpha


def fit_z(table: StatsTable, train_runs: np.ndarray, z: int, n_pca: int,
          iters: int, lr: float, seed: int, device: str | None) -> dict:
    """Fit one source redshift; returns the per-z arrays of the artifact."""
    from sklearn.decomposition import PCA

    rows = table.rows_of(train_runs, z)
    X, Y = table.X[rows], table.Y[rows]
    pca = PCA(n_pca).fit(Y)
    C = pca.transform(Y)
    c_sd = C.std(axis=0) + 1e-12
    h = _fit_gp_batch(X, C / c_sd, iters, lr, seed, device)
    return {
        "pca_components": pca.components_, "pca_mean": pca.mean_, "c_sd": c_sd,
        "alpha": _alpha_from_hypers(X, C / c_sd, h), **h,
    }


def within_run_covariance(table: StatsTable, z: int) -> np.ndarray:
    """Covariance of the physical statistics vector of a single field
    (across the ``N`` map realizations of a run), averaged over all runs."""
    f = table.f
    per_real = np.concatenate(
        [np.asarray(f[k], np.float64)[:, :, z, :] for k in BLOCKS], axis=2)  # (R,N,D)
    d = per_real - per_real.mean(axis=1, keepdims=True)
    return np.einsum("rnd,rne->de", d, d) / (table.R * (table.N - 1))


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def build_arrays(table: StatsTable, train_runs: np.ndarray, n_pca: int,
                 iters: int, lr: float, seed: int, device: str | None,
                 with_cov: bool = True, verbose: bool = True) -> dict:
    """Fit all source redshifts and assemble the full artifact array dict."""
    from bind import params as bp

    f = table.f
    astro = [i for i in range(bp.N_PARAMS) if i not in COSMO_IDX]
    arrays: dict = {
        "format_version": np.array(1),
        "block_names": np.array(list(BLOCKS)),
        "block_sizes": np.array([table.slices[k].stop - table.slices[k].start
                                 for k in BLOCKS]),
        "source_redshifts": np.asarray(f["source_redshifts"], np.float64),
        "fov_deg": np.asarray(f["fov_deg"], np.float64),
        "smoothing_arcmin": np.asarray(f["smoothing_arcmin"], np.float64),
        "scat_res": np.asarray(f["scat_res"], np.int64),
        "n_real": np.array(table.N),
        "ell": np.asarray(f["ell"]), "pdf_x": np.asarray(f["pdf_x"]),
        "peak_x": np.asarray(f["peak_x"]), "mink_thr": np.asarray(f["mink_thr"]),
        "param_names": np.array([bp.PARAM_NAMES[i] for i in astro]),
        "param_descriptions": np.array([bp.PARAM_DESCRIPTIONS[i] for i in astro]),
        "param_min": bp.PARAM_MIN[astro], "param_max": bp.PARAM_MAX[astro],
        "param_log": bp.PARAM_LOG_FLAG[astro].astype(bool),
        "param_fiducial": bp.PARAM_FIDUCIAL[astro],
        "fixed_cosmology_names": np.array(list(FIXED_COSMOLOGY)),
        "fixed_cosmology_values": np.array(list(FIXED_COSMOLOGY.values())),
        "y_mu": table.mu, "y_sd": table.sd,
        "X_train": table.params[np.sort(train_runs)],
        "provenance": np.array(json.dumps({
            "created": datetime.date.today().isoformat(),
            "stats_cache": table.path,
            "n_runs_total": table.R, "n_train_runs": len(train_runs),
            "n_real": table.N, "n_pca": n_pca, "gp_iters": iters, "gp_lr": lr,
            "seed": seed, "bind_git_sha": _git_sha(),
            "training_data": ("BIND-baryonified IllustrisTNG-DMO lightcones, "
                              "raytraced 5x5 deg kappa maps at 1024^2; Sobol "
                              "design over the 30 SB35 astro parameters, "
                              "cosmology fixed (TNG fiducial)"),
        })),
    }
    for z in range(table.Z):
        t0 = time.time()
        d = fit_z(table, np.sort(train_runs), z, n_pca, iters, lr, seed, device)
        if with_cov:
            d["cov_phys"] = within_run_covariance(table, z).astype(np.float32)
        for k, v in d.items():
            arrays[f"z{z}_{k}"] = v
        if verbose:
            print(f"  z index {z} fit in {time.time() - t0:.1f}s", flush=True)
    return arrays


# ------------------------------------------------------------- validation ----

def crosscheck_vs_gpytorch(arrays: dict, table: StatsTable,
                           train_runs: np.ndarray, n_query: int = 8) -> float:
    """Verify the numpy posterior mean reproduces gpytorch's exact posterior
    at random query points. Returns the max |rel diff| over blocks/points."""
    import gpytorch
    import torch

    emu = WLEmulator(arrays)
    rng = np.random.default_rng(1)
    Xq = rng.uniform(0.1, 0.9, size=(n_query, table.X.shape[1]))
    worst = 0.0
    for z in range(table.Z):
        rows = table.rows_of(np.sort(train_runs), z)
        X, Y = table.X[rows], table.Y[rows]
        m = emu._z_models[z]
        C = (Y - m.pca_mean) @ m.components.T / m.c_sd
        B, n = m.alpha.shape

        Xt = torch.as_tensor(X, dtype=torch.float64)
        Xb = Xt.unsqueeze(0).expand(B, *Xt.shape)
        kern = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1],
                                          batch_shape=torch.Size([B])),
            batch_shape=torch.Size([B])).to(torch.float64)
        kern.base_kernel.lengthscale = torch.as_tensor(m.ls)[:, None, :]
        kern.outputscale = torch.as_tensor(m.outputscale)
        with torch.no_grad():
            K = kern(Xb).to_dense()
            K += torch.as_tensor(m.noise)[:, None, None] * torch.eye(n, dtype=torch.float64)
            Xqb = torch.as_tensor(Xq, dtype=torch.float64).unsqueeze(0).expand(B, n_query, -1)
            ks = kern(Xqb, Xb).to_dense()
            y = torch.as_tensor(C.T) - torch.as_tensor(m.mean_const)[:, None]
            ref = (torch.as_tensor(m.mean_const)[:, None]
                   + torch.einsum("bqn,bn->bq", ks, torch.linalg.solve(K, y))).numpy().T
        got, _ = m.posterior(Xq, return_std=False)
        worst = max(worst, float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-12))))
    return worst


def kfold_validation(stats_path: str, n_folds: int, n_pca: int, iters: int,
                     lr: float, seed: int, device: str | None) -> dict:
    """Refit the full recipe with each fold of runs held out; every run gets a
    held-out prediction. Returns arrays for the validation npz (physical
    units, shaped (R, Z, D))."""
    f0 = np.load(stats_path)
    R, N, Z = f0["Cl"].shape[:3]
    rng = np.random.default_rng(seed)
    fold_of_run = rng.permutation(R) % n_folds

    pred = pred_std = truth = sem = None
    for fold in range(n_folds):
        held = np.where(fold_of_run == fold)[0]
        train = np.where(fold_of_run != fold)[0]
        print(f"fold {fold + 1}/{n_folds}: {len(held)} held-out runs", flush=True)
        table = StatsTable(stats_path, train_runs=train)
        if truth is None:
            D = table.D
            pred = np.full((R, Z, D), np.nan)
            pred_std = np.full((R, Z, D), np.nan)
            truth = table.Y_phys.reshape(R, Z, D)
            sem = table.sem_phys.reshape(R, Z, D)
        arrays = build_arrays(table, train, n_pca, iters, lr, seed, device,
                              with_cov=False, verbose=False)
        emu = WLEmulator(arrays)
        for z in range(Z):
            v, s = emu.predict_vector(table.params[held], z_idx=z, clip=False)
            pred[held, z] = v
            pred_std[held, z] = s
    return {
        "pred": pred.astype(np.float32), "pred_std": pred_std.astype(np.float32),
        "truth": truth.astype(np.float32), "sem": sem.astype(np.float32),
        "fold_of_run": fold_of_run, "params_unit": np.asarray(f0["params_unit"]),
        "source_redshifts": np.asarray(f0["source_redshifts"]),
        "block_names": np.array(list(BLOCKS)),
        "block_sizes": np.array([np.asarray(f0[k]).shape[-1] for k in BLOCKS]),
        "ell": np.asarray(f0["ell"]), "pdf_x": np.asarray(f0["pdf_x"]),
        "peak_x": np.asarray(f0["peak_x"]), "mink_thr": np.asarray(f0["mink_thr"]),
        "n_real": np.array(N),
        "provenance": np.array(json.dumps({
            "created": datetime.date.today().isoformat(), "protocol": "k-fold",
            "n_folds": n_folds, "n_pca": n_pca, "gp_iters": iters, "gp_lr": lr,
            "seed": seed, "bind_git_sha": _git_sha(),
            "note": ("held-out predictions: each run predicted by an emulator "
                     "fit with that run's fold excluded (identical recipe to "
                     "the shipped artifact)")})),
    }


def frac_err_by_block(truth, pred, slices) -> dict:
    """Mean |pred-truth|/|truth| per block over populated bins."""
    out = {}
    for k, s in slices.items():
        t, g = truth[:, s], pred[:, s]
        errs = []
        for ti, gi in zip(t, g):
            m = np.isfinite(ti) & np.isfinite(gi)
            m &= np.abs(ti) > 1e-6 * np.abs(ti[m]).max()
            errs.append(np.mean(np.abs(gi[m] - ti[m]) / np.abs(ti[m])))
        out[k] = float(np.mean(errs))
    return out


def sigma_err_by_block(truth, pred, sem, slices) -> dict:
    """Median |pred-truth| / SEM per block (1 = realization-noise floor)."""
    out = {}
    for k, s in slices.items():
        t, g, e = truth[:, s], pred[:, s], np.maximum(sem[:, s], 1e-300)
        m = np.isfinite(t) & np.isfinite(g) & (sem[:, s] > 0)
        out[k] = float(np.median(np.abs(g[m] - t[m]) / e[m]))
    return out


def standard_split_check(stats_path: str, n_pca: int, iters: int, lr: float,
                         seed: int, device: str | None):
    """Reproduce the historical protocol: fit on the 213 train runs of the
    fixed split (seed 0), score on the 20 test runs."""
    f0 = np.load(stats_path)
    R = f0["Cl"].shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(R)
    test, train = np.sort(perm[:20]), np.sort(perm[40:])
    table = StatsTable(stats_path, train_runs=train)
    arrays = build_arrays(table, train, n_pca, iters, lr, seed, device,
                          with_cov=False)
    emu = WLEmulator(arrays)
    rows = table.rows_of(test)                      # run-major, test runs sorted
    pred = np.empty((len(rows), table.D))
    for z in range(table.Z):
        v = emu.predict_vector(table.params[test], z_idx=z, return_std=False,
                               clip=False)
        pred[table.z_of_row[rows] == z] = v         # same ascending run order
    print("frac:", {k: round(v, 4) for k, v in
                    frac_err_by_block(table.Y_phys[rows], pred, table.slices).items()})
    print("sigma:", {k: round(v, 2) for k, v in
                     sigma_err_by_block(table.Y_phys[rows], pred,
                                        table.sem_phys[rows], table.slices).items()})


# ------------------------------------------------------------------- CLI -----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", required=True, help="stats_cache.npz path")
    ap.add_argument("--artifact-out", default=None,
                    help="write the production artifact (fit on all runs) here")
    ap.add_argument("--kfold", type=int, default=0,
                    help="also run k-fold held-out validation with this many folds")
    ap.add_argument("--validation-out", default=None,
                    help="write k-fold held-out predictions npz here")
    ap.add_argument("--standard-check", action="store_true",
                    help="score the historical fixed train/test split and exit")
    ap.add_argument("--n-pca", type=int, default=32)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets (smoke test): n_pca=8, iters=60, 2 folds")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_pca, args.iters = 8, 60
        if args.kfold:
            args.kfold = min(args.kfold, 2)

    if args.standard_check:
        standard_split_check(args.stats, args.n_pca, args.iters, args.lr,
                             args.seed, args.device)
        return

    if args.kfold:
        out = kfold_validation(args.stats, args.kfold, args.n_pca, args.iters,
                               args.lr, args.seed, args.device)
        if args.validation_out:
            Path(args.validation_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.validation_out, **out)
            print(f"wrote {args.validation_out}")
        D = out["truth"].shape[-1]
        sizes = out["block_sizes"]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        slices = {str(b): slice(int(offs[i]), int(offs[i + 1]))
                  for i, b in enumerate(out["block_names"])}
        t = out["truth"].reshape(-1, D)
        p = out["pred"].reshape(-1, D)
        e = out["sem"].reshape(-1, D)
        print("k-fold frac:", {k: round(v, 4) for k, v in
                               frac_err_by_block(t, p, slices).items()})
        print("k-fold sigma:", {k: round(v, 2) for k, v in
                                sigma_err_by_block(t, p, e, slices).items()})

    if args.artifact_out:
        f0 = np.load(args.stats)
        R = f0["Cl"].shape[0]
        all_runs = np.arange(R)
        table = StatsTable(args.stats, train_runs=all_runs)
        print(f"production fit on all {R} runs")
        arrays = build_arrays(table, all_runs, args.n_pca, args.iters, args.lr,
                              args.seed, args.device)
        worst = crosscheck_vs_gpytorch(arrays, table, all_runs)
        print(f"numpy-vs-gpytorch posterior mean max rel diff: {worst:.2e}")
        if worst > 1e-6:
            raise RuntimeError("numpy predictor does not reproduce the exact GP "
                               f"posterior (max rel diff {worst:.2e})")
        Path(args.artifact_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.artifact_out, **arrays)
        mb = Path(args.artifact_out).stat().st_size / 1e6
        print(f"wrote {args.artifact_out} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
