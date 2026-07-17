"""Fit the WP-A4 gas-observable emulator from ``gasemu_dataset.npz``.

Training-side counterpart of :mod:`gasemu` (numpy-only inference). Requires
``gpytorch`` and ``scikit-learn`` (the torch3 venv). The recipe is the
validated ``bind.wlemu`` construction, ported: per snapshot, standardize the
log/linear-transformed observable vector over the valid dims, compress with
PCA, fit one exact ARD Matern-5/2 GP per PCA coefficient (batched, float64),
export a plain-numpy artifact, and verify the numpy posterior against
gpytorch before shipping.

Usage::

    python -m analysis.paper3a.emulator.fit \
        --dataset /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/gasemu_dataset.npz \
        --artifact-out /mnt/ceph/users/mlee1/paper3/A/wp4_emulator/gasemu_gp.npz \
        --kfold 8 --validation-out .../gasemu_kfold.npz \
        --outdesign-out .../gasemu_outdesign.npz
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .gasemu import GasEmulator, _matern25, is_log_block


# ------------------------------------------------------------------ data -----

class GasTable:
    """(X, Y) for one snapshot from the dataset npz, with the log/linear
    transform and train-row standardization (no leakage)."""

    def __init__(self, f: dict, snap: str, train_rows: np.ndarray):
        self.snap = snap
        self.block_names = [str(b) for b in f["block_names"]]
        sizes = np.asarray(f["block_sizes"], int)
        offs = np.concatenate([[0], np.cumsum(sizes)])
        self.block_slices = {b: slice(int(offs[i]), int(offs[i + 1]))
                             for i, b in enumerate(self.block_names)}
        self.valid = np.asarray(f[f"snap{snap}_valid"], bool)

        self.X = np.asarray(f[f"snap{snap}_X_sb35"], np.float64)
        self.Y_phys = np.asarray(f[f"snap{snap}_Y_sb35"], np.float64)
        self.sem_phys = np.asarray(f[f"snap{snap}_sem_sb35"], np.float64)
        self.R = len(self.X)

        self.Y_t = self.transform(self.Y_phys)[:, self.valid]      # (R, Dv)
        self.mu = self.Y_t[train_rows].mean(axis=0)
        self.sd = self.Y_t[train_rows].std(axis=0) + 1e-12
        self.Y = (self.Y_t - self.mu) / self.sd

    def transform(self, Yp: np.ndarray) -> np.ndarray:
        Yt = np.array(Yp, np.float64, copy=True)
        for b in self.block_names:
            if not is_log_block(b):
                continue
            s = self.block_slices[b]
            Yt[..., s] = np.log10(np.maximum(Yp[..., s], 1e-300))
        return Yt


# ------------------------------------------------------------------- fit -----

def _fit_gp_batch(X: np.ndarray, C_scaled: np.ndarray, iters: int, lr: float,
                  seed: int, device: str | None):
    """Batch of exact GPs (one per column of ``C_scaled``), constant mean +
    ScaleKernel(ARD Matern-5/2); hyperparameters back as numpy."""
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
    """alpha = (K + noise I)^-1 (y - m) with the numpy kernel the emulator
    predicts with (fit/predict consistency guaranteed)."""
    B, n = C_scaled.shape[1], X.shape[0]
    alpha = np.empty((B, n))
    for b in range(B):
        diff = (X[:, None, :] - X[None, :, :]) / h["lengthscale"][b]
        K = h["outputscale"][b] * _matern25(np.sqrt((diff**2).sum(-1)))
        K[np.diag_indices(n)] += h["noise"][b]
        alpha[b] = np.linalg.solve(K, C_scaled[:, b] - h["mean_const"][b])
    return alpha


def fit_snap(table: GasTable, train_rows: np.ndarray, n_pca: int, iters: int,
             lr: float, seed: int, device: str | None) -> dict:
    """Fit one snapshot on ``train_rows``; returns per-snap artifact arrays."""
    from sklearn.decomposition import PCA

    X, Y = table.X[train_rows], table.Y[train_rows]
    n_pca = min(n_pca, Y.shape[1], len(train_rows))
    pca = PCA(n_pca).fit(Y)
    C = pca.transform(Y)
    c_sd = C.std(axis=0) + 1e-12
    h = _fit_gp_batch(X, C / c_sd, iters, lr, seed, device)
    return {
        "valid": table.valid,
        "pca_components": pca.components_, "pca_mean": pca.mean_, "c_sd": c_sd,
        "alpha": _alpha_from_hypers(X, C / c_sd, h),
        "y_mu": table.mu, "y_sd": table.sd, **h,
    }


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def build_arrays(f: dict, train_rows: np.ndarray, n_pca: int, iters: int,
                 lr: float, seed: int, device: str | None,
                 verbose: bool = True) -> dict:
    """Fit every snapshot; assemble the full artifact dict."""
    snaps = [str(s) for s in f["snaps"]]
    arrays: dict = {k: np.asarray(f[k]) for k in (
        "block_names", "block_sizes", "snaps", "snap_z", "radii_arcmin",
        "logm500_bin_edges", "ksz_bin0_range", "ksz_bin1_range",
        "param_names", "param_min", "param_max", "param_log", "param_fiducial",
        "fixed_cosmology_names", "fixed_cosmology_values")}
    arrays["X_train"] = np.asarray(f[f"snap{snaps[0]}_X_sb35"])[np.sort(train_rows)]
    for s in snaps:
        t0 = time.time()
        table = GasTable(f, s, np.sort(train_rows))
        d = fit_snap(table, np.sort(train_rows), n_pca, iters, lr, seed, device)
        for k, v in d.items():
            arrays[f"snap{s}_{k}"] = v
        if verbose:
            print(f"  snap {s} fit in {time.time() - t0:.1f}s", flush=True)
    arrays["provenance"] = np.array(json.dumps({
        "created": datetime.date.today().isoformat(),
        "dataset_provenance": json.loads(str(f["provenance"])),
        "n_train_rows": int(len(train_rows)), "n_pca": n_pca,
        "gp_iters": iters, "gp_lr": lr, "seed": seed, "git_sha": _git_sha(),
        "recipe": "per-snap PCA + exact ARD Matern-5/2 GPs (bind.wlemu port)",
    }))
    return arrays


# ------------------------------------------------------------- validation ----

def crosscheck_vs_gpytorch(arrays: dict, f: dict, train_rows: np.ndarray,
                           n_query: int = 8) -> float:
    """Verify the numpy posterior mean reproduces the exact GP posterior
    computed with gpytorch's own kernel matrices from the training
    coefficients (independent of the stored ``alpha``); max |rel diff|."""
    import gpytorch
    import torch

    emu = GasEmulator(arrays)
    rng = np.random.default_rng(1)
    Xq = rng.uniform(0.1, 0.9, size=(n_query, emu.n_params))
    rows = np.sort(train_rows)
    worst = 0.0
    for s in emu.snaps:
        m = emu._models[s]
        table = GasTable(f, s, rows)
        C_scaled = ((table.Y[rows] - m.pca_mean) @ m.components.T) / m.c_sd  # (n, P)
        B, n = m.alpha.shape
        Xt = torch.as_tensor(m.X, dtype=torch.float64)
        Xb = Xt.unsqueeze(0).expand(B, *Xt.shape)
        kern = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=m.X.shape[1],
                                          batch_shape=torch.Size([B])),
            batch_shape=torch.Size([B])).to(torch.float64)
        kern.base_kernel.lengthscale = torch.as_tensor(m.ls)[:, None, :]
        kern.outputscale = torch.as_tensor(m.outputscale)
        with torch.no_grad():
            K = kern(Xb).to_dense().numpy()
            K += m.noise[:, None, None] * np.eye(n)
            Xqb = torch.as_tensor(Xq).unsqueeze(0).expand(B, n_query, -1)
            ks = kern(Xqb, Xb).to_dense().numpy()
        got, _ = m.posterior(Xq, return_std=False)
        ref = np.empty_like(got)
        for b in range(B):
            ref[:, b] = m.mean_const[b] + ks[b] @ np.linalg.solve(
                K[b], C_scaled[:, b] - m.mean_const[b])
        # compare in the standardized prediction frame: near-degenerate PCA
        # tail components (c_sd ~ eps) carry no prediction weight and must
        # not fail the check on amplified roundoff
        dY = ((got - ref) * m.c_sd) @ m.components
        worst = max(worst, float(np.max(np.abs(dY))))
    return worst


def kfold_validation(f: dict, n_folds: int, n_pca: int, iters: int, lr: float,
                     seed: int, device: str | None) -> dict:
    """Every Sobol run predicted by an emulator fit with its fold held out.
    Returns arrays for the validation npz (physical units)."""
    snaps = [str(s) for s in f["snaps"]]
    R = len(np.asarray(f[f"snap{snaps[0]}_X_sb35"]))
    D = int(np.asarray(f["block_sizes"], int).sum())
    rng = np.random.default_rng(seed)
    fold_of_run = rng.permutation(R) % n_folds

    pred = np.full((R, len(snaps), D), np.nan)
    pred_std = np.full((R, len(snaps), D), np.nan)
    truth = np.stack([np.asarray(f[f"snap{s}_Y_sb35"]) for s in snaps], axis=1)
    sem = np.stack([np.asarray(f[f"snap{s}_sem_sb35"]) for s in snaps], axis=1)
    X_all = np.asarray(f[f"snap{snaps[0]}_X_sb35"])

    for fold in range(n_folds):
        held = np.where(fold_of_run == fold)[0]
        train = np.where(fold_of_run != fold)[0]
        print(f"fold {fold + 1}/{n_folds}: {len(held)} held-out runs", flush=True)
        arrays = build_arrays(f, train, n_pca, iters, lr, seed, device, verbose=False)
        emu = GasEmulator(arrays)
        for zi, s in enumerate(snaps):
            v, sd = emu.predict_vector(X_all[held], s)
            pred[held, zi] = v
            pred_std[held, zi] = sd
    return {
        "pred": pred.astype(np.float32), "pred_std": pred_std.astype(np.float32),
        "truth": truth.astype(np.float32), "sem": sem.astype(np.float32),
        "fold_of_run": fold_of_run,
        "snaps": np.asarray(f["snaps"]), "block_names": np.asarray(f["block_names"]),
        "block_sizes": np.asarray(f["block_sizes"]),
        "provenance": np.array(json.dumps({
            "created": datetime.date.today().isoformat(), "protocol": "k-fold",
            "n_folds": n_folds, "n_pca": n_pca, "gp_iters": iters, "gp_lr": lr,
            "seed": seed, "git_sha": _git_sha()})),
    }


def outdesign_test(f: dict, arrays: dict) -> dict:
    """Predict the 60 twobound 1P extremes + the fiducial (none in the Sobol
    training design) with the production emulator; return comparison arrays."""
    emu = GasEmulator(arrays)
    snaps = emu.snaps
    out: dict = {"snaps": np.asarray(f["snaps"]),
                 "block_names": np.asarray(f["block_names"]),
                 "block_sizes": np.asarray(f["block_sizes"])}
    for bundle in ("twobound", "fiducial"):
        X = np.asarray(f[f"snap{snaps[0]}_X_{bundle}"])
        pred = np.stack([emu.predict_vector(X, s, return_std=False) for s in snaps], axis=1)
        truth = np.stack([np.asarray(f[f"snap{s}_Y_{bundle}"]) for s in snaps], axis=1)
        out[f"{bundle}_pred"] = pred.astype(np.float32)
        out[f"{bundle}_truth"] = truth.astype(np.float32)
        out[f"{bundle}_X"] = X
    out["twobound_param"] = np.asarray(f[f"snap{snaps[0]}_twobound_param"])
    out["twobound_bound"] = np.asarray(f[f"snap{snaps[0]}_twobound_bound"])
    return out


def frac_err_by_block(truth, pred, slices) -> dict:
    """Median |pred-truth|/|truth| per block over finite dims."""
    out = {}
    for k, s in slices.items():
        t, g = truth[..., s].ravel(), pred[..., s].ravel()
        m = np.isfinite(t) & np.isfinite(g) & (np.abs(t) > 0)
        out[k] = float(np.median(np.abs(g[m] - t[m]) / np.abs(t[m]))) if m.any() else np.nan
    return out


# ------------------------------------------------------------------- CLI -----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--artifact-out", default=None)
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--validation-out", default=None)
    ap.add_argument("--outdesign-out", default=None)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: n_pca=6, iters=60, kfold<=2")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_pca, args.iters = 6, 60
        if args.kfold:
            args.kfold = min(args.kfold, 2)

    with np.load(args.dataset, allow_pickle=False) as fz:
        f = {k: fz[k] for k in fz.files}
    snaps = [str(s) for s in f["snaps"]]
    R = len(np.asarray(f[f"snap{snaps[0]}_X_sb35"]))

    if args.kfold:
        out = kfold_validation(f, args.kfold, args.n_pca, args.iters, args.lr,
                               args.seed, args.device)
        if args.validation_out:
            Path(args.validation_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.validation_out, **out)
            print(f"wrote {args.validation_out}")
        sizes = np.asarray(out["block_sizes"], int)
        offs = np.concatenate([[0], np.cumsum(sizes)])
        slices = {str(b): slice(int(offs[i]), int(offs[i + 1]))
                  for i, b in enumerate(out["block_names"])}
        print("k-fold frac:", {k: round(v, 4) for k, v in
                               frac_err_by_block(out["truth"], out["pred"], slices).items()})

    if args.artifact_out:
        print(f"production fit on all {R} Sobol runs")
        arrays = build_arrays(f, np.arange(R), args.n_pca, args.iters, args.lr,
                              args.seed, args.device)
        worst = crosscheck_vs_gpytorch(arrays, f, np.arange(R))
        print(f"numpy-vs-gpytorch posterior mean max rel diff: {worst:.2e}")
        if worst > 1e-6:
            raise RuntimeError(f"numpy predictor disagrees with exact GP ({worst:.2e})")
        Path(args.artifact_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.artifact_out, **arrays)
        print(f"wrote {args.artifact_out} "
              f"({Path(args.artifact_out).stat().st_size/1e6:.1f} MB)")

        if args.outdesign_out:
            od = outdesign_test(f, arrays)
            np.savez_compressed(args.outdesign_out, **od)
            sizes = np.asarray(od["block_sizes"], int)
            offs = np.concatenate([[0], np.cumsum(sizes)])
            slices = {str(b): slice(int(offs[i]), int(offs[i + 1]))
                      for i, b in enumerate(od["block_names"])}
            print("out-of-design (twobound) frac:",
                  {k: round(v, 4) for k, v in
                   frac_err_by_block(od["twobound_truth"], od["twobound_pred"],
                                     slices).items()})
            print(f"wrote {args.outdesign_out}")


if __name__ == "__main__":
    main()
