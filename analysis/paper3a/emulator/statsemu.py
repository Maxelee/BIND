"""WP-A6 lightcone-statistics emulator (theta_SB35 -> map summary statistics).

Implements the A6 decision (Max, 2026-07-18: NO new lightcone generation):
statistics-level emulation over the EXISTING raytraced SB35 suite. The
training table is the Popeye measurement pass
``wp6_propagation/sb35_stats/emulator_dataset.npz`` — 253 valid Sobol runs
x 5 source redshifts, with every summary block the A6 plan names
(C_ell^{kappa kappa, kappa y, yy, kappa tau, tau tau, y tau}, peak/minima
counts, PDF, Minkowski functionals, moments, WST, kappa-peak profiles,
halo scaling relations, and the WL suppression R(ell)).

Recipe: the validated wlemu/gasemu construction, per TARGET (not per snap):
flatten the per-run statistic, apply the dataset manifest's transform
(log10 / log1p / raw), drop dims that are non-finite (or non-positive under
log) on any training run, standardize with train-row statistics, compress
with PCA, one exact ARD Matern-5/2 GP per PCA coefficient (batched
gpytorch in training; plain-numpy posterior at predict time — reuses
`gasemu._matern25` and the `fit.py` GP batch, so the kernel math is
byte-identical to the validated emulators).

Targets with fewer valid runs (wst: 40, dm_*: 177) train on their own
valid subset; `train_rows` is stored per target.

Training (serial or disBatch fan-out — one task per target + merge)::

    python -m analysis.paper3a.emulator.statsemu fit \
        [--holdout 32 | --kfold 8] [--targets suppression,cl_kappa_y,...] \
        [--artifact-out parts/suppression.npz]
    python -m analysis.paper3a.emulator.statsemu merge \
        --parts-dir <dir of per-target part npz> [--artifact-out ...]

``--kfold K`` replaces the quick holdout with a full K-fold CV over the
target's valid runs (out-of-fold predictions for every run), reporting
both the median fractional error and the SEM-relative error (the wlemu
noise-floor acceptance frame).

Prediction (numpy-only)::

    emu = StatsEmulator.load()
    r = emu.predict(u, "suppression")        # (N, 5, 724) R(ell) per z_s
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .gasemu import _matern25, _SnapModel  # noqa: F401  (kernel + posterior reuse)

WP6 = Path("/mnt/ceph/users/mlee1/paper3/A/wp6_propagation")
DATASET = WP6 / "sb35_stats" / "emulator_dataset.npz"
ARTIFACT = WP6 / "statsemu_gp.npz"


# --------------------------------------------------------------- transforms --

def _apply_transform(Y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log10(Y)
    if transform == "log1p":
        return np.log1p(Y)
    return Y


def _invert_transform(Yt: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return 10.0 ** Yt
    if transform == "log1p":
        return np.expm1(Yt)
    return Yt


def _std_to_physical(sd_t: np.ndarray, phys: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.abs(phys) * np.log(10.0) * sd_t
    if transform == "log1p":
        return np.abs(phys + 1.0) * sd_t
    return sd_t


# -------------------------------------------------------------------- table --

class TargetTable:
    """(X, Y) for one manifest target: flattened, transformed, standardized
    over the target's own valid+train rows (no leakage)."""

    def __init__(self, f: dict, name: str, manifest: dict,
                 train_rows: np.ndarray, transform_override: str | None = None):
        spec = manifest["targets"][name]
        self.name = name
        # An override replaces the manifest transform (recorded in the
        # artifact provenance); dims that a log override cannot represent
        # (any nonpositive training value) are masked invalid below —
        # e.g. cl_kappa_y is >0 on 97.4% of dims, and the log dynamic-range
        # compression is the candidate fix for its GP roughness.
        self.transform = transform_override or spec["transform"]
        val = np.asarray(f[f"t__{name}__value"], np.float64)
        self.feature_shape = val.shape[1:]
        Y = val.reshape(len(val), -1)
        self.valid_runs = np.asarray(f[f"t__{name}__valid"], bool)
        self.train_rows = self.valid_runs & train_rows
        if self.train_rows.sum() < 10:
            raise ValueError(f"{name}: only {self.train_rows.sum()} training runs")

        with np.errstate(divide="ignore", invalid="ignore"):
            Y_t = _apply_transform(Y, self.transform)
        tr = Y_t[self.train_rows]
        self.valid_dims = np.all(np.isfinite(tr), axis=0)
        if self.transform == "log":
            self.valid_dims &= np.all(Y[self.train_rows] > 0, axis=0)
        if self.valid_dims.sum() == 0:
            raise ValueError(f"{name}: no finite dims after transform")

        self.Y_phys = Y
        self.Y_t = np.where(np.isfinite(Y_t), Y_t, 0.0)[:, self.valid_dims]
        self.mu = self.Y_t[self.train_rows].mean(axis=0)
        self.sd = self.Y_t[self.train_rows].std(axis=0) + 1e-12
        self.Y = (self.Y_t - self.mu) / self.sd


def _fit_target(X: np.ndarray, table: TargetTable, rows: np.ndarray,
                n_pca: int, iters: int, lr: float, seed: int,
                device: str | None) -> dict:
    from sklearn.decomposition import PCA

    from .fit import _alpha_from_hypers, _fit_gp_batch

    Xr, Yr = X[rows], table.Y[rows]
    n_pca = min(n_pca, Yr.shape[1], len(Xr) - 1)
    pca = PCA(n_pca).fit(Yr)
    C = pca.transform(Yr)
    c_sd = C.std(axis=0) + 1e-12
    h = _fit_gp_batch(Xr, C / c_sd, iters, lr, seed, device)
    return {
        "valid": table.valid_dims,
        "pca_components": pca.components_, "pca_mean": pca.mean_, "c_sd": c_sd,
        "alpha": _alpha_from_hypers(Xr, C / c_sd, h),
        "y_mu": table.mu, "y_sd": table.sd, **h,
    }


# ------------------------------------------------------------------ runtime --

class StatsEmulator:
    """Numpy-only posterior over the fitted statistics targets."""

    def __init__(self, arrays: dict):
        self._raw = arrays
        self.provenance = json.loads(str(arrays["provenance"]))
        self.targets = [str(t) for t in arrays["targets"]]
        self.param_names = [str(p) for p in arrays["param_names"]]
        self.source_redshifts = np.asarray(arrays["source_redshifts"], float)
        self.X_unit = np.asarray(arrays["X_unit"], np.float64)
        self._models: dict[str, _SnapModel] = {}
        self._meta = {}
        for t in self.targets:
            d = {k.split("__", 1)[1]: arrays[k] for k in arrays
                 if k.startswith(t + "__")}
            rows = d.pop("train_rows").astype(bool)
            self._meta[t] = {
                "feature_shape": tuple(int(x) for x in d.pop("feature_shape")),
                "transform": str(d.pop("transform")),
            }
            self._models[t] = _SnapModel(self.X_unit[rows], d)

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "StatsEmulator":
        return cls(dict(np.load(path, allow_pickle=False)))

    def predict(self, params, target: str, return_std: bool = False):
        """Physical-frame prediction ``(N, *feature_shape)`` (NaN at dims the
        training set could not constrain), plus the same-shaped GP std when
        ``return_std``."""
        if target not in self._models:
            raise KeyError(f"target {target!r} not emulated; have {self.targets}")
        u = np.atleast_2d(np.asarray(params, np.float64))
        single = np.asarray(params).ndim == 1
        m = self._models[target]
        meta = self._meta[target]

        mean_s, std_s = m.posterior(u, return_std)
        C = mean_s * m.c_sd
        Y_std = C @ m.components + m.pca_mean
        Y_t = Y_std * m.y_sd + m.y_mu
        phys_v = _invert_transform(Y_t, meta["transform"])

        D = int(np.prod(meta["feature_shape"])) if meta["feature_shape"] else 1
        out = np.full((len(u), D), np.nan)
        out[:, m.valid] = phys_v
        out = out.reshape((len(u),) + meta["feature_shape"])
        if not return_std:
            return out[0] if single else out
        sd_C = std_s * m.c_sd
        sd_t = np.sqrt((sd_C**2) @ (m.components**2)) * m.y_sd
        sd_v = _std_to_physical(sd_t, phys_v, meta["transform"])
        sd = np.full((len(u), D), np.nan)
        sd[:, m.valid] = sd_v
        sd = sd.reshape((len(u),) + meta["feature_shape"])
        return (out[0], sd[0]) if single else (out, sd)


# ----------------------------------------------------------------- training --

def _load_dataset(path: Path):
    f = dict(np.load(path, allow_pickle=False))
    manifest = json.loads(str(f["manifest"]))
    return f, manifest


def _frac_err(truth: np.ndarray, pred: np.ndarray) -> float:
    """Median |pred-truth| / scale over holdout rows x valid dims; the scale
    per dim is median|truth| (frac err is ill-posed at zero crossings of the
    raw-transform blocks)."""
    scale = np.maximum(np.nanmedian(np.abs(truth), axis=0), 1e-300)
    ok = np.isfinite(truth) & np.isfinite(pred)
    return float(np.median((np.abs(pred - truth) / scale)[ok]))


def _predict_physical(Xq: np.ndarray, table: TargetTable, model: _SnapModel) -> np.ndarray:
    mean_s, _ = model.posterior(Xq, False)
    Y_t = (mean_s * model.c_sd) @ model.components + model.pca_mean
    Y_t = Y_t * model.y_sd + model.y_mu
    return _invert_transform(Y_t, table.transform)


def _validation_entry(f, table: TargetTable, rows_mask: np.ndarray,
                      pred: np.ndarray) -> dict:
    """Score out-of-sample predictions (rows_mask selects the scored runs)."""
    truth = table.Y_phys[rows_mask][:, table.valid_dims]
    entry = {"n_scored": int(rows_mask.sum()),
             "frac_err_med": _frac_err(truth, pred)}
    err_key = f"t__{table.name}__err"
    if err_key in f:
        sem = np.asarray(f[err_key], np.float64).reshape(len(table.Y_phys), -1)
        sem = sem[rows_mask][:, table.valid_dims]
        ok = np.isfinite(truth) & np.isfinite(pred) & (sem > 0)
        if ok.any():
            # The dataset mixes err conventions: the C_ell family /
            # peak_counts / wst carry LINEAR (value-unit) realization SEMs,
            # while the binned scaling_* errs are dex-scale numbers from the
            # assembler (scaling_T err ~0.08 vs value ~1e7 K). Only compute
            # the noise-floor metric where err is plausibly in value units;
            # otherwise flag it for confirmation against
            # bind.cli.emulator_assemble (Popeye) instead of guessing.
            ratio_med = float(np.median((sem / np.maximum(np.abs(truth), 1e-300))[ok]))
            if 1e-6 < ratio_med < 3.0:
                # wlemu acceptance frame: out-of-sample residual in units of
                # the realization SEM (the noise floor any emulator of these
                # maps is judged against)
                entry["err_rel_med"] = float(
                    np.median((np.abs(pred - truth) / sem)[ok]))
            else:
                entry["err_units_flag"] = (
                    f"median err/|value| = {ratio_med:.2e} — err not in value "
                    "units (likely dex); convention unresolved, confirm in "
                    "bind.cli.emulator_assemble")
    return entry


def _validate_target(f, table: TargetTable, X: np.ndarray, args, i: int) -> dict:
    R = len(X)
    rng = np.random.default_rng(args.seed + 7000 + i)
    if args.kfold > 0:
        vr = rng.permutation(np.flatnonzero(table.valid_runs))
        pred = np.full((R, int(table.valid_dims.sum())), np.nan)
        for j, fold in enumerate(np.array_split(vr, args.kfold)):
            mask = np.zeros(R, bool)
            mask[fold] = True
            d = _fit_target(X, table, table.train_rows & ~mask, args.n_pca,
                            args.iters, args.lr,
                            args.seed + 1000 + 100 * i + j, args.device)
            model = _SnapModel(X[table.train_rows & ~mask], d)
            pred[mask] = _predict_physical(X[mask], table, model)
        entry = _validation_entry(f, table, table.valid_runs,
                                  pred[table.valid_runs])
        entry["kfold"] = args.kfold
        return entry
    vr = np.flatnonzero(table.valid_runs)
    n_hold = min(args.holdout, max(4, len(vr) // 5))
    hold_mask = np.zeros(R, bool)
    hold_mask[rng.choice(vr, size=n_hold, replace=False)] = True
    d = _fit_target(X, table, table.train_rows & ~hold_mask, args.n_pca,
                    args.iters, args.lr, args.seed + 1000 + i, args.device)
    model = _SnapModel(X[table.train_rows & ~hold_mask], d)
    entry = _validation_entry(f, table, hold_mask,
                              _predict_physical(X[hold_mask], table, model))
    entry["holdout"] = int(n_hold)
    return entry


def fit_main(args) -> None:
    f, manifest = _load_dataset(Path(args.dataset))
    X = np.asarray(f["X_unit"], np.float64)
    R = len(X)
    names = (args.targets.split(",") if args.targets
             else list(manifest["targets"]))

    arrays = {
        "targets": np.array(names),
        "param_names": f["param_names"],
        "source_redshifts": f["source_redshifts"],
        "X_unit": X,
        "run_ids": f["run_ids"],
    }
    overrides = {}
    if getattr(args, "transform_override", None):
        for item in args.transform_override.split(","):
            t, tr = item.split("=")
            overrides[t] = tr

    validation = {}
    all_rows = np.ones(R, bool)
    for i, name in enumerate(names):
        table = TargetTable(f, name, manifest, all_rows,
                            transform_override=overrides.get(name))
        d = _fit_target(X, table, table.train_rows, args.n_pca, args.iters,
                        args.lr, args.seed + i, args.device)
        arrays[f"{name}__train_rows"] = table.train_rows
        arrays[f"{name}__feature_shape"] = np.array(table.feature_shape, int)
        arrays[f"{name}__transform"] = np.array(table.transform)
        for k, v in d.items():
            arrays[f"{name}__{k}"] = v
        print(f"[fit] {name}: D={table.Y.shape[1]} runs={table.train_rows.sum()} "
              f"pca={d['pca_components'].shape[0]}", flush=True)

        if args.holdout > 0 or args.kfold > 0:
            validation[name] = _validate_target(f, table, X, args, i)
            v = validation[name]
            print(f"[validate] {name}: frac_err_med = {v['frac_err_med']:.4f}"
                  + (f", err_rel_med = {v['err_rel_med']:.2f}x SEM"
                     if "err_rel_med" in v else ""), flush=True)

    arrays["provenance"] = np.array(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "settings": {"n_pca": args.n_pca, "iters": args.iters, "lr": args.lr,
                     "seed": args.seed, "holdout": args.holdout,
                     "kfold": args.kfold,
                     "transform_override": overrides or None},
        "recipe": "per-target PCA + batched ARD Matern-5/2 GP "
                  "(wlemu/gasemu construction)",
    }))
    out = Path(args.artifact_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out}")
    if validation:
        vpath = out.with_name(out.stem + "_validation.json")
        vpath.write_text(json.dumps(validation, indent=2))
        print(f"wrote {vpath}")


def merge_main(args) -> None:
    """Merge per-target part artifacts (disBatch fan-out) into one."""
    parts = sorted(Path(args.parts_dir).glob("*.npz"))
    if not parts:
        raise SystemExit(f"no part npz files in {args.parts_dir}")
    arrays, targets, validation, provs = {}, [], {}, []
    for p in parts:
        d = dict(np.load(p, allow_pickle=False))
        for t in [str(x) for x in d["targets"]]:
            targets.append(t)
            for k, v in d.items():
                if k.startswith(t + "__"):
                    arrays[k] = v
        for k in ("param_names", "source_redshifts", "X_unit", "run_ids"):
            arrays.setdefault(k, d[k])
        provs.append(json.loads(str(d["provenance"])))
        vp = p.with_name(p.stem + "_validation.json")
        if vp.exists():
            validation.update(json.loads(vp.read_text()))
    arrays["targets"] = np.array(targets)
    arrays["provenance"] = np.array(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "merged_from": [str(p) for p in parts],
        "part_settings": provs[0].get("settings"),
        "dataset": provs[0].get("dataset"),
        "recipe": provs[0].get("recipe"),
    }))
    out = Path(args.artifact_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"merged {len(targets)} targets from {len(parts)} parts -> {out}")
    if validation:
        vpath = out.with_name(out.stem + "_validation.json")
        vpath.write_text(json.dumps(validation, indent=2))
        print(f"wrote {vpath}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "merge"])
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--artifact-out", default=str(ARTIFACT))
    ap.add_argument("--parts-dir", default=str(WP6 / "statsemu_parts"))
    ap.add_argument("--targets", default=None,
                    help="comma list; default = every manifest target")
    ap.add_argument("--transform-override", default=None,
                    help="e.g. 'cl_kappa_y=log,cl_yy=log' — replaces the "
                         "manifest transform for those targets (recorded "
                         "in provenance)")
    ap.add_argument("--holdout", type=int, default=32)
    ap.add_argument("--kfold", type=int, default=0,
                    help="K-fold CV over valid runs (replaces --holdout)")
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.mode == "merge":
        merge_main(args)
    else:
        fit_main(args)


if __name__ == "__main__":
    main()
