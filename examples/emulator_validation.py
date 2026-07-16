"""Validate the lightcone-statistics emulator: k-fold CV + twobound OOD stress test.

Two tests, both reduced to "is the emulator error below the cosmic-variance noise
floor?" — the only bar that matters for a forward model used in inference:

1. **k-fold CV** on the SB35 Sobol set (a space-filling design): per-statistic
   median fractional error, R^2, and chi^2/dof against the per-run realization
   error (``Target.err``, the 50-realization scatter).

2. **Out-of-design** on the 60 ``twobound`` prior-corner runs — the Stage-B
   stress test of ``docs/wl_tsz_plan.md`` §5: train on the interior Sobol cloud,
   predict the extreme corners, where any over-fitting shows up first.

Writes a metrics table (printed + ``emulator_validation.npz``) and figures into
``examples/figures_lightcone/``.

    python examples/emulator_validation.py --backend mlp --kfolds 5
    python examples/emulator_validation.py --backend gp --quick
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from bind.emulator import EmulatorDataset
from bind.emulator.core import Emulator
from bind.emulator.dataset import DEFAULT_DMO, assemble

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "figures_lightcone"
DEFAULT_DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")


def _frac_err(pred: np.ndarray, truth: np.ndarray, floor_frac: float = 1e-3) -> np.ndarray:
    scale = np.maximum(np.abs(truth), floor_frac * np.nanmax(np.abs(truth)))
    return np.abs(pred - truth) / scale


def _stat_metrics(pred, truth) -> dict:
    """Response-AWARE metrics over a stack of test predictions ``(n, *shape)``.

    The key numbers isolate the *parameter response* (how each run deviates from
    the ensemble mean) — pooled per-element error is dominated by the bin-shape
    every run shares and hides whether the emulator tracks feedback at all.

    * ``frac_err_med`` — pooled median fractional error (absolute accuracy).
    * ``response_r2``  — per-bin across-run R^2 (predicting each run's deviation
      from the mean), averaged over bins.  This is the metric that matters.
    * ``amp_ratio``    — std(pred)/std(truth) across runs, per bin, averaged: how
      much of the response *amplitude* is recovered (1.0 = perfect).
    """
    p = np.asarray(pred, float).reshape(len(pred), -1)
    t = np.asarray(truth, float).reshape(len(truth), -1)
    m = np.isfinite(p) & np.isfinite(t)
    fe = _frac_err(p[m], t[m])
    out = {"frac_err_med": float(np.median(fe)),
           "frac_err_p84": float(np.percentile(fe, 84))}
    # per-bin across-run statistics
    tvar = np.nanvar(t, axis=0)
    good = tvar > 1e-12 * np.nanmax(tvar + 1e-30)        # bins with real run-to-run signal
    if good.any():
        res = np.nansum((p[:, good] - t[:, good]) ** 2, axis=0)
        tot = np.nansum((t[:, good] - np.nanmean(t[:, good], 0)) ** 2, axis=0) + 1e-30
        out["response_r2"] = float(np.median(1 - res / tot))
        out["amp_ratio"] = float(np.median(np.nanstd(p[:, good], 0)
                                           / (np.nanstd(t[:, good], 0) + 1e-30)))
    return out


def kfold_cv(ds: EmulatorDataset, *, backend: str, kfolds: int, n_components: int,
             backend_kwargs: dict | None, stats=None, seed: int = 0) -> dict:
    """Out-of-fold predictions for every run, then response-aware metrics per stat."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(ds.n_runs)
    folds = np.array_split(order, kfolds)
    oof: dict[str, np.ndarray] = {}                      # name -> (N, *shape) OOF preds
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(order, test_idx)
        em = Emulator(backend=backend, n_components=n_components,
                      backend_kwargs=backend_kwargs).fit(ds.subset(train_idx), stats=stats,
                                                         verbose=False)
        pred = em.predict(ds.X_native[test_idx])
        for name in em.statistics:
            val = ds.targets[name].value
            if name not in oof:
                oof[name] = np.full_like(val, np.nan, dtype=float)
            oof[name][test_idx] = pred[name]
        print(f"  fold {f + 1}/{kfolds} done")
    out = {}
    for name, pf in oof.items():
        t = ds.targets[name]
        mask = t.valid_mask() & np.isfinite(pf.reshape(len(pf), -1)).all(1)
        if mask.sum() >= 3:
            out[name] = _stat_metrics(pf[mask], t.value[mask])
    return out


def ood_twobound(ds: EmulatorDataset, *, backend: str, n_components: int,
                 backend_kwargs: dict | None, stats=None) -> dict:
    if not TWOBOUND.exists():
        print("  [ood] no twobound suite — skipping")
        return {}
    tb = assemble(TWOBOUND, dmo_dir=DEFAULT_DMO, parquet=None, verbose=False)
    em = Emulator(backend=backend, n_components=n_components,
                  backend_kwargs=backend_kwargs).fit(ds, stats=stats, verbose=False)
    res = {}
    for name in em.statistics:
        if name not in tb.targets:
            continue
        t = tb.targets[name]
        mask = t.valid_mask()
        if not mask.any():
            continue
        pred = em.predict(tb.X_native[mask])
        if mask.sum() >= 3:
            res[name] = _stat_metrics(pred[name], t.value[mask])
    return res


def _print_table(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'statistic':16s} {'fracErr_med':>11s} {'response_R2':>11s} {'amp_ratio':>9s}")
    for name, m in metrics.items():
        print(f"{name:16s} {m.get('frac_err_med', np.nan):11.4f} "
              f"{m.get('response_r2', np.nan):11.3f} {m.get('amp_ratio', np.nan):9.2f}")


def _figures(ds, cv, ood, backend) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)

    # (1) per-statistic RESPONSE-R2 bar chart: CV vs OOD (the meaningful metric)
    names = list(cv.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, [cv[n].get("response_r2", np.nan) for n in names], 0.38, label="Sobol CV")
    if ood:
        ax.bar(x + 0.2, [ood.get(n, {}).get("response_r2", np.nan) for n in names],
               0.38, label="twobound OOD", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("response $R^2$ (across-run, per bin)")
    ax.axhline(0.9, ls=":", c="k", lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Emulator feedback-response fidelity ({backend} backend)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"emulator_accuracy_{backend}.png", dpi=130)
    plt.close(fig)

    # (2) overlay: emulated vs truth suppression S(ell) + peaks at z_s=1 for a few runs
    em = Emulator(backend=backend, n_components=12).fit(ds, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ell = ds.targets["suppression"].axes.get("ell")
    nu = ds.targets["peak_counts"].axes.get("nu")
    for r in range(0, ds.n_runs, max(1, ds.n_runs // 7)):
        o = em.predict(ds.X_native[r], z_s=1.0)
        c = plt.cm.viridis(r / ds.n_runs)
        axes[0].semilogx(ell, o["suppression"], color=c, lw=1.2)
        axes[0].semilogx(ell, ds.targets["suppression"].value[r, 1], ":", color=c, lw=1)
        axes[1].plot(nu, o["peak_counts"], color=c, lw=1.2)
        axes[1].plot(nu, ds.targets["peak_counts"].value[r, 1], ":", color=c, lw=1)
    axes[0].axhline(1, ls="-", c="k", lw=0.5)
    axes[0].set(xlabel=r"$\ell$", ylabel=r"$S(\ell)=C/C_{\rm DMO}$ ($z_s=1$)",
                title="solid=emulator, dotted=truth")
    axes[1].set(xlabel=r"$\nu$", ylabel=r"$N_{\rm peak}$ ($z_s=1$)", yscale="log")
    fig.tight_layout()
    fig.savefig(OUT / f"emulator_overlay_{backend}.png", dpi=130)
    plt.close(fig)
    print(f"\n[figures] wrote emulator_accuracy_{backend}.png + emulator_overlay_{backend}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DS)
    p.add_argument("--backend", default="auto", choices=["auto", "mlp", "gp", "gpgpu", "flow"])
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--n_components", type=int, default=12)
    p.add_argument("--quick", action="store_true", help="fast backend config for a smoke run")
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()

    ds = (EmulatorDataset.load(args.dataset) if args.dataset.exists()
          else assemble(verbose=True))
    print(ds.summary())
    bkw = None
    if args.quick and args.backend == "mlp":
        bkw = {"epochs": 300, "n_models": 3}
    elif args.quick and args.backend == "gp":
        bkw = {"n_restarts": 0}
    elif args.quick and args.backend in ("gpgpu", "auto"):
        bkw = {"epochs": 150}
    elif args.quick and args.backend == "flow":
        bkw = {"epochs": 400}

    print(f"\n[cv] {args.kfolds}-fold on {ds.n_runs} runs ({args.backend})")
    cv = kfold_cv(ds, backend=args.backend, kfolds=args.kfolds,
                  n_components=args.n_components, backend_kwargs=bkw)
    _print_table("k-fold CV (Sobol)", cv)

    print("\n[ood] twobound corners")
    ood = ood_twobound(ds, backend=args.backend, n_components=args.n_components,
                       backend_kwargs=bkw)
    if ood:
        _print_table("OOD (twobound corners)", ood)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"emulator_validation_{args.backend}.npz",
             cv=np.array(cv, dtype=object), ood=np.array(ood, dtype=object))
    if not args.no_figures:
        _figures(ds, cv, ood, args.backend)


if __name__ == "__main__":
    main()
