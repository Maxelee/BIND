"""``bind-emulate``: train the lightcone-statistics emulator, or predict with it.

Train a bundle from an assembled dataset::

    bind-emulate train --dataset emulator_dataset.npz --backend mlp \
        --out lightcone_emulator.pt

Predict every statistic for a parameter vector (defaults to the TNG fiducial)::

    bind-emulate predict --emulator lightcone_emulator.pt --z_s 1.0 \
        --params my_params.npy --out prediction.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _train(args) -> None:
    from bind.emulator import EmulatorDataset
    from bind.emulator.core import Emulator
    ds = EmulatorDataset.load(args.dataset)
    print(ds.summary())
    bkw = {}
    if args.epochs:
        bkw["epochs"] = args.epochs
    em = Emulator(backend=args.backend, n_components=args.n_components,
                  backend_kwargs=bkw or None, min_valid=args.min_valid)
    print(f"\n[train] backend={args.backend} n_components={args.n_components}")
    em.fit(ds, stats=args.stats, verbose=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    em.save(args.out)
    print(f"\n[train] {len(em.statistics)} statistics → {args.out}")


def _predict(args) -> None:
    from bind.emulator.core import Emulator
    em = Emulator.load(args.emulator)
    if args.list:
        print("[predict] available statistics (name: bin-axis):")
        for s in em.statistics:
            print(f"    {s:16s} {em.bin_axis.get(s) or '-'}")
        print("    derived: cl_kappa_auto, suppression (axis: ell)")
        return
    if args.params:
        params = np.load(args.params)
    else:
        from bind.params import fiducial_params
        params = fiducial_params()
        print("[predict] no --params; using TNG fiducial")
    grids = {}
    if args.ell:
        grids["ell"] = np.asarray(args.ell, float)
    if args.nu:
        grids["nu"] = np.asarray(args.nu, float)
    if args.mass:
        grids["log_mass_bins"] = np.asarray(args.mass, float)
    out = em.predict(params, z_s=args.z_s, stats=args.stats, grids=grids or None)
    flat = {}
    for k, v in out.items():
        if k == "axes":
            for ak, av in v.items():
                flat[f"axis__{ak}"] = av
        elif v is not None:
            flat[k] = np.asarray(v)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **{k: v for k, v in flat.items() if v is not None})
        print(f"[predict] wrote {args.out}")
    keys = [k for k in out if k not in ("axes", "source_redshifts", "z_s") and not k.endswith("_err")]
    print(f"[predict] z_s={args.z_s}; statistics: {keys}")
    for k in keys:
        print(f"  {k:16s} {np.asarray(out[k]).shape}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit an emulator from a dataset")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--backend", default="auto",
                   choices=["auto", "mlp", "gp", "gpgpu", "flow"],
                   help="auto = GPU exact-GP (gpgpu) if CUDA else sklearn gp")
    t.add_argument("--n_components", type=int, default=20)
    t.add_argument("--min_valid", type=int, default=20)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--stats", nargs="+", default=None, help="subset of statistics to fit")
    t.set_defaults(func=_train)

    q = sub.add_parser("predict", help="predict statistics for a parameter vector")
    q.add_argument("--emulator", type=Path, required=True)
    q.add_argument("--params", type=Path, default=None, help="35- or 30-vector .npy")
    q.add_argument("--z_s", type=float, default=None, help="source redshift (default: all planes)")
    q.add_argument("--stats", nargs="+", default=None, help="subset of statistics to output")
    q.add_argument("--ell", nargs="+", type=float, default=None,
                   help="custom multipoles for the Cl/suppression statistics")
    q.add_argument("--nu", nargs="+", type=float, default=None,
                   help="custom S/N thresholds for peaks/minima/MFs/R")
    q.add_argument("--mass", nargs="+", type=float, default=None,
                   help="custom log10(M) bins for the scaling relations")
    q.add_argument("--list", action="store_true", help="list available statistics and exit")
    q.add_argument("--out", type=Path, default=None)
    q.set_defaults(func=_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
