"""``bind-wlemu``: predict WL convergence summary statistics from the CLI.

Examples::

    # fiducial parameters, z_s = 1.0, print the power spectrum
    bind-wlemu --z 1.0

    # override named physical parameters, save everything to an npz
    bind-wlemu --z 1.0 --set WindEnergyIn1e51erg=7.2 --set RadioFeedbackFactor=2.0 \
        --out pred.npz

    # batch: unit-cube parameter matrix from an .npy file (N, 30)
    bind-wlemu --z 0.5 --params-file theta.npy --out pred.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z", type=float, required=True,
                    help="source redshift (one of the emulated planes)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="physical parameter override (repeatable); "
                         "unspecified parameters stay at the fiducial")
    ap.add_argument("--params-file", default=None,
                    help=".npy with (30,) or (N, 30) unit-cube parameters "
                         "(overrides --set)")
    ap.add_argument("--artifact", default=None, help="emulator artifact path "
                    "(default: the packaged one)")
    ap.add_argument("--out", default=None, help="write predictions npz here")
    ap.add_argument("--list-params", action="store_true",
                    help="print the parameter table and exit")
    args = ap.parse_args(argv)

    from bind.wlemu import WLEmulator
    emu = WLEmulator.load(args.artifact)

    if args.list_params:
        print(emu.param_table().to_string())
        return

    if args.params_file:
        params = np.load(args.params_file)
    elif args.set:
        params = {}
        for kv in args.set:
            name, _, val = kv.partition("=")
            params[name] = float(val)
    else:
        params = emu.fiducial_params()

    pred = emu.predict(params, z_source=args.z)
    if args.out:
        np.savez(args.out, **{k: np.asarray(v) for k, v in pred.items()},
                 ell=emu.ell, pdf_x=emu.pdf_x, peak_x=emu.peak_x,
                 mink_thr=emu.mink_thr,
                 source_redshift=np.array(args.z))
        print(f"wrote {args.out} (blocks: {', '.join(emu.block_names)})")
    else:
        cl, cl_std = np.atleast_2d(pred["Cl"]), np.atleast_2d(pred["Cl_std"])
        print(f"# C(ell) at z_s={args.z} "
              f"({'fiducial' if not (args.set or args.params_file) else 'custom'} params)")
        print(f"# {'ell':>10s} {'C_ell':>12s} {'sigma':>12s}")
        for ell, c, s in zip(emu.ell, cl[0], cl_std[0]):
            print(f"{ell:12.1f} {c:12.4e} {s:12.4e}")
        print("# (use --out pred.npz for all statistic blocks)")


if __name__ == "__main__":
    main()
