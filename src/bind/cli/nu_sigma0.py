"""``python -m bind.cli.nu_sigma0``: the FIDUCIAL kappa_rms that defines nu.

Writes one small npz holding ``sigma0`` — the fiducial run's convergence rms per
source plane, unsmoothed and at each smoothing scale — so that every run's PDF,
peak, minimum and Minkowski statistics are expressed on ONE common S/N axis
``nu = kappa / kappa_rms(fiducial)``.  Normalising each map by its own rms would
absorb the few-percent sigma_kappa response that the feedback variations are
supposed to show.

Realizations are streamed one at a time out of the npz zip member: a
(1000, 5, 1024, 1024) cube is 21 GB decompressed and must never be materialised.

    python -m bind.cli.nu_sigma0 --run_dir <fiducial run> --output nu_sigma0.npz
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

from bind.inference.stats import _gaussian_smooth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="FIDUCIAL run dir holding kappa_maps.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoothing_arcmin", type=float, nargs="+",
                   default=[1.0, 2.0, 5.0, 8.0],
                   help="scales to tabulate (must cover every scale the stats use: "
                        "peaks default 2.0, Minkowski use the first nongaussian scale)")
    p.add_argument("--n_real", type=int, default=50,
                   help="realizations averaged (sigma is stable; 50 is plenty)")
    p.add_argument("--fov_deg", type=float, default=None,
                   help="default: read from the npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.run_dir / "kappa_maps.npz"
    with np.load(src) as f:                      # tiny scalars only
        fov = float(args.fov_deg if args.fov_deg is not None else f["fov_deg"])

    scales = np.asarray(args.smoothing_arcmin, float)
    with zipfile.ZipFile(src) as z, z.open("kappa.npy") as f:
        v = fmt.read_magic(f)
        shape, _, dt = (fmt.read_array_header_1_0(f) if v == (1, 0)
                        else fmt.read_array_header_2_0(f))
        n_real = min(args.n_real, shape[0])
        n_src = shape[1]
        per = int(np.prod(shape[1:])) * dt.itemsize
        acc_u = np.zeros(n_src)
        acc_s = np.zeros((len(scales), n_src))
        for r in range(n_real):
            cube = np.frombuffer(f.read(per), dtype=dt).reshape(shape[1:])
            for i in range(n_src):
                m = cube[i].astype(np.float64)
                acc_u[i] += m.std()
                for k, sc in enumerate(scales):
                    acc_s[k, i] += _gaussian_smooth(m, sc, fov).std()
            if (r + 1) % 10 == 0:
                print(f"[nu_sigma0] {r + 1}/{n_real} reals", flush=True)
    sigma_unsmoothed = acc_u / n_real
    sigma_smoothed = acc_s / n_real

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, sigma_unsmoothed=sigma_unsmoothed,
             sigma_smoothed=sigma_smoothed, scales_arcmin=scales,
             fov_deg=fov, n_real=n_real, source_run=str(args.run_dir),
             n_real_available=shape[0])
    print(f"[nu_sigma0] fiducial = {args.run_dir}  ({n_real} of {shape[0]} reals)")
    print("[nu_sigma0] sigma_unsmoothed per z_s: "
          + "  ".join(f"{s:.5e}" for s in sigma_unsmoothed))
    for k, sc in enumerate(scales):
        print(f"[nu_sigma0] sigma @ {sc:g}' : "
              + "  ".join(f"{s:.5e}" for s in sigma_smoothed[k]))
    print(f"[nu_sigma0] -> {args.output}")


if __name__ == "__main__":
    main()
