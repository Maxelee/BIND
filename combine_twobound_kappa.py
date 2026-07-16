"""Pool the per-run twobound κ suite into a single export pair.

Reads every ``<runs_dir>/run_NNNN/{kappa_maps.npz,params.npy}`` and writes:
  - ``twobound_kappa.npz``  : kappa (n_run, n_real, n_zs, npix, npix) float32,
                              plus params, run_names, source_redshifts, fov_deg,
                              npix, n_real.
  - ``twobound_params.npy`` : params (n_run, 35) float64.

The κ array is large (~58 GB at 60 runs × 50 real × 5 z_s × 1024²), so it is
**streamed** run-by-run straight into the .npz member; peak memory stays at one
run (~1 GB), which is what lets this run inside a memory-capped session.

    python combine_twobound_kappa.py \
        --runs_dir /mnt/home/mlee1/ceph/bind_science/runs/twobound

Env-free; only numpy required.
"""
import argparse
import glob
import io
import os
import zipfile

import numpy as np
from numpy.lib import format as npformat


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True,
                    help="directory containing run_NNNN/ subdirs")
    ap.add_argument("--out_dir", default=None,
                    help="where to write the export (default: runs_dir)")
    ap.add_argument("--glob", default="run_*", help="run-dir glob (default run_*)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.runs_dir
    dirs = sorted(glob.glob(os.path.join(args.runs_dir, args.glob)))
    if not dirs:
        raise SystemExit(f"no run dirs matching {args.glob} in {args.runs_dir}")
    n = len(dirs)
    print(f"{n} runs in {args.runs_dir}", flush=True)

    # shape/metadata from the first run
    z0 = np.load(os.path.join(dirs[0], "kappa_maps.npz"))
    ksh = z0["kappa"].shape          # (n_real, n_zs, npix, npix)
    kdt = z0["kappa"].dtype          # float32
    src_z = z0["source_redshifts"]
    fov, npix, n_real = z0["fov_deg"], z0["npix"], z0["n_real"]
    del z0
    full_shape = (n, *ksh)
    print(f"full kappa shape {full_shape} {kdt}", flush=True)

    # params + names first (small, also catches missing files early)
    params = np.empty((n, 35), dtype=np.float64)
    names = []
    for i, d in enumerate(dirs):
        names.append(os.path.basename(d))
        params[i] = np.load(os.path.join(d, "params.npy"))
    names = np.array(names)

    out_p = os.path.join(out_dir, "twobound_params.npy")
    np.save(out_p, params)
    print(f"wrote {out_p}", flush=True)

    out_k = os.path.join(out_dir, "twobound_kappa.npz")
    with zipfile.ZipFile(out_k, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for name, arr in [("params", params), ("run_names", names),
                          ("source_redshifts", src_z),
                          ("fov_deg", np.asarray(fov)),
                          ("npix", np.asarray(npix)),
                          ("n_real", np.asarray(n_real))]:
            zf.writestr(name + ".npy", npy_bytes(arr))

        zi = zipfile.ZipInfo("kappa.npy")
        with zf.open(zi, mode="w", force_zip64=True) as f:
            npformat.write_array_header_1_0(f, {
                "descr": npformat.dtype_to_descr(kdt),
                "fortran_order": False,
                "shape": full_shape,
            })
            for i, d in enumerate(dirs):
                z = np.load(os.path.join(d, "kappa_maps.npz"))
                k = z["kappa"]
                assert k.shape == ksh and k.dtype == kdt, \
                    f"{names[i]}: {k.shape} {k.dtype} != {ksh} {kdt}"
                for r in range(k.shape[0]):           # stream per realization
                    f.write(np.ascontiguousarray(k[r]).tobytes())
                del z, k
                print(f"  [{i + 1}/{n}] {names[i]} written", flush=True)

    print(f"done → {out_k} ({os.path.getsize(out_k) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
