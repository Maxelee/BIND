"""``bind-paint-tauplane`` (stage 3, kSZ/FRB): composite gas -> lux tau-planes.

Writes ``tauplane{PP:02d}.dat`` for one snapshot into the lensplane directory,
using the **same plane numbering** as ``bind-paint-lensplane`` / ``bind-paint-yplane``
so the modified lux build (``compute_tau = true``) integrates them along the rays
with the identical per-realization randomization as the lensing/y planes.  This
yields kSZ/FRB electron-column (tau) maps that are pixel-consistent with the
ray-traced kappa and y maps (for the 1a kappa x tau and y x tau cross-spectra).

The per-slab tau is the electron column of the composited **gas** mass channel,
``tau = sigma_T x_e Sigma_gas / m_p`` with the per-plane physical-area factor
(scale factor ``a_l``) — the lux-RT counterpart of the Born ``tau`` already built
by :func:`bind.inference.lightcone_maps.assemble_lightcone`.  The plane is an
intensive sky-surface quantity (independent of the angular output pixel size), so
lux integrates it additively along the deflected ray with no kernel and no 1/a
factor (the ``a_l`` area factor is baked in here).  ``DM[pc/cm^3] = tau / TAU_PER_DM``.

    bind-paint-tauplane --generate_dir /ceph/.../snap_096 \\
        --output_dir /ceph/.../lensplanes --lc_snap_idx 0 --lc_n_snaps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.lensplane import write_tauplane
from bind.inference.lightcone_maps import _tau_per_gas_pixel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate_dir", type=Path, required=True,
                   help="Directory with composite_slab*.npz (stage 2)")
    p.add_argument("--stage1_dir", type=Path, default=None,
                   help="Stage-1 dir for the manifest (defaults to generate_dir/stage1)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Lensplane directory (tauplane*.dat written alongside lenspot*.dat)")
    p.add_argument("--lc_snap_idx", type=int, required=True)
    p.add_argument("--lc_n_snaps", type=int, required=True)
    p.add_argument("--planes_per_snapshot", type=int, default=None)
    p.add_argument("--lp_grid", type=int, default=4096)
    p.add_argument("--gas_channel", type=int, default=1,
                   help="Channel index of the gas mass map in 'composite' (default 1)")
    return p.parse_args()


def _tauslab(npz_path: Path, box_size: float, a_l: float, gas_channel: int) -> np.ndarray:
    """Electron-column tau of the composited gas channel (intensive surface field)."""
    d = np.load(npz_path)
    gas = d["composite"][gas_channel].astype(np.float64)   # gas mass [Msun/h] / pixel
    return _tau_per_gas_pixel(box_size, gas.shape[0], a_l) * gas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = args.stage1_dir or (args.generate_dir / "stage1")
    manifest = json.loads((stage1_dir / "stage1_manifest.json").read_text())
    n_slabs = int(manifest["n_slabs"])
    box_size = float(manifest["box_size"])
    a_l = float(manifest["scale_factor"])
    pps = args.planes_per_snapshot or n_slabs
    if pps != n_slabs:
        raise SystemExit(f"--planes_per_snapshot={pps} must equal n_slabs={n_slabs}")

    plane_offset = args.lc_snap_idx * pps + 1  # lux numbers planes from 1
    for si in range(n_slabs):
        plane_idx = plane_offset + si
        comp = args.generate_dir / f"composite_slab{si:02d}.npz"
        if not comp.exists():
            print(f"[tauplane] WARNING: {comp} missing — skipping slab {si}")
            continue
        tau = _tauslab(comp, box_size, a_l, args.gas_channel)   # (N, N)

        # center-crop to lp_grid (must match bind-paint-lensplane exactly)
        n = tau.shape[0]
        if n != args.lp_grid:
            if args.lp_grid > n:
                raise SystemExit(f"--lp_grid {args.lp_grid} > map size {n}")
            off = (n - args.lp_grid) // 2
            tau = tau[off:off + args.lp_grid, off:off + args.lp_grid]

        out = args.output_dir / f"tauplane{plane_idx:02d}.dat"
        write_tauplane(tau, out)
        print(f"[tauplane] plane {plane_idx:02d} (snap={args.lc_snap_idx} slab={si}) "
              f"tau_max={tau.max():.3e} -> {out.name}")

    print(f"[tauplane] snapshot {args.lc_snap_idx} done.")


if __name__ == "__main__":
    main()
