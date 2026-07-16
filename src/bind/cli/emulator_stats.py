"""``bind-emulator-stats``: backfill the emulator-only field statistics (WST + DM).

Adds the two statistics not produced by the original stats stage —  the wavelet
scattering transform (``wst.npz``, from ``kappa_maps.npz``) and the
dispersion-measure field stats (``dm_stats.npz``, from ``tau_maps.npz``) — onto a
run directory whose maps are already on disk.  Skip-if-exists by default, so it is
safe to run as an idempotent SLURM array over a suite that is still generating
(WST wants a GPU; see ``run_emulator_stats.sh``).

    bind-emulator-stats --run_dir /ceph/bind_sb35/runs/run_0000
    bind-emulator-stats --run_dir ... --wst_n_real 10 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bind.inference import stats as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Run dir holding kappa_maps.npz / tau_maps.npz; outputs go here.")
    p.add_argument("--no_wst", action="store_true", help="Skip the scattering transform.")
    p.add_argument("--no_dm", action="store_true", help="Skip the DM field stats.")
    p.add_argument("--wst_J", type=int, default=4)
    p.add_argument("--wst_L", type=int, default=4)
    p.add_argument("--wst_n_real", type=int, default=10,
                   help="Realizations used for WST (heaviest stat; default 10).")
    p.add_argument("--device", default=None, help="torch device for WST (default: auto).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if wst.npz / dm_stats.npz already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rd = args.run_dir

    wst_path = rd / "wst.npz"
    if not args.no_wst and (args.overwrite or not wst_path.exists()):
        kpath = rd / "kappa_maps.npz"
        if not kpath.exists():
            print(f"[emu-stats] no kappa_maps.npz in {rd} — skipping WST")
        else:
            kappa = np.load(kpath)["kappa"]
            w = S.wst(kappa, J=args.wst_J, L=args.wst_L,
                      n_real=args.wst_n_real, device=args.device,
                      return_realizations=True)
            np.savez(wst_path, **w)
            print(f"[emu-stats] wst.npz  wst{w['wst'].shape} (J={args.wst_J},"
                  f"L={args.wst_L},n_real={w['n_real_used']})")
    elif not args.no_wst:
        print(f"[emu-stats] wst.npz exists — skipping ({wst_path})")

    dm_path = rd / "dm_stats.npz"
    if not args.no_dm and (args.overwrite or not dm_path.exists()):
        tpath = rd / "tau_maps.npz"
        if not tpath.exists():
            print(f"[emu-stats] no tau_maps.npz in {rd} — skipping DM")
        else:
            tau = np.load(tpath)["tau"]
            fov = float(np.load(tpath)["fov_deg"]) if "fov_deg" in np.load(tpath).files else 5.0
            dm = S.dm_stats(tau, fov_deg=fov, return_realizations=True)
            np.savez(dm_path, **dm)
            print(f"[emu-stats] dm_stats.npz  pdf{dm['dm_pdf'].shape} F{dm['F'].shape}")
    elif not args.no_dm:
        print(f"[emu-stats] dm_stats.npz exists — skipping ({dm_path})")


if __name__ == "__main__":
    main()
