"""``bind-emulator-assemble``: pack a lightcone-run suite into one emulator dataset.

Walks every completed run under ``--runs_dir`` (default the SB35 Sobol suite),
collects all summary statistics + the parquet Y–M family + the DMO suppression
reference, and writes a single ``emulator_dataset.npz`` consumed by
``bind.emulator.Emulator.fit``.

    bind-emulator-assemble --out /ceph/bind_sb35/emulator_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bind.emulator.dataset import (
    DEFAULT_DMO,
    DEFAULT_PARQUET,
    DEFAULT_RUNS,
    assemble,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--out", type=Path, required=True, help="Output emulator_dataset.npz")
    p.add_argument("--dmo_dir", type=Path, default=DEFAULT_DMO,
                   help="DMO run for the suppression S(ell) reference (or 'none').")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                   help="integrated.parquet for the Y-M family (or 'none').")
    p.add_argument("--scaling_snap", type=int, default=96,
                   help="Snapshot used for the scaling relations (default z~0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dmo = None if str(args.dmo_dir).lower() == "none" else args.dmo_dir
    parquet = None if str(args.parquet).lower() == "none" else args.parquet
    ds = assemble(args.runs_dir, dmo_dir=dmo, parquet=parquet,
                  scaling_snap=args.scaling_snap, verbose=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(args.out)
    print("\n" + ds.summary())
    print(f"\n[assemble] wrote {args.out}")


if __name__ == "__main__":
    main()
