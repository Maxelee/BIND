"""Driver for the A3 gate compute: one Slurm-array task = one design point.

Maps a global index onto the three painted bundles on rusty and evaluates
the WP-A2 operators at the gate snapshots (resumable — existing outputs are
skipped):

    index 0-255   -> /mnt/ceph/users/mlee1/bind_sb35/runs/run_XXXX          (Sobol)
    index 256-315 -> /mnt/home/mlee1/ceph/bind_portable_twobound/runs/run_XXXX (1P extremes)
    index 316     -> /mnt/ceph/users/mlee1/paper3/A/wp3_gate/1p_runs/run_0000  (fiducial, once painted)

Usage:  python analysis/paper3a/scripts/run_gate_operators.py INDEX
Output: /mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables/<bundle>_<run>_<snap>.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.gate import GateConfig, process_run_snapshot  # noqa: E402

SB35 = Path("/mnt/ceph/users/mlee1/bind_sb35")
TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_portable_twobound")
FIDUCIAL = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/1p_runs")
OUT = Path("/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables_v3")

# Gate snapshots: z~0 anchor (eROSITA regime) + the LRG range the kSZ bins
# actually occupy. Snapshot redshifts come from the conditions manifests.
GATE_SNAPS = ("096", "071", "067", "063", "056", "049")


def resolve(index: int) -> tuple[str, Path]:
    if 0 <= index < 256:
        return "sb35", SB35 / "runs" / f"run_{index:04d}"
    if 256 <= index < 316:
        return "twobound", TWOBOUND / "runs" / f"run_{index - 256:04d}"
    if index == 316:
        return "fiducial", FIDUCIAL / "run_0000"
    raise SystemExit(f"index {index} out of range [0, 316]")


def snap_redshift(snap: str) -> float:
    manifest = SB35 / "conditions" / f"snap_{snap}" / "stage1_manifest.json"
    return float(json.loads(manifest.read_text())["redshift"])


def main() -> None:
    index = int(sys.argv[1])
    bundle, run_dir = resolve(index)
    if not (run_dir / "params.npy").exists():
        if bundle == "fiducial":
            # expected until Max's fiducial paint lands — exit clean so the
            # array job doesn't report a failure for a known-pending slot
            print(f"{run_dir} not painted yet — skipping fiducial slot")
            return
        raise SystemExit(f"{run_dir} has no params.npy")

    # CAP radii from the frozen Qu et al. 2026 vector; falls back to the
    # frozen grid values if the wp1 data mount is unavailable.
    try:
        from analysis.paper3a.data_vectors.data_vectors import load_ksz_qu2026_lrg_fiducial

        radii = load_ksz_qu2026_lrg_fiducial().bins
    except (FileNotFoundError, OSError):
        radii = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])

    config = GateConfig(radii_arcmin=radii)
    for snap in GATE_SNAPS:
        out_path = OUT / f"{bundle}_{run_dir.name}_snap{snap}.npz"
        if out_path.exists():
            print(f"  {out_path.name}: exists, skipping")
            continue
        if not (run_dir / f"snap_{snap}").exists():
            print(f"  {run_dir.name}/snap_{snap}: missing, skipping")
            continue
        z = snap_redshift(snap)
        print(f"  {bundle}/{run_dir.name} snap {snap} (z={z:.3f}) ...")
        process_run_snapshot(run_dir, snap, z, out_path, config)
    print(f"done: index {index} ({bundle}/{run_dir.name})")


if __name__ == "__main__":
    main()
