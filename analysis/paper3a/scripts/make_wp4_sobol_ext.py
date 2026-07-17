"""Extend the SB35 Sobol design: points 256-511 of the SAME sequence.

The A4 convergence test showed the group-bin f_gas emulation error is
design-limited (~1/n_train); Max approved a second 256-run paint at the six
gate snapshots. The original design records ``sobol_seed: 0`` and this
script reproduces its first 256 points to 2e-15 before writing the
continuation, so the union is a genuine 512-point scrambled-Sobol design
(not two overlapping 256-designs).

Writes /mnt/ceph/users/mlee1/bind_sb35_ext/runs/run_{0256..0511}/params.npy
(35-dim physical vectors, TNG300 cosmology fixed) + design_ext.json.
Conditions and flow weights are shared from the original bundle — only run
dirs live here. Paint with run_wp4_paint_ext.sh (GPU, ⛔ Max).

Usage (CPU, seconds):
    python analysis/paper3a/scripts/make_wp4_sobol_ext.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402

BUNDLE = Path("/mnt/ceph/users/mlee1/bind_sb35")
EXT = Path("/mnt/ceph/users/mlee1/bind_sb35_ext")


def main() -> None:
    orig = np.load(BUNDLE / "design" / "astro_params_sobol.npy")
    design = json.loads((BUNDLE / "design" / "design.json").read_text())
    assert design["sobol_seed"] == 0 and design["n_sobol"] == 256

    eng = qmc.Sobol(d=30, scramble=True, seed=design["sobol_seed"])
    u512 = eng.random(512)
    phys_check = pm.astro_unit_to_physical(u512[:256])
    rel = np.abs(phys_check - orig[:, pm.ASTRO_IDX]) / np.maximum(np.abs(orig[:, pm.ASTRO_IDX]), 1e-12)
    if rel.max() > 1e-10:
        raise SystemExit(f"first-256 reproduction failed (max rel {rel.max():.2e}) — do not paint")
    print(f"first-256 reproduction OK (max rel {rel.max():.2e}); writing 256-511")

    cosmo_row = orig[0].copy()                       # fixed cosmology template
    full = np.tile(cosmo_row, (256, 1))
    full[:, pm.ASTRO_IDX] = pm.astro_unit_to_physical(u512[256:])

    for j in range(256):
        run_dir = EXT / "runs" / f"run_{256 + j:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "params.npy", full[j])
    (EXT / "design_ext.json").write_text(json.dumps({
        "created": datetime.date.today().isoformat(),
        "parent_bundle": str(BUNDLE), "sobol_seed": design["sobol_seed"],
        "points": "sequence indices 256-511 (verified continuation)",
        "gate_snaps_only": ["096", "071", "067", "063", "056", "049"],
        "purpose": "WP-A4 design densification: halve group-bin f_gas emulation error",
    }, indent=2))
    print(f"wrote 256 run dirs under {EXT}/runs + design_ext.json")


if __name__ == "__main__":
    main()
