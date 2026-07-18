"""WP-B5 interpretation-ladder item 4: transfer-SHAPE movement numbers.

Post-processing for the two bind-only variant grids run with the
denominator-swap transfer shapes (transfer_robustness/run_0000 and
run_0004, jobs 2451614/2451615): compare each variant anchor's y_mean
against the FROZEN tf-grid fiducial anchor, per fit nu bin x CAP radius,
normalized by the anchors' grid-MC errors. If the movement at the 4'
fiducial radius is small compared to the ~3-6x deficit (and to the MC
tier), the transfer-shape lever closes as NULL in the movement table.

Job-free; reads the three unit npz files only. Safe to run any time — it
reports which variant outputs exist and skips missing ones.

Outputs -> /mnt/home/mlee1/ceph/paper3/B/wp5_inference/transfer_movement.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B_ROOT = Path("/mnt/home/mlee1/ceph/paper3/B")
FROZEN_UNIT = B_ROOT / "wp4_mocks" / "grid_tfwiener" / "bind_run_0000.npz"
VARIANTS = {
    "T_denom_run_0000": B_ROOT / "wp4_mocks" / "transfer_robustness"
        / "run_0000" / "grid_tfwiener" / "bind_run_0000.npz",
    "T_denom_run_0004": B_ROOT / "wp4_mocks" / "transfer_robustness"
        / "run_0004" / "grid_tfwiener" / "bind_run_0000.npz",
}
FIT_BINS = [1, 2, 3, 4]     # nu [1,2),[2,3),[3,4),[4,12); 0-1 diagnostic-only
RAD_IDX_4AM = 2


def main() -> None:
    f0 = np.load(FROZEN_UNIT)
    y0, e0 = f0["y_mean"], f0["y_mc_err"]
    out = {"frozen_anchor": str(FROZEN_UNIT),
           "cap_radii_arcmin": f0["cap_radii_arcmin"].tolist(),
           "nu_edges": f0["nu_edges"].tolist(), "variants": {}}
    print(f"frozen anchor y_mean(4') fit bins: "
          + " ".join(f"{y0[b, RAD_IDX_4AM]:.3e}" for b in FIT_BINS))
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"[{name}] not on disk yet ({path}) — skipping")
            out["variants"][name] = {"status": "missing"}
            continue
        fv = np.load(path)
        yv, ev = fv["y_mean"], fv["y_mc_err"]
        ratio = yv / y0
        pull = (yv - y0) / np.sqrt(ev ** 2 + e0 ** 2)
        out["variants"][name] = {
            "status": "ok", "path": str(path),
            "ratio_fitbins_all_radii": ratio[FIT_BINS].tolist(),
            "pull_fitbins_all_radii": pull[FIT_BINS].tolist(),
            "ratio_4am": ratio[FIT_BINS, RAD_IDX_4AM].tolist(),
            "pull_4am": pull[FIT_BINS, RAD_IDX_4AM].tolist(),
            "max_abs_pull_fitbins": float(np.max(np.abs(pull[FIT_BINS]))),
        }
        print(f"[{name}] ratio(4') = "
              + " ".join(f"{r:.3f}" for r in ratio[FIT_BINS, RAD_IDX_4AM])
              + "   pull(4') = "
              + " ".join(f"{p:+.2f}" for p in pull[FIT_BINS, RAD_IDX_4AM])
              + f"   max|pull| all radii = "
                f"{out['variants'][name]['max_abs_pull_fitbins']:.2f}")
    dst = B_ROOT / "wp5_inference" / "transfer_movement.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
