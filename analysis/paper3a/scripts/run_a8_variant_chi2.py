"""Does the EXCLUSION survive each WP-A8 systematics variant?

The systematics grid reports where the posterior MEDIAN moves. That is
not the question the paper turns on. The paper's claim is that no
corner of the feedback space FITS -- so for each variant we need the
per-block chi2 at that variant's own MAP, under that variant's own
covariance, and its posterior-predictive p.

This matters because the fresh chains showed `emul2x` moving the
headline coordinate by +9.2 sigma -- which the importance-reweighted
pass had also reported (+9.04) and which was wrongly written off as an
IS artifact. It is real: doubling the emulator-error tier in all three
blocks relaxes the inter-probe tension and lets the fit sit much closer
to the fiducial. Whether the model then becomes ACCEPTABLE is the
question here.

Run: python analysis/paper3a/scripts/run_a8_variant_chi2.py
Out: wp8_robustness/a8_variant_chi2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402
from analysis.paper3a.scripts.run_joint_ab_fit import VARIANTS, _blocks  # noqa: E402

WP8 = Path("/mnt/ceph/users/mlee1/paper3/A/wp8_robustness")
NDIM = 32


def load(tag):
    """Flat chain + logp for a variant ('' = fiducial). None if absent."""
    suf = "" if not tag else f"_{tag}"
    fs = [CHAINS / f"joint_ab{suf}_seed{k}.npz" for k in range(4)]
    if not all(f.exists() for f in fs):
        return None, None
    ch, lp = [], []
    for f in fs:
        d = np.load(f)
        ch.append(d["chain"].astype(float).reshape(-1, NDIM))
        lp.append(d["logp"].astype(float).reshape(-1))
    return np.concatenate(ch), np.concatenate(lp)


def main() -> None:
    out = {}
    for tag in ("",) + VARIANTS:
        flat, logp = load(tag)
        name = tag or "fiducial"
        if flat is None:
            out[name] = {"status": "chains not present"}
            print(f"{name:14s} chains not present")
            continue

        # each variant is scored under ITS OWN likelihood
        fgas, ksz, jb = _blocks(tag or None)
        u_map = flat[int(np.argmax(logp))]

        # A drop-one variant carries a zeroed stub in place of a block.
        # Its chi2 is still worth having -- it is the OUT-OF-SAMPLE score
        # of the probe that was withheld, at the MAP the other probes
        # chose -- but it must not enter the fitted total or its dof.
        chi2_b, _, _ = jb.chi2(u_map)
        blocks = {
            "fgas": {"chi2": float((getattr(fgas, "real", fgas))
                                   .chi2(u_map)[0]),
                     "n": int(len(fgas.data.values)),
                     "fitted": not getattr(fgas, "dropped", False)},
            "ksz": {"chi2": float((getattr(ksz, "real", ksz))
                                  .chi2(u_map)[0]),
                    "n": int(len((getattr(ksz, "real", ksz)).values)),
                    "fitted": not getattr(ksz, "dropped", False)},
            "b_kappa_y_peaks": {"chi2": float(chi2_b), "n": 4,
                                "fitted": True},
        }
        fit = {k: v for k, v in blocks.items() if v["fitted"]}
        held = {k: v for k, v in blocks.items() if not v["fitted"]}
        tot = sum(b["chi2"] for b in fit.values())
        dof = sum(b["n"] for b in fit.values())
        out[name] = {
            "status": "ok",
            "n_samples": int(len(flat)),
            "map_coords_mirror": jb.coords(u_map[None, :30])[0][0].tolist(),
            "chi2_per_block": blocks,
            "chi2_total": tot, "n_total": dof,
            "totals_cover": sorted(fit),
            "held_out": {k: {"chi2": v["chi2"], "n": v["n"]}
                         for k, v in held.items()},
        }
        print(f"{name:14s} total {tot:8.1f}/{dof:<3d}  " +
              "  ".join(f"{k} {v['chi2']:7.1f}/{v['n']}"
                        + ("" if v["fitted"] else " [held out]")
                        for k, v in blocks.items()))

    (WP8 / "a8_variant_chi2.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {WP8}/a8_variant_chi2.json")


if __name__ == "__main__":
    main()
