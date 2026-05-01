"""
Patch norm_stats.npz to correct the Stars channel target_mean.

offset.ipynb confirmed that ns.target_mean[2] was computed from a different
snapshot of the training set and is 0.147 dex higher than the current data
mean. The model is unbiased in latent space (Test 2), so shifting target_mean
by +log10(correction_factor) is the complete and sufficient fix.

Correction factor: 1.3853x (global median of m_truth/m_gen across all CV halos).
Run once, then re-run inference with --regenerate to refresh generated_halos.npz.
"""

import numpy as np
from pathlib import Path

NORM_PATH = Path("/mnt/home/mlee1/ceph/fm_runs/fm_base/norm_stats.npz")
CORRECTION_STARS = 1.3853  # truth/gen median from offset.ipynb


def main() -> None:
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"norm_stats.npz not found at {NORM_PATH}")

    d = dict(np.load(NORM_PATH))
    old_mean = d["target_mean"].copy()

    d["target_mean"] = old_mean.copy()
    shift = np.log10(CORRECTION_STARS)
    d["target_mean"][2] += shift

    print("norm_stats patch:")
    print(f"  Stars target_mean:  {old_mean[2]:.6f} → {d['target_mean'][2]:.6f}  (+{shift:.6f} dex)")
    print(f"  DM_hydro unchanged: {d['target_mean'][0]:.6f}")
    print(f"  Gas      unchanged: {d['target_mean'][1]:.6f}")

    np.savez(str(NORM_PATH), **d)
    print(f"\nSaved to {NORM_PATH}")
    print("Next step: re-run inference with --regenerate to refresh generated_halos.npz")


if __name__ == "__main__":
    main()
