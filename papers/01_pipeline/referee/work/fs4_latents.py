#!/usr/bin/env python3
"""fidswap step 3 -- the 8 shipped latents for the new fiducial.

Uses the shipped reducer verbatim (referee/work/r5_common.measure_lambda, which is
family_basis_all.py L408-457 copied line for line) on the rebuilt halo atlas.

Reports, per replica and for the trio mean:
    lam_new, lam_old, delta, delta / sigma_design, and the design percentile
sigma_design is the standard deviation of the latent over the Sobol design rows
that the family model was fit on (emulator_dataset_nu05.npz run_ids).

Self-checks printed first:
  * lam_old measured from halo_atlas/fid_snap096.npz must equal the shipped
    X_FID[CHOSEN] in figs_preview/agnostic_lambda_results_obs.npz
  * the atlas-cube fid_* block must give the same lam_old as the atlas file

    python referee/work/fs4_latents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import r5_common as R  # noqa: E402

FS = Path("/mnt/home/mlee1/ceph/referee_work/fidswap")
ATL = FS / "halo_atlas"
P1 = HERE.parents[1]
REPLICAS = (18, 49, 53)
CANON = 49


def lam_from_atlas(path: Path) -> np.ndarray:
    z = np.load(path)
    return R.measure_lambda({k: np.asarray(z[k], float) for k in R.FIELD_KEYS})


def main():
    SOB, FID, rows, dsn = R.load_cube()
    LAT = R.sobol_LAT(SOB)
    sd = LAT.std(0, ddof=1)
    lam_old_cube = R.measure_lambda(FID)
    lam_old_atlas = lam_from_atlas(Path(R.ATLAS) / "fid_snap096.npz")
    print(f"design rows: {len(LAT)}   (emulator_dataset_nu05 run_ids)")
    print(f"CHECK atlas-file vs atlas-cube lam_old: max|diff| = "
          f"{np.max(np.abs(lam_old_atlas - lam_old_cube)):.3e}")
    try:
        g = np.load(P1 / "figs_preview/agnostic_lambda_results_obs.npz", allow_pickle=True)
        xf = np.asarray(g["X_FID"], float)[np.asarray(g["CHOSEN"], int)] \
            if "CHOSEN" in g.files else np.asarray(g["lam_fid"], float)
        print(f"CHECK shipped X_FID[CHOSEN] vs measured lam_old: max|diff| = "
              f"{np.max(np.abs(xf - lam_old_cube)):.3e}")
    except Exception as e:                      # noqa: BLE001
        print("CHECK shipped X_FID: unavailable --", e)

    lam_new = {}
    for r in REPLICAS:
        p = ATL / f"fidtb{r:02d}_snap096.npz"
        if not p.exists():
            print(f"  (missing {p.name}; skipping)")
            continue
        lam_new[r] = lam_from_atlas(p)
    trio = np.stack([lam_new[r] for r in sorted(lam_new)])
    lam_mean = trio.mean(0)

    hdr = (f"{'latent':<20s} {'old':>11s} " +
           " ".join(f"{'tb%d' % r:>11s}" for r in sorted(lam_new)) +
           f" {'trio mean':>11s} {'sig_design':>11s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, nm in enumerate(R.LAT_NAMES):
        print(f"{nm:<20s} {lam_old_cube[i]:11.5f} " +
              " ".join(f"{lam_new[r][i]:11.5f}" for r in sorted(lam_new)) +
              f" {lam_mean[i]:11.5f} {sd[i]:11.5f}")

    print(f"\nshift / sigma_design (new - old):")
    print(f"{'latent':<20s} " + " ".join(f"{'tb%d' % r:>9s}" for r in sorted(lam_new))
          + f" {'trio mean':>10s} {'pct(design)':>12s} {'paint sd/sig':>13s}")
    for i, nm in enumerate(R.LAT_NAMES):
        pct = 100.0 * (LAT[:, i] < lam_new[CANON][i]).mean()
        print(f"{nm:<20s} " +
              " ".join(f"{(lam_new[r][i]-lam_old_cube[i])/sd[i]:+9.3f}" for r in sorted(lam_new))
              + f" {(lam_mean[i]-lam_old_cube[i])/sd[i]:+10.3f} {pct:11.1f}% "
              + f"{trio.std(0, ddof=1)[i]/sd[i]:12.4f}")

    out = FS / "lam_fidswap_v2.npz"
    np.savez(out, lat_names=np.array(R.LAT_NAMES), lam_old=lam_old_cube,
             lam_replicas=trio, replica_ids=np.array(sorted(lam_new)),
             lam_trio_mean=lam_mean, sigma_design=sd, LAT_sobol=LAT,
             run_ids=np.asarray(rows), canonical=CANON,
             lam_canonical=lam_new[CANON])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
