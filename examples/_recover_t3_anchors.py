#!/usr/bin/env python
"""Recovery for `LC/T3_theta200_anchors.npz` (docs/REPRODUCING.md gap #1).

No producing script was ever committed for this npz -- it was originally
computed inline during T3 (`lightcone_m2r_ycap_real.py`) via
`act_ycap_measure.theta200_arcmin()` at the mock/data anchor mass+redshift,
then cached here. But `lightcone_m2r_ycap_real.py` (lines ~101-103, ~362)
only ever READS `t200_mock_arcmin`/`t200_data_arcmin` back in and passes
both values straight through, byte-for-byte, into `act_ycap_lrg_real.npz`
under the renamed keys `theta200_mock_arcmin`/`theta200_data_arcmin` --
verified equal to 16 significant figures. So this npz's content already
survives, verbatim, inside a downstream product; this script just reverses
the rename to reconstruct the original cache if it is ever lost.
Run: `python examples/_recover_t3_anchors.py [--out PATH]`.
"""
import argparse
import os
from pathlib import Path

import numpy as np

KS = Path(os.environ.get("BIND_KSZ_PRODUCTS", "/mnt/home/mlee1/ceph/bind_science/ksz_confront"))
LC = KS / "lightcone"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, default=LC / "T3_theta200_anchors.npz")
args = ap.parse_args()

real = np.load(LC / "act_ycap_lrg_real.npz")
np.savez(args.out, t200_mock_arcmin=real["theta200_mock_arcmin"],
         t200_data_arcmin=real["theta200_data_arcmin"])
print(f"wrote {args.out}: t200_mock_arcmin={float(real['theta200_mock_arcmin']):.10f}, "
      f"t200_data_arcmin={float(real['theta200_data_arcmin']):.10f}")
