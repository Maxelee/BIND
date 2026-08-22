"""Refresh the MF entries (v0/v1/v2) of figs_preview/amplitude_sets.npz to
the 22-bin canonical grid, after the 2026-08-12 twobound MF remeasurement
(remeasure_twobound_nu05.py) and the family_basis_all.py rebuild.

Replaces, for st in (v0, v1, v2):
  {st}__basis / {st}__mean / {st}__a_sobol / {st}__family_names / {st}__grid
      <- from the rebuilt figs_preview/family_model_bundle.npz
  {st}__fid_measured  <- the fiducial BIND paint's canonical-grid MFs
      (runs/bind/run_0000/nu05_stats.npz, ZI=1)
  {st}__a_fid / {st}__a_fid_lstsq  <- OLS (pinv) fit of the fiducial
      deviation on the new basis (one estimator everywhere)
  {st}__a_twobound / {st}__a_twobound_lstsq / {st}__tb_resid
      <- projections of the 30 remeasured twobound responses
All other keys pass through unchanged. The prior file is preserved at
amplitude_sets_mf14.npz.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python refresh_amplitude_sets_mf22.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
ZI = 1

AMP_PATH = Path("figs_preview/amplitude_sets.npz")
BAK_PATH = Path("figs_preview/amplitude_sets_mf14.npz")
BUNDLE = np.load("figs_preview/family_model_bundle.npz", allow_pickle=True)
FID = np.load(CEPH / "bind_science/runs/bind/run_0000/nu05_stats.npz")

# twobound pairs (mirror of family_basis_all.py)
names35 = list(pd.read_csv(
    "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")["ParamName"])
tbp = np.load(TB / "twobound_params.npy")
pairs = {}
for i in range(30):
    dcol = np.where(tbp[2 * i] != tbp[2 * i + 1])[0]
    assert len(dcol) == 1
    rr = [2 * i, 2 * i + 1]
    vv = [float(tbp[r, int(dcol[0])]) for r in rr]
    o = np.argsort(vv)
    pairs[names35[int(dcol[0])]] = [rr[k] for k in o]
NGC = {r: np.load(TB / f"run_{r:04d}/nu05_stats.npz") for r in range(60)}

old = dict(np.load(AMP_PATH, allow_pickle=True))
if not BAK_PATH.exists():
    np.savez(BAK_PATH, **old)
    print(f"backed up -> {BAK_PATH}")

new = dict(old)
for st in ("v0", "v1", "v2"):
    B = np.asarray(BUNDLE[f"{st}__basis"], float)
    mean = np.asarray(BUNDLE[f"{st}__mean"], float)
    assert B.shape[1] == 22 and mean.shape[0] == 22, \
        f"{st}: bundle not on the 22-bin grid -- rebuild family_basis_all first"
    pinvB = np.linalg.pinv(B)
    fid_meas = np.asarray(FID[f"mf_{st}"], float)[ZI]
    a_fid = (fid_meas - mean) @ pinvB
    dTB = np.stack([np.asarray(NGC[hi][f"mf_{st}"], float)[ZI]
                    - np.asarray(NGC[lo][f"mf_{st}"], float)[ZI]
                    for lo, hi in pairs.values()])
    a_tb = dTB @ pinvB
    new[f"{st}__basis"] = B
    new[f"{st}__mean"] = mean
    new[f"{st}__a_sobol"] = np.asarray(BUNDLE[f"{st}__amps"], float)
    new[f"{st}__family_names"] = np.asarray(BUNDLE[f"{st}__fam_names"])
    new[f"{st}__grid"] = np.asarray(BUNDLE[f"{st}__x"], float)
    new[f"{st}__fid_measured"] = fid_meas
    new[f"{st}__a_fid"] = a_fid
    new[f"{st}__a_fid_lstsq"] = a_fid
    new[f"{st}__a_twobound"] = a_tb
    new[f"{st}__a_twobound_lstsq"] = a_tb
    new[f"{st}__tb_resid"] = np.abs(dTB - a_tb @ B).max(1)
    print(f"{st}: basis {B.shape}, families "
          f"{list(np.asarray(BUNDLE[f'{st}__fam_names']))}, "
          f"span-resid fid {np.abs((fid_meas - mean) - a_fid @ B).max():.2e}")

np.savez(AMP_PATH, **new)
print(f"rewrote {AMP_PATH} ({len(new)} keys)")
