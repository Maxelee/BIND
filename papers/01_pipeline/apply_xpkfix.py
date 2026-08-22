"""Re-apply the xpk cross-normalization correction to a reassembled dataset.

WHY THIS EXISTS. The per-run Sobol caches (Cl_kappa_y.npz / Cl_tau.npz) still carry the
PRE-xpkfix cross-spectrum normalization (see the stale-caches memory + WORKLOG 2026-08-04):
the released emulator_dataset_xpkfix.npz got its name from a dataset-level correction that
plain `bind-emulator-assemble` knows nothing about. The 2026-08-03 253->256 reassembly
therefore silently dropped the fix: C^ky/C^kt/C^yt came out a uniform ~8.32e-08 of their
corrected values (autos and S(ell) bit-identical).

WHAT THIS DOES. Derives the correction EMPIRICALLY from the kept pre-reassembly backup
(emulator_dataset_xpkfix.bak253.npz): for every t__*__value key that is not bit-identical
on the 253 common runs, it forms the per-element ratio old/new, verifies the ratio is
uniform (scalar) or at worst constant-per-bin across runs/planes, and applies it to the
full 256-run array (the 3 repainted runs' caches carry the same uncorrected normalization,
so the same deterministic factor applies). Keys identical on the common runs are untouched.

Exactness check: after correction, the 253 common rows must match bak253 to float
round-off. The corrected dataset REPLACES emulator_dataset_xpkfix.npz in place after
writing a .prefix backup of the broken file.

CORRECTED 2026-08-05 (see apply_repaint3fix.py): the original version of this script
scaled the WHOLE 256-run array, on the assumption that the 3 repainted runs'
(0114/0115/0117) caches carried the same uncorrected normalization as the other 253.
They do not — the repaint rebuilt their stats with the FIXED pipeline, so their
XPk-path entries are already physical and the blind scaling double-corrected them
(the fig07 "only three lines" bug). This version therefore (a) scales ONLY rows whose
run is present in bak253, leaving fresh-pipeline rows at their cache convention, and
(b) also corrects t__*__err keys, which the released vintage carried in the same
physical convention as the values (the original script's __value-only filter silently
left the 253 cross errs 1.2e7 low).

    python apply_xpkfix.py            # verify + correct + verify
    python apply_xpkfix.py --dry-run  # report per-key ratios only
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
DS = CEPH / "bind_sb35" / "emulator_dataset_xpkfix.npz"
BAK = CEPH / "bind_sb35" / "emulator_dataset_xpkfix.bak253.npz"
BROKEN_KEEP = CEPH / "bind_sb35" / "emulator_dataset_xpkfix.broken_norm_20260803.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    old = np.load(BAK, allow_pickle=True)
    new = np.load(DS, allow_pickle=True)
    ids_old = [int(r) for r in old["run_ids"]]
    ids_new = [int(r) for r in new["run_ids"]]
    assert len(ids_new) == 256, f"expected the 256-run dataset, got {len(ids_new)}"
    common = sorted(set(ids_old) & set(ids_new))
    io = [ids_old.index(r) for r in common]
    inw = [ids_new.index(r) for r in common]
    print(f"common runs: {len(common)} (new-only: {sorted(set(ids_new)-set(ids_old))})")

    store, corrections = {}, {}
    for k in new.files:
        v = new[k]
        if not (k.startswith("t__") and k.endswith(("__value", "__err"))) \
                or v.dtype.kind != "f" or k not in old.files:
            store[k] = v
            continue
        vo, vn = old[k][io], v[inw]
        if np.array_equal(vo, vn, equal_nan=True):
            store[k] = v
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = vo / vn
        rat_f = rat[np.isfinite(rat) & (vn != 0)]
        med = float(np.median(rat_f))
        spread = float(np.nanmax(np.abs(rat_f / med - 1))) if rat_f.size else np.nan
        print(f"{k}: ratio old/new median {med:.6e}, max |dev from uniform| {spread:.2e}, "
              f"finite-ratio elements {rat_f.size}/{rat.size}")
        if spread < 1e-6:
            corrections[k] = med
            # scale ONLY the rows whose run is in bak253 (pre-fix caches). Runs
            # absent from bak253 (the repaints) have fixed-pipeline caches that
            # are already physical — scaling them double-corrects (2026-08-05).
            arr = np.array(v, dtype=float, copy=True)
            arr[inw] *= med
            store[k] = arr
        elif abs(med - 1) < 0.05:
            # ratio ~1 with scatter = recompute drift in a partial-coverage
            # statistic (wst/peak_R re-derived at reassembly), NOT a dropped
            # normalization. Keep the new values; report only.
            print(f"   ratio ~1 (recompute drift, e.g. partial-coverage stat) -- "
                  f"left uncorrected by design")
            store[k] = v
        else:
            raise AssertionError(
                f"{k}: neither uniform-factor nor ratio~1 -- refusing to guess")

    if not corrections:
        print("nothing to correct -- dataset already matches the backup on common runs")
        return 0
    print(f"corrected keys: {sorted(corrections)}")

    # exactness check BEFORE writing
    for k in corrections:
        resid = np.nanmax(np.abs(store[k][inw] - old[k][io]))
        scale = np.nanmax(np.abs(old[k][io])) or 1.0
        print(f"{k}: post-correction max |resid| vs bak253 = {resid:.3e} "
              f"(rel {resid/scale:.3e})")
        assert resid / scale < 1e-9, f"{k}: correction does not reproduce the backup"

    if a.dry_run:
        print("dry run -- nothing written")
        return 0

    if not BROKEN_KEEP.exists():
        shutil.copy2(DS, BROKEN_KEEP)
        print(f"broken file preserved -> {BROKEN_KEEP.name}")
    np.savez_compressed(DS, **store)
    print(f"corrected dataset written in place -> {DS.name} "
          f"({DS.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
