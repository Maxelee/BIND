"""Harmonize the 3 repainted Sobol runs' XPk-path spectra in the assembled datasets.

WHY THIS EXISTS. The 2026-08-04 `apply_xpkfix.py` repair assumed the repainted runs
0114/0115/0117's per-run caches "carry the same uncorrected normalization" as the other
253 and scaled the whole 256-run array by the xpk factor F=1.2015796e7. That assumption
was wrong: the repainted caches were rebuilt (2026-08-03, post-repaint) by the FIXED
stats pipeline, so their XPk-path entries were already physical and the blind scaling
double-corrected them. Verified 2026-08-05 (fig07_covariation panels j/k/l showing "only
three lines" was the symptom):

  key                              253 clean rows           3 repainted rows
  t__cl_{kappa_y,kappa_tau,yt}__value   cache x F (physical, OK)   cache x F = phys x F (BAD)
  t__cl_{kappa_y,kappa_tau,yt}__err     cache (1.2e7 LOW, BAD)     cache (physical, OK)
  t__cl_kappa__{value,err}              cache (legacy: diag phys,  cache (ALL phys ->
                                        off-diag 1.2e7 low, OK)    off-diag BAD)

The err rows are a second `apply_xpkfix` gap: it only scaled `__value` keys, but the
released vintage (bak253) had the cross ERRS in the physical convention too, so the
reassembly regression also dropped the err scaling for the 253.

WHAT THIS DOES, per dataset (emulator_dataset_nu05.npz + emulator_dataset_xpkfix.npz):
  1. cross VALUE rows of 0114/0115/0117 := their own per-run cache arrays (bit-exact
     restore of the physical convention, consistent with the corrected 253);
  2. cross ERR rows of the 253 := the bak253 (released-vintage) rows (bit-exact restore);
     repainted err rows verified == cache and left alone;
  3. t__cl_kappa__{value,err} repainted rows: OFF-DIAGONAL (z_s_i x plane_j, i != j;
     the XPk plane-cross path) divided by F, so the whole column stays in the legacy
     bak253 convention (diag physical / off-diag 1.2e7 low). Diagonal untouched --
     it is the only part any current consumer reads ([:, ZI, ZI, :]).
F is derived empirically from the dataset itself (clean-row value/cache ratio, asserted
uniform), never hardcoded. Every precondition is asserted before writing; if the dataset
does not look exactly like the broken state described above, this refuses rather than
guesses. Post-fix, all six cross keys match bak253 on the 253 common rows to round-off,
and no t__* key flags the repainted runs as >100x outliers.

    python apply_repaint3fix.py            # verify + fix both datasets + verify
    python apply_repaint3fix.py --dry-run  # all checks, nothing written
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
SB35 = CEPH / "bind_sb35"
RUNS = SB35 / "runs"
BAK = SB35 / "emulator_dataset_xpkfix.bak253.npz"
DATASETS = [SB35 / "emulator_dataset_nu05.npz", SB35 / "emulator_dataset_xpkfix.npz"]
REPAINTED = (114, 115, 117)

# dataset key -> (per-run cache file, cache array key)
CROSS_VAL = {"t__cl_kappa_y__value": ("Cl_kappa_y.npz", "cl_ky"),
             "t__cl_kappa_tau__value": ("Cl_tau.npz", "cl_kt"),
             "t__cl_yt__value": ("Cl_tau.npz", "cl_yt")}
CROSS_ERR = {"t__cl_kappa_y__err": ("Cl_kappa_y.npz", "cl_ky_err"),
             "t__cl_kappa_tau__err": ("Cl_tau.npz", "cl_kt_err"),
             "t__cl_yt__err": ("Cl_tau.npz", "cl_yt_err")}
KAPPA = {"t__cl_kappa__value": ("Cl_kappa.npz", "cl"),
         "t__cl_kappa__err": ("Cl_kappa.npz", "cl_err")}
F_EXPECT = 1.2015796e7   # sanity anchor only; the applied F is derived empirically


def cache_arr(rid: int, fn: str, key: str) -> np.ndarray:
    return np.load(RUNS / f"run_{rid:04d}" / fn)[key].astype(float)


def uniform_ratio(num: np.ndarray, den: np.ndarray, what: str) -> float:
    """Median of num/den over finite, nonzero-den elements; asserts uniformity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = num / den
    rf = rat[np.isfinite(rat) & (den != 0)]
    assert rf.size, f"{what}: no finite ratio elements"
    med = float(np.median(rf))
    dev = float(np.max(np.abs(rf / med - 1)))
    assert dev < 1e-9, f"{what}: ratio not uniform (max dev {dev:.3e})"
    return med


def outlier_scan(store: dict, ids: np.ndarray, bad: np.ndarray) -> list[str]:
    flagged = []
    for k, v in store.items():
        if not (k.startswith("t__") and k.endswith(("__value", "__err"))):
            continue
        v = np.asarray(v, float)
        if v.ndim < 1 or v.shape[0] != len(ids) or v.dtype.kind != "f":
            continue
        with np.errstate(all="ignore"):
            m = np.nanmedian(np.abs(v.reshape(v.shape[0], -1)), axis=1)
        g = np.nanmedian(m[~bad])
        if np.isfinite(g) and g > 0 and np.all(m[bad] / g > 100):
            flagged.append(k)
    return flagged


def fix_dataset(path: Path, dry: bool) -> None:
    print(f"\n== {path.name}")
    d = np.load(path, allow_pickle=True)
    old = np.load(BAK, allow_pickle=True)
    ids = np.array([int(r) for r in d["run_ids"]])
    ids_old = [int(r) for r in old["run_ids"]]
    assert len(ids) == 256, f"expected the 256-run dataset, got {len(ids)}"
    bad = np.isin(ids, REPAINTED)
    common = sorted(set(ids_old) & set(ids.tolist()))
    io = [ids_old.index(r) for r in common]
    inw = [int(np.where(ids == r)[0][0]) for r in common]
    assert not (set(REPAINTED) & set(common)), "repainted runs unexpectedly in bak253"
    store = {k: d[k] for k in d.files}

    # ── derive F empirically: clean-row value / cache, asserted uniform ──────
    r0 = common[0]
    F = uniform_ratio(np.asarray(store["t__cl_kappa_y__value"], float)[inw[0]],
                      cache_arr(r0, *CROSS_VAL["t__cl_kappa_y__value"]),
                      f"run{r0:04d} cl_ky value/cache")
    assert abs(F / F_EXPECT - 1) < 1e-3, f"derived F={F:.7e} != expected ~{F_EXPECT:.7e}"
    print(f"   derived xpk factor F = {F:.7e}")

    # ── preconditions: dataset must look exactly like the known broken state ──
    pre_scan = outlier_scan(store, ids, bad)
    assert pre_scan, "no repainted-run outlier keys found -- dataset already fixed?"
    print(f"   pre-fix outlier keys: {sorted(pre_scan)}")
    for k, (fn, ck) in CROSS_VAL.items():
        for rid in REPAINTED:
            i = int(np.where(ids == rid)[0][0])
            f_i = uniform_ratio(np.asarray(store[k], float)[i], cache_arr(rid, fn, ck),
                                f"{k} run{rid:04d}")
            assert abs(f_i / F - 1) < 1e-9, \
                f"{k} run{rid:04d}: row/cache={f_i:.7e} != F (not double-scaled?)"
    for k, (fn, ck) in CROSS_ERR.items():
        v = np.asarray(store[k], float)
        for rid in (r0, *REPAINTED):   # errs are at cache convention for EVERY run
            i = int(np.where(ids == rid)[0][0])
            c = cache_arr(rid, fn, ck)
            assert np.array_equal(v[i], c, equal_nan=True), \
                f"{k} run{rid:04d}: err row != cache (unexpected state)"
    offdiag = ~np.eye(5, dtype=bool)[:, :, None]   # (5,5,1) mask over (z_s, plane, ell)
    for k, (fn, ck) in KAPPA.items():
        v = np.asarray(store[k], float)
        for rid in REPAINTED:
            i = int(np.where(ids == rid)[0][0])
            c = cache_arr(rid, fn, ck)
            assert np.array_equal(v[i], c, equal_nan=True), \
                f"{k} run{rid:04d}: row != cache (unexpected state)"
            # off-diag must sit ~F above the clean population (idempotency guard)
            g = np.nanmedian(np.abs(np.asarray(store[k], float)[~bad]), axis=0)
            with np.errstate(all="ignore"):
                r_off = np.nanmedian((np.abs(v[i]) / g)[np.broadcast_to(offdiag, v[i].shape)])
            assert r_off > 1e3, f"{k} run{rid:04d}: off-diag not elevated (already fixed?)"

    # ── apply ────────────────────────────────────────────────────────────────
    for k, (fn, ck) in CROSS_VAL.items():
        arr = np.array(store[k], dtype=float, copy=True)
        for rid in REPAINTED:
            i = int(np.where(ids == rid)[0][0])
            c = cache_arr(rid, fn, ck)
            assert arr[i].shape == c.shape
            arr[i] = c                      # bit-exact physical restore
        store[k] = arr
    for k, (fn, ck) in CROSS_ERR.items():
        arr = np.array(store[k], dtype=float, copy=True)
        arr[inw] = np.asarray(old[k], float)[io]   # bit-exact released-vintage restore
        store[k] = arr                              # repainted rows: cache = physical, kept
    for k, (fn, ck) in KAPPA.items():
        arr = np.array(store[k], dtype=float, copy=True)
        for rid in REPAINTED:
            i = int(np.where(ids == rid)[0][0])
            arr[i] = np.where(np.broadcast_to(offdiag, arr[i].shape), arr[i] / F, arr[i])
        store[k] = arr

    # ── verify ───────────────────────────────────────────────────────────────
    for k in (*CROSS_VAL, *CROSS_ERR, *KAPPA):
        vo, vn = np.asarray(old[k], float)[io], np.asarray(store[k], float)[inw]
        resid = np.nanmax(np.abs(vn - vo))
        scale = np.nanmax(np.abs(vo)) or 1.0
        assert resid / scale < 1e-9, f"{k}: common rows no longer match bak253"
    post_scan = outlier_scan(store, ids, bad)
    assert not post_scan, f"outlier keys survive the fix: {post_scan}"
    print("   verify OK: common rows match bak253 on all 8 keys; no repainted-run "
          "outlier left in any t__*__value/err key")

    if dry:
        print("   dry run -- nothing written")
        return
    bak_out = path.with_name(path.stem + ".pre_repaint3fix_20260805.npz")
    if not bak_out.exists():
        shutil.copy2(path, bak_out)
        print(f"   pre-fix file preserved -> {bak_out.name}")
    np.savez_compressed(path, **store)
    print(f"   fixed dataset written in place -> {path.name} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for p in DATASETS:
        fix_dataset(p, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
