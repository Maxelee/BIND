#!/usr/bin/env python3
"""Build (and verify) the corrected 35-dim conditioning vector for the repaint.

THE BUG.  Every snapshot of the released fiducial lightcone
(``bind_lightcone_tng/snap_*/stage1/{params,fiducial_params}.npy``) carries the
**CAMELS SB35 fiducial cosmology** instead of TNG300's, because
``run_lightcone_project.sh`` wrote ``bind.fiducial_params()`` — whose cosmology
block is CAMELS' — as the stage-1 params file, and ``run_lightcone_generate.sh``
never passed ``--params``, so stage 2 fell back to it.  The 30 astrophysical
entries were always correct; only indices 0, 1, 6, 7, 8 are wrong:

    Omega0  0.3000 -> 0.3089    sigma8 0.8000 -> 0.8159   OmegaBaryon 0.0490 -> 0.0486
    HubbleParam 0.6711 -> 0.6774                          n_s 0.9624 -> 0.9667

The conditioned baryon fraction Ob/Om was 1.038141x too high, which is why the
gas/tau planes carry a +7.7% power excess (1.038141^2 = 1.0777).

WHAT THIS SCRIPT DOES
    1. builds the corrected vector = astro fiducial + TNG300 cosmology,
    2. HARD-asserts it equals bind_science/runs/fiducial/run_0000/params.npy
       bit-for-bit (that run is the verified-correct sibling),
    3. prints a diff table against the released (wrong) vector, including the
       shift in *network-input* (min-max normalised) units,
    4. writes it to the NEW campaign tree only.

It is idempotent (re-writing identical bytes is a no-op) and refuses to write
into any released tree.

Usage (nothing is submitted; this is a plain CPU script)::

    python repaint/make_corrected_params.py                    # write $REPAINT_ROOT/params.npy
    python repaint/make_corrected_params.py --write_snapshots  # + per-snapshot stage1 copies
    python repaint/make_corrected_params.py --check_only       # verify, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repaint_paths as rp  # noqa: E402


def build_corrected() -> np.ndarray:
    """Corrected 35-dim vector: fiducial astrophysics + TNG300 cosmology."""
    import bind

    # Preferred: the permanent helper added by the code fix (bind.params).
    fn = getattr(bind, "tng300_params", None)
    if callable(fn):
        p = np.asarray(fn(), dtype=np.float64)
    else:  # older install without the fix — construct explicitly
        p = bind.vary_params(dict(rp.TNG300_COSMOLOGY))
    return np.asarray(p, dtype=np.float64)


def normalised(p: np.ndarray) -> np.ndarray:
    """Map the raw vector into the model's [0,1] input space (log10 where flagged)."""
    from bind.params import PARAM_LOG_FLAG, PARAM_MAX, PARAM_MIN

    lg = PARAM_LOG_FLAG == 1
    # np.where would evaluate log10 on the linear entries too (MinVal can be 0),
    # so branch by assignment instead.
    q, lo, hi = np.array(p, float), np.array(PARAM_MIN, float), np.array(PARAM_MAX, float)
    q[lg] = np.log10(q[lg])
    lo[lg] = np.log10(PARAM_MIN[lg])
    hi[lg] = np.log10(PARAM_MAX[lg])
    return (q - lo) / (hi - lo)


def print_diff_table(old: np.ndarray, new: np.ndarray) -> None:
    from bind.params import PARAM_MAX, PARAM_MIN, PARAM_NAMES

    du = normalised(new) - normalised(old)
    idx = np.nonzero(old != new)[0]
    print("\n  corrected vector vs the released (buggy) vector")
    print("  " + "-" * 92)
    print(f"  {'i':>3}  {'parameter':<22} {'released':>12} {'corrected':>12} "
          f"{'ratio':>9} {'d(norm)':>9}  in prior?")
    print("  " + "-" * 92)
    for i in idx:
        inb = "yes" if PARAM_MIN[i] <= new[i] <= PARAM_MAX[i] else "OUT OF PRIOR"
        print(f"  {i:>3}  {PARAM_NAMES[i]:<22} {old[i]:>12.6g} {new[i]:>12.6g} "
              f"{new[i] / old[i]:>9.5f} {du[i]:>+9.5f}  {inb}")
    print("  " + "-" * 92)
    print(f"  {len(idx)} of 35 entries differ; the other {35 - len(idx)} "
          f"(all astrophysical) are identical.")
    print(f"  max |d(norm)| = {np.abs(du).max():.5f} at index {int(np.argmax(np.abs(du)))} "
          f"({PARAM_NAMES[int(np.argmax(np.abs(du)))]})")
    print(f"  f_b = Ob/Om : released {rp.FB_WRONG:.6f} -> corrected {rp.FB_RIGHT:.6f} "
          f"(ratio {rp.FB_RATIO:.6f}, squared {rp.FB_RATIO ** 2:.6f})")
    print(f"  => expected gas/tau plane POWER ratio new/old = {rp.POWER_RATIO_EXPECTED:.6f} "
          f"(the +7.7% excess removed)")
    print("  NOTE sigma8 moves MORE in normalised input units than Omega_b, and\n"
          "       Omega_m/h/n_s move too: the repaint changes texture as well as\n"
          "       amplitude.  It is NOT the old maps divided by 1.0777.")


def write_if_changed(path: Path, vec: np.ndarray, force: bool) -> str:
    """Idempotent write.  Returns 'written' | 'unchanged' | 'REFUSED'."""
    rp.assert_writable(path, "params")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            cur = rp.load_params(path)
        except Exception:
            cur = None
        if cur is not None and np.array_equal(cur, vec):
            return "unchanged"
        if not force:
            print(f"  REFUSED: {path} exists and differs (pass --force to overwrite)",
                  file=sys.stderr)
            return "REFUSED"
    np.save(path, vec.astype(np.float64))
    return "written"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repaint_root", type=Path, default=rp.REPAINT_ROOT,
                    help="new campaign tree (default $REPAINT_ROOT)")
    ap.add_argument("--reference", type=Path, default=rp.PARAMS_REFERENCE,
                    help="verified-correct params.npy to match bit-for-bit")
    ap.add_argument("--old", type=Path,
                    default=rp.LC_OLD / "snap_096" / "stage1" / "params.npy",
                    help="released (buggy) params.npy, for the diff table")
    ap.add_argument("--write_snapshots", action="store_true",
                    help="also write $ROOT/snap_NNN/stage1/params.npy for all 20 snapshots")
    ap.add_argument("--check_only", action="store_true", help="verify only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing params.npy that differs")
    args = ap.parse_args()

    print("=" * 96)
    print("BIND fiducial repaint — corrected 35-dim conditioning vector")
    print("=" * 96)

    vec = build_corrected()

    # ── gate 1: bit-for-bit against the verified-correct sibling ──────────────
    if not args.reference.exists():
        print(f"FAIL: reference not found: {args.reference}", file=sys.stderr)
        return 2
    ref = rp.load_params(args.reference)
    if not np.array_equal(vec, ref):
        bad = np.nonzero(vec != ref)[0]
        print(f"FAIL: constructed vector != {args.reference}", file=sys.stderr)
        for i in bad:
            print(f"   idx {i}: constructed {vec[i]!r} vs reference {ref[i]!r}",
                  file=sys.stderr)
        return 2
    print(f"  [OK] constructed == {args.reference} (bit-for-bit, float64, all 35)")

    # ── gate 2: every entry inside the SB35 prior box (model not extrapolating) ─
    from bind.params import PARAM_MAX, PARAM_MIN, PARAM_NAMES
    out = [i for i in range(35) if not (PARAM_MIN[i] <= vec[i] <= PARAM_MAX[i])]
    if out:
        print("FAIL: entries outside the SB35 prior box: "
              + ", ".join(f"{i}:{PARAM_NAMES[i]}" for i in out), file=sys.stderr)
        return 2
    print("  [OK] all 35 entries inside the SB35 prior box")

    # ── the diff table ────────────────────────────────────────────────────────
    if args.old.exists():
        print_diff_table(rp.load_params(args.old), vec)
    else:
        print(f"  (skipping diff table — released vector not found at {args.old})")

    if args.check_only:
        print("\n  --check_only: nothing written.")
        return 0

    # ── write into the NEW tree only ──────────────────────────────────────────
    root = rp.assert_writable(args.repaint_root, "repaint root")
    targets = [Path(root) / "params.npy"]
    if args.write_snapshots:
        targets += [rp.snap_dir(root, s) / "stage1" / "params.npy" for s in rp.SNAPSHOTS]

    print(f"\n  writing corrected vector into {root}")
    status = {}
    for t in targets:
        status[t] = write_if_changed(Path(t), vec, args.force)
        print(f"    {status[t]:>9}  {t}")

    if any(v == "REFUSED" for v in status.values()):
        print("\nFAIL: at least one target refused (existing file differs).", file=sys.stderr)
        return 3
    print("\n  PASS — corrected params in place.  Next: repaint/run_repaint_stage1_link.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
