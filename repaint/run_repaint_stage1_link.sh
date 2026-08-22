#!/bin/bash
# ── FIDUCIAL REPAINT, STAGE 1: link the existing DMO conditioning ─────────────
# NO Slurm needed — this makes ~100 symlinks and writes 60 small .npy files.
# Run it directly on a workstation/login node (seconds):
#
#     bash repaint/run_repaint_stage1_link.sh
#     bash repaint/run_repaint_stage1_link.sh --verify-only     # re-check, write nothing
#
# WHY LINK AND NOT RE-PROJECT.  Stage 1 is COSMOLOGY-INDEPENDENT: it projects DM
# particles into z-slabs, extracts per-halo DMO cutouts, and writes
# stage1_slab*.npz + stage1_manifest.json.  The 35-dim parameter vector is never
# used by any of that math — project_and_extract() only np.save()s it
# (src/bind/inference/paint_stages.py:266) and echoes its filename into the
# manifest (:337).  The lensing geometry is likewise unaffected: the manifest's
# Omega_m = 0.3089 was read from the TNG300-Dark snapshot HEADER, not from the
# buggy params.npy, so distances and the lensing kernel were always correct.
# Re-projecting would burn ~4 nodes x 3 h x 20 snapshots to reproduce 14.5 GiB of
# byte-identical output.  We therefore SYMLINK the slabs and the manifest
# (read-only inputs, never copied, never modified) and write the CORRECTED
# params.npy as a real file alongside them, inside the new tree.
#
# The corrected vector is what stage 2 reads: generate_from_stage1() loads
# stage1_dir/params.npy when --params is omitted (paint_stages.py:460-462).  The
# repaint passes --params explicitly as well (belt and braces).
#
# Env: see repaint/repaint_env.sh (REPAINT_ROOT, LC_OLD, ...).

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

VERIFY_ONLY=0
[[ "${1:-}" == "--verify-only" ]] && VERIFY_ONLY=1

repaint_preamble
repaint_lock

# ── the corrected 35-dim vector must exist first ──────────────────────────────
if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "ERROR: $PARAMS_FILE not found." >&2
    echo "       Run first:  python repaint/make_corrected_params.py" >&2
    exit 1
fi

# ── the shared lightcone geometry (read-only input) ───────────────────────────
if [[ ! -f "$LC_OLD/lightcone_transforms.json" ]]; then
    echo "ERROR: $LC_OLD/lightcone_transforms.json missing" >&2; exit 1
fi
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    ln -sfn "$LC_OLD/lightcone_transforms.json" "$REPAINT_ROOT/lightcone_transforms.json"
fi

echo
echo "=== linking stage-1 DMO conditioning: $LC_OLD -> $REPAINT_ROOT ==="
for IDX in $(seq 0 $((N_SNAPS - 1))); do
    S3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
    SRC="$LC_OLD/snap_${S3}/stage1"
    DST="$REPAINT_ROOT/snap_${S3}/stage1"
    repaint_guard "$DST" "stage1 dst" || exit 1

    [[ -d "$SRC" ]] || { echo "ERROR: missing source stage1 dir $SRC" >&2; exit 1; }
    [[ -f "$SRC/stage1_manifest.json" ]] || { echo "ERROR: no manifest in $SRC" >&2; exit 1; }

    if [[ "$VERIFY_ONLY" -eq 0 ]]; then
        mkdir -p "$DST"
        # slabs + manifest: symlinks into the released tree (never copied).
        # nullglob: without it an unmatched glob would `ln -sfn` the LITERAL string
        # "$SRC/stage1_slab*.npz" into the new tree as a dangling link.
        shopt -s nullglob
        SLABS=("$SRC"/stage1_slab*.npz)
        shopt -u nullglob
        WANT_SLABS=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_slabs'])" \
                     "$SRC/stage1_manifest.json")
        if [[ "${#SLABS[@]}" -ne "$WANT_SLABS" ]]; then
            echo "ERROR: $SRC has ${#SLABS[@]} stage1_slab*.npz but the manifest says n_slabs=$WANT_SLABS" >&2
            exit 1
        fi
        for f in "${SLABS[@]}"; do
            ln -sfn "$f" "$DST/$(basename "$f")"
        done
        ln -sfn "$SRC/stage1_manifest.json" "$DST/stage1_manifest.json"

        # the CORRECTED conditioning vector: real files in the new tree.
        # params.npy is what stage 2 / recomposite read.  tng300_params.npy is the
        # honestly-named provenance twin; fiducial_params.npy is kept only as a
        # symlink for layout compatibility with the released tree — it was the
        # decoy name that caused this bug (bind.fiducial_params() is the CAMELS
        # fiducial, NOT TNG300's).
        cp -f "$PARAMS_FILE" "$DST/params.npy"
        cp -f "$PARAMS_FILE" "$DST/tng300_params.npy"
        ln -sfn tng300_params.npy "$DST/fiducial_params.npy"
    fi
done

# ── verification: manifest fields present, cosmology now self-consistent ──────
echo
echo "=== verifying linked stage 1 ==="
python - "$REPAINT_ROOT" "$LC_OLD" "$PARAMS_FILE" "${SNAPSHOTS[@]}" <<'PY'
import json, os, sys
import numpy as np

root, old, params_file = sys.argv[1], sys.argv[2], sys.argv[3]
snaps = [int(s) for s in sys.argv[4:]]
want = np.load(params_file).astype(np.float64).ravel()

hdr = f"{'snap':>5} {'z':>8} {'a':>8} {'Om(man)':>8} {'Om(par)':>8} {'n_halos':>8} " \
      f"{'slabs':>6} {'links':>6}  status"
print(hdr); print("-" * len(hdr))
bad = 0
for s in snaps:
    d = f"{root}/snap_{s:03d}/stage1"
    man_p = f"{d}/stage1_manifest.json"
    msgs = []
    if not os.path.islink(man_p):
        msgs.append("manifest NOT a symlink")
    try:
        man = json.load(open(man_p))
    except Exception as e:
        print(f"{s:>5}  FAIL: cannot read manifest ({e})"); bad += 1; continue

    # required per-snapshot geometry/redshift fields
    for k in ("redshift", "scale_factor", "n_slabs", "npix", "box_size", "Omega_m",
              "n_halos", "transforms_snap_idx", "proj_dir", "disp", "flip"):
        if k not in man:
            msgs.append(f"manifest missing '{k}'")
    z, a = man.get("redshift", float("nan")), man.get("scale_factor", float("nan"))
    if not (a > 0 and abs(1.0 / a - 1.0 - z) < 1e-6):
        msgs.append("redshift/scale_factor inconsistent")

    # slab symlinks resolve, and point back into the released tree
    n_sl = int(man.get("n_slabs", 0))
    n_lk = 0
    for si in range(n_sl):
        p = f"{d}/stage1_slab{si:02d}.npz"
        if not os.path.islink(p):
            msgs.append(f"slab{si:02d} not a symlink")
        elif not os.path.exists(p):
            msgs.append(f"slab{si:02d} dangling")
        else:
            n_lk += 1
            if not os.path.realpath(p).startswith(os.path.realpath(old)):
                msgs.append(f"slab{si:02d} does not point into {old}")

    # params.npy is a REAL file holding the corrected vector
    pp = f"{d}/params.npy"
    if os.path.islink(pp):
        msgs.append("params.npy is a symlink (must be a real file)")
    try:
        got = np.load(pp).astype(np.float64).ravel()
        if not np.array_equal(got, want):
            msgs.append("params.npy != corrected vector")
    except Exception as e:
        msgs.append(f"params.npy unreadable ({e})")
        got = np.full(35, np.nan)

    # THE GATE THAT WOULD HAVE CAUGHT THE ORIGINAL BUG:
    om_man, om_par = float(man.get("Omega_m", float("nan"))), float(got[0])
    if not abs(om_par - om_man) / om_man < 1e-3:
        msgs.append(f"Omega_m mismatch: manifest {om_man} vs params {om_par}")

    ok = not msgs
    bad += (not ok)
    print(f"{s:>5} {z:>8.4f} {a:>8.4f} {om_man:>8.4f} {om_par:>8.4f} "
          f"{man.get('n_halos', -1):>8} {n_sl:>6} {n_lk:>6}  "
          f"{'OK' if ok else 'FAIL: ' + '; '.join(msgs)}")

print("-" * len(hdr))
if bad:
    print(f"FAIL: {bad}/{len(snaps)} snapshots have problems"); sys.exit(1)
print(f"PASS: {len(snaps)}/{len(snaps)} snapshots linked, manifests complete, "
      f"params corrected and consistent with the substrate cosmology")
PY

echo
echo "=== stage 1 (link) done ==="
echo "    disk used by the new tree so far: symlinks only (~0 B); "
echo "    the released 14.5 GiB of stage1_slab*.npz is reused in place."
echo "    Next:  sbatch repaint/run_repaint_generate.sh"
