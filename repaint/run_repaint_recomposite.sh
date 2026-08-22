#!/bin/bash
#SBATCH --job-name=repaint_repaste
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repaint_repaste_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repaint_repaste_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── FIDUCIAL REPAINT, STAGE 2b (CPU): circular re-composite (r200_factor=4.0) ─
# Calls bind.cli.paint_recomposite DIRECTLY with explicit absolute paths.
#
# It used to delegate to run_lightcone_recomposite.sh with `export OUTPUT_ROOT`.
# That script defaults OUTPUT_ROOT to the RELEASED tree
# (/mnt/home/mlee1/ceph/bind_lightcone_tng) and writes IN-PLACE, so a single
# dropped export — or an operator running the inner script directly out of habit —
# would have overwritten 22 GB of released composites with no rollback, and
# repaint_guard could not see it because it only resolves the paths this wrapper
# itself computes.  The delegated logic was four lines; it is inlined below, which
# means every path this stage writes is one the guard has checked.
#
# WHY THIS STAGE EXISTS.  bind.cli.paint_generate defaults to --r200_factor 0.0
# (square taper) while the RELEASED tree was built with the circular 4xR200c
# paste (snap_*/summary.json: r200_factor 4.0, "recomposited_from" ...).  The
# repaint reproduces that two-step provenance exactly.  If you set
# GENERATE_R200=4.0 in stage 2, SKIP this stage — the patches and the compositing
# math are identical, it is only a question of how many times the maps are
# written.
#
# Submit (author only), after stage 2:
#     sbatch --dependency=afterok:$JID_GEN repaint/run_repaint_recomposite.sh
# afterok on the WHOLE job, not aftercorr: stage 2 is submitted as --array=1-19 (Tier 0
# covers snap_096 separately), so a per-task aftercorr chain from this 0-19 array would
# wait forever on a source task 0 that does not exist.
#
# Env: see repaint/repaint_env.sh.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

repaint_preamble
repaint_lock

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19}
SNAP3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
SNAP_DIR="$REPAINT_ROOT/snap_${SNAP3}"
STAGE1_DIR="$SNAP_DIR/stage1"
repaint_assert_under_root "$SNAP_DIR" "recomposite output" || exit 1

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    exit 1
fi

if [[ ! -f "$SNAP_DIR/composite_slab00.npz" ]]; then
    echo "ERROR: no stage-2 composites in $SNAP_DIR — run run_repaint_generate.sh first" >&2
    exit 1
fi

# skip-if-done: summary.json records the paste settings actually used
if [[ "$FORCE" != "1" && -f "$SNAP_DIR/summary.json" ]]; then
    if python - "$SNAP_DIR/summary.json" "$R200_FACTOR" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if abs(float(s.get("r200_factor", -1)) - float(sys.argv[2])) < 1e-9
         and s.get("recomposited_from") else 1)
PY
    then
        echo "=== snap ${SNAP3}: already recomposited at r200_factor=$R200_FACTOR — skipping ==="
        exit 0
    fi
fi

# ── re-composite IN PLACE inside the new tree (never the released one) ────────
echo "=== recomposite snap ${SNAP3}: $SNAP_DIR (r200_factor=$R200_FACTOR, taper=$TAPER_FRAC) ==="
python -u -m bind.cli.paint_recomposite \
    --stage1_dir    "$STAGE1_DIR" \
    --generated_dir "$SNAP_DIR" \
    --output_dir    "$SNAP_DIR" \
    --taper_frac    "$TAPER_FRAC" \
    --r200_factor   "$R200_FACTOR"

# confirm the paste settings landed in summary.json — run_repaint_planes.sh
# hard-fails on this, so catch a mismatch here where it is cheap to fix.
python - "$SNAP_DIR/summary.json" "$R200_FACTOR" "$TAPER_FRAC" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
r, t = float(s.get("r200_factor", -1)), float(s.get("taper_frac", -1))
if abs(r - float(sys.argv[2])) > 1e-9 or abs(t - float(sys.argv[3])) > 1e-9:
    raise SystemExit(f"ERROR: summary.json r200_factor={r} taper_frac={t} after recomposite")
print(f"[ok] summary.json r200_factor={r} taper_frac={t} recomposited_from={s.get('recomposited_from')}")
PY

echo "=== snap ${SNAP3} recomposited (r200_factor=$R200_FACTOR) ==="
