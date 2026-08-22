#!/bin/bash
#SBATCH --job-name=repaint_planes
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repaint_planes_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repaint_planes_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint="cascadelake|skylake"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── FIDUCIAL REPAINT, STAGE 3 (CPU): composites -> lux planes ─────────────────
# One array task = one snapshot = 4 planes of each kind, written by direct calls
# to the three CLIs:
#
#     bind.cli.paint_lensplane  -> lenspot{PP}.dat  (mass; task 0 also config.dat)
#     bind.cli.paint_yplane     -> yplane{PP}.dat   (Compton-y)
#     bind.cli.paint_tauplane   -> tauplane{PP}.dat (electron column)
#
# The CLIs are invoked DIRECTLY (not via run_lightcone_{lensplane,yplane,tauplane}.sh)
# on purpose.  Those scripts default OUTPUT_ROOT/LENSPLANE_DIR to the RELEASED tree
# and would write there if an export were ever dropped; calling the CLIs with
# explicit absolute paths means every path this stage touches is one repaint_guard
# has seen.  The flags below reproduce those scripts exactly.
#
# ALL THREE PLANE TYPES ARE REPAINTED.  The mass channel is nearly (not exactly)
# immune: patch_mass_match pins each halo's 3-channel TOTAL to the DMO cutout, so
# C_l^kk moves by <0.3% below ell=4e3 — but by -0.9% at ell 4e3-1e4 and -1.9% at
# ell>1e4, inside the plotted/trusted range (ELL_TRUST=3e4).
#
# REUSE_MASS_PLANES IS GONE — DO NOT REINTRODUCE IT.  It symlinked the released
# lenspot*.dat + config.dat into $LENSPLANE_DIR.  bind.inference.lensplane writes
# planes with a plain open(path,"wb") (lines 193/214/401), which FOLLOWS SYMLINKS
# and truncates the target: any later plane repaint (FORCE=1, or one plane failing
# the completeness test) would have destroyed 53.7 GB of released lenspot planes
# in place, and repaint_guard could not see it because $LENSPLANE_DIR itself
# legitimately lives outside the released trees.  Measured plane cost is ~25-50 s
# per lenspot task and ~17-34 s per yplane task (sacct 2431124/2431144/2431884),
# i.e. the whole saving was a few node-minutes.  A hard pre-flight refusal is kept
# below in case an operator recreates such a link by hand.
#
# SAFETY INVARIANT (copied from n1000_body.sh:279-283, which relies on the same
# property): with simulation_format=PreProjected the current lux build only READS
# LP_output_dir / tsz_input_dir / tau_input_dir — all writes go to RT_output_dir,
# because `lenspot Lenspot(...)` is commented out at lux main.cpp:47.  lux's
# lenspot::write_phi (lenspot.cpp:451) would otherwise fopen("%s/lenspot%02d.dat",
# LPoutput_dir, "wb").  RE-VERIFY THIS IF LUX IS EVER REBUILT.
#
# Submit (author only), after stage 2b:
#     sbatch --dependency=afterok:$JID_RC repaint/run_repaint_planes.sh
#
# Env: see repaint/repaint_env.sh.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

repaint_preamble
repaint_lock

LENSPLANE_DIR=${LENSPLANE_DIR:-$REPAINT_ROOT/lensplanes}
repaint_assert_under_root "$LENSPLANE_DIR" "lensplane dir" || exit 1
mkdir -p "$LENSPLANE_DIR"
repaint_assert_no_released_symlinks "$LENSPLANE_DIR" "lensplane dir" || exit 1

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19}
SNAP3=$(printf '%03d' "${SNAPSHOTS[$IDX]}")
SNAP_DIR="$REPAINT_ROOT/snap_${SNAP3}"
STAGE1_DIR="$SNAP_DIR/stage1"
TRANSFORMS_FILE="$REPAINT_ROOT/lightcone_transforms.json"

[[ -f "$SNAP_DIR/composite_slab00.npz" ]] || {
    echo "ERROR: no composites in $SNAP_DIR" >&2; exit 1; }
[[ -f "$TRANSFORMS_FILE" ]] || {
    echo "ERROR: $TRANSFORMS_FILE missing (stage1_link writes it)" >&2; exit 1; }
[[ -f "$STAGE1_DIR/stage1_manifest.json" ]] || {
    echo "ERROR: $STAGE1_DIR/stage1_manifest.json missing" >&2; exit 1; }

# ── HARD GATE: the composites must already carry the FINAL paste geometry ─────
# Why this is not optional.  The current code writes a `composite_thermo` array
# into every composite_slab*.npz (pipeline.py:803, _save_composite_slab:441) — the
# RELEASED June-9 files do NOT have that key (verified by npz listing).  Because
# paint_yplane._yslab (cli/paint_yplane.py:49-52) PREFERS the stored
# composite_thermo when present, `--r200_factor 4.0` becomes a silent NO-OP for the
# y planes on this new tree.  So if stage 2b were skipped or only partly run after a
# GENERATE_R200=0.0 paint, BOTH lenspot and yplane would be built from square-tapered
# composites and 75 GB of planes plus a multi-thousand-core-hour trace would be spent
# on the wrong product.  summary.json records what was actually used; assert it.
python - "$SNAP_DIR/summary.json" "$R200_FACTOR" "$TAPER_FRAC" <<'PY'
import json, sys
p, want_r200, want_taper = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    s = json.load(open(p))
except Exception as e:
    raise SystemExit(f"ERROR: cannot read {p}: {e}")
got_r200 = float(s.get("r200_factor", -1))
got_taper = float(s.get("taper_frac", -1))
if abs(got_r200 - want_r200) > 1e-9:
    raise SystemExit(
        f"ERROR: {p} says r200_factor={got_r200} but this campaign wants {want_r200}.\n"
        "  The composites on disk were pasted with a different aperture. Run stage 2b\n"
        "  (repaint/run_repaint_recomposite.sh) before painting planes — otherwise the\n"
        "  stored composite_thermo silently overrides --r200_factor for the y planes.")
if abs(got_taper - want_taper) > 1e-9:
    raise SystemExit(f"ERROR: {p} says taper_frac={got_taper}, campaign wants {want_taper}")
print(f"[gate] composites OK: r200_factor={got_r200} taper_frac={got_taper} "
      f"n_halos={s.get('n_halos')} recomposited_from={s.get('recomposited_from', 'n/a (one-step)')}")
PY

# planes of this snapshot: lux numbers from 1, 4 planes per snapshot
PLANES=()
for si in 0 1 2 3; do PLANES+=("$(printf '%02d' $((IDX * 4 + 1 + si)))"); done

have_all() {  # have_all <prefix> <expected_bytes>
    # EXACT size, and not a symlink.  `-s` would (a) happily accept a plane
    # truncated by a job killed mid-write — these are single ~640 MB writes — and
    # (b) follow a symlink into the released tree and call it done.
    local pre=$1 want=$2 p f
    for p in "${PLANES[@]}"; do
        f="$LENSPLANE_DIR/${pre}${p}.dat"
        [[ -f "$f" && ! -L "$f" ]] || return 1
        [[ "$(stat -c%s "$f")" -eq "$want" ]] || return 1
    done
    return 0
}

echo "    lensplane_dir=$LENSPLANE_DIR"
echo "    expected sizes: lenspot=$LENSPOT_BYTES B  yplane/tauplane=$YTAU_BYTES B"

# ── 1. mass planes (lenspot); task 0 additionally writes config.dat ───────────
if [[ "$FORCE" != "1" ]] && have_all lenspot "$LENSPOT_BYTES" \
   && { [[ "$IDX" -ne 0 ]] || { [[ -f "$LENSPLANE_DIR/config.dat" ]] && [[ ! -L "$LENSPLANE_DIR/config.dat" ]]; }; }; then
    echo "=== [1/3] lenspot planes ${PLANES[*]} already present (exact size) — skipping ==="
else
    echo "=== [1/3] lenspot planes ${PLANES[*]} ==="
    LP_OPT=()
    if [[ "$IDX" -eq 0 ]]; then          # task 0 writes config.dat, needs every snap dir
        ALL_SNAP_DIRS=()
        for s in "${SNAPSHOTS[@]}"; do
            ALL_SNAP_DIRS+=("$REPAINT_ROOT/snap_$(printf '%03d' "$s")")
        done
        LP_OPT+=(--all_snap_dirs "${ALL_SNAP_DIRS[@]}")
    fi
    python -u -m bind.cli.paint_lensplane \
        --generate_dir "$SNAP_DIR" \
        --stage1_dir   "$STAGE1_DIR" \
        --output_dir   "$LENSPLANE_DIR" \
        --transforms   "$TRANSFORMS_FILE" \
        --lc_snap_idx  "$IDX" \
        --lc_n_snaps   "$N_SNAPS" \
        --lp_grid      "$LP_GRID" \
        "${LP_OPT[@]}"
fi

# ── 2. y planes ───────────────────────────────────────────────────────────────
if [[ "$FORCE" != "1" ]] && have_all yplane "$YTAU_BYTES"; then
    echo "=== [2/3] yplane ${PLANES[*]} already present (exact size) — skipping ==="
else
    echo "=== [2/3] yplane ${PLANES[*]} ==="
    python -u -m bind.cli.paint_yplane \
        --generate_dir "$SNAP_DIR" \
        --stage1_dir   "$STAGE1_DIR" \
        --output_dir   "$LENSPLANE_DIR" \
        --lc_snap_idx  "$IDX" \
        --lc_n_snaps   "$N_SNAPS" \
        --lp_grid      "$LP_GRID" \
        --r200_factor  "$R200_FACTOR"
fi

# ── 3. tau planes ─────────────────────────────────────────────────────────────
if [[ "$FORCE" != "1" ]] && have_all tauplane "$YTAU_BYTES"; then
    echo "=== [3/3] tauplane ${PLANES[*]} already present (exact size) — skipping ==="
else
    echo "=== [3/3] tauplane ${PLANES[*]} ==="
    python -u -m bind.cli.paint_tauplane \
        --generate_dir "$SNAP_DIR" \
        --stage1_dir   "$STAGE1_DIR" \
        --output_dir   "$LENSPLANE_DIR" \
        --lc_snap_idx  "$IDX" \
        --lc_n_snaps   "$N_SNAPS" \
        --lp_grid      "$LP_GRID"
fi

have_all lenspot "$LENSPOT_BYTES" && have_all yplane "$YTAU_BYTES" \
    && have_all tauplane "$YTAU_BYTES" || {
    echo "ERROR: planes missing or wrong size for snapshot ${SNAP3} after painting" >&2
    exit 1; }
repaint_assert_no_released_symlinks "$LENSPLANE_DIR" "lensplane dir (post-paint)" || exit 1
echo "=== planes done for snap ${SNAP3} (planes ${PLANES[*]}) ==="
