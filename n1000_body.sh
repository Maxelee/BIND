#!/bin/bash
# ── N1000 campaign: shared per-run body (sourced by run_n1000_{cca,preempt}.sh) ──
# One task = one run traced to N_REAL=1000 seed-paired lightcone realizations,
# written into a FRESH tree ($OUT_ROOT, default bind_n1000) — the existing 50-real
# products under bind_science / bind_sb35 are never touched (inputs are read-only).
#
# Manifest (task index -> run):
#   0        bind      trace-only from the fiducial lensplanes (kappa+y+tau)
#   1        dmo       DMO replane from stage1, kappa-only trace
#   2        truth     recomposite+paint from runs/truth/run_0000, kappa+y+tau
#   3..62    twobound  run_0000..run_0059, recomposite+paint, kappa+y+tau
#   63..318  sb35      run_0000..run_0255, recomposite+paint, kappa+y+tau
#
# Realizations are traced in NCHUNK chunks of CHUNK reals.  Chunk c covers global
# realizations [c*CHUNK+1 .. (c+1)*CHUNK] with lux base seed RT_SEED + 7*c*CHUNK,
# so per-realization seeds are exactly RT_SEED + 7*r for r = 1..N_REAL — the same
# ladder as the existing 50/550-real traces (pairing preserved; verified by
# n1000_seed_gate.sh before launch).  Chunk outputs run001..run<CHUNK> are renamed
# to global run%04d.  Each chunk writes a .done sentinel: a requeued/preempted job
# redoes only the interrupted chunk (~5 h max loss).
#
# Two submission lanes share this body and the work list without duplication:
# an atomic claim dir (.claims/task_%04d) is taken at start; a task claimed by a
# live job in the other lane exits immediately.  Claims record the owning job;
# after a permanent failure remove the claim dir by hand to release the task.
#
# Env overrides: OUT_ROOT, N_REAL, CHUNK, RT_SEED, FOV_DEG, LP_GRID, RT_GRID,
#                PAINT_PARALLEL, LUX_BIN, FORCE (=1 retrace even if complete).

set -euo pipefail

FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng}      # shared geometry (read-only)
SCI=${SCI:-/mnt/home/mlee1/ceph/bind_science}            # old tree (read-only)
SB35=${SB35:-/mnt/home/mlee1/ceph/bind_sb35}             # old tree (read-only)
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/bind_n1000}    # NEW campaign tree
N_REAL=${N_REAL:-1000}
CHUNK=${CHUNK:-125}
RT_SEED=${RT_SEED:-1992}          # base of the canonical seed ladder — never change
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}
FOV_DEG=${FOV_DEG:-5.0}
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
PAINT_PARALLEL=${PAINT_PARALLEL:-6}
FORCE=${FORCE:-0}
# Diffuse-gas correction (validated 2026-08-11, docs/diffuse_gas_validation.md):
# gas_diffuse = f_b * DMO * (1-alpha) at T=1e4K outside the paste apertures.
# Fixes the pasted tau under-count (mean 0.16-0.37 of true -> 1.01) with kappa
# bit-unchanged (mass moved between channels, not added).  DIFFUSE=0 reverts to
# the pasted-only construction.
DIFFUSE=${DIFFUSE:-1}
T_DIFFUSE_K=${T_DIFFUSE_K:-1e4}
F_B_DIFFUSE=${F_B_DIFFUSE:-}       # empty = CLI default (0.0486/0.3089)
N_SNAPS=20
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

# Checked before any per-task work (claims/mkdir): a bad CHUNK fails every task
# instantly with zero side effects, never a partial tree with a broken seed ladder.
(( N_REAL % CHUNK == 0 )) || { echo "ERROR: N_REAL=$N_REAL not divisible by CHUNK=$CHUNK" >&2; exit 1; }
NCHUNK=$(( N_REAL / CHUNK ))

# The seed ladder must not depend on the job environment: GSL_RNG_TYPE would
# change gsl_rng_default (the generator lux seeds per realization), silently
# breaking pairing between lanes/nodes.  Pin the default.
unset GSL_RNG_TYPE GSL_RNG_SEED || true

IDX=${TASK_IDX:?n1000_body.sh: set TASK_IDX (0..318) before sourcing}

# ── manifest ──────────────────────────────────────────────────────────────────
if   (( IDX == 0 )); then CAT=bind;     SRC_RUN=;                                          COMPUTE_TSZ=True;  COMPUTE_TAU=True
elif (( IDX == 1 )); then CAT=dmo;      SRC_RUN=;                                          COMPUTE_TSZ=False; COMPUTE_TAU=False
elif (( IDX == 2 )); then CAT=truth;    SRC_RUN="$SCI/runs/truth/run_0000";                COMPUTE_TSZ=True;  COMPUTE_TAU=True
elif (( IDX <= 62 )); then CAT=twobound; SRC_RUN="$SCI/runs/twobound/$(printf 'run_%04d' $((IDX - 3)))";  COMPUTE_TSZ=True; COMPUTE_TAU=True
elif (( IDX <= 318 )); then CAT=sb35;    SRC_RUN="$SB35/runs/$(printf 'run_%04d' $((IDX - 63)))";         COMPUTE_TSZ=True; COMPUTE_TAU=True
else echo "ERROR: task index $IDX out of range (0..318)" >&2; exit 1
fi
case "$CAT" in
  bind|dmo|truth) OUT_RUN="$OUT_ROOT/$CAT/run_0000" ;;
  twobound)       OUT_RUN="$OUT_ROOT/twobound/$(printf 'run_%04d' $((IDX - 3)))" ;;
  sb35)           OUT_RUN="$OUT_ROOT/sb35/$(printf 'run_%04d' $((IDX - 63)))" ;;
esac
[[ -z "$SRC_RUN" || -d "$SRC_RUN" ]] || { echo "ERROR: missing source run $SRC_RUN" >&2; exit 1; }

WORK="$OUT_RUN/_work"
LP_DIR="$WORK/lensplanes"
# bind traces straight off the master lensplanes ONLY in pasted-only mode; with
# DIFFUSE=1 it symlinks the master's mass planes (kappa unchanged) and paints
# fresh y/tau planes from diffuse-augmented master composites.
[[ "$CAT" == bind && "$DIFFUSE" != "1" ]] && LP_DIR="$FID/lensplanes"
RT_DIR="$WORK/rt_output"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"

echo "=== n1000 | task $IDX -> $CAT -> $OUT_RUN | N_REAL=$N_REAL CHUNK=$CHUNK seed=$RT_SEED | $(date) ==="
echo "    nodes=${SLURM_NNODES:-?} tasks=${SLURM_NTASKS:-?} job=$JOB_TAG"

# ── done / claim gates ────────────────────────────────────────────────────────
if [[ -f "$OUT_RUN/.trace_complete" && "$FORCE" != "1" ]]; then
  echo "=== already complete — skipping (FORCE=1 to redo) ==="; exit 0
fi

# Cross-lane campaign-parameter lock: both lanes must trace the same ladder.
# First task records the resolved params; every later task (either lane) must match.
mkdir -p "$OUT_ROOT"
PARAMS_LINE="RT_SEED=$RT_SEED N_REAL=$N_REAL CHUNK=$CHUNK LP_GRID=$LP_GRID RT_GRID=$RT_GRID FOV_DEG=$FOV_DEG DIFFUSE=$DIFFUSE T_DIFFUSE_K=$T_DIFFUSE_K F_B=${F_B_DIFFUSE:-default}"
if [[ ! -f "$OUT_ROOT/.campaign_params" ]]; then
  echo "$PARAMS_LINE" > "$OUT_ROOT/.campaign_params.$$" && mv -n "$OUT_ROOT/.campaign_params.$$" "$OUT_ROOT/.campaign_params"
  rm -f "$OUT_ROOT/.campaign_params.$$"
fi
if [[ "$(cat "$OUT_ROOT/.campaign_params")" != "$PARAMS_LINE" ]]; then
  echo "ERROR: campaign param mismatch — this job has [$PARAMS_LINE] but $OUT_ROOT/.campaign_params has [$(cat "$OUT_ROOT/.campaign_params")]" >&2
  exit 1
fi

CLAIM="$OUT_ROOT/.claims/task_$(printf '%04d' "$IDX")"
mkdir -p "$OUT_ROOT/.claims"
[[ "$FORCE" == "1" ]] && rm -rf "$CLAIM"       # a FORCE redo always retakes the claim
if mkdir "$CLAIM" 2>/dev/null; then
  echo "$JOB_TAG" > "$CLAIM/owner"
elif [[ ! -s "$CLAIM/owner" ]]; then
  # claim dir exists but owner never landed (crash between mkdir and echo):
  # self-heal, then re-read after a pause so simultaneous healers arbitrate
  echo "$JOB_TAG" > "$CLAIM/owner"; sleep 5
  [[ "$(cat "$CLAIM/owner" 2>/dev/null)" == "$JOB_TAG" ]] || { echo "=== task $IDX taken during claim self-heal — exiting ==="; exit 0; }
  echo "    reclaimed orphaned claim"
elif [[ "$(cat "$CLAIM/owner" 2>/dev/null)" == "$JOB_TAG" ]]; then
  echo "    resuming own claim (requeue)"
else
  echo "=== task $IDX claimed by job $(cat "$CLAIM/owner" 2>/dev/null || echo '?') — exiting (rm -r $CLAIM to release) ==="
  exit 0
fi
mkdir -p "$OUT_RUN" "$WORK"

module restore lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND

# ── stage 1: lensplanes ───────────────────────────────────────────────────────
count_planes() { ls "$LP_DIR"/"$1"*.dat 2>/dev/null | wc -l; }
planes_complete() {
  [[ "$(count_planes lenspot)" -eq $((N_SNAPS * 4)) ]] || return 1
  [[ "$COMPUTE_TSZ" == True ]] && { [[ "$(count_planes yplane)"   -eq $((N_SNAPS * 4)) ]] || return 1; }
  [[ "$COMPUTE_TAU" == True ]] && { [[ "$(count_planes tauplane)" -eq $((N_SNAPS * 4)) ]] || return 1; }
  return 0
}

ALL_FID_SNAPS=()
for s in "${SNAPSHOTS[@]}"; do ALL_FID_SNAPS+=("$FID/snap_$(printf '%03d' "$s")"); done

manifest_a() {   # scale_factor of snapshot $1 from the fiducial stage1 manifest
  python -c "import json,sys; print(json.load(open(sys.argv[1]))['scale_factor'])" \
      "$FID/snap_$1/stage1/stage1_manifest.json"
}

add_diffuse() {  # augment composite dir $1 -> $2 for snapshot $3 (3-digit)
  local fb_opt=()
  [[ -n "$F_B_DIFFUSE" ]] && fb_opt=(--f_b "$F_B_DIFFUSE")
  python -u -m bind.cli.paint_diffuse_composite \
      --composite_dir "$1" --output_dir "$2" \
      --scale_factor "$(manifest_a "$3")" --t_diffuse_K "$T_DIFFUSE_K" \
      --y_convention physical "${fb_opt[@]}"
}

paint_snapshot() {   # truth / twobound / sb35: recomposite this run's patches, paint planes
  local idx=$1 s3 fid_snap log src
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  fid_snap="$FID/snap_${s3}"; log="$WORK/paint_snap_${s3}.log"
  : > "$log"
  # completeness = summary.json (recomposite writes it LAST); a torn partial dir
  # from a killed attempt is wiped and redone, never fed downstream
  if [[ ! -f "$WORK/snap_${s3}/summary.json" ]]; then
    rm -rf "$WORK/snap_${s3}"
    python -u -m bind.cli.paint_recomposite \
        --stage1_dir "$fid_snap/stage1" --generated_dir "$SRC_RUN/snap_${s3}" \
        --output_dir "$WORK/snap_${s3}" >> "$log" 2>&1
  fi
  src="$WORK/snap_${s3}"
  if [[ "$DIFFUSE" == "1" ]]; then
    # all three planes come from the augmented composite: mass is MOVED
    # ch0->ch1 so the lensing planes are unchanged; y/tau gain the diffuse term
    add_diffuse "$WORK/snap_${s3}" "$WORK/snap_diff_${s3}" "$s3" >> "$log" 2>&1
    rm -rf "$WORK/snap_${s3}"
    src="$WORK/snap_diff_${s3}"
  fi
  python -u -m bind.cli.paint_lensplane \
      --generate_dir "$src" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --transforms "$FID/lightcone_transforms.json" \
      --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" \
      --all_snap_dirs "${ALL_FID_SNAPS[@]}" >> "$log" 2>&1
  python -u -m bind.cli.paint_yplane \
      --generate_dir "$src" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$src" --stage1_dir "$fid_snap/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  rm -rf "$src"
  echo "[paint] snap ${s3} done"
}

paint_snapshot_bind_diffuse() {  # bind: master composites + diffuse -> y/tau planes only
  local idx=$1 s3 log src
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  log="$WORK/paint_snap_${s3}.log"
  src="$WORK/snap_diff_${s3}"
  add_diffuse "$FID/snap_${s3}" "$src" "$s3" > "$log" 2>&1
  python -u -m bind.cli.paint_yplane \
      --generate_dir "$src" --stage1_dir "$FID/snap_${s3}/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  python -u -m bind.cli.paint_tauplane \
      --generate_dir "$src" --stage1_dir "$FID/snap_${s3}/stage1" \
      --output_dir "$LP_DIR" --lc_snap_idx "$idx" --lc_n_snaps "$N_SNAPS" \
      --lp_grid "$LP_GRID" >> "$log" 2>&1
  rm -rf "$src"
  echo "[paint] bind+diffuse snap ${s3} done"
}

paint_snapshot_dmo() {   # dmo: mass lensplanes straight from the shared stage1
  local idx=$1 s3 extra=() log
  s3=$(printf '%03d' "${SNAPSHOTS[$idx]}")
  log="$WORK/paint_snap_${s3}.log"
  [[ "$idx" -eq 0 ]] && extra=(--all_snap_dirs "${ALL_FID_SNAPS[@]}")
  python -u -m bind.cli.paint_lensplane --dmo \
      --stage1_dir "$FID/snap_${s3}/stage1" --output_dir "$LP_DIR" \
      --transforms "$FID/lightcone_transforms.json" --lc_snap_idx "$idx" \
      --lc_n_snaps "$N_SNAPS" --lp_grid "$LP_GRID" "${extra[@]}" > "$log" 2>&1
  echo "[paint] dmo snap ${s3} done"
}

if [[ "$CAT" == bind && "$DIFFUSE" != "1" ]]; then
  echo "=== [1/3] using fiducial lensplanes at $LP_DIR (no paint) ==="
  planes_complete || { echo "ERROR: fiducial lensplanes incomplete" >&2; exit 1; }
elif planes_complete; then
  echo "=== [1/3] lensplanes already complete — skipping paint ==="
else
  echo "=== [1/3] paint planes (20 snapshots, -P $PAINT_PARALLEL, DIFFUSE=$DIFFUSE) ==="
  rm -rf "$LP_DIR"; mkdir -p "$LP_DIR"      # partial paints are cheap to redo whole
  if [[ "$CAT" == bind ]]; then
    # kappa is untouched by the diffuse move -> reuse the master's mass planes
    # (+ its config.dat) via symlinks; only y/tau are painted fresh below
    ln -s "$FID"/lensplanes/lenspot*.dat "$LP_DIR"/
    [[ -f "$FID/lensplanes/config.dat" ]] || { echo "ERROR: $FID/lensplanes/config.dat missing (lux requires it)" >&2; exit 1; }
    ln -s "$FID/lensplanes/config.dat" "$LP_DIR/config.dat"
  fi
  for I in $(seq 0 $((N_SNAPS - 1))); do
    case "$CAT" in
      dmo)  paint_snapshot_dmo "$I" & ;;
      bind) paint_snapshot_bind_diffuse "$I" & ;;
      *)    paint_snapshot "$I" & ;;
    esac
    while (( $(jobs -r | wc -l) >= PAINT_PARALLEL )); do wait -n; done
  done
  wait
  planes_complete || { echo "ERROR: missing lensplanes after paint" >&2; exit 1; }
fi

# ── stage 2: lux trace in seed-offset chunks ──────────────────────────────────
mkdir -p "$RT_DIR"
expected_files() {   # per-realization completeness: config + 5 planes per enabled field
  local d=$1 p
  [[ -f "$d/config.dat" ]] || return 1
  for p in 26 45 59 70 78; do
    [[ -f "$d/kappa${p}.dat" ]] || return 1
    [[ "$COMPUTE_TSZ" == True && ! -f "$d/y${p}.dat" ]] && return 1
    [[ "$COMPUTE_TAU" == True && ! -f "$d/tau${p}.dat" ]] && return 1
  done
  return 0
}

for (( c = 0; c < NCHUNK; c++ )); do
  OFFSET=$(( c * CHUNK ))
  SENTINEL="$WORK/.chunk${c}.done"
  if [[ -f "$SENTINEL" ]]; then echo "=== [2/3] chunk $c (reals $((OFFSET+1))-$((OFFSET+CHUNK))) already done ==="; continue; fi
  CHUNK_DIR="$WORK/rt_chunk${c}"
  rm -rf "$CHUNK_DIR"; mkdir -p "$CHUNK_DIR"
  for i in $(seq 1 "$CHUNK"); do mkdir -p "$CHUNK_DIR/$(printf 'run%03d' "$i")"; done

  INI="$CHUNK_DIR/lux.ini"
  # SAFETY INVARIANT: with simulation_format=PreProjected the current lux build
  # only READS LP_output_dir/tsz_input_dir/tau_input_dir (all writes go to
  # RT_output_dir; lenspot construction is commented out in lux main.cpp).  For
  # CAT=bind, LP_DIR is the sacred fiducial lensplanes — re-verify this if lux
  # is ever rebuilt (the seed gate doubles as the regression check).
  cat > "$INI" <<EOF
LP_output_dir = $LP_DIR
RT_output_dir = $CHUNK_DIR
tsz_input_dir = $LP_DIR
tau_input_dir = $LP_DIR
input_dir = $LP_DIR
LP_grid = $LP_GRID
RT_grid = $RT_GRID
planes_per_snapshot = 4
angle = $FOV_DEG
simulation_format = PreProjected
snapshot_list = 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack = false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
output_planes = 26, 45, 59, 70, 78
projection_direction = -1
translation_rotation = False
RT_random_seed = $(( RT_SEED + 7 * OFFSET ))
RT_randomization = True
compute_tsz = $COMPUTE_TSZ
compute_tau = $COMPUTE_TAU
n_realizations = $CHUNK
verbose = True
EOF
  echo "=== [2/3] chunk $c: reals $((OFFSET+1))-$((OFFSET+CHUNK)), base seed $((RT_SEED + 7 * OFFSET)) | $(date) ==="
  srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$INI"

  for i in $(seq 1 "$CHUNK"); do
    SRC="$CHUNK_DIR/$(printf 'run%03d' "$i")"
    DST="$RT_DIR/$(printf 'run%04d' $((OFFSET + i)))"
    expected_files "$SRC" || { echo "ERROR: chunk $c realization $i incomplete after lux" >&2; exit 1; }
    rm -rf "$DST"       # leftover from an interrupted earlier attempt at this chunk
    mv "$SRC" "$DST"
  done
  rm -rf "$CHUNK_DIR"
  touch "$SENTINEL"
done

# ── stage 3: collect maps into the release tree ───────────────────────────────
# maps_complete: all expected map cubes exist with the full N_REAL realization
# axis.  Checked BEFORE requiring raw rt dirs so that a kill during/after
# lux_collect --delete_raw (valid npz on disk, raw already deleted) converges to
# .trace_complete on retry instead of wedging on the raw-count guard forever.
maps_complete() {
  python - "$OUT_RUN" "$N_REAL" "$COMPUTE_TSZ" "$COMPUTE_TAU" <<'EOF'
import sys, numpy as np
out, n, tsz, tau = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "True", sys.argv[4] == "True"
want = [("kappa_maps.npz", "kappa")] + ([("y_maps.npz", "y")] if tsz else []) \
       + ([("tau_maps.npz", "tau")] if tau else [])
for fname, key in want:
    try:
        with np.load(f"{out}/{fname}") as f:
            got = f[key].shape[0]
    except Exception:
        sys.exit(1)
    if got != n:
        print(f"[verify] {fname}: {got} reals != {n}", file=sys.stderr)
        sys.exit(1)
    print(f"[verify] {fname} n_real={got} OK")
sys.exit(0)
EOF
}

if maps_complete; then
  echo "=== [3/3] maps already collected for $OUT_RUN — skipping collect ==="
else
  echo "=== [3/3] collect -> $OUT_RUN | $(date) ==="
  N_GLOBAL=$(find "$RT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'run*' | wc -l)
  [[ "$N_GLOBAL" -eq "$N_REAL" ]] || { echo "ERROR: $N_GLOBAL/$N_REAL realizations in $RT_DIR" >&2; exit 1; }
  srun -N1 -n1 python -u -m bind.cli.lux_collect \
      --rt_root "$RT_DIR" --output_dir "$OUT_RUN" --n_real "$N_REAL" \
      --fov_deg "$FOV_DEG" --delete_raw
  maps_complete || { echo "ERROR: collected maps failed verification" >&2; exit 1; }
fi

rm -rf "$WORK"
touch "$OUT_RUN/.trace_complete"
echo "=== task $IDX ($CAT) complete | $(date) ==="
ls -la "$OUT_RUN"/*_maps.npz
