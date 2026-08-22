#!/bin/bash
#SBATCH --job-name=repaint_generate
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repaint_generate_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repaint_generate_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── FIDUCIAL REPAINT, STAGE 2 (GPU): re-generate at the CORRECT cosmology ─────
# Identical to run_lightcone_generate.sh in SBATCH shape, checkpoint, sampler
# settings (N_STEPS=50, BATCH_SIZE=16) and CLI — with exactly two differences:
#
#   1. --params "$PARAMS_FILE"   <-- THE FIX.  The released run omitted --params,
#      so bind.cli.paint_generate fell back to the stage-1 params.npy, which
#      carried the CAMELS SB35 cosmology (paint_stages.py:460-462).  Here the
#      corrected TNG300 vector is passed explicitly AND is what the linked
#      stage-1 dir holds, so both routes agree.
#   2. --output_dir points into the NEW tree; the released tree is read-only.
#
# Per-snapshot completion guard: a snapshot whose composite slabs are already
# complete is skipped, so a resubmit after a preemption/partial failure costs
# only the missing snapshots (FORCE=1 to redo regardless).
#
# COST: ~0.43 s/halo on an A100; 33,678 halos over 20 snapshots => ~4.0-4.7 GPU-hr
# in total.  The array does NOT run 20-wide: the `gpu` QOS caps the user at
# cpu=40, gres/gpu=4 (sacctmgr show qos), so with --cpus-per-task=8 exactly four
# tasks run concurrently (4x8=32 CPU <= 40, 4 GPU <= 4) — expect ~1-2.5 h of wall,
# not the ~25 min a fully parallel array would take.  (--cpus-per-task=16 would
# have allowed only TWO concurrent tasks; the sampler is GPU-bound, not
# dataloader-bound.)  Output: ~22 GiB+ of composite_slab*.npz — the current code
# also writes a (4, npix, npix) float32 `composite_thermo` per slab that the
# released files do not have, so measure the real size after the Tier-0 run.
#
# ── BEFORE SUBMITTING: CHECK THAT AN A100 EXISTS ─────────────────────────────
#     sinfo -p gpu -O NodeList,StateLong,Gres,Reason
# pcn-16-06 is the ONLY a100 node in the gpu partition and it was
# State=IDLE+DRAIN (Reason="health cuda != 6") at design time — a `--constraint=a100`
# array would then sit PENDING forever.  Fallback, on the idle V100S nodes:
#
#     NO_AMP=1 sbatch --constraint=v100s repaint/run_repaint_generate.sh
#
# NO_AMP=1 is MANDATORY there: bind/inference/paint.py:389 hard-codes
# torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) and V100 (sm_70) has
# no hardware bf16.  Painting in fp32 is scientifically harmless (the sampler is
# unseeded, so this paint is one draw either way) but it MUST be provenance-visible:
# the script records amp/device/host into $OUTPUT_DIR/.repaint_provenance.json.
#
# Submit (author only):
#     sbatch --array=0 repaint/run_repaint_generate.sh     # Tier 0: snap_096 only
#     sbatch --array=1-19 repaint/run_repaint_generate.sh  # the rest
# Use --array=1-19 for the second submission, NOT the full 0-19: `generate_complete`
# is evaluated once at job start with no lock held, so a full array launched while
# the Tier-0 job is still running would re-paint snap_096 concurrently into the same
# composite_slab*.npz files.
#
# Env: see repaint/repaint_env.sh.  Extra knob here:
#     GENERATE_R200  0.0 (default) = reproduce the released two-step build
#                    (square taper now, circular 4xR200c in run_repaint_recomposite.sh).
#                    Set 4.0 to composite circularly in one step and skip the
#                    recomposite stage (same patches, same math, fewer writes).
#                    EITHER WAY, run_repaint_planes.sh hard-asserts that the
#                    composites on disk carry r200_factor=$R200_FACTOR before it
#                    writes a single plane.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

repaint_preamble
repaint_lock

GENERATE_R200=${GENERATE_R200:-0.0}

IDX=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..19 (or submit as an array)}
SNAP=${SNAPSHOTS[$IDX]}
SNAP3=$(printf '%03d' "$SNAP")

STAGE1_DIR="$REPAINT_ROOT/snap_${SNAP3}/stage1"
OUTPUT_DIR="$REPAINT_ROOT/snap_${SNAP3}"
repaint_guard "$OUTPUT_DIR" "generate output" || exit 1

if [[ ! -f "$STAGE1_DIR/stage1_manifest.json" ]]; then
    echo "ERROR: stage-1 manifest not found: $STAGE1_DIR/stage1_manifest.json" >&2
    echo "       Run: bash repaint/run_repaint_stage1_link.sh" >&2
    exit 1
fi
if [[ ! -f "$STAGE1_DIR/params.npy" ]]; then
    echo "ERROR: $STAGE1_DIR/params.npy missing (run make_corrected_params.py)" >&2
    exit 1
fi

# ── refuse to paint with the wrong cosmology, whatever the file says ──────────
python - "$STAGE1_DIR" "$PARAMS_FILE" <<'PY'
import json, sys
import numpy as np
s1, pf = sys.argv[1], sys.argv[2]
man = json.load(open(f"{s1}/stage1_manifest.json"))
p_stage1 = np.load(f"{s1}/params.npy").astype(np.float64).ravel()
p_cli = np.load(pf).astype(np.float64).ravel()
if not np.array_equal(p_stage1, p_cli):
    raise SystemExit(f"ERROR: {s1}/params.npy differs from {pf} — refusing to paint")
om = float(man["Omega_m"])
if abs(p_cli[0] - om) / om >= 1e-3:
    raise SystemExit(
        f"ERROR: cosmology mismatch — params Omega_m={p_cli[0]} but the snapshot "
        f"header (manifest) says {om}. This is exactly the released bug; refusing.")
print(f"[gate] cosmology OK: Omega_m={om:.4f}  Ob/Om={p_cli[6] / p_cli[0]:.6f}  "
      f"z={man['redshift']:.4f}")
PY

# ── per-snapshot completion guard ─────────────────────────────────────────────
generate_complete() {
    python - "$STAGE1_DIR" "$OUTPUT_DIR" <<'PY'
import json, sys
import numpy as np
s1, out = sys.argv[1], sys.argv[2]
man = json.load(open(f"{s1}/stage1_manifest.json"))
n_slabs, n_halos = int(man["n_slabs"]), int(man["n_halos"])
tot = 0
for si in range(n_slabs):
    try:
        with np.load(f"{out}/composite_slab{si:02d}.npz") as f:
            n = int(f["n_halos"])
            if n and "generated_patches" not in f.files:
                sys.exit(1)          # patches are required downstream
            if n and "composite" not in f.files:
                sys.exit(1)
            tot += n
    except Exception:
        sys.exit(1)
sys.exit(0 if tot == n_halos else 1)
PY
}

if [[ "$FORCE" != "1" ]] && generate_complete; then
    echo "=== snap ${SNAP3}: composites already complete — skipping (FORCE=1 to redo) ==="
    exit 0
fi

echo "=== repaint stage 2 (generate): task ${IDX} / snapshot ${SNAP3} ==="
echo "    stage1_dir=$STAGE1_DIR   (symlinked DMO conditioning, cosmology-independent)"
echo "    params=$PARAMS_FILE      (CORRECTED TNG300 vector)"
echo "    run_dir=$RUN_DIR  n_steps=$N_STEPS  batch_size=$BATCH_SIZE  r200=$GENERATE_R200"
echo "    output=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
AMP_OPT=()
[[ "$NO_AMP" == "1" ]] && AMP_OPT+=(--no_amp)
echo "    amp=$([[ "$NO_AMP" == 1 ]] && echo 'OFF (fp32)' || echo 'bf16')  host=$(hostname)"

python -u -m bind.cli.paint_generate \
    --stage1_dir "$STAGE1_DIR" \
    --params "$PARAMS_FILE" \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --taper_frac "$TAPER_FRAC" \
    --r200_factor "$GENERATE_R200" \
    "${AMP_OPT[@]}" \
    --device auto
# NOTE: per-halo generated_patches / thermo_patches MUST be saved (no
# --no_save_patches): recomposite, the y/tau planes and any future re-paste all
# consume them.  The sampler is UNSEEDED (src/bind/model.py:454) — this paint is
# a fresh replica, not a reproduction of the released one.

generate_complete || { echo "ERROR: composites incomplete after generate" >&2; exit 1; }

# provenance: which GPU / precision actually painted this snapshot.  The fp32
# fallback is harmless but must not be invisible.
python - "$OUTPUT_DIR/.repaint_provenance.json" "$NO_AMP" "$RUN_DIR" \
        "$N_STEPS" "$BATCH_SIZE" "$GENERATE_R200" "$TAPER_FRAC" <<'PY'
import json, os, platform, subprocess, sys
out, no_amp, run_dir, n_steps, bs, r200, taper = sys.argv[1:8]
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout.strip()
except Exception:
    gpu = "unknown"
json.dump({"amp": "fp32" if no_amp == "1" else "bf16", "gpu": gpu,
           "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
           "run_dir": run_dir, "n_steps": int(n_steps), "batch_size": int(bs),
           "generate_r200_factor": float(r200), "taper_frac": float(taper)},
          open(out, "w"), indent=1)
print(f"[provenance] {out}: amp={'fp32' if no_amp == '1' else 'bf16'} gpu={gpu}")
PY

touch "$OUTPUT_DIR/.generate_complete"
echo "=== snap ${SNAP3} done -> $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR"/composite_slab*.npz
