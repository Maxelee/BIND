#!/bin/bash
#SBATCH --job-name=fmthermo_eval
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fmthermo_eval_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fmthermo_eval_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-9   # set to 0-(N_CHUNKS-1); override with --array at sbatch time

# Test-suite eval for the joint mass+thermo emulator (fm_thermo), deploying the
# EMA-baked epoch-64 checkpoint. Adapted from run_test_suite_parallel.sh with two
# changes: it passes --checkpoint_path (so the EMA weights are used, not last.ckpt)
# and it is parametrized by SUITE so you can target 1P, the SB35 test-split
# manifest, or everything.
#
# Submit examples (this script cannot sbatch itself):
#   1P (small, one task):
#     SUITE=1p   N_CHUNKS=1  sbatch --array=0     run_test_suite_thermo.sh
#   SB35 held-out test split (chunked array; chunk 0 builds the manifest):
#     SUITE=test N_CHUNKS=10 sbatch --array=0-9   run_test_suite_thermo.sh
#   Everything (CV + 1P + SB35-test) in one array:
#     SUITE=all  N_CHUNKS=10 sbatch --array=0-9   run_test_suite_thermo.sh
# Useful overrides: N_STEPS, BATCH_SIZE, REGEN=1 (force --regenerate_all),
#   PREP_ONLY=1, SKIP_TRUTH=1, CHECKPOINT_PATH=...

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

# ── Configuration ─────────────────────────────────────────────────────────────
SUITE=${SUITE:-all}                       # all | cv | 1p | test | sb35
N_CHUNKS=${N_CHUNKS:-10}                   # must match --array upper bound + 1
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_thermo}
MODEL_NAME=${MODEL_NAME:-fm_thermo_ema}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$RUN_DIR/checkpoints/kept/keep_epoch064_ema.ckpt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
TEST_MANIFEST="$MANIFEST_DIR/sb35_test_manifest.json"

SNAPSHOT=${SNAPSHOT:-90}
NPIX=${NPIX:-1024}
PATCH_PIX=${PATCH_PIX:-128}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}
N_STEPS=${N_STEPS:-20}                     # 20 matched truth to <0.03 dex in validation
BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-auto}

SB35_PARAM_FILE=${SB35_PARAM_FILE:-/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt}
SB35_DM_ROOT=${SB35_DM_ROOT:-/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35}
SB35_HYDRO_ROOT=${SB35_HYDRO_ROOT:-/mnt/ceph/users/camels/Sims/IllustrisTNG_extras/L50n512/SB35}
SB35_GROUP_ROOT=${SB35_GROUP_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35}
CV_FOF_ROOT=${CV_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/CV}
ONEP_FOF_ROOT=${ONEP_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/1P}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test}

mkdir -p "$OUTPUT_ROOT" "$MANIFEST_DIR" /mnt/home/mlee1/ceph/logs

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

# ── SB35 test manifest (only needed for SUITE in {all,test}) ──────────────────
NEED_MANIFEST=0
[[ "$SUITE" == "all" || "$SUITE" == "test" ]] && NEED_MANIFEST=1
MANIFEST_LOCK="$MANIFEST_DIR/.manifest.lock"

if [[ "$NEED_MANIFEST" == "1" ]]; then
    if [[ "$CHUNK_ID" == "0" ]]; then
        echo "=== [chunk 0] Building SB35 test manifest ==="
        rm -f "$MANIFEST_LOCK"
        export SNAPSHOT TEST_MANIFEST SB35_PARAM_FILE SB35_DM_ROOT SB35_HYDRO_ROOT SB35_GROUP_ROOT TEST_DATA_ROOT
        /mnt/home/mlee1/venvs/torch3/bin/python - <<'PY'
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

snapshot     = int(os.environ.get("SNAPSHOT", "90"))
param_file   = Path(os.environ["SB35_PARAM_FILE"])
dm_root      = Path(os.environ["SB35_DM_ROOT"])
hydro_root   = Path(os.environ["SB35_HYDRO_ROOT"])
group_root   = Path(os.environ["SB35_GROUP_ROOT"])
test_data_root = Path(os.environ["TEST_DATA_ROOT"])
manifest_path  = Path(os.environ["TEST_MANIFEST"])

df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)
param_map = {str(row.iloc[0]).strip(): row.iloc[1:36].to_numpy(dtype=np.float32)
             for _, row in df.iterrows() if str(row.iloc[0]).strip()}

dm_sims    = {p.name for p in dm_root.glob("SB35_*")    if p.is_dir()}
hydro_sims = {p.name for p in hydro_root.glob("SB35_*") if p.is_dir()}
group_sims = {p.name for p in group_root.glob("SB35_*") if p.is_dir()}

selected_ids = set()
for p in test_data_root.glob("sim_*"):
    m = re.fullmatch(r"sim_(\d+)", p.name)
    if m: selected_ids.add(int(m.group(1)))
if not selected_ids:
    for name in ("file_list_cache_no_lowmass.txt", "file_list_cache.txt"):
        cp = test_data_root / name
        if cp.exists():
            for line in cp.read_text().splitlines():
                m = re.search(r"sim_(\d+)", line)
                if m: selected_ids.add(int(m.group(1)))

available = dm_sims & hydro_sims & group_sims
common    = sorted(available & {f"SB35_{i}" for i in selected_ids},
                   key=lambda s: int(s.split("_")[1]))

entries, skipped = [], 0
for sim_name in common:
    params = param_map.get(sim_name)
    if params is None or params.shape[0] != 35: skipped += 1; continue
    hydro_snapdir = hydro_root / sim_name / f"snapdir_{snapshot:03d}"
    group_catalog = group_root / sim_name
    if not (dm_root / sim_name).exists() or not hydro_snapdir.exists() or not group_catalog.exists():
        skipped += 1; continue
    entries.append({"suite": "Test", "sim_id": sim_name,
                    "nbody_path":    str(dm_root / sim_name),
                    "hydro_snapdir": str(hydro_snapdir),
                    "group_catalog": str(group_catalog),
                    "params": params.tolist()})

manifest_path.write_text(json.dumps({"simulations": entries,
    "meta": {"snapshot": snapshot, "n_entries": len(entries), "skipped": skipped}}, indent=2))
print(f"Manifest: {len(entries)} entries, {skipped} skipped")
PY
        touch "$MANIFEST_LOCK"
    else
        echo "=== [chunk $CHUNK_ID] Waiting for manifest ==="
        for i in $(seq 1 60); do
            [[ -f "$MANIFEST_LOCK" ]] && break
            sleep 5
        done
        [[ -f "$MANIFEST_LOCK" ]] || { echo "ERROR: manifest lock not found after 5 min" >&2; exit 1; }
    fi
fi

# ── Build flags ───────────────────────────────────────────────────────────────
EXTRA_FLAGS=()
[[ "${REGEN:-0}" == "1" ]]      && EXTRA_FLAGS+=(--regenerate_all)
[[ "${PREP_ONLY:-0}" == "1" ]]  && EXTRA_FLAGS+=(--prep_only)
[[ "${SKIP_TRUTH:-0}" == "1" ]] && EXTRA_FLAGS+=(--skip_truth)
[[ "${NO_AMP:-0}" == "1" ]]     && EXTRA_FLAGS+=(--no_amp)
[[ "$NEED_MANIFEST" == "1" ]]   && EXTRA_FLAGS+=(--test_manifest "$TEST_MANIFEST")

echo "=== [chunk $CHUNK_ID/$N_CHUNKS] suite=$SUITE model=$MODEL_NAME ckpt=$CHECKPOINT_PATH ==="

/mnt/home/mlee1/venvs/torch3/bin/python run_test_suite.py \
    --suite "$SUITE" \
    --snapshot "$SNAPSHOT" \
    --npix "$NPIX" \
    --patch_pix "$PATCH_PIX" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --run_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max_workers 1 \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --cv_fof_root "$CV_FOF_ROOT" \
    --onep_fof_root "$ONEP_FOF_ROOT" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "=== [chunk $CHUNK_ID] Done ==="
