#!/bin/bash
#SBATCH --job-name=fm_testsuite_all
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_testsuite_all_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_testsuite_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
TEST_MANIFEST="$MANIFEST_DIR/sb35_test_manifest.json"

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_base}
MODEL_NAME=${MODEL_NAME:-fm_base}

SNAPSHOT=${SNAPSHOT:-90}
NPIX=${NPIX:-1024}
PATCH_PIX=${PATCH_PIX:-128}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}
N_STEPS=${N_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_WORKERS=${MAX_WORKERS:-1}
DEVICE=${DEVICE:-auto}

SB35_PARAM_FILE=${SB35_PARAM_FILE:-/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt}
SB35_DM_ROOT=${SB35_DM_ROOT:-/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/SB35}
SB35_HYDRO_ROOT=${SB35_HYDRO_ROOT:-/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35}
SB35_GROUP_ROOT=${SB35_GROUP_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_extras/L50n512/SB35}
TEST_DATA_ROOT=${TEST_DATA_ROOT:-/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test}

export SNAPSHOT
export TEST_MANIFEST
export SB35_PARAM_FILE
export SB35_DM_ROOT
export SB35_HYDRO_ROOT
export SB35_GROUP_ROOT
export TEST_DATA_ROOT

mkdir -p "$OUTPUT_ROOT" "$MANIFEST_DIR" /mnt/home/mlee1/ceph/logs

echo "=== Building SB35 test manifest ==="
/mnt/home/mlee1/venvs/torch3/bin/python - <<'PY'
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

snapshot = int(os.environ.get("SNAPSHOT", "90"))
param_file = Path(os.environ["SB35_PARAM_FILE"])
dm_root = Path(os.environ["SB35_DM_ROOT"])
hydro_root = Path(os.environ["SB35_HYDRO_ROOT"])
group_root = Path(os.environ["SB35_GROUP_ROOT"])
test_data_root = Path(os.environ["TEST_DATA_ROOT"])
manifest_path = Path(os.environ["TEST_MANIFEST"])

if not param_file.exists():
    raise SystemExit(f"Missing SB35 parameter file: {param_file}")
if not dm_root.exists():
    raise SystemExit(f"Missing SB35 DMO root: {dm_root}")
if not hydro_root.exists():
    raise SystemExit(f"Missing SB35 hydro root: {hydro_root}")
if not group_root.exists():
    raise SystemExit(f"Missing SB35 group root: {group_root}")
if not test_data_root.exists():
    raise SystemExit(f"Missing test data root: {test_data_root}")

df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)

param_map = {}
for _, row in df.iterrows():
    sim_name = str(row.iloc[0]).strip()
    if not sim_name:
        continue
    params = row.iloc[1:36].to_numpy(dtype=np.float32)
    if params.shape[0] != 35:
        continue
    param_map[sim_name] = params

dm_sims = {p.name for p in dm_root.glob("SB35_*") if p.is_dir()}
hydro_sims = {p.name for p in hydro_root.glob("SB35_*") if p.is_dir()}
group_sims = {p.name for p in group_root.glob("SB35_*") if p.is_dir()}

selected_ids = set()
for p in test_data_root.glob("sim_*"):
    if not p.is_dir():
        continue
    match = re.fullmatch(r"sim_(\d+)", p.name)
    if match:
        selected_ids.add(int(match.group(1)))

if not selected_ids:
    for cache_name in ("file_list_cache_no_lowmass.txt", "file_list_cache.txt"):
        cache_path = test_data_root / cache_name
        if not cache_path.exists():
            continue
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.search(r"sim_(\d+)", line)
                if match:
                    selected_ids.add(int(match.group(1)))

if not selected_ids:
    raise SystemExit(f"No sim_<id> entries found in test data root: {test_data_root}")

selected_sims = {f"SB35_{i}" for i in selected_ids}
available = dm_sims & hydro_sims & group_sims
common = sorted(available & selected_sims, key=lambda s: int(s.split("_")[1]))

entries = []
skipped_missing = 0
skipped_params = 0

for sim_name in common:
    params = param_map.get(sim_name)
    if params is None:
        skipped_params += 1
        continue

    nbody_path = dm_root / sim_name
    hydro_sim_root = hydro_root / sim_name
    group_sim_root = group_root / sim_name
    hydro_snapdir = hydro_sim_root / f"snapdir_{snapshot:03d}"
    group_catalog = group_sim_root / f"groups_{snapshot:03d}"

    if not nbody_path.exists() or not hydro_snapdir.exists() or not group_catalog.exists():
        skipped_missing += 1
        continue

    entries.append(
        {
            "suite": "Test",
            "sim_id": sim_name,
            "nbody_path": str(nbody_path),
            "hydro_snapdir": str(hydro_snapdir),
            "group_catalog": str(group_catalog),
            "params": params.tolist(),
        }
    )

if not entries:
    raise SystemExit("No valid SB35 entries found for manifest generation after test-split filtering")

payload = {
    "simulations": entries,
    "meta": {
        "snapshot": snapshot,
        "source": "auto-generated by run_test_suite_all.sh",
        "n_entries": len(entries),
        "test_data_root": str(test_data_root),
        "n_selected_test_sims": len(selected_sims),
        "n_common_available_selected": len(common),
        "skipped_missing_paths": skipped_missing,
        "skipped_missing_params": skipped_params,
    },
}

manifest_path.write_text(json.dumps(payload, indent=2))
print(f"Manifest written: {manifest_path}")
print(f"SB35 entries: {len(entries)}")
print(f"Selected SB35 sims from test split: {len(selected_sims)}")
print(f"Skipped missing paths: {skipped_missing}")
print(f"Skipped missing params: {skipped_params}")
PY

EXTRA_FLAGS=()
if [[ "${PREP_ONLY:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--prep_only)
fi
if [[ "${SKIP_TRUTH:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--skip_truth)
fi
if [[ "${NO_AMP:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--no_amp)
fi

echo "=== Launching test-suite run (suite=all) ==="
echo "Output root: $OUTPUT_ROOT"
echo "Run dir: $RUN_DIR"
echo "Manifest: $TEST_MANIFEST"

/mnt/home/mlee1/venvs/torch3/bin/python run_test_suite.py \
    --suite all \
    --snapshot "$SNAPSHOT" \
    --npix "$NPIX" \
    --patch_pix "$PATCH_PIX" \
    --halo_mass_min "$HALO_MASS_MIN" \
    --run_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --output_root "$OUTPUT_ROOT" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max_workers "$MAX_WORKERS" \
    --test_manifest "$TEST_MANIFEST" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "=== Completed. Summary: $OUTPUT_ROOT/run_summary_all.json ==="