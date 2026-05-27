#!/bin/bash
#SBATCH --job-name=fm_testsuite_cube
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_testsuite_cube_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_testsuite_cube_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-9   # set to 0-(N_CHUNKS-1); override with --array at sbatch time

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

# ── Configuration ─────────────────────────────────────────────────────────────
# N_CHUNKS must match the --array upper bound + 1 (e.g. --array=0-9 → N_CHUNKS=10)
N_CHUNKS=${N_CHUNKS:-10}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite_cube}
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
TEST_MANIFEST="$MANIFEST_DIR/sb35_test_manifest_cube.json"

RUN_DIR=${RUN_DIR:-/mnt/home/mlee1/ceph/fm_runs/fm_cube_two_head}
MODEL_NAME=${MODEL_NAME:-fm_cube_two_head}

SNAPSHOT=${SNAPSHOT:-90}
NPIX=${NPIX:-1024}       # 3D voxelization resolution (and 2D full-box resolution)
PATCH_PIX=${PATCH_PIX:-128}
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}
N_STEPS=${N_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-auto}

SB35_PARAM_FILE=${SB35_PARAM_FILE:-/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt}
SB35_DM_ROOT=${SB35_DM_ROOT:-/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35}
SB35_HYDRO_ROOT=${SB35_HYDRO_ROOT:-/mnt/ceph/users/camels/Sims/IllustrisTNG_extras/L50n512/SB35}
SB35_GROUP_ROOT=${SB35_GROUP_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35}
CV_FOF_ROOT=${CV_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/CV}
ONEP_FOF_ROOT=${ONEP_FOF_ROOT:-/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/1P}

# Cube test-data root — used to determine which SB35 sims belong to the test split.
CUBE_TEST_ROOT=${CUBE_TEST_ROOT:-/mnt/home/mlee1/ceph/diffusion_new/test}

mkdir -p "$OUTPUT_ROOT" "$MANIFEST_DIR" /mnt/home/mlee1/ceph/logs

# ── Manifest generation (task 0 builds it; others wait) ──────────────────────
MANIFEST_LOCK="$MANIFEST_DIR/.manifest_cube.lock"

if [[ "$CHUNK_ID" == "0" ]]; then
    echo "=== [chunk 0] Building SB35 cube test manifest ==="
    export SNAPSHOT TEST_MANIFEST SB35_PARAM_FILE SB35_DM_ROOT SB35_HYDRO_ROOT SB35_GROUP_ROOT CUBE_TEST_ROOT

    /mnt/home/mlee1/venvs/torch3/bin/python - <<'PY'
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

snapshot       = int(os.environ.get("SNAPSHOT", "90"))
param_file     = Path(os.environ["SB35_PARAM_FILE"])
dm_root        = Path(os.environ["SB35_DM_ROOT"])
hydro_root     = Path(os.environ["SB35_HYDRO_ROOT"])
group_root     = Path(os.environ["SB35_GROUP_ROOT"])
cube_test_root = Path(os.environ["CUBE_TEST_ROOT"])
manifest_path  = Path(os.environ["TEST_MANIFEST"])

df = pd.read_csv(param_file, sep=r"\s+", comment="#", header=None, skiprows=1)
param_map = {str(row.iloc[0]).strip(): row.iloc[1:36].to_numpy(dtype=np.float32)
             for _, row in df.iterrows() if str(row.iloc[0]).strip()}

dm_sims    = {p.name for p in dm_root.glob("SB35_*")    if p.is_dir()}
hydro_sims = {p.name for p in hydro_root.glob("SB35_*") if p.is_dir()}
group_sims = {p.name for p in group_root.glob("SB35_*") if p.is_dir()}

# Collect sim IDs from the cube test-data split.
selected_ids = set()
for p in cube_test_root.glob("sim_*"):
    m = re.fullmatch(r"sim_(\d+)", p.name)
    if m:
        selected_ids.add(int(m.group(1)))
# Fallback: use any file_list_cache inside the cube test split.
if not selected_ids:
    for name in ("file_list_cache.txt", "file_list_cache_no_lowmass.txt"):
        cp = cube_test_root / name
        if cp.exists():
            for line in cp.read_text().splitlines():
                m = re.search(r"sim_(\d+)", line)
                if m:
                    selected_ids.add(int(m.group(1)))

available = dm_sims & hydro_sims & group_sims
common    = sorted(available & {f"SB35_{i}" for i in selected_ids},
                   key=lambda s: int(s.split("_")[1]))

entries, skipped = [], 0
for sim_name in common:
    params = param_map.get(sim_name)
    if params is None or params.shape[0] != 35:
        skipped += 1
        continue
    hydro_snapdir = hydro_root / sim_name / f"snapdir_{snapshot:03d}"
    group_catalog = group_root / sim_name
    if not (dm_root / sim_name).exists() or not hydro_snapdir.exists() or not group_catalog.exists():
        skipped += 1
        continue
    entries.append({"suite": "Test", "sim_id": sim_name,
                    "nbody_path":    str(dm_root / sim_name),
                    "hydro_snapdir": str(hydro_snapdir),
                    "group_catalog": str(group_catalog),
                    "params": params.tolist()})

manifest_path.write_text(json.dumps({"simulations": entries,
    "meta": {"snapshot": snapshot, "n_entries": len(entries), "skipped": skipped}}, indent=2))
print(f"Cube manifest: {len(entries)} entries, {skipped} skipped")
PY
    touch "$MANIFEST_LOCK"
else
    # Wait up to 5 minutes for task 0 to finish the manifest
    echo "=== [chunk $CHUNK_ID] Waiting for manifest ==="
    for i in $(seq 1 60); do
        [[ -f "$MANIFEST_LOCK" ]] && break
        sleep 5
    done
    if [[ ! -f "$MANIFEST_LOCK" ]]; then
        echo "ERROR: manifest lock not found after 5 min" >&2
        exit 1
    fi
fi

# ── Build extra flags ─────────────────────────────────────────────────────────
EXTRA_FLAGS=(--regenerate)
[[ "${PREP_ONLY:-0}" == "1" ]]  && EXTRA_FLAGS+=(--prep_only)
[[ "${SKIP_TRUTH:-0}" == "1" ]] && EXTRA_FLAGS+=(--skip_truth)
[[ "${NO_AMP:-0}" == "1" ]]     && EXTRA_FLAGS+=(--no_amp)

echo "=== [chunk $CHUNK_ID/$N_CHUNKS] Running cube generation ==="

# Note: --npix controls both the 3D voxelization grid (1024^3) and the full-box
# 2D DMO/truth map resolution. The cube model cutout procedure:
#   1. Voxelize full DMO box to NPIX^3 using MASL CIC
#   2. Extract PATCH_PIX^3 sub-cube centred on each halo (periodic BC)
#   3. Sum along z → PATCH_PIX×PATCH_PIX condition map
# This matches exactly how the cube training data (diffusion_new) was generated.

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
    --max_workers 1 \
    --test_manifest "$TEST_MANIFEST" \
    --n_chunks "$N_CHUNKS" \
    --chunk_id "$CHUNK_ID" \
    --cv_fof_root "$CV_FOF_ROOT" \
    --onep_fof_root "$ONEP_FOF_ROOT" \
    "${EXTRA_FLAGS[@]}" \
    "$@"

echo "=== [chunk $CHUNK_ID] Done ==="
