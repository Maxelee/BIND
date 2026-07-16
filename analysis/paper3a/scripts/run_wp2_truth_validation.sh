#!/bin/bash
#SBATCH --job-name=wp2_truth_val
#SBATCH --output=/mnt/home/mlee1/ceph/logs/wp2_truth_val_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/wp2_truth_val_%A_%a.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── WP-A2 Popeye half: TNG300 truth validation (plan task 5-6) ────────────────
# One array task per validation snapshot:  0->096  1->071  2->067  3->063
# (z ~ 0.03 / 0.42 / 0.50 / 0.60 — the last three double as the z>0 thermo
# verdict, SHARED_CONTEXT caveat 6). CPU + big memory: each task streams the
# full TNG300-hydro snapshot (~600 files) into the transformed-frame slab maps
# and runs the operators truth-vs-painted. NO GPU.
#
# NO PAINT NEEDED: the painted side defaults to the EXISTING fiducial paint at
# bind_lightcone_tng (per-halo composites + co-located stage1 conditions). The
# painted-side path is smoke-verified on those real composites for all 4 snaps.
# (A fresh paint is only needed for the >=8-draw multi-sample — task 6 — which is
# OFF by default here; enable it by setting MULTISAMPLE_ROOT to an 8-draw paint.)
# Restart-safe at snapshot granularity (each writes its own summary + sigma_model).
#
# ⛔ HUMAN CHECKPOINT — Max submits:
#   mkdir -p /mnt/home/mlee1/ceph/logs
#   sbatch /mnt/home/mlee1/BIND-paper3a/analysis/paper3a/scripts/run_wp2_truth_validation.sh
#
# Memory: if 192G is tight (the CylToSph cKDTree pass over full-box gas), the
# dedicated big-mem node is `--partition=mem` (pcn-8-17). The `cca` general
# nodes have ample RAM for the projection itself. This runs in paper3b_popeye
# (CPU only — no CUDA init, so its cu130 torch is fine here, unlike the paint).

set -euo pipefail

VENV=${VENV:-/mnt/home/mlee1/venvs/paper3b_popeye}
source "$VENV/bin/activate"
mkdir -p /mnt/home/mlee1/ceph/logs

REPO=/mnt/home/mlee1/BIND-paper3a
cd "$REPO"

# Painted composites + their stage1 conditions both come from the existing
# fiducial paint (the driver's defaults); override only to point at a re-paint.
PAINTED_ROOT=${PAINTED_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
CONDITIONS_ROOT=${CONDITIONS_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
OUT_DIR=${OUT_DIR:-/mnt/home/mlee1/ceph/paper3/A/wp2_validation}

ARGS=(--array-index "$SLURM_ARRAY_TASK_ID"
      --painted-root "$PAINTED_ROOT"
      --conditions-root "$CONDITIONS_ROOT"
      --stage1-subdir stage1
      --out-dir "$OUT_DIR")
# Opt-in task 6: set MULTISAMPLE_ROOT to a dir with snap_063_s{0..7} draws.
[[ -n "${MULTISAMPLE_ROOT:-}" ]] && ARGS+=(--multisample-root "$MULTISAMPLE_ROOT")

python -u analysis/paper3a/scripts/run_wp2_truth_validation.py "${ARGS[@]}"

echo "=== WP-A2 truth validation task ${SLURM_ARRAY_TASK_ID} done ==="
