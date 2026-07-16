#!/bin/bash
#SBATCH --job-name=fgas_cap
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fgas_cap_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fgas_cap_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Low-mass f~gas-M in the exact CAP filter (kSZ paper §6b) ───────────────────
# Array index 0 -> snap085 (BGS, z=0.18), 1 -> snap046 (ELG, z=1.16). Each task
# takes one whole node and parallelises the 256-node Sobol loop with --nproc
# (each node is independent), so a ~1.5 h serial reduction finishes in ~1-2 min.
# Writes ceph/.../ksz_confront/fgas_lowmass_snap{085,046}.npz (drop-in for the nb).
#   sbatch run_fgas_lowmass.sh
set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

SNAPS=(085 046)
IDXS=(2 12)                                   # lightcone-transforms snap index (085->2, 046->12)
IDX=${SLURM_ARRAY_TASK_ID:?run with --array=0-1 (0=BGS snap085, 1=ELG snap046)}
SNAP=${SNAPS[$IDX]}
SNAP_IDX=${IDXS[$IDX]}
NPROC=${SLURM_CPUS_ON_NODE:-32}               # whole node (exclusive) -> all cores

echo "=== fgas_lowmass CAP: snap ${SNAP} (snap_idx ${SNAP_IDX}), nproc ${NPROC} ==="
python -u examples/_reduce_fgas_lowmass.py --snap "$SNAP" --snap_idx "$SNAP_IDX" --nproc "$NPROC"
echo "=== done snap ${SNAP} ==="
