#!/bin/bash
#SBATCH --job-name=bind_mapstats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_mapstats_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_mapstats_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── D3 map leg via disBatch (docs/tsz_des_data_plan.md) ─────────────────────
# Replaces the monolithic --bind_map_stats, which hit the 6h wall of job
# 2453719 at node 200/253 (~96 s/node sequential, 65 GB RSS). 253 independent
# per-node shard tasks (~2 min, ~3 GB each; idempotent — existing shards skip
# instantly, safe to resubmit), then merge + the analysis tail:
# D4 map-leg chi2 -> Fig M7 -> Capstone X.

set -uo pipefail
module load disBatch
E=/mnt/home/mlee1/BIND-ksz2/examples

disBatch -p /mnt/home/mlee1/ceph/logs/mapstats_db "$E/_mapstats_tasks.disbatch"

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
FAILED=0
run() { echo "=== $(date +%H:%M:%S) $* ==="; "$@" || { echo "!! FAILED: $*"; FAILED=1; }; }

run python "$E/des_bind_forward.py" --bind_map_stats_merge
run python "$E/des_bind_consistency.py" --maps

# revised T3 needs the fine-res T2f measurement (runs on the workstation);
# wait up to 45 min for it in case this job races ahead of it
LCDIR=/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone
for i in $(seq 45); do
  [ -f "$LCDIR/T2f_lrg_z0406_baseline.npz" ] && [ -f "$LCDIR/T2f_lrg_z0406_cib1.7.npz" ] && break
  echo "waiting for T2f fine-res npz ($i min)"; sleep 60
done
run python "$E/lightcone_m2r_ycap_real.py"      # revised T3
run python "$E/lightcone_m7_des.py"
run python "$E/lightcone_capstone_x.py"

echo "=== $(date +%H:%M:%S) MAPSTATS PIPELINE DONE (FAILED=$FAILED) ==="
exit $FAILED
