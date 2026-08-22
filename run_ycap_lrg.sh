#!/bin/bash
#SBATCH --job-name=ycap_lrg
#SBATCH --output=/mnt/home/mlee1/ceph/logs/ycap_lrg_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/ycap_lrg_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── tSZ y-CAP at the DESI-LRG host mass (kSZ paper §6d) ───────────────────────
# snap067 (z=0.5), LRG band logM200~13.18 Msun/h, mean-stack, ACT-beam 1.6'.
# One whole node; parallelises the 256-node Sobol loop with --nproc so the paper
# can show all 256 run profiles. Writes ceph/.../ksz_confront/ycap_lrg_snap067.npz.
#   sbatch run_ycap_lrg.sh
set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

NPROC=${SLURM_CPUS_ON_NODE:-32}               # whole node (exclusive) -> all cores
echo "=== ycap_lrg: snap067, 256 runs, nproc ${NPROC} ==="
python -u examples/_reduce_ycap_lrg.py --nproc "$NPROC"
echo "=== done ==="
