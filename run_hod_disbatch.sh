#!/bin/bash
#SBATCH --job-name=hod_stacks
#SBATCH --output=/mnt/home/mlee1/ceph/logs/hod_stacks_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/hod_stacks_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=cascadelake
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── R2 final (docs/p4c_referee_hardening_plan.md): per-node HOD-population
# y-CAP stacks (254 tasks: fid + 253 nodes; cen_anchor + kappa-calibrated
# satellite mixes at f_eff {0.04,0.08,0.12}; ~4 min / ~2GB each; idempotent).
# The fiducial satellite TEMPLATE cannot be applied across nodes (±35-39%
# node spread — gas-poor strong-feedback hosts boost less), hence per-node.
# Tail: merge -> final T3/L/X regeneration with per-node HOD models.
module load disBatch
disBatch -p /mnt/home/mlee1/ceph/logs/hod_db /mnt/home/mlee1/BIND-ksz2/examples/_hod_tasks.disbatch

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
E=/mnt/home/mlee1/BIND-ksz2/examples
FAILED=0
run() { echo "=== $(date +%H:%M:%S) $* ==="; "$@" || { echo "!! FAILED: $*"; FAILED=1; }; }
run python "$E/lightcone_hod_stack.py" --merge
run python "$E/lightcone_m2r_ycap_real.py"
run python "$E/lightcone_latent_corner.py"
run python "$E/lightcone_capstone_x.py"
echo "=== HOD PIPELINE DONE (FAILED=$FAILED) ==="
exit $FAILED
