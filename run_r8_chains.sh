#!/bin/bash
#SBATCH --job-name=r8_chains
#SBATCH --output=/mnt/home/mlee1/ceph/logs/r8_chains_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/r8_chains_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── R8 chain extension (docs/p4c_referee_hardening_plan.md) ────────────────
# The R8 MCMC engine is checkpointed (emcee HDFBackend): each invocation
# CONTINUES the existing chains toward the 50-autocorrelation-time
# convergence target the workstation run could not reach (nsteps/tau 20-29).
# Four legs run concurrently; --budget is per-call seconds of sampling.
# Idempotent/resumable: safe to resubmit until verdicts/R8.json convergence
# flips to pass.
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
E=/mnt/home/mlee1/BIND-ksz2/examples/_r8_gp_mcmc.py
# R8 state now lives on ceph (engine SCRATCH patched 2026-07-27: node-local
# /tmp killed the first submission); rebuild the GP ingredients in-job if
# absent so the script is fully self-contained on any node.
ING=/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/r8_state/r8_ingredients.pkl
[ -f "$ING" ] || python $E --stage gp || { echo "GP STAGE FAILED"; exit 1; }
for leg in ksz tsz joint joint_noA2h; do
  srun --exact -n1 -c4 python $E --stage chain --leg $leg --budget 25000 &
done
wait
python $E --stage finalize
echo "R8 CHAINS DONE"
