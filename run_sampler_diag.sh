#!/bin/bash
#SBATCH --job-name=fm_sampdiag
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm_sampdiag_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm_sampdiag_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Sampler diagnostic for the small-scale (high-k) power deficit.
# Answers: is it an under-integrated SAMPLER (fixable with more ODE steps) or a
# mean-seeking TRAINING issue (needs a spectral/adversarial loss or post-hoc
# sharpening)?  Writes sampler_diag.{png,npz} + a printed verdict.
#
#   sbatch run_sampler_diag.sh
#
# Env overrides (all optional):
#   RUN=/path/to/run_dir            # checkpoint dir (default fm_two_head)
#   SIMS=CV/sim_0,CV/sim_1,CV/sim_2 # comma-sep sims under $ROOT
#   NMAX=24      # most-massive halos per sim
#   NSAMP=8      # samples for the dispersion test
#   NSTEPS=20,50,100,200
#   OUTDIR=/mnt/home/mlee1/ceph/fm_diag

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
python sampler_diag.py
