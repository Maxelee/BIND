#!/bin/bash
#SBATCH --job-name=val_vdm
#SBATCH --output=/mnt/home/mlee1/ceph/logs/val_vdm_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/val_vdm_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Compare the in-progress VDM checkpoint vs flow matching, per channel, vs truth:
# integrated mass, radial profiles, power spectra. VDM uses many more sampling
# steps (DDIM). Read-only; safe to run repeatedly while the VDM trains.
#   sbatch run_validate_vdm.sh
#   VDM_STEPS=500 FM_CKPT=epoch047-val_loss0.2138.ckpt N_TEST=300 sbatch run_validate_vdm.sh

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
export LD_LIBRARY_PATH=/mnt/sw/nix/store/1fycpximqhwabw5a9z5myspx61lpx8gi-gcc-11.5.0/lib64:${LD_LIBRARY_PATH:-}
cd /mnt/home/mlee1/vdm_bind2

python validate_vdm.py \
    --n_test "${N_TEST:-200}" \
    --fm_steps "${FM_STEPS:-50}" \
    --vdm_steps "${VDM_STEPS:-250}" \
    --fm_ckpt "${FM_CKPT:-last.ckpt}" \
    --vdm_ckpt "${VDM_CKPT:-last.ckpt}" \
    --out "${OUT:-ceph/fm_diag/vdm_vs_fm.png}"
