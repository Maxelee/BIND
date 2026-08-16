#!/bin/bash
#SBATCH --job-name=fm2h_gpu
#SBATCH --output=/mnt/home/mlee1/ceph/logs/fm2h_gpu_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/fm2h_gpu_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3
#
# fm_two_head GPU reruns for the paper (referee I-4 + I-16), split to fit a
# 4-hour allocation:
#   task 0: posterior ensemble, halos   0:100  (100 halos x 100 draws @ n_steps=50)
#   task 1: posterior ensemble, halos 100:200
#   task 2: ODE convergence ladder, CV,  10 sims (N=20/50/100/200 vs N=400 ref)
#   task 3: ODE convergence ladder, SB35, 10 sims
#
# Submit:      sbatch run_paper_gpu.sh
# Then (CPU, after all four tasks finish):
#   source /mnt/home/mlee1/venvs/torch3/bin/activate
#   cd /mnt/home/mlee1/vdm_bind2/tools/paper_cache
#   python merge_ensemble_slices.py \
#       /mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_200x100.npz \
#       /mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_slice0.npz \
#       /mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_slice1.npz
#   python analyze_posterior_ensemble.py \
#       --npz /mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_200x100.npz \
#       --fig /mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/pit_calibration_perhalo.png
#   PIT_NPZ=/mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_200x100.npz \
#       python referee_figs/f9_pit_calibration.py     # regenerates fig_pit_perhalo for the paper

set -e
source /mnt/home/mlee1/venvs/torch3/bin/activate
export PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_testsuite
export PAPER_MODEL_SUBDIR=fm_two_head
export PAPER_MASS_DIR=mass_threshold_1p000e13
export PAPER_MODEL_TAG=fm_two_head

CKPT=/mnt/home/mlee1/ceph/fm_runs/fm_two_head/checkpoints/last.ckpt
NORM=/mnt/home/mlee1/ceph/fm_runs/fm_two_head/norm_stats.npz
OUT=/mnt/home/mlee1/ceph/paper_cache/fm_two_head
TOOLS=/mnt/home/mlee1/vdm_bind2/tools/paper_cache
mkdir -p "$OUT/posterior_calibration"

case "${SLURM_ARRAY_TASK_ID}" in
  0|1)
    A=$(( SLURM_ARRAY_TASK_ID * 100 )); B=$(( A + 100 ))
    python "$TOOLS/build_posterior_ensemble.py" \
      --suite_root /mnt/home/mlee1/ceph/fm_testsuite --mass_dir mass_threshold_1p000e13 \
      --checkpoint "$CKPT" --norm_stats "$NORM" --n_steps 50 \
      --halo_slice "${A}:${B}" \
      --out "$OUT/posterior_calibration/ensemble_slice${SLURM_ARRAY_TASK_ID}.npz"
    ;;
  2)
    python "$TOOLS/ode_convergence.py" --suite CV --n-sims 10 \
      --ckpt "$CKPT" --norm "$NORM" --out "$OUT/ode_conv_cv.npz"
    ;;
  3)
    python "$TOOLS/ode_convergence.py" --suite Test --n-sims 10 \
      --ckpt "$CKPT" --norm "$NORM" --out "$OUT/ode_conv_sb35.npz"
    ;;
esac
