#!/bin/bash
#SBATCH --job-name=scatter_decomp
#SBATCH --output=/mnt/home/mlee1/ceph/logs/scatter_decomp_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/scatter_decomp_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Program 1: decompose group-scale observable scatter into
#   assembly (between halos) vs physics (between theta) vs intrinsic (model noise).
#
# Smoke-test defaults (SN + AGN axes, 30 halos, K=15, 5 levels) ≈ minutes on one H100.
# Scale up for the production ensemble by raising --levels / --n-halos / --k.
#
# Modes:
#   axes      — SN/AGN line sweeps + paired-counterfactual cube (fast smoke test)
#   per-param — sweep each of the 30 astro knobs individually (1P-style marginal sensitivity)
#   joint     — Sobol design over all 30 astro knobs at once (interactions + Sobol indices)
# Cosmology (5 params) is held at fiducial: varying it with a fixed DMO field is out-of-distribution.
#
# Submit:
#   sbatch run_scatter_decomposition.sh                                  # axes smoke test
#   MODE=joint N_DESIGN=128 N_HALOS=40 K=12 sbatch run_scatter_decomposition.sh   # joint Sobol
#   MODE=per-param LEVELS=5 sbatch run_scatter_decomposition.sh          # per-parameter scan

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2

export PYTHONUNBUFFERED=1

# ── Configuration (env-overridable) ──────────────────────────────────────────
MODE=${MODE:-axes}            # axes | per-param | joint
SIM=${SIM:-1P_p1_0}           # fixed-halo source (CAMELS fiducial)
N_HALOS=${N_HALOS:-30}
K=${K:-15}                     # noise draws per (halo, theta)
LEVELS=${LEVELS:-5}            # theta levels per param (axes / per-param modes)
N_DESIGN=${N_DESIGN:-128}      # Sobol design points (joint mode; rounded to power of 2)
LO=${LO:-0.15}                 # min normalized level
HI=${HI:-0.85}                 # max normalized level
N_STEPS=${N_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
N_BOOT=${N_BOOT:-200}          # bootstrap resamples over halos for CIs (0 = off)
AXES=${AXES:-"SN AGN"}         # subset of {SN, AGN} (axes mode only)
INCLUDE_COSMO=${INCLUDE_COSMO:-0}   # 1 = also vary cosmology (UNPHYSICAL with fixed DMO)

COSMO_FLAG=""
[ "$INCLUDE_COSMO" = "1" ] && COSMO_FLAG="--include-cosmo"

mkdir -p /mnt/home/mlee1/ceph/logs \
         figures/scatter_diagnostics \
         outputs/scatter_diagnostics

echo "=== Scatter decomposition: mode=$MODE sim=$SIM N_halos=$N_HALOS K=$K levels=$LEVELS n_design=$N_DESIGN n_boot=$N_BOOT ==="
python -m scatter.scatter_decomposition \
    --mode       "$MODE" \
    --sim        "$SIM" \
    --n-halos    "$N_HALOS" \
    --k          "$K" \
    --levels     "$LEVELS" \
    --n-design   "$N_DESIGN" \
    --lo         "$LO" \
    --hi         "$HI" \
    --n-steps    "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --n-boot     "$N_BOOT" \
    --axes       $AXES \
    $COSMO_FLAG

echo "=== Scatter decomposition complete ==="
echo "Outputs (mode-dependent):"
echo "  axes      -> scatter_decomposition.json + _cube.npz"
echo "  per-param -> scatter_decomposition_perparam.{json,npz} + _sensitivity.{pdf,png}"
echo "  joint     -> scatter_decomposition_joint.{json,npz} + _joint_sobol.{pdf,png}"
