#!/bin/bash
#SBATCH --job-name=ksz_val
#SBATCH --output=/mnt/home/mlee1/ceph/logs/ksz_val_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/ksz_val_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Paper-2 validation plots A, B, C, D, E, F, G — per-halo τ recovery /
# annular profiles / Spearman parameter sensitivity / stacked τ(M) /
# leave-one-out SBI coverage / v_los robustness / HMF coverage.
# All consume pre-existing artifacts from run_test_suite_parallel.sh, so this
# is CPU-only and short.
#
# Outputs (per model):
#   analysis_physics_cache/ksz_validation_{a,b,c,d,e,f}_<MODEL>.npz
#   analysis_physics_cache/ksz_validation_g.npz   (model-independent)
#   figures/ksz_validation_{a,b,c,d,e,f,g}_<MODEL>.pdf
#
# Examples:
#   sbatch run_ksz_validation_a.sh
#   sbatch --export=ALL,MODEL_NAME=fm_thermo,SUITES="CV 1P" run_ksz_validation_a.sh
#   bash run_ksz_validation_a.sh                                  # interactive
#   PLOTS=EF bash run_ksz_validation_a.sh                          # only E + F

set -euo pipefail

source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
mkdir -p /mnt/home/mlee1/ceph/logs

# ── Configuration (env-overridable) ──────────────────────────────────────────
TESTSUITE_ROOT=${TESTSUITE_ROOT:-/mnt/home/mlee1/ceph/fm_testsuite}
MODEL_NAME=${MODEL_NAME:-fm_two_head}
# Suite names match the directory tree produced by run_test_suite_parallel.sh
# (case-sensitive): CV, 1P, Test (SB35-holdout).
SUITES=${SUITES:-"CV 1P Test"}
SUITE_C=${SUITE_C:-Test}   # single suite for plot C (broad param variation)
HALO_MASS_MIN=${HALO_MASS_MIN:-1e13}

APERTURE=${APERTURE:-r200}          # r200 | fixed
R200_FACTOR=${R200_FACTOR:-1.0}
FIXED_R_PIX=${FIXED_R_PIX:-8.0}

BOX_SIZE=${BOX_SIZE:-50.0}
PATCH_SIZE_MPC_H=${PATCH_SIZE_MPC_H:-6.25}
HUBBLE=${HUBBLE:-0.6711}

# Annular bin edges in R/R200 for plot B
R_EDGES_B=${R_EDGES_B:-"0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0"}

# ACT-DR6-like stacked-aperture geometry for plot D
D_APERTURE=${D_APERTURE:-cap}                # disk | cap
R_AP_MPC_H=${R_AP_MPC_H:-0.5}                # aperture (or CAP-inner) radius
MASS_BINS_D=${MASS_BINS_D:-"1e13 2e13 5e13 1e14 1e15"}

# HMF coverage mass bins for plot G (model-independent)
MASS_BINS_G=${MASS_BINS_G:-"1e13 2e13 5e13 1e14 1e15"}
MIN_PER_SIM_G=${MIN_PER_SIM_G:-1.0}

# SBI-coverage (E) and v_los robustness (F) settings — both use the same
# stacked τ(M) observable as D and the Test suite as the (θ, x) training pool.
SUITES_EF=${SUITES_EF:-Test}
MASS_BINS_EF=${MASS_BINS_EF:-"1e13 2e13 5e13 1e14 1e15"}
NOISE_FRAC_EF=${NOISE_FRAC_EF:-0.05}
N_REALIZATIONS_EF=${N_REALIZATIONS_EF:-8}
RIDGE_EF=${RIDGE_EF:-1e-2}
PRIOR_STD_EF=${PRIOR_STD_EF:-3.0}
LEVEL_EF=${LEVEL_EF:-0.6827}
VLOS_SIGMAS=${VLOS_SIGMAS:-"0 0.05 0.10 0.20 0.30"}

CACHE_DIR=${CACHE_DIR:-analysis_physics_cache}
FIG_DIR=${FIG_DIR:-figures}

# Choose which plots to run: any non-empty substring match of A/B/C/D/E/F/G
PLOTS=${PLOTS:-ABCDEFG}

mkdir -p "$CACHE_DIR" "$FIG_DIR"

# ── Plot A ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *A* ]]; then
    OUT_NPZ_A=${OUT_NPZ_A:-${CACHE_DIR}/ksz_validation_a_${MODEL_NAME}.npz}
    OUT_PDF_A=${OUT_PDF_A:-${FIG_DIR}/ksz_validation_a_${MODEL_NAME}.pdf}
    echo "=== Validation A — per-halo τ recovery ==="
    python -m analysis.ksz.validation_a \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suites $SUITES --halo_mass_min "$HALO_MASS_MIN" \
        --aperture "$APERTURE" --r200_factor "$R200_FACTOR" \
        --fixed_r_pix "$FIXED_R_PIX" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --out "$OUT_NPZ_A"
    python -m analysis.ksz.plot_validation_a \
        --input "$OUT_NPZ_A" --out "$OUT_PDF_A"
fi

# ── Plot B ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *B* ]]; then
    OUT_NPZ_B=${OUT_NPZ_B:-${CACHE_DIR}/ksz_validation_b_${MODEL_NAME}.npz}
    OUT_PDF_B=${OUT_PDF_B:-${FIG_DIR}/ksz_validation_b_${MODEL_NAME}.pdf}
    echo "=== Validation B — annular τ(R/R200) profiles ==="
    python -m analysis.ksz.validation_b \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suites $SUITES --halo_mass_min "$HALO_MASS_MIN" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --r_edges $R_EDGES_B \
        --out "$OUT_NPZ_B"
    python -m analysis.ksz.plot_validation_b \
        --input "$OUT_NPZ_B" --out "$OUT_PDF_B"
fi

# ── Plot C ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *C* ]]; then
    OUT_NPZ_C=${OUT_NPZ_C:-${CACHE_DIR}/ksz_validation_c_${MODEL_NAME}.npz}
    OUT_PDF_C=${OUT_PDF_C:-${FIG_DIR}/ksz_validation_c_${MODEL_NAME}.pdf}
    echo "=== Validation C — Spearman τ-parameter sensitivity (suite=$SUITE_C) ==="
    python -m analysis.ksz.validation_c \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suite "$SUITE_C" --halo_mass_min "$HALO_MASS_MIN" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --r200_factor "$R200_FACTOR" \
        --fixed_r_pix "$FIXED_R_PIX" --out "$OUT_NPZ_C"
    python -m analysis.ksz.plot_validation_c \
        --input "$OUT_NPZ_C" --out "$OUT_PDF_C"
fi

# ── Plot D ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *D* ]]; then
    OUT_NPZ_D=${OUT_NPZ_D:-${CACHE_DIR}/ksz_validation_d_${MODEL_NAME}.npz}
    OUT_PDF_D=${OUT_PDF_D:-${FIG_DIR}/ksz_validation_d_${MODEL_NAME}.pdf}
    echo "=== Validation D — stacked τ(M) in ACT-DR6-like apertures ==="
    python -m analysis.ksz.validation_d \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suites $SUITES --halo_mass_min "$HALO_MASS_MIN" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --aperture "$D_APERTURE" \
        --r_ap_mpc_h "$R_AP_MPC_H" --mass_bins $MASS_BINS_D \
        --out "$OUT_NPZ_D"
    python -m analysis.ksz.plot_validation_d \
        --input "$OUT_NPZ_D" --out "$OUT_PDF_D"
fi

# ── Plot E ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *E* ]]; then
    OUT_NPZ_E=${OUT_NPZ_E:-${CACHE_DIR}/ksz_validation_e_${MODEL_NAME}.npz}
    OUT_PDF_E=${OUT_PDF_E:-${FIG_DIR}/ksz_validation_e_${MODEL_NAME}.pdf}
    echo "=== Validation E — SBI leave-one-out coverage ==="
    python -m analysis.ksz.validation_e \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suites $SUITES_EF --halo_mass_min "$HALO_MASS_MIN" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --aperture "$D_APERTURE" \
        --r_ap_mpc_h "$R_AP_MPC_H" --mass_bins $MASS_BINS_EF \
        --ridge "$RIDGE_EF" --prior_std "$PRIOR_STD_EF" \
        --noise_frac "$NOISE_FRAC_EF" --n_realizations "$N_REALIZATIONS_EF" \
        --level "$LEVEL_EF" --out "$OUT_NPZ_E"
    python -m analysis.ksz.plot_validation_e \
        --input "$OUT_NPZ_E" --out "$OUT_PDF_E"
fi

# ── Plot F ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *F* ]]; then
    OUT_NPZ_F=${OUT_NPZ_F:-${CACHE_DIR}/ksz_validation_f_${MODEL_NAME}.npz}
    OUT_PDF_F=${OUT_PDF_F:-${FIG_DIR}/ksz_validation_f_${MODEL_NAME}.pdf}
    echo "=== Validation F — v_los robustness ==="
    python -m analysis.ksz.validation_f \
        --testsuite_root "$TESTSUITE_ROOT" --model "$MODEL_NAME" \
        --suites $SUITES_EF --halo_mass_min "$HALO_MASS_MIN" \
        --box_size "$BOX_SIZE" --patch_size_mpc_h "$PATCH_SIZE_MPC_H" \
        --hubble "$HUBBLE" --aperture "$D_APERTURE" \
        --r_ap_mpc_h "$R_AP_MPC_H" --mass_bins $MASS_BINS_EF \
        --ridge "$RIDGE_EF" --prior_std "$PRIOR_STD_EF" \
        --noise_frac "$NOISE_FRAC_EF" --n_realizations "$N_REALIZATIONS_EF" \
        --vlos_sigmas $VLOS_SIGMAS --level "$LEVEL_EF" \
        --out "$OUT_NPZ_F"
    python -m analysis.ksz.plot_validation_f \
        --input "$OUT_NPZ_F" --out "$OUT_PDF_F"
fi

# ── Plot G ───────────────────────────────────────────────────────────────────
if [[ "$PLOTS" == *G* ]]; then
    OUT_NPZ_G=${OUT_NPZ_G:-${CACHE_DIR}/ksz_validation_g.npz}
    OUT_PDF_G=${OUT_PDF_G:-${FIG_DIR}/ksz_validation_g.pdf}
    echo "=== Validation G — HMF coverage ==="
    python -m analysis.ksz.validation_g \
        --testsuite_root "$TESTSUITE_ROOT" --suites $SUITES \
        --halo_mass_min "$HALO_MASS_MIN" --mass_bins $MASS_BINS_G \
        --out "$OUT_NPZ_G"
    python -m analysis.ksz.plot_validation_g \
        --input "$OUT_NPZ_G" --min_per_sim "$MIN_PER_SIM_G" \
        --out "$OUT_PDF_G"
fi

echo "=== Done.  PLOTS=$PLOTS ==="

