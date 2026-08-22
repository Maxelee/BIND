#!/bin/bash
# Submission driver for the multi-redshift `fm_redshift` suite evaluation.
#
# WHY: the paper's Fig. 3 (fig_mass_error) medians are over CV+1P+SB35, while the
# redshift figures are currently SB35-only, so the two quote different stellar
# accuracies (+4.3% vs -9.9%) purely from suite composition. This eval produces
# CV+1P+Test at every training snapshot so the redshift figures can be recomputed
# on the same population -- and, at snapshot 90, it is also the long-planned
# `fm_redshift` z=0 spine eval that would let every paper figure come from one
# model. See docs/WORKLOG.md 2026-08-13.
#
# REQUIRES the scale_factor fix (2026-08-13): before it, this eval would have
# generated z=0 baryons at every snapshot and blamed the model (stars -39.6%
# instead of -8.5% at z=1.045).
#
# This script does NOT submit by default -- it prints the sbatch commands.
# Review them, then re-run with --submit.
#
#   ./run_zsuite_submit.sh              # dry run, prints commands
#   ./run_zsuite_submit.sh --submit     # actually submit
#
# COST (measured: 3.32 min/sim, 70 MB/sim on one A100; I/O-bound, ~47% CPU):
#   268 sims/snapshot -> ~14.8 h serial per snapshot, ~89 h over 6 snapshots
#   ~19 GB/snapshot, ~113 GB total (composite.npz is 48% of that)
# With 8-way chunking per suite the wall clock is ~1-2 h per (suite, snapshot).
set -euo pipefail

SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

REPO=/mnt/home/mlee1/vdm_bind2
cd "$REPO"

# ── The redshift-conditioned model ───────────────────────────────────────────
export RUN_DIR=/mnt/home/mlee1/ceph/fm_runs/fm_redshift
export MODEL_NAME=fm_redshift
export CHECKPOINT_PATH=$RUN_DIR/checkpoints/last.ckpt
export OUTPUT_ROOT=/mnt/home/mlee1/ceph/fm_redshift_suite
export HALO_MASS_MIN=1e13      # the trained regime the paper restricts to
export N_STEPS=20

# Snapshots with enough held-out halos to support a percentile estimate.
# 32 (z=3.01) and 24 (z=4.01) are trained but too sparse -- excluded, matching
# tools/paper_cache/make_z_figures.py MAIN_SNAPS.
SNAPSHOTS=(90 82 74 60 52 44)

# 1P stores snapdir_080, NOT snapdir_082, so it cannot contribute at z=0.209.
# Dropping it there keeps the population composition fixed across the figure,
# which is the whole point of this eval; mixing in a different epoch would not.
declare -A SKIP=( ["1p:82"]=1 )

# Chunks per suite (must match the --array upper bound).
declare -A CHUNKS=( ["cv"]=4 ["1p"]=8 ["test"]=8 )

emit () {   # suite snapshot
  local suite=$1 snap=$2 n=${CHUNKS[$1]}
  if [[ -n "${SKIP[$suite:$snap]:-}" ]]; then
    echo "#  SKIP  suite=$suite snapshot=$snap  (no hydro snapdir_0$snap for 1P)"
    return
  fi
  local cmd="SUITE=$suite SNAPSHOT=$snap N_CHUNKS=$n RUN_DIR=$RUN_DIR"
  cmd+=" MODEL_NAME=$MODEL_NAME CHECKPOINT_PATH=$CHECKPOINT_PATH"
  cmd+=" OUTPUT_ROOT=$OUTPUT_ROOT HALO_MASS_MIN=$HALO_MASS_MIN N_STEPS=$N_STEPS"
  cmd+=" sbatch --array=0-$((n-1)) --job-name=zsuite_${suite}_${snap} run_lowmass_suite.sh"
  if [[ $SUBMIT == 1 ]]; then
    echo "SUBMITTING: $cmd"; eval "$cmd"
  else
    echo "$cmd"
  fi
}

# For SUITE=test, chunk 0 builds the SB35 manifest and the other chunks wait on a
# lock. As of 2026-08-13 both the manifest and the lock are namespaced by
# snapshot (run_lowmass_suite.sh), because the manifest hardcodes
# snapdir_<SNAPSHOT> in every entry -- with the old shared filename a
# multi-snapshot run would have paired one snapshot's DMO field with another
# snapshot's hydro truth. All snapshots can therefore be submitted together.
for snap in "${SNAPSHOTS[@]}"; do
  for suite in cv 1p test; do emit "$suite" "$snap"; done
done

echo
if [[ $SUBMIT == 0 ]]; then
  echo "# Dry run -- nothing submitted. Re-run with --submit once the above looks right."
else
  echo "# Submitted. Track with:  squeue -u \$USER"
fi
echo "# Then rebuild the figures on the matched population:"
echo "#   PAPER_SUITE_ROOT=$OUTPUT_ROOT PAPER_MODEL_SUBDIR=$MODEL_NAME \\"
echo "#   PAPER_MODEL_TAG=fm_redshift python tools/paper_cache/make_z_figures.py"
