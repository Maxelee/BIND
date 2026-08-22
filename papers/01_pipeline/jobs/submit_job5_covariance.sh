#!/bin/bash
# ── Job 5 convenience submitter — 550-realization covariance sets ─────────────
# NOT executed automatically by anyone but you. Read jobs/README.md Job 5 first.
# This just saves re-typing the four sbatch invocations and chains the
# truth replane -> truth retrace -> truth stats dependency with
# --dependency=afterok so you can fire it once and walk away.
#
# Usage:
#   cd /mnt/home/mlee1/BIND
#   bash papers/01_pipeline/jobs/submit_job5_covariance.sh            # fid+truth+dmo
#   bash papers/01_pipeline/jobs/submit_job5_covariance.sh fid        # just fiducial
#   bash papers/01_pipeline/jobs/submit_job5_covariance.sh truth
#   bash papers/01_pipeline/jobs/submit_job5_covariance.sh dmo
#
# Seed convention: RT_random_seed stays at its default 1992 (1992+7r per
# realization) in every sub-job below -- do NOT override it, or the fid/truth/dmo
# realizations stop being paired.

set -euo pipefail
cd /mnt/home/mlee1/BIND

TARGET="${1:-all}"
CEPH=/mnt/home/mlee1/ceph
SCI="$CEPH/bind_science"
FID_LC="$CEPH/bind_lightcone_tng"
TIME_LUX=${TIME_LUX:-3-00:00:00}     # 72h ceiling for the 550-real lux trace (partition MaxTime=7-00:00:00)
TIME_STATS=${TIME_STATS:-12:00:00}   # generous ceiling for collect+stats at 550 real
N_REAL=${N_REAL:-550}

run_fid() {
  echo "[5b] fiducial: fresh ${N_REAL}-realization kappa+y+tau retrace ($FID_LC)"
  J=$(N_REAL="$N_REAL" sbatch --parsable --time="$TIME_LUX" run_lightcone_tau.sh)
  echo "     submitted job $J"
}

run_truth() {
  echo "[5a] truth: one-time replane (recomposite + lenspot/yplane/tauplane)"
  JA=$(DESIGN=truth OUTPUT_ROOT="$SCI" STAGE1_ROOT="$FID_LC" \
       sbatch --parsable --array=0-0 run_sobol_paste.sh)
  echo "     submitted job $JA"

  echo "[5c] truth: fresh ${N_REAL}-realization kappa+y+tau retrace (waits on 5a)"
  JC=$(DESIGN=truth OUTPUT_ROOT="$SCI" N_REAL="$N_REAL" \
       sbatch --parsable --dependency=afterok:"$JA" --time="$TIME_LUX" --array=0-0 run_sobol_lux.sh)
  echo "     submitted job $JC (dependency afterok:$JA)"

  JC2=$(DESIGN=truth OUTPUT_ROOT="$SCI" N_REAL="$N_REAL" \
        sbatch --parsable --dependency=afterok:"$JC" --time="$TIME_STATS" --array=0-0 run_sobol_stats.sh)
  echo "     submitted job $JC2 (dependency afterok:$JC)"
}

run_dmo() {
  echo "[5d] dmo: fresh ${N_REAL}-realization kappa-only retrace (lensplanes already built, no replane needed)"
  JD=$(COMPUTE_TSZ=False DESIGN=dmo OUTPUT_ROOT="$SCI" N_REAL="$N_REAL" \
       sbatch --parsable --time="$TIME_LUX" --array=0-0 run_sobol_lux.sh)
  echo "     submitted job $JD"

  JD2=$(DESIGN=dmo OUTPUT_ROOT="$SCI" N_REAL="$N_REAL" \
        sbatch --parsable --dependency=afterok:"$JD" --time="$TIME_STATS" --array=0-0 run_sobol_stats.sh)
  echo "     submitted job $JD2 (dependency afterok:$JD)"
}

case "$TARGET" in
  all)   run_fid; run_truth; run_dmo ;;
  fid)   run_fid ;;
  truth) run_truth ;;
  dmo)   run_dmo ;;
  *) echo "usage: $0 [all|fid|truth|dmo]" >&2; exit 1 ;;
esac
