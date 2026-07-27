#!/bin/bash
# P5 sweep (P4b plan): CAP-stack shards for all 256 Sobol runs on the local
# workstation (no SLURM; ~12 s/combo measured in P4 -> ~35 min at 12 workers).
# BGS-only pass; ELG shards are produced after P2 (haloplane positions).
# Idempotent: lightcone_cap_stack.py skips existing shards. Re-run to resume.
set -u
PY=/mnt/home/mlee1/venvs/BIND_env/bin/python
CLI=/mnt/home/mlee1/BIND-ksz2/examples/lightcone_cap_stack.py
OUT=/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/shards
NPAR=${NPAR:-12}
LOG=${LOG:-/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/shards/_sweep.log}

combo() {  # run map sample beam src
  local r=$1 m=$2 s=$3 b=$4 k=$5
  if [[ "$b" == "none" ]]; then
    $PY "$CLI" --run "$r" --map "$m" --sample "$s" --src_idx "$k" --out_dir "$OUT" >>"$LOG" 2>&1
  else
    $PY "$CLI" --run "$r" --map "$m" --sample "$s" --src_idx "$k" \
        --beam_fwhm_arcmin "$b" --out_dir "$OUT" >>"$LOG" 2>&1
  fi
}

for RUN in $(seq 0 255); do
  for spec in "tau bgs110 none 4" "tau bgs1125 none 4" \
              "y bgs110 none 4"  "y bgs1125 none 4" \
              "y bgs110 1.6 4"   "y bgs1125 1.6 4" \
              "kappa bgs110 none 1" "kappa bgs1125 none 1"; do
    set -- $spec
    combo "$RUN" "$1" "$2" "$3" "$4" &
    while (( $(jobs -r | wc -l) >= NPAR )); do wait -n; done
  done
done
wait
echo "P5 sweep complete: $(ls "$OUT" | grep -c 'run[0-9]') run-shards" | tee -a "$LOG"
