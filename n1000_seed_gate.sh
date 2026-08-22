#!/bin/bash
#SBATCH --job-name=bind_n1000_gate
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_n1000_gate_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_n1000_gate_%j.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── N1000 seed gate: prove the seed-offset "append" trick before the campaign ─
# The campaign traces realizations in chunks whose lux base seed is
# RT_SEED + 7*OFFSET, relying on the fact (verified in lux main.cpp) that
# realization i draws its randomization from a FRESH RNG seeded with
# base + 7*i and nothing else.  This job proves it end-to-end: trace 2
# realizations with base seed 1992 + 7*50 = 2342 (kappa-only, ~10 min) and
# compare their config.dat (Np, Ns, a[], chi[], rot[], disp[] — no paths, no
# timestamps) byte-for-byte against realizations 51 and 52 of the retained
# 550-real master trace at $FID/rt_output.  PASS = chunking preserves the
# canonical 1992+7r ladder exactly; the campaign is safe to launch.
# The rest of this file makes no changes outside $OUT_ROOT/.seed_gate.

set -euo pipefail

FID=${FID:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
OUT_ROOT=${OUT_ROOT:-/mnt/home/mlee1/ceph/bind_n1000}
LUX_BIN=${LUX_BIN:-/mnt/home/mlee1/lux/lux}
GATE="$OUT_ROOT/.seed_gate"
RT_DIR="$GATE/rt"
OFFSET=50

module restore lux
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs
# Pin the GSL default generator (n1000_body.sh does the same) — with these unset,
# one gate environment certifies both lanes: the draws are integer-only mt19937 on
# rank 0 and broadcast, so neither node type nor rank count (64 here vs 192 in
# production) can change config.dat.  Optional belt-and-braces second gate:
#   sbatch --partition=preempt --qos=preempt --constraint="cascadelake|skylake" n1000_seed_gate.sh
unset GSL_RNG_TYPE GSL_RNG_SEED || true
rm -rf "$GATE"; mkdir -p "$RT_DIR/run001" "$RT_DIR/run002"

cat > "$GATE/lux.ini" <<EOF
LP_output_dir = $FID/lensplanes
RT_output_dir = $RT_DIR
tsz_input_dir = $FID/lensplanes
tau_input_dir = $FID/lensplanes
input_dir = $FID/lensplanes
LP_grid = 4096
RT_grid = 1024
planes_per_snapshot = 4
angle = 5.0
simulation_format = PreProjected
snapshot_list = 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack = false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
output_planes = 26, 45, 59, 70, 78
projection_direction = -1
translation_rotation = False
RT_random_seed = $(( 1992 + 7 * OFFSET ))
RT_randomization = True
compute_tsz = False
compute_tau = False
n_realizations = 2
verbose = True
EOF

echo "=== seed gate: tracing reals ${OFFSET}+1..${OFFSET}+2 with base seed $((1992 + 7 * OFFSET)) ==="
srun -n "${SLURM_NTASKS}" "$LUX_BIN" "$GATE/lux.ini"

PASS=1
for i in 1 2; do
  NEW="$RT_DIR/run00${i}/config.dat"
  REF="$FID/rt_output/run0$((OFFSET + i))/config.dat"
  if cmp -s "$NEW" "$REF"; then
    echo "PASS: realization $((OFFSET + i)) config.dat is byte-identical to the master trace"
  else
    echo "FAIL: config.dat mismatch for realization $((OFFSET + i)) ($NEW vs $REF)"
    cmp "$NEW" "$REF" | head -3 || true
    PASS=0
  fi
done

# informative (non-gating): kappa map agreement — tiny FP differences from a
# different MPI decomposition are acceptable, geometry (above) is the contract
python - "$RT_DIR/run001/kappa78.dat" "$FID/rt_output/run0$((OFFSET + 1))/kappa78.dat" <<'EOF' || true
import sys
from bind.inference.lux_io import read_lux_map
import numpy as np
a, b = read_lux_map(sys.argv[1]), read_lux_map(sys.argv[2])
d = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)
print(f"[info] kappa78 max relative deviation vs master real 51: {d:.3e} "
      f"({'bit-identical' if d == 0 else 'FP-level' if d < 1e-6 else 'INVESTIGATE'})")
EOF

if [[ "$PASS" == 1 ]]; then
  rm -rf "$RT_DIR"
  echo "=== SEED GATE PASSED — campaign lanes are safe to submit ==="
else
  echo "=== SEED GATE FAILED — do NOT submit the campaign; investigate first ===" >&2
  echo "    (mismatching outputs kept under $RT_DIR for inspection)" >&2
  exit 1
fi
