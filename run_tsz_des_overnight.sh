#!/bin/bash
#SBATCH --job-name=tsz_des_overnight
#SBATCH --output=/mnt/home/mlee1/ceph/logs/tsz_des_overnight_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/tsz_des_overnight_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── docs/tsz_des_data_plan.md — overnight completion of the remaining pipeline ──
# Everything is IDEMPOTENT: T2 engine runs skip existing npz outputs, so this
# script is safe to resubmit; it only (re)does what is missing.
#
# Sequence:
#   1. T2 mop-up: the 6 measurement runs killed by the workstation-session
#      ~10GB cgroup (baseline, cib1.7, rotated null, z0.45-0.9, EBV, cib1.0)
#      + verifies the 9 already-complete ones (instant skips).
#   2. Fig V-T2 + verdicts/T2.json  (examples/_t2_fig_verdict.py)
#   3. T3: Fig M2R money plot + act_ycap_lrg_real.npz + tsz_consistent_nodes.npz
#   4. D3 map leg: per-node full-n(z) kappa peak stats (253 nodes, ~1 h)
#   5. D4 map leg: chi2 vs the 45 DES patches
#   6. Fig M7 (Stream-D money plot)
#   7. Capstone X multi-probe figure + verdicts/X.json
#
# All outputs land under
#   /mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone/{,figs/,verdicts/}
# Review figures in the morning: VT2_lrg_measurement, M2R_ycap_real,
# M7_des_wl, X_multiprobe (.png).

set -uo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
E=/mnt/home/mlee1/BIND-ksz2/examples
FAILED=0

run() { echo "=== $(date +%H:%M:%S) $* ==="; "$@" || { echo "!! FAILED: $*"; FAILED=1; }; }

# ── 1. T2 mop-up (idempotent over the full battery) ─────────────────────────
run python "$E/act_ycap_measure.py" --mode lrg --zmin 0.4 --zmax 0.6 --grid both --map_variant baseline --out T2_lrg_z0406_baseline.npz
run python "$E/act_ycap_measure.py" --mode lrg --zmin 0.4 --zmax 0.6 --grid both --map_variant cib1.7 --out T2_lrg_z0406_cib1.7.npz
run python "$E/act_ycap_measure.py" --mode random --zmin 0.4 --zmax 0.6 --grid rap --n_boot 100 --out T2_random_null_z0406.npz
run python "$E/act_ycap_measure.py" --mode rotated --zmin 0.4 --zmax 0.6 --grid both --out T2_rotated_null_z0406.npz
run python "$E/act_ycap_measure.py" --mode lrg --zmin 0.45 --zmax 0.9 --grid both --map_variant baseline --out T2_lrg_z04509_baseline.npz
run python "$E/act_ycap_measure.py" --mode lrg --zmin 0.4 --zmax 0.6 --grid both --ebv_max 0.15 --map_variant baseline --out T2_lrg_z0406_ebv015.npz
for v in cib1.0 cib1.2 cib1.4 cib1.6 cib1.8 cib2.0 cib1.7_24 cibdBeta cibdBetadT; do
  run python "$E/act_ycap_measure.py" --mode lrg --zmin 0.4 --zmax 0.6 --grid both --map_variant "$v" --out "T2_lrg_z0406_${v}.npz"
done

# ── 2-7. analysis tail ──────────────────────────────────────────────────────
run python "$E/_t2_fig_verdict.py"
run python "$E/lightcone_m2r_ycap_real.py"
run python "$E/des_bind_forward.py" --bind_map_stats
run python "$E/des_bind_consistency.py" --maps
run python "$E/lightcone_m7_des.py"
run python "$E/lightcone_capstone_x.py"

echo "=== $(date +%H:%M:%S) OVERNIGHT PIPELINE DONE (FAILED=$FAILED) ==="
exit $FAILED
