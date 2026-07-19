#!/bin/bash
#SBATCH --job-name=xpkfix_retrain
#SBATCH --output=/mnt/home/mlee1/ceph/logs/xpkfix_retrain_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/xpkfix_retrain_%j.err
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── xpkfix retrain: cross-family statsemu targets on the fixed dataset ──────
# Stage 1: 3 disBatch fits (cl_kappa_y / cl_kappa_tau / cl_yt, log
#          transform, 8-fold CV) against emulator_dataset_xpkfix.npz.
# Stage 2: merge all 23 parts -> statsemu_gp.npz (20 untouched parts are
#          bit-identical inputs).
# Stage 2b: constant-shift regression — retrained predictions must equal
#          the superseded ones x 1.201580e7 (log transform => exact).
# Stage 3: regenerate the cross products: posterior envelopes
#          (--with-gp-sigma), DES-weighted kappa-y prep, Pandey compare
#          (consumer-side XPK_CROSS_FIX now retired — data/model ratios
#          must REPRODUCE the recorded 0.37-0.62).
#
# ⛔ Submit:  sbatch /mnt/home/mlee1/vdm_bind2-paper3a/analysis/paper3a/scripts/run_xpkfix_retrain.sh

set -euo pipefail
module load gcc disBatch
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2-paper3a

WP6=/mnt/ceph/users/mlee1/paper3/A/wp6_propagation
[ -f $WP6/statsemu_gp_prexpkfix.npz ] || \
    cp $WP6/statsemu_gp.npz $WP6/statsemu_gp_prexpkfix.npz

echo "=== stage 1: 3 cross-family fits ==="
disBatch -p /mnt/home/mlee1/ceph/logs/wp6_statsemu/db_xpkfix_ \
    analysis/paper3a/scripts/xpkfix_retrain.disbatch

echo "=== stage 1b: verify the parts actually refit ==="
for T in cl_kappa_y cl_kappa_tau cl_yt; do
    if [ ! $WP6/statsemu_parts/$T.npz -nt $WP6/sb35_stats/emulator_dataset_xpkfix.npz ]; then
        echo "part $T is OLDER than the xpkfix dataset — stage 1 failed; see xpkfix task logs"
        exit 1
    fi
done

echo "=== stage 2: merge 23 parts ==="
python -u -m analysis.paper3a.emulator.statsemu merge

echo "=== stage 2b: constant-shift regression ==="
python - <<'PY'
import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from analysis.paper3a.emulator.statsemu import StatsEmulator
from analysis.paper3a.emulator import params_meta as pm
import numpy.random as npr
old = StatsEmulator.load("/mnt/ceph/users/mlee1/paper3/A/wp6_propagation/statsemu_gp_prexpkfix.npz")
new = StatsEmulator.load()
U = npr.default_rng(3).uniform(0.1, 0.9, size=(20, 30))
for t in ("cl_kappa_y", "cl_kappa_tau", "cl_yt"):
    a, b = old.predict(U, t), new.predict(U, t)
    m = np.isfinite(a) & (a > 0)
    r = np.nanmedian(b[m] / a[m])
    spread = np.nanstd(np.log(b[m] / a[m]))
    ok = abs(r / 1.201580e7 - 1) < 0.02 and spread < 0.05
    print(f"{t}: new/old = {r:.5e} (target 1.20158e7) spread {spread:.3f} -> {'PASS' if ok else 'FAIL'}")
    assert ok, f"{t} constant-shift regression FAILED"
PY

echo "=== stage 3: regenerate cross products ==="
python -u analysis/paper3a/scripts/run_a6_posterior_stats.py --with-gp-sigma
python -u analysis/paper3a/scripts/run_a6_pandey_prep.py
python -u analysis/paper3a/scripts/run_a6_pandey_compare.py

echo "DONE"
