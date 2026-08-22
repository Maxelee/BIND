#!/bin/bash
#SBATCH --job-name=repaint_stats
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repaint_stats_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repaint_stats_%A_%a.err
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── FIDUCIAL REPAINT, STAGE 5 (CPU): summary statistics on the new maps ───────
# One array task per statistics variant.  Tasks 0-4 reproduce the product set the
# released fiducial carries; tasks 5-7 close the three gaps an audit found between
# "what this stage writes" and "what Paper I actually reads".
#
#   task 0  base      lightcone_stats                -> Cl_kappa, Cl_kappa_y, Cl_tau,
#                                                       dm_stats, peak_counts,
#                                                       peak_cross, nongaussian_stats,
#                                                       halo_scaling, scaling_relations
#   task 1  multi     --peaks_only --smoothing 2 1 5 8 --out_suffix _multi
#   task 2  ngal10    --peaks_only --shape_noise_ngal $NGAL      (-> *_ngal10.npz)
#   task 3  nufid     --peaks_only --nu_sigma0_from $NUFID_FROM  (-> *_nufid.npz)
#   task 4  paired    bind.cli.paired_stats --n_real $PAIRED_N_REAL
#                                                       -> paired_perreal_fid.npz
#   task 5  nu05      nu_grid.compute on kappa_maps   -> nu05_stats.npz
#   task 6  dmo-den   paired 50-real DMO denominator  -> Cl_kappa_dmo_n50.npz
#   task 7  n50       50-real sibling of a 550-real tree (skipped when N_REAL==50)
#
# Tasks are independent (they all just read kappa_maps.npz); submit as one array.
#
# WHY TASK 5 EXISTS.  nu05_stats.npz is the CANONICAL 22-point nu grid
# (dnu=0.5, -2.75..7.75) that the family-basis / fig07 machinery consumes
# (papers/01_pipeline/family_basis_all.py:115, refresh_amplitude_sets_mf22.py:32).
# It is NOT written by bind.cli.lightcone_stats — that writes nongaussian_stats.npz
# on the legacy 29-point nu=-3..4 grid.  It is written by nu_grid.compute (via
# papers/01_pipeline/remeasure_twobound_nu05.py).  Without this task a re-pointed
# fiducial would silently mix the OLD fiducial's nu05 with new everything else.
#
# WHY TASK 6 EXISTS.  S(ell) is NOT stored: bind.inference.stats only writes
# `suppression`/`cl_dmo` when a kappa_dmo cube is passed in-process (stats.py:103-112),
# and the released Cl_kappa.npz carries only ell/cl/cl_err.  Downstream code forms
# S(ell) by dividing by bind_science/runs/dmo/run_0000, whose kappa cube is
# (550, 5, 1024, 1024).  Dividing a 50-real BIND mean by a 550-real DMO mean is the
# already-documented "DMO 550-real denominator mismatch" (~3% low-ell wiggle).  This
# task writes a realization-matched denominator from the FIRST $DMO_N reals of the
# same (read-only) DMO cube, which is seed-paired with the repaint.
#
# WHY TASK 4 PASSES --n_real.  _load_or_build_fid derives the fixed-nu sigma0 from
# the whole fiducial cube.  The released convention
# (bind_science/runs/bind/run_0000/paired_perreal_fid.npz, 1.5 MB = 50 reals) used a
# 50-realization sigma; at N_REAL=550 an unqualified run would silently change that
# convention.  PAIRED_N_REAL pins it at 50 regardless of how many were traced.
#
# NOTE ON *_nufid: the fixed-nu convention normalises by ONE sigma per source
# bin taken from the fiducial.  This repaint CHANGES the fiducial, so every
# twobound/sb35 *_nufid.npz that was normalised by the old fiducial's sigma must
# be regenerated against the new one (run_nufid_restats.sh with
# FID_DIR=$REPAINT_ROOT) before any response figure is redrawn.  See
# docs/fiducial_repaint_plan.md §10.5.
#
# Submit (author only), after stage 4:
#     sbatch repaint/run_repaint_stats.sh      # only once .trace_complete exists
#
# Env: see repaint/repaint_env.sh; plus SMOOTHING, NGAL, SIGMA_E, NUFID_FROM,
#      PAIRED_N_REAL (50), DMO_RUN, DMO_N (50).

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=repaint_env.sh
source "$HERE/repaint_env.sh"

repaint_preamble
repaint_lock          # stage 5 is otherwise the ONE stage that could run against a
                      # tree assembled under a different RT_SEED/LP_GRID/RUN_DIR

SMOOTHING=${SMOOTHING:-2.0}
NGAL=${NGAL:-10}
SIGMA_E=${SIGMA_E:-0.26}
NUFID_FROM=${NUFID_FROM:-$REPAINT_ROOT}
PAIRED_N_REAL=${PAIRED_N_REAL:-50}      # released fixed-nu convention
DMO_RUN=${DMO_RUN:-/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000}   # READ-ONLY
DMO_N=${DMO_N:-50}

VAR=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID 0..7}
repaint_assert_under_root "$REPAINT_ROOT" "stats output" || exit 1

if [[ ! -f "$REPAINT_ROOT/kappa_maps.npz" ]]; then
    echo "ERROR: $REPAINT_ROOT/kappa_maps.npz missing — run run_repaint_lux.sh first" >&2
    exit 1
fi
if [[ ! -f "$REPAINT_ROOT/.trace_complete" ]]; then
    echo "ERROR: no .trace_complete marker — refusing to compute statistics on a" >&2
    echo "       partial trace (resubmit run_repaint_lux.sh until it completes)." >&2
    exit 1
fi

# Statistics must never outlive the trace they were computed from: record the
# realization count they were built at and redo them when the trace grows.
TRACE_N=$(cat "$REPAINT_ROOT/.n_real" 2>/dev/null || echo 0)
STATS_N=$(cat "$REPAINT_ROOT/.stats_n_real" 2>/dev/null || echo 0)
[[ "$TRACE_N" =~ ^[0-9]+$ ]] || TRACE_N=0
[[ "$STATS_N" =~ ^[0-9]+$ ]] || STATS_N=0
STALE=0
(( STATS_N != TRACE_N )) && STALE=1
echo "    trace n_real=$TRACE_N   stats last built at n_real=$STATS_N   stale=$STALE"

skip_if() {   # skip_if <file> -> exit 0 when present, current and FORCE != 1
    [[ "$FORCE" != "1" && "$STALE" == "0" && -f "$REPAINT_ROOT/$1" ]] && {
        echo "=== $1 already present at n_real=$TRACE_N — skipping (FORCE=1 to redo) ==="
        exit 0; }
    return 0
}

case "$VAR" in
  0) skip_if Cl_kappa.npz
     echo "=== [0] base statistics -> $REPAINT_ROOT ==="
     python -u -m bind.cli.lightcone_stats \
         --run_dir "$REPAINT_ROOT" --snap_root "$REPAINT_ROOT" \
         --smoothing_arcmin $SMOOTHING ;;
  1) skip_if peak_counts_multi.npz
     echo "=== [1] multi-scale peaks -> *_multi.npz ==="
     python -u -m bind.cli.lightcone_stats --run_dir "$REPAINT_ROOT" --peaks_only \
         --smoothing_arcmin 2 1 5 8 --out_suffix _multi ;;
  2) skip_if "peak_counts_ngal${NGAL%.*}.npz"
     echo "=== [2] shape-noise peaks (ngal=$NGAL) ==="
     python -u -m bind.cli.lightcone_stats --run_dir "$REPAINT_ROOT" --peaks_only \
         --smoothing_arcmin 2 --shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E" ;;
  3) skip_if peak_counts_nufid.npz
     echo "=== [3] fixed-nu peaks (sigma0 from $NUFID_FROM) ==="
     python -u -m bind.cli.lightcone_stats --run_dir "$REPAINT_ROOT" --peaks_only \
         --nu_sigma0_from "$NUFID_FROM" ;;
  4) skip_if paired_perreal_fid.npz
     echo "=== [4] paired per-realization cache (fiducial cube, n_real=$PAIRED_N_REAL) ==="
     python -u -m bind.cli.paired_stats \
         --run_dir "$REPAINT_ROOT" --fid_dir "$REPAINT_ROOT" --fov_deg "$FOV_DEG" \
         --n_real "$PAIRED_N_REAL" ;;
  5) skip_if nu05_stats.npz
     echo "=== [5] canonical nu05 grid (22 centres, dnu=0.5) -> nu05_stats.npz ==="
     PYTHONPATH="$BIND_REPO/papers/01_pipeline:${PYTHONPATH:-}" \
     python -u - "$REPAINT_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np
from nu_grid import compute            # papers/01_pipeline/nu_grid.py
root = Path(sys.argv[1])
res = compute(root / "kappa_maps.npz")
out = root / "nu05_stats.npz"
np.savez_compressed(out, **{k: v for k, v in res.items() if not k.endswith("_real")})
print(f"[nu05] wrote {out}  keys={sorted(k for k in res if not k.endswith('_real'))}")
PY
     ;;
  6) skip_if Cl_kappa_dmo_n50.npz
     echo "=== [6] realization-matched DMO denominator ($DMO_N reals of $DMO_RUN) ==="
     python -u - "$DMO_RUN" "$REPAINT_ROOT" "$DMO_N" "$FOV_DEG" <<'PY'
import sys
from pathlib import Path
import numpy as np
import bind.inference.stats as S
dmo_run, out_root, n, fov = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
with np.load(dmo_run / "kappa_maps.npz") as f:
    k = np.asarray(f["kappa"][:n])
    fov_file = float(f["fov_deg"])
if k.shape[0] != n:
    raise SystemExit(f"ERROR: {dmo_run} holds only {k.shape[0]} reals, need {n}")
if abs(fov_file - fov) > 1e-9:
    raise SystemExit(f"ERROR: DMO fov_deg={fov_file} != campaign {fov}")
ck = S.cl_kappa(k, fov_deg=fov_file)
dst = out_root / f"Cl_kappa_dmo_n{n}.npz"
np.savez(dst, n_real=np.array(n), source=np.array(str(dmo_run)), **ck)
print(f"[dmo] {dst}  cl{ck['cl'].shape} from the FIRST {n} realizations "
      f"(seed-paired with the repaint: ladder 1992+7r)")
PY
     ;;
  7) if [[ "$TRACE_N" -le 50 ]]; then
         echo "=== [7] n50 sibling not needed (trace is $TRACE_N reals) — skipping ==="
         exit 0
     fi
     N50="$REPAINT_ROOT/n50"
     repaint_assert_under_root "$N50" "n50 sibling" || exit 1
     [[ "$FORCE" != "1" && -f "$N50/Cl_kappa.npz" ]] && {
         echo "=== [7] n50 sibling already built — skipping ==="; exit 0; }
     [[ -d "$REPAINT_ROOT/rt_output/run0050" ]] || {
         echo "ERROR: raw rt_output was deleted (KEEP_RAW=0) — cannot build the n50 sibling" >&2
         exit 1; }
     echo "=== [7] 50-real sibling (drop-in for bind_science/runs/bind/run_0000) ==="
     mkdir -p "$N50"
     python -u -m bind.cli.lux_collect \
         --rt_root "$REPAINT_ROOT/rt_output" --output_dir "$N50" \
         --n_real 50 --fov_deg "$FOV_DEG"
     python -u -m bind.cli.lightcone_stats --run_dir "$N50" --snap_root "$REPAINT_ROOT" \
         --smoothing_arcmin $SMOOTHING
     python -u -m bind.cli.lightcone_stats --run_dir "$N50" --peaks_only \
         --smoothing_arcmin 2 1 5 8 --out_suffix _multi
     python -u -m bind.cli.lightcone_stats --run_dir "$N50" --peaks_only \
         --smoothing_arcmin 2 --shape_noise_ngal "$NGAL" --sigma_e "$SIGMA_E"
     python -u -m bind.cli.lightcone_stats --run_dir "$N50" --peaks_only \
         --nu_sigma0_from "$N50"
     python -u -m bind.cli.paired_stats --run_dir "$N50" --fid_dir "$N50" \
         --fov_deg "$FOV_DEG" --n_real 50
     PYTHONPATH="$BIND_REPO/papers/01_pipeline:${PYTHONPATH:-}" \
     python -u - "$N50" <<'PY'
import sys
from pathlib import Path
import numpy as np
from nu_grid import compute
root = Path(sys.argv[1])
res = compute(root / "kappa_maps.npz")
np.savez_compressed(root / "nu05_stats.npz",
                    **{k: v for k, v in res.items() if not k.endswith("_real")})
print(f"[nu05] wrote {root / 'nu05_stats.npz'}")
PY
     ;;
  *) echo "ERROR: unknown variant $VAR (expect 0..7)" >&2; exit 1 ;;
esac

# Task 0 owns the stats-freshness marker (it is the one that must always run).
if [[ "$VAR" == "0" ]]; then echo "$TRACE_N" > "$REPAINT_ROOT/.stats_n_real"; fi

echo "=== stats variant $VAR done ==="
ls -la "$REPAINT_ROOT"/*.npz 2>/dev/null | tail -20
