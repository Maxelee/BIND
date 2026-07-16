#!/bin/bash
#SBATCH --job-name=bind_sph_slabs
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sph_slabs_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sph_slabs_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --array=0-19
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Spherical-BCM control tree (WL anisotropy study) ──────────────────────────
# Stage A.  Array index = snapshot.  Reads a run's per-halo composite_slab*.npz
# and writes a TRIPLE-total slab tree (each key a single-channel total-matter map):
#   composite       faithful radial-displacement BCM (triaxiality co-moves) -> kappa_sph
#   composite_mono  circular monopole-only correction                       -> kappa_mono
#   composite_bind  full BIND total                                         -> kappa_bind
#
# Build the BIND, BCM, monopole and DMO kappa on MATCHED seeds, then decompose:
#   dX_aniso = X_bind - X_sph    (feedback anisotropy a radial BCM misses)
#   X_sph  - X_mono              (triaxiality-tracing a radial BCM reproduces)
#
# Run-agnostic via RUN (fid|truth|tbNNNN): selects the source composite tree.
# Submit the chain (per run):
#   RUN=fid   jid=$(sbatch --parsable run_spherical_slabs.sh)
#   RUN=fid   sbatch --dependency=afterok:$jid run_spherical_maps.sh   # -> maps+stats
# For the parameter scan, loop RUN over twobound ids (tb0000 ... tb0059) — or just
# use the orchestrator run_wl_anisotropy.sh, which submits fid then all twobound.
#
# Env overrides: RUN, OUT_BASE, STAGE1_ROOT, R200_FACTOR, TAPER_FRAC.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RUN=${RUN:-fid}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}   # shared DMO/manifests
SCI=${SCI:-/mnt/home/mlee1/ceph/bind_science}
OUT_BASE=${OUT_BASE:-$SCI/wl_anisotropy}
R200_FACTOR=${R200_FACTOR:-4.0}
TAPER_FRAC=${TAPER_FRAC:-0.15}

# ── Resolve the source composite tree from RUN ────────────────────────────────
case "$RUN" in
    fid)        SRC_SNAP_ROOT="$STAGE1_ROOT" ;;
    truth)      SRC_SNAP_ROOT="$SCI/runs/truth/run_0000" ;;
    tb[0-9]*)   SRC_SNAP_ROOT="$SCI/runs/twobound/run_${RUN#tb}" ;;
    *)          echo "ERROR: unknown RUN='$RUN'" >&2; exit 1 ;;
esac
OUT_ROOT="$OUT_BASE/$RUN/slabs"      # triple-total: composite=BCM-warp, composite_mono, composite_bind

SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "=== spherical slabs: RUN=$RUN snapshot=$SNAPSHOT ==="
echo "    src=$SRC_SNAP_ROOT"
echo "    out=$OUT_ROOT"

if [[ ! -d "$SRC_SNAP_ROOT/snap_$(printf '%03d' "$SNAPSHOT")" ]]; then
    echo "    [skip] no source snap dir for snapshot $SNAPSHOT"
    exit 0
fi

# twobound/truth composites don't store the shared DMO background -> read it from
# the fid lightcone tree (STAGE1_ROOT). For RUN=fid the slab carries its own dmo.
python -u -m bind.cli.spherical_slabs \
    --src_snap_root "$SRC_SNAP_ROOT" \
    --out_root "$OUT_ROOT" \
    --snapshot "$SNAPSHOT" \
    --dmo_root "$STAGE1_ROOT" \
    --r200_factor "$R200_FACTOR" \
    --taper_frac "$TAPER_FRAC"

echo "=== done snapshot $SNAPSHOT ==="
