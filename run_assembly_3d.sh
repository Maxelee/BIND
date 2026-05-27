#!/bin/bash
#SBATCH --job-name=assembly_3d
#SBATCH --output=/mnt/home/mlee1/ceph/logs/assembly_3d_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/assembly_3d_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Assembly attribution: baryon residual vs DMO 3D structure (incl. formation time) over 27 CV sims.
# CPU-only (reads Subfind catalogs + our generations). Formation tracing reads many snapshots and is
# the bulk of the runtime. Adjust --partition to your CPU queue.
#   sbatch run_assembly_3d.sh
#   CV_MAX=3 sbatch run_assembly_3d.sh             # quick subset
#   NO_FORMATION=1 sbatch run_assembly_3d.sh       # structure only (fast, no snapshot tracing)

set -euo pipefail
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_bind2
export PYTHONUNBUFFERED=1

CV_MAX=${CV_MAX:-27}
M200_MIN=${M200_MIN:-1e13}
SNAP_MIN=${SNAP_MIN:-33}
EXTRA=""
[ "${NO_FORMATION:-0}" = "1" ] && EXTRA="--no-formation"

mkdir -p /mnt/home/mlee1/ceph/logs figures/scatter_diagnostics outputs/scatter_diagnostics
echo "=== Assembly 3D attribution: CV_0..$((CV_MAX-1)), M200>$M200_MIN, snap_min=$SNAP_MIN ==="
python -m scatter.assembly_3d --cv-max "$CV_MAX" --m200-min "$M200_MIN" --snap-min "$SNAP_MIN" $EXTRA
echo "=== done -> outputs/scatter_diagnostics/assembly_3d.json + figures/scatter_diagnostics/assembly_3d.{pdf,png} ==="
