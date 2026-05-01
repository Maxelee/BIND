#!/bin/bash
# upload_to_modal.sh
#
# Uploads all files required by modal_app.py to a Modal Volume.
# Run this ONCE from the HPC (or any machine with ceph access) after:
#
#   pip install modal
#   python -m modal setup     # links your account (opens browser)
#   modal volume create halo-explorer-data
#
# Then:
#   bash upload_to_modal.sh
#
# Files are stored permanently in the Volume; you only need to re-upload
# if you update the model checkpoint or data files.

set -e

# Activate the Python venv that has the modal CLI installed
source /mnt/home/mlee1/venvs/torch3/bin/activate

VOLUME="halo-explorer-data"

# ─── Source paths (HPC) ───────────────────────────────────────────────────────
CKPT_PATH="/mnt/home/mlee1/ceph/fm_runs/fm_base/checkpoints/last.ckpt"
NORM_STATS="/mnt/home/mlee1/ceph/fm_runs/fm_base/norm_stats.npz"
CUTOUTS="/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_12/snap_090/mass_threshold_1p000e13/halo_cutouts.npz"
CATALOG="/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_12/snap_090/mass_threshold_1p000e13/halo_catalog.npz"
SB35_CSV="/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv"
CV_PARAMS="/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/CV/CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt"

# ─── Check all source files exist ─────────────────────────────────────────────
echo "Checking source files…"
for f in "$CKPT_PATH" "$NORM_STATS" "$CUTOUTS" "$CATALOG" "$SB35_CSV" "$CV_PARAMS"; do
    if [[ ! -f "$f" ]]; then
        echo "  ERROR: not found: $f"
        exit 1
    fi
    echo "  OK: $f"
done

# ─── Upload ───────────────────────────────────────────────────────────────────
echo ""
echo "Uploading to Modal Volume: $VOLUME"
echo "(The checkpoint may be several hundred MB — this can take a few minutes)"
echo ""

modal volume put "$VOLUME" "$CKPT_PATH"    last.ckpt        && echo "  ✓ last.ckpt"
modal volume put "$VOLUME" "$NORM_STATS"   norm_stats.npz   && echo "  ✓ norm_stats.npz"
modal volume put "$VOLUME" "$CUTOUTS"      halo_cutouts.npz && echo "  ✓ halo_cutouts.npz"
modal volume put "$VOLUME" "$CATALOG"      halo_catalog.npz && echo "  ✓ halo_catalog.npz"
modal volume put "$VOLUME" "$SB35_CSV"     SB35_param_minmax.csv && echo "  ✓ SB35_param_minmax.csv"
modal volume put "$VOLUME" "$CV_PARAMS"    CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt && echo "  ✓ CV params"

echo ""
echo "Upload complete. Verify with:"
echo "  modal volume ls $VOLUME"
echo ""
echo "Next step — deploy the app:"
echo "  cd /mnt/home/mlee1/vdm_bind2"
echo "  modal deploy modal_app.py"
echo ""
echo "Modal will print the public URL, e.g.:"
echo "  https://mlee1--halo-explorer-web.modal.run"
