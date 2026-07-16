#!/bin/bash
#SBATCH --job-name=bind_sph_maps
#SBATCH --output=/mnt/home/mlee1/ceph/logs/bind_sph_maps_%A.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/bind_sph_maps_%A.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=END,FAIL

# ── Spherical-BCM: assemble kappa (BIND / BCM / monopole / DMO) + statistics ───
# Stage B (single task).  Builds four kappa lightcones on the SAME realization
# seeds so cosmic variance and shape noise cancel in the differences:
#   bind/  kappa from the full BIND total       (mass_key=composite_bind) + kappa_dmo
#   sph/   kappa from the radial-warp BCM        (mass_key=composite)       + kappa_dmo
#   mono/  kappa from the circular monopole      (mass_key=composite_mono)  + kappa_dmo
#   dmo/   kappa from the DMO background
# then runs bind-lightcone-stats (Cl, peaks/minima, PDF/moments/Minkowski) on each.
# Decomposition (dX_aniso = X_bind - X_sph; triaxiality-tracing = X_sph - X_mono)
# lives in examples/wl_anisotropy_paper.ipynb.
#
#   RUN=fid sbatch run_spherical_maps.sh        # after run_spherical_slabs.sh
#
# Env overrides: RUN, OUT_BASE, STAGE1_ROOT, SRCS_Z, NPIX, N_REAL, FOV_DEG, SEED0,
#                SMOOTH_ARCMIN, NGAL.

set -euo pipefail
source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND
mkdir -p /mnt/home/mlee1/ceph/logs

RUN=${RUN:-fid}
STAGE1_ROOT=${STAGE1_ROOT:-/mnt/home/mlee1/ceph/bind_lightcone_tng}
SCI=${SCI:-/mnt/home/mlee1/ceph/bind_science}
OUT_BASE=${OUT_BASE:-$SCI/wl_anisotropy}
SRCS_Z=${SRCS_Z:-"0.5 1.0 1.5 2.0"}
NPIX=${NPIX:-1024}
N_REAL=${N_REAL:-8}
FOV_DEG=${FOV_DEG:-5.0}
SEED0=${SEED0:-0}
SMOOTH_ARCMIN=${SMOOTH_ARCMIN:-"1.0 2.0 5.0"}
NGAL=${NGAL:-}                      # optional shape-noise gal/arcmin^2 (e.g. 30)

SLAB_ROOT="$OUT_BASE/$RUN/slabs"     # triple-total (composite/composite_mono/composite_bind)
BIND_DIR="$OUT_BASE/$RUN/bind"
SPHK_DIR="$OUT_BASE/$RUN/sph"
MONO_DIR="$OUT_BASE/$RUN/mono"
DMO_DIR="$OUT_BASE/$RUN/dmo"
DMO_SHARED=${DMO_SHARED:-$OUT_BASE/fid/dmo/kappa_maps.npz}   # kappa_dmo is run-independent
mkdir -p "$BIND_DIR" "$SPHK_DIR" "$MONO_DIR" "$DMO_DIR"

if [[ ! -f "$DMO_SHARED" ]]; then
    echo "ERROR: shared DMO kappa not found: $DMO_SHARED (build RUN=fid first)" >&2
    exit 1
fi

COMMON=(--source_redshifts $SRCS_Z --npix "$NPIX" --n_real "$N_REAL" \
        --fov_deg "$FOV_DEG" --seed0 "$SEED0" --no_y --no_tau \
        --manifest_root "$STAGE1_ROOT")

# kappa_bind, kappa_sph and kappa_mono all come from the one triple-total slab tree,
# differing only by which stored total is summed (--mass_key). kappa_dmo is identical
# across runs (same DMO + seeds) so it is reused from the fid lightcone, not recomputed.
echo "=== [1/5] BIND kappa from $SLAB_ROOT (mass_key=composite_bind) ==="
python -u -m bind.cli.lightcone_maps \
    --snap_root "$SLAB_ROOT" --output_dir "$BIND_DIR" --mass_key composite_bind "${COMMON[@]}"

echo "=== [2/5] BCM (radial-warp) kappa from $SLAB_ROOT (mass_key=composite) ==="
python -u -m bind.cli.lightcone_maps \
    --snap_root "$SLAB_ROOT" --output_dir "$SPHK_DIR" --mass_key composite "${COMMON[@]}"

echo "=== [3/5] monopole kappa from $SLAB_ROOT (mass_key=composite_mono) ==="
python -u -m bind.cli.lightcone_maps \
    --snap_root "$SLAB_ROOT" --output_dir "$MONO_DIR" --mass_key composite_mono "${COMMON[@]}"

echo "=== [4/5] attach shared kappa_dmo (from $DMO_SHARED) ==="
python - "$BIND_DIR" "$SPHK_DIR" "$MONO_DIR" "$DMO_DIR" "$DMO_SHARED" <<'PY'
import sys, numpy as np
from pathlib import Path
bind_dir, sph_dir, mono_dir, dmo_dir, dmo_shared = map(Path, sys.argv[1:6])
kd = np.load(dmo_shared)["kappa"]                    # (n_real, n_src, npix, npix)
# standalone DMO dir (so peaks/PDF/MFs run on the DMO field too)
b = np.load(bind_dir / "kappa_maps.npz")
if kd.shape != b["kappa"].shape:
    raise SystemExit(f"shape mismatch: dmo {kd.shape} vs bind {b['kappa'].shape} "
                     f"(npix/n_real/source_z must match the fid DMO run)")
meta = {k: b[k] for k in ("source_redshifts", "fov_deg", "npix", "n_real")}
np.savez_compressed(dmo_dir / "kappa_maps.npz", kappa=kd, **meta)
# give bind/sph/mono dirs kappa_dmo so their Cl suppression S = cl/cl_dmo is defined
for dd in (bind_dir, sph_dir, mono_dir):
    s = dict(np.load(dd / "kappa_maps.npz")); s["kappa_dmo"] = kd
    np.savez_compressed(dd / "kappa_maps.npz", **s)
print(f"[dmo] attached shared kappa_dmo {kd.shape} -> dmo/, bind/, sph/, mono/")
PY

echo "=== [5/5] statistics on bind / sph / mono / dmo ==="
# nu (peaks/minima) must share ONE sigma0 across the fields or the bins are not
# comparable. BIND defines the reference; ALL fields use --nu_norm fixed from it,
# so every field writes peak_counts_nufid.npz with identical sigma0.
STAT_ARGS=(--smoothing_arcmin $SMOOTH_ARCMIN --no_halo_scaling
           --nu_norm fixed --nu_sigma0_from "$BIND_DIR")
[[ -n "$NGAL" ]] && STAT_ARGS+=(--shape_noise_ngal "$NGAL")

for d in "$BIND_DIR" "$SPHK_DIR" "$MONO_DIR" "$DMO_DIR"; do
    echo "--- stats: $d (nu fixed from BIND) ---"
    python -u -m bind.cli.lightcone_stats --run_dir "$d" "${STAT_ARGS[@]}"
done

echo "=== spherical maps+stats done for RUN=$RUN ==="
echo "    -> $OUT_BASE/$RUN/{bind,sph,mono,dmo}/{Cl_kappa,peak_counts_nufid,nongaussian_stats}.npz"
