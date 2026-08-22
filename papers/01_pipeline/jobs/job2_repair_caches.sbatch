#!/bin/bash
#SBATCH --job-name=repair3_caches
#SBATCH --output=/mnt/home/mlee1/ceph/logs/repair3_caches_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/repair3_caches_%j.err
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00

# ── Job 2: nu8 shards for the 3 repaired Sobol runs + 256-run dataset reassembly ──
# Prereq: Job 1 (run_sobol_lightcone.sh --array=114,115,117) has completed — each of
# run_0114/run_0115/run_0117 under /mnt/home/mlee1/ceph/bind_sb35/runs/ must already
# have Cl_kappa.npz (and kappa/y/tau_maps.npz). This script refuses to run otherwise.
#
# WHY mf_cache.py / field_cache.py are NOT touched here (verified by reading both):
#   - mf_cache.py:   SIDES = ("bind", "truth")            -- fiducial pair only
#   - field_cache.py: TASKS = [(product, side) for side in ("bind","truth")]
# Neither iterates bind_sb35 Sobol runs at all, so the 253->256 repaint cannot
# affect either cache; only build_nu_cache.py is keyed per Sobol run.
#
# Steps: (1) compute the 3 missing nu8 shards, (2) back up + re-run
# bind-emulator-assemble to fold the 3 repaired runs into emulator_dataset_xpkfix.npz
# (this is what makes run_emulator_fit.sbatch auto-refit -- n_runs/train_idx change
# the fit's cache fingerprint), (3) reassemble + verify emulator_dataset_nu8.npz on
# the now-256-run dataset.
#
#   sbatch papers/01_pipeline/jobs/job2_repair_caches.sbatch

set -euo pipefail

source /mnt/home/mlee1/venvs/BIND_env/bin/activate
cd /mnt/home/mlee1/BIND/papers/01_pipeline
mkdir -p /mnt/home/mlee1/ceph/logs

SB35=/mnt/home/mlee1/ceph/bind_sb35
DS="$SB35/emulator_dataset_xpkfix.npz"

echo "=== [0/4] prereq check: Job 1 must have produced Cl_kappa.npz for the 3 runs ==="
for r in run_0114 run_0115 run_0117; do
  if [[ ! -f "$SB35/runs/$r/Cl_kappa.npz" ]]; then
    echo "ERROR: $SB35/runs/$r/Cl_kappa.npz missing -- Job 1 has not finished for $r" >&2
    exit 1
  fi
  echo "  OK: $r has Cl_kappa.npz"
done

echo "=== [1/4] nu8 shards for the 3 repaired runs (build_nu_cache.py --run) ==="
for r in run_0114 run_0115 run_0117; do
  python build_nu_cache.py --run "$r"
done

echo "=== [1b/4] dm_stats backfill: every run with tau_maps.npz but no dm_stats.npz ==="
# The DM panel's 177/253 gap is an interrupted 2026-06 backfill pass, not missing
# data (see audits/newimage2_memo.md): tau_maps.npz exists for all completed runs
# and dm_stats.npz is a ~50 s/run CPU post-process of it. This fills the 76 old
# gaps plus the 3 repaired runs, mirroring bind.cli.lightcone_stats exactly
# (dm = S.dm_stats(tau, fov_deg=fov); savez), so the reassembly below picks up
# t__dm_pdf__valid for the full run set and fig13's stamp updates on the next
# figure run. ~79 runs x ~50 s serial => well inside the 2.5 h ceiling.
python - <<'PY'
import numpy as np
from pathlib import Path
import bind.inference.stats as S
runs = sorted(Path("/mnt/home/mlee1/ceph/bind_sb35/runs").glob("run_*"))
todo = [r for r in runs if (r / "tau_maps.npz").exists() and not (r / "dm_stats.npz").exists()]
print(f"[dm backfill] {len(todo)} runs to fill")
for i, rd in enumerate(todo):
    tau = np.load(rd / "tau_maps.npz")["tau"]
    fov = float(np.load(rd / "kappa_maps.npz")["fov_deg"])
    dm = S.dm_stats(tau, fov_deg=fov)
    np.savez(rd / "dm_stats.npz", **dm)
    print(f"[dm backfill] {i+1}/{len(todo)} {rd.name} pdf{dm['dm_pdf'].shape}", flush=True)
PY

echo "=== [2/4] backup + reassemble the 256-run emulator dataset ==="
cp -v "$DS" "$SB35/emulator_dataset_xpkfix.bak253.npz"
bind-emulator-assemble --out "$DS"
python - <<PY
import numpy as np
d = np.load("$DS", allow_pickle=True)
n = len(d["run_ids"])
print(f"emulator_dataset_xpkfix.npz now has {n} runs")
assert n == 256, f"expected 256 runs, got {n}"
missing = sorted(set(range(256)) - {int(r) for r in d["run_ids"]})
assert not missing, f"still missing runs: {missing}"
PY

echo "=== [2b/4] re-apply the xpk cross-normalization correction ==="
# bind-emulator-assemble reads the per-run caches, which still carry the
# PRE-xpkfix cross norm (crosses ~1.2e7 low) -- the dataset-level fix that gave
# emulator_dataset_xpkfix.npz its name is NOT part of the assemble CLI. This
# bit us on 2026-08-03 (see WORKLOG 2026-08-04); apply_xpkfix.py derives the
# factor empirically from the kept bak253 backup and refuses anything
# non-deterministic.
python apply_xpkfix.py

echo "=== [3/4] reassemble + verify emulator_dataset_nu8.npz on the new 256-run dataset ==="
python build_nu_cache.py --assemble
python build_nu_cache.py --verify

echo "=== [4/4] done ==="
ls -la "$DS" "$SB35/emulator_dataset_nu8.npz"
