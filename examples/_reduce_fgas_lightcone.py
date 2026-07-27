"""_reduce_fgas_lightcone.py -- lightcone-native f~gas(M) (kSZ paper Task D).

Builds a *self-consistent* pair of projected sky maps -- kSZ electron column
tau (kSZ) and RAW total-matter column Sigma_tot (DM+Gas+Star) -- on the SAME
periodic-tiling geometry (mirroring `bind.inference.lightcone_maps.assemble_
lightcone`'s Born-approximation tau block, extended with a second, matching
"total mass" accumulator), then CAP-stacks the same >=1e12 FoF halos used by
`_reduce_fgas_lowmass.py` directly on those MAPS instead of on individual
per-halo painted patches.

Why a *new* map pair instead of reusing the cached tau_maps.npz/y_maps.npz:
those are produced by a different, lux-raytraced pipeline (run_sobol_light-
cone.sh) whose ray-pixel geometry is opaque outside the compiled `lux`
binary. `assemble_lightcone`'s own simpler "Born approximation" path (no lux;
exact to first order for tau/y per its docstring) is fully specified in pure
Python, so we build our OWN tau + Sigma_tot pair with a chosen, reproducible
`seed`, and derive halo pixel positions analytically from the SAME periodic-
bilinear + per-snapshot-random-shift recipe (verified against
`bind/inference/lightcone_maps.py`).

Per-run pipeline:
  1. (Sobol runs only; the fiducial/truth already carry a full grid) recompo-
     site this run's saved `generated_patches`/`thermo_patches` onto the
     shared fiducial DMO background for every needed snapshot, via
     `bind.inference.paint_stages.recomposite_from_saved` (CPU, no GPU, no
     model -- just re-pasting already-painted patches).
  2. Assemble the tau + Sigma_tot maps (`assemble_gas_and_mass`), accumulating
     every plane nearer than a chosen source redshift with the SAME
     per-snapshot random transverse shift, seeded and hence reproducible.
  3. Cross-match the FoF (>=1e12) catalog to the lightcone geometry (the SAME
     `LightconeTransforms` + slab assignment as `_reduce_fgas_lowmass.py`),
     find every PERIODIC TILE COPY of each halo's pixel position on the map
     (the box is smaller than the field of view at these comoving depths, so
     the periodic tiling repeats each halo's patch of sky several times), and
     CAP-stack (disk r<r200 minus equal-area ring r200<r<sqrt2 r200) tau and
     Sigma_tot at every valid copy, averaging the CAP sums across copies.
  4. f~gas_lightcone(M) = median_h [ CAP_tau(h)/(SIGMA_T*X_E_PER_MASS) ] /
                          median_h [ CAP_sigmatot(h) ]        per mass bin
     (equivalently: since the conversion tau->gas-column is the SAME
     multiplicative constant for every halo, f~gas = CAP_gas_col/CAP_tot_col
     per halo, then we bin+median).
  5. Delete the transient recomposited grids (kept only if --keep_scratch).

Writes one shard per run: KS/fgas_lightcone_shards/{run}_snap{snap}.npz
(preempt-safe / disBatch-friendly: re-running skips runs whose shard exists,
matching the `bind_mpi_worker.py` shard convention). Merge shards with
`--merge` into `KS/fgas_lightcone_snap{snap}.npz` (fiducial + truth + N nodes,
drop-in for the `figD_lightcone_confront` companion of `figE_lowmass`).

Usage (single run, mechanics test):
    python examples/_reduce_fgas_lightcone.py --run fiducial --snap 085
    python examples/_reduce_fgas_lightcone.py --run truth    --snap 085
    python examples/_reduce_fgas_lightcone.py --run 0        --snap 085   # Sobol run_0000

Batch (one shell line per run -- ideal disBatch task, or SLURM --array):
    for i in $(seq 0 255); do
        echo "python examples/_reduce_fgas_lightcone.py --run $i --snap 085"
    done > tasks_fgas_lightcone_085.txt
    disBatch tasks_fgas_lightcone_085.txt
    # or: sbatch --array=0-255 run_fgas_lightcone.sh   (SLURM array; see that file)

Merge once all shards exist:
    python examples/_reduce_fgas_lightcone.py --merge --snap 085
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bind.inference.lightcone_maps import (
    SIGMA_T, X_E_PER_MASS, _periodic_bilinear, _slab_chi, _tau_per_gas_pixel,
)
from bind.inference.lensplane import comoving_distance_from_a
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import _assign_halos_to_slabs
from bind.inference.paint_stages import recomposite_from_saved
from bind.inference import io_gadget

CEPH = Path("/mnt/home/mlee1/ceph")
FID = CEPH / "bind_lightcone_tng"                      # shared fiducial geometry (stage1 + transforms)
RUNS_ROOT = CEPH / "bind_sb35/runs"                     # 256 Sobol runs (generated_patches only)
FID_RUN_DIR = CEPH / "bind_science/runs/bind/run_0000"  # fiducial: already a full composite grid
TRUTH_RUN_DIR = CEPH / "bind_science/runs/truth/run_0000"
FOF_PARENT = "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output"
KS = CEPH / "bind_science/ksz_confront"
SHARDS = KS / "fgas_lightcone_shards"
SCRATCH = CEPH / "bind_science/ksz_confront/_lightcone_scratch"

# snapshot order MUST match run_lightcone_generate.sh / run_sobol_lightcone.sh
SNAPSHOTS = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAP_IDX = {s: i for i, s in enumerate(SNAPSHOTS)}       # snap 85 -> 2, snap 46 -> 12
# a source redshift comfortably above each science snapshot's own z (matches the
# production tomographic bins 0.5/1.0/1.5/2.0; picks the closest one with margin)
Z_SOURCE = {85: 0.5, 46: 1.5}
Om = 0.3089
FOV_DEG_DEFAULT = 5.0
NPIX_DEFAULT = 1024
F_B = 0.0490 / 0.3089
MB = np.array([12.0, 12.33, 12.66, 13.0, 13.4, 13.8, 14.2, 14.8])   # logM200 bins (matches _reduce_fgas_lowmass.py)


# ---------------------------------------------------------------------------
# Step 1: recomposite (Sobol runs only) -- pure numpy, no GPU
# ---------------------------------------------------------------------------

def recomposite_run(run_dir: Path, out_dir: Path, snapshots=SNAPSHOTS, force: bool = False) -> list[int]:
    """Recomposite every snapshot with an on-disk generated_dir onto `out_dir`.

    Returns the list of snapshots successfully recomposited (others are skipped
    -- e.g. if this run's generation didn't cover that snapshot).
    """
    done = []
    for s in snapshots:
        gen_dir = run_dir / f"snap_{s:03d}"
        stage1_dir = FID / f"snap_{s:03d}" / "stage1"
        out_snap = out_dir / f"snap_{s:03d}"
        if out_snap.exists() and not force and (out_snap / "composite_slab00.npz").exists():
            done.append(s); continue
        if not gen_dir.exists() or not stage1_dir.exists():
            continue
        recomposite_from_saved(stage1_dir, gen_dir, out_snap,
                                save_per_halo_patches=False, progress=False)
        done.append(s)
    return done


# ---------------------------------------------------------------------------
# Step 2: assemble tau (kSZ) + Sigma_tot (DM+Gas+Star) on the SAME geometry
# ---------------------------------------------------------------------------

def assemble_gas_and_mass(snap_root: Path, snapshots, z_source: float, *,
                           fov_deg: float = FOV_DEG_DEFAULT, npix: int = NPIX_DEFAULT,
                           seed: int = 0, manifest_root: Path = FID, verbose: bool = False):
    """Mirrors `assemble_lightcone`'s tau accumulation, but also accumulates the
    RAW total-matter column (no lensing kernel -- same LOS-additive recipe as
    tau/y, matching the module's own "Born approximation, exact to first order"
    treatment of these two fields).

    Returns (tau_map, sigmatot_map, geom) where geom[s] = dict(shift, box,
    n_slabs, slab_depth, chi_l=[per slab], L_ang=[per slab]) for every
    snapshot that contributed at least one plane -- everything a caller needs
    to place a halo on the map.
    """
    fov = np.deg2rad(fov_deg)
    chi_s = comoving_distance_from_a(1.0 / (1.0 + z_source), Om)
    rng = np.random.default_rng(seed)
    tau_map = np.zeros((npix, npix), dtype=np.float64)
    tot_map = np.zeros((npix, npix), dtype=np.float64)
    geom: dict[int, dict] = {}

    for s in snapshots:
        man_path = manifest_root / f"snap_{s:03d}" / "stage1" / "stage1_manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        a_l = float(man["scale_factor"]); box = float(man["box_size"])
        n_slabs = int(man["n_slabs"]); slab_depth = float(man["slab_depth"])
        chi_snap = comoving_distance_from_a(a_l, Om)
        shift = rng.uniform(0.0, box, size=2)     # decorrelation, SAME draw order as assemble_lightcone

        chi_l_list, L_ang_list = [], []
        for sl in range(n_slabs):
            f2 = snap_root / f"snap_{s:03d}" / f"composite_slab{sl:02d}.npz"
            chi_l = _slab_chi(chi_snap, n_slabs, sl, slab_depth)
            chi_l_list.append(chi_l); L_ang_list.append(fov * chi_l if chi_l > 0 else 0.0)
            if chi_l <= 0 or chi_l >= chi_s or not f2.exists():
                continue
            L_ang = fov * chi_l
            d = np.load(f2)
            gas = d["composite"][1].astype(np.float64)
            tot = d["composite"].sum(0).astype(np.float64)
            K = _tau_per_gas_pixel(box, gas.shape[0], a_l)         # tau per unit gas-mass pixel
            base_conv = K / (SIGMA_T * X_E_PER_MASS)                # raw-mass -> physical column [g/cm^2 per Msun/h]
            tau_map += _periodic_bilinear(K * gas, box, L_ang, npix, *shift)
            tot_map += _periodic_bilinear(base_conv * tot, box, L_ang, npix, *shift)
        geom[s] = dict(shift=shift, box=box, n_slabs=n_slabs, slab_depth=slab_depth,
                       chi_l=chi_l_list, L_ang=L_ang_list)
        if verbose:
            print(f"  snap {s} (a={a_l:.3f}) done")
    return tau_map, tot_map, geom


# ---------------------------------------------------------------------------
# Step 3: halo -> (possibly several) pixel positions on the periodically-tiled map
# ---------------------------------------------------------------------------

def halo_tile_pixels(x: float, y: float, shift, box: float, L_ang: float, npix: int):
    """All periodic tile-copy pixel positions of a box position (x,y) on the map.

    Inverts `_periodic_bilinear`'s forward map `cx=((shift+u)%box)/box*N` for
    `u` in `[0, L_ang)`: `u = ((pos-shift) % box) + k*box`, `k=0..n_tiles-1`.
    """
    if L_ang <= 0:
        return []
    n_tiles = int(np.ceil(L_ang / box)) + 1
    out = []
    ux0 = (x - shift[0]) % box
    uy0 = (y - shift[1]) % box
    for kx in range(n_tiles):
        ux = ux0 + kx * box
        if ux >= L_ang:
            continue
        for ky in range(n_tiles):
            uy = uy0 + ky * box
            if uy >= L_ang:
                continue
            out.append((ux / L_ang * npix, uy / L_ang * npix))
    return out


def cap_on_map(m2d: np.ndarray, px: float, py: float, theta_pix: float):
    """Disk r<theta_pix minus equal-area ring theta_pix<r<sqrt2*theta_pix."""
    P = m2d.shape[-1]; w = int(np.ceil(np.sqrt(2) * theta_pix)) + 2
    x0, x1 = max(0, int(px - w)), min(P, int(px + w + 1))
    y0, y1 = max(0, int(py - w)), min(P, int(py + w + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = m2d[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - px, yy - py)
    disk = r < theta_pix; ring = (r >= theta_pix) & (r < np.sqrt(2) * theta_pix)
    if disk.sum() < 3 or ring.sum() < 5:
        return None
    w_area = disk.sum() / ring.sum()
    return float(sub[disk].sum() - sub[ring].sum() * w_area)


# ---------------------------------------------------------------------------
# Step 4: CAP-stack the FoF halos on (tau_map, tot_map) -> f~gas(M)
# ---------------------------------------------------------------------------

def fgas_lightcone_of_run(snap_root: Path, snap: int, *, fov_deg=FOV_DEG_DEFAULT,
                          npix=NPIX_DEFAULT, seed=0, verbose=False):
    tf = LightconeTransforms.load(FID / "lightcone_transforms.json")
    gc = f"{FOF_PARENT}/groups_{snap:03d}"
    cat = io_gadget.read_fof_catalog(gc, snapshot=snap, halo_mass_min=1e12, mass_field="Group_M_Crit200")
    snap_idx = SNAP_IDX[snap]
    xyz = tf.apply(cat["positions"], snap_idx, 205.0)
    slab = _assign_halos_to_slabs(xyz[:, 2], 205.0, 4)     # n_slabs=4 (matches stage1_manifest)

    tau_map, tot_map, geom = assemble_gas_and_mass(
        snap_root, SNAPSHOTS, Z_SOURCE[snap], fov_deg=fov_deg, npix=npix, seed=seed, verbose=verbose)
    if snap not in geom:
        raise RuntimeError(f"snap {snap} produced no planes -- check {snap_root}")
    g = geom[snap]

    rows = []   # (logM200, f~gas)
    for i in range(len(cat["mass"])):
        sl = int(slab[i]); chi_l = g["chi_l"][sl]; L_ang = g["L_ang"][sl]
        if chi_l <= 0 or L_ang <= 0:
            continue
        tiles = halo_tile_pixels(xyz[i, 0], xyz[i, 1], g["shift"], g["box"], L_ang, npix)
        if not tiles:
            continue
        theta_pix = max(cat["r200"][i] / (L_ang / npix), 2.0)
        caps_g, caps_t = [], []
        for px, py in tiles:
            cg = cap_on_map(tau_map, px, py, theta_pix)
            ct = cap_on_map(tot_map, px, py, theta_pix)
            if cg is not None and ct is not None and ct > 0:
                caps_g.append(cg / (SIGMA_T * X_E_PER_MASS)); caps_t.append(ct)
        if not caps_g:
            continue
        cg_mean, ct_mean = float(np.mean(caps_g)), float(np.mean(caps_t))
        if ct_mean <= 0:
            continue
        rows.append((np.log10(cat["mass"][i]), cg_mean / ct_mean / F_B))

    rows = np.asarray(rows)
    if rows.size == 0:
        return np.full(len(MB) - 1, np.nan), 0
    lm, fg = rows[:, 0], rows[:, 1]
    ok = np.isfinite(fg) & (fg > -0.5) & (fg < 3)
    out = np.full(len(MB) - 1, np.nan)
    for i in range(len(MB) - 1):
        m = ok & (lm >= MB[i]) & (lm < MB[i + 1])
        if m.sum() > 5:
            out[i] = np.median(fg[m])
    return out, int(ok.sum())


# ---------------------------------------------------------------------------
# CLI: one run per invocation (disBatch/SLURM-array friendly) + a merge mode
# ---------------------------------------------------------------------------

def _resolve_run(run_arg: str):
    """--run fiducial|truth|<int> -> (label, snap_root, needs_recomposite)."""
    if run_arg == "fiducial":
        return "fiducial", FID_RUN_DIR, False
    if run_arg == "truth":
        return "truth", TRUTH_RUN_DIR, False
    n = int(run_arg)
    return f"run_{n:04d}", RUNS_ROOT / f"run_{n:04d}", True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="'fiducial' | 'truth' | Sobol node index (0-255)")
    ap.add_argument("--snap", type=int, default=85, choices=[85, 46], help="85=BGS z~0.18, 46=ELG z~1.16")
    ap.add_argument("--npix", type=int, default=NPIX_DEFAULT)
    ap.add_argument("--fov_deg", type=float, default=FOV_DEG_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_scratch", action="store_true", help="don't delete recomposited grids afterward")
    ap.add_argument("--force", action="store_true", help="redo even if a shard already exists")
    ap.add_argument("--merge", action="store_true", help="merge all shards for --snap into one npz and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)

    if a.merge:
        merge_shards(a.snap); return

    if not a.run:
        ap.error("--run is required unless --merge")
    label, snap_root, needs_recomp = _resolve_run(a.run)
    shard = SHARDS / f"{label}_snap{a.snap:03d}.npz"
    if shard.exists() and not a.force:
        print(f"[skip] {shard} already exists (--force to redo)"); return

    t0 = time.time()
    scratch_dir = None
    try:
        if needs_recomp:
            scratch_dir = SCRATCH / f"{label}_snap{a.snap:03d}"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            done = recomposite_run(snap_root, scratch_dir, SNAPSHOTS)
            print(f"[{label}] recomposited {len(done)}/{len(SNAPSHOTS)} snapshots [{time.time()-t0:.0f}s]")
            work_root = scratch_dir
        else:
            work_root = snap_root

        curve, n_halos = fgas_lightcone_of_run(work_root, a.snap, fov_deg=a.fov_deg,
                                               npix=a.npix, seed=a.seed, verbose=a.verbose)
        print(f"[{label}] snap {a.snap}: n_halos_used={n_halos}  f~gas={np.round(curve, 2)}  "
              f"[{time.time()-t0:.0f}s]")
        mc = 0.5 * (MB[:-1] + MB[1:])
        np.savez(shard, label=label, logM=mc, fgas=curve, n_halos=n_halos,
                fov_deg=a.fov_deg, npix=a.npix, seed=a.seed)
        print(f"wrote {shard}")
    finally:
        if scratch_dir is not None and scratch_dir.exists() and not a.keep_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)


def merge_shards(snap: int):
    files = sorted(SHARDS.glob(f"*_snap{snap:03d}.npz"))
    if not files:
        print(f"no shards found for snap {snap} in {SHARDS}"); return
    fid = tru = None; nodes, sb = [], []
    mc = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        mc = d["logM"]
        if str(d["label"]) == "fiducial":
            fid = d["fgas"]
        elif str(d["label"]) == "truth":
            tru = d["fgas"]
        else:
            nodes.append(int(str(d["label"]).split("_")[1])); sb.append(d["fgas"])
    out = KS / f"fgas_lightcone_snap{snap:03d}.npz"
    np.savez(out, logM=mc, fiducial=fid if fid is not None else np.full_like(mc, np.nan),
             truth=tru if tru is not None else np.full_like(mc, np.nan),
             sb35=np.array(sb) if sb else np.zeros((0, len(mc))),
             node_ids=np.array(nodes))
    print(f"merged {len(files)} shards -> {out}  (fiducial={fid is not None}, truth={tru is not None}, nodes={len(nodes)})")


if __name__ == "__main__":
    main()
