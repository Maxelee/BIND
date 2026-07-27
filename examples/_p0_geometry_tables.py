#!/usr/bin/env python
"""
P0: Build geometry/transform tables for BIND SB35 lightcone analysis.

Parse config.dat files from 50 realizations, extract randomization (rot, disp),
verify geometry arrays, build snap↔z table, load lightcone_transforms, and
generate verdict JSON + V0 figure.
"""
import json
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List

# Configuration
FID = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng")
OUT_DIR = Path("/mnt/home/mlee1/ceph/bind_science/ksz_confront/lightcone")
SNAP_IDX_TO_NUM = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SOURCE_PLANE_INDICES = [26, 45, 59, 70, 78]  # 1-indexed in lux; convert to 0-indexed for chi_out
SOURCE_Z_EXPECTED = [0.5, 1.0, 1.5, 2.0, 2.44]


def parse_config_dat(fpath: Path) -> Dict:
    """
    Parse binary config.dat file.

    Format (little-endian):
    int32 Np(=80), int32 Ns(=20), float64 a[81], float64 chi[81], float64 chi_out[80],
    float64 Ll[20], float64 Lt[20], int32 rot[20], int32 disp[40] (x0,y0,x1,y1,…)
    """
    with open(fpath, "rb") as f:
        data = f.read()

    # Calculate expected size
    expected_size = (
        4 +  # int32 Np
        4 +  # int32 Ns
        81 * 8 +  # float64 a[81]
        81 * 8 +  # float64 chi[81]
        80 * 8 +  # float64 chi_out[80]
        20 * 8 +  # float64 Ll[20]
        20 * 8 +  # float64 Lt[20]
        20 * 4 +  # int32 rot[20]
        40 * 4    # int32 disp[40]
    )

    actual_size = len(data)
    if actual_size != expected_size:
        raise ValueError(
            f"{fpath.name}: size mismatch. Expected {expected_size}, got {actual_size}"
        )

    offset = 0

    # Parse header
    Np, Ns = struct.unpack_from("<ii", data, offset)
    offset += 8
    assert Np == 80 and Ns == 20, f"Unexpected Np={Np}, Ns={Ns}"

    # Parse geometry arrays
    a = np.array(struct.unpack_from(f"<{81}d", data, offset))
    offset += 81 * 8

    chi = np.array(struct.unpack_from(f"<{81}d", data, offset))
    offset += 81 * 8

    chi_out = np.array(struct.unpack_from(f"<{80}d", data, offset))
    offset += 80 * 8

    Ll = np.array(struct.unpack_from(f"<{20}d", data, offset))
    offset += 20 * 8

    Lt = np.array(struct.unpack_from(f"<{20}d", data, offset))
    offset += 20 * 8

    # Parse randomization
    rot = np.array(struct.unpack_from(f"<{20}i", data, offset), dtype=np.int32)
    offset += 20 * 4

    disp_flat = np.array(struct.unpack_from(f"<{40}i", data, offset), dtype=np.int32)
    disp = disp_flat.reshape((20, 2))  # (snap_idx, x/y)

    return {
        "Np": Np,
        "Ns": Ns,
        "a": a,
        "chi": chi,
        "chi_out": chi_out,
        "Ll": Ll,
        "Lt": Lt,
        "rot": rot,
        "disp": disp,
        "expected_size": expected_size,
        "actual_size": actual_size,
    }


def load_stage1_manifest(snap_num: int) -> Dict:
    """Load stage1_manifest.json for a snapshot."""
    snap_dir = FID / f"snap_{snap_num:03d}" / "stage1"
    manifest_file = snap_dir / "stage1_manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing {manifest_file}")

    with open(manifest_file) as f:
        return json.load(f)


def load_lightcone_transforms() -> Dict:
    """Load lightcone_transforms.json."""
    transforms_file = FID / "lightcone_transforms.json"
    if not transforms_file.exists():
        raise FileNotFoundError(f"Missing {transforms_file}")

    with open(transforms_file) as f:
        return json.load(f)


def z_from_chi(chi_val: float, a_array: np.ndarray, chi_array: np.ndarray) -> float:
    """Interpolate redshift from comoving distance."""
    # Find the a value at this chi by interpolation
    a_at_chi = np.interp(chi_val, chi_array, a_array)
    z = 1.0 / a_at_chi - 1.0
    return z


def main():
    """Execute P0 phase."""
    verdict = {
        "phase": "P0",
        "pass": True,
        "metrics": {},
        "figs": [],
        "notes": [],
        "next": "P1",
    }

    print("=" * 80)
    print("P0: Geometry & Transform Tables")
    print("=" * 80)

    # ========== Step 1: Parse config.dat files ==========
    print("\n[1] Parsing 50 config.dat files...")
    rot_array = []
    disp_array = []
    geometry = None
    byte_size_ok = True

    for r in range(1, 51):
        config_path = FID / f"rt_output/run{r:03d}/config.dat"
        try:
            parsed = parse_config_dat(config_path)

            # Verify byte size
            if parsed["expected_size"] != parsed["actual_size"]:
                byte_size_ok = False
                verdict["notes"].append(
                    f"run{r:03d}: byte size mismatch: {parsed['actual_size']} != {parsed['expected_size']}"
                )

            rot_array.append(parsed["rot"])
            disp_array.append(parsed["disp"])

            # Verify geometry is identical
            if geometry is None:
                geometry = {k: parsed[k] for k in ["a", "chi", "chi_out", "Ll", "Lt"]}
            else:
                for key in ["a", "chi", "chi_out", "Ll", "Lt"]:
                    if not np.allclose(geometry[key], parsed[key]):
                        verdict["pass"] = False
                        verdict["notes"].append(
                            f"run{r:03d}: {key} mismatch from run001"
                        )
        except Exception as e:
            verdict["pass"] = False
            verdict["notes"].append(f"run{r:03d}: {str(e)}")

    rot_array = np.array(rot_array, dtype=np.int32)  # (50, 20)
    disp_array = np.array(disp_array, dtype=np.int32)  # (50, 20, 2)

    verdict["metrics"]["n_records"] = 50
    verdict["metrics"]["byte_size_check"] = byte_size_ok
    print(f"  ✓ Parsed 50 files, byte sizes OK: {byte_size_ok}")

    # ========== Step 2: Gate checks on rot/disp ==========
    print("\n[2] Gate checks on randomization arrays...")
    rot_ok = np.all((rot_array >= 0) & (rot_array <= 3))
    disp_ok = np.all((disp_array >= 0) & (disp_array < 4096))

    verdict["metrics"]["rot_ok"] = bool(rot_ok)
    verdict["metrics"]["disp_ok"] = bool(disp_ok)
    verdict["pass"] = verdict["pass"] and rot_ok and disp_ok

    print(f"  rot ∈ {{0,1,2,3}}: {rot_ok}")
    print(f"  disp ∈ [0,4096): {disp_ok}")

    # ========== Step 3: Gate checks on geometry ==========
    print("\n[3] Gate checks on geometry arrays...")

    a = geometry["a"]
    chi = geometry["chi"]
    chi_out = geometry["chi_out"]
    Ll = geometry["Ll"]
    Lt = geometry["Lt"]

    # Check monotonicity
    chi_monotonic = np.all(np.diff(chi) > 0)
    verdict["metrics"]["chi_monotonic"] = bool(chi_monotonic)
    verdict["pass"] = verdict["pass"] and chi_monotonic
    print(f"  chi strictly increasing: {chi_monotonic}")

    # Report Lt values
    Lt_unique = np.unique(Lt)
    verdict["metrics"]["Lt_values"] = Lt_unique.tolist()
    print(f"  Lt unique values: {Lt_unique}")
    if not np.allclose(Lt_unique, [200.0]):
        verdict["notes"].append(
            f"Lt values unexpected: {Lt_unique.tolist()}. Expected ~[200.0]"
        )

    # ========== Step 4: Build snap↔z table ==========
    print("\n[4] Building snap↔z table from manifests...")
    snap_z_table = []

    for snap_idx, snap_num in enumerate(SNAP_IDX_TO_NUM):
        manifest = load_stage1_manifest(snap_num)
        a_snap = manifest["scale_factor"]
        z_snap = manifest["redshift"]

        snap_z_table.append({
            "snap": snap_num,
            "snap_idx": snap_idx,
            "z": z_snap,
            "a": a_snap,
        })

    snap_z_table_list = [
        [row["snap"], row["snap_idx"], row["z"]] for row in snap_z_table
    ]
    verdict["metrics"]["snap_z_table"] = snap_z_table_list
    print(f"  ✓ Built 20-row snap↔z table")

    # ========== Step 5: Source plane z check ==========
    print("\n[5] Source plane z check...")

    # Load source_redshifts from tau_maps.npz
    tau_maps_file = FID / "tau_maps.npz"
    with np.load(tau_maps_file, allow_pickle=False) as npz:
        source_z_in_file = npz["source_redshifts"]

    source_z_check = []
    z_check_ok = True

    for m, z_expected in zip(SOURCE_PLANE_INDICES, SOURCE_Z_EXPECTED):
        # m is 1-indexed in lux; chi_out is 0-indexed [0..79]
        # m ∈ {26,45,59,70,78} → chi_out[m-1] ∈ {25,44,58,69,77}
        chi_out_idx = m - 1
        chi_val = chi_out[chi_out_idx]
        z_computed = z_from_chi(chi_val, a, chi)

        # Compare to source_redshifts (index = position in {26,45,59,70,78})
        z_file = source_z_in_file[SOURCE_PLANE_INDICES.index(m)]
        error_pct = abs(z_computed - z_file) / z_file * 100

        source_z_check.append([m, round(z_computed, 4), round(z_file, 4), round(error_pct, 2)])

        if error_pct > 2.0:
            z_check_ok = False
            verdict["notes"].append(
                f"Source plane m={m}: z_computed={z_computed:.4f}, z_file={z_file:.4f}, error={error_pct:.2f}%"
            )

    verdict["metrics"]["source_z_check"] = source_z_check
    verdict["metrics"]["source_z_ok"] = z_check_ok
    verdict["pass"] = verdict["pass"] and z_check_ok
    print(f"  ✓ Source plane z check OK: {z_check_ok}")
    for row in source_z_check:
        print(f"    m={row[0]}: z_computed={row[1]}, z_file={row[2]}, error={row[3]}%")

    # ========== Step 6: Load lightcone_transforms ==========
    print("\n[6] Loading lightcone_transforms.json...")
    lc_transforms = load_lightcone_transforms()

    # Convert to arrays
    lc_proj_dirs = np.array(lc_transforms.get("proj_dirs", []), dtype=np.int32)
    lc_disp = np.array(lc_transforms.get("disp", []), dtype=np.float64)
    lc_flip = np.array(lc_transforms.get("flip", []), dtype=np.int32)

    # Verify consistency with manifests
    lc_manifest_check_ok = True
    for snap_idx, snap_num in enumerate(SNAP_IDX_TO_NUM):
        manifest = load_stage1_manifest(snap_num)
        proj_dir_manifest = manifest.get("proj_dir")
        disp_manifest = manifest.get("disp")
        flip_manifest = manifest.get("flip")

        # proj_dir should match lc_proj_dirs[snap_idx]
        if proj_dir_manifest is not None and proj_dir_manifest != lc_proj_dirs[snap_idx]:
            lc_manifest_check_ok = False
            verdict["notes"].append(
                f"snap_idx {snap_idx} (snap {snap_num}): proj_dir mismatch: "
                f"manifest={proj_dir_manifest}, lc_transforms={lc_proj_dirs[snap_idx]}"
            )

    verdict["metrics"]["lc_manifest_check"] = lc_manifest_check_ok
    verdict["pass"] = verdict["pass"] and lc_manifest_check_ok
    print(f"  ✓ Loaded lightcone_transforms, manifest check OK: {lc_manifest_check_ok}")

    # ========== Step 7: Save to geometry.npz ==========
    print("\n[7] Saving geometry.npz...")

    np.savez(
        OUT_DIR / "geometry.npz",
        # Randomization per-realization
        rot=rot_array,
        disp=disp_array,
        # Geometry (shared)
        a=a,
        chi=chi,
        chi_out=chi_out,
        Ll=Ll,
        Lt=Lt,
        # Snap↔z table
        snap=np.array([row["snap"] for row in snap_z_table], dtype=np.int32),
        snap_idx=np.array([row["snap_idx"] for row in snap_z_table], dtype=np.int32),
        z=np.array([row["z"] for row in snap_z_table], dtype=np.float64),
        a_snap=np.array([row["a"] for row in snap_z_table], dtype=np.float64),
        # Lightcone transforms
        lc_proj_dirs=lc_proj_dirs,
        lc_disp=lc_disp,
        lc_flip=lc_flip,
    )

    print(f"  ✓ Saved to {OUT_DIR / 'geometry.npz'}")

    # ========== Step 8: Generate V0 figure ==========
    print("\n[8] Generating V0 figure...")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: chi(z) curve with plane bands and source planes
    z_array = 1.0 / a - 1.0
    ax_left.plot(z_array, chi, "b-", linewidth=2, label="chi(z)")

    # Shade plane bands (4 planes per snapshot)
    for snap_idx, snap_num in enumerate(SNAP_IDX_TO_NUM):
        for slab in range(4):
            plane_idx = 4 * snap_idx + slab
            chi_start = chi[plane_idx]
            chi_end = chi[plane_idx + 1]
            ax_left.axvspan(z_array[np.searchsorted(chi, chi_start)],
                            z_array[np.searchsorted(chi, chi_end)],
                            alpha=0.1, color="blue")

    # Mark source planes
    colors = ["red", "orange", "green", "purple", "brown"]
    for m, z_expected, color in zip(SOURCE_PLANE_INDICES, SOURCE_Z_EXPECTED, colors):
        chi_val = chi_out[m - 1]
        z_at_source = z_from_chi(chi_val, a, chi)
        ax_left.axvline(z_at_source, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax_left.text(z_at_source, ax_left.get_ylim()[1] * 0.95, f"z≈{z_expected:.2f}",
                    rotation=90, fontsize=9, color=color)

    ax_left.set_xlabel("Redshift z", fontsize=11)
    ax_left.set_ylabel("Comoving distance χ [Mpc/h]", fontsize=11)
    ax_left.set_title("Lightcone Geometry: χ(z) with plane bands and sources", fontsize=12)
    ax_left.grid(True, alpha=0.3)

    # Right panel: imshow of rot array (50 realizations × 20 snapshots)
    im = ax_right.imshow(rot_array, cmap="tab20", aspect="auto", origin="lower")
    ax_right.set_xlabel("Snapshot index", fontsize=11)
    ax_right.set_ylabel("Realization (1-indexed)", fontsize=11)
    ax_right.set_title("Per-realization plane rotations (0=0°, 1=90°, 2=180°, 3=270°)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax_right)
    cbar.set_label("Rotation", fontsize=10)

    status_str = "P0 PASS" if verdict["pass"] else "P0 FAIL"
    fig.suptitle(f"P0 Geometry Tables — {status_str}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig_path = OUT_DIR / "figs" / "V0_geometry.png"
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved V0 figure to {fig_path}")
    verdict["figs"].append("figs/V0_geometry.png")

    # ========== Step 9: Print per-realization table (r=1..3) ==========
    print("\n[9] Per-realization (rot, disp) for r=1..3:")
    print("     snap_idx |  rot  | disp_x | disp_y |")
    print("     " + "-" * 35)
    for r in range(3):  # r=1,2,3 → array indices 0,1,2
        for snap_idx in range(20):
            rot_val = rot_array[r, snap_idx]
            disp_x = disp_array[r, snap_idx, 0]
            disp_y = disp_array[r, snap_idx, 1]
            if snap_idx == 0:
                print(f"  r={r+1:2d}  {snap_idx:2d}  |  {rot_val}   | {disp_x:5d}  | {disp_y:5d}  |")
            else:
                print(f"        {snap_idx:2d}  |  {rot_val}   | {disp_x:5d}  | {disp_y:5d}  |")

    # ========== Step 10: Save verdict JSON ==========
    print("\n[10] Saving verdict JSON...")
    verdict_path = OUT_DIR / "verdicts" / "P0.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"  ✓ Saved verdict to {verdict_path}")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print(f"P0 Status: {'PASS ✓' if verdict['pass'] else 'FAIL ✗'}")
    print("=" * 80)

    if not verdict["pass"]:
        print("\nFailed checks:")
        for note in verdict["notes"]:
            print(f"  - {note}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    exit(0 if verdict["pass"] else 1)
