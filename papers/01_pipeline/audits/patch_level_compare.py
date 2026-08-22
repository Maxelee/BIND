"""Investigation 2: patch-level (pre-composite) gas/tau amplitude bias.

Compares raw `generated_patches` (model output, PRE patch_mass_match, PRE
scale_global) in bind_lightcone_tng/snap_096 against the raw particle-projected
`generated_patches` in bind_science/runs/truth/run_0000/snap_096, at the SAME
666+723+743+801 = 2933 halos by row index (verified centers/masses/r200 match
exactly -> identical halo ordering).

Channels: generated_patches[:, 0]=DM_hydro, [:,1]=Gas, [:,2]=Stars
          thermo_patches[:, 0]=compton_y, [:,1]=T, [:,2]=entropy, [:,3]=P_e
"""
import numpy as np
import gc

FID_DIR = "/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096"
TRUTH_DIR = "/mnt/home/mlee1/ceph/bind_science/runs/truth/run_0000/snap_096"

slabs = [0, 1, 2, 3]

results = {
    "gas_fid": [], "gas_truth": [],
    "dm_fid": [], "dm_truth": [],
    "star_fid": [], "star_truth": [],
    "y_fid": [], "y_truth": [],
    "T_fid": [], "T_truth": [],
    "mass": [],
}

for si in slabs:
    f = np.load(f"{FID_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)
    t = np.load(f"{TRUTH_DIR}/composite_slab{si:02d}.npz", allow_pickle=True)

    n_f = int(f["n_halos"]); n_t = int(t["n_halos"])
    assert n_f == n_t, (si, n_f, n_t)
    if n_f == 0:
        continue

    # sanity: same halos by row index
    assert np.allclose(f["halo_masses"], t["halo_masses"]), "halo order mismatch!"

    gp_f = f["generated_patches"]  # (N,3,128,128) raw model output, physical units
    gp_t = t["generated_patches"]  # (N,3,128,128) raw particle projection

    results["dm_fid"].append(gp_f[:, 0].sum(axis=(1, 2)))
    results["dm_truth"].append(gp_t[:, 0].sum(axis=(1, 2)))
    results["gas_fid"].append(gp_f[:, 1].sum(axis=(1, 2)))
    results["gas_truth"].append(gp_t[:, 1].sum(axis=(1, 2)))
    results["star_fid"].append(gp_f[:, 2].sum(axis=(1, 2)))
    results["star_truth"].append(gp_t[:, 2].sum(axis=(1, 2)))

    if "thermo_patches" in f.files and "thermo_patches" in t.files:
        tp_f = f["thermo_patches"]
        tp_t = t["thermo_patches"]
        results["y_fid"].append(tp_f[:, 0].sum(axis=(1, 2)))
        results["y_truth"].append(tp_t[:, 0].sum(axis=(1, 2)))
        # mass-weighted T proxy: mean T over patch (already mass-weighted per-pixel in truth build)
        results["T_fid"].append(tp_f[:, 1].mean(axis=(1, 2)))
        results["T_truth"].append(tp_t[:, 1].mean(axis=(1, 2)))

    results["mass"].append(f["halo_masses"])

    del f, t, gp_f, gp_t
    gc.collect()

for k in results:
    results[k] = np.concatenate(results[k])

n = len(results["mass"])
print(f"Total halos: {n}")

def report(name, num, den):
    ratio = num / np.where(den > 0, den, np.nan)
    finite = np.isfinite(ratio)
    med = np.nanmedian(ratio[finite])
    mean = np.nanmean(ratio[finite])
    p16, p84 = np.nanpercentile(ratio[finite], [16, 84])
    print(f"{name:20s} median={med:.4f}  mean={mean:.4f}  16/84%=({p16:.4f},{p84:.4f})  n_finite={finite.sum()}/{len(ratio)}")
    return ratio

r_dm = report("DM_hydro fid/truth", results["dm_fid"], results["dm_truth"])
r_gas = report("Gas fid/truth", results["gas_fid"], results["gas_truth"])
r_star = report("Stars fid/truth", results["star_fid"], results["star_truth"])
r_y = report("Compton-y fid/truth", results["y_fid"], results["y_truth"])
r_T = report("T(mean) fid/truth", results["T_fid"], results["T_truth"])

np.savez(
    "/tmp/claude-2107/-mnt-home-mlee1-BIND/eaad8798-3159-4ecc-ad3b-687c176db4cc/scratchpad/inv2/patch_level_ratios.npz",
    mass=results["mass"],
    r_dm=r_dm, r_gas=r_gas, r_star=r_star, r_y=r_y, r_T=r_T,
    gas_fid=results["gas_fid"], gas_truth=results["gas_truth"],
    dm_fid=results["dm_fid"], dm_truth=results["dm_truth"],
    y_fid=results["y_fid"], y_truth=results["y_truth"],
)
print("saved ratios")

# mass-binned check: is the ratio mass-independent?
mass = results["mass"]
bins = np.percentile(mass, [0, 25, 50, 75, 100])
print("\nMass-binned gas ratio (patch-level, pre-composite):")
for i in range(4):
    sel = (mass >= bins[i]) & (mass <= bins[i+1])
    print(f"  M in [{bins[i]:.2e},{bins[i+1]:.2e}]  n={sel.sum():4d}  median gas ratio={np.nanmedian(r_gas[sel]):.4f}  median y ratio={np.nanmedian(r_y[sel]):.4f}")
