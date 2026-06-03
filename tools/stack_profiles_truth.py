"""Stacked TRUTH (CAMELS hydro) profiles for the real-data validation of the
Observable -> f_b map.

For every CV simulation (fiducial cosmology AND fiducial feedback; the 27-sim
cosmic-variance set) we take the **hydro-truth** fields, stack the in-bin halos at
the field level exactly as `tools/stack_profiles_reduce.py` stacks the BIND maps,
and build truth Y(r), SX(r), T/S/P(r) and the truth f_b(r) profile.  These are the
ground truth the BIND-learned map will be tested against.

Truth fields per CV sim:
  snap_090/full_maps.npz              truth_maps (3,1024,1024) = [DM_hydro,Gas,Stars]
  .../halo_catalog.npz                centers (Mpc/h), masses(=M200), radii(=R200)
  .../truth_thermo_patches.npz        truth_thermo (n,4,128,128) = [y,T,entropy,P_e]
Truth mass patches are cut from full-box truth_maps at each halo centre with the
crop convention validated to reproduce the stored DMO `condition` patch to corr=1.

We accumulate, per (sim, mass bin), the azimuthal profile of the **summed** maps
(sum over the sim's halos), so any later aggregation -- all-CV or a bootstrap over
sims -- combines *exactly* by summing over sims (means/ratios then follow):
  Y(r)=Σ y_sum/Σ count   SX(r)=Σ SX_sum/Σ count
  T(r)=Σ Tgas_sum/Σ gas_sum (gas-weighted)   f_b(r)=Σ baryon_sum/Σ tot_sum

Output -> /mnt/home/mlee1/ceph/sobol_ss_cv/truth_stacked_profiles.npz
Run:  python tools/stack_profiles_truth.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# reuse the EXACT geometry / radial binning / azimuthal averaging used for BIND
_spec = importlib.util.spec_from_file_location(
    "spr", Path(__file__).with_name("stack_profiles_reduce.py"))
spr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(spr)
azim, R_CEN, NR = spr.azim, spr.R_CEN, spr.NR
MASS_BINS, MASS_LBL = spr.MASS_BINS, spr.MASS_LBL

CV_ROOT = Path("/mnt/home/mlee1/ceph/fm_testsuite/CV")
SNAP = "snap_090"
MASS_TAG = "mass_threshold_1p000e13"
NPIX = 1024
BOX = 50.0
H = 64
OUT = Path("/mnt/home/mlee1/ceph/sobol_ss_cv/truth_stacked_profiles.npz")


def crop(field, cx, cy):
    ix = (cx - H + np.arange(128)) % NPIX
    iy = (cy - H + np.arange(128)) % NPIX
    return field[np.ix_(ix, iy)]


def truth_mass_patches(truth_maps, centers):
    """(n,3,128,128) [DM,Gas,Stars] cut from the full box at each halo centre."""
    out = np.empty((len(centers), 3, 128, 128), np.float32)
    for i, c in enumerate(centers):
        cx = int(c[0] / BOX * NPIX); cy = int(c[1] / BOX * NPIX)
        for ch in range(3):
            out[i, ch] = crop(truth_maps[ch], cx, cy)
    return out


def main():
    sims = sorted(p for p in CV_ROOT.iterdir() if p.is_dir())
    keys = ["y_sum", "SX_sum", "gas_sum", "baryon_sum", "tot_sum",
            "Tgas_sum", "Sgas_sum", "Pgas_sum"]
    per_sim = {k: [] for k in keys}
    counts = []; used = []; allM = []
    for sd in sims:
        md = sd / SNAP / MASS_TAG
        cat_f = md / "halo_catalog.npz"; th_f = md / "truth_thermo_patches.npz"
        fm_f = sd / SNAP / "full_maps.npz"
        if not (cat_f.exists() and th_f.exists() and fm_f.exists()):
            print(f"[skip] {sd.name}: missing files"); continue
        cat = np.load(cat_f)
        if "radii" not in cat.files or len(cat["radii"]) == 0:
            print(f"[skip] {sd.name}: empty radii"); continue
        centers = cat["centers"]; M200 = np.asarray(cat["masses"], float)
        logM = np.log10(M200)
        truth_maps = np.load(fm_f)["truth_maps"]                 # (3,1024,1024)
        thermo = np.load(th_f)["truth_thermo"].astype(np.float32)  # (n,4,128,128) y,T,S,P
        mass = truth_mass_patches(truth_maps, centers)           # (n,3,128,128)
        DM, GAS, STAR = mass[:, 0], mass[:, 1], mass[:, 2]
        Y, T, S, P = thermo[:, 0], thermo[:, 1], thermo[:, 2], thermo[:, 3]
        SXpix = np.clip(GAS, 0, None) ** 2 * np.sqrt(np.clip(T, 0, None))
        row = {k: np.full((3, NR), np.nan) for k in keys}; cnt = np.zeros(3, int)
        for b, (lo, hi) in enumerate(MASS_BINS):
            idx = np.where((logM >= lo) & (logM < hi))[0]
            cnt[b] = len(idx)
            if len(idx) == 0:
                continue
            row["y_sum"][b]      = azim(Y[idx].sum(0))
            row["SX_sum"][b]     = azim(SXpix[idx].sum(0))
            row["gas_sum"][b]    = azim(GAS[idx].sum(0))
            row["baryon_sum"][b] = azim((GAS + STAR)[idx].sum(0))
            row["tot_sum"][b]    = azim((DM + GAS + STAR)[idx].sum(0))
            row["Tgas_sum"][b]   = azim((T * GAS)[idx].sum(0))
            row["Sgas_sum"][b]   = azim((S * GAS)[idx].sum(0))
            row["Pgas_sum"][b]   = azim((P * GAS)[idx].sum(0))
        for k in keys:
            per_sim[k].append(row[k])
        counts.append(cnt); used.append(sd.name); allM.append(logM)
        print(f"[{sd.name}] n={len(M200)} per-bin {cnt.tolist()}", flush=True)

    per_sim = {k: np.array(v) for k, v in per_sim.items()}     # (n_sim,3,nr)
    counts = np.array(counts)                                  # (n_sim,3)
    print("total halos:", counts.sum(), "over", len(used), "sims")
    np.savez_compressed(OUT, r_kpc=R_CEN, mass_lbl=np.array(MASS_LBL),
                        counts=counts, sims=np.array(used), **per_sim)
    # quick all-CV f_b sanity
    fb = per_sim["baryon_sum"].sum(0) / per_sim["tot_sum"].sum(0)
    for b, l in enumerate(MASS_LBL):
        print(f"  truth f_b {l}: ", fb[b].round(3))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
