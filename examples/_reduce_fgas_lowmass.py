"""Low-mass (reuse) f~gas-M relation: extend the kSZ gas-fraction confrontation
from the 10^13 floor down to 10^12 using halos that already live inside the
painted 6.25 Mpc/h patches (no new generation).

Method (one consistent measurement across the whole mass range):
  * >=1e13 "centrals": the painted patch is centred on them -> background-subtracted
    compensated aperture at the patch centre.
  * 1e12-1e13 "secondaries": cross-match the FoF catalogue to the existing patches
    (lightcone transform + slab assignment, as `lightcone_lowmass_reuse.py`), find the
    off-centre pixel position, same bg-subtracted aperture.
  f~gas(<r200) = [gas_disk - bg*A] / [sum(dm,gas,star)_disk - bg*A] / (Omega_b/Omega_m),
  with bg = mean surface density in a 1.4-2.2 r200 annulus (removes the 50 Mpc/h
  line-of-sight projection that otherwise drives a small aperture to ~cosmic).

Validation (BIND vs truth, same halos/footprints, snap 85): BIND/truth ~1.08-1.17 at
1e12-1e13 -> BIND paints the low-mass gas to ~10%. Caveat: captured low-mass halos are
near a >=1e13 host (median ~1.9 Mpc/h), so they are filament/outskirt, not field.

    python examples/_reduce_fgas_lowmass.py        # fiducial + truth + 24 SB35 nodes
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `examples.*`
from examples.lightcone_lowmass_reuse import load_patch_set, _periodic_delta
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import _assign_halos_to_slabs
from bind.inference import io_gadget

CEPH = Path("/mnt/home/mlee1/ceph")
TRANSF = CEPH / "bind_lightcone_tng/lightcone_transforms.json"
SB35 = CEPH / "bind_sb35/runs"
FOF_PARENT = "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output"   # groups_<snap> within
F_B = 0.0490 / 0.3089
MB = np.array([12.0, 12.33, 12.66, 13.0, 13.4, 13.8, 14.2, 14.8])   # logM200 bins (reuse + centrals)
NODES = list(range(256))                            # full Sobol set for the feedback band


def _comp_fgas(patch, px, py, rp, win):
    """Exact CAP [dm,gas,star]: disk r<rp minus EQUAL-AREA ring rp<r<sqrt2 rp at (px,py)
    (the paper's filter; matches `_reduce_fgas_cap.py`). win = local half-window."""
    P = patch.shape[-1]
    x0, x1 = max(0, int(px - win)), min(P, int(px + win + 1))
    y0, y1 = max(0, int(py - win)), min(P, int(py + win + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - px, yy - py); disk = r < rp; ring = (r >= rp) & (r < np.sqrt(2) * rp)  # CAP equal-area ring
    if disk.sum() < 3 or ring.sum() < 5:
        return None
    sub = patch[:3, y0:y1, x0:x1]; w = disk.sum() / ring.sum()
    return np.array([sub[c][disk].sum() - sub[c][ring].sum() * w for c in range(3)])


def fgas_of_mass(snap_dir, fof):
    """median f~gas(<r200) per logM200 bin: >=1e13 centrals (centred) + 1e12-1e13 reuse."""
    ps = load_patch_set(snap_dir)
    box, npix, n_slabs = ps["box"] if "box" in ps else ps["box_size"], ps["npix"], ps["n_slabs"]
    pix = box / npix
    P = next(iter(s for s in ps["slabs"].values() if s["n"]))["gen"].shape[-1]
    half = (P // 2) * pix
    rows = []                                        # (logM200, f~gas)
    # --- centrals (>=1e13): patch is centred on them ---
    for sl in ps["slabs"].values():
        if not sl["n"]:
            continue
        for i in range(sl["n"]):
            rp = max(sl["r200"][i] / pix, 2.0)
            e = _comp_fgas(sl["gen"][i], P / 2, P / 2, rp, int(2.4 * rp) + 2)
            if e is not None and e.sum() > 0:
                rows.append((np.log10(sl["mass"][i]), e[1] / e.sum() / F_B))
    # --- secondaries (1e12-1e13): reuse cross-match ---
    xy, mass, r200 = fof["xy"], fof["mass"], fof["r200"]
    band = mass < 1e13; slab = _assign_halos_to_slabs(xy[:, 2], box, n_slabs)
    for si, sl in ps["slabs"].items():
        if not sl["n"]:
            continue
        idx = np.where(band & (slab == si))[0]
        if not len(idx):
            continue
        tree = cKDTree(sl["centers"], boxsize=box)
        for k in idx:
            p = xy[k, :2]; cand = tree.query_ball_point(p, half)
            if not cand:
                continue
            best, bd = -1, 1e9
            for c in cand:
                dd = float(np.hypot(*_periodic_delta(p, sl["centers"][c], box)))
                if dd < bd:
                    bd, best = dd, c
            dxy = _periodic_delta(p, sl["centers"][best], box)
            rp = max(r200[k] / pix, 2.0)
            e = _comp_fgas(sl["gen"][best], P / 2 + dxy[0] / pix, P / 2 + dxy[1] / pix, rp, int(2.4 * rp) + 2)
            if e is not None and e.sum() > 0:
                rows.append((np.log10(mass[k]), e[1] / e.sum() / F_B))
    rows = np.array(rows)
    lm, fg = rows[:, 0], rows[:, 1]
    ok = np.isfinite(fg) & (fg > -0.5) & (fg < 3)
    out = np.full(len(MB) - 1, np.nan)
    for i in range(len(MB) - 1):
        s = ok & (lm >= MB[i]) & (lm < MB[i + 1])
        if s.sum() > 20:
            out[i] = np.median(fg[s])
    return out


_FOF = None; _SNAP = None      # set in main() before the Pool; forked workers inherit (Linux fork)


def _node_task(n):
    """Worker: one Sobol node -> (n, f~gas-M array | None). Reads the global _FOF (fork-inherited)."""
    d = SB35 / f"run_{n:04d}/snap_{_SNAP}"
    if not d.exists():
        return n, None
    try:
        return n, fgas_of_mass(d, _FOF)
    except Exception:
        return n, None


def main():
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="085", help="snapshot string: 085 (z0.18, BGS) or 046 (z1.16, ELG)")
    ap.add_argument("--snap_idx", type=int, default=2, help="lightcone transforms index (manifest transforms_snap_idx; 085->2, 046->12)")
    ap.add_argument("--nproc", type=int, default=1, help="parallelise the 256-node loop across this many cores (each node is independent)")
    a = ap.parse_args(); snap = a.snap
    gc = f"{FOF_PARENT}/groups_{snap}"
    fid_dir = CEPH / f"bind_lightcone_tng/snap_{snap}"
    truth_dir = CEPH / f"bind_science/runs/truth/run_0000/snap_{snap}"
    out = CEPH / f"bind_science/ksz_confront/fgas_lowmass_snap{snap}.npz"
    t0 = time.time()
    tf = LightconeTransforms.load(TRANSF)
    cat = io_gadget.read_fof_catalog(gc, snapshot=int(snap), halo_mass_min=1e12, mass_field="Group_M_Crit200")
    fof = {"xy": tf.apply(cat["positions"], a.snap_idx, 205.0), "mass": cat["mass"], "r200": cat["r200"]}
    print(f"[snap {snap}] FoF {len(cat['mass']):,} halos >=1e12  [{time.time()-t0:.0f}s]; nproc={a.nproc}", flush=True)
    global _FOF, _SNAP; _FOF = fof; _SNAP = snap                 # share with forked workers
    fid = fgas_of_mass(fid_dir, fof); print(f"fiducial [{time.time()-t0:.0f}s]:", np.round(fid, 2), flush=True)
    tru = fgas_of_mass(truth_dir, fof); print(f"truth    [{time.time()-t0:.0f}s]:", np.round(tru, 2), flush=True)
    if a.nproc > 1:
        with Pool(a.nproc) as pool:
            results = pool.map(_node_task, NODES, chunksize=1)
    else:
        results = [_node_task(n) for n in NODES]
    ok, sb = [], []
    for n, r in results:                                        # results preserve NODES order
        if r is not None:
            ok.append(n); sb.append(r)
    sb = np.array(sb); mc = 0.5 * (MB[:-1] + MB[1:]); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logM=mc, fiducial=fid, truth=tru, sb35=sb, node_ids=np.array(ok), mass_edges=MB)
    print(f"\nwrote {out}  (snap {snap}: fiducial + truth + {len(ok)} nodes)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
