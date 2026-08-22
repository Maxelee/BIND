"""Reuse-only low-mass baryon recovery + BIND-vs-truth comparison.

Going down to ~10^11 halos with *new* per-halo generation is ~78x the GPU cost of
the 10^13 lightcone (at 10^11, halos fill space: ~50x redundant per 6.25 Mpc/h
patch).  Instead this exploits a free fact: every existing 10^13 patch is a
6.25 Mpc/h cutout that already contains the *true*/painted hydro of all the
smaller halos sharing that column.  So the 10^12-10^13 halos that fall inside an
existing patch footprint are recovered with **no new generation** -- on *both*
sides symmetrically:

* BIND   : `composite_slab*.npz` `generated_patches`/`thermo_patches` (already on disk).
* truth  : `bind-truth-halos` projects the full TNG300 hydro at the *same* 10^13
           DMO halos, so its patches carry the true low-mass hydro in the *same*
           footprints -- a drop-in match (run it once; no low-mass-specific job).

Capture is set by the paste footprint.  The production composite circular-tapers
to `r200_factor` * R200 (~1.76 Mpc/h median for a 10^13 halo), which keeps ~29%
of 10^12-10^13 halos; widening to the full 6.25 Mpc/h patch keeps ~52%.  The
remaining ~48% are field halos too far from any cluster -- accepted as the
reuse-only ceiling.

This module cross-matches the 10^12-10^13 FoF halos against the existing patch
centres (same lightcone transform + slab assignment the lightcone used), tags the
captured ones, and reads each captured halo's aperture (gas/Y/stars/T) straight
out of its host patch.  Run on a BIND snap_dir and a truth snap_dir to get a
matched per-halo BIND-vs-truth table for the recovered low-mass population.

    # BIND side only (existing fiducial, no new compute):
    python -m examples.lightcone_lowmass_reuse \
        --snap_dir /ceph/bind_lightcone_tng/snap_096 \
        --group_catalog /.../L205n2500TNG_DM/output --snapshot_index 96 \
        --transforms /ceph/bind_lightcone_tng/lightcone_transforms.json --snap_idx 0

    # + truth comparison once bind-truth-halos has produced truth patches:
        --truth_dir /ceph/bind_truth/snap_096
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import (
    NATIVE_PIXEL_SIZE_MPCH, _assign_halos_to_slabs, _round_npix,
)
from bind.inference import io_gadget

# patch thermo channel order (bind.data.THERMO_KEYS): compton_y, T, entropy, P_e
THERMO = ("compton_y", "temperature", "entropy", "pressure")


# ---------------------------------------------------------------------------
# patch set (BIND composite_slab*.npz or truth composite_slab*.npz -- same format)
# ---------------------------------------------------------------------------

def load_patch_set(snap_dir: Path) -> dict:
    """Load per-slab patches + halo metadata from a composite_slab*.npz dir."""
    snap_dir = Path(snap_dir)
    slabs: dict[int, dict] = {}
    box_size = npix = n_slabs = None
    for sp in sorted(snap_dir.glob("composite_slab*.npz")):
        d = np.load(sp)
        si = int(d["slab_idx"])
        box_size = float(d["box_size"])
        n_slabs = int(d["n_slabs"])
        if "npix" in d.files:
            npix = int(d["npix"])
        if int(d["n_halos"]) == 0:
            slabs[si] = {"n": 0}
            continue
        slabs[si] = {
            "n": int(d["n_halos"]),
            "centers": d["halo_centers"].astype(np.float64),   # (N,2) transformed xy
            "r200": d["halo_r200"].astype(np.float64),
            "mass": d["halo_masses"].astype(np.float64),
            "gen": d["generated_patches"],                     # (N,3,P,P) DM,Gas,Star
            "thermo": d["thermo_patches"] if "thermo_patches" in d.files else None,
        }
    if npix is None:                          # truth npz omits npix -> derive
        npix = _round_npix(box_size, NATIVE_PIXEL_SIZE_MPCH)
    return {"slabs": slabs, "box_size": box_size, "npix": npix, "n_slabs": n_slabs}


def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = a - b
    return d - box * np.round(d / box)


def _aperture(patch: np.ndarray, px: float, py: float, ap_px: float) -> np.ndarray:
    """Sum each channel of `patch` (C,P,P) over a circle of radius ap_px at (px,py).

    Returns (sums[C], covered_fraction).  covered_fraction < 1 means the aperture
    clipped the patch edge (identical on BIND/truth, so a matched comparison is
    still fair).
    """
    C, P, _ = patch.shape
    ap = max(ap_px, 1.5)
    lo_x, hi_x = max(0, int(px - ap - 1)), min(P, int(px + ap + 2))
    lo_y, hi_y = max(0, int(py - ap - 1)), min(P, int(py + ap + 2))
    if lo_x >= hi_x or lo_y >= hi_y:
        return np.zeros(C), 0.0
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x]
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= ap ** 2
    n_full = np.pi * ap ** 2
    sums = np.array([patch[c, lo_y:hi_y, lo_x:hi_x][mask].sum() for c in range(C)])
    return sums, float(mask.sum()) / max(n_full, 1.0)


def measure(
    snap_dir: Path,
    group_catalog: Path,
    snapshot_index: int,
    transforms: LightconeTransforms | None,
    snap_idx: int | None,
    *,
    mass_lo: float = 1e12,
    mass_hi: float = 1e13,
    r200_factor: float = 4.0,
    full_patch: bool = False,
    aperture_r200: float = 1.0,
    min_aperture_px: float = 2.0,
) -> dict:
    """Cross-match the [mass_lo, mass_hi) FoF halos to the patch set and read
    each captured halo's aperture out of its host patch.

    `full_patch=True` uses the full 6.25 Mpc/h footprint (reuse-only ceiling);
    otherwise the capture radius is min(r200_factor * R200_host, half_patch),
    matching the production circular paste.
    """
    ps = load_patch_set(snap_dir)
    box, npix, n_slabs = ps["box_size"], ps["npix"], ps["n_slabs"]
    pix = box / npix
    half_patch_mpch = (next(iter(s for s in ps["slabs"].values() if s["n"]))["gen"]
                       .shape[-1] // 2) * pix

    cat = io_gadget.read_fof_catalog(group_catalog, snapshot=snapshot_index,
                                     halo_mass_min=mass_lo, mass_field="Group_M_Crit200")
    xy = cat["positions"]
    if transforms is not None:
        xy = transforms.apply(xy, snap_idx, box)
    mass, r200 = cat["mass"], cat["r200"]
    band = mass < mass_hi
    z = xy[:, 2]
    slab = _assign_halos_to_slabs(z, box, n_slabs)

    recs = []  # per captured halo
    n_band = int(band.sum())
    for si, sl in ps["slabs"].items():
        if not sl["n"]:
            continue
        in_slab = band & (slab == si)
        if not in_slab.any():
            continue
        idx = np.where(in_slab)[0]
        lo_xy = xy[idx, :2].astype(np.float64)
        tree = cKDTree(sl["centers"], boxsize=box)
        rad_host = (np.full(sl["n"], half_patch_mpch) if full_patch
                    else np.minimum(r200_factor * sl["r200"], half_patch_mpch))
        for k, p in zip(idx, lo_xy):
            cand = tree.query_ball_point(p, half_patch_mpch)
            if not cand:
                continue
            best = -1
            best_d = 1e9
            for c in cand:
                dxy = _periodic_delta(p, sl["centers"][c], box)
                dd = float(np.hypot(*dxy))
                if dd <= rad_host[c] and dd < best_d:
                    best_d, best = dd, c
            if best < 0:
                continue
            dxy = _periodic_delta(p, sl["centers"][best], box)
            px = sl["gen"].shape[-1] / 2 + dxy[0] / pix
            py = sl["gen"].shape[-1] / 2 + dxy[1] / pix
            ap_px = max(aperture_r200 * r200[k] / pix, min_aperture_px)
            m_sums, cov = _aperture(sl["gen"][best], px, py, ap_px)
            rec = {"halo_id": int(k), "slab": si, "mass": float(mass[k]),
                   "r200": float(r200[k]), "host": int(best), "d_mpch": best_d,
                   "cov": cov, "dm": m_sums[0], "gas": m_sums[1], "star": m_sums[2]}
            if sl["thermo"] is not None:
                t_sums, _ = _aperture(sl["thermo"][best], px, py, ap_px)
                rec["y"] = t_sums[0]
                rec["T"] = t_sums[1]
            recs.append(rec)

    return {"n_band": n_band, "n_captured": len(recs),
            "capture_frac": len(recs) / max(n_band, 1),
            "half_patch_mpch": half_patch_mpch, "records": recs}


def _to_table(recs: list[dict]) -> dict[str, np.ndarray]:
    if not recs:
        return {}
    keys = recs[0].keys()
    return {k: np.array([r[k] for r in recs]) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap_dir", type=Path, required=True, help="BIND composite_slab dir")
    ap.add_argument("--truth_dir", type=Path, default=None, help="truth composite_slab dir")
    ap.add_argument("--group_catalog", type=Path, required=True)
    ap.add_argument("--snapshot_index", type=int, required=True)
    ap.add_argument("--transforms", type=Path, default=None)
    ap.add_argument("--snap_idx", type=int, default=None)
    ap.add_argument("--mass_lo", type=float, default=1e12)
    ap.add_argument("--mass_hi", type=float, default=1e13)
    ap.add_argument("--r200_factor", type=float, default=4.0)
    ap.add_argument("--full_patch", action="store_true",
                    help="use full 6.25 Mpc/h footprint (reuse-only ceiling)")
    ap.add_argument("--out", type=Path, default=None, help="save comparison .npz")
    args = ap.parse_args()

    tf = LightconeTransforms.load(args.transforms) if args.transforms else None
    common = dict(group_catalog=args.group_catalog, snapshot_index=args.snapshot_index,
                  transforms=tf, snap_idx=args.snap_idx, mass_lo=args.mass_lo,
                  mass_hi=args.mass_hi, r200_factor=args.r200_factor,
                  full_patch=args.full_patch)

    b = measure(args.snap_dir, **common)
    print(f"[BIND ] band {args.mass_lo:.0e}-{args.mass_hi:.0e}: {b['n_band']:,} halos  "
          f"captured {b['n_captured']:,} ({b['capture_frac']*100:.1f}%)  "
          f"[{'full-patch' if args.full_patch else f'{args.r200_factor}xR200'}]")
    bt = _to_table(b["records"])
    out: dict[str, np.ndarray] = {f"bind_{k}": v for k, v in bt.items()}

    if args.truth_dir is not None:
        t = measure(args.truth_dir, **common)
        tt = _to_table(t["records"])
        # match BIND<->truth by halo_id (same FoF catalog / ordering)
        bid, tid = bt["halo_id"], tt["halo_id"]
        common_ids = np.intersect1d(bid, tid)
        bi = {h: i for i, h in enumerate(bid)}
        ti = {h: i for i, h in enumerate(tid)}
        bidx = np.array([bi[h] for h in common_ids])
        tidx = np.array([ti[h] for h in common_ids])
        out = {f"bind_{k}": v[bidx] for k, v in bt.items()}
        out.update({f"truth_{k}": v[tidx] for k, v in tt.items()})
        print(f"[truth] captured {t['n_captured']:,} ({t['capture_frac']*100:.1f}%); "
              f"matched both sides: {len(common_ids):,}")
        for fld in ("gas", "y", "star"):
            if f"bind_{fld}" in out and f"truth_{fld}" in out:
                bb, tv = out[f"bind_{fld}"], out[f"truth_{fld}"]
                ok = (tv > 0) & (bb > 0)
                if ok.sum() > 10:
                    lr = np.log10(bb[ok] / tv[ok])
                    print(f"   {fld:>5}: BIND/truth median={10**np.median(lr):.3f}  "
                          f"scatter={np.std(lr):.3f} dex  (N={ok.sum():,})")

    if args.out:
        np.savez_compressed(args.out, **out,
                            meta=json.dumps({"mass_lo": args.mass_lo, "mass_hi": args.mass_hi,
                                             "r200_factor": args.r200_factor,
                                             "full_patch": args.full_patch}))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
