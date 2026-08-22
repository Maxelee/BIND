#!/usr/bin/env python3
"""fidswap -- per-halo radial profiles + halo-population medians for the NEW fiducial.

Part A: build_perhalo_profiles.py::reduce_slab VERBATIM, run on a twobound replica.
        Writes referee_work/fidswap/profiles/perhalo_fidtb{run}_snap096.npz, a
        drop-in for bind_science/profiles/perhalo_fid_snap096.npz.  The paired
        guards (identical halo list / npix vs truth) are enforced as in the
        shipped script; the truth leg is unaffected and reused from the shipped
        cache.

Part B: the halo-level median ratios main.tex:502 quotes -- Y_200c, tau_200c
        (= m_gas_200c), M_star,200c, plus the f_gas / thermo companions -- with
        bootstrap SEs, for the old fiducial and every replica.  Verified against
        the shipped numbers (1.037 / 1.065 / 0.943) on the old fiducial first.

    python referee/work/fs7_profiles.py --run 49
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
sys.path.insert(0, str(P1))
from build_perhalo_profiles import N_X, X_EDGES, reduce_slab  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
FS = CEPH / "referee_work/fidswap"
ATL = SCI / "halo_atlas"
SNAP = 96
REPLICAS = (18, 49, 53)

# (label, atlas key) for the halo-population medians
RATIOS = [("Y_200c", "Y_200c"), ("tau_200c (m_gas_200c)", "m_gas_200c"),
          ("M_star_200c", "m_star_200c"), ("f_gas_200c_bg", None),
          ("f_gas_500c_bg", None), ("Y_500c", "Y_500c"),
          ("T_mw_500c", "T_mw_500c"), ("Pe_mw_500c", "Pe_mw_500c")]


def _field(z, name):
    if name == "f_gas_200c_bg":
        return np.asarray(z["m_gas_200c_bg"], float) / np.asarray(z["m_tot_200c_bg"], float)
    if name == "f_gas_500c_bg":
        return np.asarray(z["m_gas_500c_bg"], float) / np.asarray(z["m_tot_500c_bg"], float)
    return np.asarray(z[name], float)


def med_ratio(a, b, n_boot=2000, seed=0):
    r = a / b
    ok = np.isfinite(r)
    r = r[ok]
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(rng.choice(r, len(r))) for _ in range(n_boot)])
    return float(np.median(r)), float(bs.std(ddof=1)), int(len(r))


def medians(atlas_path: Path, truth: dict, tag: str):
    z = np.load(atlas_path)
    print(f"\n{tag}  ({atlas_path.name})")
    print(f"{'quantity':<24s} {'median BIND/truth':>18s} {'boot SE':>9s} "
          f"{'bias %':>8s} {'n_sig':>7s} {'n_halo':>7s}")
    out = {}
    for label, key in RATIOS:
        name = key or label
        m, se, n = med_ratio(_field(z, name), truth[name])
        out[label] = (m, se, n)
        print(f"{label:<24s} {m:18.4f} {se:9.4f} {100*(m-1):+8.2f} "
              f"{abs(m-1)/se:7.1f} {n:7d}")
    return out


def profiles(run: int):
    src = SCI / f"runs/twobound/run_{run:04d}"
    t0 = time.time()
    slabs = sorted((src / f"snap_{SNAP:03d}").glob("composite_slab*.npz"))
    parts = [r for r in (reduce_slab(s) for s in slabs) if r is not None]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(cat["M"]) == 2933, f"{len(cat['M'])} halos != atlas 2933"
    ref = np.load(SCI / f"profiles/perhalo_truth_snap{SNAP:03d}.npz")
    assert np.allclose(cat["M"], ref["M"]), "halo list differs from the truth leg"
    assert np.allclose(cat["r200"], ref["r200"])
    assert np.array_equal(cat["npix"], ref["npix"])
    cat["x_edges"] = X_EDGES
    cat["z"], cat["snap"] = ref["z"], ref["snap"]
    (FS / "profiles").mkdir(parents=True, exist_ok=True)
    f = FS / f"profiles/perhalo_fidtb{run:02d}_snap{SNAP:03d}.npz"
    np.savez_compressed(f, **cat)
    fin = np.isfinite(cat["prof_y"]).mean(0)
    print(f"[tb{run}] {time.time()-t0:.1f}s  wrote {f.name}: {len(cat['M'])} halos x "
          f"{N_X} annuli, paired guards vs truth PASS; finite frac min/med "
          f"{fin.min():.2f}/{np.median(fin):.2f}")
    return cat


def profile_ratios(cat, tag):
    ref = np.load(SCI / f"profiles/perhalo_truth_snap{SNAP:03d}.npz")
    lm = np.log10(np.asarray(cat["M"], float))
    xc = np.sqrt(X_EDGES[1:] * X_EDGES[:-1])
    inner = xc <= 1.0
    print(f"\n{tag}: median BIND/truth of the stacked profile, r <= R200c")
    print(f"{'mass bin':<16s} " + " ".join(f"{c:>9s}" for c in ("DM", "gas", "star", "y")))
    for lo, hi in ((13.0, 13.5), (13.5, 14.0), (14.0, 15.5)):
        sel = (lm >= lo) & (lm < hi)
        row = []
        for ch in ("DM", "gas", "star", "y"):
            a = np.nanmedian(cat[f"prof_{ch}"][sel][:, inner], 0)
            b = np.nanmedian(ref[f"prof_{ch}"][sel][:, inner], 0)
            row.append(np.nanmedian(a / b))
        print(f"{f'{lo}-{hi} ({sel.sum()})':<16s} " + " ".join(f"{v:9.4f}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=49)
    a = ap.parse_args()

    tz = np.load(ATL / f"truth_snap{SNAP:03d}.npz")
    truth = {n or l: _field(tz, n or l) for l, n in RATIOS}
    medians(ATL / f"fid_snap{SNAP:03d}.npz", truth,
            "OLD FIDUCIAL (bind_lightcone_tng)")
    for r in REPLICAS:
        p = FS / f"halo_atlas/fidtb{r:02d}_snap{SNAP:03d}.npz"
        if p.exists():
            medians(p, truth, f"NEW FIDUCIAL twobound/run_{r:04d}")

    cat = profiles(a.run)
    profile_ratios(cat, f"twobound/run_{a.run:04d}")
    old = {k: np.load(SCI / f"profiles/perhalo_fid_snap{SNAP:03d}.npz")[k]
           for k in ("M", "prof_DM", "prof_gas", "prof_star", "prof_y")}
    profile_ratios(old, "OLD FIDUCIAL")


if __name__ == "__main__":
    main()
