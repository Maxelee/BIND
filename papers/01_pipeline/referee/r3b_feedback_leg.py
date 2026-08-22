#!/usr/bin/env python3
"""R3b feedback leg: does the sampler's stochastic texture grow with feedback?

The direct version of this test -- repaint N times at a strong-feedback node --
needs a GPU, and the assigned node has none (see the memo).  What *is* possible
with existing data is a calibrated extrapolation, because the twobound campaign
painted every parameter extreme on the SAME halos:

  1. At the fiducial ensemble (twobound 0018/0049/0053, three independent draws)
     measure the per-halo stochastic fraction f = sigma_draw / rms and calibrate
     how it depends on halo mass and on gas diffuseness.
  2. At each extreme-feedback node a single draw is enough to measure the halo
     gas diffuseness (no ensemble needed).
  3. Push the measured diffuseness shift through the fiducial-calibrated relation
     to predict how much the stochastic texture would grow at that node.

Assumption stated plainly: the f(mass, diffuseness) relation is itself taken to be
parameter-independent.  That is the first-order prediction, not a measurement -- it
brackets the size of the effect the referee is worried about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TWOBOUND = Path("/mnt/home/mlee1/ceph/bind_science/runs/twobound")
STAGE1 = Path("/mnt/home/mlee1/ceph/bind_lightcone_tng/snap_096/stage1")
WORK = Path("/mnt/home/mlee1/ceph/referee_work/texture")

ENSEMBLE = [18, 49, 53]
GAS_CH, Y_CH = 1, 0
PATCH_MPCH = 6.25

# extreme-feedback nodes (index, label) resolved from twobound_params.npy
NODES = {
    "fiducial(ens)": None,
    "VarWindVel_max": 5,
    "VarWindVel_min": 4,
    "WindEnergy_max": 1,
    "WindEnergy_min": 0,
    "RadioFB_max": 3,
    "RadioFB_min": 2,
}
SLABS = [0, 1, 2, 3]


def load_patches(run: int, slab: int):
    d = np.load(TWOBOUND / f"run_{run:04d}" / f"snap_096/composite_slab{slab:02d}.npz")
    return (
        d["generated_patches"][:, GAS_CH].astype(np.float32),
        d["thermo_patches"][:, Y_CH].astype(np.float32),
    )


def diffuseness(gas_patches: np.ndarray, r200: np.ndarray) -> np.ndarray:
    """1 - M_gas(<R200) / M_gas(patch): fraction of in-patch gas outside R200."""
    n, pp, _ = gas_patches.shape
    yy, xx = np.mgrid[:pp, :pp]
    rr = np.sqrt((xx - pp / 2 + 0.5) ** 2 + (yy - pp / 2 + 0.5) ** 2)
    pix = PATCH_MPCH / pp
    out = np.empty(n)
    for i in range(n):
        m = rr <= (float(r200[i]) / pix)
        tot = gas_patches[i].sum()
        out[i] = 1.0 - gas_patches[i][m].sum() / (tot + 1e-30)
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    mass, r200, f_gas, f_y, diff_fid = [], [], [], [], []

    # ── 1. fiducial ensemble: per-halo stochastic fraction + diffuseness ─────
    for slab in SLABS:
        s1 = np.load(STAGE1 / f"stage1_slab{slab:02d}.npz")
        G, T = [], []
        for r in ENSEMBLE:
            g, t = load_patches(r, slab)
            G.append(g)
            T.append(t)
        G = np.stack(G)
        T = np.stack(T)
        n = G.shape[1]
        gm, tm = G.mean(0), T.mean(0)
        gv = ((G - gm) ** 2).sum(0) / (len(ENSEMBLE) - 1)
        tv = ((T - tm) ** 2).sum(0) / (len(ENSEMBLE) - 1)
        f_gas.append(np.sqrt(gv.reshape(n, -1).mean(1)) / (np.sqrt((gm**2).reshape(n, -1).mean(1)) + 1e-30))
        f_y.append(np.sqrt(tv.reshape(n, -1).mean(1)) / (np.sqrt((tm**2).reshape(n, -1).mean(1)) + 1e-30))
        mass.append(s1["halo_masses"])
        r200.append(s1["halo_r200"])
        diff_fid.append(diffuseness(gm, s1["halo_r200"]))
        print(f"  fiducial ensemble slab {slab}: {n} halos", flush=True)

    mass = np.concatenate(mass)
    r200 = np.concatenate(r200)
    f_gas = np.concatenate(f_gas)
    f_y = np.concatenate(f_y)
    diff_fid = np.concatenate(diff_fid)
    lm = np.log10(mass)

    # ── 2. calibrate f(mass, diffuseness) in mass bins ──────────────────────
    edges = np.quantile(lm, np.linspace(0, 1, 7))
    edges[-1] += 1e-6
    cal = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lm >= a) & (lm < b)
        if m.sum() < 25:
            cal.append(None)
            continue
        e = {}
        for nm, f in (("gas", f_gas), ("y", f_y)):
            A = np.vstack([np.ones(m.sum()), diff_fid[m]]).T
            coef, *_ = np.linalg.lstsq(A, f[m], rcond=None)
            e[nm] = coef
        e["lo"], e["hi"] = a, b
        e["mean_diff"] = float(diff_fid[m].mean())
        cal.append(e)

    def predict(diff_arr):
        out = {"gas": np.full(len(diff_arr), np.nan), "y": np.full(len(diff_arr), np.nan)}
        for e in cal:
            if e is None:
                continue
            m = (lm >= e["lo"]) & (lm < e["hi"])
            for nm in ("gas", "y"):
                c = e[nm]
                out[nm][m] = c[0] + c[1] * diff_arr[m]
        return out

    base = predict(diff_fid)
    summary = {
        "n_halos": int(len(lm)),
        "fiducial_measured": {
            "f_gas_median": float(np.median(f_gas)),
            "f_y_median": float(np.median(f_y)),
            "diffuse_mean": float(diff_fid.mean()),
        },
        "calibration_slopes": [
            None if e is None
            else {"logM": [float(e["lo"]), float(e["hi"])],
                  "gas_intercept_slope": [float(e["gas"][0]), float(e["gas"][1])],
                  "y_intercept_slope": [float(e["y"][0]), float(e["y"][1])]}
            for e in cal
        ],
        "nodes": {},
    }

    # ── 3. extreme-feedback nodes: measure diffuseness, predict f ────────────
    for label, run in NODES.items():
        if run is None:
            continue
        dif = []
        for slab in SLABS:
            s1 = np.load(STAGE1 / f"stage1_slab{slab:02d}.npz")
            g, _ = load_patches(run, slab)
            dif.append(diffuseness(g, s1["halo_r200"]))
        dif = np.concatenate(dif)
        pred = predict(dif)
        summary["nodes"][label] = {
            "run": run,
            "diffuse_mean": float(dif.mean()),
            "diffuse_shift": float(dif.mean() - diff_fid.mean()),
            "pred_f_gas_ratio": float(np.nanmean(pred["gas"]) / np.nanmean(base["gas"])),
            "pred_f_y_ratio": float(np.nanmean(pred["y"]) / np.nanmean(base["y"])),
        }
        print(
            f"  {label:>16s} run_{run:04d}: <diffuse>={dif.mean():.4f} "
            f"(fid {diff_fid.mean():.4f})  pred f_gas x{summary['nodes'][label]['pred_f_gas_ratio']:.3f} "
            f" f_y x{summary['nodes'][label]['pred_f_y_ratio']:.3f}",
            flush=True,
        )

    np.savez_compressed(
        WORK / "r3b_feedback_leg.npz",
        lm=lm, f_gas=f_gas, f_y=f_y, diff_fid=diff_fid, mass=mass, r200=r200,
    )
    (WORK / "r3b_feedback_leg.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["nodes"], indent=2))
    print("wrote", WORK / "r3b_feedback_leg.json")


if __name__ == "__main__":
    main()
