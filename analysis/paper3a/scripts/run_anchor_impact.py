"""Does merging the bind (0,0) anchor into L_B move the joint posterior?

The anchored 61-unit grid now matches B5's training set. That changes
L_B slightly (the fiducial chi2 anchor goes 445.3 -> 443.6 against B5's
recorded 443.8, and the dilution profile unit switches from
twobound/run_0027 to bind). The question this answers is whether a full
re-fit is warranted or the change is immaterial.

Method: importance reweighting of the frozen joint chain under the
anchored B block, w = exp(L_B,anchored - L_B,frozen). This is the
well-behaved direction for IS -- it is a mild re-training, not a
covariance widening, so the proposal and target have similar width
(contrast the A8 grid, where three error-widening variants collapsed to
ESS 24-141 and had to go to fresh chains).

Run: python analysis/paper3a/scripts/run_anchor_impact.py
Out: wp6_propagation/anchor_impact.json
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator.statsemu import WP6  # noqa: E402
from analysis.paper3a.inference.sampler import CHAINS  # noqa: E402

ANCHORED = ("/mnt/ceph/users/mlee1/paper3/B/wp4_mocks/"
            "model_grid_tfwiener_anchored.npz")
N_SUB = 8_000
ESS_MIN = 500.0


def coords_and_logl(grid_env):
    """(coords, coord_err, logL_B) for the subsample under one grid."""
    if grid_env is None:
        os.environ.pop("BIND_PAPER3A_BGRID", None)
    else:
        os.environ["BIND_PAPER3A_BGRID"] = grid_env
    import analysis.paper3a.inference.bblock as bb
    import analysis.paper3a.inference.jointab as ja
    importlib.reload(bb)
    importlib.reload(ja)
    jb = ja.JointBBlock()

    flat = np.concatenate(
        [np.load(CHAINS / f"joint_ab_seed{k}.npz")["chain"]
         .astype(float).reshape(-1, 32) for k in range(4)])
    rng = np.random.default_rng(7)
    U = flat[rng.choice(len(flat), N_SUB, replace=False)]
    c, e = jb.coords(U[:, :30])
    lb = jb.b.loglike_batch(c, e, U[:, 31] * 3.6)
    return U, c, lb, jb.b.train_names, jb.b.profile_unit


def wq(x, w, q):
    i = np.argsort(x)
    cc = np.cumsum(w[i])
    return float(np.interp(q * cc[-1], cc, x[i]))


def main() -> None:
    U, c, lb_frozen, names0, prof0 = coords_and_logl(None)
    _, _, lb_anch, names1, prof1 = coords_and_logl(ANCHORED)

    dl = lb_anch - lb_frozen
    w = np.exp(dl - dl.max())
    ess = float(w.sum() ** 2 / (w ** 2).sum())

    out = {"n_train_frozen": len(names0), "n_train_anchored": len(names1),
           "profile_unit_frozen": prof0, "profile_unit_anchored": prof1,
           "ess": round(ess, 1), "reliable": bool(ess >= ESS_MIN),
           "delta_logL_B": {"median": float(np.median(dl)),
                            "p16": float(np.percentile(dl, 16)),
                            "p84": float(np.percentile(dl, 84))},
           "coords": {}}
    for j, nm in enumerate(("dln_mgas", "dln_t")):
        x = c[:, j]
        f = (wq(x, np.ones(len(x)), 0.16), wq(x, np.ones(len(x)), 0.50),
             wq(x, np.ones(len(x)), 0.84))
        a = (wq(x, w, 0.16), wq(x, w, 0.50), wq(x, w, 0.84))
        sig = 0.5 * (f[2] - f[0])
        out["coords"][nm] = {
            "frozen_p16_p50_p84": [round(v, 4) for v in f],
            "anchored_p16_p50_p84": [round(v, 4) for v in a],
            "shift_in_sigma": round((a[1] - f[1]) / sig, 3)}

    worst = max(abs(v["shift_in_sigma"]) for v in out["coords"].values())
    out["max_abs_shift_sigma"] = worst
    out["verdict"] = ("re-fit NOT warranted (shift < 0.5 sigma)"
                      if worst < 0.5 and ess >= ESS_MIN else
                      "re-fit WARRANTED" if ess >= ESS_MIN else
                      "INCONCLUSIVE — ESS collapsed, needs fresh chains")
    (WP6 / "anchor_impact.json").write_text(json.dumps(out, indent=2))

    print(f"training set: {len(names0)} -> {len(names1)} units")
    print(f"dilution profile unit: {prof0} -> {prof1}")
    print(f"\ndelta logL_B median {np.median(dl):+.3f} "
          f"[{np.percentile(dl,16):+.3f}, {np.percentile(dl,84):+.3f}]")
    print(f"ESS = {ess:.0f} / {N_SUB}  -> "
          f"{'usable' if ess >= ESS_MIN else 'COLLAPSED'}")
    print()
    for nm, v in out["coords"].items():
        print(f"  {nm:9s} frozen {v['frozen_p16_p50_p84'][1]:+.4f}  "
              f"anchored {v['anchored_p16_p50_p84'][1]:+.4f}  "
              f"shift {v['shift_in_sigma']:+.3f} sigma")
    print(f"\nVERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()
