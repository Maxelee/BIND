"""Merge the bind (0,0) anchor row into the twobound model grid.

B5 trained L_B on 61 units: the 60 twobound units plus the **bind
(0,0) anchor** — the bind reference run itself, which sits at the
origin of the coordinate system by construction (every delta in
`sb35_coords.npz` is measured "vs bind/run_0000"). The rusty mirror had
only the 60, which is the disclosed departure recorded in
`bblock.py`'s docstring and quantified by its chi2-anchor validation:
the `corner` anchor reproduced at **172.4 against B5's recorded
161.8** (ratio 1.065), while `fiducial` and `run_0004` agreed to 1.003.

`bind_run_0000.npz` has now been rsync'd, so this script builds the
61-unit table. It writes a NEW file and never touches the frozen grid;
`bblock` picks it up via $BIND_PAPER3A_BGRID.

Checks performed before merging (all must pass):
  - CAP radii identical to the frozen stack's,
  - nu edges give the 5 bins the grid's nbin declares,
  - y_mean/y_mc_err shapes match a grid row,
  - the recipe matches the model freeze (transfer=wiener,
    weight_scheme=default, n_seeds=32),
  - no existing unit already sits at (0,0).

`params` and `abundance_per_deg2` are filled with NaN for the anchor: it
is the bind reference, not a CAMELS design unit, so it HAS no 35-vector
(the source file stores an empty array). `BBlock` consumes only
run_names / delta_ln_* / y_mean / y_mc_err, so this is inert — but the
NaNs are deliberate, so anything that later reaches for the anchor's
params fails loudly instead of silently reading a zero.

Run: python analysis/paper3a/scripts/make_anchored_grid.py
Out: wp4_mocks/model_grid_tfwiener_anchored.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))

B = Path("/mnt/ceph/users/mlee1/paper3/B")
GRID = B / "wp4_mocks/model_grid_tfwiener.npz"
ANCHOR = B / "wp4_mocks/grid_tfwiener/bind_run_0000.npz"
STACK = B / "wp2_measurement/stack_wiener_sm2am_fid.npz"
OUT = B / "wp4_mocks/model_grid_tfwiener_anchored.npz"


def main() -> None:
    g = np.load(GRID, allow_pickle=True)
    a = np.load(ANCHOR, allow_pickle=True)
    s = np.load(STACK, allow_pickle=True)

    # ---- gates -------------------------------------------------------
    if not np.allclose(a["cap_radii_arcmin"], s["cap_radii_arcmin"]):
        raise SystemExit("CAP radii differ between anchor and frozen stack")
    nbin, nrad = int(g["nbin"]), int(g["nrad"])
    if len(a["nu_edges"]) - 1 != nbin:
        raise SystemExit(f"anchor gives {len(a['nu_edges'])-1} nu bins, "
                         f"grid declares {nbin}")
    if a["y_mean"].shape != (nbin, nrad):
        raise SystemExit(f"anchor y_mean {a['y_mean'].shape} != "
                         f"({nbin}, {nrad})")
    recipe = (str(a["transfer"]), str(a["weight_scheme"]), int(a["n_seeds"]))
    if recipe != ("wiener", "default", 32):
        raise SystemExit(f"anchor recipe {recipe} != the frozen "
                         "(wiener, default, 32)")
    if bool(((g["delta_ln_mgas"] == 0) & (g["delta_ln_t"] == 0)).any()):
        raise SystemExit("a unit already sits at (0,0) — refusing to add "
                         "a second origin")

    n = len(g["run_names"])
    print(f"gates PASS — merging the bind anchor into {n} twobound units")

    out = {}
    for k in g.files:
        v = g[k]
        if v.ndim == 0:                       # scalars carry through
            out[k] = v
            continue
        if k == "run_names":
            out[k] = np.array(list(map(str, v)) + ["bind"])
        elif k in ("y_mean", "y_mc_err"):
            out[k] = np.concatenate([np.asarray(v, float),
                                     np.asarray(a[k], float)[None]], axis=0)
        elif k == "n_per_bin":
            out[k] = np.concatenate([np.asarray(v, float),
                                     np.asarray(a[k], float)[None]], axis=0)
        elif k in ("delta_ln_mgas", "delta_ln_t"):
            # the anchor IS the reference: every delta is measured
            # against it, so it sits at exactly 0 by construction
            out[k] = np.concatenate([np.asarray(v, float), [0.0]])
        elif k in ("params", "abundance_per_deg2"):
            pad = np.full((1,) + np.asarray(v).shape[1:], np.nan)
            out[k] = np.concatenate([np.asarray(v, float), pad], axis=0)
        else:
            raise SystemExit(f"unhandled grid key {k} — refusing to guess")

    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}")
    print(f"  units: {n} -> {len(out['run_names'])}  "
          f"(last = {out['run_names'][-1]})")
    print(f"  anchor coords: ({out['delta_ln_mgas'][-1]}, "
          f"{out['delta_ln_t'][-1]})")
    print(f"  anchor y_mean @4' [1e6]: "
          f"{np.round(out['y_mean'][-1][:, 2] * 1e6, 3)}")
    print("\nvalidate with:")
    print(f"  BIND_PAPER3A_BGRID={OUT} \\")
    print("    python -m analysis.paper3a.inference.bblock")


if __name__ == "__main__":
    main()
