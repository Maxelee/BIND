"""Shared paths, constants and the no-overwrite guard for the fiducial repaint.

Imported by ``make_corrected_params.py`` and ``verify_repaint.py``; the shell
equivalent lives in ``repaint_env.sh`` (keep the two in sync).

The campaign never writes into a released tree.  Every write path is funnelled
through :func:`assert_writable`, which resolves symlinks (``/mnt/home/mlee1/ceph``
is a symlink to ``/mnt/sdceph/users/mlee1``) on *both* sides before comparing, so
the guard cannot be dodged by an alias.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── trees that are READ-ONLY for this campaign ────────────────────────────────
# Nothing in repaint/ ever writes into any of these.  bind_n1000 stays on the list
# even though the plan calls for a FORCE=1 redo of ``bind_n1000/bind/run_0000``
# after the repaint lands: that redo is performed by n1000_body.sh (with
# ``FID=$REPAINT_ROOT``), which owns that tree and has its own claim arbitration —
# it is NOT a write this campaign's scripts are allowed to make.  See
# docs/fiducial_repaint_plan.md §10.5 for the sequencing constraint (the redo waits
# until the running 319-task N1000 campaign drains, so a FORCE=1 ``rm -rf $CLAIM``
# cannot race live claims).
RELEASED_TREES = [
    Path("/mnt/home/mlee1/ceph/bind_lightcone_tng"),
    Path("/mnt/home/mlee1/ceph/bind_science"),
    Path("/mnt/home/mlee1/ceph/bind_sb35"),
    Path("/mnt/home/mlee1/ceph/bind_n1000"),
    Path("/mnt/home/mlee1/ceph/tng_full_validation"),
]

# ── the released (buggy) fiducial lightcone and its correct-cosmology sibling ──
LC_OLD = Path(os.environ.get("LC_OLD", "/mnt/home/mlee1/ceph/bind_lightcone_tng"))
# bind_science/runs/fiducial/run_0000/params.npy is the verified-correct TNG300
# 35-dim vector (astro fiducial + TNG300 cosmology).  It is the bit-for-bit
# reference the corrected vector must reproduce.
PARAMS_REFERENCE = Path(os.environ.get(
    "PARAMS_REFERENCE",
    "/mnt/home/mlee1/ceph/bind_science/runs/fiducial/run_0000/params.npy",
))

# ── the NEW campaign tree (never inside a released tree) ───────────────────────
REPAINT_ROOT = Path(os.environ.get(
    "REPAINT_ROOT", "/mnt/home/mlee1/ceph/bind_lightcone_tng_fixed"))

# 20 lightcone snapshots, low-z → high-z (must match run_lightcone_project.sh)
SNAPSHOTS = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38,
             35, 33, 31, 29]

# The five cosmology entries of the 35-dim SB35 vector that were wrong.
# TNG300-1 / IllustrisTNG cosmology (Planck 2015 XIII); Omega0, OmegaBaryon and
# HubbleParam are also carried in the TNG300-Dark snapshot Header, sigma8 and
# n_s are not (they cannot be header-checked — see docs/fiducial_repaint_plan.md).
TNG300_COSMOLOGY = {
    "Omega0": 0.3089,        # idx 0   (header Header/Omega0)
    "sigma8": 0.8159,        # idx 1   (not in header)
    "OmegaBaryon": 0.0486,   # idx 6   (header Header/OmegaBaryon)
    "HubbleParam": 0.6774,   # idx 7   (header Header/HubbleParam)
    "n_s": 0.9667,           # idx 8   (not in header)
}
COSMO_INDICES = [0, 1, 6, 7, 8]

# Ob/Om of the released (wrong) conditioning vs the correct one.  The painted
# gas/tau amplitude scales ~linearly with this ratio, plane power with its square.
FB_WRONG = 0.049 / 0.300
FB_RIGHT = TNG300_COSMOLOGY["OmegaBaryon"] / TNG300_COSMOLOGY["Omega0"]
FB_RATIO = FB_WRONG / FB_RIGHT              # 1.038141
POWER_RATIO_EXPECTED = 1.0 / FB_RATIO ** 2  # 0.927887  (new / old gas plane power)


def _resolve(p: Path | str) -> Path:
    """Resolve symlinks/.. for a path that need not exist yet."""
    return Path(os.path.realpath(os.path.abspath(str(p))))


def is_inside_released(path: Path | str) -> Path | None:
    """Return the released tree containing *path*, or None."""
    r = _resolve(path)
    for tree in RELEASED_TREES:
        t = _resolve(tree)
        if r == t or str(r).startswith(str(t) + os.sep):
            return tree
    return None


def assert_writable(path: Path | str, what: str = "output") -> Path:
    """Raise unless *path* is outside every released tree.  Returns the path."""
    tree = is_inside_released(path)
    if tree is not None:
        raise SystemExit(
            f"REFUSING TO WRITE: {what} path {path} resolves inside the released "
            f"tree {tree}.\nThe repaint campaign is strictly additive — set "
            f"REPAINT_ROOT to a fresh tree (default {REPAINT_ROOT})."
        )
    return Path(path)


def snap_dir(root: Path | str, snap: int) -> Path:
    return Path(root) / f"snap_{snap:03d}"


def load_params(path: Path | str) -> np.ndarray:
    a = np.load(str(path))
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size != 35:
        raise ValueError(f"{path}: expected 35 params, got {a.size}")
    return a
