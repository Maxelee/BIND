"""Which realization campaign a figure is built against — 50-real or N=1000.

The paper's figures read from two parallel trees with identical layout:

    sci50  (default)  ceph/bind_science/runs/...   ceph/bind_sb35/runs/...   50 reals
    n1000             ceph/bind_n1000/...                                  1000 reals

Set ``BIND_CAMPAIGN=n1000`` to point the builders at the campaign tree; every
builder that imports from here follows.  This mirrors the existing
``BIND_FID_REPLICA`` swap in ``papers/01_pipeline/referee/work/fsfig_common.py``
(the two axes compose: replica x campaign).

NOT everything has an n1000 twin.  Per-halo products (halo_atlas, profiles,
composite patches) and the assembled Sobol ``emulator_dataset*.npz`` exist only
for the 50-real tree, and the campaign is still tracing, so most Sobol/1P runs
have no stats yet.  Call :func:`require` in a builder to fail loudly with a
useful message instead of silently mixing realization counts.

    from campaign_roots import CAMPAIGN, SCI_ROOT, SB35_ROOT, IMGS_DIR, tag
"""
from __future__ import annotations

import os
from pathlib import Path

CEPH = Path("/mnt/home/mlee1/ceph")
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
if CAMPAIGN not in ("sci50", "n1000"):
    raise SystemExit(f"BIND_CAMPAIGN must be sci50|n1000, got {CAMPAIGN!r}")

N1000 = CEPH / "bind_n1000"
_SCI50 = CEPH / "bind_science"
_SB3550 = CEPH / "bind_sb35"


def is_n1000() -> bool:
    return CAMPAIGN == "n1000"


def SCI_ROOT() -> Path:
    """Root under which ``runs/{bind,dmo,truth,twobound}/run_NNNN`` live."""
    return N1000 if is_n1000() else _SCI50


def run_dir(category: str, index: int = 0) -> Path:
    """Path to one run. Handles the two trees' different nesting.

    sci50:  bind_science/runs/twobound/run_0049
    n1000:  bind_n1000/twobound/run_0049          (no 'runs/' level)
    """
    name = f"run_{index:04d}"
    if is_n1000():
        return N1000 / category / name
    return _SCI50 / "runs" / category / name


def SB35_ROOT() -> Path:
    """Root under which the Sobol ``run_NNNN`` dirs live."""
    return N1000 / "sb35" if is_n1000() else _SB3550 / "runs"


def IMGS_DIR() -> Path:
    """Where the paper images for this campaign are staged."""
    base = Path("/mnt/home/mlee1/BIND/BINDing_the_lightcone")
    return base / ("imgs_1000" if is_n1000() else "imgs")


def tag() -> str:
    """Filename tag so a campaign's outputs can never overwrite the other's."""
    return "_n1000" if is_n1000() else ""


def require(*paths: Path, what: str = "") -> None:
    """Fail loudly (not silently on stale/50-real data) if inputs are missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"[campaign_roots] BIND_CAMPAIGN={CAMPAIGN} — missing input(s)"
            + (f" for {what}" if what else "") + ":\n  "
            + "\n  ".join(missing)
            + "\nThis figure cannot be built for this campaign yet."
        )
