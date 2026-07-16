"""Stage-1 frame transforms for TNG300 truth projection (WP-A2 Popeye half).

The BIND generation bundle (``bind_portable_twobound``, the SB35 Sobol bundle,
and the fiducial paint) builds its DMO *conditions* by applying a per-snapshot
geometric transform to the particle box before projecting into z-slabs — a
random axis permutation (which original axis is the line of sight), a periodic
displacement, and per-axis flips. The painted patches therefore live in that
*transformed* frame. To validate painted-vs-truth (task 5) the TNG300-hydro
projection must apply the **same** transform and slab decomposition, or the
halos will not register (NEXT_POPEYE_SESSION.md workstream 1, step 1: "Frame
matching is the trap to avoid").

The transform values are stored per snapshot in each condition directory's
``stage1_manifest.json`` (keys ``proj_dir``, ``disp``, ``flip``,
``slab_depth``, ``box_size``, ``npix``, ...), so this module only needs to
*apply* them — it does not regenerate them. The application convention is
copied verbatim from ``bind.inference.lightcone_transforms.LightconeTransforms``
on the BIND ``lightcone`` branch (the code that produced the conditions); see
that module's docstring. It is reproduced here — pure numpy, no bind import —
because the observables package must run in the Popeye ``paper3b_popeye`` venv,
whose ``bind`` is installed from a ``main``-derived branch that does not carry
``lightcone_transforms``.

Transform applied to each particle for a snapshot:
  1. Permute axes via ``proj_dir``: choose which original axis is the LOS.
  2. Displace: ``new[ax] = pos[original_ax] + disp[original_ax]``.
  3. Flip:  ``if flip[original_ax]: new[ax] = -new[ax]``.
  4. Periodic wrap to ``[0, box_size)``.
Output columns: 0 = transverse x, 1 = transverse y, 2 = LOS.

``disp`` and ``flip`` are indexed by *original* (pre-permutation) axis, matching
the manifest and lux's ``disp[j + 3*s]`` storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (ix, iy, iz): original-axis indices mapping to (transverse_x, transverse_y, LOS).
# Verbatim from bind.inference.lightcone_transforms.PROJ_DIR_AXES (lightcone branch).
PROJ_DIR_AXES: dict[int, tuple[int, int, int]] = {
    0: (1, 2, 0),  # LOS along original x → transverse = (y, z)
    1: (2, 0, 1),  # LOS along original y → transverse = (z, x)
    2: (0, 1, 2),  # LOS along original z → transverse = (x, y)
}


def apply_frame_transform(
    pos_mpch: np.ndarray,
    proj_dir: int,
    disp: np.ndarray,
    flip: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """Apply one snapshot's stage-1 transform to particle/halo positions.

    Parameters
    ----------
    pos_mpch : (N, 3) float
        Positions in Mpc/h, columns = (x, y, z) in the *original* box frame.
    proj_dir : int
        Projection direction 0/1/2 (the ``proj_dir`` manifest value).
    disp : (3,) float
        Periodic displacement in Mpc/h, indexed by original axis.
    flip : (3,) bool
        Per-axis flip flags, indexed by original axis.
    box_size : float
        Periodic box side in Mpc/h.

    Returns
    -------
    (N, 3) float32
        Transformed positions: col 0/1 = transverse x/y, col 2 = LOS; all
        wrapped to ``[0, box_size)``. Identical convention to the condition
        maps, so ``pixelize_z_projection`` (which projects cols 0,1) and
        z-slabbing (on col 2) reproduce the painted frame.
    """
    pos = np.asarray(pos_mpch, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos_mpch must be (N, 3); got {pos.shape}")
    d = np.asarray(disp, dtype=np.float64).reshape(3)
    f = np.asarray(flip).reshape(3).astype(bool)
    ix, iy, iz = PROJ_DIR_AXES[int(proj_dir)]

    # Displacement uses the *original*-axis index (matching lux's disp[ix + 3*s]).
    x = pos[:, ix] + d[ix]
    y = pos[:, iy] + d[iy]
    z = pos[:, iz] + d[iz]
    if f[ix]:
        x = -x
    if f[iy]:
        y = -y
    if f[iz]:
        z = -z
    x %= box_size
    y %= box_size
    z %= box_size
    return np.stack([x, y, z], axis=1).astype(np.float32)


def assign_slabs(los_mpch: np.ndarray, box_size: float, n_slabs: int) -> np.ndarray:
    """Assign LOS coordinates to z-slabs (mirrors paint._assign_halos_to_slabs).

    slab_h = box_size / n_slabs; slab index = clip(floor(los / slab_h), 0,
    n_slabs-1). The transverse projection of a slab integrates ``slab_h`` Mpc/h
    of material along the LOS — the projection depth the CylToSph correction is
    locked to.
    """
    slab_h = box_size / n_slabs
    los = np.asarray(los_mpch, dtype=np.float64)
    return np.clip((los / slab_h).astype(np.int64), 0, n_slabs - 1)


@dataclass(frozen=True)
class Stage1Manifest:
    """Parsed ``stage1_manifest.json`` — the frame + geometry of one condition set.

    Only the fields the truth projection needs are surfaced as attributes; the
    full dict is kept in ``raw``.
    """

    proj_dir: int
    disp: np.ndarray          # (3,) Mpc/h, original-axis indexed
    flip: np.ndarray          # (3,) bool, original-axis indexed
    box_size: float           # Mpc/h
    npix: int
    n_slabs: int
    slab_depth: float         # Mpc/h
    snapshot_index: int
    redshift: float
    scale_factor: float
    halo_mass_field: str
    dmo_snapshot: str         # path recorded at condition-build time (DMO)
    dmo_group_catalog: str
    raw: dict

    @property
    def slab_depth_hmpc(self) -> float:
        """Projection depth per slab (box_size / n_slabs) — locks CylToSph."""
        return self.box_size / self.n_slabs

    def hydro_snapdir(self) -> str:
        """Derive the TNG300-*hydro* snapshot dir from the recorded DMO path.

        The manifest records the DMO snapshot (``L205n2500TNG_DM``); truth uses
        the hydro twin (``L205n2500TNG``). This swaps the ``_DM`` suffix on the
        simulation directory only.
        """
        return self.dmo_snapshot.replace("L205n2500TNG_DM", "L205n2500TNG")

    def hydro_group_catalog(self) -> str:
        return self.dmo_group_catalog.replace("L205n2500TNG_DM", "L205n2500TNG")


def load_stage1_manifest(path: str | Path) -> Stage1Manifest:
    """Load a condition directory's ``stage1_manifest.json`` into a Stage1Manifest.

    ``path`` may be the manifest file itself or the directory containing it.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "stage1_manifest.json"
    m = json.loads(p.read_text())
    return Stage1Manifest(
        proj_dir=int(m["proj_dir"]),
        disp=np.asarray(m["disp"], dtype=np.float64),
        flip=np.asarray(m["flip"], dtype=bool),
        box_size=float(m["box_size"]),
        npix=int(m["npix"]),
        n_slabs=int(m["n_slabs"]),
        slab_depth=float(m["slab_depth"]),
        snapshot_index=int(m.get("snapshot_index", m.get("transforms_snap_idx", -1))),
        redshift=float(m["redshift"]),
        scale_factor=float(m["scale_factor"]),
        halo_mass_field=str(m.get("halo_mass_field", "Group_M_Crit200")),
        dmo_snapshot=str(m.get("snapshot", "")),
        dmo_group_catalog=str(m.get("group_catalog", "")),
        raw=m,
    )
