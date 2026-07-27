"""Halo (box coords) → lux lightcone map-pixel engine (P4b plan, phase P1).

Implements the four-step chain in ``docs/ksz_lightcone_map_plan.md`` §1.2 for
turning a halo's 3-D box position into the pixel(s) it occupies in the traced
``{tau,y,kappa}_maps.npz`` products, **without touching lux or re-deriving any
map**.  Everything here reproduces what the *actual* pipeline
(``bind.inference.{lightcone_transforms,lensplane}`` + the ``lux`` C++
raytracer) did, including one deliberate quirk (see "The Lt=205 convention"
below) that must be replicated bit-for-bit rather than "fixed".

The chain, box → sky pixel
---------------------------
**Step A — snapshot box transform.**  ``LightconeTransforms.apply(pos, snap_idx,
box_size=205.0)`` permutes axes (LOS choice), adds a per-snapshot displacement,
flips sign on chosen axes, and wraps mod 205.0 Mpc/h.  Output columns are
``(transverse_x, transverse_y, LOS)``.  One transform per snapshot, shared by
all 256 Sobol runs and all 50 realizations (seed 2020, baked into
``lightcone_transforms.json`` / ``geometry.npz``'s ``lc_*`` fields).

**Step B — slab assignment.**  ``slab = clip(floor(LOS / 51.25), 0, 3)``
(51.25 = 205/4 Mpc/h, the transverse-box's quarter-depth).  The global
1-indexed lux plane number is ``p = 4*snap_idx + slab + 1`` (1..80).

**Step C — plane pixelization (this module's key finding).**  Stage 1 builds
each snapshot's mass map on a **native** grid: ``NATIVE_NPIX = 4198`` px at
``NATIVE_PIXEL_SIZE_MPCH = 0.048828125`` Mpc/h (``= 200/4096 = 50/1024``,
covering 205.02 Mpc/h ⊃ the nominal 205.0 Mpc/h box).  ``paint_lensplane.py``
then writes the ``lenspot``/``tauplane``/``yplane`` files by **literally
slicing** the central ``LP_GRID = 4096`` pixels out of that native array
(``mass_map[off:off+4096, off:off+4096]`` with ``off = (4198-4096)//2 = 51``)
— a plain numpy index crop, *not* a resample/rescale.  The native pixel
spacing is therefore preserved exactly inside the written plane array: a halo
at native pixel index ``i_native = floor(x / 0.048828125)`` lands at plane
array index ``i_plane = i_native - 51`` (real-valued here; not floored so
downstream sub-pixel comparisons stay meaningful).  Halos with
``i_native < 51`` or ``>= 4147`` (on either axis) fall off the crop and are
dropped — ``1 - (4096/4198)**2 ≈ 4.8%`` of the transverse area (not the
"~2.4%" quoted in an earlier draft of the plan, which used the 1-D removed
fraction instead of the 2-D area fraction).

**The Lt=205 convention.**  ``paint_lensplane.py`` computes an *effective* box
size for the cropped array (``box_size_eff = 205.0 * 4096/4198 ≈ 200.02``
Mpc/h) but only uses it for the density/Poisson normalisation
(``mass_map_to_delta_scaled`` / ``density_to_lensplane``) — it is **not** what
gets written to ``config.dat``.  ``Lt`` in ``config.dat`` (and in
``geometry.npz``) is the raw **native** ``box_size = 205.0`` Mpc/h read
straight from the stage-1 manifest.  So when lux (and this module, to match
it) converts a plane array index to a sky angle in Step D, it uses
``dLt = Lt/4096 = 205/4096 ≈ 0.050049`` Mpc/h/px — *not* the true array pixel
spacing of 0.048828125 Mpc/h/px.  This is an internal inconsistency in the
pipeline (P0's "Lt=205 not 200" finding), but because both the *actual* traced
maps and every prediction made by this module apply the **same** mismatched
``dLt`` consistently, positions still round-trip correctly — Step C fixes
*where in the array* a halo's flux sits (true native pixel size), Step D
fixes *what sky angle lux thinks that array index corresponds to* (the
Lt=205-derived, slightly-too-coarse pixel size).  Validated empirically in
P1's checks 2-3 (see ``examples/_p1_validate_geometry.py``).

**Step D — per-realization scatter + ray geometry.**  For realization
``r = 1..50``, snapshot ``snap_idx``: ``rot[r-1, snap_idx] ∈ {0,1,2,3}`` and
``disp[r-1, snap_idx] ∈ [0,4096)²`` (read verbatim from
``rt_output/run{r:03d}/config.dat``, identical across all 256 Sobol nodes
because they share ``RT_SEED=1992``).  Applied to the *raw* Step-C plane
position ``(i, j)``, **literally** (note ``4096-j``, not ``4096-1-j``):

    rot 0: (I, J) = (i,        j)
    rot 1: (I, J) = (4096-j,   i)
    rot 2: (I, J) = (4096-i,   4096-j)
    rot 3: (I, J) = (j,        4096-i)

then ``(I, J) = (I + disp_x, J + disp_y) mod 4096``.  Ray geometry (Born,
small-angle): plane position ``X = (I+0.5)*dLt`` (``dLt = Lt[snap_idx]/4096``),
``beta = (X - 0.5*Lt) / chi[p]`` where ``chi`` is the **plane-center** comoving
distance (``geometry.npz``'s ``chi`` array, 81 entries indexed ``chi[p]`` for
``p=1..80`` — *not* ``chi_out``, which holds the far-edge/cumulative distances
used for source-plane bookkeeping).  Map pixel
``i_map = beta_x/dtheta + 512`` (``dtheta = (5°·pi/180)/1024``), same for
``j_map``.  Because the transverse box (``Lt≈205`` Mpc/h) can be narrower than
the FOV's physical width at the plane's distance (``W = chi[p]*5°_rad``,
increasingly true at high z), the plane tiles periodically across the sky:
every image ``X + n*Lt`` (independently for x and y) that lands inside
``[0, 1024)`` is a valid copy — **all** are kept (a single halo can appear
several times in one map at high z).

**Map array axis order.**  ``{tau,y,kappa}_maps.npz['tau'/'y'/'kappa']`` has
shape ``(50, 5, 1024, 1024)``.  P1 check 3 empirically pins whether the last
two axes are ``[i_map, j_map]`` or ``[j_map, i_map]`` — see
``MAP_AXIS_TRANSPOSE`` below (frozen after the validation ladder passed; do
not change without re-running check 3).

Usage
-----
>>> geom = load_geometry(".../geometry.npz")
>>> i_map, j_map = halo_map_pixels(pos_mpch, snap_idx=0, realization=1, geom=geom)[1:]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bind.inference.lightcone_transforms import LightconeTransforms

# ---------------------------------------------------------------------------
# Frozen geometric constants (P0/P1-verified; do not "fix" — they reproduce
# what the actual pipeline did, mismatches and all).
# ---------------------------------------------------------------------------

BOX_SIZE_MPCH = 205.0                    # native TNG300 box (Mpc/h)
NATIVE_NPIX = 4198                       # stage-1 native grid (px/side)
NATIVE_PIXEL_SIZE_MPCH = 0.048828125     # = 200/4096 = 50/1024 (Mpc/h/px)
LP_GRID = 4096                           # lux plane grid (px/side)
CROP_OFF = (NATIVE_NPIX - LP_GRID) // 2  # = 51 px, center-crop offset
RT_GRID = 1024                           # ray-traced map grid (px/side)
FOV_DEG = 5.0                            # ray-traced map field of view
PLANES_PER_SNAP = 4                      # pps; slabs per snapshot
N_REALIZATIONS = 50
N_SNAPSHOTS = 20

# Empirically pinned in P1 check 3: does array[..., a, b] correspond to
# (a=i_map, b=j_map) [False] or (a=j_map, b=i_map) [True]?  See docstring.
MAP_AXIS_TRANSPOSE = False

FOV_RAD = np.deg2rad(FOV_DEG)
DTHETA_RAD = FOV_RAD / RT_GRID  # per-pixel angle on the ray-traced map


# ---------------------------------------------------------------------------
# geometry.npz loader
# ---------------------------------------------------------------------------

@dataclass
class LuxGeometry:
    """Wraps ``geometry.npz`` (built by P0 / ``examples/_p0_geometry_tables.py``).

    Attributes mirror the npz keys exactly (see ``docs/ksz_lightcone_map_plan.md``
    P0 phase).  ``chi`` has 81 entries (plane *centers*, index 0..80; ``chi[p]``
    for ``p=1..80`` is the center distance of plane ``p``).  ``chi_out`` has 80
    entries (plane *far edges* / cumulative distances, used for source-plane
    bookkeeping only — not for the ray-geometry ``beta`` computation).
    """

    rot: np.ndarray        # (50, 20) int, realization x snap_idx
    disp: np.ndarray       # (50, 20, 2) int, realization x snap_idx x (dx,dy)
    a: np.ndarray          # (81,) scale factor at plane edges
    chi: np.ndarray        # (81,) comoving distance to plane *centers* (see above)
    chi_out: np.ndarray    # (80,) comoving distance to plane *far edges*
    Ll: np.ndarray         # (20,) LOS box size per snapshot [Mpc/h]
    Lt: np.ndarray         # (20,) transverse box size per snapshot [Mpc/h]
    snap: np.ndarray       # (20,) snapshot number, low-z first
    snap_idx: np.ndarray   # (20,) 0..19
    z: np.ndarray          # (20,) redshift
    a_snap: np.ndarray     # (20,) snapshot scale factor
    lc_proj_dirs: np.ndarray  # (20,) int, LightconeTransforms.proj_dirs
    lc_disp: np.ndarray       # (20, 3) float, LightconeTransforms.disp
    lc_flip: np.ndarray       # (20, 3) bool/int, LightconeTransforms.flip

    _transforms: LightconeTransforms | None = None

    @property
    def transforms(self) -> LightconeTransforms:
        """Lazily-built :class:`LightconeTransforms` from the ``lc_*`` fields."""
        if self._transforms is None:
            self._transforms = LightconeTransforms(
                proj_dirs=self.lc_proj_dirs,
                disp=self.lc_disp,
                flip=self.lc_flip,
            )
        return self._transforms


def load_geometry(path: str | Path) -> LuxGeometry:
    """Load ``geometry.npz`` (P0 output) into a :class:`LuxGeometry`."""
    d = np.load(Path(path), allow_pickle=False)
    return LuxGeometry(
        rot=d["rot"], disp=d["disp"], a=d["a"], chi=d["chi"], chi_out=d["chi_out"],
        Ll=d["Ll"], Lt=d["Lt"], snap=d["snap"], snap_idx=d["snap_idx"], z=d["z"],
        a_snap=d["a_snap"], lc_proj_dirs=d["lc_proj_dirs"], lc_disp=d["lc_disp"],
        lc_flip=d["lc_flip"],
    )


# ---------------------------------------------------------------------------
# Step A-C: box position -> raw plane pixel (pre-randomization)
# ---------------------------------------------------------------------------

def box_to_plane(
    pos_mpch: np.ndarray,
    snap_idx: int,
    geom: LuxGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Box position -> (slab, global plane index, raw plane pixel i, j, valid).

    Implements §1.2 Steps A-C.  ``pos_mpch`` is ``(N, 3)`` in the *original*
    (pre-lightcone-transform) simulation frame, Mpc/h.

    Returns
    -------
    p_slab : (N,) int64
        Local slab index 0..3 within the snapshot.
    p : (N,) int64
        Global 1-indexed lux plane number, ``4*snap_idx + p_slab + 1`` (1..80).
    i, j : (N,) float64
        Continuous plane-pixel coordinates in the **raw, un-randomized**
        4096-frame (i.e. what indexes directly into
        ``lensplanes/{tauplane,yplane,lenspot}{p:02d}.dat`` — no Step D
        rotation/displacement applied yet).  Not floored, so callers can do
        sub-pixel offset checks; round/floor only when indexing an array.
    valid : (N,) bool
        False where the halo's transformed transverse position falls outside
        the ``paint_lensplane`` center-crop (~4.8% of the transverse area) and
        therefore never made it onto any plane at all.
    """
    pos_mpch = np.asarray(pos_mpch, dtype=np.float64).reshape(-1, 3)
    transformed = geom.transforms.apply(pos_mpch, snap_idx, BOX_SIZE_MPCH)
    x = transformed[:, 0].astype(np.float64)
    y = transformed[:, 1].astype(np.float64)
    los = transformed[:, 2].astype(np.float64)

    # Step B: slab assignment.
    slab_depth = BOX_SIZE_MPCH / PLANES_PER_SNAP  # 51.25 Mpc/h
    p_slab = np.clip(np.floor(los / slab_depth).astype(np.int64), 0, PLANES_PER_SNAP - 1)
    p = (PLANES_PER_SNAP * snap_idx + p_slab + 1).astype(np.int64)

    # Step C: native-pixel address minus the center-crop offset (see module
    # docstring — literal index crop, native pixel size preserved).
    i_native = x / NATIVE_PIXEL_SIZE_MPCH
    j_native = y / NATIVE_PIXEL_SIZE_MPCH
    i = i_native - CROP_OFF
    j = j_native - CROP_OFF

    valid = (i >= 0) & (i < LP_GRID) & (j >= 0) & (j < LP_GRID)

    return p_slab, p, i, j, valid


# ---------------------------------------------------------------------------
# Step D: raw plane pixel -> ray-traced map pixel(s)
# ---------------------------------------------------------------------------

def _rotate_scatter(i: np.ndarray, j: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the lux per-snapshot plane rotation (literal ``4096-j`` quirk)."""
    I = np.empty_like(i)
    J = np.empty_like(j)
    m0 = rot == 0
    m1 = rot == 1
    m2 = rot == 2
    m3 = rot == 3
    I[m0], J[m0] = i[m0], j[m0]
    I[m1], J[m1] = LP_GRID - j[m1], i[m1]
    I[m2], J[m2] = LP_GRID - i[m2], LP_GRID - j[m2]
    I[m3], J[m3] = j[m3], LP_GRID - i[m3]
    return I, J


def plane_to_map(
    i: np.ndarray,
    j: np.ndarray,
    plane_p: np.ndarray,
    realization_r: int,
    geom: LuxGeometry,
) -> list[np.ndarray]:
    """Raw plane pixel (i, j) -> all periodic-copy ray-traced map pixels.

    Implements §1.2 Step D.  ``i``, ``j``, ``plane_p`` are ``(N,)`` arrays
    (output of :func:`box_to_plane`); ``realization_r`` is 1-indexed (1..50).

    Returns
    -------
    list of (K_n, 2) float64 arrays, one per input halo, each row an
    ``(i_map, j_map)`` periodic-copy position in ``[0, 1024)``.  ``K_n`` varies
    per halo (0 if none of its periodic images land in the FOV, >1 at high z
    where the box repeats inside the 5° field).  Axis order matches
    ``MAP_AXIS_TRANSPOSE`` (see module docstring): callers indexing
    ``tau_maps.npz['tau'][r-1, k]`` should use ``[i_map, j_map]`` unless
    ``MAP_AXIS_TRANSPOSE`` is True, in which case swap to ``[j_map, i_map]``.
    """
    i = np.asarray(i, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    plane_p = np.asarray(plane_p, dtype=np.int64)
    n = len(i)

    snap_idx_arr = (plane_p - 1) // PLANES_PER_SNAP  # (N,)
    r0 = realization_r - 1

    rot = geom.rot[r0, snap_idx_arr].astype(np.int64)           # (N,)
    disp_x = geom.disp[r0, snap_idx_arr, 0].astype(np.float64)  # (N,)
    disp_y = geom.disp[r0, snap_idx_arr, 1].astype(np.float64)  # (N,)

    I, J = _rotate_scatter(i, j, rot)
    I = np.mod(I + disp_x, LP_GRID)
    J = np.mod(J + disp_y, LP_GRID)

    Lt = geom.Lt[snap_idx_arr]              # (N,) Mpc/h, per-snapshot (== 205.0)
    dLt = Lt / LP_GRID                      # (N,)
    chi_p = geom.chi[plane_p]               # (N,) plane-CENTER comoving distance

    X0 = (I + 0.5) * dLt                    # (N,) raw (n=0) plane position
    Y0 = (J + 0.5) * dLt

    out: list[list[tuple[float, float]]] = [[] for _ in range(n)]

    # Periodic copies: X0 + n*Lt for n such that beta stays inside the FOV.
    # W = chi_p * FOV_RAD is the physical FOV width at this plane's distance;
    # n ranges over roughly [-W/(2Lt)-1, W/(2Lt)+1] to be safe.
    with np.errstate(invalid="ignore"):
        n_max = np.ceil(0.5 * chi_p * FOV_RAD / np.maximum(Lt, 1e-9)).astype(np.int64) + 2
    n_max_global = int(n_max.max()) if n > 0 else 0

    for nx in range(-n_max_global, n_max_global + 1):
        Xc = X0 + nx * Lt
        beta_x = (Xc - 0.5 * Lt) / chi_p
        i_map = beta_x / DTHETA_RAD + 0.5 * RT_GRID
        x_ok = (i_map >= 0) & (i_map < RT_GRID)
        if not x_ok.any():
            continue
        for ny in range(-n_max_global, n_max_global + 1):
            Yc = Y0 + ny * Lt
            beta_y = (Yc - 0.5 * Lt) / chi_p
            j_map = beta_y / DTHETA_RAD + 0.5 * RT_GRID
            y_ok = (j_map >= 0) & (j_map < RT_GRID)
            ok = x_ok & y_ok
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            for k in idx:
                out[k].append((float(i_map[k]), float(j_map[k])))

    return [np.asarray(rows, dtype=np.float64).reshape(-1, 2) for rows in out]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def halo_map_pixels(
    pos_mpch: np.ndarray,
    snap_idx: int,
    realization: int,
    geom: LuxGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full chain: box position -> all ray-traced map pixel copies.

    Composes :func:`box_to_plane` and :func:`plane_to_map`.

    Parameters
    ----------
    pos_mpch : (N, 3) array
        Halo positions in Mpc/h, original (pre-lightcone-transform) frame.
    snap_idx : int
        0..19, low-z first.
    realization : int
        1-indexed, 1..50.
    geom : LuxGeometry

    Returns
    -------
    halo_idx : (M,) int64
        Index into ``pos_mpch`` for each returned copy (M >= N in general;
        halos with multiple periodic images repeat, halos that fell off the
        plane crop or land outside the FOV are absent).
    i_map, j_map : (M,) float64
        Ray-traced map pixel positions in ``[0, 1024)``.  Axis order per
        ``MAP_AXIS_TRANSPOSE`` — see :func:`plane_to_map`.
    """
    p_slab, p, i, j, valid = box_to_plane(pos_mpch, snap_idx, geom)
    n = len(i)
    if not valid.any():
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))

    idx_valid = np.nonzero(valid)[0]
    copies = plane_to_map(i[idx_valid], j[idx_valid], p[idx_valid], realization, geom)

    halo_idx_list = []
    i_map_list = []
    j_map_list = []
    for local_k, orig_idx in enumerate(idx_valid):
        rows = copies[local_k]
        if rows.shape[0] == 0:
            continue
        halo_idx_list.append(np.full(rows.shape[0], orig_idx, dtype=np.int64))
        i_map_list.append(rows[:, 0])
        j_map_list.append(rows[:, 1])

    if not halo_idx_list:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))

    return (np.concatenate(halo_idx_list), np.concatenate(i_map_list),
            np.concatenate(j_map_list))
