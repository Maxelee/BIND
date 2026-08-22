"""Lightcone geometric transforms: random axis permutation, displacement, and axis flip.

Matches the convention of lux (``/mnt/home/mlee1/lux``) so that BIND-produced
lensplanes are directly compatible with lux raytracing.

Transform applied to each particle for snapshot s:
  1. Permute axes via ``proj_dirs[s]``: choose which original axis is the LOS.
  2. Displace: ``new_coord[ax] = pos[original_ax] + disp[s, original_ax]``
  3. Flip:  ``if flip[s, original_ax]: new_coord[ax] = -new_coord[ax]``
  4. Periodic wrap to ``[0, box_size)``.

Output positions: axis 0 = transverse x, axis 1 = transverse y, axis 2 = LOS.

Storage in ``disp`` / ``flip`` arrays matches lux's ``disp[j + 3*s]`` / ``flip[j + 3*s]``
where ``j`` indexes the *original* (pre-permutation) axis.  The permutation tells
us which original axis maps to the transverse x, transverse y, and LOS slots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# (ix, iy, iz): original-axis indices that map to (transverse_x, transverse_y, LOS)
PROJ_DIR_AXES: dict[int, tuple[int, int, int]] = {
    0: (1, 2, 0),  # LOS along original x-axis → transverse = (y, z)
    1: (2, 0, 1),  # LOS along original y-axis → transverse = (z, x)
    2: (0, 1, 2),  # LOS along original z-axis → transverse = (x, y)
}


class LightconeTransforms:
    """Per-snapshot geometric transforms for a multi-snapshot lightcone.

    Attributes
    ----------
    proj_dirs : (Ns,) int array
        Projection direction per snapshot (0, 1, or 2).
    disp : (Ns, 3) float array
        Displacement in Mpc/h, indexed by *original* axis.
    flip : (Ns, 3) bool array
        Axis-flip flags, indexed by *original* axis.
    """

    def __init__(
        self,
        proj_dirs: np.ndarray,
        disp: np.ndarray,
        flip: np.ndarray,
    ) -> None:
        self.proj_dirs = np.asarray(proj_dirs, dtype=np.int32)
        self.disp = np.asarray(disp, dtype=np.float64)    # (Ns, 3)
        self.flip = np.asarray(flip, dtype=bool)           # (Ns, 3)
        n = len(self.proj_dirs)
        if self.disp.shape != (n, 3) or self.flip.shape != (n, 3):
            raise ValueError("disp and flip must have shape (n_snapshots, 3)")

    @property
    def n_snapshots(self) -> int:
        return len(self.proj_dirs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_seed(
        cls,
        n_snapshots: int,
        seed: int,
        box_size: float,
        random_proj_dir: bool = True,
    ) -> "LightconeTransforms":
        """Generate transforms from a random seed.

        The draw order matches lux's GSL RNG sequence:
          1. proj_dirs (one draw per snapshot, if random)
          2. For each snapshot i, for each dimension j=0,1,2:
               disp[i, j] ← uniform [0, box_size)
               flip[i, j] ← Bernoulli(0.5)

        Uses numpy's default (MT19937) RNG; to match lux's GSL MT19937
        exactly you would need to verify the seeding convention, but for
        a BIND-only pipeline any consistent seed is fine.

        Parameters
        ----------
        n_snapshots : int
        seed : int
        box_size : float
            Simulation box side length in Mpc/h (used as upper bound for disp).
        random_proj_dir : bool
            If False all snapshots use proj_dir=2 (LOS along z).
        """
        rng = np.random.default_rng(seed)

        if random_proj_dir:
            proj_dirs = rng.integers(0, 3, size=n_snapshots)
        else:
            proj_dirs = np.full(n_snapshots, 2, dtype=np.int32)

        # Draw disp then flip for each (snapshot, axis) pair, interleaved
        # to match lux's inner-j-outer-i loop.
        disp = np.zeros((n_snapshots, 3), dtype=np.float64)
        flip = np.zeros((n_snapshots, 3), dtype=bool)
        for i in range(n_snapshots):
            for j in range(3):
                disp[i, j] = rng.uniform(0.0, box_size)
                flip[i, j] = bool(rng.integers(0, 2) == 0)

        return cls(proj_dirs, disp, flip)

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    def apply(
        self,
        pos_mpch: np.ndarray,
        snap_idx: int,
        box_size: float,
    ) -> np.ndarray:
        """Apply the transform for snapshot *snap_idx* to particle positions.

        Parameters
        ----------
        pos_mpch : (N, 3) float array
            Particle positions in Mpc/h, columns = (x, y, z) in original frame.
        snap_idx : int
        box_size : float
            Periodic box size in Mpc/h.

        Returns
        -------
        (N, 3) float32 array
            Transformed positions: col 0/1 = transverse x/y, col 2 = LOS.
            All coordinates wrapped to ``[0, box_size)``.
        """
        pd = int(self.proj_dirs[snap_idx])
        ix, iy, iz = PROJ_DIR_AXES[pd]
        d = self.disp[snap_idx]   # shape (3,), indexed by original axis
        f = self.flip[snap_idx]   # shape (3,), indexed by original axis

        # Displacement uses the original-axis index (matching lux's disp[ix+3*s])
        x = pos_mpch[:, ix].astype(np.float64) + d[ix]
        y = pos_mpch[:, iy].astype(np.float64) + d[iy]
        z = pos_mpch[:, iz].astype(np.float64) + d[iz]

        if f[ix]:
            x = -x
        if f[iy]:
            y = -y
        if f[iz]:
            z = -z

        x = x % box_size
        y = y % box_size
        z = z % box_size

        return np.stack([x, y, z], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "proj_dirs": self.proj_dirs.tolist(),
            "disp": self.disp.tolist(),
            "flip": self.flip.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LightconeTransforms":
        return cls(d["proj_dirs"], d["disp"], d["flip"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LightconeTransforms":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return (f"LightconeTransforms(n_snapshots={self.n_snapshots}, "
                f"proj_dirs={self.proj_dirs.tolist()})")
