"""Generic ``bind.paint`` API: DMO snapshot + halo catalog -> baryonified maps.

Three layers:

* :class:`Simulation` — DMO particles + halo catalog (loaded from a snapshot
  or supplied as numpy arrays).
* :class:`Model` — loaded flow-matching checkpoint + normalization stats with
  a low-level :meth:`Model.generate` that runs the sampler on a list of
  per-halo cutouts.
* :func:`paint` — one-call orchestrator: project the box into z-slabs, extract
  halo cutouts, run the model, and paste the generated hydro fields back into
  per-slab composite maps.

Native scale (the model was trained at):

* pixel scale : ``50.0 / 1024 ~= 0.0488`` Mpc/h per pixel  (~50 kpc/h)
* slab depth  : 50 Mpc/h

Off-spec values are accepted but emit a warning — the model will still run but
quality may degrade as you move away from the trained scale.

Example::

    import bind

    sim = bind.Simulation.from_paths(
        snapshot="path/to/snap_090.hdf5",
        group_catalog="path/to/fof_subhalo_tab_090.hdf5",
        halo_mass_min=1e13,
    )
    model = bind.Model.from_local("weights/fm_two_head")
    result = bind.paint(sim, model, params=my_35_dim_params,
                        output_dir="bind_output/my_run")
"""

from __future__ import annotations

import json
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from bind.data import NormStats

from . import io_gadget
from .pipeline import (
    _denormalize_to_physical,
    build_bind_composite,
    extract_multiscale,
    normalize_cutout,
    pixelize_z_projection,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NATIVE_PIXEL_SIZE_MPCH = 50.0 / 1024.0       # ~0.0488 Mpc/h (~50 kpc/h)
NATIVE_SLAB_DEPTH_MPCH = 50.0                # Mpc/h
PATCH_PIX = 128                              # halo cutout resolution (model fixed)


# ---------------------------------------------------------------------------
# Simulation: pure data
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    """A DMO simulation snapshot + halo catalog, in physical units (Mpc/h, Msun/h).

    Construct via :meth:`from_paths` (Gadget/Arepo HDF5) or :meth:`from_arrays`
    (in-memory). Halos are already mass-cut; further cuts can be applied with
    :meth:`filter_halos`.
    """

    dmo_positions: np.ndarray         # (N_part, 3)  Mpc/h
    particle_mass: float              # Msun/h, single value (uniform DMO grid)
    box_size: float                   # Mpc/h
    halo_positions: np.ndarray        # (N_halo, 3)  Mpc/h
    halo_masses: np.ndarray           # (N_halo,)    Msun/h
    halo_r200: np.ndarray             # (N_halo,)    Mpc/h (zeros if unknown)

    # ---- factory constructors -------------------------------------------
    @classmethod
    def from_paths(
        cls,
        snapshot: Path | str,
        group_catalog: Path | str,
        *,
        snapshot_index: int | None = None,
        halo_mass_min: float = 1e13,
        halo_mass_field: str = "Group_M_Crit200",
    ) -> "Simulation":
        """Load DMO particles + FOF halos from Gadget/Arepo HDF5."""
        dmo_pos, particle_mass, box = io_gadget.read_dmo_particles(
            snapshot, snapshot=snapshot_index
        )
        cat = io_gadget.read_fof_catalog(
            group_catalog, snapshot=snapshot_index,
            halo_mass_min=halo_mass_min, mass_field=halo_mass_field,
        )
        return cls(
            dmo_positions=dmo_pos,
            particle_mass=float(particle_mass),
            box_size=float(box),
            halo_positions=cat["positions"],
            halo_masses=cat["mass"],
            halo_r200=cat["r200"],
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        dmo_positions: np.ndarray,
        particle_mass: float,
        box_size: float,
        halo_positions: np.ndarray,
        halo_masses: np.ndarray,
        halo_r200: np.ndarray | None = None,
        halo_mass_min: float = 0.0,
    ) -> "Simulation":
        """Build a Simulation from in-memory arrays (units: Mpc/h, Msun/h)."""
        halo_positions = np.asarray(halo_positions, dtype=np.float32)
        halo_masses = np.asarray(halo_masses, dtype=np.float32)
        if halo_r200 is None:
            halo_r200 = np.zeros(len(halo_masses), dtype=np.float32)
        else:
            halo_r200 = np.asarray(halo_r200, dtype=np.float32)
        keep = halo_masses > float(halo_mass_min)
        return cls(
            dmo_positions=np.asarray(dmo_positions, dtype=np.float32),
            particle_mass=float(particle_mass),
            box_size=float(box_size),
            halo_positions=halo_positions[keep],
            halo_masses=halo_masses[keep],
            halo_r200=halo_r200[keep],
        )

    # ---- views ----------------------------------------------------------
    @property
    def n_halos(self) -> int:
        return int(len(self.halo_masses))

    @property
    def n_particles(self) -> int:
        return int(len(self.dmo_positions))

    def __repr__(self) -> str:
        return (
            f"Simulation(box={self.box_size:.2f} Mpc/h, "
            f"n_part={self.n_particles}, n_halos={self.n_halos}, "
            f"M_halo_min={self.halo_masses.min():.2e} Msun/h)"
            if self.n_halos
            else f"Simulation(box={self.box_size:.2f} Mpc/h, n_part={self.n_particles}, n_halos=0)"
        )

    # ---- transforms -----------------------------------------------------
    def filter_halos(
        self,
        *,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mask: np.ndarray | None = None,
    ) -> "Simulation":
        """Return a new Simulation with halo cuts applied."""
        keep = np.ones(self.n_halos, dtype=bool)
        if mass_min is not None:
            keep &= self.halo_masses > mass_min
        if mass_max is not None:
            keep &= self.halo_masses < mass_max
        if mask is not None:
            keep &= np.asarray(mask, dtype=bool)
        return Simulation(
            dmo_positions=self.dmo_positions,
            particle_mass=self.particle_mass,
            box_size=self.box_size,
            halo_positions=self.halo_positions[keep],
            halo_masses=self.halo_masses[keep],
            halo_r200=self.halo_r200[keep],
        )

    def project(
        self,
        *,
        pixel_size: float = NATIVE_PIXEL_SIZE_MPCH,
        slab_depth: float = NATIVE_SLAB_DEPTH_MPCH,
    ) -> np.ndarray:
        """CIC-project DMO particles into ``n_slabs`` 2D maps along z.

        Returns ``(n_slabs, npix, npix)`` of column-density-like values.
        """
        npix = _round_npix(self.box_size, pixel_size)
        n_slabs = _round_n_slabs(self.box_size, slab_depth)
        masses = np.full(self.n_particles, self.particle_mass, dtype=np.float32)
        return _project_zslabs(self.dmo_positions, masses, self.box_size, npix, n_slabs)

    def slab_assignment(self, *, slab_depth: float = NATIVE_SLAB_DEPTH_MPCH) -> np.ndarray:
        """Per-halo slab index (length = n_halos)."""
        n_slabs = _round_n_slabs(self.box_size, slab_depth)
        return _assign_halos_to_slabs(self.halo_positions[:, 2], self.box_size, n_slabs)


# ---------------------------------------------------------------------------
# Model: trained checkpoint + norm_stats
# ---------------------------------------------------------------------------

def _resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


class Model:
    """Wrapper around a trained ``FlowMatchingLit`` checkpoint + ``NormStats``.

    Construct via :meth:`from_local`, :meth:`from_files`, or pass an explicit
    fm/norm_stats pair to the initializer.

    The low-level :meth:`generate` runs the sampler on a list of per-halo
    cutouts (each a ``{"condition": (1,128,128), "large_scale": (3,128,128)}``
    dict) and returns physical-space ``(N, 3, 128, 128)`` hydro maps.
    """

    def __init__(
        self,
        fm,
        norm_stats: NormStats,
        *,
        n_params: int,
        no_large_scale: bool,
        device: torch.device,
    ):
        self.fm = fm
        self.norm_stats = norm_stats
        self.n_params = int(n_params)
        self.no_large_scale = bool(no_large_scale)
        self.device = device
        cosmo_idx = [0, 1, 7, 8]
        self.param_indices: np.ndarray | None = (
            np.array([i for i in range(35) if i not in cosmo_idx])
            if self.n_params < 35 else None
        )

    # ---- factory constructors -----------------------------------------
    @classmethod
    def from_files(
        cls,
        checkpoint: Path | str,
        norm_stats: Path | str,
        *,
        device: str | torch.device = "auto",
    ) -> "Model":
        from bind.train import FlowMatchingLit  # lazy

        dev = _resolve_device(device)
        lit = FlowMatchingLit.load_from_checkpoint(str(checkpoint), map_location=dev)
        lit.eval().to(dev)
        n_params = int(getattr(lit.hparams, "n_params", 35))
        no_large_scale = bool(getattr(lit.hparams, "no_large_scale", False))
        ns = NormStats.load(str(norm_stats))
        return cls(lit.fm, ns, n_params=n_params, no_large_scale=no_large_scale, device=dev)

    @classmethod
    def from_local(
        cls,
        run_dir: Path | str,
        *,
        checkpoint_name: str = "last.ckpt",
        norm_stats_name: str = "norm_stats.npz",
        device: str | torch.device = "auto",
    ) -> "Model":
        """Load from a directory layout ``<run_dir>/{last.ckpt,norm_stats.npz}``."""
        d = Path(run_dir)
        return cls.from_files(d / checkpoint_name, d / norm_stats_name, device=device)

    # ---- inference -----------------------------------------------------
    def __repr__(self) -> str:
        return (f"Model(n_params={self.n_params}, "
                f"no_large_scale={self.no_large_scale}, device={self.device})")

    @torch.no_grad()
    def generate(
        self,
        cutouts: list[dict],
        params: np.ndarray,
        *,
        n_steps: int = 50,
        batch_size: int = 16,
        use_amp: bool = True,
        progress: bool = True,
    ) -> np.ndarray:
        """Run the flow-matching sampler on per-halo cutouts.

        Parameters
        ----------
        cutouts
            List of dicts with keys ``condition`` (shape ``(1, H, W)``) and
            ``large_scale`` (shape ``(3, H, W)``). Build with
            :func:`bind.extract_halo_cutouts` or directly via
            :func:`bind.inference.pipeline.extract_multiscale`.
        params
            35-dim cosmology + astrophysics vector.
        n_steps, batch_size, use_amp, progress
            Sampling controls.

        Returns
        -------
        np.ndarray
            ``(N, 3, H, W)`` hydro fields ``[DM_hydro, Gas, Stars]`` in physical
            (un-normalized) units.
        """
        params = _validate_params(params)
        if not cutouts:
            return np.zeros((0, 3, PATCH_PIX, PATCH_PIX), dtype=np.float32)

        outputs: list[np.ndarray] = []
        rng = range(0, len(cutouts), batch_size)
        if progress:
            rng = tqdm(rng, desc="Generating hydro")
        for s in rng:
            batch = cutouts[s : s + batch_size]
            conds, lss, par = zip(
                *[normalize_cutout(hc, self.norm_stats, params) for hc in batch]
            )
            cond_t = torch.from_numpy(np.stack(conds).astype(np.float32)).to(self.device)
            ls_t = (
                None if self.no_large_scale
                else torch.from_numpy(np.stack(lss).astype(np.float32)).to(self.device)
            )
            par_np = np.stack(par).astype(np.float32)
            if self.param_indices is not None:
                par_np = par_np[:, self.param_indices]
            par_t = torch.from_numpy(par_np).to(self.device)

            ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and self.device.type == "cuda" else nullcontext()
            )
            with ctx:
                gen = self.fm.sample(cond_t, ls_t, par_t, n_steps=n_steps)
            outputs.append(_denormalize_to_physical(
                gen.float().cpu().numpy().astype(np.float32), self.norm_stats
            ))
        return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Geometry helpers (kept module-level; reused by Simulation + paint)
# ---------------------------------------------------------------------------

def _round_npix(box_size: float, pixel_size: float) -> int:
    n = int(round(box_size / pixel_size))
    if n <= 0:
        raise ValueError(f"derived npix={n} from box_size={box_size}, pixel_size={pixel_size}")
    return n


def _round_n_slabs(box_size: float, slab_depth: float) -> int:
    return max(1, int(round(box_size / slab_depth)))


def _project_zslabs(positions, masses, box_size, npix, n_slabs):
    """Project particles into ``n_slabs`` separate 2D maps along z."""
    out = np.zeros((n_slabs, npix, npix), dtype=np.float32)
    slab_h = box_size / n_slabs
    z = positions[:, 2]
    for i in range(n_slabs):
        lo = i * slab_h
        hi = (i + 1) * slab_h if i < n_slabs - 1 else box_size + 1e-6
        m = (z >= lo) & (z < hi)
        if m.any():
            out[i] = pixelize_z_projection(positions[m], masses[m], box_size, npix)
    return out


def _assign_halos_to_slabs(halo_z: np.ndarray, box_size: float, n_slabs: int) -> np.ndarray:
    slab_h = box_size / n_slabs
    return np.clip((halo_z / slab_h).astype(np.int64), 0, n_slabs - 1)


def _validate_params(params) -> np.ndarray:
    arr = np.asarray(params, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 35:
        raise ValueError(f"`params` must be a 35-dim vector; got shape {np.shape(params)}")
    return arr


def _warn_off_native(pixel_size: float, slab_depth: float) -> None:
    if not np.isclose(pixel_size, NATIVE_PIXEL_SIZE_MPCH, rtol=5e-2):
        warnings.warn(
            f"pixel_size={pixel_size:.4f} Mpc/h differs from native "
            f"{NATIVE_PIXEL_SIZE_MPCH:.4f}; model quality may degrade.",
            stacklevel=3,
        )
    if not np.isclose(slab_depth, NATIVE_SLAB_DEPTH_MPCH, rtol=5e-2):
        warnings.warn(
            f"slab_depth={slab_depth} Mpc/h differs from native "
            f"{NATIVE_SLAB_DEPTH_MPCH}; model quality may degrade.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Halo cutout extraction (module-level, exposed for power users)
# ---------------------------------------------------------------------------

def extract_halo_cutouts(
    slab_map: np.ndarray,
    halo_positions_xy: np.ndarray,
    *,
    box_size: float,
    patch_pix: int = PATCH_PIX,
) -> list[dict]:
    """Extract ``(condition, large_scale)`` cutouts at each halo's xy position.

    Parameters
    ----------
    slab_map
        2D DMO column-density map of shape ``(npix, npix)``.
    halo_positions_xy
        ``(N, 2)`` halo (x, y) positions in Mpc/h.
    box_size
        Side length of the slab in Mpc/h.
    patch_pix
        Cutout resolution; the model expects 128.
    """
    npix = slab_map.shape[0]
    pixels_per_mpc = npix / box_size
    out: list[dict] = []
    for hx, hy in halo_positions_xy:
        cx = int(hx * pixels_per_mpc) % npix
        cy = int(hy * pixels_per_mpc) % npix
        cond, ls = extract_multiscale(slab_map, cx, cy, target_res=patch_pix)
        out.append({"condition": cond, "large_scale": ls})
    return out


# ---------------------------------------------------------------------------
# Result + main paint() entry point
# ---------------------------------------------------------------------------

@dataclass
class PaintResult:
    """Outputs of a :func:`paint` run."""

    n_halos: int
    n_slabs: int
    box_size: float
    npix: int
    output_dir: Path
    summary_path: Path
    composite_paths: list[Path] = field(default_factory=list)
    per_slab: list[dict] = field(default_factory=list)


def paint(
    sim: Simulation,
    model: Model,
    *,
    params: np.ndarray,
    output_dir: Path | str = "bind_output",
    pixel_size: float = NATIVE_PIXEL_SIZE_MPCH,
    slab_depth: float = NATIVE_SLAB_DEPTH_MPCH,
    patch_pix: int = PATCH_PIX,
    n_steps: int = 50,
    batch_size: int = 16,
    use_amp: bool = True,
    patch_mass_match: bool = True,
    taper_frac: float = 0.15,
    r200_factor: float = 0.0,
    save_per_halo_patches: bool = True,
    progress: bool = True,
) -> PaintResult:
    """Generate baryonified hydro maps from a DMO simulation + halo catalog.

    Parameters
    ----------
    sim, model
        A :class:`Simulation` and a loaded :class:`Model`.
    params
        35-dim cosmology + astrophysics vector that the model conditions on.
    output_dir
        Destination directory. One ``composite_slab{NN}.npz`` is written per
        z-slab plus a top-level ``summary.json``.
    pixel_size, slab_depth
        Geometry. Defaults match the native trained scale (~50 kpc/h pixel,
        50 Mpc/h slab depth). Off-spec values trigger a warning.
    patch_pix, n_steps, batch_size, use_amp
        Sampling controls. ``patch_pix`` should stay at 128 (model contract).
    patch_mass_match, taper_frac, r200_factor
        Compositing controls (see :func:`bind.inference.pipeline.build_bind_composite`).
    """
    params = _validate_params(params)
    _warn_off_native(pixel_size, slab_depth)
    if model.no_large_scale:
        warnings.warn(
            "Model was trained without large-scale conditioning (cube model). "
            "The 2D paint() workflow extracts halo cutouts from per-slab full-box "
            "projections — this differs from the cube training geometry. Consider "
            "a 'fm_two_head' style checkpoint instead.",
            stacklevel=2,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npix = _round_npix(sim.box_size, pixel_size)
    n_slabs = _round_n_slabs(sim.box_size, slab_depth)

    print(f"[paint] box={sim.box_size:.2f} Mpc/h, npix={npix}, n_slabs={n_slabs}")
    print(f"[paint] {sim.n_halos} halos, {sim.n_particles} particles")
    print(f"[paint] projecting DMO into {n_slabs} z-slab(s)...")

    # 1. Project DMO into z-slabs.
    dmo_slabs = sim.project(pixel_size=pixel_size, slab_depth=slab_depth)
    halo_slab_idx = sim.slab_assignment(slab_depth=slab_depth)

    composite_paths: list[Path] = []
    per_slab: list[dict] = []

    for si in range(n_slabs):
        slab_map = dmo_slabs[si]
        in_slab = np.where(halo_slab_idx == si)[0]
        if len(in_slab) == 0:
            slab_path = _save_empty_slab(output_dir, si, n_slabs, sim.box_size, slab_map)
            composite_paths.append(slab_path)
            per_slab.append({"slab_idx": si, "n_halos": 0})
            continue

        halo_xy = sim.halo_positions[in_slab, :2]
        halo_m = sim.halo_masses[in_slab]
        halo_r = sim.halo_r200[in_slab]

        # 2. Extract per-halo cutouts.
        cutouts = extract_halo_cutouts(
            slab_map, halo_xy, box_size=sim.box_size, patch_pix=patch_pix
        )

        # 3. Run the model.
        gen = model.generate(
            cutouts, params,
            n_steps=n_steps, batch_size=batch_size, use_amp=use_amp,
            progress=progress,
        )

        # 4. Composite.  build_bind_composite expects per-halo dicts.
        halos_dicts = [
            {"halo_center": halo_xy[i], "halo_mass": float(halo_m[i]),
             "r200": float(halo_r[i]), "params": params.astype(np.float32)}
            for i in range(len(in_slab))
        ]
        bundle = build_bind_composite(
            slab_map, halos_dicts, gen, cutouts,
            box_size=sim.box_size, npix=npix, patch_pix=patch_pix,
            patch_mass_match=patch_mass_match, taper_frac=taper_frac,
            r200_factor=r200_factor,
        )

        slab_path = output_dir / f"composite_slab{si:02d}.npz"
        save_kwargs: dict[str, Any] = dict(
            dmo=slab_map,
            composite=bundle["composite"],
            alpha=bundle["alpha"],
            patch_scales=bundle["patch_scales"],
            scale_global=bundle["scale_global"],
            coverage_pct=bundle["coverage_pct"],
            n_halos=len(in_slab),
            slab_idx=si,
            n_slabs=n_slabs,
            box_size=sim.box_size,
            halo_centers=halo_xy.astype(np.float32),
            halo_masses=halo_m.astype(np.float32),
            halo_r200=halo_r.astype(np.float32),
        )
        if save_per_halo_patches:
            save_kwargs["generated_patches"] = gen
        np.savez_compressed(slab_path, **save_kwargs)

        composite_paths.append(slab_path)
        per_slab.append({
            "slab_idx": si,
            "n_halos": int(len(in_slab)),
            "coverage_pct": float(bundle["coverage_pct"]),
            "scale_global": float(bundle["scale_global"]),
        })
        print(f"[paint] slab {si}: {len(in_slab)} halos, "
              f"coverage={bundle['coverage_pct']:.1f}%")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "box_size": sim.box_size,
        "pixel_size": pixel_size,
        "slab_depth": slab_depth,
        "npix": npix,
        "n_slabs": n_slabs,
        "n_halos": sim.n_halos,
        "model": repr(model),
        "n_steps": n_steps,
        "patch_mass_match": patch_mass_match,
        "taper_frac": taper_frac,
        "r200_factor": r200_factor,
        "per_slab": per_slab,
    }, indent=2))

    return PaintResult(
        n_halos=sim.n_halos,
        n_slabs=n_slabs,
        box_size=sim.box_size,
        npix=npix,
        output_dir=output_dir,
        summary_path=summary_path,
        composite_paths=composite_paths,
        per_slab=per_slab,
    )


def _save_empty_slab(output_dir, si, n_slabs, box_size, slab_map):
    composite = np.stack([slab_map, np.zeros_like(slab_map), np.zeros_like(slab_map)])
    p = output_dir / f"composite_slab{si:02d}.npz"
    np.savez_compressed(
        p, dmo=slab_map, composite=composite, n_halos=0,
        slab_idx=si, n_slabs=n_slabs, box_size=box_size,
    )
    return p
