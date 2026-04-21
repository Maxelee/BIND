"""Dataclasses shared by the test-suite runner."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SimulationSpec:
    """Inputs needed to run the pipeline for one simulation."""

    suite: str
    sim_id: str
    snapshot: int
    nbody_path: Path
    hydro_snapdir: Path
    group_catalog: Path
    params: np.ndarray
    box_size: float = 50.0
    npix: int = 1024
    patch_pix: int = 128
    proj_frac: float = 1.0
    halo_mass_min: float = 1e13

    @property
    def sim_label(self) -> str:
        return f"{self.suite}_{self.sim_id}"


@dataclass(frozen=True)
class RunConfig:
    """Execution options shared across simulations."""

    run_dir: Path
    checkpoint_path: Path
    output_root: Path
    model_name: str
    n_steps: int = 50
    batch_size: int = 16
    patch_mass_match: bool = True
    taper_frac: float = 0.15
    use_amp: bool = True
    device: str = "cuda"
    prep_only: bool = False
    regenerate: bool = False
    regenerate_all: bool = False
    repaste: bool = False
