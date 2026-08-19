"""The packaged assets must resolve from the installed package.

These are the tests that catch a broken wheel: `bind` ships three data files in
``src/bind/assets/`` (declared in ``[tool.setuptools.package-data]``) and three
different subsystems resolve them relative to ``__file__``.  If package-data
stops being installed, or a new asset extension is added without updating the
build config, everything below fails while `import bind` still succeeds.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import bind
from bind import data as D
from bind import params as PA

PACKAGE_DIR = Path(bind.__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"

# The glob suffixes declared in pyproject's [tool.setuptools.package-data].
# Adding an asset with a new suffix requires updating that list too.
PACKAGED_SUFFIXES = {".csv", ".txt", ".npz"}


def test_assets_directory_ships_with_the_package():
    assert ASSETS_DIR.is_dir(), f"missing packaged assets dir: {ASSETS_DIR}"
    files = [p for p in ASSETS_DIR.iterdir() if p.is_file()]
    assert files, "assets directory is empty"
    unpackaged = sorted(p.name for p in files if p.suffix not in PACKAGED_SUFFIXES)
    assert not unpackaged, (
        f"assets with suffixes outside the package-data globs {PACKAGED_SUFFIXES}: "
        f"{unpackaged} -- update [tool.setuptools.package-data] in pyproject.toml"
    )


def test_sb35_minmax_csv_resolves_from_the_package():
    csv = Path(D.SB35_CSV).resolve()
    assert csv.parent == ASSETS_DIR and csv.exists()
    assert Path(PA.SB35_CSV).resolve() == csv     # data.py and params.py agree

    for arr in (D.PARAM_LOG_FLAG, D.PARAM_MIN_RAW, D.PARAM_MAX_RAW,
                D.PARAM_MIN_NORM, D.PARAM_MAX_NORM):
        assert arr.shape == (35,)
    assert set(np.unique(D.PARAM_LOG_FLAG)) <= {0, 1}
    assert (D.PARAM_MIN_RAW < D.PARAM_MAX_RAW).all()
    assert (D.PARAM_MIN_NORM < D.PARAM_MAX_NORM).all()
    # Log-flagged bounds are stored in log10 space.
    log = D.PARAM_LOG_FLAG == 1
    np.testing.assert_allclose(D.PARAM_MIN_NORM[log], np.log10(D.PARAM_MIN_RAW[log]),
                               rtol=1e-6)


def test_sb35_param_table_resolves_and_matches_the_csv_ordering():
    txt = Path(D.SB35_PARAMS_TXT).resolve()
    assert txt.parent == ASSETS_DIR and txt.exists()

    table = D.load_sb35_param_table()
    assert table.ndim == 2 and table.shape[1] == PA.N_PARAMS == 35
    assert table.shape[0] > 0
    assert np.isfinite(table).all()

    # The two assets must agree on column order, or every parameter vector
    # looked up by sim id is silently permuted relative to the bounds.
    header = txt.read_text().splitlines()[0].split()
    assert header[1:-1] == PA.PARAM_NAMES     # drop '#Name' and the trailing 'seed'


def test_wlemu_loads_its_packaged_artifact_and_predicts():
    from bind.wlemu import BLOCKS, WLEmulator

    artifact = (ASSETS_DIR / "wlemu_gp.npz")
    assert artifact.exists()

    emu = WLEmulator.load()
    assert emu.n_params == 30
    assert len(emu.param_names) == 30

    pred = emu.predict(emu.fiducial_params(), z_source=1.0)
    total = 0
    for block in BLOCKS:
        assert block in pred and f"{block}_std" in pred
        y, std = pred[block], pred[f"{block}_std"]
        assert y.ndim == 1 and y.shape == std.shape
        assert np.isfinite(y).all() and np.isfinite(std).all()
        assert (std >= 0).all()
        total += y.shape[0]
    assert total == 383      # the documented summary-statistic dimensionality


def test_wlemu_rejects_an_out_of_range_source_redshift():
    from bind.wlemu import WLEmulator

    emu = WLEmulator.load()
    with pytest.raises(ValueError):
        emu.predict(emu.fiducial_params(), z_source=float(emu.source_redshifts[-1]) + 1.0)


def test_top_level_api_is_importable():
    """The names README/docs tell users to import must exist on `bind`."""
    for name in ("paint", "Simulation", "Model", "PaintResult",
                 "fiducial_params", "random_params", "vary_param", "vary_params",
                 "param_dataframe", "THERMO_KEYS", "N_THERMO", "PATCH_PIX"):
        assert hasattr(bind, name), f"bind.{name} is missing"


def test_version_matches_pyproject():
    """src/bind/__init__.py and pyproject.toml must not drift apart."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():          # running against an installed wheel
        pytest.skip("pyproject.toml not available")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert match, "no version field in pyproject.toml"
    assert bind.__version__ == match.group(1)
