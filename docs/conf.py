"""Sphinx configuration for BIND."""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the package importable for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project ----------------------------------------------------------------
project = "BIND"
author = "Max E. Lee"
copyright = "2024–2026, Max E. Lee"

# The docs build deliberately does NOT import the package: `import bind` pulls in
# torch / lightning / Pylians, which we do not install here (see
# `autodoc_mock_imports` below).  Read the version straight out of the source
# instead, so this file has no runtime dependency on the package at all.
_INIT = Path(__file__).resolve().parents[1] / "src" / "bind" / "__init__.py"
_m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', _INIT.read_text(), re.M)
if _m is None:  # pragma: no cover - only if src/bind/__init__.py is restructured
    raise RuntimeError(f"could not parse __version__ from {_INIT}")
release = _m.group(1)
version = ".".join(release.split(".")[:2])

# -- General ----------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
]

# Heavy runtime deps that must not be required at docs-build time.
#
# CONTRACT: the docs build installs `docs/requirements.txt` only — Sphinx, the
# theme, and the two lightweight libraries that BIND's *documented* modules use
# at import time (numpy, pandas).  The package itself is NOT pip-installed; it is
# imported from ../src via the sys.path entry above, with everything below
# replaced by autodoc mocks.  Keep this list in sync with the third-party
# module-scope imports under src/bind/ (`grep -rn '^import\|^from' src/bind`), or
# autodoc will fail with an ImportError on a real dependency.
autodoc_mock_imports = [
    "torch",
    "lightning",
    "pytorch_lightning",
    "torch_ema",
    "MAS_library",   # Pylians
    "Pk_library",    # Pylians
    "h5py",
    "huggingface_hub",
    "tqdm",
    "scipy",
    "mpi4py",
    "gpytorch",
    "sklearn",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
# WORKLOG.md is the in-repo session log and plans/ holds internal working notes;
# neither is user documentation, and both would otherwise warn as orphan pages.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "WORKLOG.md", "plans"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- HTML -------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "BIND"
html_logo = None
html_favicon = None

html_theme_options = {
    "repository_url": "https://github.com/Maxelee/BIND",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "logo": {"text": "BIND"},
}

ogp_site_url = "https://github.com/Maxelee/BIND"
ogp_image = "_static/fig1_showcase.png"
