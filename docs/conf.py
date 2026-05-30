"""Sphinx configuration for BIND."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project ----------------------------------------------------------------
project = "BIND"
author = "Matthew Ho Lee"
copyright = "2024–2026, Matthew Ho Lee"

try:
    from bind import __version__ as release  # type: ignore
except Exception:  # pragma: no cover - fallback if heavy deps not installed
    release = "0.1.0"
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
autodoc_mock_imports = [
    "torch",
    "lightning",
    "pytorch_lightning",
    "torch_ema",
    "MAS_library",
    "h5py",
    "huggingface_hub",
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "WORKLOG.md"]

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

ogp_site_url = "https://bind.readthedocs.io"
ogp_image = "_static/fig1_showcase.png"
