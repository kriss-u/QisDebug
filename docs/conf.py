"""Sphinx configuration for QisDebug documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import qisdebug  # noqa: E402

project = "QisDebug"
copyright = "2026, QisDebug contributors"
author = "QisDebug contributors"
release = qisdebug.__version__
version = qisdebug.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"QisDebug {version}"
