"""Sphinx configuration for QisDebug documentation."""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR.parent / "src"))

import qisdebug  # noqa: E402

project = "QisDebug"
copyright = "2026, Krishna Upadhyay"
author = "Krishna Upadhyay"
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
myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "model_config",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Methods overriding a third-party base (e.g. Textual's Screen.compose) would
# otherwise inherit that library's Markdown docstring, which isn't valid reST.
autodoc_inherit_docstrings = False
# The MCP SDK is an optional extra imported at module level by the server, so
# it's mocked to let the server module be documented without installing it.
autodoc_mock_imports = ["mcp"]
# Every module uses postponed annotations, so a bare ``type`` hint reaches the
# Python domain as the string "type", whose fuzzy lookup then links it to one
# of the event models' ``type`` fields. Qualifying it avoids that wrong link.
autodoc_type_aliases = {"type": "builtins.type"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer", None),
}

html_theme = "furo"
html_title = f"QisDebug {version}"
# Read the Docs sets this to the project's canonical domain, so every version
# points search engines at one URL; empty for local builds.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

REFERENCE_DIR = DOCS_DIR / "reference"
# Documented on api.md from their __all__; also listing them here would
# describe the re-exported objects twice under the same name.
PUBLIC_MODULES = {"qisdebug", "qisdebug.visualizer"}


def _public_names_defined_in(module_name: str) -> list[tuple[str, str]]:
    """Return ``(public module, name)`` for each public object defined in ``module_name``.

    Those objects are documented on api.md under their public path, so the
    internals page skips them rather than giving cross-references two
    equally valid targets.
    """
    found = []
    for public in sorted(PUBLIC_MODULES):
        module = importlib.import_module(public)
        for name in getattr(module, "__all__", ()):
            if getattr(getattr(module, name), "__module__", None) == module_name:
                found.append((public, name))
    return sorted(found, key=lambda pair: pair[1])


def _skip_foreign_members(_app, what, _name, obj, skip, _options):
    """Skip module-level names imported from other packages.

    Sphinx's autodoc-skip-member event fixes this signature. A module's
    ``from pydantic import Field`` would otherwise be inspected as one of its
    members, sending sphinx-autodoc-typehints into third-party annotations it
    can't resolve.
    """
    defined_in = getattr(obj, "__module__", None)
    if what == "module" and isinstance(defined_in, str) and not defined_in.startswith("qisdebug"):
        return True
    return skip


def setup(app) -> None:
    """Register the docs build's event handlers."""
    app.connect("autodoc-skip-member", _skip_foreign_members)


def _write_reference_pages() -> None:
    """Write one autodoc page per qisdebug module, private modules included.

    Walking the package at build time means a new module is documented the
    next time the docs build, with no hand-maintained list to forget. A page
    is only rewritten when its content changes, so incremental builds don't
    treat every page as modified.
    """
    REFERENCE_DIR.mkdir(exist_ok=True)
    modules = sorted(
        info.name
        for info in pkgutil.walk_packages(qisdebug.__path__, prefix="qisdebug.")
        if info.name not in PUBLIC_MODULES
    )
    expected = set()
    for name in modules:
        page = REFERENCE_DIR / f"{name}.rst"
        expected.add(page.name)
        public = _public_names_defined_in(name)
        content = (
            f"{name}\n{'=' * len(name)}\n\n"
            f".. automodule:: {name}\n"
            "   :members:\n"
            "   :undoc-members:\n"
            "   :private-members:\n"
            "   :show-inheritance:\n"
        )
        if public:
            names = ", ".join(name for _, name in public)
            links = ", ".join(f":py:obj:`{module}.{name}`" for module, name in public)
            content += f"   :exclude-members: model_config, {names}\n"
            content += f"\nAlso defined here, and documented in :doc:`/api`: {links}.\n"
        if not page.exists() or page.read_text(encoding="utf-8") != content:
            page.write_text(content, encoding="utf-8")
    for stale in REFERENCE_DIR.glob("*.rst"):
        if stale.name not in expected:
            stale.unlink()


_write_reference_pages()
