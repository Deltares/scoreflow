import os
import shutil
import sys
from pathlib import Path

from veriflow.constants import VERSION

sys.path.insert(0, os.path.abspath("../../src"))


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples"
DOC_NOTEBOOKS = Path(__file__).parent / "examples"

if DOC_NOTEBOOKS.exists():
    shutil.rmtree(DOC_NOTEBOOKS)

shutil.copytree(NOTEBOOKS, DOC_NOTEBOOKS)

is_github_main = (
    os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REF_NAME") == "main"
)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "veriflow"
copyright = "2026, Deltares"
author = "Jurian Beunk"

release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "myst_parser",  # Markdown support
    "nbsphinx",
    "sphinxcontrib.mermaid",  # Mermaid diagrams in docstrings/pages
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]

# ``examples/index.ipynb`` is a stray overview notebook that is superseded by the
# case-based Gallery pages; exclude it so it is not built as an orphan document.
exclude_patterns = ["examples/index.ipynb"]

# Links to external documentation pages
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scores": ("https://scores.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# Copy the versioned JSON schemas into the published site at the root, so they
# are reachable at e.g. https://deltares.github.io/veriflow/v0/config.schema.json.
# This is the canonical URL referenced from YAML configs via the
# ``# yaml-language-server: $schema=...`` modeline for IDE validation.
html_extra_path = ["../../schemas"]
html_title = "Verification "
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # "switcher": {
    #     "json_url": "_static/versions.json",  # see below
    #     "version_match": "current",
    # },
    "navigation_depth": 3,
}

# Autodoc
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "exclude-members": "model_config",
    "members": True,
    "undoc-members": True,  # show StrEnum members from constants
    # Document fields inherited from our own base classes (e.g. the shared
    # ``import_adapter``/``source``/``data_type`` fields on BaseDatasourceConfig,
    # or the ``score_adapter`` field on BaseScoreConfig) instead of only the
    # attributes redefined on the leaf class. The listed ancestors are excluded
    # so Pydantic's ``BaseModel``/``BaseSettings`` internals stay out of the docs.
    "inherited-members": "BaseModel, BaseSettings",
}

# We always execute notebooks in the docs build on GitHub main, to make sure they are up to date
# and working. Locally, it can take a long time to execute all notebooks, so we skip execution
# unless explicitly requested.
nbsphinx_execute = "auto" if is_github_main else "never"  # options: 'auto', 'always', 'never'
nbsphinx_kernel_name = "python3"  # kernel to use for notebook execution
nbsphinx_timeout = 600  # seconds per notebook

# Repository coordinates used to build "view source" / Binder links for notebooks.
GITHUB_REPO = "Deltares/veriflow"
GITHUB_BRANCH = "main"

# Prepended to every rendered notebook. Adds a banner with a "view on GitHub"
# link, an "Open in Binder" badge, and a direct download link for the notebook.
# ``env.doc2path`` yields the source path relative to ``docs/source`` (e.g.
# ``examples/The Rhine Case/1a_basics.ipynb``), which matches the layout of the
# notebooks inside the repository, so the same path works for GitHub and Binder.
nbsphinx_prolog = (
    r"""
{% set docpath = env.doc2path(env.docname, base=None) %}
{% set notebook = env.docname.split('/')|last + '.ipynb' %}
{% set displaypath = docpath|replace('\\', '/') %}
{% set urlpath = docpath|replace('\\', '/')|replace(' ', '%20')|replace('(', '%28')|replace(')', '%29') %}

.. raw:: html

    <div class="admonition note nbsphinx-prolog">
      <p class="admonition-title">Notebook</p>
      <p>

        Run it live:
        <a class="reference external" href="https://mybinder.org/v2/gh/"""
    + GITHUB_REPO
    + r"""/"""
    + GITHUB_BRANCH
    + r"""?labpath={{ urlpath }}"><img alt="Open in Binder" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>
      </p>
    </div>

.. nbinfo::

    Download this notebook: :download:`{{ notebook }} <{{ notebook }}>`
"""
)
