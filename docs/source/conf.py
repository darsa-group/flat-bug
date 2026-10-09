# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# The site has four parts: the home page (index.md), the user guide, the models page and the API
# reference. The models table and the per-model pages are generated at build time by
# _ext/fb_models.py from _static/models/, so publishing a model or a benchmark result means adding
# files there, not editing pages.

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))

# -- Project information -----------------------------------------------------

project = "flat-bug"
copyright = "2024-2026, the flat-bug authors"
author = "Asger Svenning, Quentin Geissmann and contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "fb_models",  # _ext/fb_models.py: the models table and one page per model, from _static/models/
]
myst_enable_extensions = ["colon_fence", "attrs_inline", "attrs_block"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_generated"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "flat-bug"
html_static_path = ["_static"]
html_css_files = ["flatbug.css"]
# DARSA green (darsa.info), with a lighter shade for dark backgrounds.
html_theme_options = {
    "source_repository": "https://github.com/darsa-group/flat-bug/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#017b33",
        "color-brand-content": "#017b33",
        "color-brand-visited": "#015220",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3fcf8e",
        "color-brand-content": "#3fcf8e",
        "color-brand-visited": "#2fae74",
    },
}

# The figure on the home page is the one in the repository README.
_here = os.path.dirname(__file__)
os.makedirs(os.path.join(_here, "_static"), exist_ok=True)
shutil.copy(os.path.join(_here, "..", "..", "prediction.jpg"), os.path.join(_here, "_static", "prediction.jpg"))
