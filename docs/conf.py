# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("mmcontext")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

# Bibtex configuration
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_encoding = "utf-8"

templates_path = ["_templates"]
nitpicky = False  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "mengerj",
    "github_repo": project_name,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]


autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_imported_members = False
autosummary_ignore_module_all = False
autosummary_mock_imports = []
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_use_ivar = True  # Make sure napoleon respects instance variables
autodoc_typehints = "none"  # Prevent autodoc from trying to automatically link type hints
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Add more comprehensive warning suppression for Read the Docs
suppress_warnings = [
    "docutils",
    "docutils.parsers.rst",
    "docutils.parsers.rst.states",
    "docutils.parsers.rst.directives",
    "docutils.utils",
    "ref.class",  # To suppress warnings related to unresolved class references
    "ref.attr",
    "ref.func",
    "ref.meth",
    "autosummary",
    "toc.not_readable",
    "epub.unknown_project_files",
    "app.add_source_suffix",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project_name

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "torch.utils.hooks.RemovableHandle"),
    ("py:attr", "torch.nn.modules.module.Module"),
]
