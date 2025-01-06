#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# QPyTorch documentation build configuration file, created by
# sphinx-quickstart on Mon Feb 11 00:48:31 2019.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# Mechanism to mock out modules
class ModuleMock(object):
    def __init__(self, *args, **kwargs):
        pass


# Putting all of our dirty hacks together
class Mock(MagicMock):
    __metaclass__ = type

    @classmethod
    def __getattr__(cls, name):
        if "Module" == name:
            return ModuleMock
        elif "Distribution" in name:
            return _Distribution
        elif "Normal" in name or "Gamma" in name or "Wishart" in name:
            return _SubDistribution
        elif "Kernel" in name or "Parallel" in name:
            return _Kernel
        else:
            res = MagicMock()
            res.Module = ModuleMock
            res.__metaclass__ = type
            return res


MOCK_MODULES = [
    "torch",
    "torch.autograd",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.parallel",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.cpp_extension",
    "numpy",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

import sphinx_rtd_theme

examples_source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples", "tutorial")
)
examples_dest = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "examples", "tutorial")
)

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.makedirs(examples_dest)

for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)
            shutil.copyfile(source_filename, dest_filename)

    exclude_patterns = ["_build", "**.ipynb_checkpoints", "examples/**/README.md"]

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["ntemplates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "QPyTorch"
copyright = "2019, Tianyi Zhang, Zhiqiu Lin, Guandao Yang, Christopher De Sa"
author = "Tianyi Zhang, Zhiqiu Lin, Guandao Yang, Christopher De Sa"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.0.1"
# The full version, including alpha/beta/rc tags.
release = "0.0.1 alpha"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["nstatic"]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "QPyTorchdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "QPyTorch.tex",
        "QPyTorch Documentation",
        "Tianyi Zhang, Zhiqiu Lin, Christopher De Sa",
        "manual",
    ),
]

autodoc_inherit_docstrings = False
# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "qpytorch", "QPyTorch Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "QPyTorch",
        "QPyTorch Documentation",
        author,
        "QPyTorch",
        "One line description of project.",
        "Miscellaneous",
    ),
]
