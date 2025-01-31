import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
print(sys.path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SnapFISH2'
copyright = '2024, Hongyu Yu'
author = 'Hongyu Yu'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',  # jupyter notebook
    'sphinx.ext.napoleon',  # numpy style
    'sphinx.ext.mathjax',  # math symbols
    'myst_parser'  # markdown files
]
numpydoc_show_class_members = False

root_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

autodoc_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Don't add a source link in the sidebar
html_show_sourcelink = False

# Allow shorthand references for main function interface
rst_prolog = """
.. currentmodule:: snapfish2
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# html_static_path = ['_static']
