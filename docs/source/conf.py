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
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'nbsphinx',  # jupyter notebook
    'sphinx.ext.napoleon',  # numpy style
    'sphinx.ext.mathjax',  # math symbols
    'myst_parser'  # markdown files
]
numpydoc_show_class_members = False

# templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
