# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'EquiViT'
copyright = '2026, Chih-Chun Wang'
author = 'Chih-Chun Wang'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx.ext.intersphinx'
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'breeze'
html_theme_options = {
    'header_tabs': False,
    # 'navigation_depth': 4,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']



### Custom settings (2026-04-14)
autodoc_typehints = "description"
# maximum_signature_line_length = 80

# pygments_style = 'catppuccin-mocha'

# Enable Catppuccin for code syntax highlighting
# pygments_style = "catppuccin-mocha"  # Options: latte, frappe, macchiato, mocha
# pygments_dark_style = "catppuccin-mocha"

# html_theme_options = {
#     "light_css_variables": {
#         # Latte (Light) palette
#         "color-brand-primary": "#1e66f5",        # Blue
#         "color-brand-content": "#8839ef",        # Mauve
#         "color-admonition-background": "#eff1f5", # Base
#     },
#     "dark_css_variables": {
#         # Mocha (Dark) palette
#         "color-brand-primary": "#89b4fa",        # Blue
#         "color-brand-content": "#cba6f7",        # Mauve
#         "color-sidebar-background": "#181825",   # Mantle
#         "color-admonition-background": "#1e1e2e", # Base
#     },
# }


# conf.py

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}