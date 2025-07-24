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
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(1, os.path.abspath('../'))
# sys.path.insert(2, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(1, os.path.abspath('../..'))

print(os.path.abspath(os.path.dirname(__file__)))
# print(os.path.abspath('../'))
print(os.path.abspath('../..'))
from datetime import date
import re
import netflow
                
# -- Project information -----------------------------------------------------

project = 'netflow'
copyright = '2023, rena elkin'
author = 'rena elkin'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinxcontrib.fulltoc',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
    # "nbsphinx",
    'sphinx_copybutton',  #  for adding “copy to clipboard” buttons to all text/code boxes
    'sphinxcontrib.bibtex', # for references in jupyter notebook tutorials
]
bibtex_bibfiles = ['references.bib']
# pygments_style = 'sphinx'
# highlight_language = 'python'

# For using with MyST Parser, for Markdown documentation, it is recommended to use the colon_fence syntax extension:
# extensions.append("myst_parser")
# myst_enable_extensions = ["colon_fence"]

napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.DS_Store']


# -- Additional options -----------------------------------------
project = 'netflow'  # The documented project’s name.

# author : The author name(s) of the document. The default value is 'unknown'.
# author = ""

# A copyright statement in the style '2008, Author Name'.
# Changed in version 7.1: The value may now be a sequence of copyright statements in the above
# form, which will be displayed each to their own line.
copyright = '2023-{}, The {} community'.format(date.today().year, project)

version = re.sub(r'\.dev.*$', r'.dev', netflow.__version__)  # The major project version, e.g,, 2.6
release = version  # The full project version, e.g., 2.6.1

source_suffix = {'.rst': 'restructuredtext',   # The file extensions of source files.
                 '.ipynb': 'myst-nb',
                 '.myst': 'myst-nb',
                 }

master_doc = 'index'  # document with the root toctree directive.

# exclude_patterns = []  # glob

# A boolean that decides whether parentheses are appended to function and method role
# text (e.g. the content of :func:`input`) to signify that the name is callable.
# Default is True.
add_function_parentheses = True


# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for py:function
# directives. Default is True.
# add_module_names = True

nitpicky = False  # True
geo_ennitpick_ignore = []

# Create table of contents entries for domain objects (e.g. functions, classes,
# attributes, etc.). Default is True.
# toc_object_entries = True

# A string that determines how domain objects (e.g. functions, classes, attributes, etc.)
# are displayed in their table of contents entry.
# - Use domain to allow the domain to determine the appropriate number of parents to show.
#   For example, the Python domain would show Class.method() and function(), leaving out
#   the module. level of parents. This is the default setting.
# - Use hide to only show the name of the element without any parents (i.e. method()).
# - Use all to show the fully-qualified name for the object (i.e. module.Class.method()),
#   displaying all parents.
# toc_object_entries_show_parents = 'all'

# decides whether codeauthor and sectionauthor directives produce output in the built files.
# show_authors = False

# Set this option to True if you want all displayed math to be numbered. The default is False.
# math_number_all = False

# A string used for formatting the labels of references to equations.
# The {number} place-holder stands for the equation number.
# Example: 'Eq.{number}' gets rendered as, for example, Eq.10.
# math_eqref_format = 'Eq.{number}'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'  # 'classic'  # 'alabaster'  # 

# for all theme options, see:
# https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
html_theme_options = {
    # 'stickysidebar': False,  # alabaster
    # 'collapsiblesidebar': True,  # alabaster
    "collapse_navigation": False,  # pydata_sphinx_theme
    "show_nav_level": 4,
    "navigation_depth": 6,
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
    # "show_toc_level": 2,  # pydata_sphinx_theme
    # "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],

    # "icon_links": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/Cistrome/MIRA",
    #         "icon": "fab fa-github-square",
    #     }
    # ],
    # "use_edit_page_button": False,
}  # Make the sidebar “fixed” so that it doesn’t scroll out of view for long body content.
                      

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Custamize HTML side bar - alabaster -------------------------------------
# The default html sidebar consists of 4 templates:
# ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
# you could change localtoc.html to globaltoc.html with the following: 
# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
# html_sidebars = {
#     'index' : []
# }

html_title = f"{project} v{version} Manual"
html_logo = "_static/NF_logo3.png"  # "_static/netflow_logo.png"
# html_favicon = "_static/NF_logo3.png"  # "_static/netflow_logo.png"

# html_css_files
html_css_files = ['custom.css']

html_last_updated_fmt = '%b %d, %Y'

###
html_copy_source = True
html_domain_indices = True  # If true, generate domain-specific indices in addition to the general index.
html_use_modindex = True

html_show_sphinx = True

htmlhelp_basename = project

autosummary_generate = True

maximum_signature_line_length = 3

html_show_sourcelink = True

# plot_html_show_source_link = False

nb_execution_mode = 'off' # 'cache'  # Do not execute the notebook
# nb_kernel_rgx_aliases = {"*": "geo_env_test"}

# html_sourcelink_suffix = ''
# nbsphinx_execute = 'never'  # Whether to execute notebooks before conversion or not. ('auto', 'always', 'never')
# nbsphinx_kernel_name = 'geo_env_test'  # 'python3'

# -- intersphinx configuration ----------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}


# -- autodoc -----------------
autodoc_default_options = {
    'members': None, # Include all members (methods).
    'special-members': None,  # “special” members (e.g.,  __special__)
    'exclude-members': '__dict__,__weakref__', # Exclude "standard" methods.
    'inherited-members': None,  # For classes and exceptions, to show members inherited from base classes 
    'undoc-members': None,  # Members without docstrings
    'private-members': None,  # “Private” members, e.g., _private
}


# This value controls how to represent typehints. The setting takes the following values:
# 'signature' – Show typehints in the signature (default)
# 'description' – Show typehints as content of the function or method The typehints of
#                 overloaded functions or methods will still be represented in the signature.
# 'none' – Do not show typehints
# 'both' – Show typehints in the signature and as content of the function or method
# autodoc_typehints = 'description'

# autodoc_class_signature = "separated"

