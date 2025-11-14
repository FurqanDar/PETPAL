# # Configuration file for the Sphinx documentation builder.
#
from datetime import datetime

project = 'PETPAL (Positron Emission Tomography Analysis Library)'
year = datetime.today().year
copyright = f'2024-{year}, Furqan Dar, Bradley Judge, Noah Goldman, Kenan Oestreich'
author = 'Furqan Dar, Bradley Judge, Noah Goldman, Kenan Oestreich'
release = '0.1.0'

extensions = [
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
]

language = 'en'

html_static_path = ['_static']
html_title = 'PETPAL'
html_split_index = False
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "secondary_sidebar_items": {
        "**/*": ["page-toc", "edit-this-page", "sourcelink"]
    },
    "show_toc_level" : 3,
    "icon_links": [
            {
            "name": "GitHub",
            "url" : "https://github.com/FurqanDar/PETPAL",
            "icon": "fa-brands fa-github",
            },
                 ],
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
}

exclude_patterns = ['_build']
templates_path = ["_templates"]
source_suffix = '.rst'
master_doc = 'index'

toc_object_entries_show_parents = "hide"
todo_include_todos = True

# autoapi configuration
autoapi_type = 'python'
autoapi_dirs = ['../petpal']

# Options: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#customisation-options
autoapi_options = [
    'members',
    'undoc-members',
    'inherited-members',
    'private-members',
    'special-members',
    'show-inheritance',
    'show-module-summary'
]

autoapi_ignore = ['*cli*']
autoapi_own_page_level = 'function'
autoapi_template_dir = '_templates'
autoapi_python_class_content = 'both'
autoapi_member_order = 'bysource'
autoapi_keep_files = True
autoapi_generate_api_docs = True
autoapi_toctree_entries = True
autoapi_use_implicit_namespaces= False

napoleon_use_ivar = True
napoleon_use_rtype = False

intersphinx_mapping = {
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'ants': ('https://antspy.readthedocs.io/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'lmfit': ('https://lmfit.github.io/lmfit-py/', None),
}

def skip_main_funcs(app, what, name, obj, skip, options):
    if "main" in name and what == "function":
       skip = True
    elif what=="attribute":
       skip = True
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_main_funcs)