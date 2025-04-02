# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'flat-bug'
copyright = '2024, Asger Svenning, Quentin Geissmann'
author = 'Asger Svenning, Quentin Geissmann'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "navigation_depth": 2,
}
html_static_path = ["_static"]

# Myst setup
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.abspath('../src/bin'))

def copy_and_adjust_readme():
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "_static"))
    readme_path = os.path.join(source_dir, "README.md")
    processed_readme_path = os.path.join(os.path.dirname(__file__), "README_sphinx.md")

    os.makedirs(dest_dir, exist_ok=True)

    with open(readme_path, "r") as file:
        content = file.read()

        # Find and copy images while adjusting paths in README content
        image_paths = re.findall(r'<img src="([^"]+)"', content)
        for image_path in image_paths:
            src_path = os.path.join(source_dir, image_path)
            dest_path = os.path.join(dest_dir, os.path.basename(image_path))
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                # Replace original path with the path pointing to _static
                content = content.replace(image_path, f"_static/{os.path.basename(image_path)}")
            else:
                print(f"Warning: {src_path} not found.")

    # Write modified README content for Sphinx
    with open(processed_readme_path, "w") as file:
        file.write(content)

copy_and_adjust_readme()

# Ensure Sphinx recognizes _static
html_static_path = ["_static"]
