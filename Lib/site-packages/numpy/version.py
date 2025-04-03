
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.0.2"
__version__ = version
full_version = version

git_revision = "854252ded83e6b9c21c4ee80558d354d8a72484c"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
