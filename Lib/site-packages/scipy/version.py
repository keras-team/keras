
"""
Module to expose more detailed version info for the installed `scipy`
"""
version = "1.15.2"
full_version = version
short_version = version.split('.dev')[0]
git_revision = "0f1fd4a7268b813fa2b844ca6038e4dfdf90084a"
release = 'dev' not in version and '+' not in version

if not release:
    version = full_version
