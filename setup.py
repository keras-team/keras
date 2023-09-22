"""Setup script."""
import os
from setuptools import find_packages, setup

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as file:
        return file.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

# Get the version from the appropriate source file
if os.path.exists("keras/version.py"):
    VERSION = get_version("keras/version.py")
else:
    VERSION = get_version("keras/__init__.py")

setup(
    name="keras",
    description="Multi-backend Keras.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/keras-team/keras",
    author="Keras team",
    author_email="keras-users@googlegroups.com",
    license="Apache License 2.0",
    install_requires=[
        "absl-py",
        "numpy",
        "rich",
        "namex",
        "h5py",
        "dm-tree",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
