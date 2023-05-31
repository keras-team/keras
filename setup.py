"""Setup script."""

import os
import pathlib

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = get_version("keras_core/__init__.py")

setup(
    name="keras-core",
    description="Multi-backend Keras.",
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/keras-team/keras-core",
    author="Keras team",
    author_email="keras@google.com",
    license="Apache License 2.0",
    install_requires=[
        "absl-py",
        "numpy",
        "packaging",
        "rich",
    ],
    # Supported Python versions
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
