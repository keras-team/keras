"""Setup script."""

import pathlib

from setuptools import find_packages
from setuptools import setup

HERE = pathlib.Path(__file__).parent

setup(
    name="keras-core",
    description="Multi-backend Keras.",
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/keras-team/keras-core",
    author="Keras team",
    author_email="keras@google.com",
    license="Apache License 2.0",
    install_requires=[
        "absl-py",
        "numpy",
        "packaging",
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
