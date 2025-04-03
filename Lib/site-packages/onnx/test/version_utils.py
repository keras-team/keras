# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from packaging.version import parse as version


def numpy_older_than(ver: str) -> bool:
    """Returns True if the numpy version is older than the given version."""
    import numpy  # pylint: disable=import-outside-toplevel

    return version(numpy.__version__) < version(ver)


def pillow_older_than(ver: str) -> bool:
    """Returns True if the pillow version is older than the given version."""
    import PIL  # pylint: disable=import-outside-toplevel

    return version(PIL.__version__) < version(ver)
