# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]

from onnx._custom_element_types import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
    int4,
    uint4,
)

_supported_types = [
    (bfloat16, "bfloat16", "bfloat16"),
    (float8e4m3fn, "e4m3fn", "float8_e4m3fn"),
    (float8e4m3fnuz, "e4m3fnuz", "float8_e4m3fnuz"),
    (float8e5m2, "e5m2", "float8_e5m2"),
    (float8e5m2fnuz, "e5m2fnuz", "float8_e5m2fnuz"),
    (int4, "int4", "int4"),
    (uint4, "uint4", "uint4"),
]


def convert_from_ml_dtypes(array: np.ndarray) -> np.ndarray:
    """Detects the type and changes into one of the ONNX
    defined custom types when ``ml_dtypes`` is installed.

    Args:
        array: Numpy array with a dtype from ml_dtypes.

    Returns:
        numpy array
    """
    if not ml_dtypes:
        return array
    for dtype, _, ml_name in _supported_types:
        if array.dtype == getattr(ml_dtypes, ml_name):
            return array.view(dtype=dtype)
    return array


def convert_to_ml_dtypes(array: np.ndarray) -> np.ndarray:
    """Detects the type and changes into one of the type
    defined in ``ml_dtypes`` if installed.

    Args:
        array: array

    Returns:
        numpy Numpy array with a dtype from ml_dtypes.
    """
    new_dt = None

    for dtype, name, ml_name in _supported_types:
        if array.dtype == dtype and array.dtype.descr[0][0] == name:
            assert ml_dtypes, (
                f"ml_dtypes is not installed and the tensor cannot "
                f"be converted into ml_dtypes.{array.dtype.descr[0][0]}"
            )
            new_dt = getattr(ml_dtypes, ml_name)
            break

    if new_dt:
        # int4, uint4, the representation uses 1 byte per element,
        # only onnx storage uses 1 byte for two elements
        return array.view(dtype=new_dt)

    return array
