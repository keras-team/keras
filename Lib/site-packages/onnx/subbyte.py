# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


def float32_to_4bit_unpacked(
    x: np.ndarray | np.dtype | float, signed: bool
) -> np.ndarray:
    """Cast to 4bit via rounding and clipping (without packing).

    Args:
        x: element to be converted
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int4 element (sign-extended to int8/uint8)
    """
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    clipped = np.clip(x, clip_low, clip_high)
    return np.rint(clipped).astype(dtype)  # type: ignore[no-any-return]


def float32x2_to_4bitx2(
    val_low: np.dtype, val_high: np.dtype, signed: bool
) -> np.ndarray:
    """Cast two elements to 4bit (via rounding and clipping) and pack
    to a single byte
    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int8/uint8 element, containing both int4 elements
    """
    i8_high = float32_to_4bit_unpacked(val_high, signed)
    i8_low = float32_to_4bit_unpacked(val_low, signed)
    return i8_high << 4 | i8_low & 0x0F  # type: ignore[operator]


def unpack_single_4bitx2(
    x: np.ndarray | np.dtype | float, signed: bool
) -> tuple[np.ndarray, np.ndarray]:
    unpack_signed = lambda x: np.where((x >> 3) == 0, x, x | 0xF0)  # noqa: E731
    """Unpack a single byte 4bitx2 to two 4 bit elements
    Args:
        x: Input data
        signed: boolean, whether to interpret as signed int4.
    Returns:
        A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    x_low = x & 0x0F
    x_high = x >> 4
    x_low = unpack_signed(x_low) if signed else x_low
    x_high = unpack_signed(x_high) if signed else x_high
    dtype = np.int8 if signed else np.uint8
    return (x_low.astype(dtype), x_high.astype(dtype))
