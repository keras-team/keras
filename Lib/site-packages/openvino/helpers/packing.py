# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import numpy as np
from typing import Union
from openvino import Type, Shape


def pack_data(array: np.ndarray, type: Type) -> np.ndarray:
    """Represent array values as u1,u4 or i4 openvino element type and pack them into uint8 numpy array.

    If the number of elements in array is odd we pad them with zero value to be able to fit the bit
    sequence into the uint8 array.

    Example: two uint8 values - [7, 8] can be represented as uint4 values and be packed into one int8
             value - [120], because [7, 8] bit representation is [0111, 1000] will be viewed
             as [01111000], which is bit representation of [120].

    :param array: numpy array with values to pack.
    :type array: numpy array
    :param type: Type to interpret the array values. Type must be u1, u4, i4, nf4 or f4e2m1.
    :type type: openvino.Type
    """
    assert type in [Type.u1, Type.u4, Type.i4, Type.nf4, Type.f4e2m1], "Packing algorithm for the" "data types stored in 1, 2 or 4 bits"

    minimum_regular_dtype = np.int8 if type == Type.i4 else np.uint8
    casted_to_regular_type = array.astype(dtype=minimum_regular_dtype, casting="unsafe")
    if not np.array_equal(casted_to_regular_type, array):
        raise RuntimeError(f'The conversion of array "{array}" to dtype' f' "{casted_to_regular_type}" results in rounding')

    data_size = casted_to_regular_type.size
    num_bits = type.bitwidth

    assert num_bits < 8 and 8 % num_bits == 0, "Packing algorithm for the" "data types stored in 1, 2 or 4 bits"
    num_values_fitting_into_uint8 = 8 // num_bits
    pad = (-data_size) % num_values_fitting_into_uint8

    flattened = casted_to_regular_type.flatten()
    padded = np.concatenate((flattened, np.zeros([pad], dtype=minimum_regular_dtype)))  # type: ignore
    assert padded.size % num_values_fitting_into_uint8 == 0

    bit_order_little = (padded[:, None] & (1 << np.arange(num_bits)) > 0).astype(minimum_regular_dtype)
    bit_order_big = np.flip(bit_order_little, axis=1)  # type: ignore
    bit_order_big_flattened = bit_order_big.flatten()

    return np.packbits(bit_order_big_flattened)


def unpack_data(array: np.ndarray, type: Type, shape: Union[list, Shape]) -> np.ndarray:
    """Extract openvino element type values from array into new uint8/int8 array given shape.

    Example: uint8 value [120] can be represented as two u4 values and be unpacked into [7, 8]
             because [120] bit representation is [01111000] will be viewed as [0111, 1000],
             which is bit representation of [7, 8].

    :param array: numpy array to unpack.
    :type array: numpy array
    :param type: Type to extract from array values. Type must be u1, u4, i4, nf4 or f4e2m1.
    :type type: openvino.Type
    :param shape: the new shape for the unpacked array.
    :type shape: Union[list, openvino.Shape]
    """
    assert type in [Type.u1, Type.u4, Type.i4, Type.nf4, Type.f4e2m1], "Unpacking algorithm for the" "data types stored in 1, 2 or 4 bits"
    unpacked = np.unpackbits(array.view(np.uint8))
    shape = list(shape)
    if type.bitwidth == 1:
        return np.resize(unpacked, shape)
    else:
        unpacked = unpacked.reshape(-1, type.bitwidth)
        padding_shape = (unpacked.shape[0], 8 - type.bitwidth)
        padding = np.ndarray(padding_shape, np.uint8)  # type: np.ndarray
        if type == Type.i4:
            for axis, bits in enumerate(unpacked):
                if bits[0] == 1:
                    padding[axis] = np.ones((padding_shape[1],), np.uint8)
                else:
                    padding[axis] = np.zeros((padding_shape[1],), np.uint8)
        else:
            padding = np.zeros(padding_shape, np.uint8)
        padded = np.concatenate((padding, unpacked), 1)  # type: ignore
        packed = np.packbits(padded, 1)
        if type == Type.i4:
            return np.resize(packed, shape).astype(dtype=np.int8)
        else:
            return np.resize(packed, shape)
