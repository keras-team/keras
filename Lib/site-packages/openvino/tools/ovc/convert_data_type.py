# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.ovc.error import Error

"""
Packed data of custom types are stored in numpy uint8 data type.
To distinguish true uint8 and custom data we introduce this class not to store,
but to have unique data type in SUPPORTED_DATA_TYPES map
"""


class packed_U1(np.generic):
    pass


class packed_U4(np.generic):
    pass


class packed_I4(np.generic):
    pass


SUPPORTED_DATA_TYPES = {
    'float': (np.float32, 'FP32', 'f32'),
    'half': (np.float16, 'FP16', 'f16'),
    'FP32': (np.float32, 'FP32', 'f32'),
    'FP64': (np.float64, 'FP64', 'f64'),
    'FP16': (np.float16, 'FP16', 'f16'),
    'I32': (np.int32, 'I32', 'i32'),
    'I64': (np.int64, 'I64', 'i64'),
    'int8': (np.int8, 'I8', 'i8'),
    'int32': (np.int32, 'I32', 'i32'),
    'int64': (np.int64, 'I64', 'i64'),
    'bool': (bool, 'BOOL', 'boolean'),
    'uint8': (np.uint8, 'U8', 'u8'),
    'uint32': (np.uint32, 'U32', 'u32'),
    'uint64': (np.uint64, 'U64', 'u64'),

    # custom types
    'U1': (packed_U1, 'U1', 'u1'),
    'int4': (packed_I4, 'I4', 'i4'),
    'uint4': (packed_U4, 'U4', 'u4'),
    'I4': (packed_I4, 'I4', 'i4'),
    'U4': (packed_U4, 'U4', 'u4'),
}


def data_type_str_to_np(data_type_str: str):
    return SUPPORTED_DATA_TYPES[data_type_str][0] if data_type_str in SUPPORTED_DATA_TYPES else None


def data_type_str_to_precision(data_type_str: str):
    return SUPPORTED_DATA_TYPES[data_type_str][1] if data_type_str in SUPPORTED_DATA_TYPES else None


def data_type_str_to_destination_type(data_type_str: str):
    return SUPPORTED_DATA_TYPES[data_type_str][2] if data_type_str in SUPPORTED_DATA_TYPES else None


def np_data_type_to_precision(np_data_type):
    for np_t, precision, _ in SUPPORTED_DATA_TYPES.values():
        if np_t == np_data_type:
            return precision
    raise Error('Data type "{}" is not supported'.format(np_data_type))


def np_data_type_to_destination_type(np_data_type):
    for np_t, _, destination_type in SUPPORTED_DATA_TYPES.values():
        if np_t == np_data_type:
            return destination_type
    raise Error('Data type "{}" is not supported'.format(np_data_type))


def precision_to_destination_type(data_type_str):
    for _, precision, destination_type in SUPPORTED_DATA_TYPES.values():
        if precision == data_type_str:
            return destination_type
    raise Error('Data type "{}" is not supported'.format(data_type_str))
