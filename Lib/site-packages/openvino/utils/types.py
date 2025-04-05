# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Functions related to converting between Python and numpy types and openvino types."""

import logging
from typing import List, Union, Optional

import numpy as np

from openvino.exceptions import OVTypeError
from openvino import Node, Shape, Output, Type
from openvino.op import Constant

log = logging.getLogger(__name__)

TensorShape = List[int]
NumericData = Union[int, float, np.ndarray]
NumericType = Union[type, np.dtype]
ScalarData = Union[int, float]
NodeInput = Union[Node, NumericData]

openvino_to_numpy_types_map = [
    (Type.boolean, bool),
    (Type.boolean, np.bool_),
    (Type.f16, np.float16),
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.i8, np.int8),
    (Type.i16, np.int16),
    (Type.i32, np.int32),
    (Type.i64, np.int64),
    (Type.u8, np.uint8),
    (Type.u16, np.uint16),
    (Type.u32, np.uint32),
    (Type.u64, np.uint64),
    (Type.bf16, np.uint16),
    (Type.string, str),
    (Type.string, np.str_),
    (Type.string, bytes),
    (Type.string, np.bytes_),
]

openvino_to_numpy_types_str_map = [
    ("boolean", bool),
    ("boolean", np.bool_),
    ("f16", np.float16),
    ("f32", np.float32),
    ("f64", np.float64),
    ("i8", np.int8),
    ("i16", np.int16),
    ("i32", np.int32),
    ("i64", np.int64),
    ("u8", np.uint8),
    ("u16", np.uint16),
    ("u32", np.uint32),
    ("u64", np.uint64),
    ("string", str),
    ("string", np.str_),
    ("string", bytes),
    ("string", np.bytes_),
]


def get_element_type(data_type: NumericType) -> Type:
    """Return an ngraph element type for a Python type or numpy.dtype."""
    if data_type is int:
        log.warning("Converting int type of undefined bitwidth to 32-bit ngraph integer.")
        return Type.i32

    if data_type is float:
        log.warning("Converting float type of undefined bitwidth to 32-bit ngraph float.")
        return Type.f32

    ov_type = next(
        (ov_type for (ov_type, np_type) in openvino_to_numpy_types_map if np_type == data_type),
        None,
    )
    if ov_type:
        return ov_type

    raise OVTypeError("Unidentified data type %s", data_type)


def get_element_type_str(data_type: NumericType) -> str:
    """Return an ngraph element type string representation for a Python type or numpy dtype."""
    if data_type is int:
        log.warning("Converting int type of undefined bitwidth to 32-bit ngraph integer.")
        return "i32"

    if data_type is float:
        log.warning("Converting float type of undefined bitwidth to 32-bit ngraph float.")
        return "f32"

    ov_type = next(
        (ov_type for (ov_type, np_type) in openvino_to_numpy_types_str_map if np_type == data_type),
        None,
    )
    if ov_type:
        return ov_type

    raise OVTypeError("Unidentified data type %s", data_type)


def get_dtype(openvino_type: Type) -> np.dtype:
    """Return a numpy.dtype for an openvino element type."""
    np_type = next(
        (np_type for (ov_type, np_type) in openvino_to_numpy_types_map if ov_type == openvino_type),
        None,
    )

    if np_type:
        return np.dtype(np_type)

    raise OVTypeError("Unidentified data type %s", openvino_type)


def get_numpy_ctype(openvino_type: Type) -> type:
    """Return numpy ctype for an openvino element type."""
    np_type = next(
        (np_type for (ov_type, np_type) in openvino_to_numpy_types_map if ov_type == openvino_type),
        None,
    )

    if np_type:
        return np_type

    raise OVTypeError("Unidentified data type %s", openvino_type)


def get_ndarray(data: NumericData) -> np.ndarray:
    """Wrap data into a numpy ndarray."""
    if isinstance(data, np.ndarray):
        return data  # type: ignore
    return np.array(data)


def get_shape(data: NumericData) -> TensorShape:
    """Return a shape of NumericData."""
    if isinstance(data, np.ndarray):
        return data.shape  # type: ignore
    if isinstance(data, list):
        return [len(data)]  # type: ignore
    return []


def make_constant_node(value: NumericData, dtype: Union[NumericType, Type] = None, *, name: Optional[str] = None) -> Constant:
    """Return an openvino Constant node with the specified value."""
    ndarray = get_ndarray(value)
    if dtype is not None:
        element_type = get_element_type(dtype) if isinstance(dtype, (type, np.dtype)) else dtype
    else:
        element_type = get_element_type(ndarray.dtype)

    const = Constant(element_type, Shape(ndarray.shape), ndarray.flatten().tolist())

    if name:
        const.friendly_name = name + "/" + const.friendly_name

    return const


def as_node(input_value: NodeInput, name: Optional[str] = None) -> Node:
    """Return input values as nodes. Scalars will be converted to Constant nodes."""
    if issubclass(type(input_value), Node):
        return input_value
    if issubclass(type(input_value), Output):
        return input_value
    return make_constant_node(input_value, name=name)


def as_nodes(*input_values: NodeInput, name: Optional[str] = None) -> List[Node]:
    """Return input values as nodes. Scalars will be converted to Constant nodes."""
    return [as_node(input_value, name=name) for input_value in input_values]
