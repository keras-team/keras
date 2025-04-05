# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset13."""
from functools import partial, singledispatch
from typing import Literal, Optional, Union
import logging

import numpy as np

log = logging.getLogger(__name__)

from openvino import Node, Shape, Type, Output, Tensor
from openvino.op import Constant, Result
from openvino.opset1 import convert_like
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import binary_op, nameable_op, unary_op, overloading
from openvino.utils.types import (
    NumericData,
    NodeInput,
    NumericType,
    as_nodes,
    as_node,
)

_get_node_factory_opset13 = partial(_get_node_factory, "opset13")


# -------------------------------------------- ops ------------------------------------------------
@binary_op
def bitwise_and(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise AND operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_and.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise AND operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseAnd",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def bitwise_not(
    node: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise NOT operation on input node element-wise.

    For boolean input tensors, operator is equivalent to logical_not.

    :param node: Tensor of integer or boolean datatype providing data.
    :param name: The optional new name for output node.
    :return: The node performing bitwise NOT operation on the given tensor.
    """
    return _get_node_factory_opset13().create(
        "BitwiseNot",
        [node],
    )


@binary_op
def bitwise_or(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise OR operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_or.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise OR operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseOr",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def bitwise_xor(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise XOR operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_xor.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise XOR operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseXor",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def fake_convert(
    data: NodeInput,
    scale: NodeInput,
    shift: Optional[NodeInput] = None,
    destination_type: Literal["f8e4m3", "f8e5m2"] = "f8e4m3",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs FakeConvert.

    FakeConvert is experimental and may change in the future.
    .. warning:: FakeConvert is experimental and may change in the future.

    :param data: The node with data tensor with FP16, BF16 or FP32 datatype.
    :param scale: Tensor with a scale factor for the data input value,
                  of the same type as the data, and shape Numpy-broadcastable to data.
    :param shift: Optional tensor with value to subtract before and add after conversion of the data input value,
                  of the same type as the data, and shape Numpy-broadcastable to data.
    :param destination_type: Type to emulate, string of either "f8e4m3" or "f8e5m2".
    :param name: The optional new name for output node.

    :return: The new node performing FakeConvert operation.
    """
    nodes = [data, scale]
    if shift is not None:
        nodes.append(shift)
    return _get_node_factory_opset13().create(
        "FakeConvert",
        as_nodes(*nodes, name=name),
        {"destination_type": destination_type},
    )


@nameable_op
def multinomial(
    probs: NodeInput,
    num_samples: NodeInput,
    convert_type: str,
    with_replacement: bool,
    log_probs: bool,
    global_seed: int = 0,
    op_seed: int = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates a sequence of class indices sampled from the multinomial distribution.

    :param probs: Tensor with probabilities of floating-point type, and shape [batch_size, class_size].
    :param num_samples: Tensor (scalar or 1D) a single element of type i32 or i64,
                        specifying the number of samples to draw from the multinomial distribution.
    :param convert_type: Specifies the output tensor type, possible values: 'i64', 'i32'.
    :param with_replacement: Flag that specifies whether to sample with replacement.
    :param log_probs: Flag that specifies whether *probs* should be treated as unnormalized log probabilities.
    :param global_seed: Specifies global seed value. Required to be a positive integer or 0.
    :param op_seed: Specifies operational seed value. Required to be a positive integer or 0.
    :param name: The optional new name for output node.

    :return: The new node performing Multinomial operation.
    """
    inputs = as_nodes(probs, num_samples, name=name)

    if global_seed < 0:
        raise RuntimeError(f"global_seed should be positive or 0. Got: {global_seed}")

    if op_seed < 0:
        raise RuntimeError(f"op_seed should be positive or 0. Got: {op_seed}")

    attributes = {
        "convert_type": convert_type,
        "with_replacement": with_replacement,
        "log_probs": log_probs,
        "global_seed": global_seed,
        "op_seed": op_seed,
    }
    return _get_node_factory_opset13().create("Multinomial", inputs, attributes)


@nameable_op
def nms_rotated(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: NodeInput,
    iou_threshold: NodeInput,
    score_threshold: NodeInput,
    sort_result_descending: bool = True,
    output_type: str = "i64",
    clockwise: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NMSRotated.

    :param boxes: Tensor with box coordinates of floating point type and shape [num_batches, num_boxes, 5],
                  where the last dimension is defined as [x_ctr, y_ctr, width, height, angle_radians].
    :param scores: Tensor with box scores of floating point type and shape [num_batches, num_classes, num_boxes].
    :param max_output_boxes_per_class: Tensor (scalar or 1D) of integer type, specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor (scalar or 1D) of floating point type, specifying intersection over union threshold
    :param score_threshold: Tensor (scalar or 1D) of floating point type, specifying minimum score to consider box for the processing.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :param clockwise: Flag that specifies direction of the box rotation.
    :return: The new node which performs NMSRotated
    """
    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, name=name)

    attributes = {
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
        "clockwise": clockwise,
    }

    return _get_node_factory_opset13().create("NMSRotated", inputs, attributes)


@nameable_op
def scaled_dot_product_attention(
    query: NodeInput,
    key: NodeInput,
    value: NodeInput,
    attention_mask: Optional[NodeInput] = None,
    scale: Optional[NodeInput] = None,
    causal: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which implements Scaled Dot Product Attention.

    :param query: Query tensor of shape [N, ..., L, E] and floating-point datatype.
    :param key: Key tensor of shape [N, ..., S, E] and floating-point datatype.
    :param value: Value tensor of shape [N, ..., S, Ev] and floating-point datatype.
    :param attention_mask: Optional attention mask tensor of shape [N, ..., L, S] or scalar float type zero value.
                           Refer to the operation specification for a complete description.
    :param scale: Optional alternative scale, a floating-point type scalar.
    :param causal: If true, then autogenerates causal attention mask instead of using attention_mask input.
                   In this case attention_mask input is ignored.
    :param name: The optional new name for output node.

    :return: The new node performing Scaled Dot Product Attention operation.
    """
    inputs = as_nodes(query, key, value, name=name)
    if attention_mask is not None:
        inputs.append(as_node(attention_mask, name=name))
    elif scale is not None:
        inputs.append(as_node(convert_like(constant(np.array(0, np.int32)), inputs[0]), name=name))
    if scale is not None:
        inputs.append(as_node(scale, name=name))

    attributes = {
        "causal": causal,
    }
    return _get_node_factory_opset13().create("ScaledDotProductAttention", inputs, attributes)


@overloading(Union[NumericData, np.number, bool, np.bool_, list], Union[NumericType, Type], Optional[str], bool)  # type: ignore
@nameable_op
def constant(
    value: Union[NumericData, np.number, bool, np.bool_, list],
    dtype: Union[NumericType, Type] = None,
    name: Optional[str] = None,
    *,
    shared_memory: bool = False,
) -> Constant:
    """Create a Constant node from provided value.

    :param value: One of: array of values or scalar to initialize node with.
    :param dtype: The data type of provided data.
                  If dtype does not match, data will be converted.
                  Note: disables sharing of the memory when convertion occurs.
    :param name: Optional name for output node.
    :param shared_memory: keyword-only argument.
                          If `True`, this Constant's memory is being shared with a host,
                          that means the responsibility of keeping host memory is
                          on the side of a user. Any action performed on the host
                          memory is reflected on this Constant's memory!
                          If `False`, data is being copied to this Constant.
                          Requires data to be C_CONTIGUOUS if `True`.
                          Disabled by default if:
                          - value is a scalar.
                          - dtype is one of: Type.u1, Type.i4, Type.u4, Type.nf4, Type.bf16.
                          - dtype force conversion of data.
    :return: The Constant node initialized with provided data.
    """

    def display_shared_memory_warning(warning_message: str) -> None:
        if shared_memory:
            log.warning(f"{warning_message}. Memory sharing is disabled by default. Set shared_memory=False to hide this warning.")

    if isinstance(value, np.ndarray):
        _value, _shared_memory = value, shared_memory
    else:
        _value, _shared_memory = np.array(value), False
        display_shared_memory_warning(f"Converting scalar to corresponding type of {_value.dtype}")
    # Handle type casting, when dtype is not None:
    if dtype:
        # Expect packed data, use different constructor to handle it correctly:
        if dtype in [Type.u1, Type.i4, Type.u4, Type.nf4, Type.f4e2m1]:
            display_shared_memory_warning(f"Constant initialized with packed type of {dtype}")
            return Constant(dtype, Shape(_value.shape), _value.flatten().tolist())
        elif dtype in [Type.bf16, Type.f8e8m0, Type.f8e4m3, Type.f8e5m2]:
            display_shared_memory_warning(f"Constant initialized with OpenVINO custom {dtype}")
            return Constant(dtype, Shape(_value.shape), _value.flatten().tolist())
        # General use-case for all other types:
        else:
            _dtype = dtype.to_dtype() if isinstance(dtype, Type) else dtype
            if _dtype is int:
                display_shared_memory_warning("Converting scalar type of undefined bitwidth to 32-bit integer")
                _value, _shared_memory = _value.astype(np.int32), False
            elif _dtype is float:
                display_shared_memory_warning("Converting scalar type of undefined bitwidth to 32-bit float")
                _value, _shared_memory = _value.astype(np.float32), False
            elif _dtype is bool:
                display_shared_memory_warning("Converting bool type to numpy bool")
                _value, _shared_memory = _value.astype(np.bool_), False
            else:
                if _dtype != _value.dtype:
                    display_shared_memory_warning(f"Converting value of {_value.dtype} to {_dtype}")
                    _value, _shared_memory = _value.astype(_dtype), False
    # Create Constant itself:
    return Constant(_value, shared_memory=_shared_memory)


@overloading(Tensor, bool, Optional[str])  # type: ignore
@nameable_op
def constant(  # noqa: F811
    tensor: Tensor,
    shared_memory: bool = True,
    name: Optional[str] = None,
) -> Constant:
    return Constant(tensor, shared_memory=shared_memory)


@unary_op
def result(data: Union[Node, Output, NumericData], name: Optional[str] = None) -> Node:
    """Return a node which represents an output of a graph (Model).

    :param data: The tensor containing the input data
    :return: Result node
    """
    if isinstance(data, Node):
        return Result(data.output(0))
    return Result(data)


@nameable_op
def fake_quantize(
    data: NodeInput,
    input_low: NodeInput,
    input_high: NodeInput,
    output_low: NodeInput,
    output_high: NodeInput,
    levels: int,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    r"""Perform an element-wise linear quantization on input data.

    :param data:           The node with data tensor.
    :param input_low:      The node with the minimum for input values.
    :param input_high:     The node with the maximum for input values.
    :param output_low:     The node with the minimum quantized value.
    :param output_high:    The node with the maximum quantized value.
    :param levels:         The number of quantization levels. Integer value.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name:           Optional name of the new node.
    :return: New node with quantized value.

    Input floating point values are quantized into a discrete set of floating point values.

    .. code-block:: python

        if x <= input_low:
            output = output_low
        if x > input_high:
            output = output_high
        else:
            output = fake_quantize(output)

    Fake quantize uses the following logic:

    \f[ output =
            \dfrac{round( \dfrac{data - input\_low}{(input\_high - input\_low)\cdot (levels-1)})}
            {(levels-1)\cdot (output\_high - output\_low)} + output\_low \f]
    """
    return _get_node_factory_opset13().create(
        "FakeQuantize",
        as_nodes(data, input_low, input_high, output_low, output_high, name=name),
        {"levels": levels, "auto_broadcast": auto_broadcast.upper()},
    )
