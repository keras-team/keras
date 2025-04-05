# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset14."""
from functools import partial

from typing import Union, Optional, List

from openvino import Node, Type
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import TensorShape
from openvino.utils.decorators import nameable_op
from openvino.utils.types import NodeInput, as_node, as_nodes

_get_node_factory_opset14 = partial(_get_node_factory, "opset14")


# -------------------------------------------- ops ------------------------------------------------

@nameable_op
def convert_promote_types(
    left_node: NodeInput,
    right_node: NodeInput,
    promote_unsafe: bool = False,
    pytorch_scalar_promotion: bool = False,
    u64_integer_promotion_target: Union[str, Type] = "f32",
    name: Optional[str] = None,
) -> Node:
    """Return a node performing conversion to common type based on promotion rules.

    :param left_node: input node with type to be promoted to common one.
    :param right_node: input node with type to be promoted to common one.
    :param promote_unsafe: Bool attribute whether to allow promotions that might result in bit-widening, precision loss and undefined behaviors.
    :param pytorch_scalar_promotion: Bool attribute whether to promote scalar input to type provided by non-scalar input when number format is matching.
    :param u64_integer_promotion_target: Element type attribute to select promotion result when inputs are u64 and signed integer.
    :param name: Optional name for the new output node.

    :return: The new node performing ConvertPromoteTypes operation.
    """
    inputs = as_nodes(left_node, right_node, name=name)

    attributes = {
        "promote_unsafe": promote_unsafe,
        "pytorch_scalar_promotion": pytorch_scalar_promotion,
        "u64_integer_promotion_target": u64_integer_promotion_target,
    }
    return _get_node_factory_opset14().create("ConvertPromoteTypes", inputs, attributes)


@nameable_op
def inverse(
    data: NodeInput,
    adjoint: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node with inverse matrices of the input.

    :param data: Tensor with matrices to invert. Last two dimensions must be of the same size.
    :param adjoint: Whether to return adjoint instead of inverse matrices. Defaults to false.
    :param name: Optional name for the new output node.

    :return: The new node performing Inverse operation.
    """
    inputs = as_nodes(data, name=name)

    attributes = {
        "adjoint": adjoint,
    }

    return _get_node_factory_opset14().create("Inverse", inputs, attributes)


@nameable_op
def max_pool(
    data: NodeInput,
    strides: List[int],
    dilations: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    kernel_shape: TensorShape,
    rounding_type: str = "floor",
    auto_pad: Optional[str] = None,
    index_element_type: Optional[Union[str, Type]] = "i64",
    axis: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Perform max pooling operation and return both values and indices of the selected elements.

    :param  data:                The node providing input data.
    :param  strides:             The distance (in pixels) to slide the filter on the feature map
                                 over the axes.
    :param  dilations:           The dilation of filter elements(distance between elements).
    :param  pads_begin:          The number of pixels to add at the beginning along each axis.
    :param  pads_end:            The number of pixels to add at the end along each axis.
    :param  kernel_shape:        The pooling operation kernel shape.
    :param  rounding_type:       Determines used rounding schema when computing output shape.
                                 Acceptable values are: ['floor', 'ceil', 'ceil_torch']. Defaults to 'floor'.
    :param  auto_pad:            Determines how the padding is calculated. Acceptable values:
                                 [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
    :param  index_element_type:  The data type used for the indices output of this operator.
                                 Defaults to i64.
    :param  axis:                The first dimension in the data shape used to determine the maximum
                                 returned index value. The value is the product of all dimensions
                                 starting at the provided axis. Defaults to 0.
    :param  name:                The optional name for the created output node.

    :return:   The new node performing max pooling operation.
    """
    if auto_pad is None:
        auto_pad = "explicit"
    return _get_node_factory_opset14().create(
        "MaxPool",
        [as_node(data, name=name)],
        {
            "strides": strides,
            "dilations": dilations,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "kernel": kernel_shape,
            "rounding_type": rounding_type.upper(),
            "auto_pad": auto_pad.upper(),
            "index_element_type": index_element_type,
            "axis": axis,
        },
    )


@nameable_op
def avg_pool(
    data_batch: NodeInput,
    strides: List[int],
    pads_begin: TensorShape,
    pads_end: TensorShape,
    kernel_shape: TensorShape,
    exclude_pad: bool,
    rounding_type: str = "floor",
    auto_pad: Optional[str] = None,
    name: Optional[str] = None,
) -> Node:
    """Return average pooling node.

    :param data_batch:      The input node providing data.
    :param strides:         The window movement strides.
    :param pads_begin:      The number of pixels to add at the beginning along each axis.
    :param pads_end:        The number of pixels to add at the end along each axis.
    :param kernel_shape:    The pooling window shape.
    :param exclude_pad:     Whether or not to include zero padding in average computations.
    :param rounding_type:   Determines used rounding schema when computing output shape. Acceptable
                            values are: ['floor', 'ceil', 'ceil_torch']. Defaults to 'floor'.
    :param auto_pad:        Determines how the padding is calculated. Acceptable values:
                            [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
    :param name:            Optional name for the new output node.

    :return: New node with AvgPool operation applied on its data.
    """
    if auto_pad is None:
        auto_pad = "explicit"
    return _get_node_factory_opset14().create(
        "AvgPool",
        [as_node(data_batch, name=name)],
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "kernel": kernel_shape,
            "exclude-pad": exclude_pad,
            "rounding_type": rounding_type.upper(),
            "auto_pad": auto_pad.upper(),
        },
    )
