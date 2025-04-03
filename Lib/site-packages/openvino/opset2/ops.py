# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np
from functools import partial
import warnings

from openvino import Node, Shape
from openvino.op import Constant, Parameter
from openvino.utils.decorators import binary_op, nameable_op, unary_op
from openvino.utils.input_validation import (
    assert_list_of_ints,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from openvino.utils.node_factory import NodeFactory, _get_node_factory
from openvino.utils.types import (
    NodeInput,
    NumericData,
    NumericType,
    ScalarData,
    TensorShape,
    as_node,
    as_nodes,
    get_dtype,
    get_element_type,
    get_element_type_str,
    make_constant_node,
)

_get_node_factory_opset2 = partial(_get_node_factory, "opset2")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def batch_to_space(
    data: NodeInput,
    block_shape: NodeInput,
    crops_begin: NodeInput,
    crops_end: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform BatchToSpace operation on the input tensor.

    BatchToSpace permutes data from the batch dimension of the data tensor into spatial dimensions.

    :param data: Node producing the data tensor.
    :param block_shape: The sizes of the block of values to be moved.
    :param crops_begin: Specifies the amount to crop from the beginning along each axis of `data`.
    :param crops_end: Specifies the amount to crop from the end along each axis of `data`.
    :param name: Optional output node name.
    :return: The new node performing a BatchToSpace operation.
    """
    return _get_node_factory_opset2().create(
        "BatchToSpace",
        as_nodes(data, block_shape, crops_begin, crops_end, name=name),
    )


@unary_op
def gelu(node: NodeInput, name: Optional[str] = None) -> Node:
    r"""Perform Gaussian Error Linear Unit operation element-wise on data from input node.

    Computes GELU function:

    \f[ f(x) = 0.5\cdot x\cdot(1 + erf( \dfrac{x}{\sqrt{2}}) \f]

    For more information refer to [Gaussian Error Linear Unit (GELU)](https://arxiv.org/pdf/1606.08415.pdf>)

    :param node: Input tensor. One of: input node, array or scalar.
    :param name: Optional output node name.
    :return: The new node performing a GELU operation on its input data element-wise.
    """
    return _get_node_factory_opset2().create("Gelu", [node])


@nameable_op
def mvn(
    data: Node,
    across_channels: bool = False,
    normalize_variance: bool = False,
    eps: float = 1e-9,
    name: Optional[str] = None,
) -> Node:
    r"""Perform Mean Variance Normalization operation on data from input node.

    Computes MVN on the input tensor `data` (called `X`) using formula:

    \f[ Y = \dfrac{X-EX}{\sqrt{E(X-EX)^2}} \f]

    :param data: The node with data tensor.
    :param across_channels: Denotes if mean values are shared across channels.
    :param normalize_variance: Denotes whether to perform variance normalization.
    :param eps: The number added to the variance to avoid division by zero
                when normalizing the value. Scalar value.
    :param name: Optional output node name.
    :return: The new node performing a MVN operation on input tensor.
    """
    return _get_node_factory_opset2().create(
        "MVN",
        [data],
        {
            "across_channels": across_channels,
            "normalize_variance": normalize_variance,
            "eps": eps,
        },
    )


@nameable_op
def reorg_yolo(input: Node, stride: List[int], name: Optional[str] = None) -> Node:
    """Return a node which produces the ReorgYolo operation.

    :param input:   Input data.
    :param stride:  Stride to reorganize input by.
    :param name:    Optional name for output node.
    :return: ReorgYolo node.
    """
    return _get_node_factory_opset2().create("ReorgYolo", [input], {"stride": stride})


@nameable_op
def roi_pooling(
    input: NodeInput,
    coords: NodeInput,
    output_roi: Optional[TensorShape] = None,
    spatial_scale: Optional[NumericData] = None,
    method: str = "max",
    name: Optional[str] = None,
    *,
    output_size: Optional[TensorShape] = None,
) -> Node:
    """Return a node which produces an ROIPooling operation.

    :param input:          Input feature map `{N, C, ...}`.
    :param coords:         Coordinates of bounding boxes.
    :param output_roi:     Height/Width of ROI output features (shape).
    :param spatial_scale:  Ratio of input feature map over input image size (float).
    :param method:         Method of pooling - string: "max" or "bilinear". Default: "max"
    :param output_size:    (DEPRECATED!) Height/Width of ROI output features (shape).
                           Will override `output_roi` if used and change behavior of the operator.
    :return:               ROIPooling node.
    """
    # Allow either one of these attributes to be passed.
    if output_roi is None and output_size is None:
        raise AttributeError("One of the following arguments must be defined: `output_roi`, `output_size`!")
    # Force checking of spatial_scale.
    if spatial_scale is None:
        raise AttributeError("The following arguments must be defined: `spatial_scale`!")

    def _deprecated_output_size_arg(output_roi: Optional[TensorShape], output_size: Optional[TensorShape]) -> Optional[TensorShape]:
        if output_size is not None:
            warnings.warn(
                "`output_size` is deprecated and will be removed in future. "
                "Value of `output_size` is going to override `output_roi` value and "
                "`get_output_size` will behave like `get_output_roi` function."
                "Please use only `output_roi` explicitly.",
                DeprecationWarning,
                stacklevel=3,
            )
            return output_size
        return output_roi

    method = method.lower()
    roi_shape = _deprecated_output_size_arg(output_roi, output_size)
    node = _get_node_factory_opset2().create(
        "ROIPooling",
        as_nodes(input, coords, name=name),
        {
            "output_size": Shape(roi_shape),
            "output_roi": Shape(roi_shape),
            "spatial_scale": spatial_scale,
            "method": method,
        },
    )

    # Override behavior when deprecated value was used.
    if output_size is not None:
        node.get_output_size = node.get_output_roi
        node.set_output_size = node.set_output_roi

    return node


@nameable_op
def space_to_batch(
    data: NodeInput,
    block_shape: NodeInput,
    pads_begin: NodeInput,
    pads_end: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform SpaceToBatch operation on the input tensor.

    SpaceToBatch permutes data tensor blocks of spatial data into batch dimension.
    The operator returns a copy of the input tensor where values from spatial blocks dimensions
    are moved in the batch dimension

    :param data: Node producing the data tensor.
    :param block_shape: The sizes of the block of values to be moved.
    :param pads_begin: Specifies the padding for the beginning along each axis of `data`.
    :param pads_end: Specifies the padding for the ending along each axis of `data`.
    :param name: Optional output node name.
    :return: The new node performing a SpaceToBatch operation.
    """
    return _get_node_factory_opset2().create(
        "SpaceToBatch",
        as_nodes(data, block_shape, pads_begin, pads_end, name=name),
    )
