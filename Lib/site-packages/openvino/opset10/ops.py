# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from functools import partial
from typing import List, Optional

from openvino import Node
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import nameable_op
from openvino.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
    make_constant_node,
)

_get_node_factory_opset4 = partial(_get_node_factory, "opset4")
_get_node_factory_opset10 = partial(_get_node_factory, "opset10")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def interpolate(
    image: NodeInput,
    output_shape: NodeInput,
    scales: NodeInput,
    mode: str,
    shape_calculation_mode: str,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    coordinate_transformation_mode: str = "half_pixel",
    nearest_mode: str = "round_prefer_floor",
    antialias: bool = False,
    cube_coeff: float = -0.75,
    axes: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform interpolation of independent slices in input tensor.

    :param  image:         The node providing input tensor with data for interpolation.
    :param  output_shape:  1D tensor describing output shape for spatial axes.
    :param  scales:        1D tensor describing scales for spatial axes.
    :param  mode:          Specifies type of interpolation. Possible values are: nearest, linear,
                           linear_onnx, cubic.
    :param  shape_calculation_mode:
                           Specifies which input, sizes or scales, is used to calculate an output
                           shape.
    :param  pads_begin:    Specifies the number of pixels to add to the beginning of the image
                           being interpolated. Default is None.
    :param  pads_end:      Specifies the number of pixels to add to the end of the image being
                           interpolated. Default is None.
    :param  coordinate_transformation_mode:
                           Specifies how to transform the coordinate in the resized tensor to the
                           coordinate in the original tensor. Default is "half_pixel".
    :param  nearest_mode:  Specifies round mode when mode == nearest and is used only when
                           mode == nearest. Default is "round_prefer_floor".
    :param  antialias:     Specifies whether to perform anti-aliasing. Default is False.
    :param  cube_coeff:    Specifies the parameter a for cubic interpolation. Default is -0.75.
    :param  axes:          1D tensor specifying dimension indices where interpolation is applied.
                           Default is None.
    :param  name:          Optional name for the output node. Default is None.
    :return: Node representing interpolation operation.
    """
    attrs = {
        "mode": mode,
        "shape_calculation_mode": shape_calculation_mode,
        "coordinate_transformation_mode": coordinate_transformation_mode,
        "nearest_mode": nearest_mode,
        "antialias": antialias,
        "cube_coeff": cube_coeff,
    }

    attrs["pads_begin"] = [] if pads_begin is None else pads_begin
    attrs["pads_end"] = [] if pads_end is None else pads_end

    inputs = as_nodes(image, output_shape, scales, name=name) if axes is None else as_nodes(image, output_shape, scales, axes, name=name)

    # This is an update of the operator version, so even though this is opset 10,
    # the operator is taken from opset 4.
    return _get_node_factory_opset4().create("Interpolate", inputs, attrs)


@nameable_op
def is_finite(data: NodeInput, name: Optional[str] = None) -> Node:
    """Performs element-wise mapping from NaN and Infinity to False. Other values are mapped to True.

    :param  data:          A tensor of floating-point numeric type and arbitrary shape.
    :param  name:          Optional name for the output node. The default is None.
    :return: Node representing is_finite operation.
    """
    return _get_node_factory_opset10().create("IsFinite", as_nodes(data, name=name))


@nameable_op
def is_inf(
    data: NodeInput,
    attributes: Optional[dict] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs IsInf operation.

    :param data: The input tensor.
    :param attributes: Optional dictionary containing IsInf attributes.
    :param name: Optional name of the node.

    Available attributes:

    * detect_negative   Specifies whether to map negative infinities to true in output map.
                        Range of values: true, false
                        Default value: true
                        Required: no
    * detect_positive   Specifies whether to map positive infinities to true in output map.
                        Range of values: true, false
                        Default value: true
                        Required: no

    :return: A new IsInf node.
    """
    if not attributes:
        attributes = {}
    return _get_node_factory_opset10().create("IsInf", as_nodes(data, name=name), attributes)


@nameable_op
def is_nan(data: NodeInput, name: Optional[str] = None) -> Node:
    """Performs element-wise mapping from NaN to True. Other values are mapped to False.

    :param  data:          A tensor of floating point numeric type and arbitrary shape.
    :param  name:          Optional name for the output node. Default is None.
    :return: Node representing is_nan operation.
    """
    return _get_node_factory_opset10().create("IsNaN", as_nodes(data, name=name))


@nameable_op
def unique(
    data: NodeInput,
    axis: Optional[NodeInput] = None,
    sorted: Optional[bool] = True,
    index_element_type: Optional[str] = "i64",
    count_element_type: Optional[str] = "i64",
    name: Optional[str] = None,
) -> Node:
    """Operator which selects and returns unique elements or unique slices of the input tensor.

    :param  data:               Input data tensor.
    :param  axis:               (Optional) An input tensor containing the axis value.
                                If not provided or None, data input is considered as a flattened tensor.
                                Default value: None.
    :param  sorted:             (Optional) Controls the order of the returned unique values,
                                sorts ascendingly when true.
                                Default value: True.
    :param  index_element_type: (Optional) The data type set for outputs containing indices.
                                Default value: "i64".
    :param  count_element_type: (Optional) The data type set for the output with repetition count.
                                Default value: "i64".
    :param name:                (Optional) A name for the output node. Default value: None.
    :return: Node representing Unique operation.
    """
    if axis is None:
        inputs = as_nodes(data, name=name)
    else:
        inputs = as_nodes(data, axis, name=name)

    attributes = {
        "sorted": sorted,
        "index_element_type": index_element_type,
        "count_element_type": count_element_type,
    }
    return _get_node_factory_opset10().create("Unique", inputs, attributes)
