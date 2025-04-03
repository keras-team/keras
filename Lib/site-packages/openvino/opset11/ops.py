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
)

_get_node_factory_opset11 = partial(_get_node_factory, "opset11")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def interpolate(
    image: NodeInput,
    scales_or_sizes: NodeInput,
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
    """Perfors the interpolation of the input tensor.

    :param  image:         The node providing input tensor with data for interpolation.
    :param  scales_or_sizes:
                           1D tensor providing information used to calculate the output shape
                           of the operation. It might contain floats (scales) or integers(sizes).
    :param  mode:          Specifies type of interpolation. Possible values are: nearest, linear,
                           linear_onnx, cubic, bilinear_pillow, bicubic_pillow.
    :param  shape_calculation_mode:
                           Specifies how the scales_or_sizes input should be interpreted.
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
                           The default is None.
    :param  name:          Optional name for the output node. The default is None.
    :return: Node representing the interpolation operation.
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

    inputs = as_nodes(image, scales_or_sizes, name=name) if axes is None else as_nodes(image, scales_or_sizes, axes, name=name)

    return _get_node_factory_opset11().create("Interpolate", inputs, attrs)


@nameable_op
def topk(
    data: NodeInput,
    k: NodeInput,
    axis: int,
    mode: str,
    sort: str,
    index_element_type: str = "i32",
    stable: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs TopK.

    :param data: Input data.
    :param k: K.
    :param axis: TopK Axis.
    :param mode: Compute TopK largest ('max') or smallest ('min')
    :param sort: Order of output elements (sort by: 'none', 'index' or 'value')
    :param index_element_type: Type of output tensor with indices.
    :param stable: Specifies whether the equivalent elements should maintain
                   their relative order from the input tensor during sorting.
    :return: The new node which performs TopK
    """
    return _get_node_factory_opset11().create(
        "TopK",
        as_nodes(data, k, name=name),
        {"axis": axis, "mode": mode, "sort": sort, "index_element_type": index_element_type, "stable": stable},
    )
