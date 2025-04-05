# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import nameable_op
from openvino.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
)

_get_node_factory_opset12 = partial(_get_node_factory, "opset12")


# -------------------------------------------- ops ------------------------------------------------

@nameable_op
def pad(
    arg: NodeInput,
    pads_begin: NodeInput,
    pads_end: NodeInput,
    pad_mode: str,
    arg_pad_value: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a generic padding operation.

    :param arg: The node producing input tensor to be padded.
    :param pads_begin: Number of padding elements to be added before position 0
                       on each axis of arg. Negative values are supported.
    :param pads_end: Number of padding elements to be added after the last element.
                     Negative values are supported.
    :param pad_mode: "constant", "edge", "reflect" or "symmetric"
    :param arg_pad_value: value used for padding if pad_mode is "constant"
    :return: Pad operation node.
    """
    input_nodes = as_nodes(arg, pads_begin, pads_end, name=name)
    if arg_pad_value:
        input_nodes.append(as_node(arg_pad_value, name=name))

    pad_mode = pad_mode.upper()
    return _get_node_factory_opset12().create("Pad", input_nodes, {"pad_mode": pad_mode})


@nameable_op
def scatter_elements_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    axis: NodeInput,
    reduction: Optional[str] = "None",
    use_init_val: Optional[bool] = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a ScatterElementsUpdate operation.

    :param data:    The input tensor to be updated.
    :param indices: The tensor with indexes which will be updated. Negative indices are supported.
    :param updates: The tensor with update values.
    :param axis:    The axis for scatter.
    :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                      "prod", "min", "max", "mean".
    :param: use_init_val: Controls whether the elements in the data input tensor are used as
                          initial value for reduce operations.
    :return: ScatterElementsUpdate node

    ScatterElementsUpdate creates a copy of the first input tensor with updated elements
    specified with second and third input tensors.

    For each entry in `updates`, the target index in `data` is obtained by combining
    the corresponding entry in `indices` with the index of the entry itself: the
    index-value for dimension equal to `axis` is obtained from the value of the
    corresponding entry in `indices` and the index-value for dimension not equal
    to `axis` is obtained from the index of the entry itself.
    """
    input_nodes = as_nodes(data, indices, updates, axis, name=name)
    return _get_node_factory_opset12().create(
        "ScatterElementsUpdate",
        input_nodes,
        {
            "reduction": reduction,
            "use_init_val": use_init_val,
        },
    )


@nameable_op
def group_normalization(
    data: NodeInput,
    scale: NodeInput,
    bias: NodeInput,
    num_groups: int,
    epsilon: float,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a GroupNormalization operation.

    :param data:    The input tensor to be normalized.
    :param scale:   The tensor containing the scale values for each channel.
    :param bias:    The tensor containing the bias values for each channel.
    :param num_groups: Specifies the number of groups that the channel dimension will be divided into.
    :param epsilon: A very small value added to the variance for numerical stability.
                    Ensures that division by zero does not occur for any normalized element.
    :return: GroupNormalization node
    """
    input_nodes = as_nodes(data, scale, bias, name=name)
    return _get_node_factory_opset12().create(
        "GroupNormalization",
        input_nodes,
        {
            "num_groups": num_groups,
            "epsilon": epsilon,
        },
    )
