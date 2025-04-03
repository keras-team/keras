# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset15."""
from functools import partial
from typing import List, Literal, Optional

import numpy as np
from openvino import Node, Type
from openvino.opset1 import convert_like
from openvino.opset14 import constant
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import binary_op, nameable_op
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset15 = partial(_get_node_factory, "opset15")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def scatter_nd_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    reduction: Optional[Literal["none", "sum", "sub", "prod", "min", "max"]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ScatterNDUpdate.

    :param data: Node input representing the tensor to be updated.
    :param indices: Node input representing the indices at which updates will be applied.
    :param updates: Node input representing the updates to be applied.
    :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                      "sub", "prod", "min", "max".
    :param name: Optional name for the output node.
    :return: New node performing the ScatterNDUpdate.
    """
    inputs = as_nodes(data, indices, updates, name=name)
    attributes = {}
    if reduction:
        attributes["reduction"] = reduction
    return _get_node_factory_opset15().create("ScatterNDUpdate", inputs, attributes)


@nameable_op
def col2im(
    data: NodeInput,
    output_size: NodeInput,
    kernel_size: NodeInput,
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform data movement operation which combines sliding blocks into an image tensor.

    :param  data:                The node providing input data.
    :param  output_size:         Shape of the spatial dimensions of the output image.
    :param  kernel_size:         Size of the sliding blocks.
    :param  strides:             Stride on the sliding blocks in the input spatial dimensions. Defaults to [1, 1].
    :param  dilations:           The dilation of filter elements (distance between elements). Defaults to [1, 1].
    :param  pads_begin:          The number of pixels added at the beginning along each axis. Defaults to [0, 0].
    :param  pads_end:            The number of pixels added at the end along each axis. Defaults to [0, 0].
    :param  name:                The optional name for the created output node.

    :return:   The new node performing Col2Im operation.
    """
    if strides is None:
        strides = [1, 1]
    if dilations is None:
        dilations = [1, 1]
    if pads_begin is None:
        pads_begin = [0, 0]
    if pads_end is None:
        pads_end = [0, 0]
    return _get_node_factory_opset15().create(
        "Col2Im",
        as_nodes(data, output_size, kernel_size, name=name),
        {
            "strides": strides,
            "dilations": dilations,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
        },
    )


@nameable_op
def embedding_bag_offsets(
    emb_table: NodeInput,
    indices: NodeInput,
    offsets: NodeInput,
    default_index: Optional[NodeInput] = None,
    per_sample_weights: Optional[NodeInput] = None,
    reduction: Literal["sum", "mean"] = "sum",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs sums or means of bags of embeddings without the intermediate embeddings.

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: 1D Tensor with indices.
    :param offsets: 1D Tensor containing the starting index positions of each bag in indices.
    :param per_sample_weights: Tensor with weights for each sample.
    :param default_index: Scalar containing default index in embedding table to fill empty bags.
                          If unset or set to -1, empty bags will be filled with 0.
                          Reverse indexing using negative indices is not supported.
    :param reduction: String to select algorithm used to perform reduction of elements in bag.
    :param name: Optional name for output node.
    :return: The new node performing EmbeddingBagOffsets operation.
    """
    inputs = [emb_table, indices, offsets]
    if default_index is not None:
        inputs.append(default_index)
    elif per_sample_weights is not None:
        inputs.append(convert_like(constant(np.array(-1, np.int32)), inputs[1]))
    if per_sample_weights is not None:
        inputs.append(per_sample_weights)

    return _get_node_factory_opset15().create("EmbeddingBagOffsets", as_nodes(*inputs, name=name), {"reduction": reduction})


@nameable_op
def embedding_bag_packed(
    emb_table: NodeInput,
    indices: NodeInput,
    per_sample_weights: Optional[NodeInput] = None,
    reduction: Literal["sum", "mean"] = "sum",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs sums or means of "bags" of embeddings, without the intermediate embeddings.

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: 2D Tensor of shape [batch, indices_per_bag] with indices.
    :param per_sample_weights: Tensor of weights to be multiplied with embedding table with same shape as indices.
    :param reduction: Operator to perform reduction of elements in bag.
    :param name: Optional name for output node.
    :return: The new node performing EmbeddingBagPacked operation.
    """
    inputs = [emb_table, indices]
    if per_sample_weights is not None:
        inputs.append(per_sample_weights)

    return _get_node_factory_opset15().create("EmbeddingBagPacked", as_nodes(*inputs, name=name), {"reduction": reduction})


@nameable_op
def roi_align_rotated(
    data: NodeInput,
    rois: NodeInput,
    batch_indices: NodeInput,
    pooled_h: int,
    pooled_w: int,
    sampling_ratio: int,
    spatial_scale: float,
    clockwise_mode: bool,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ROIAlignRotated operation.

    :param data: Input data.
    :param rois: RoIs (Regions of Interest) to pool over.
    :param batch_indices: Tensor with each element denoting the index of
                          the corresponding image in the batch.
    :param pooled_h: Height of the ROI output feature map.
    :param pooled_w: Width of the ROI output feature map.
    :param sampling_ratio: Number of bins over height and width to use to calculate
                           each output feature map element.
    :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
    :param clockwise_mode:  If true, rotation angle is interpreted as clockwise,
                            otherwise as counterclockwise
    :param name: The optional name for the output node

    :return: The new node which performs ROIAlignRotated
    """
    return _get_node_factory_opset15().create(
        "ROIAlignRotated",
        as_nodes(data, rois, batch_indices, name=name),
        {
            "pooled_h": pooled_h,
            "pooled_w": pooled_w,
            "sampling_ratio": sampling_ratio,
            "spatial_scale": spatial_scale,
            "clockwise_mode": clockwise_mode,
        },
    )


@nameable_op
def string_tensor_unpack(
    data: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform an operation which unpacks a batch of strings into three tensors.

    :param data: The node providing input data.

    :return: The new node performing StringTensorUnpack operation.
    """
    return _get_node_factory_opset15().create(
        "StringTensorUnpack",
        as_nodes(data, name=name)
    )


@nameable_op
def string_tensor_pack(
    begins: NodeInput,
    ends: NodeInput,
    symbols: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform an operation which packs a concatenated batch of strings into a batched string tensor.

    :param begins: ND tensor of non-negative integer numbers containing indices of each string's beginnings.
    :param ends: ND tensor of non-negative integer numbers containing indices of each string's endings.
    :param symbols: 1D tensor of concatenated strings data encoded in utf-8 bytes.

    :return: The new node performing StringTensorPack operation.
    """
    return _get_node_factory_opset15().create(
        "StringTensorPack",
        as_nodes(begins, ends, symbols, name=name)
    )


@binary_op
def bitwise_left_shift(
    arg0: NodeInput,
    arg1: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs BitwiseLeftShift operation on input nodes element-wise.

    :param arg0: Node with data to be shifted.
    :param arg1: Node with number of shifts.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors.
                           Defaults to “NUMPY”.

    :return: The new node performing BitwiseLeftShift operation.
    """
    return _get_node_factory_opset15().create(
        "BitwiseLeftShift",
        as_nodes(arg0, arg1, name=name),
        {
            "auto_broadcast": auto_broadcast.upper(),
        },
    )


@binary_op
def bitwise_right_shift(
    arg0: NodeInput,
    arg1: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs BitwiseRightShift operation on input nodes element-wise.

    :param arg0: Tensor with data to be shifted.
    :param arg1: Tensor with number of shifts.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors.
                           Defaults to “NUMPY”.

    :return: The new node performing BitwiseRightShift operation.
    """
    return _get_node_factory_opset15().create(
        "BitwiseRightShift",
        as_nodes(arg0, arg1, name=name),
        {
            "auto_broadcast": auto_broadcast.upper(),
        },
    )


@nameable_op
def slice_scatter(
    data: NodeInput,
    updates: NodeInput,
    start: NodeInput,
    stop: NodeInput,
    step: NodeInput,
    axes: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates SliceScatter operation.

    :param  data: The node providing input data.
    :param  updates: The node providing updates data.
    :param  start: The node providing start indices (inclusively).
    :param  stop: The node providing stop indices (exclusively).
    :param  step: The node providing step values.
    :param  axes: The optional node providing axes to slice, default [0, 1, ..., len(start)-1].
    :param  name: The optional name for the created output node.
    :return: The new node performing SliceScatter operation.
    """
    if axes is None:
        inputs = as_nodes(data, updates, start, stop, step, name=name)
    else:
        inputs = as_nodes(data, updates, start, stop, step, axes, name=name)

    return _get_node_factory_opset15().create("SliceScatter", inputs)


@nameable_op
def stft(
    data: NodeInput,
    window: NodeInput,
    frame_size: NodeInput,
    frame_step: NodeInput,
    transpose_frames: bool,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates STFT operation.

    :param  data: The node providing input data.
    :param  window: The node providing window data.
    :param  frame_size: The node with scalar value representing the size of Fourier Transform.
    :param  frame_step: The distance (number of samples) between successive window frames.
    :param  transpose_frames: Flag to set output shape layout. If true the `frames` dimension is at out_shape[2],
                              otherwise it is at out_shape[1].
    :param  name: The optional name for the created output node.
    :return: The new node performing STFT operation.
    """
    inputs = as_nodes(data, window, frame_size, frame_step, name=name)
    return _get_node_factory_opset15().create("STFT", inputs, {"transpose_frames": transpose_frames})


@nameable_op
def search_sorted(
    sorted_sequence: NodeInput,
    values: NodeInput,
    right_mode: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates SearchSorted operation.

    :param sorted_sequence: The node providing sorted sequence to search in.
    :param values: The node providing searched values.
    :param right_mode: If set to False, return the first suitable index that is found for given value.
                       If set to True, return the last such index. Defaults to False.
    :param name: The optional name for the created output node.
    :return: The new node performing SearchSorted operation.
    """
    inputs = as_nodes(sorted_sequence, values, name=name)
    attributes = {"right_mode": right_mode}
    return _get_node_factory_opset15().create("SearchSorted", inputs, attributes)


@nameable_op
def squeeze(
    data: NodeInput,
    axes: Optional[NodeInput] = None,
    allow_axis_skip: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Perform squeeze operation on input tensor.

    :param data: The node with data tensor.
    :param axes: Optional list of integers, indicating the dimensions to squeeze.
                  Negative indices are supported. One of: input node or array.
    :param allow_axis_skip: If true, shape inference results in a dynamic rank, when
                  selected axis has value 1 in its dynamic range. Used only if axes input
                  is given. Defaults to false.
    :param name: Optional new name for output node.
    :return: The new node performing a squeeze operation on input tensor.

    Remove single-dimensional entries from the shape of a tensor.
    Takes an optional parameter `axes` with a list of axes to squeeze.
    If `axes` is not provided, all the single dimensions will be removed from the shape.

    For example:

       Inputs: tensor with shape [1, 2, 1, 3, 1, 1], axes=[2, 4]

       Result: tensor with shape [1, 2, 3, 1]
    """
    if axes is None:
        inputs = as_nodes(data, name=name)
    else:
        inputs = as_nodes(data, axes, name=name)
    return _get_node_factory_opset15().create(
        "Squeeze",
        inputs,
        {"allow_axis_skip": allow_axis_skip}
    )
