# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np
from functools import partial

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

_get_node_factory_opset3 = partial(_get_node_factory, "opset3")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def assign(new_value: NodeInput, variable_id: str, name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param new_value:    Node producing a value to be assigned to a variable.
    :param variable_id:  Id of a variable to be updated.
    :param name:         Optional name for output node.
    :return: Assign node
    """
    return _get_node_factory_opset3().create(
        "Assign",
        [as_node(new_value, name=name)],
        {"variable_id": variable_id},
    )


@nameable_op
def broadcast(
    data: NodeInput,
    target_shape: NodeInput,
    axes_mapping: Optional[NodeInput] = None,
    broadcast_spec: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Create a node which broadcasts the input node's values along specified axes to a desired shape.

    :param data: The node with input tensor data.
    :param target_shape: The node with a new shape we want to broadcast tensor to.
    :param axes_mapping: The node with a axis positions (0-based) in the result
                           that are being broadcast.
    :param broadcast_spec: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: NUMPY, EXPLICIT, BIDIRECTIONAL.
    :param name: Optional new name for output node.
    :return: New node with broadcast shape.
    """
    inputs = as_nodes(data, target_shape, name=name)
    if broadcast_spec.upper() == "EXPLICIT":
        inputs.append(as_node(axes_mapping, name=name))
    return _get_node_factory_opset3().create(
        "Broadcast",
        inputs,
        {"mode": broadcast_spec.upper()},
    )


@nameable_op
def bucketize(
    data: Node,
    buckets: NodeInput,
    output_type: str = "i64",
    with_right_bound: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces the Bucketize operation.

    :param data:              Input data to bucketize
    :param buckets:           1-D of sorted unique boundaries for buckets
    :param output_type:       Output tensor type, "i64" or "i32", defaults to i64
    :param with_right_bound:  indicates whether bucket includes the right or left
                              edge of interval. default true = includes right edge
    :param name:              Optional name for output node.
    :return: Bucketize node
    """
    return _get_node_factory_opset3().create(
        "Bucketize",
        [data, as_node(buckets, name=name)],
        {"output_type": output_type, "with_right_bound": with_right_bound},
    )


@nameable_op
def cum_sum(
    arg: NodeInput,
    axis: NodeInput,
    exclusive: bool = False,
    reverse: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Construct a cumulative summation operation.

    :param arg: The tensor to be summed.
    :param axis: zero dimension tensor specifying axis position along which sum will be performed.
    :param exclusive: if set to true, the top element is not included
    :param reverse: if set to true, will perform the sums in reverse direction
    :return: New node performing the operation
    """
    return _get_node_factory_opset3().create(
        "CumSum",
        as_nodes(arg, axis, name=name),
        {"exclusive": exclusive, "reverse": reverse},
    )


@nameable_op
def embedding_bag_offsets_sum(
    emb_table: Node,
    indices: NodeInput,
    offsets: NodeInput,
    default_index: Optional[NodeInput] = None,
    per_sample_weights: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs sums of bags of embeddings without the intermediate embeddings.

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: Tensor with indices.
    :param offsets: Tensor containing the starting index positions of each bag in indices.
    :param per_sample_weights: Tensor with weights for each sample.
    :param default_index: Scalar containing default index in embedding table to fill empty bags.
    :param name: Optional name for output node.
    :return: The new node which performs EmbeddingBagOffsetsSum
    """
    inputs = [emb_table, as_node(indices, name=name), as_node(offsets, name=name)]
    if per_sample_weights is not None:
        inputs.append(default_index)
        inputs.append(per_sample_weights)
    elif default_index is not None:
        inputs.append(default_index)

    return _get_node_factory_opset3().create("EmbeddingBagOffsetsSum", inputs, {})


@nameable_op
def embedding_bag_packed_sum(
    emb_table: NodeInput,
    indices: NodeInput,
    per_sample_weights: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return an EmbeddingBagPackedSum node.

    EmbeddingSegmentsSum constructs an output tensor by replacing every index in a given
    input tensor with a row (from the weights matrix) at that index

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: Tensor with indices.
    :param per_sample_weights: Weights to be multiplied with embedding table.
    :param name: Optional name for output node.
    :return: EmbeddingBagPackedSum node
    """
    inputs = [as_node(emb_table, name=name), as_node(indices, name=name)]
    if per_sample_weights is not None:
        inputs.append(as_node(per_sample_weights, name=name))

    return _get_node_factory_opset3().create("EmbeddingBagPackedSum", inputs, {})


@nameable_op
def embedding_segments_sum(
    emb_table: Node,
    indices: NodeInput,
    segment_ids: NodeInput,
    num_segments: Optional[NodeInput] = None,
    default_index: Optional[NodeInput] = None,
    per_sample_weights: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return an EmbeddingSegmentsSum node.

    EmbeddingSegmentsSum constructs an output tensor by replacing every index in a given
    input tensor with a row (from the weights matrix) at that index

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: Tensor with indices.
    :param segment_ids: Tensor with indices into the output Tensor
    :param num_segments: Tensor with number of segments.
    :param default_index: Scalar containing default index in embedding table to fill empty bags.
    :param per_sample_weights: Weights to be multiplied with embedding table.
    :param name: Optional name for output node.
    :return: EmbeddingSegmentsSum node
    """
    inputs = [as_node(emb_table, name=name), as_node(indices, name=name), as_node(segment_ids, name=name)]
    if per_sample_weights is not None:
        inputs.append(as_node(num_segments, name=name))
        inputs.append(as_node(default_index, name=name))
        inputs.append(as_node(per_sample_weights, name=name))
    elif default_index is not None:
        inputs.append(as_node(num_segments, name=name))
        inputs.append(as_node(default_index, name=name))
    elif num_segments is not None:
        inputs.append(as_node(num_segments, name=name))

    return _get_node_factory_opset3().create("EmbeddingSegmentsSum", inputs, {})


@nameable_op
def extract_image_patches(
    image: NodeInput,
    sizes: TensorShape,
    strides: List[int],
    rates: TensorShape,
    auto_pad: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces the ExtractImagePatches operation.

    :param image:     4-D Input data to extract image patches.
    :param sizes:     Patch size in the format of [size_rows, size_cols].
    :param strides:   Patch movement stride in the format of [stride_rows, stride_cols]
    :param rates:     Element seleciton rate for creating a patch.
    :param auto_pad:  Padding type.
    :param name:      Optional name for output node.
    :return: ExtractImagePatches node
    """
    return _get_node_factory_opset3().create(
        "ExtractImagePatches",
        [as_node(image, name=name)],
        {"sizes": sizes, "strides": strides, "rates": rates, "auto_pad": auto_pad},
    )


@nameable_op
def gru_cell(
    X: NodeInput,
    initial_hidden_state: NodeInput,
    W: NodeInput,
    R: NodeInput,
    B: NodeInput,
    hidden_size: int,
    activations: Optional[List[str]] = None,
    activations_alpha: Optional[List[float]] = None,
    activations_beta: Optional[List[float]] = None,
    clip: float = 0.0,
    linear_before_reset: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Perform GRUCell operation on the tensor from input node.

    GRUCell represents a single GRU Cell that computes the output
    using the formula described in the paper: https://arxiv.org/abs/1406.1078

    Note this class represents only single *cell* and not whole *layer*.

    :param X:                       The input tensor with shape: [batch_size, input_size].
    :param initial_hidden_state:    The hidden state tensor at current time step with shape:
                                    [batch_size, hidden_size].
    :param W:                       The weights for matrix multiplication, gate order: zrh.
                                    Shape: [3*hidden_size, input_size].
    :param R:                       The recurrence weights for matrix multiplication.
                                    Shape: [3*hidden_size, hidden_size].
    :param B:                       The sum of biases (weight and recurrence).
                                    For linear_before_reset set True the shape is [4*hidden_size].
                                    Otherwise the shape is [3*hidden_size].
    :param hidden_size:             The number of hidden units for recurrent cell.
                                    Specifies hidden state size.
    :param activations:             The vector of activation functions used inside recurrent cell.
    :param activation_alpha:        The vector of alpha parameters for activation functions in
                                    order respective to activation list.
    :param activation_beta:         The vector of beta parameters for activation functions in order
                                    respective to activation list.
    :param clip:                    The value defining clipping range [-clip, clip] on input of
                                    activation functions.
    :param linear_before_reset:     Flag denotes if the layer behaves according to the modification
                                    of GRUCell described in the formula in the ONNX documentation.
    :param name:                    Optional output node name.
    :return:   The new node performing a GRUCell operation on tensor from input node.
    """
    if activations is None:
        activations = ["sigmoid", "tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    input_nodes = as_nodes(X, initial_hidden_state, W, R, B, name=name)
    attributes = {
        "hidden_size": hidden_size,
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "linear_before_reset": linear_before_reset,
        "clip": clip,
    }
    return _get_node_factory_opset3().create("GRUCell", input_nodes, attributes)


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    box_encoding: str = "corner",
    sort_result_descending: bool = True,
    output_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NonMaxSuppression.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor specifying intersection over union threshold
    :param score_threshold: Tensor specifying minimum score to consider box for the processing.
    :param box_encoding: Format of boxes data encoding.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :return: The new node which performs NonMaxSuppression
    """
    if max_output_boxes_per_class is None:
        max_output_boxes_per_class = make_constant_node(0, np.int64)
    if iou_threshold is None:
        iou_threshold = make_constant_node(0, np.float32)
    if score_threshold is None:
        score_threshold = make_constant_node(0, np.float32)

    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, name=name)
    attributes = {
        "box_encoding": box_encoding,
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
    }

    return _get_node_factory_opset3().create("NonMaxSuppression", inputs, attributes)


@nameable_op
def non_zero(data: NodeInput, output_type: str = "i64", name: Optional[str] = None) -> Node:
    """Return the indices of the elements that are non-zero.

    :param data: Input data.
    :param output_type: Output tensor type.

    :return: The new node which performs NonZero
    """
    return _get_node_factory_opset3().create(
        "NonZero",
        [as_node(data, name=name)],
        {"output_type": output_type},
    )


@nameable_op
def read_value(init_value: NodeInput, variable_id: str, name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param init_value:   Node producing a value to be returned instead of an unassigned variable.
    :param variable_id:  Id of a variable to be read.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    return _get_node_factory_opset3().create(
        "ReadValue",
        [as_node(init_value, name=name)],
        {"variable_id": variable_id},
    )


@nameable_op
def rnn_cell(
    X: NodeInput,
    initial_hidden_state: NodeInput,
    W: NodeInput,
    R: NodeInput,
    B: NodeInput,
    hidden_size: int,
    activations: List[str],
    activations_alpha: List[float],
    activations_beta: List[float],
    clip: float = 0.0,
    name: Optional[str] = None,
) -> Node:
    """Perform RNNCell operation on tensor from input node.

    It follows notation and equations defined as in ONNX standard:
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN

    Note this class represents only single *cell* and not whole RNN *layer*.

    :param X:                       The input tensor with shape: [batch_size, input_size].
    :param initial_hidden_state:    The hidden state tensor at current time step with shape:
                                    [batch_size, hidden_size].
    :param W:                       The weight tensor with shape: [hidden_size, input_size].
    :param R:                       The recurrence weight tensor with shape: [hidden_size,
                                    hidden_size].
    :param B:                       The sum of biases (weight and recurrence) with shape: [hidden_size].
    :param hidden_size:             The number of hidden units for recurrent cell.
                                    Specifies hidden state size.
    :param activations:             The vector of activation functions used inside recurrent cell.
    :param activation_alpha:        The vector of alpha parameters for activation functions in
                                    order respective to activation list.
    :param activation_beta:         The vector of beta parameters for activation functions in order
                                    respective to activation list.
    :param clip:                    The value defining clipping range [-clip, clip] on input of
                                    activation functions.
    :param name:                    Optional output node name.
    :return:   The new node performing a RNNCell operation on tensor from input node.
    """
    if activations is None:
        activations = ["tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    input_nodes = as_nodes(X, initial_hidden_state, W, R, B, name=name)
    attributes = {
        "hidden_size": hidden_size,
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "clip": clip,
    }
    return _get_node_factory_opset3().create("RNNCell", input_nodes, attributes)


@nameable_op
def roi_align(
    data: NodeInput,
    rois: NodeInput,
    batch_indices: NodeInput,
    pooled_h: int,
    pooled_w: int,
    sampling_ratio: int,
    spatial_scale: float,
    mode: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ROIAlign.

    :param data: Input data.
    :param rois: RoIs (Regions of Interest) to pool over.
    :param batch_indices: Tensor with each element denoting the index of
                          the corresponding image in the batch.
    :param pooled_h: Height of the ROI output feature map.
    :param pooled_w: Width of the ROI output feature map.
    :param sampling_ratio: Number of bins over height and width to use to calculate
                           each output feature map element.
    :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
    :param mode: Method to perform pooling to produce output feature map elements.

    :return: The new node which performs ROIAlign
    """
    inputs = as_nodes(data, rois, batch_indices, name=name)
    attributes = {
        "pooled_h": pooled_h,
        "pooled_w": pooled_w,
        "sampling_ratio": sampling_ratio,
        "spatial_scale": spatial_scale,
        "mode": mode,
    }
    return _get_node_factory_opset3().create("ROIAlign", inputs, attributes)


@nameable_op
def scatter_elements_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    axis: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a ScatterElementsUpdate operation.

    :param data:    The input tensor to be updated.
    :param indices: The tensor with indexes which will be updated.
    :param updates: The tensor with update values.
    :param axis:    The axis for scatter.
    :return: ScatterElementsUpdate node

    ScatterElementsUpdate creates a copy of the first input tensor with updated elements
    specified with second and third input tensors.

    For each entry in `updates`, the target index in `data` is obtained by combining
    the corresponding entry in `indices` with the index of the entry itself: the
    index-value for dimension equal to `axis` is obtained from the value of the
    corresponding entry in `indices` and the index-value for dimension not equal
    to `axis` is obtained from the index of the entry itself.

    """
    return _get_node_factory_opset3().create(
        "ScatterElementsUpdate",
        as_nodes(data, indices, updates, axis, name=name),
    )


@nameable_op
def scatter_update(
    data: Node,
    indices: NodeInput,
    updates: NodeInput,
    axis: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a ScatterUpdate operation.

    ScatterUpdate sets new values to slices from data addressed by indices.

    :param data:    The input tensor to be updated.
    :param indices: The tensor with indexes which will be updated.
    :param updates: The tensor with update values.
    :param axis:    The axis at which elements will be updated.
    :return: ScatterUpdate node
    """
    return _get_node_factory_opset3().create(
        "ScatterUpdate",
        as_nodes(data, indices, updates, axis, name=name),
    )


@nameable_op
def shape_of(data: NodeInput, output_type: str = "i64", name: Optional[str] = None) -> Node:
    """Return a node which produces a tensor containing the shape of its input data.

    :param data: The tensor containing the input data.
    :param output_type: Output element type.
    :return: ShapeOf node
    """
    return _get_node_factory_opset3().create(
        "ShapeOf",
        [as_node(data, name=name)],
        {"output_type": output_type},
    )


@nameable_op
def shuffle_channels(data: Node, axis: int, group: int, name: Optional[str] = None) -> Node:
    """Perform permutation on data in the channel dimension of the input tensor.

    :param data: The node with input tensor.
    :param axis: Channel dimension index in the data tensor.
                 A negative value means that the index should be calculated
                 from the back of the input data shape.
    :param group: The channel dimension specified by the axis parameter
                 should be split into this number of groups.
    :param name: Optional output node name.
    :return: The new node performing a permutation on data in the channel dimension
             of the input tensor.

    The operation is the equivalent with the following transformation of the input tensor
    `data` of shape [N, C, H, W]:

    `data_reshaped` = reshape(`data`, [N, group, C / group, H * W])

    `data_transposed` = transpose(`data_reshaped`, [0, 2, 1, 3])

    `output` = reshape(`data_transposed`, [N, C, H, W])

    For example:

    .. code-block:: python

        Inputs: tensor of shape [1, 6, 2, 2]

                data = [[[[ 0.,  1.], [ 2.,  3.]],
                         [[ 4.,  5.], [ 6.,  7.]],
                         [[ 8.,  9.], [10., 11.]],
                         [[12., 13.], [14., 15.]],
                         [[16., 17.], [18., 19.]],
                         [[20., 21.], [22., 23.]]]]

                axis = 1
                groups = 3

        Output: tensor of shape [1, 6, 2, 2]

                output = [[[[ 0.,  1.], [ 2.,  3.]],
                           [[ 8.,  9.], [10., 11.]],
                           [[16., 17.], [18., 19.]],
                           [[ 4.,  5.], [ 6.,  7.]],
                           [[12., 13.], [14., 15.]],
                           [[20., 21.], [22., 23.]]]]
    """
    return _get_node_factory_opset3().create(
        "ShuffleChannels",
        [as_node(data, name=name)],
        {"axis": axis, "group": group},
    )


@nameable_op
def topk(
    data: NodeInput,
    k: NodeInput,
    axis: int,
    mode: str,
    sort: str,
    index_element_type: str = "i32",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs TopK.

    :param data: Input data.
    :param k: K.
    :param axis: TopK Axis.
    :param mode: Compute TopK largest ('max') or smallest ('min')
    :param sort: Order of output elements (sort by: 'none', 'index' or 'value')
    :param index_element_type: Type of output tensor with indices.
    :return: The new node which performs TopK (both indices and values)
    """
    return _get_node_factory_opset3().create(
        "TopK",
        as_nodes(data, k, name=name),
        {"axis": axis, "mode": mode, "sort": sort, "index_element_type": index_element_type},
    )
