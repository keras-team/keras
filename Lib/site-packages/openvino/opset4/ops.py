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

_get_node_factory_opset4 = partial(_get_node_factory, "opset4")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def ctc_loss(
    logits: NodeInput,
    logit_length: NodeInput,
    labels: NodeInput,
    label_length: NodeInput,
    blank_index: Optional[NodeInput] = None,
    preprocess_collapse_repeated: bool = False,
    ctc_merge_repeated: bool = True,
    unique: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs CTCLoss.

    :param logits:                        3-D tensor of logits.
    :param logit_length:                  1-D tensor of lengths for each object from a batch.
    :param labels:                        2-D tensor of labels for which likelihood is estimated using logits.
    :param label_length:                  1-D tensor of length for each label sequence.
    :param blank_index:                   Scalar used to mark a blank index.
    :param preprocess_collapse_repeated:  Flag for preprocessing labels before loss calculation.
    :param ctc_merge_repeated:            Flag for merging repeated characters in a potential alignment.
    :param unique:                        Flag to find unique elements in a target.
    :return: The new node which performs CTCLoss
    """
    if blank_index is not None:
        inputs = as_nodes(logits, logit_length, labels, label_length, blank_index, name=name)
    else:
        inputs = as_nodes(logits, logit_length, labels, label_length, name=name)

    attributes = {
        "preprocess_collapse_repeated": preprocess_collapse_repeated,
        "ctc_merge_repeated": ctc_merge_repeated,
        "unique": unique,
    }

    return _get_node_factory_opset4().create("CTCLoss", inputs, attributes)


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

    return _get_node_factory_opset4().create("NonMaxSuppression", inputs, attributes)


@nameable_op
def softplus(data: NodeInput, name: Optional[str] = None) -> Node:
    """Apply SoftPlus operation on each element of input tensor.

    :param data: The tensor providing input data.
    :return: The new node with SoftPlus operation applied on each element.
    """
    return _get_node_factory_opset4().create("SoftPlus", as_nodes(data, name=name), {})


@nameable_op
def mish(data: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which performs Mish.

    :param data: Tensor with input data floating point type.
    :return: The new node which performs Mish
    """
    return _get_node_factory_opset4().create("Mish", as_nodes(data, name=name), {})


@nameable_op
def hswish(data: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which performs HSwish (hard version of Swish).

    :param data: Tensor with input data floating point type.
    :return: The new node which performs HSwish
    """
    return _get_node_factory_opset4().create("HSwish", as_nodes(data, name=name), {})


@nameable_op
def swish(
    data: NodeInput,
    beta: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performing Swish activation function Swish(x, beta=1.0) = x * sigmoid(x * beta)).

    :param data: Tensor with input data floating point type.
    :return: The new node which performs Swish
    """
    if beta is None:
        beta = make_constant_node(1.0, np.float32)
    return _get_node_factory_opset4().create("Swish", as_nodes(data, beta, name=name), {})


@nameable_op
def acosh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply hyperbolic inverse cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arccosh operation applied on it.
    """
    return _get_node_factory_opset4().create("Acosh", as_nodes(node, name=name))


@nameable_op
def asinh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply hyperbolic inverse sinus function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arcsinh operation applied on it.
    """
    return _get_node_factory_opset4().create("Asinh", as_nodes(node, name=name))


@nameable_op
def atanh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply hyperbolic inverse tangent function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arctanh operation applied on it.
    """
    return _get_node_factory_opset4().create("Atanh", as_nodes(node, name=name))


@nameable_op
def proposal(
    class_probs: Node,
    bbox_deltas: Node,
    image_shape: NodeInput,
    attrs: dict,
    name: Optional[str] = None,
) -> Node:
    """Filter bounding boxes and outputs only those with the highest prediction confidence.

    :param  class_probs:        4D input floating point tensor with class prediction scores.
    :param  bbox_deltas:        4D input floating point tensor with corrected predictions of bounding boxes
    :param  image_shape:        The 1D input tensor with 3 or 4 elements describing image shape.
    :param  attrs:              The dictionary containing key, value pairs for attributes.
    :param  name:               Optional name for the output node.

    * base_size     The size of the anchor to which scale and ratio attributes are applied.
                    Range of values: a positive unsigned integer number
                    Default value: None
                    Required: yes
    * pre_nms_topn  The number of bounding boxes before the NMS operation.
                    Range of values: a positive unsigned integer number
                    Default value: None
                    Required: yes
    * post_nms_topn The number of bounding boxes after the NMS operation.
                    Range of values: a positive unsigned integer number
                    Default value: None
                    Required: yes
    * nms_thresh    The minimum value of the proposal to be taken into consideration.
                    Range of values: a positive floating-point number
                    Default value: None
                    Required: yes
    * feat_stride   The step size to slide over boxes (in pixels).
                    Range of values: a positive unsigned integer
                    Default value: None
                    Required: yes
    * min_size      The minimum size of box to be taken into consideration.
                    Range of values: a positive unsigned integer number
                    Default value: None
                    Required: yes
    * ratio         The ratios for anchor generation.
                    Range of values: a list of floating-point numbers
                    Default value: None
                    Required: yes
    * scale         The scales for anchor generation.
                    Range of values: a list of floating-point numbers
                    Default value: None
                    Required: yes
    * clip_before_nms   The flag that specifies whether to perform clip bounding boxes before
                        non-maximum suppression or not.
                        Range of values: True or False
                        Default value: True
                        Required: no
    * clip_after_nms    The flag that specifies whether to perform clip bounding boxes after
                        non-maximum suppression or not.
                        Range of values: True or False
                        Default value: False
                        Required: no
    * normalize     The flag that specifies whether to perform normalization of output boxes to
                    [0,1] interval or not.
                    Range of values: True or False
                    Default value: False
                    Required: no
    * box_size_scale    Specifies the scale factor applied to logits of box sizes before decoding.
                        Range of values: a positive floating-point number
                        Default value: 1.0
                        Required: no
    * box_coordinate_scale  Specifies the scale factor applied to logits of box coordinates
                            before decoding.
                            Range of values: a positive floating-point number
                            Default value: 1.0
                            Required: no
    * framework     Specifies how the box coordinates are calculated.
                    Range of values: "" (empty string) - calculate box coordinates like in Caffe*
                                     tensorflow - calculate box coordinates like in the TensorFlow*
                                                  Object Detection API models

                    Default value: "" (empty string)
                    Required: no

    Example of attribute dictionary:

    .. code-block:: python

        # just required ones
        attrs = {
            'base_size': 85,
            'pre_nms_topn': 10,
            'post_nms_topn': 20,
            'nms_thresh': 0.34,
            'feat_stride': 16,
            'min_size': 32,
            'ratio': [0.1, 1.5, 2.0, 2.5],
            'scale': [2, 3, 3, 4],
        }

    Optional attributes which are absent from dictionary will be set with corresponding default.
    :return: Node representing Proposal operation.
    """
    requirements = [
        ("base_size", True, np.unsignedinteger, is_positive_value),
        ("pre_nms_topn", True, np.unsignedinteger, is_positive_value),
        ("post_nms_topn", True, np.unsignedinteger, is_positive_value),
        ("nms_thresh", True, np.floating, is_positive_value),
        ("feat_stride", True, np.unsignedinteger, is_positive_value),
        ("min_size", True, np.unsignedinteger, is_positive_value),
        ("ratio", True, np.floating, None),
        ("scale", True, np.floating, None),
        ("clip_before_nms", False, np.bool_, None),
        ("clip_after_nms", False, np.bool_, None),
        ("normalize", False, np.bool_, None),
        ("box_size_scale", False, np.floating, is_positive_value),
        ("box_coordinate_scale", False, np.floating, is_positive_value),
        ("framework", False, np.str_, None),
    ]

    check_valid_attributes("Proposal", attrs, requirements)

    return _get_node_factory_opset4().create(
        "Proposal",
        [class_probs, bbox_deltas, as_node(image_shape, name=name)],
        attrs,
    )


@nameable_op
def reduce_l1(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """L1-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to mean-reduce.
    :param reduction_axes: The axes to eliminate through mean operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction
    :param name:           Optional name for output node.
    :return: The new node performing mean-reduction operation.
    """
    return _get_node_factory_opset4().create(
        "ReduceL1",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_l2(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """L2-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to mean-reduce.
    :param reduction_axes: The axes to eliminate through mean operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction
    :param name:           Optional name for output node.
    :return: The new node performing mean-reduction operation.
    """
    return _get_node_factory_opset4().create(
        "ReduceL2",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def lstm_cell(
    X: NodeInput,
    initial_hidden_state: NodeInput,
    initial_cell_state: NodeInput,
    W: NodeInput,
    R: NodeInput,
    B: NodeInput,
    hidden_size: int,
    activations: Optional[List[str]] = None,
    activations_alpha: Optional[List[float]] = None,
    activations_beta: Optional[List[float]] = None,
    clip: float = 0.0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs LSTMCell operation.

    :param X: The input tensor with shape: [batch_size, input_size].
    :param initial_hidden_state: The hidden state tensor with shape: [batch_size, hidden_size].
    :param initial_cell_state: The cell state tensor with shape: [batch_size, hidden_size].
    :param W: The weight tensor with shape: [4*hidden_size, input_size].
    :param R: The recurrence weight tensor with shape: [4*hidden_size, hidden_size].
    :param B: The bias tensor for gates with shape: [4*hidden_size].
    :param hidden_size: Specifies hidden state size.
    :param activations: The list of three activation functions for gates.
    :param activations_alpha: The list of alpha parameters for activation functions.
    :param activations_beta: The list of beta parameters for activation functions.
    :param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
    :param name: An optional name of the output node.

    :return: The new node represents LSTMCell. Node outputs count: 2.
    """
    if activations is None:
        activations = ["sigmoid", "tanh", "tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    node_inputs = as_nodes(X, initial_hidden_state, initial_cell_state, W, R, B, name=name)

    attributes = {
        "hidden_size": hidden_size,
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "clip": clip,
    }
    return _get_node_factory_opset4().create("LSTMCell", node_inputs, attributes)


@nameable_op
def range(
    start: Node,
    stop: NodeInput,
    step: NodeInput,
    output_type: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces the Range operation.

    :param start:       The start value of the generated range.
    :param stop:        The stop value of the generated range.
    :param step:        The step value for the generated range.
    :param output_type: The output tensor type.
    :param name:        Optional name for output node.
    :return: Range node
    """
    return _get_node_factory_opset4().create(
        "Range",
        as_nodes(start, stop, step, name=name),
        {
            "output_type": output_type,
        },
    )


@nameable_op
def scatter_nd_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ScatterNDUpdate.

    :param data: Node input representing the tensor to be updated.
    :param indices: Node input representing the indices at which updates will be applied.
    :param updates: Node input representing the updates to be applied.
    :param name: Optional name for the output node.
    :return: New node performing the ScatterNDUpdate.
    """
    inputs = as_nodes(data, indices, updates, name=name)

    return _get_node_factory_opset4().create("ScatterNDUpdate", inputs, {})
