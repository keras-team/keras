# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
from openvino.exceptions import UserInputError
from openvino.op import Constant, Parameter, if_op
from openvino import Node
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import nameable_op
from openvino.utils.input_validation import (
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from openvino.utils.types import (
    NodeInput,
    TensorShape,
    as_node,
    as_nodes,
)

_get_node_factory_opset8 = partial(_get_node_factory, "opset8")


# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def deformable_convolution(
    data: NodeInput,
    offsets: NodeInput,
    filters: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    dilations: List[int],
    mask: Optional[NodeInput] = None,
    auto_pad: str = "EXPLICIT",
    group: int = 1,
    deformable_group: int = 1,
    bilinear_interpolation_pad: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs deformable convolution operation.

    :param data: The node providing data batch tensor.
    :param offsets: The node providing offset tensor.
    :param filters: The node providing filters tensor.
    :param strides: The distance (in pixels) to slide the filter on the feature map over the axes.
    :param pads_begin: The number of pixels to add to the beginning along each axis.
    :param pads_end: The number of pixels to add to the end along each axis.
    :param dilations: The distance in width and height between elements (weights) in the filter.
    :param mask: The node providing modulation scalar (mask) tensor.
    :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
    :param group: The number of groups which both output and input should be split into.
    :param deformable_group: The number of groups which deformable values and output should be split
                             into along the channel axis.
    :param bilinear_interpolation_pad: The flag that determines the mode of bilinear interpolation
                                               execution.
    :param name: The optional new name for output node.
    :return: New node performing deformable convolution operation.
    """
    if mask is None:
        inputs = as_nodes(data, offsets, filters, name=name)
    else:
        inputs = as_nodes(data, offsets, filters, mask, name=name)

    return _get_node_factory_opset8().create(
        "DeformableConvolution",
        inputs,
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad,
            "group": group,
            "deformable_group": deformable_group,
            "bilinear_interpolation_pad": bilinear_interpolation_pad,
        },
    )


@nameable_op
def adaptive_avg_pool(
    data: NodeInput,
    output_shape: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs AdaptiveAvgPool operation.

    :param data: The list of input nodes
    :param output_shape: the shape of spatial dimentions after operation
    :param name: Optional output node name.
    :return: The new node performing AdaptiveAvgPool operation on the data
    """
    inputs = as_nodes(data, output_shape, name=name)
    return _get_node_factory_opset8().create("AdaptiveAvgPool", inputs)


@nameable_op
def adaptive_max_pool(
    data: NodeInput,
    output_shape: NodeInput,
    index_element_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs AdaptiveMaxPool operation.

    :param data: The list of input nodes
    :param output_shape: the shape of spatial dimentions after operation
    :param index_element_type: Type of indices output.
    :param name: Optional output node name.
    :return: The new node performing AdaptiveMaxPool operation on the data
    """
    inputs = as_nodes(data, output_shape, name=name)

    attributes = {
        "index_element_type": index_element_type,
    }

    return _get_node_factory_opset8().create("AdaptiveMaxPool", inputs, attributes)


@nameable_op
def multiclass_nms(
    boxes: NodeInput,
    scores: NodeInput,
    sort_result_type: str = "none",
    sort_result_across_batch: bool = False,
    output_type: str = "i64",
    iou_threshold: float = 0.0,
    score_threshold: float = 0.0,
    nms_top_k: int = -1,
    keep_top_k: int = -1,
    background_class: int = -1,
    nms_eta: float = 1.0,
    normalized: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MulticlassNms.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param sort_result_type: Specifies order of output elements, possible values:
                             'class': sort selected boxes by class id (ascending)
                             'score': sort selected boxes by score (descending)
                             'none': do not guarantee the order.
    :param sort_result_across_batch: Specifies whenever it is necessary to sort selected boxes
                                     across batches or not
    :param output_type: Specifies the output tensor type, possible values:
                        'i64', 'i32'
    :param iou_threshold: Specifies intersection over union threshold
    :param score_threshold: Specifies minimum score to consider box for the processing
    :param nms_top_k: Specifies maximum number of boxes to be selected per class, -1 meaning
                      to keep all boxes
    :param keep_top_k: Specifies maximum number of boxes to be selected per batch element, -1
                       meaning to keep all boxes
    :param background_class: Specifies the background class id, -1 meaning to keep all classes
    :param nms_eta: Specifies eta parameter for adpative NMS, in close range [0, 1.0]
    :param normalized: Specifies whether boxes are normalized or not
    :param name: Optional output node name.
    :return: The new node which performs MuticlassNms
    """
    inputs = as_nodes(boxes, scores, name=name)

    attributes = {
        "sort_result_type": sort_result_type,
        "sort_result_across_batch": sort_result_across_batch,
        "output_type": output_type,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "nms_top_k": nms_top_k,
        "keep_top_k": keep_top_k,
        "background_class": background_class,
        "nms_eta": nms_eta,
        "normalized": normalized,
    }

    return _get_node_factory_opset8().create("MulticlassNms", inputs, attributes)


@nameable_op
def matrix_nms(
    boxes: NodeInput,
    scores: NodeInput,
    sort_result_type: str = "none",
    sort_result_across_batch: bool = False,
    output_type: str = "i64",
    score_threshold: float = 0.0,
    nms_top_k: int = -1,
    keep_top_k: int = -1,
    background_class: int = -1,
    decay_function: str = "linear",
    gaussian_sigma: float = 2.0,
    post_threshold: float = 0.0,
    normalized: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MatrixNms.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param sort_result_type: Specifies order of output elements, possible values:
                             'class': sort selected boxes by class id (ascending)
                             'score': sort selected boxes by score (descending)
                             'none': do not guarantee the order.
    :param sort_result_across_batch: Specifies whenever it is necessary to sort selected boxes
                                     across batches or not
    :param output_type: Specifies the output tensor type, possible values:
                        'i64', 'i32'
    :param score_threshold: Specifies minimum score to consider box for the processing
    :param nms_top_k: Specifies maximum number of boxes to be selected per class, -1 meaning
                      to keep all boxes
    :param keep_top_k: Specifies maximum number of boxes to be selected per batch element, -1
                       meaning to keep all boxes
    :param background_class: Specifies the background class id, -1 meaning to keep all classes
    :param decay_function: Specifies decay function used to decay scores, possible values:
                           'gaussian', 'linear'
    :param gaussian_sigma: Specifies gaussian_sigma parameter for gaussian decay_function
    :param post_threshold: Specifies threshold to filter out boxes with low confidence score
                           after decaying
    :param normalized: Specifies whether boxes are normalized or not
    :param name: Optional output node name.
    :return: The new node which performs MatrixNms
    """
    inputs = as_nodes(boxes, scores, name=name)

    attributes = {
        "sort_result_type": sort_result_type,
        "sort_result_across_batch": sort_result_across_batch,
        "output_type": output_type,
        "score_threshold": score_threshold,
        "nms_top_k": nms_top_k,
        "keep_top_k": keep_top_k,
        "background_class": background_class,
        "decay_function": decay_function,
        "gaussian_sigma": gaussian_sigma,
        "post_threshold": post_threshold,
        "normalized": normalized,
    }

    return _get_node_factory_opset8().create("MatrixNms", inputs, attributes)


@nameable_op
def gather(
    data: NodeInput,
    indices: NodeInput,
    axis: NodeInput,
    batch_dims: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs Gather with support of negative indices.

    :param data:         N-D tensor with data for gathering
    :param indices:      N-D tensor with indices by which data is gathered. Negative indices
                         indicate reverse indexing from the end
    :param axis:         axis along which elements are gathered
    :param batch_dims:   number of batch dimensions
    :param name:         Optional output node name.
    :return:             The new node which performs Gather
    """
    inputs = as_nodes(data, indices, axis, name=name)
    attributes = {
        "batch_dims": batch_dims,
    }
    return _get_node_factory_opset8().create("Gather", inputs, attributes)


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
    index_element_type: Optional[str] = "i64",
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
                                 Acceptable values are: ['floor', 'ceil']. Defaults to 'floor'.
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
    return _get_node_factory_opset8().create(
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
def random_uniform(
    output_shape: NodeInput,
    min_val: NodeInput,
    max_val: NodeInput,
    output_type: str,
    global_seed: int = 0,
    op_seed: int = 0,
    alignment: str = "tensorflow",
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates sequence of random values from uniform distribution.

    :param output_shape: Tensor with shape of the output tensor.
    :param min_val: Tensor with the lower bound on the range of random values to generate.
    :param max_val: Tensor with the upper bound on the range of random values to generate.
    :param output_type: Specifies the output tensor type, possible values:
                                'i64', 'i32', 'f64', 'f32', 'f16', 'bf16'.
    :param global_seed: Specifies global seed value. Required to be a positive integer or 0.
    :param op_seed: Specifies operational seed value. Required to be a positive integer or 0.
    :param alignment: Specifies alignment of the randomly generated numbers to a given framework.
                                Possible values: 'tensorflow', 'pytorch'. Default is 'tensorflow'.
    :param name: Optional output node name.

    :return: The new node which performs generation of random values from uniform distribution.
    """
    inputs = as_nodes(output_shape, min_val, max_val, name=name)

    if global_seed < 0:
        raise RuntimeError(f"global_seed should be positive or 0. Got: {global_seed}")

    if op_seed < 0:
        raise RuntimeError(f"op_seed should be positive or 0. Got: {op_seed}")

    attributes = {
        "output_type": output_type,
        "global_seed": global_seed,
        "op_seed": op_seed,
        "alignment": alignment.lower(),
    }
    return _get_node_factory_opset8().create("RandomUniform", inputs, attributes)


@nameable_op
def slice(
    data: NodeInput,
    start: NodeInput,
    stop: NodeInput,
    step: NodeInput,
    axes: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates Slice operation.

    :param  data: The node providing input data.
    :param  start: The node providing start indices (inclusively).
    :param  stop: The node providing stop indices (exclusively).
    :param  step: The node providing step values.
    :param  axes: The optional node providing axes to slice, default [0, 1, ..., len(start)-1].
    :param  name: The optional name for the created output node.
    :return: The new node performing Slice operation.
    """
    if axes is None:
        inputs = as_nodes(data, start, stop, step, name=name)
    else:
        inputs = as_nodes(data, start, stop, step, axes, name=name)

    return _get_node_factory_opset8().create("Slice", inputs)


@nameable_op
def gather_nd(
    data: NodeInput,
    indices: NodeInput,
    batch_dims: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherND.

    :param data:       N-D tensor with data for gathering
    :param indices:    K-D tensor of tuples with indices by which data is gathered
    :param batch_dims: Scalar value of batch dimensions
    :return: The new node which performs GatherND
    """
    inputs = as_nodes(data, indices, name=name)

    attributes = {
        "batch_dims": batch_dims,
    }

    return _get_node_factory_opset8().create("GatherND", inputs, attributes)


@nameable_op
def prior_box(
    layer_shape: Node,
    image_shape: NodeInput,
    attrs: dict,
    name: Optional[str] = None,
) -> Node:
    """Generate prior boxes of specified sizes and aspect ratios across all dimensions.

    :param  layer_shape:  Shape of layer for which prior boxes are computed.
    :param  image_shape:  Shape of image to which prior boxes are scaled.
    :param  attrs:        The dictionary containing key, value pairs for attributes.
    :param  name:         Optional name for the output node.
    :return: Node representing prior box operation.

    Available attributes are:
    * min_size                      The minimum box size (in pixels).
                                    Range of values: positive floating point numbers
                                    Default value: []
                                    Required: no

    * max_size                      The maximum box size (in pixels).
                                    Range of values: positive floating point numbers
                                    Default value: []
                                    Required: no
    * aspect_ratio                  Aspect ratios of prior boxes.
                                    Range of values: set of positive floating point numbers
                                    Default value: []
                                    Required: no
    * flip                          The flag that denotes that each aspect_ratio is duplicated and flipped.
                                    Range of values: {True, False}
                                    Default value: False
                                    Required: no
    * clip                          The flag that denotes if each value in the output tensor should be clipped
                                    to [0,1] interval.
                                    Range of values: {True, False}
                                    Default value: False
                                    Required: no
    * step                          The distance between box centers.
                                    Range of values: floating point non-negative number
                                    Default value: 0
                                    Required: no
    * offset                        This is a shift of box respectively to top left corner.
                                    Range of values: floating point non-negative number
                                    Default value: None
                                    Required: yes
    * variance                      The variance denotes a variance of adjusting bounding boxes. The attribute
                                    could contain 0, 1 or 4 elements.
                                    Range of values: floating point positive numbers
                                    Default value: []
                                    Required: no
    * scale_all_sizes               The flag that denotes type of inference.
                                    Range of values: False - max_size is ignored
                                                     True  - max_size is used

                                    Default value: True
                                    Required: no
    * fixed_ratio                   This is an aspect ratio of a box.
                                    Range of values: a list of positive floating-point numbers
                                    Default value: None
                                    Required: no
    * fixed_size                    This is an initial box size (in pixels).
                                    Range of values: a list of positive floating-point numbers
                                    Default value: None
                                    Required: no
    * density                       This is the square root of the number of boxes of each type.
                                    Range of values: a list of positive floating-point numbers
                                    Default value: None
                                    Required: no
    * min_max_aspect_ratios_order   The flag that denotes the order of output prior box.
                                    Range of values: False - the output prior box is in order of
                                                             [min, aspect_ratios, max]

                                                     True  - the output prior box is in order of
                                                             [min, max, aspect_ratios]

                                    Default value: True
                                    Required: no

    Example of attribute dictionary:

    .. code-block:: python

        # just required ones
        attrs = {
            'offset': 85,
        }
        attrs = {
            'offset': 85,
            'flip': True,
            'clip': True,
            'fixed_size': [32, 64, 128]
        }

    Optional attributes which are absent from dictionary will be set with corresponding default.
    """
    requirements = [
        ("offset", True, np.floating, is_non_negative_value),
        ("min_size", False, np.floating, is_positive_value),
        ("max_size", False, np.floating, is_positive_value),
        ("aspect_ratio", False, np.floating, is_positive_value),
        ("flip", False, np.bool_, None),
        ("clip", False, np.bool_, None),
        ("step", False, np.floating, is_non_negative_value),
        ("variance", False, np.floating, is_positive_value),
        ("scale_all_sizes", False, np.bool_, None),
        ("fixed_ratio", False, np.floating, is_positive_value),
        ("fixed_size", False, np.floating, is_positive_value),
        ("density", False, np.floating, is_positive_value),
        ("min_max_aspect_ratios_order", False, np.bool_, None),
    ]

    check_valid_attributes("PriorBox", attrs, requirements)

    return _get_node_factory_opset8().create("PriorBox", [layer_shape, as_node(image_shape, name=name)], attrs)


@nameable_op
def i420_to_bgr(
    arg: NodeInput,
    arg_u: Optional[NodeInput] = None,
    arg_v: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs I420toBGR operation.

    :param  arg: The node providing single or Y plane data.
    :param  arg_u: The node providing U plane data. Required for separate planes.
    :param  arg_v: The node providing V plane data. Required for separate planes.
    :param  name: The optional name for the created output node.
    :return: The new node performing I420toBGR operation.
    """
    if arg_u is None and arg_v is None:
        inputs = as_nodes(arg, name=name)
    elif arg_u is not None and arg_v is not None:
        inputs = as_nodes(arg, arg_u, arg_v, name=name)
    else:
        raise UserInputError(
            "Operation I420toBGR must have one (single plane) or three (separate planes) inputs provided.",
        )

    return _get_node_factory_opset8().create("I420toBGR", inputs)


@nameable_op
def i420_to_rgb(
    arg: NodeInput,
    arg_u: Optional[NodeInput] = None,
    arg_v: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs I420toRGB operation.

    :param  arg: The node providing single or Y plane data.
    :param  arg_u: The node providing U plane data. Required for separate planes.
    :param  arg_v: The node providing V plane data. Required for separate planes.
    :param  name: The optional name for the created output node.
    :return: The new node performing I420toRGB operation.
    """
    if arg_u is None and arg_v is None:
        inputs = as_nodes(arg, name=name)
    elif arg_u is not None and arg_v is not None:
        inputs = as_nodes(arg, arg_u, arg_v, name=name)
    else:
        raise UserInputError(
            "Operation I420toRGB must have one (single plane) or three (separate planes) inputs provided.",
        )

    return _get_node_factory_opset8().create("I420toRGB", inputs)


@nameable_op
def nv12_to_bgr(
    arg: NodeInput,
    arg_uv: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NV12toBGR operation.

    :param  arg: The node providing single or Y plane data.
    :param  arg_uv: The node providing UV plane data. Required for separate planes.
    :param  name: The optional name for the created output node.
    :return: The new node performing NV12toBGR operation.
    """
    if arg_uv is None:
        inputs = as_nodes(arg, name=name)
    else:
        inputs = as_nodes(arg, arg_uv, name=name)

    return _get_node_factory_opset8().create("NV12toBGR", inputs)


@nameable_op
def nv12_to_rgb(
    arg: NodeInput,
    arg_uv: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NV12toRGB operation.

    :param  arg: The node providing single or Y plane data.
    :param  arg_uv: The node providing UV plane data. Required for separate planes.
    :param  name: The optional name for the created output node.
    :return: The new node performing NV12toRGB operation.
    """
    if arg_uv is None:
        inputs = as_nodes(arg, name=name)
    else:
        inputs = as_nodes(arg, arg_uv, name=name)

    return _get_node_factory_opset8().create("NV12toRGB", inputs)


@nameable_op
def detection_output(
    box_logits: NodeInput,
    class_preds: NodeInput,
    proposals: NodeInput,
    attrs: dict,
    aux_class_preds: Optional[NodeInput] = None,
    aux_box_preds: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Generate the detection output using information on location and confidence predictions.

    :param  box_logits:         The 2D input tensor with box logits.
    :param  class_preds:        The 2D input tensor with class predictions.
    :param  proposals:          The 3D input tensor with proposals.
    :param  attrs:              The dictionary containing key, value pairs for attributes.
    :param  aux_class_preds:    The 2D input tensor with additional class predictions information.
    :param  aux_box_preds:      The 2D input tensor with additional box predictions information.
    :param  name:               Optional name for the output node.
    :return: Node representing DetectionOutput operation.
     Available attributes are:

    * background_label_id   The background label id.
                            Range of values: integer value
                            Default value: 0
                            Required: no
    * top_k                 Maximum number of results to be kept per batch after NMS step.
                            Range of values: integer value
                            Default value: -1
                            Required: no
    * variance_encoded_in_target    The flag that denotes if variance is encoded in target.
                                    Range of values: {False, True}
                                    Default value: False
                                    Required: no
    * keep_top_k            Maximum number of bounding boxes per batch to be kept after NMS step.
                            Range of values: integer values
                            Default value: None
                            Required: yes
    * code_type             The type of coding method for bounding boxes.
                            Range of values: {'caffe.PriorBoxParameter.CENTER_SIZE',
                                             'caffe.PriorBoxParameter.CORNER'}

                            Default value: 'caffe.PriorBoxParameter.CORNER'
                            Required: no
    * share_location        The flag that denotes if bounding boxes are shared among different
                            classes.
                            Range of values: {True, False}
                            Default value: True
                            Required: no
    * nms_threshold         The threshold to be used in the NMS stage.
                            Range of values: floating point value
                            Default value: None
                            Required: yes
    * confidence_threshold  Specifies the minimum confidence threshold for detection boxes to be
                            considered.
                            Range of values: floating point value
                            Default value: 0
                            Required: no
    * clip_after_nms        The flag that denotes whether to perform clip bounding boxes after
                            non-maximum suppression or not.
                            Range of values: {True, False}
                            Default value: False
                            Required: no
    * clip_before_nms       The flag that denotes whether to perform clip bounding boxes before
                            non-maximum suppression or not.
                            Range of values: {True, False}
                            Default value: False
                            Required: no
    * decrease_label_id     The flag that denotes how to perform NMS.
                            Range of values: False - perform NMS like in Caffe*.
                                             True  - perform NMS like in MxNet*.

                            Default value: False
                            Required: no
    * normalized            The flag that denotes whether input tensors with boxes are normalized.
                            Range of values: {True, False}
                            Default value: False
                            Required: no
    * input_height          The input image height.
                            Range of values: positive integer number
                            Default value: 1
                            Required: no
    * input_width           The input image width.
                            Range of values: positive integer number
                            Default value: 1
                            Required: no
    * objectness_score      The threshold to sort out confidence predictions.
                            Range of values: non-negative float number
                            Default value: 0
                            Required: no

    Example of attribute dictionary:

    .. code-block:: python

        # just required ones
        attrs = {
            'keep_top_k': [1, 2, 3],
            'nms_threshold': 0.645,
        }
        attrs = {
            'keep_top_k': [1, 2, 3],
            'nms_threshold': 0.645,
            'normalized': True,
            'clip_before_nms': True,
            'input_height': [32],
            'input_width': [32],
        }

    Optional attributes which are absent from dictionary will be set with corresponding default.
    """
    requirements = [
        ("background_label_id", False, np.integer, None),
        ("top_k", False, np.integer, None),
        ("variance_encoded_in_target", False, np.bool_, None),
        ("keep_top_k", True, np.integer, None),
        ("code_type", False, np.str_, None),
        ("share_location", False, np.bool_, None),
        ("nms_threshold", True, np.floating, None),
        ("confidence_threshold", False, np.floating, None),
        ("clip_after_nms", False, np.bool_, None),
        ("clip_before_nms", False, np.bool_, None),
        ("decrease_label_id", False, np.bool_, None),
        ("normalized", False, np.bool_, None),
        ("input_height", False, np.integer, is_positive_value),
        ("input_width", False, np.integer, is_positive_value),
        ("objectness_score", False, np.floating, is_non_negative_value),
    ]

    check_valid_attributes("DetectionOutput", attrs, requirements)

    inputs = [box_logits, class_preds, proposals]
    if aux_class_preds is not None:
        inputs.append(aux_class_preds)
    if aux_box_preds is not None:
        inputs.append(aux_box_preds)
    inputs = as_nodes(*inputs, name=name)

    return _get_node_factory_opset8().create("DetectionOutput", inputs, attrs)


@nameable_op
def softmax(data: NodeInput, axis: int, name: Optional[str] = None) -> Node:
    """Apply softmax operation on each element of input tensor.

    :param data: The tensor providing input data.
    :param axis: An axis along which Softmax should be calculated. Can be positive or negative.
    :param name: Optional name for the node.
    :return: The new node with softmax operation applied on each element.
    """
    return _get_node_factory_opset8().create("Softmax", [as_node(data, name=name)], {"axis": axis})
