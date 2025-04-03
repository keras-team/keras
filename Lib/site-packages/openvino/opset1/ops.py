# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from typing import List, Optional, Union, get_args

import numpy as np
from functools import partial

from openvino import Node, PartialShape, Type
from openvino.op import Constant, Parameter, tensor_iterator
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import binary_op, nameable_op, unary_op
from openvino.utils.input_validation import (
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from openvino.utils.node_factory import NodeFactory
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
from openvino.utils import deprecated

_get_node_factory_opset1 = partial(_get_node_factory, "opset1")

# -------------------------------------------- ops ------------------------------------------------


@unary_op
def absolute(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies f(x) = abs(x) to the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with Abs operation applied on it.
    """
    return _get_node_factory_opset1().create("Abs", [node])


@unary_op
def acos(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply inverse cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arccos operation applied on it.
    """
    return _get_node_factory_opset1().create("Acos", [node])


@binary_op
def add(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies f(A,B) = A+B to the input nodes element-wise.

    :param left_node: The first input node for add operation.
    :param right_node: The second input node for add operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors. Defaults to "NUMPY".
    :param name: The optional name for output new node.
    :return: The node performing element-wise addition.
    """
    return _get_node_factory_opset1().create(
        "Add",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def asin(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply inverse sine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arcsin operation applied on it.
    """
    return _get_node_factory_opset1().create("Asin", [node])


@unary_op
def atan(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply inverse tangent function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arctan operation applied on it.
    """
    return _get_node_factory_opset1().create("Atan", [node])


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
    :param pads_begin:      The input data optional padding below filled with zeros.
    :param pads_end:        The input data optional padding below filled with zeros.
    :param kernel_shape:    The pooling window shape.
    :param exclude_pad:     Whether or not to include zero padding in average computations.
    :param rounding_type:   Determines used rounding schema when computing output shape. Acceptable
                            values are: ['floor', 'ceil']
    :param auto_pad:        Determines how the padding is calculated. Acceptable values:
                            [None, 'same_upper', 'same_lower', 'valid']
    :param name:            Optional name for the new output node.

    :return: New node with AvgPool operation applied on its data.
    """
    if auto_pad is None:
        auto_pad = "explicit"
    return _get_node_factory_opset1().create(
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


@nameable_op
def batch_norm_inference(
    data: NodeInput,
    gamma: NodeInput,
    beta: NodeInput,
    mean: NodeInput,
    variance: NodeInput,
    epsilon: float,
    name: Optional[str] = None,
) -> Node:
    """Perform layer normalizes a input tensor by mean and variance with appling scale and offset.

    :param data: The input tensor with data for normalization.
    :param gamma: The scalar scaling for normalized value.
    :param beta: The bias added to the scaled normalized value.
    :param mean: The value for mean normalization.
    :param variance: The value for variance normalization.
    :param epsilon: The  number to be added to the variance to avoid division
                    by zero when normalizing a value.
    :param name: The optional name of the output node.
    :return: The new node which performs BatchNormInference.
    """
    inputs = as_nodes(gamma, beta, data, mean, variance, name=name)
    return _get_node_factory_opset1().create("BatchNormInference", inputs, {"epsilon": epsilon})


@nameable_op
def binary_convolution(
    data: NodeInput,
    filters: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    dilations: List[int],
    mode: str,
    pad_value: float,
    auto_pad: str = "EXPLICIT",
    name: Optional[str] = None,
) -> Node:
    """Create node performing convolution with binary weights, binary input and integer output.

    :param data: The node providing data batch tensor.
    :param filter: The node providing filters tensor.
    :param strides: The kernel window movement strides.
    :param pads_begin: The number of pixels to add to the beginning along each axis.
    :param pads_end: The number of pixels to add to the end along each axis.
    :param dilations: The distance in width and height between elements (weights) in the filter.
    :param mode: Defines how input tensor 0/1 values and weights 0/1 are interpreted.
    :param pad_value: Floating-point value used to fill pad area.
    :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
    :param name: The optional new name for output node.
    :return: New node performing binary convolution operation.
    """
    return _get_node_factory_opset1().create(
        "BinaryConvolution",
        as_nodes(data, filters, name=name),
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "mode": mode,
            "pad_value": pad_value,
            "auto_pad": auto_pad,
        },
    )


@nameable_op
def broadcast(
    data: NodeInput,
    target_shape: NodeInput,
    axes_mapping: Optional[NodeInput] = None,
    mode: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Create a node which broadcasts the input node's values along specified axes to a desired shape.

    :param data: The node with input tensor data.
    :param target_shape: The node with a new shape we want to broadcast tensor to.
    :param axes_mapping: The node with a axis positions (0-based) in the result
                           that are being broadcast.
    :param mode: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: NUMPY, EXPLICIT.
    :param name: Optional new name for output node.
    :return: New node with broadcast shape.
    """
    inputs = as_nodes(data, target_shape, name=name)
    if mode.upper() == "EXPLICIT":
        inputs.append(as_node(axes_mapping, name=name))
    return _get_node_factory_opset1().create(
        "Broadcast",
        inputs,
        {"mode": mode.upper()},
    )


@nameable_op
def ctc_greedy_decoder(
    data: NodeInput,
    sequence_mask: NodeInput,
    merge_repeated: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Perform greedy decoding on the logits given in input (best path).

    :param data: Logits on which greedy decoding is performed.
    :param sequence_mask: The tensor with sequence masks for each sequence in the batch.
    :param merge_repeated: The flag for merging repeated labels during the CTC calculation.
    :param name: Optional name for output node.
    :return: The new node performing an CTCGreedyDecoder operation on input tensor.
    """
    node_inputs = as_nodes(data, sequence_mask, name=name)
    return _get_node_factory_opset1().create(
        "CTCGreedyDecoder",
        node_inputs,
        {"ctc_merge_repeated": merge_repeated},
    )


@unary_op
def ceiling(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies ceiling to the input node element-wise.

    :param node: The node providing data to ceiling operation.
    :param name: Optional name for output node.
    :return: The node performing element-wise ceiling.
    """
    return _get_node_factory_opset1().create("Ceiling", [node])


@nameable_op
def clamp(
    data: NodeInput,
    min_value: ScalarData,
    max_value: ScalarData,
    name: Optional[str] = None,
) -> Node:
    """Perform clamp element-wise on data from input node.

    :param data: Input tensor. One of: input node, array or scalar.
    :param min_value: The lower bound of the <min_value;max_value> range. Scalar value.
    :param max_value: The upper bound of the <min_value;max_value> range. Scalar value.
    :param name: Optional output node name.
    :return: The new node performing a clamp operation on its input data element-wise.

    Performs a clipping operation on an input value between a pair of boundary values.

    For each element in `data`, if the element's value is lower than `min_value`,
    it will be replaced with `min_value`. If the value is higher than `max_value`,
    it will be replaced by `max_value`.
    Intermediate values of `data` are returned without change.

    Clamp uses the following logic:

    .. code-block:: python

        if data < min_value:
            data=min_value
        elif data > max_value:
            data=max_value
    """
    return _get_node_factory_opset1().create(
        "Clamp",
        [as_node(data, name=name)],
        {"min": min_value, "max": max_value},
    )


@nameable_op
def concat(nodes: List[NodeInput], axis: int, name: Optional[str] = None) -> Node:
    """Concatenate input nodes into single new node along specified axis.

    :param nodes: The nodes we want concatenate into single new node.
    :param axis: The axis along which we want to concatenate input nodes.
    :param name: The optional new name for output node.
    :return: Return new node that is a concatenation of input nodes.
    """
    return _get_node_factory_opset1().create("Concat", as_nodes(*nodes, name=name), {"axis": axis})


@nameable_op
def constant(
    value: NumericData,
    dtype: Union[NumericType, Type] = None,
    name: Optional[str] = None,
) -> Constant:
    """Create a Constant node from provided value.

    :param value: One of: array of values or scalar to initialize node with.
    :param dtype: The data type of provided data.
    :param name: Optional name for output node.
    :return: The Constant node initialized with provided data.
    """
    if value is None or (isinstance(value, np.ndarray) and value.size == 0):
        raise ValueError("Cannot create an empty Constant. Please provide valid data.")
    return make_constant_node(value, dtype)


@nameable_op
def convert(
    data: NodeInput,
    destination_type: Union[str, NumericType, Type],
    name: Optional[str] = None,
) -> Node:
    """Return node which casts input node values to specified type.

    :param data: Node which produces the input tensor.
    :param destination_type: Provides the target type for the conversion.
    :param name: Optional name for the output node.
    :return: New node performing the conversion operation.
    """
    _destination_type = None  # type: Union[str, Type]
    if isinstance(destination_type, get_args(NumericType)):
        _destination_type = get_element_type_str(destination_type).lower()
    else:
        _destination_type = destination_type
    return _get_node_factory_opset1().create(
        "Convert",
        [as_node(data, name=name)],
        {"destination_type": _destination_type},
    )


@binary_op
def convert_like(data: NodeInput, like: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which casts data node values to the type of another node.

    :param data: Node which produces the input tensor
    :param like: Node which provides the target type information for the conversion
    :param name: Optional name for the output node.
    :return: New node performing the conversion operation.
    """
    return _get_node_factory_opset1().create("ConvertLike", [data, like])


@nameable_op
def convolution(
    data: NodeInput,
    filters: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    dilations: List[int],
    auto_pad: str = "EXPLICIT",
    name: Optional[str] = None,
) -> Node:
    """Return node performing batched convolution operation.

    :param data: The node providing data batch tensor.
    :param filter: The node providing filters tensor.
    :param strides: The kernel window movement strides.
    :param pads_begin: The number of zero padding elements to add on each axis below 0 coordinate.
    :param pads_end: The number of zero padding elements to add on each axis above max coordinate
    :param dilations: The data batch dilation strides.
    :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
    :param name: The optional new name for output node.
    :return: New node performing batched convolution operation.
    """
    return _get_node_factory_opset1().create(
        "Convolution",
        as_nodes(data, filters, name=name),
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad,
        },
    )


@nameable_op
def convolution_backprop_data(
    data: NodeInput,
    filters: NodeInput,
    strides: List[int],
    output_shape: Optional[NodeInput] = None,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    auto_pad: Optional[str] = None,
    output_padding: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Create node performing a batched-convolution backprop data operation.

    :param      data:         The node producing data from forward-prop
    :param      filters:      The node producing the filters from forward-prop.
    :param      output_shape: The node producing output delta.
    :param      strides:      The distance (in pixels) to slide the filter on the feature map
                              over the axes.
    :param      pads_begin:   The number of pixels to add to the beginning along each axis.
    :param      pads_end:     The number of pixels to add to the end along each axis.
    :param      dilations:    The distance in width and height between elements (weights)
                              in the filter.
    :param      name:         The node name.

    :return:   The node object representing ConvolutionBackpropData  operation.
    """
    spatial_dim_count = len(strides)
    if pads_begin is None:
        pads_begin = [0] * spatial_dim_count
    if pads_end is None:
        pads_end = [0] * spatial_dim_count
    if dilations is None:
        dilations = [1] * spatial_dim_count
    if auto_pad is None:
        auto_pad = "explicit"
    if output_padding is None:
        output_padding = [0] * spatial_dim_count
    args = as_nodes(data, filters, name=name)
    if output_shape is not None:
        args.append(as_node(output_shape, name=name))

    return _get_node_factory_opset1().create(
        "ConvolutionBackpropData",
        args,
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad.upper(),
            "output_padding": output_padding,
        },
    )


@unary_op
def cos(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with cos operation applied on it.
    """
    return _get_node_factory_opset1().create("Cos", [node])


@unary_op
def cosh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply hyperbolic cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with cosh operation applied on it.
    """
    return _get_node_factory_opset1().create("Cosh", [node])


@nameable_op
def deformable_convolution(
    data: NodeInput,
    deformable_values: NodeInput,
    filters: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    dilations: List[int],
    auto_pad: str = "EXPLICIT",
    group: int = 1,
    deformable_group: int = 1,
    name: Optional[str] = None,
) -> Node:
    """Create node performing deformable convolution.

    :param data: The node providing data batch tensor.
    :param filter: The node providing filters tensor.
    :param strides: The distance (in pixels) to slide the filter on the feature map over the axes.
    :param pads_begin: The number of pixels to add to the beginning along each axis.
    :param pads_end: The number of pixels to add to the end along each axis.
    :param dilations: The distance in width and height between elements (weights) in the filter.
    :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
    :param group: The number of groups which both output and input should be split into.
    :param deformable_group: The number of groups which deformable values and output should be split
                             into along the channel axis.
    :param name: The optional new name for output node.
    :return: New node performing deformable convolution operation.
    """
    return _get_node_factory_opset1().create(
        "DeformableConvolution",
        as_nodes(data, deformable_values, filters, name=name),
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad,
            "group": group,
            "deformable_group": deformable_group,
        },
    )


@nameable_op
def deformable_psroi_pooling(
    feature_maps: NodeInput,
    coords: NodeInput,
    output_dim: int,
    spatial_scale: float,
    group_size: int = 1,
    mode: str = "bilinear_deformable",
    spatial_bins_x: int = 1,
    spatial_bins_y: int = 1,
    trans_std: float = 1.0,
    part_size: int = 1,
    offsets: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return node performing DeformablePSROIPooling operation.

    DeformablePSROIPooling computes position-sensitive pooling
    on regions of interest specified by input.

    :param feature_maps: 4D tensor with feature maps.
    :param coords: 2D tensor describing box consisting of tuples: [batch_id, x_1, y_1, x_2, y_2].
    :param output_dim: A pooled output channel number.
    :param spatial_scale: A multiplicative spatial scale factor to translate ROI.
    :param group_size: The number of groups to encode position-sensitive score.
    :param mode: Specifies mode for pooling. Range of values: ['bilinear_deformable'].
    :param spatial_bins_x: Specifies numbers of bins to divide the input feature maps over width.
    :param spatial_bins_y: Specifies numbers of bins to divide the input feature maps over height.
    :param trans_std: The value that all transformation (offset) values are multiplied with.
    :param part_size: The number of parts the output tensor spatial dimensions are divided into.
    :param offsets: Optional node. 4D input blob with transformation values (offsets).
    :param name: The optional new name for output node.
    :return: New node performing DeformablePSROIPooling operation.
    """
    node_inputs = as_nodes(feature_maps, coords, name=name)
    if offsets is not None:
        node_inputs.append(as_node(offsets, name=name))

    return _get_node_factory_opset1().create(
        "DeformablePSROIPooling",
        node_inputs,
        {
            "output_dim": output_dim,
            "spatial_scale": spatial_scale,
            "group_size": group_size,
            "mode": mode,
            "spatial_bins_x": spatial_bins_x,
            "spatial_bins_y": spatial_bins_y,
            "trans_std": trans_std,
            "part_size": part_size,
        },
    )


@nameable_op
def depth_to_space(node: Node, mode: str, block_size: int = 1, name: Optional[str] = None) -> Node:
    """Rearranges input tensor from depth into blocks of spatial data.

    Values from the height and width dimensions are moved to the depth dimension.

    Input tensor has shape [N,C,H,W], where N is the batch axis, C is the channel or depth,
    H is the height and W is the width.

    Output node produces a tensor with shape:

    [N, C * `block_size` * `block_size`, H / `block_size`, W / `block_size`]

    :param node: The node with input tensor data.
    :param mode: Specifies how the input depth dimension is split to block coordinates

                 blocks_first: The input is divided to [block_size, ..., block_size, new_depth]
                 depth_first: The input is divided to [new_depth, block_size, ..., block_size]

    :param block_size: The size of the spatial block of values describing
                       how the tensor's data is to be rearranged.
    :param name: Optional output node name.
    :return: The new node performing an DepthToSpace operation on its input tensor.
    """
    return _get_node_factory_opset1().create(
        "DepthToSpace",
        [node],
        {"mode": mode, "block_size": block_size},
    )


@nameable_op
def detection_output(
    box_logits: Node,
    class_preds: Node,
    proposals: Node,
    attrs: dict,
    aux_class_preds: NodeInput = None,
    aux_box_preds: NodeInput = None,
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

    * num_classes       The number of classes to be predicted.
                        Range of values: positive integer number
                        Default value: None
                        Required: yes

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
            'num_classes': 85,
            'keep_top_k': [1, 2, 3],
            'nms_threshold': 0.645,

        }

        attrs = {
            'num_classes': 85,
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
        ("num_classes", True, np.integer, is_positive_value),
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

    return _get_node_factory_opset1().create("DetectionOutput", inputs, attrs)


@binary_op
def divide(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies f(x) = A/B to the input nodes element-wise.

    :param left_node: The node providing dividend data.
    :param right_node: The node providing divisor data.
    :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
    :param name: Optional name for output node.
    :return: The node performing element-wise division.
    """
    return _get_node_factory_opset1().create(
        "Divide",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def elu(data: NodeInput, alpha: NumericType, name: Optional[str] = None) -> Node:
    """Perform Exponential Linear Unit operation element-wise on data from input node.

    Computes exponential linear: alpha * (exp(data) - 1) if < 0, data otherwise.

    For more information refer to:
    [Fast and Accurate Deep Network Learning by Exponential Linear Units](http://arxiv.org/abs/1511.07289)

    :param data: Input tensor. One of: input node, array or scalar.
    :param alpha: Scalar multiplier for negative values.
    :param name: Optional output node name.
    :return: The new node performing an ELU operation on its input data element-wise.
    """
    return _get_node_factory_opset1().create("Elu", [as_node(data, name=name)], {"alpha": alpha})


@binary_op
def equal(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if input nodes are equal element-wise.

    :param left_node: The first input node for equal operation.
    :param right_node: The second input node for equal operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional name for output new node.
    :return: The node performing element-wise equality check.
    """
    return _get_node_factory_opset1().create(
        "Equal",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def erf(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which calculates Gauss error function element-wise with given tensor.

    :param node: The node providing data for operation.
    :param name: The optional name for new output node.
    :return: The new node performing element-wise Erf operation.
    """
    return _get_node_factory_opset1().create("Erf", [node])


@unary_op
def exp(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies exponential function to the input node element-wise.

    :param node: The node providing data for operation.
    :param name: The optional name for new output node.
    :return: The new node performing natural exponential operation.
    """
    return _get_node_factory_opset1().create("Exp", [node])


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
    return _get_node_factory_opset1().create(
        "FakeQuantize",
        as_nodes(data, input_low, input_high, output_low, output_high, name=name),
        {"levels": levels, "auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def floor(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies floor to the input node element-wise.

    :param node: The input node providing data.
    :param name: The optional name for new output node.
    :return: The node performing element-wise floor operation.
    """
    return _get_node_factory_opset1().create("Floor", [node])


@binary_op
def floor_mod(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node performing element-wise FloorMod (division reminder) with two given tensors.

    :param left_node: The first input node for FloorMod operation.
    :param right_node: The second input node for FloorMod operation.
    :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
    :param name: Optional name for output node.
    :return: The node performing element-wise FloorMod operation.
    """
    return _get_node_factory_opset1().create(
        "FloorMod",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def gather(
    data: NodeInput,
    indices: NodeInput,
    axis: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return Gather node which takes slices from axis of data according to indices.

    :param data: The tensor from which slices are gathered.
    :param indices: Tensor with indexes to gather.
    :param axis: The dimension index to gather data from.
    :param name: Optional name for output node.
    :return: The new node performing a Gather operation on the data input tensor.
    """
    node_inputs = as_nodes(data, indices, axis, name=name)
    return _get_node_factory_opset1().create("Gather", node_inputs)


@nameable_op
def gather_tree(
    step_ids: NodeInput,
    parent_idx: NodeInput,
    max_seq_len: NodeInput,
    end_token: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform GatherTree operation.

    :param step_ids: The tensor with indices from per each step.
    :param parent_idx: The tensor with with parent beam indices.
    :param max_seq_len: The tensor with maximum lengths for each sequence in the batch.
    :param end_token: The scalar tensor with value of the end marker in a sequence.
    :param name: Optional name for output node.
    :return: The new node performing a GatherTree operation.

    The GatherTree node generates the complete beams from the indices per each step
    and the parent beam indices.
    GatherTree uses the following logic:

    .. code-block:: python

        for batch in range(BATCH_SIZE):
            for beam in range(BEAM_WIDTH):
                max_sequence_in_beam = min(MAX_TIME, max_seq_len[batch])

                parent = parent_idx[max_sequence_in_beam - 1, batch, beam]

                for level in reversed(range(max_sequence_in_beam - 1)):
                    final_idx[level, batch, beam] = step_idx[level, batch, parent]

                    parent = parent_idx[level, batch, parent]
    """
    node_inputs = as_nodes(step_ids, parent_idx, max_seq_len, end_token, name=name)
    return _get_node_factory_opset1().create("GatherTree", node_inputs)


@binary_op
def greater(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if left input node is greater than the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is greater than right_node.
    """
    return _get_node_factory_opset1().create(
        "Greater",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def greater_equal(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if left node is greater or equal to the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional new name for output node.

    :return: The node performing element-wise check whether left_node is greater than or equal right_node.
    """
    return _get_node_factory_opset1().create(
        "GreaterEqual",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


def grn(data: Node, bias: float, name: Optional[str] = None) -> Node:
    r"""Perform Global Response Normalization with L2 norm (across channels only).

    Computes GRN operation on channels for input tensor:

    \f[ output_i = \dfrac{input_i}{\sqrt{\sum_{i}^{C} input_i}} \f]

    :param data: The node with data tensor.
    :param bias: The bias added to the variance. Scalar value.
    :param name: Optional output node name.
    :return: The new node performing a GRN operation on tensor's channels.
    """
    return _get_node_factory_opset1().create("GRN", [data], {"bias": bias})


@nameable_op
def group_convolution(
    data: NodeInput,
    filters: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    dilations: List[int],
    auto_pad: str = "EXPLICIT",
    name: Optional[str] = None,
) -> Node:
    """Perform Group Convolution operation on data from input node.

    :param data:        The node producing input data.
    :param filters:     The node producing filters data.
    :param strides:     The distance (in pixels) to slide the filter on the feature map
                        over the axes.
    :param pads_begin:  The number of pixels to add at the beginning along each axis.
    :param pads_end:    The number of pixels to add at the end along each axis.
    :param dilations:   The distance in width and height between elements (weights) in the filter.
    :param auto_pad:    Describes how to perform padding. Possible values:
                        EXPLICIT:   Pad dimensions are explicity specified
                        SAME_LOWER: Pad dimensions computed to match input shape
                        Ceil(num_dims/2) at the beginning and
                        Floor(num_dims/2) at the end

                        SAME_UPPER: Pad dimensions computed to match input shape
                                    Floor(num_dims/2) at the beginning and
                                    Ceil(num_dims/2) at the end

                        VALID:      No padding
    :param name: Optional output node name.
    :return: The new node performing a Group Convolution operation on tensor from input node.
    """
    return _get_node_factory_opset1().create(
        "GroupConvolution",
        as_nodes(data, filters, name=name),
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad.upper(),
        },
    )


@nameable_op
def group_convolution_backprop_data(
    data: NodeInput,
    filters: NodeInput,
    strides: List[int],
    output_shape: Optional[NodeInput] = None,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    auto_pad: str = "EXPLICIT",
    output_padding: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform Group Convolution operation on data from input node.

    :param data:            The node producing input data.
    :param filters:         The node producing filter data.
    :param strides:         The distance (in pixels) to slide the filter on the feature map
                            over the axes.
    :param output_shape:    The node that specifies spatial shape of the output.
    :param pads_begin:      The number of pixels to add at the beginning along each axis.
    :param pads_end:        The number of pixels to add at the end along each axis.
    :param dilations:       The distance in width and height between elements (weights)
                            in the filter.
    :param auto_pad:        Describes how to perform padding. Possible values:
                            EXPLICIT:   Pad dimensions are explicity specified
                            SAME_LOWER: Pad dimensions computed to match input shape
                            Ceil(num_dims/2) at the beginning and
                            Floor(num_dims/2) at the end

                            SAME_UPPER: Pad dimensions computed to match input shape
                                        Floor(num_dims/2) at the beginning and
                                        Ceil(num_dims/2) at the end

                            VALID:      No padding

    :param output_padding:  The additional amount of paddings added per each spatial axis
                            in the output tensor.
    :param name: Optional output node name.
    :return: The new node performing a Group Convolution operation on tensor from input node.
    """
    spatial_dim_count = len(strides)
    if dilations is None:
        dilations = [1] * spatial_dim_count
    if output_padding is None:
        output_padding = [0] * spatial_dim_count

    attributes = {
        "strides": strides,
        "dilations": dilations,
        "auto_pad": auto_pad.upper(),
        "output_padding": output_padding,
    }
    args = as_nodes(data, filters, name=name)

    if output_shape is not None:
        args.append(as_node(output_shape, name=name))
    else:
        if pads_begin is None:
            pads_begin = [0] * spatial_dim_count
        if pads_end is None:
            pads_end = [0] * spatial_dim_count
        attributes["pads_begin"] = pads_begin
        attributes["pads_end"] = pads_end

    return _get_node_factory_opset1().create("GroupConvolutionBackpropData", args, attributes)


@nameable_op
def hard_sigmoid(
    data: Node,
    alpha: NodeInput,
    beta: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform Hard Sigmoid operation element-wise on data from input node.

    :param data: The node with data tensor.
    :param alpha: A node producing the alpha parameter.
    :param beta: A node producing the beta parameter
    :param name: Optional output node name.
    :return: The new node performing a Hard Sigmoid element-wise on input tensor.

    Hard Sigmoid uses the following logic:

    .. code-block:: python

        y = max(0, min(1, alpha * data + beta))
    """
    return _get_node_factory_opset1().create("HardSigmoid", [data, as_node(alpha, name=name), as_node(beta, name=name)])


@nameable_op
def interpolate(
    image: Node,
    output_shape: NodeInput,
    attrs: dict,
    name: Optional[str] = None,
) -> Node:
    """Perform interpolation of independent slices in input tensor.

    :param  image:         The node providing input tensor with data for interpolation.
    :param  output_shape:  1D tensor describing output shape for spatial axes.
    :param  attrs:         The dictionary containing key, value pairs for attributes.
    :param  name:          Optional name for the output node.
    :return: Node representing interpolation operation.

    Available attributes are:

    * axes              Specify spatial dimension indices where interpolation is applied.
                        Type: List of non-negative integer numbers.
                        Required: yes.

    * mode              Specifies type of interpolation.
                        Range of values: one of {nearest, linear, cubic, area}
                        Type: string
                        Required: yes

    * align_corners     A flag that specifies whether to align corners or not. True means the
                        alignment is applied, False means the alignment isn't applied.
                        Range of values: True or False. Default: True.
                        Required: no

    * antialias         A flag that specifies whether to perform anti-aliasing.
                        Range of values: False - do not perform anti-aliasing
                                         True - perform anti-aliasing

                        Default value: False
                        Required: no

    * pads_begin        Specify the number of pixels to add to the beginning of the image being
                        interpolated. A scalar that specifies padding for each spatial dimension.
                        Range of values: list of non-negative integer numbers. Default value: 0
                        Required: no

    * pads_end          Specify the number of pixels to add to the beginning of the image being
                        interpolated. A scalar that specifies padding for each spatial dimension.
                        Range of values: list of non-negative integer numbers. Default value: 0
                        Required: no

    Example of attribute dictionary:

    .. code-block:: python

        # just required ones
        attrs = {
            'axes': [2, 3],
            'mode': 'cubic',
        }

        attrs = {
            'axes': [2, 3],
            'mode': 'cubic',
            'antialias': True,
            'pads_begin': [2, 2, 2],
        }

    Optional attributes which are absent from dictionary will be set with corresponding default.
    """
    requirements = [
        ("axes", True, np.integer, is_non_negative_value),
        ("mode", True, np.str_, None),
        ("align_corners", False, np.bool_, None),
        ("antialias", False, np.bool_, None),
        ("pads_begin", False, np.integer, is_non_negative_value),
        ("pads_end", False, np.integer, is_non_negative_value),
    ]

    check_valid_attributes("Interpolate", attrs, requirements)

    return _get_node_factory_opset1().create("Interpolate", [image, as_node(output_shape, name=name)], attrs)


@binary_op
def less(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if left input node is less than the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is less than the right_node.
    """
    return _get_node_factory_opset1().create(
        "Less",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def less_equal(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if left input node is less or equal the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is less than or equal the
             right_node.
    """
    return _get_node_factory_opset1().create(
        "LessEqual",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def log(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies natural logarithm to the input node element-wise.

    :param node: The input node providing data for operation.
    :param name: The optional new name for output node.
    :return: The new node performing log operation element-wise.
    """
    return _get_node_factory_opset1().create("Log", [node])


@binary_op
def logical_and(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which perform logical and operation on input nodes element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: numpy, explicit.
    :param name: The optional new name for output node.
    :return: The node performing logical and operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset1().create(
        "LogicalAnd",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def logical_not(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies element-wise logical negation to the input node.

    :param node: The input node providing data.
    :param name: The optional new name for output node.
    :return: The node performing element-wise logical NOT operation with given tensor.
    """
    return _get_node_factory_opset1().create("LogicalNot", [node])


@binary_op
def logical_or(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs logical OR operation on input nodes element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: numpy, explicit.
    :param name: The optional new name for output node.
    :return: The node performing logical or operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset1().create(
        "LogicalOr",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def logical_xor(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs logical XOR operation on input nodes element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: numpy, explicit.
    :param name: The optional new name for output node.
    :return: The node performing logical or operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset1().create(
        "LogicalXor",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def lrn(
    data: NodeInput,
    axes: NodeInput,
    alpha: float = 1,
    beta: float = 0.5,
    bias: float = 1,
    size: int = 5,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs element-wise Local Response Normalization (LRN) operation.

    :param data: Input data.
    :param alpha: A scale factor (usually positive).
    :param beta: An exponent.
    :param bias: An offset (usually positive) to avoid dividing by 0.
    :param size: Width of the 1-D normalization window.
    :param name: An optional name of the output node.
    :return: The new node which performs LRN.
    """
    attributes = {"alpha": alpha, "beta": beta, "bias": bias, "size": size}
    return _get_node_factory_opset1().create("LRN", as_nodes(data, axes, name=name), attributes)


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

    node_inputs = as_nodes(
        X,
        initial_hidden_state,
        initial_cell_state,
        W,
        R,
        B,
        name=name,
    )

    # P - nGraph additional input, no such input in the OV spec
    peepholes_count = 3  # nGraph default
    peepholes_shape = [peepholes_count * hidden_size]
    peepholes_array = np.zeros(peepholes_shape)  # nGraph default
    data_dtype = get_dtype(node_inputs[0].get_output_element_type(0))
    default_p = make_constant_node(peepholes_array, dtype=data_dtype)
    node_inputs.append(default_p)

    weights_format = "fico"  # OV LSTMWeightsFormat, no such attribute in the OV spec
    input_forget = False  # nGraph default, no such attribute in the OV spec

    attributes = {
        "hidden_size": hidden_size,
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "clip": clip,
        "weights_format": weights_format,
        "input_forget": input_forget,
    }
    return _get_node_factory_opset1().create("LSTMCell", node_inputs, attributes)


@nameable_op
def matmul(
    data_a: NodeInput,
    data_b: NodeInput,
    transpose_a: bool,
    transpose_b: bool,
    name: Optional[str] = None,
) -> Node:
    """Return the Matrix Multiplication operation.

    :param data_a: left-hand side matrix
    :param data_b: right-hand side matrix
    :param transpose_a: should the first matrix be transposed before operation
    :param transpose_b: should the second matrix be transposed
    :return: MatMul operation node
    """
    return _get_node_factory_opset1().create(
        "MatMul",
        as_nodes(data_a, data_b, name=name),
        {"transpose_a": transpose_a, "transpose_b": transpose_b},
    )


@nameable_op
def max_pool(
    data: NodeInput,
    strides: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    kernel_shape: TensorShape,
    rounding_type: str = "floor",
    auto_pad: Optional[str] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform max pooling operation with given parameters on provided data.

    :param  data:           The node providing input data.
    :param  strides:        The distance (in pixels) to slide the filter on the feature map
                            over the axes.
    :param  pads_begin:     The number of pixels to add at the beginning along each axis.
    :param  pads_end:       The number of pixels to add at the end along each axis.
    :param  kernel_shape:   The pooling operation kernel shape.
    :param  rounding_type:  Determines used rounding schema when computing output shape. Acceptable
                            values are: ['floor', 'ceil']
    :param  auto_pad:       Determines how the padding is calculated. Acceptable values:
                            [None, 'same_upper', 'same_lower', 'valid']
    :param  name:           The optional name for the created output node.

    :return:   The new node performing max pooling operation.
    """
    if auto_pad is None:
        auto_pad = "explicit"
    return _get_node_factory_opset1().create(
        "MaxPool",
        [as_node(data, name=name)],
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "kernel": kernel_shape,
            "rounding_type": rounding_type.upper(),
            "auto_pad": auto_pad.upper(),
        },
    )


@binary_op
def maximum(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies the maximum operation to input nodes elementwise.

    :param left_node: The first input node for maximum operation.
    :param right_node: The second input node for maximum operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors. Defaults to "NUMPY".
    :param name: The optional name for output new node.
    :return: The node performing element-wise maximum operation.
    """
    return _get_node_factory_opset1().create(
        "Maximum",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def minimum(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies the minimum operation to input nodes elementwise.

    :param left_node: The first input node for minimum operation.
    :param right_node: The second input node for minimum operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors. Defaults to "NUMPY".
    :param name: The optional name for output new node.
    :return: The node performing element-wise minimum operation.
    """
    return _get_node_factory_opset1().create(
        "Minimum",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def mod(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node performing element-wise division reminder with two given tensors.

    :param left_node: The first input node for mod operation.
    :param right_node: The second input node for mod operation.
    :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
    :param name: Optional name for output node.
    :return: The node performing element-wise Mod operation.
    """
    return _get_node_factory_opset1().create(
        "Mod",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def multiply(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies f(A,B) = A*B to the input nodes elementwise.

    :param left_node: The first input node for multiply operation.
    :param right_node: The second input node for multiply operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors. Defaults to "NUMPY".
    :param name: The optional name for output new node.
    :return: The node performing element-wise multiplication.
    """
    return _get_node_factory_opset1().create(
        "Multiply",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def negative(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies f(x) = -x to the input node elementwise.

    :param node: Input node for negative operation.
    :param name: The optional name for output new node.
    :return: The node performing element-wise multiplicaion by -1.
    """
    return _get_node_factory_opset1().create("Negative", [node])


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    box_encoding: str = "corner",
    sort_result_descending: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NonMaxSuppression.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor specifying intersection over union threshold
    :param score_threshold: Tensor specifying minimum score to consider box for the processing.
    :param box_encoding: Format of boxes data encoding. Range of values: corner or cente.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
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
    }

    return _get_node_factory_opset1().create("NonMaxSuppression", inputs, attributes)


@nameable_op
def normalize_l2(
    data: NodeInput,
    axes: NodeInput,
    eps: float,
    eps_mode: str,
    name: Optional[str] = None,
) -> Node:
    """Construct an NormalizeL2 operation.

    :param data: Node producing the input tensor
    :param axes: Node indicating axes along which L2 reduction is calculated
    :param eps: The epsilon added to L2 norm
    :param eps_mode: how eps is combined with L2 value (`add` or `max`)
    :return: New node which performs the L2 normalization.
    """
    return _get_node_factory_opset1().create(
        "NormalizeL2",
        as_nodes(data, axes, name=name),
        {"eps": eps, "mode": eps_mode},
    )


@binary_op
def not_equal(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which checks if input nodes are unequal element-wise.

    :param left_node: The first input node for not-equal operation.
    :param right_node: The second input node for not-equal operation.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :param name: The optional name for output new node.
    :return: The node performing element-wise inequality check.
    """
    return _get_node_factory_opset1().create(
        "NotEqual",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def one_hot(
    indices: NodeInput,
    depth: NodeInput,
    on_value: NodeInput,
    off_value: NodeInput,
    axis: int,
    name: Optional[str] = None,
) -> Node:
    """Create node performing one-hot encoding on input data.

    :param indices: Input tensor of rank N with indices of any supported integer data type.
    :param depth: Scalar of any supported integer type that specifies number of classes and
                  the size of one-hot dimension.
    :param on_value: Scalar of any type that is the value that the locations
                     in output tensor represented by indices in input take.
    :param off_value: Scalar of any type that is the value that the locations not represented
                      by indices in input take.

    :param name: The optional name for new output node.
    :return: New node performing one-hot operation.
    """
    return _get_node_factory_opset1().create(
        "OneHot",
        as_nodes(indices, depth, on_value, off_value, name=name),
        {"axis": axis},
    )


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
    :param pads_begin: number of padding elements to be added before position 0
                       on each axis of arg.
    :param pads_end: number of padding elements to be added after the last element.
    :param pad_mode: "constant", "edge", "reflect" or "symmetric"
    :param arg_pad_value: value used for padding if pad_mode is "constant"
    :return: Pad operation node.
    """
    input_nodes = as_nodes(arg, pads_begin, pads_end, name=name)
    if arg_pad_value:
        input_nodes.append(as_node(arg_pad_value, name=name))

    pad_mode = pad_mode.upper()
    return _get_node_factory_opset1().create("Pad", input_nodes, {"pad_mode": pad_mode})


@nameable_op
def parameter(
    shape: TensorShape,
    dtype: Union[NumericType, Type] = np.float32,
    name: Optional[str] = None,
) -> Parameter:
    """Return an openvino Parameter object.

    :param shape: The shape of the output tensor.
    :param dtype: The type of elements of the output tensor. Defaults to np.float32.
    :param name: The optional name for output new node.
    :return: The node that specifies input to the model.
    """
    return Parameter(
        get_element_type(dtype) if isinstance(dtype, (type, np.dtype)) else dtype,
        PartialShape(shape),
    )


@binary_op
def power(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which perform element-wise exponentiation operation.

    :param left_node: The node providing the base of operation.
    :param right_node: The node providing the exponent of operation.
    :param name: The optional name for the new output node.
    :param auto_broadcast: The type of broadcasting specifies rules used for
                           auto-broadcasting of input tensors.
    :return: The new node performing element-wise exponentiation operation on input nodes.
    """
    return _get_node_factory_opset1().create(
        "Power",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def prelu(data: NodeInput, slope: NodeInput, name: Optional[str] = None) -> Node:
    """Perform Parametrized Relu operation element-wise on data from input node.

    :param data: The node with data tensor.
    :param slope: The node with the multipliers for negative values.
    :param name: Optional output node name.
    :return: The new node performing a PRelu operation on tensor's channels.

    PRelu uses the following logic:

    .. code-block:: python

        if data < 0:
            data = data * slope
        elif data >= 0:
            data = data
    """
    return _get_node_factory_opset1().create("PRelu", as_nodes(data, slope, name=name))


@nameable_op
def prior_box_clustered(
    output_size: Node,
    image_size: NodeInput,
    attrs: dict,
    name: Optional[str] = None,
) -> Node:
    """Generate prior boxes of specified sizes normalized to the input image size.

    :param  output_size:    1D tensor with two integer elements [height, width]. Specifies the
                            spatial size of generated grid with boxes.
    :param  image_size:     1D tensor with two integer elements [image_height, image_width] that
                            specifies shape of the image for which boxes are generated.
    :param  attrs:          The dictionary containing key, value pairs for attributes.
    :param  name:           Optional name for the output node.
    :return: Node representing PriorBoxClustered operation.

     Available attributes are:

    * widths        Specifies desired boxes widths in pixels.
                    Range of values: floating point positive numbers.
                    Default value: 1.0
                    Required: no

    * heights       Specifies desired boxes heights in pixels.
                    Range of values: floating point positive numbers.
                    Default value: 1.0
                    Required: no

    * clip          The flag that denotes if each value in the output tensor should be clipped
                    within [0,1].
                    Range of values: {True, False}
                    Default value: True
                    Required: no

    * step_widths   The distance between box centers.
                    Range of values: floating point positive number
                    Default value: 0.0
                    Required: no

    * step_heights  The distance between box centers.
                    Range of values: floating point positive number
                    Default value: 0.0
                    Required: no

    * offset        The shift of box respectively to the top left corner.
                    Range of values: floating point positive number
                    Default value: None
                    Required: yes

    * variance      Denotes a variance of adjusting bounding boxes.
                    Range of values: floating point positive numbers
                    Default value: []
                    Required: no

    Example of attribute dictionary:

    .. code-block:: python

        # just required ones
        attrs = {
            'offset': 85,
        }

        attrs = {
            'offset': 85,
            'clip': False,
            'step_widths': [1.5, 2.0, 2.5]
        }

    Optional attributes which are absent from dictionary will be set with corresponding default.
    """
    requirements = [
        ("widths", False, np.floating, is_positive_value),
        ("heights", False, np.floating, is_positive_value),
        ("clip", False, np.bool_, None),
        ("step_widths", False, np.floating, is_positive_value),
        ("step_heights", False, np.floating, is_positive_value),
        ("offset", True, np.floating, is_positive_value),
        ("variance", False, np.floating, is_positive_value),
    ]

    check_valid_attributes("PriorBoxClustered", attrs, requirements)

    return _get_node_factory_opset1().create(
        "PriorBoxClustered",
        [output_size, as_node(image_size, name=name)],
        attrs,
    )


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

    * min_size          The minimum box size (in pixels).
                        Range of values: positive floating point numbers
                        Default value: []
                        Required: no

    * max_size          The maximum box size (in pixels).
                        Range of values: positive floating point numbers
                        Default value: []
                        Required: no

    * aspect_ratio      Aspect ratios of prior boxes.
                        Range of values: set of positive floating point numbers
                        Default value: []
                        Required: no

    * flip              The flag that denotes that each aspect_ratio is duplicated and flipped.
                        Range of values: {True, False}
                        Default value: False
                        Required: no

    * clip              The flag that denotes if each value in the output tensor should be clipped
                        to [0,1] interval.
                        Range of values: {True, False}
                        Default value: False
                        Required: no

    * step              The distance between box centers.
                        Range of values: floating point non-negative number
                        Default value: 0
                        Required: no

    * offset            This is a shift of box respectively to top left corner.
                        Range of values: floating point non-negative number
                        Default value: None
                        Required: yes

    * variance          The variance denotes a variance of adjusting bounding boxes. The attribute
                        could contain 0, 1 or 4 elements.
                        Range of values: floating point positive numbers
                        Default value: []
                        Required: no

    * scale_all_sizes   The flag that denotes type of inference.
                        Range of values: False - max_size is ignored
                                         True  - max_size is used

                        Default value: True
                        Required: no

    * fixed_ratio       This is an aspect ratio of a box.
                        Range of values: a list of positive floating-point numbers
                        Default value: None
                        Required: no

    * fixed_size        This is an initial box size (in pixels).
                        Range of values: a list of positive floating-point numbers
                        Default value: None
                        Required: no

    * density           This is the square root of the number of boxes of each type.
                        Range of values: a list of positive floating-point numbers
                        Default value: None
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
    ]

    check_valid_attributes("PriorBox", attrs, requirements)

    return _get_node_factory_opset1().create(
        "PriorBox",
        [layer_shape, as_node(image_shape, name=name)],
        attrs,
    )


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
    :param  bbox_deltas:         4D input floating point tensor with box logits.
    :param  image_shape:        The 1D input tensor with 3 or 4 elements describing image shape.
    :param  attrs:              The dictionary containing key, value pairs for attributes.
    :param  name:               Optional name for the output node.

    :return: Node representing Proposal operation.

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

    return _get_node_factory_opset1().create(
        "Proposal",
        [class_probs, bbox_deltas, as_node(image_shape, name=name)],
        attrs,
    )


@nameable_op
def psroi_pooling(
    input: NodeInput,
    coords: NodeInput,
    output_dim: int,
    group_size: int,
    spatial_scale: float,
    spatial_bins_x: int,
    spatial_bins_y: int,
    mode: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a PSROIPooling operation.

    :param input: Input feature map `{N, C, ...}`.
    :param coords: Coordinates of bounding boxes.
    :param output_dim: Output channel number.
    :param group_size: Number of groups to encode position-sensitive scores.
    :param spatial_scale: Ratio of input feature map over input image size.
    :param spatial_bins_x: Numbers of bins to divide the input feature maps over.
    :param spatial_bins_y: Numbers of bins to divide the input feature maps over.
    :param mode: Mode of pooling - "avg" or "bilinear".
    :return: PSROIPooling node
    """
    mode = mode.lower()
    return _get_node_factory_opset1().create(
        "PSROIPooling",
        as_nodes(input, coords, name=name),
        {
            "output_dim": output_dim,
            "group_size": group_size,
            "spatial_scale": spatial_scale,
            "spatial_bins_x": spatial_bins_x,
            "spatial_bins_y": spatial_bins_y,
            "mode": mode,
        },
    )


@nameable_op
def range(
    start: Node,
    stop: NodeInput,
    step: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces the Range operation.

    :param start:  The start value of the generated range.
    :param stop:   The stop value of the generated range.
    :param step:   The step value for the generated range.
    :param name:   Optional name for output node.
    :return: Range node
    """
    return _get_node_factory_opset1().create("Range", as_nodes(start, stop, step, name=name))


@unary_op
def relu(node: NodeInput, name: Optional[str] = None) -> Node:
    """Perform rectified linear unit operation on input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: The optional output node name.
    :return: The new node performing relu operation on its input element-wise.
    """
    return _get_node_factory_opset1().create("Relu", [node])


@nameable_op
def reduce_logical_and(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Logical AND reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to reduce.
    :param reduction_axes: The axes to eliminate through AND operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction.
    :param name:           Optional name for output node.
    :return: The new node performing reduction operation.
    """
    return _get_node_factory_opset1().create(
        "ReduceLogicalAnd",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_logical_or(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Logical OR reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to reduce.
    :param reduction_axes: The axes to eliminate through OR operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction.
    :param name:           Optional name for output node.
    :return: The new node performing reduction operation.
    """
    return _get_node_factory_opset1().create(
        "ReduceLogicalOr",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_max(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Max-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to max-reduce.
    :param reduction_axes: The axes to eliminate through max operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction.
    :param name: Optional name for output node.
    """
    return _get_node_factory_opset1().create(
        "ReduceMax",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_mean(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Mean-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to mean-reduce.
    :param reduction_axes: The axes to eliminate through mean operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction.
    :param name:           Optional name for output node.
    :return: The new node performing mean-reduction operation.
    """
    return _get_node_factory_opset1().create(
        "ReduceMean",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_min(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Min-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to min-reduce.
    :param reduction_axes: The axes to eliminate through min operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction
    :param name:           Optional name for output node.
    """
    return _get_node_factory_opset1().create(
        "ReduceMin",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_prod(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Product-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node:           The tensor we want to product-reduce.
    :param reduction_axes: The axes to eliminate through product operation.
    :param keep_dims:      If set to True it holds axes that are used for reduction
    :param name:           Optional name for output node.
    :return: The new node performing product-reduction operation.
    """
    return _get_node_factory_opset1().create(
        "ReduceProd",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def reduce_sum(
    node: NodeInput,
    reduction_axes: NodeInput,
    keep_dims: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Perform element-wise sums of the input tensor, eliminating the specified reduction axes.

    :param node:           The node providing data for operation.
    :param reduction_axes: The axes to eliminate through summation.
    :param keep_dims:      If set to True it holds axes that are used for reduction
    :param name:           The optional new name for output node.
    :return: The new node performing summation along `reduction_axes` element-wise.
    """
    return _get_node_factory_opset1().create(
        "ReduceSum",
        as_nodes(node, reduction_axes, name=name),
        {"keep_dims": keep_dims},
    )


@nameable_op
def region_yolo(
    input: Node,
    coords: int,
    classes: int,
    num: int,
    do_softmax: bool,
    mask: List[int],
    axis: int,
    end_axis: int,
    anchors: Optional[List[float]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces the RegionYolo operation.

    :param input:       Input data
    :param coords:      Number of coordinates for each region
    :param classes:     Number of classes for each region
    :param num:         Number of regions
    :param do_softmax:  Compute softmax
    :param mask:        Mask
    :param axis:        Axis to begin softmax on
    :param end_axis:    Axis to end softmax on
    :param anchors:     A flattened list of pairs `[width, height]` that describes prior box sizes
    :param name:        Optional name for output node.
    :return: RegionYolo node
    """
    if anchors is None:
        anchors = []

    return _get_node_factory_opset1().create(
        "RegionYolo",
        [input],
        {
            "coords": coords,
            "classes": classes,
            "num": num,
            "do_softmax": do_softmax,
            "mask": mask,
            "axis": axis,
            "end_axis": end_axis,
            "anchors": anchors,
        },
    )


@nameable_op
def reshape(
    node: NodeInput,
    output_shape: NodeInput,
    special_zero: bool,
    name: Optional[str] = None,
) -> Node:
    """Return reshaped node according to provided parameters.

    :param node: The tensor we want to reshape.
    :param output_shape: The node with a new shape for input tensor.
    :param special_zero: The boolean variable that controls how zero values in shape are
                         interpreted. If special_zero is false, then 0 is interpreted as-is
                         which means that output shape will contain a zero dimension at the
                         specified location. Input and output tensors are empty in this case.
                         If special_zero is true, then all zeros in shape implies the copying
                         of corresponding dimensions from data.shape into the output shape.
                         Range of values: False or True
    :return: The node reshaping an input tensor.
    """
    return _get_node_factory_opset1().create(
        "Reshape",
        as_nodes(node, output_shape, name=name),
        {"special_zero": special_zero},
    )


@unary_op
def result(data: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which represents an output of a graph (Model).

    :param data: The tensor containing the input data
    :return: Result node
    """
    return _get_node_factory_opset1().create("Result", [data])


@nameable_op
def reverse_sequence(
    input: NodeInput,
    seq_lengths: NodeInput,
    batch_axis: NumericData,
    seq_axis: NumericData,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces a ReverseSequence operation.

    :param input: tensor with input data to reverse
    :param seq_lengths: 1D tensor of integers with sequence lengths in the input tensor.
    :param batch_axis: index of the batch dimension.
    :param seq_axis: index of the sequence dimension.
    :return: ReverseSequence node
    """
    return _get_node_factory_opset1().create(
        "ReverseSequence",
        as_nodes(input, seq_lengths, name=name),
        {"batch_axis": batch_axis, "seq_axis": seq_axis},
    )


@nameable_op
def select(
    cond: NodeInput,
    then_node: NodeInput,
    else_node: NodeInput,
    auto_broadcast: str = "numpy",
    name: Optional[str] = None,
) -> Node:
    """Perform an element-wise selection operation on input tensors.

    :param cond: Tensor with selection mask of type `boolean`.
    :param then_node: Tensor providing data to be selected if respective `cond`
                        item value is `True`.
    :param else_node: Tensor providing data to be selected if respective `cond`
                        item value is `False`.
    :param auto_broadcast: Mode specifies rules used for auto-broadcasting of input tensors.
    :param name: The optional new name for output node.
    :return: The new node with values selected according to provided arguments.
    """
    inputs = as_nodes(cond, then_node, else_node, name=name)
    return _get_node_factory_opset1().create(
        "Select",
        inputs,
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def selu(
    data: NodeInput,
    alpha: NodeInput,
    lambda_value: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform a Scaled Exponential Linear Unit (SELU) operation on input node element-wise.

    :param data: input node, array or scalar.
    :param alpha: Alpha coefficient of SELU operation
    :param lambda_value: Lambda coefficient of SELU operation
    :param name: The optional output node name.
    :return: The new node performing relu operation on its input element-wise.
    """
    return _get_node_factory_opset1().create(
        "Selu",
        as_nodes(data, alpha, lambda_value, name=name),
    )


@nameable_op
def shape_of(data: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which produces a tensor containing the shape of its input data.

    :param data: The tensor containing the input data.
    :return: ShapeOf node
    """
    return _get_node_factory_opset1().create("ShapeOf", [as_node(data, name=name)])


@unary_op
def sigmoid(data: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which applies the sigmoid function element-wise.

    :param data: The tensor containing the input data
    :return: Sigmoid node
    """
    return _get_node_factory_opset1().create("Sigmoid", [data])


@unary_op
def sign(node: NodeInput, name: Optional[str] = None) -> Node:
    """Perform element-wise sign operation.

    :param node: One of: input node, array or scalar.
    :param name: The optional new name for output node.
    :return: The node with mapped elements of the input tensor to -1 (if it is negative),
             0 (if it is zero), or 1 (if it is positive).
    """
    return _get_node_factory_opset1().create("Sign", [node])


@unary_op
def sin(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply sine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with sin operation applied on it.
    """
    return _get_node_factory_opset1().create("Sin", [node])


@unary_op
def sinh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply hyperbolic sine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with sin operation applied on it.
    """
    return _get_node_factory_opset1().create("Sinh", [node])


@nameable_op
def softmax(data: NodeInput, axis: int, name: Optional[str] = None) -> Node:
    """Apply softmax operation on each element of input tensor.

    :param data: The tensor providing input data.
    :param axis: An axis along which Softmax should be calculated
    :return: The new node with softmax operation applied on each element.
    """
    return _get_node_factory_opset1().create("Softmax", [as_node(data, name=name)], {"axis": axis})


@nameable_op
def space_to_depth(data: Node, mode: str, block_size: int = 1, name: Optional[str] = None) -> Node:
    """Perform SpaceToDepth operation on the input tensor.

    SpaceToDepth rearranges blocks of spatial data into depth.
    The operator :return: a copy of the input tensor where values from the height
    and width dimensions are moved to the depth dimension.

    :param data: The node with data tensor.
    :param mode: Specifies how the output depth dimension is gathered from block coordinates.

                 blocks_first: The output depth is gathered from [block_size, ..., block_size, C]
                 depth_first: The output depth is gathered from [C, block_size, ..., block_size]

    :param block_size: The size of the block of values to be moved. Scalar value.
    :param name: Optional output node name.
    :return: The new node performing a SpaceToDepth operation on input tensor.
    """
    return _get_node_factory_opset1().create(
        "SpaceToDepth",
        [data],
        {"mode": mode, "block_size": block_size},
    )


@nameable_op
def split(data: NodeInput, axis: NodeInput, num_splits: int, name: Optional[str] = None) -> Node:
    """Return a node which splits the input tensor into same-length slices.

    :param data: The input tensor to be split
    :param axis: Axis along which the input data will be split
    :param num_splits: Number of the output tensors that should be produced
    :return: Split node
    """
    return _get_node_factory_opset1().create(
        "Split",
        as_nodes(data, axis, name=name),
        {"num_splits": num_splits},
    )


@unary_op
def sqrt(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies square root to the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: The new node with sqrt operation applied element-wise.
    """
    return _get_node_factory_opset1().create("Sqrt", [node])


@binary_op
def squared_difference(
    x1: NodeInput,
    x2: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    r"""Perform an element-wise squared difference between two tensors.

    \f[ y[i] = (x_1[i] - x_2[i])^2 \f]

    :param x1: The node with first input tensor.
    :param x2: The node with second input tensor.
    :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: numpy, explicit.
    :param name: Optional new name for output node.
    :return: The new node performing a squared difference between two tensors.
    """
    return _get_node_factory_opset1().create(
        "SquaredDifference",
        [x1, x2],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def squeeze(data: NodeInput, axes: NodeInput, name: Optional[str] = None) -> Node:
    """Perform squeeze operation on input tensor.

    :param data: The node with data tensor.
    :param axes: List of non-negative integers, indicate the dimensions to squeeze.
                  One of: input node or array.
    :param name: Optional new name for output node.
    :return: The new node performing a squeeze operation on input tensor.

    Remove single-dimensional entries from the shape of a tensor.
    Takes a parameter `axes` with a list of axes to squeeze.
    If `axes` is not provided, all the single dimensions will be removed from the shape.
    If an `axis` is selected with shape entry not equal to one, an error is raised.


    For example:

       Inputs: tensor with shape [1, 2, 1, 3, 1, 1], axes=[2, 4]

       Result: tensor with shape [1, 2, 3, 1]
    """
    return _get_node_factory_opset1().create("Squeeze", as_nodes(data, axes, name=name))


@nameable_op
def strided_slice(
    data: NodeInput,
    begin: NodeInput,
    end: NodeInput,
    strides: NodeInput,
    begin_mask: List[int],
    end_mask: List[int],
    new_axis_mask: Optional[List[int]] = None,
    shrink_axis_mask: Optional[List[int]] = None,
    ellipsis_mask: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which dynamically repeats(replicates) the input data tensor.

    :param      data:              The tensor to be sliced
    :param      begin:             1D tensor with begin indexes for input blob slicing
    :param      end:               1D tensor with end indexes for input blob slicing
    :param      strides:           The slicing strides
    :param      begin_mask:        A mask applied to the 'begin' input indicating which elements
                                   shoud be ignored
    :param      end_mask:          A mask applied to the 'end' input indicating which elements
                                   shoud be ignored
    :param      new_axis_mask:     A mask indicating dimensions where '1' should be inserted
    :param      shrink_axis_mask:  A mask indicating which dimensions should be deleted
    :param      ellipsis_mask:     Indicates positions where missing dimensions should be inserted
    :return:   StridedSlice node
    """
    if new_axis_mask is None:
        new_axis_mask = []
    if shrink_axis_mask is None:
        shrink_axis_mask = []
    if ellipsis_mask is None:
        ellipsis_mask = []
    attributes = {
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "ellipsis_mask": ellipsis_mask,
    }

    return _get_node_factory_opset1().create(
        "StridedSlice",
        as_nodes(data, begin, end, strides, name=name),
        attributes,
    )


@binary_op
def subtract(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which applies f(x) = A-B to the input nodes element-wise.

    :param left_node: The node providing data for left hand side of operator.
    :param right_node: The node providing data for right hand side of operator.
    :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                           to output shape axes. Range of values: numpy, explicit.
    :param name: The optional name for output node.
    :return: The new output node performing subtraction operation on both tensors element-wise.
    """
    return _get_node_factory_opset1().create(
        "Subtract",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def tan(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply tangent function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with tan operation applied on it.
    """
    return _get_node_factory_opset1().create("Tan", [node])


@unary_op
def tanh(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which applies hyperbolic tangent to the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with tanh operation applied on it.
    """
    return _get_node_factory_opset1().create("Tanh", [node])


@nameable_op
def tile(data: NodeInput, repeats: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which dynamically repeats(replicates) the input data tensor.

    :param data: The input tensor to be tiled
    :param repeats: Per-dimension replication factors
    :return: Tile node
    """
    return _get_node_factory_opset1().create("Tile", as_nodes(data, repeats, name=name))


@nameable_op
def topk(
    data: NodeInput,
    k: NodeInput,
    axis: int,
    mode: str,
    sort: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs TopK.

    :param data: Input data.
    :param k: K.
    :param axis: TopK Axis.
    :param mode: Compute TopK largest ('max') or smallest ('min')
    :param sort: Order of output elements (sort by: 'none', 'index' or 'value')
    :return: The new node which performs TopK (both indices and values)
    """
    return _get_node_factory_opset1().create(
        "TopK",
        as_nodes(data, k, name=name),
        {"axis": axis, "mode": mode, "sort": sort},
    )


@nameable_op
def transpose(data: NodeInput, input_order: NodeInput, name: Optional[str] = None) -> Node:
    """Return a node which transposes the data in the input tensor.

    :param data: The input tensor to be transposed
    :param input_order: Permutation of axes to be applied to the input tensor
    :return: Transpose node
    """
    return _get_node_factory_opset1().create("Transpose", as_nodes(data, input_order, name=name))


def unsqueeze(data: NodeInput, axes: NodeInput, name: Optional[str] = None) -> Node:
    """Perform unsqueeze operation on input tensor.

    Insert single-dimensional entries to the shape of a tensor. Takes one required argument axes,
    a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example: Inputs: tensor with shape [3, 4, 5], axes=[0, 4]
                 Result: tensor with shape [1, 3, 4, 5, 1]

    :param data: The node with data tensor.
    :param axes: List of non-negative integers, indicate the dimensions to be inserted.
                  One of: input node or array.
    :return: The new node performing an unsqueeze operation on input tensor.
    """
    return _get_node_factory_opset1().create("Unsqueeze", as_nodes(data, axes, name=name))


@nameable_op
def variadic_split(
    data: NodeInput,
    axis: NodeInput,
    split_lengths: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which splits the input tensor into variadic length slices.

    :param data: The input tensor to be split
    :param axis: Axis along which the input data will be split
    :param split_lengths: Sizes of the output tensors along the split axis
    :return: VariadicSplit node
    """
    return _get_node_factory_opset1().create(
        "VariadicSplit",
        as_nodes(data, axis, split_lengths, name=name),
    )
