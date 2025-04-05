# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from functools import partial
from typing import Optional

import numpy as np
from openvino import Node
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import nameable_op
from openvino.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
    make_constant_node,
)

_get_node_factory_opset9 = partial(_get_node_factory, "opset9")


# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def eye(
    num_rows: NodeInput,
    num_columns: NodeInput,
    diagonal_index: NodeInput,
    output_type: str,
    batch_shape: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs eye operation.

    :param num_rows: The node providing row number tensor.
    :param num_columns: The node providing column number tensor.
    :param diagonal_index: The node providing the index of the diagonal to be populated.
    :param output_type: Specifies the output tensor type, supports any numeric types.
    :param batch_shape: The node providing the leading batch dimensions of output shape. Optionally.
    :param name: The optional new name for output node.
    :return: New node performing deformable convolution operation.
    """
    if batch_shape is not None:
        inputs = as_nodes(num_rows, num_columns, diagonal_index, batch_shape, name=name)
    else:
        inputs = as_nodes(num_rows, num_columns, diagonal_index, name=name)

    return _get_node_factory_opset9().create("Eye", inputs, {"output_type": output_type})


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    soft_nms_sigma: Optional[NodeInput] = None,
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
    :param soft_nms_sigma: Tensor specifying the sigma parameter for Soft-NMS.
    :param box_encoding: Format of boxes data encoding.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :return: The new node which performs NonMaxSuppression
    """
    max_output_boxes_per_class = max_output_boxes_per_class if max_output_boxes_per_class is not None else make_constant_node(0, np.int64)
    iou_threshold = iou_threshold if iou_threshold is not None else make_constant_node(0, np.float32)
    score_threshold = score_threshold if score_threshold is not None else make_constant_node(0, np.float32)
    soft_nms_sigma = soft_nms_sigma if soft_nms_sigma is not None else make_constant_node(0, np.float32)

    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma, name=name)

    attributes = {
        "box_encoding": box_encoding,
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
    }

    return _get_node_factory_opset9().create("NonMaxSuppression", inputs, attributes)


def roi_align(
    data: NodeInput,
    rois: NodeInput,
    batch_indices: NodeInput,
    pooled_h: int,
    pooled_w: int,
    sampling_ratio: int,
    spatial_scale: float,
    mode: str,
    aligned_mode: Optional[str] = "asymmetric",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ROIAlign operation.

    :param data: Input data.
    :param rois: RoIs (Regions of Interest) to pool over.
    :param batch_indices: Tensor with each element denoting the index of
                          the corresponding image in the batch.
    :param pooled_h: Height of the ROI output feature map.
    :param pooled_w: Width of the ROI output feature map.
    :param sampling_ratio: Number of bins over height and width to use to calculate
                           each output feature map element.
    :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
    :param mode: Method to perform pooling to produce output feature map elements. Avaiable modes are:
                         - 'max' - maximum pooling
                         - 'avg' - average pooling
    :param aligned_mode: Specifies how to transform the coordinate in original tensor to the resized tensor.
                         Mode 'asymmetric' is the default value. Optional. Avaiable aligned modes are:
                         - 'asymmetric'
                         - 'half_pixel_for_nn'
                         - 'half_pixel'
    :param name: The optional name for the output node

    :return: The new node which performs ROIAlign
    """
    inputs = as_nodes(data, rois, batch_indices, name=name)
    attributes = {
        "pooled_h": pooled_h,
        "pooled_w": pooled_w,
        "sampling_ratio": sampling_ratio,
        "spatial_scale": spatial_scale,
        "mode": mode,
        "aligned_mode": aligned_mode,
    }
    return _get_node_factory_opset9().create("ROIAlign", inputs, attributes)


def softsign(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply SoftSign operation on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: The optional name for the output node.
    :return: New node with SoftSign operation applied on each element of it.
    """
    return _get_node_factory_opset9().create("SoftSign", [as_node(node, name=name)], {})


@nameable_op
def rdft(
    data: NodeInput,
    axes: NodeInput,
    signal_size: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs RDFT operation.

    :param data: Tensor with data.
    :param axes: Tensor with axes to transform.
    :param signal_size: Optional tensor specifying signal size with respect to axes from the input 'axes'.
    :param name: Optional output node name.
    :return: The new node which performs RDFT operation on the input data tensor.
    """
    if signal_size is None:
        inputs = as_nodes(data, axes, name=name)
    else:
        inputs = as_nodes(data, axes, signal_size, name=name)

    return _get_node_factory_opset9().create("RDFT", inputs)


@nameable_op
def irdft(
    data: NodeInput,
    axes: NodeInput,
    signal_size: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs IRDFT operation.

    :param data: Tensor with data.
    :param axes: Tensor with axes to transform.
    :param signal_size: Optional tensor specifying signal size with respect to axes from the input 'axes'.
    :param name: Optional output node name.
    :return: The new node which performs IRDFT operation on the input data tensor.
    """
    if signal_size is None:
        inputs = as_nodes(data, axes, name=name)
    else:
        inputs = as_nodes(data, axes, signal_size, name=name)

    return _get_node_factory_opset9().create("IRDFT", inputs)


@nameable_op
def multiclass_nms(
    boxes: NodeInput,
    scores: NodeInput,
    roisnum: Optional[NodeInput] = None,
    sort_result_type: Optional[str] = "none",
    sort_result_across_batch: Optional[bool] = False,
    output_type: Optional[str] = "i64",
    iou_threshold: Optional[float] = 0.0,
    score_threshold: Optional[float] = 0.0,
    nms_top_k: Optional[int] = -1,
    keep_top_k: Optional[int] = -1,
    background_class: Optional[int] = -1,
    nms_eta: Optional[float] = 1.0,
    normalized: Optional[bool] = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MulticlassNms.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param roisnum: Tensor with roisnum. Specifies the number of rois in each image. Required when
                    'scores' is a 2-dimensional tensor.
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
    :param name: The optional name for the output node
    :return: The new node which performs MuticlassNms
    """
    if roisnum is None:
        inputs = as_nodes(boxes, scores, name=name)
    else:
        inputs = as_nodes(boxes, scores, roisnum, name=name)

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

    return _get_node_factory_opset9().create("MulticlassNms", inputs, attributes)


@nameable_op
def generate_proposals(
    im_info: NodeInput,
    anchors: NodeInput,
    deltas: NodeInput,
    scores: NodeInput,
    min_size: float,
    nms_threshold: float,
    pre_nms_count: int,
    post_nms_count: int,
    normalized: bool = True,
    nms_eta: float = 1.0,
    roi_num_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GenerateProposals operation.

    :param im_info: Input with image info.
    :param anchors: Input anchors.
    :param deltas: Input deltas.
    :param scores: Input scores.
    :param min_size: Specifies minimum box width and height.
    :param nms_threshold: Specifies threshold to be used in the NMS stage.
    :param pre_nms_count: Specifies number of top-n proposals before NMS.
    :param post_nms_count: Specifies number of top-n proposals after NMS.
    :param normalized: Specifies whether proposal bboxes are normalized or not. Optional attribute, default value is `True`.
    :param nms_eta: Specifies eta parameter for adaptive NMS., must be in range `[0.0, 1.0]`. Optional attribute, default value is `1.0`.
    :param roi_num_type: Specifies the element type of the third output `rpnroisnum`. Optional attribute, range of values: `i64` (default) or `i32`.
    :param name: The optional name for the output node.
    :return: New node performing GenerateProposals operation.
    """
    inputs = as_nodes(im_info, anchors, deltas, scores, name=name)

    attributes = {
        "min_size": min_size,
        "nms_threshold": nms_threshold,
        "pre_nms_count": pre_nms_count,
        "post_nms_count": post_nms_count,
        "normalized": normalized,
        "nms_eta": nms_eta,
        "roi_num_type": roi_num_type,
    }

    return _get_node_factory_opset9().create("GenerateProposals", inputs, attributes)


def grid_sample(
    data: NodeInput,
    grid: NodeInput,
    attributes: dict,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GridSample operation.

    :param data: The input image.
    :param grid: Grid values (normalized input coordinates).
    :param attributes: A dictionary containing GridSample's attributes.
    :param name: Optional name of the node.

    Available attributes:

    * align_corners A flag which specifies whether to align the grid extrema values
                    with the borders or center points of the input tensor's border pixels.
                    Range of values: true, false
                    Default value: false
                    Required: no
    * mode          Specifies the type of interpolation.
                    Range of values: bilinear, bicubic, nearest
                    Default value: bilinear
                    Required: no
    * padding_mode  Specifies how the out-of-bounds coordinates should be handled.
                    Range of values: zeros, border, reflection
                    Default value: zeros
                    Required: no

    :return: A new GridSample node.
    """
    return _get_node_factory_opset9().create("GridSample", as_nodes(data, grid, name=name), attributes)
