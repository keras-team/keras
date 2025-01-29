"""Contains functions to compute ious of bounding boxes."""

import math

import keras
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes import (
    converters,
)


def _compute_area(box):
    """Computes area for bounding boxes

    Args:
      box: [N, 4] or [batch_size, N, 4] float Tensor, either batched
        or unbatched boxes.
    Returns:
      a float Tensor of [N] or [batch_size, N]
    """
    y_min, x_min, y_max, x_max = ops.split(box[..., :4], 4, axis=-1)
    return ops.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def _compute_intersection(boxes1, boxes2):
    """Computes intersection area between two sets of boxes.

    Args:
      boxes1: [N, 4] or [batch_size, N, 4] float Tensor boxes.
      boxes2: [M, 4] or [batch_size, M, 4] float Tensor boxes.
    Returns:
      a [N, M] or [batch_size, N, M] float Tensor.
    """
    y_min1, x_min1, y_max1, x_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = ops.split(boxes2[..., :4], 4, axis=-1)
    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    # [N, M] or [batch_size, N, M]
    intersect_ymax = ops.minimum(y_max1, ops.transpose(y_max2, perm))
    intersect_ymin = ops.maximum(y_min1, ops.transpose(y_min2, perm))
    intersect_xmax = ops.minimum(x_max1, ops.transpose(x_max2, perm))
    intersect_xmin = ops.maximum(x_min1, ops.transpose(x_min2, perm))

    intersect_height = intersect_ymax - intersect_ymin
    intersect_width = intersect_xmax - intersect_xmin
    zeros_t = ops.cast(0, intersect_height.dtype)
    intersect_height = ops.maximum(zeros_t, intersect_height)
    intersect_width = ops.maximum(zeros_t, intersect_width)

    return intersect_height * intersect_width


@keras_export("keras.utils.bounding_boxes.compute_iou")
def compute_iou(
    boxes1,
    boxes2,
    bounding_box_format,
    use_masking=False,
    mask_val=-1,
    image_shape=None,
):
    """Computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if
    boxes are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the
    boxes are batched.

    The users can pass `boxes1` and `boxes2` to be different ranks. For example:
    1) `boxes1`: [batch_size, M, 4], `boxes2`: [batch_size, N, 4] -> return
        [batch_size, M, N].
    2) `boxes1`: [batch_size, M, 4], `boxes2`: [N, 4] -> return
        [batch_size, M, N]
    3) `boxes1`: [M, 4], `boxes2`: [batch_size, N, 4] -> return
        [batch_size, M, N]
    4) `boxes1`: [M, 4], `boxes2`: [N, 4] -> return [M, N]

    Args:
        boxes1: a list of bounding boxes in 'corners' format. Can be batched or
            unbatched.
        boxes2: a list of bounding boxes in 'corners' format. Can be batched or
            unbatched.
        bounding_box_format: a case-insensitive string which is one of `"xyxy"`,
            `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`.
            For detailed information on the supported format, see the
        use_masking: whether masking will be applied. This will mask all
            `boxes1` or `boxes2` that have values less than 0 in all its 4
            dimensions. Default to `False`.
        mask_val: int to mask those returned IOUs if the masking is True,
            defaults to -1.
        image_shape: `Tuple[int]`. The shape of the image (height, width, 3).
            When using relative bounding box format for `box_format` the
            `image_shape` is used for normalization.

    Returns:
        iou_lookup_table: a vector containing the pairwise ious of boxes1 and
            boxes2.
    """  # noqa: E501

    boxes1_rank = len(ops.shape(boxes1))
    boxes2_rank = len(ops.shape(boxes2))

    if boxes1_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes1 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes1.shape)=2 AND or len(boxes1.shape)=3."
        )
    if boxes2_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes2 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes2.shape)=2 AND or len(boxes2.shape)=3."
        )

    target_format = "yxyx"
    if "rel" in bounding_box_format and image_shape is None:
        raise ValueError(
            "When using relative bounding box formats (e.g. `rel_yxyx`) "
            "the `image_shape` argument must be provided."
            f"Received `image_shape`: {image_shape}"
        )

    if image_shape is None:
        height, width = None, None
    else:
        height, width, _ = image_shape

    boxes1 = converters.convert_format(
        boxes1,
        source=bounding_box_format,
        target=target_format,
        height=height,
        width=width,
    )

    boxes2 = converters.convert_format(
        boxes2,
        source=bounding_box_format,
        target=target_format,
        height=height,
        width=width,
    )

    intersect_area = _compute_intersection(boxes1, boxes2)
    boxes1_area = _compute_area(boxes1)
    boxes2_area = _compute_area(boxes2)
    boxes2_area_rank = len(boxes2_area.shape)
    boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
    boxes1_area = ops.expand_dims(boxes1_area, axis=-1)
    boxes2_area = ops.expand_dims(boxes2_area, axis=boxes2_axis)
    union_area = boxes1_area + boxes2_area - intersect_area
    res = ops.divide(intersect_area, union_area + backend.epsilon())

    if boxes1_rank == 2:
        perm = [1, 0]
    else:
        perm = [0, 2, 1]

    if not use_masking:
        return res

    mask_val_t = ops.cast(mask_val, res.dtype) * ops.ones_like(res)
    boxes1_mask = ops.less(ops.max(boxes1, axis=-1, keepdims=True), 0.0)
    boxes2_mask = ops.less(ops.max(boxes2, axis=-1, keepdims=True), 0.0)
    background_mask = ops.logical_or(
        boxes1_mask, ops.transpose(boxes2_mask, perm)
    )
    iou_lookup_table = ops.where(background_mask, mask_val_t, res)
    return iou_lookup_table


@keras_export("keras.utils.bounding_boxes.compute_ciou")
def compute_ciou(boxes1, boxes2, bounding_box_format, image_shape=None):
    """
    Computes the Complete IoU (CIoU) between two bounding boxes or between
    two batches of bounding boxes.

    CIoU loss is an extension of GIoU loss, which further improves the IoU
    optimization for object detection. CIoU loss not only penalizes the
    bounding box coordinates but also considers the aspect ratio and center
    distance of the boxes. The length of the last dimension should be 4 to
    represent the bounding boxes.

    Args:
        box1 (tensor): tensor representing the first bounding box with
            shape (..., 4).
        box2 (tensor): tensor representing the second bounding box with
            shape (..., 4).
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values. For detailed
            information on the supported formats, see the [KerasCV bounding box
            documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        image_shape: `Tuple[int]`. The shape of the image (height, width, 3).
            When using relative bounding box format for `box_format` the
            `image_shape` is used for normalization.

    Returns:
        tensor: The CIoU distance between the two bounding boxes.
    """
    target_format = "xyxy"
    if "rel" in bounding_box_format:
        raise ValueError(
            "When using relative bounding box formats (e.g. `rel_yxyx`) "
            "the `image_shape` argument must be provided."
            f"Received `image_shape`: {image_shape}"
        )

    if image_shape is None:
        height, width = None, None
    else:
        height, width, _ = image_shape

    boxes1 = converters.convert_format(
        boxes1,
        source=bounding_box_format,
        target=target_format,
        height=height,
        width=width,
    )

    boxes2 = converters.convert_format(
        boxes2,
        source=bounding_box_format,
        target=target_format,
        height=height,
        width=width,
    )

    x_min1, y_min1, x_max1, y_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2 = ops.split(boxes2[..., :4], 4, axis=-1)

    width_1 = x_max1 - x_min1
    height_1 = y_max1 - y_min1 + keras.backend.epsilon()
    width_2 = x_max2 - x_min2
    height_2 = y_max2 - y_min2 + keras.backend.epsilon()

    intersection_area = ops.maximum(
        ops.minimum(x_max1, x_max2) - ops.maximum(x_min1, x_min2), 0
    ) * ops.maximum(
        ops.minimum(y_max1, y_max2) - ops.maximum(y_min1, y_min2), 0
    )
    union_area = (
        width_1 * height_1
        + width_2 * height_2
        - intersection_area
        + keras.backend.epsilon()
    )
    iou = ops.squeeze(
        ops.divide(intersection_area, union_area + keras.backend.epsilon()),
        axis=-1,
    )

    convex_width = ops.maximum(x_max1, x_max2) - ops.minimum(x_min1, x_min2)
    convex_height = ops.maximum(y_max1, y_max2) - ops.minimum(y_min1, y_min2)
    convex_diagonal_squared = ops.squeeze(
        convex_width**2 + convex_height**2 + keras.backend.epsilon(),
        axis=-1,
    )
    centers_distance_squared = ops.squeeze(
        ((x_min1 + x_max1) / 2 - (x_min2 + x_max2) / 2) ** 2
        + ((y_min1 + y_max1) / 2 - (y_min2 + y_max2) / 2) ** 2,
        axis=-1,
    )

    v = ops.squeeze(
        (4 / math.pi**2)
        * ops.power(
            (ops.arctan(width_2 / height_2) - ops.arctan(width_1 / height_1)),
            2,
        ),
        axis=-1,
    )
    alpha = v / (v - iou + (1 + keras.backend.epsilon()))

    return iou - (
        centers_distance_squared / convex_diagonal_squared + v * alpha
    )
