"""Converter functions for working with bounding box formats."""

from keras.src import ops
from keras.src.utils import tf_utils


# Internal exception to propagate the fact images was not passed to a converter
# that needs it.
class RequiresImagesException(Exception):
    pass


ALL_AXES = 4


def _center_yxhw_to_xyxy(boxes, images=None, image_shape=None):
    y, x, height, width = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
        axis=-1,
    )


def _center_xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
        axis=-1,
    )


def _xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([x, y, x + width, y + height], axis=-1)


def _xyxy_to_center_yxhw(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            (top + bottom) / 2.0,
            (left + right) / 2.0,
            bottom - top,
            right - left,
        ],
        axis=-1,
    )


def _rel_xywh_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            image_width * x,
            image_height * y,
            image_width * (x + width),
            image_height * (y + height),
        ],
        axis=-1,
    )


def _xyxy_no_op(boxes, images=None, image_shape=None):
    return boxes


def _xyxy_to_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [left, top, right - left, bottom - top],
        axis=-1,
    )


def _xyxy_to_rel_xywh(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    left, right = (
        left / image_width,
        right / image_width,
    )
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [left, top, right - left, bottom - top],
        axis=-1,
    )


def _xyxy_to_center_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            (left + right) / 2.0,
            (top + bottom) / 2.0,
            right - left,
            bottom - top,
        ],
        axis=-1,
    )


def _rel_xyxy_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )


def _xyxy_to_rel_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )


def _yxyx_to_xyxy(boxes, images=None, image_shape=None):
    y1, x1, y2, x2 = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([x1, y1, x2, y2], axis=-1)


def _rel_yxyx_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    top, left, bottom, right = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )


def _xyxy_to_yxyx(boxes, images=None, image_shape=None):
    x1, y1, x2, y2 = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([y1, x1, y2, x2], axis=-1)


def _xyxy_to_rel_yxyx(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [top, left, bottom, right],
        axis=-1,
    )


TO_XYXY_CONVERTERS = {
    "xywh": _xywh_to_xyxy,
    "center_xywh": _center_xywh_to_xyxy,
    "center_yxhw": _center_yxhw_to_xyxy,
    "rel_xywh": _rel_xywh_to_xyxy,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _rel_xyxy_to_xyxy,
    "yxyx": _yxyx_to_xyxy,
    "rel_yxyx": _rel_yxyx_to_xyxy,
}

FROM_XYXY_CONVERTERS = {
    "xywh": _xyxy_to_xywh,
    "center_xywh": _xyxy_to_center_xywh,
    "center_yxhw": _xyxy_to_center_yxhw,
    "rel_xywh": _xyxy_to_rel_xywh,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _xyxy_to_rel_xyxy,
    "yxyx": _xyxy_to_yxyx,
    "rel_yxyx": _xyxy_to_rel_yxyx,
}


def convert_format(
    boxes, source, target, images=None, image_shape=None, dtype="float32"
):
    f"""Converts bounding_boxes from one format to another.

    Supported formats are:

    - `"xyxy"`, also known as `corners` format. In this format the first four
        axes represent `[left, top, right, bottom]` in that order.
    - `"rel_xyxy"`. In this format, the axes are the same as `"xyxy"` but the x
        coordinates are normalized using the image width, and the y axes the
        image height. All values in `rel_xyxy` are in the range `(0, 1)`.
    - `"xywh"`. In this format the first four axes represent
        `[left, top, width, height]`.
    - `"rel_xywh". In this format the first four axes represent
        [left, top, width, height], just like `"xywh"`. Unlike `"xywh"`, the
        values are in the range (0, 1) instead of absolute pixel values.
    - `"center_xyWH"`. In this format the first two coordinates represent the x
        and y coordinates of the center of the bounding box, while the last two
        represent the width and height of the bounding box.
    - `"center_yxHW"`. In this format the first two coordinates represent the y
        and x coordinates of the center of the bounding box, while the last two
        represent the height and width of the bounding box.
    - `"yxyx"`. In this format the first four axes represent
        [top, left, bottom, right] in that order.
    - `"rel_yxyx"`. In this format, the axes are the same as `"yxyx"` but the x
        coordinates are normalized using the image width, and the y axes the
        image height. All values in `rel_yxyx` are in the range (0, 1).
    Formats are case insensitive. It is recommended that you capitalize width
    and height to maximize the visual difference between `"xyWH"` and `"xyxy"`.

    Relative formats, abbreviated `rel`, make use of the shapes of the `images`
    passed. In these formats, the coordinates, widths, and heights are all
    specified as percentages of the host image.

    Example:

    ```python
    boxes = {
        "boxes": [TODO],
        "labels": [TODO],
    }
    boxes_in_xywh = keras.utils.bounding_boxes.convert_format(
        boxes,
        source='xyxy',
        target='xyWH'
    )
    ```

    Args:
        boxes: tensor representing bounding boxes in the format specified in
            the `source` parameter. `boxes` can optionally have extra
            dimensions stacked on the final axis to store metadata. boxes
            should be a 3D tensor, with the shape `[batch_size, num_boxes, 4]`.
            Alternatively, boxes can be a dictionary with key 'boxes' containing
            a tensor matching the aforementioned spec.
        source:One of {" ".join([f'"{f}"' for f in TO_XYXY_CONVERTERS.keys()])}.
            Used to specify the original format of the `boxes` parameter.
        target:One of {" ".join([f'"{f}"' for f in TO_XYXY_CONVERTERS.keys()])}.
            Used to specify the destination format of the `boxes` parameter.
        images: (Optional) a batch of images aligned with `boxes` on the first
            axis. Should be at least 3 dimensions, with the first 3 dimensions
            representing: `[batch_size, height, width]`. Used in some
            converters to compute relative pixel values of the bounding box
            dimensions. Required when transforming from a rel format to a
            non-rel format.
        dtype: the data type to use when transforming the boxes, defaults to
            `"float32"`.
    """
    if isinstance(boxes, dict):
        converted_boxes = boxes.copy()
        converted_boxes["boxes"] = convert_format(
            boxes["boxes"],
            source=source,
            target=target,
            images=images,
            image_shape=image_shape,
            dtype=dtype,
        )
        return converted_boxes

    if boxes.shape[-1] is not None and boxes.shape[-1] != 4:
        raise ValueError(
            "Expected `boxes` to be a Tensor with a final dimension of "
            f"`4`. Instead, got `boxes.shape={boxes.shape}`."
        )
    if images is not None and image_shape is not None:
        raise ValueError(
            "convert_format() expects either `images` or `image_shape`, but "
            f"not both. Received images={images} image_shape={image_shape}"
        )

    _validate_image_shape(image_shape)

    source = source.lower()
    target = target.lower()
    if source not in TO_XYXY_CONVERTERS:
        raise ValueError(
            "`convert_format()` received an unsupported format for the "
            "argument `source`. `source` should be one of "
            f"{TO_XYXY_CONVERTERS.keys()}. Got source={source}"
        )
    if target not in FROM_XYXY_CONVERTERS:
        raise ValueError(
            "`convert_format()` received an unsupported format for the "
            "argument `target`. `target` should be one of "
            f"{FROM_XYXY_CONVERTERS.keys()}. Got target={target}"
        )

    boxes = ops.cast(boxes, dtype)
    if source == target:
        return boxes

    # rel->rel conversions should not require images
    if source.startswith("rel") and target.startswith("rel"):
        source = source.replace("rel_", "", 1)
        target = target.replace("rel_", "", 1)

    boxes, images, squeeze = _format_inputs(boxes, images)
    to_xyxy_fn = TO_XYXY_CONVERTERS[source]
    from_xyxy_fn = FROM_XYXY_CONVERTERS[target]

    try:
        in_xyxy = to_xyxy_fn(boxes, images=images, image_shape=image_shape)
        result = from_xyxy_fn(in_xyxy, images=images, image_shape=image_shape)
    except RequiresImagesException:
        raise ValueError(
            "convert_format() must receive `images` or `image_shape` when "
            "transforming between relative and absolute formats."
            f"convert_format() received source=`{format}`, target=`{format}, "
            f"but images={images} and image_shape={image_shape}."
        )

    return _format_outputs(result, squeeze)


def _format_inputs(boxes, images):
    boxes_rank = len(boxes.shape)
    if boxes_rank > 3:
        raise ValueError(
            "Expected len(boxes.shape)=2, or len(boxes.shape)=3, got "
            f"len(boxes.shape)={boxes_rank}"
        )
    boxes_includes_batch = boxes_rank == 3
    # Determine if images needs an expand_dims() call
    if images is not None:
        images_rank = len(images.shape)
        if images_rank > 4:
            raise ValueError(
                "Expected len(images.shape)=2, or len(images.shape)=3, got "
                f"len(images.shape)={images_rank}"
            )
        images_include_batch = images_rank == 4
        if boxes_includes_batch != images_include_batch:
            raise ValueError(
                "convert_format() expects both boxes and images to be batched, "
                "or both boxes and images to be unbatched. Received "
                f"len(boxes.shape)={boxes_rank}, "
                f"len(images.shape)={images_rank}. Expected either "
                "len(boxes.shape)=2 AND len(images.shape)=3, or "
                "len(boxes.shape)=3 AND len(images.shape)=4."
            )
        if not images_include_batch:
            images = ops.expand_dims(images, axis=0)

    if not boxes_includes_batch:
        return ops.expand_dims(boxes, axis=0), images, True
    return boxes, images, False


def _validate_image_shape(image_shape):
    # Escape early if image_shape is None and skip validation.
    if image_shape is None:
        return
    # tuple/list
    if isinstance(image_shape, (tuple, list)):
        if len(image_shape) != 3:
            raise ValueError(
                "image_shape should be of length 3, but got "
                f"image_shape={image_shape}"
            )
        return

    # tensor
    if ops.is_tensor(image_shape):
        if len(image_shape.shape) > 1:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        if image_shape.shape[0] != 3:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        return

    # Warn about failure cases
    raise ValueError(
        "Expected image_shape to be either a tuple, list, Tensor. "
        f"Received image_shape={image_shape}"
    )


def _format_outputs(boxes, squeeze):
    if squeeze:
        return ops.squeeze(boxes, axis=0)
    return boxes


def _image_shape(images, image_shape, boxes):
    if images is None and image_shape is None:
        raise RequiresImagesException()

    if image_shape is None:
        if not tf_utils.is_ragged_tensor(images):
            image_shape = ops.shape(images)
            height, width = image_shape[1], image_shape[2]
        else:
            height = ops.reshape(images.row_lengths(), (-1, 1))
            width = ops.reshape(ops.max(images.row_lengths(axis=2), 1), (-1, 1))
            height = ops.expand_dims(height, axis=-1)
            width = ops.expand_dims(width, axis=-1)
    else:
        height, width = image_shape[0], image_shape[1]
    return ops.cast(height, boxes.dtype), ops.cast(width, boxes.dtype)
