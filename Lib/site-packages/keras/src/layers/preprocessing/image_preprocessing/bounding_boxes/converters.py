from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box import (  # noqa: E501
    BoundingBox,
)
from keras.src.utils import backend_utils


@keras_export("keras.utils.bounding_boxes.convert_format")
def convert_format(
    boxes, source, target, height=None, width=None, dtype="float32"
):
    """Converts bounding boxes between formats.

    Supported formats (case-insensitive):
    `"xyxy"`: [left, top, right, bottom]
    `"yxyx"`: [top, left, bottom, right]
    `"xywh"`: [left, top, width, height]
    `"center_xywh"`: [center_x, center_y, width, height]
    `"center_yxhw"`: [center_y, center_x, height, width]
    `"rel_xyxy"`, `"rel_yxyx"`, `"rel_xywh"`, `"rel_center_xywh"`:  Relative
        versions of the above formats, where coordinates are normalized
        to the range [0, 1] based on the image `height` and `width`.

    Args:
        boxes: Bounding boxes tensor/array or dictionary of `boxes` and
            `labels`.
        source: Source format string.
        target: Target format string.
        height: Image height (required for relative target format).
        width: Image width (required for relative target format).
        dtype: Data type for conversion (optional).

    Returns:
        Converted boxes.

    Raises:
        ValueError: For invalid formats, shapes, or missing dimensions.

    Example:
    ```python
    boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    # Convert from 'xyxy' to 'xywh' format
    boxes_xywh = keras.utils.bounding_boxes.convert_format(
        boxes, source='xyxy', target='xywh'
    )  # Output: [[10. 20. 20. 20.], [50. 60. 20. 20.]]

    # Convert to relative 'rel_xyxy' format
    boxes_rel_xyxy = keras.utils.bounding_boxes.convert_format(
        boxes, source='xyxy', target='rel_xyxy', height=200, width=300
    ) # Output: [[0.03333334 0.1        0.1        0.2       ],
               #[0.16666667 0.3        0.23333333 0.4       ]]
    ```
    """
    box_utils = BoundingBox()
    # Switch to tensorflow backend if we are in tf.data pipe
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    boxes = box_utils.convert_format(
        boxes=boxes,
        source=source,
        target=target,
        height=height,
        width=width,
        dtype=dtype,
    )
    # Switch back to original backend
    box_utils.backend.reset()
    return boxes


@keras_export("keras.utils.bounding_boxes.clip_to_image_size")
def clip_to_image_size(
    bounding_boxes, height=None, width=None, bounding_box_format="xyxy"
):
    """Clips bounding boxes to be within the image dimensions.
    Args:
        bounding_boxes: A dictionary with 'boxes' shape `(N, 4)` or
            `(batch, N, 4)` and 'labels' shape `(N,)` or `(batch, N,)`.
        height: Image height.
        width: Image width.
        bounding_box_format: The format of the input bounding boxes. Defaults to
            `"xyxy"`.

    Returns:
        Clipped bounding boxes.

    Example:
    ```python
    boxes = {"boxes": np.array([[-10, -20, 150, 160], [50, 40, 70, 80]]),
             "labels": np.array([0, 1])}
    clipped_boxes = keras.utils.bounding_boxes.clip_to_image_size(
        boxes, height=100, width=120,
    )
    # Output will have boxes clipped to the image boundaries, and labels
    # potentially adjusted if the clipped area becomes zero
    ```
    """

    box_utils = BoundingBox()
    # Switch to tensorflow backend if we are in tf.data pipe
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    bounding_boxes = box_utils.clip_to_image_size(
        bounding_boxes,
        height=height,
        width=width,
        bounding_box_format=bounding_box_format,
    )
    # Switch back to original backend
    box_utils.backend.reset()
    return bounding_boxes


@keras_export("keras.utils.bounding_boxes.affine_transform")
def affine_transform(
    boxes,
    angle,
    translate_x,
    translate_y,
    scale,
    shear_x,
    shear_y,
    height,
    width,
    center_x=None,
    center_y=None,
    bounding_box_format="xyxy",
):
    """Applies an affine transformation to the bounding boxes.

    The `height` and `width` parameters are used to normalize the
    translation and scaling factors.

    Args:
        boxes: The bounding boxes to transform, a tensor/array of shape
            `(N, 4)` or `(batch_size, N, 4)`.
        angle: Rotation angle in degrees.
        translate_x: Horizontal translation fraction.
        translate_y: Vertical translation fraction.
        scale: Scaling factor.
        shear_x: Shear angle in x-direction (degrees).
        shear_y: Shear angle in y-direction (degrees).
        height: Height of the image/data.
        width: Width of the image/data.
        center_x:  x-coordinate of the transformation center (fraction).
        center_y: y-coordinate of the transformation center (fraction).
        bounding_box_format: The format of the input bounding boxes. Defaults to
            `"xyxy"`.

    Returns:
        The transformed bounding boxes, a tensor/array with the same shape
        as the input `boxes`.
    """
    if bounding_box_format != "xyxy":
        raise NotImplementedError
    box_utils = BoundingBox()
    # Switch to tensorflow backend if we are in tf.data pipe
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")

    boxes = box_utils.affine(
        boxes,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
        center_x=center_x,
        center_y=center_y,
    )
    box_utils.backend.reset()
    return boxes


@keras_export("keras.utils.bounding_boxes.crop")
def crop(boxes, top, left, height, width, bounding_box_format="xyxy"):
    """Crops bounding boxes based on the given offsets and dimensions.

    This function crops bounding boxes to a specified region defined by
    `top`, `left`, `height`, and `width`. The boxes are first converted to
    `xyxy` format, cropped, and then returned.

    Args:
        boxes: The bounding boxes to crop.  A NumPy array or tensor of shape
            `(N, 4)` or `(batch_size, N, 4)`.
        top: The vertical offset of the top-left corner of the cropping region.
        left: The horizontal offset of the top-left corner of the cropping
            region.
        height: The height of the cropping region. Defaults to `None`.
        width: The width of the cropping region. Defaults to `None`.
        bounding_box_format: The format of the input bounding boxes. Defaults to
            `"xyxy"`.

    Returns:
        The cropped bounding boxes.

    Example:
    ```python
    boxes = np.array([[10, 20, 50, 60], [70, 80, 100, 120]])  # xyxy format
    cropped_boxes = keras.utils.bounding_boxes.crop(
        boxes, bounding_box_format="xyxy", top=10, left=20, height=40, width=30
    )  # Cropping a 30x40 region starting at (20, 10)
    print(cropped_boxes)
    # Expected output:
    # array([[ 0., 10., 30., 50.],
    #        [50., 70., 80., 110.]])
    """
    if bounding_box_format != "xyxy":
        raise NotImplementedError
    box_utils = BoundingBox()
    # Switch to tensorflow backend if we are in tf.data pipe
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    outputs = box_utils.crop(boxes, top, left, height, width)
    box_utils.backend.reset()
    return outputs


@keras_export("keras.utils.bounding_boxes.pad")
def pad(boxes, top, left, height=None, width=None, bounding_box_format="xyxy"):
    """Pads bounding boxes by adding top and left offsets.

    This function adds padding to the bounding boxes by increasing the 'top'
    and 'left' coordinates by the specified amounts. The method assume the
    input bounding_box_format is `xyxy`.

    Args:
        boxes: Bounding boxes to pad. Shape `(N, 4)` or `(batch, N, 4)`.
        top: Vertical padding to add.
        left: Horizontal padding to add.
        height: Image height. Defaults to None.
        width: Image width. Defaults to None.
        bounding_box_format: The format of the input bounding boxes. Defaults to
            `"xyxy"`.

    Returns:
        Padded bounding boxes in the original format.
    """
    if bounding_box_format != "xyxy":
        raise NotImplementedError
    box_utils = BoundingBox()
    # Switch to tensorflow backend if we are in tf.data pipe
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    outputs = box_utils.pad(boxes, top, left)
    box_utils.backend.reset()
    return outputs


@keras_export("keras.utils.bounding_boxes.encode_box_to_deltas")
def encode_box_to_deltas(
    anchors,
    boxes,
    anchor_format,
    box_format,
    encoding_format="center_yxhw",
    variance=None,
    image_shape=None,
):
    """Encodes bounding boxes relative to anchors as deltas.

    This function calculates the deltas that represent the difference between
    bounding boxes and provided anchors. Deltas encode the offsets and scaling
    factors to apply to anchors to obtain the target boxes.

    Boxes and anchors are first converted to the specified `encoding_format`
    (defaulting to `center_yxhw`) for consistent delta representation.

    Args:
        anchors: `Tensors`. Anchor boxes with shape of `(N, 4)` where N is the
            number of anchors.
        boxes:  `Tensors` Bounding boxes to encode. Boxes can be of shape
            `(B, N, 4)` or `(N, 4)`.
        anchor_format: str. The format of the input `anchors`
            (e.g., "xyxy", "xywh", etc.).
        box_format: str. The format of the input `boxes`
            (e.g., "xyxy", "xywh", etc.).
        encoding_format: str. The intermediate format to which boxes and anchors
            are converted before delta calculation. Defaults to "center_yxhw".
        variance: `List[float]`. A 4-element array/tensor representing variance
            factors to scale the box deltas. If provided, the calculated deltas
            are divided by the variance. Defaults to None.
        image_shape: `Tuple[int]`. The shape of the image (height, width, 3).
            When using relative bounding box format for `box_format` the
            `image_shape` is used for normalization.
    Returns:
        Encoded box deltas. The return type matches the `encode_format`.

    Raises:
        ValueError: If `variance` is not None and its length is not 4.
        ValueError: If `encoding_format` is not `"center_xywh"` or
            `"center_yxhw"`.

    """
    if variance is not None:
        variance = ops.convert_to_tensor(variance, "float32")
        var_len = variance.shape[-1]

        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")

    if encoding_format not in ["center_xywh", "center_yxhw"]:
        raise ValueError(
            "`encoding_format` should be one of 'center_xywh' or "
            f"'center_yxhw', got {encoding_format}"
        )

    if image_shape is None:
        height, width = None, None
    else:
        height, width, _ = image_shape

    encoded_anchors = convert_format(
        anchors,
        source=anchor_format,
        target=encoding_format,
        height=height,
        width=width,
    )
    boxes = convert_format(
        boxes,
        source=box_format,
        target=encoding_format,
        height=height,
        width=width,
    )
    anchor_dimensions = ops.maximum(encoded_anchors[..., 2:], backend.epsilon())
    box_dimensions = ops.maximum(boxes[..., 2:], backend.epsilon())
    # anchors be unbatched, boxes can either be batched or unbatched.
    boxes_delta = ops.concatenate(
        [
            (boxes[..., :2] - encoded_anchors[..., :2]) / anchor_dimensions,
            ops.log(box_dimensions / anchor_dimensions),
        ],
        axis=-1,
    )
    if variance is not None:
        boxes_delta /= variance
    return boxes_delta


@keras_export("keras.utils.bounding_boxes.decode_deltas_to_boxes")
def decode_deltas_to_boxes(
    anchors,
    boxes_delta,
    anchor_format,
    box_format,
    encoded_format="center_yxhw",
    variance=None,
    image_shape=None,
):
    """Converts bounding boxes from delta format to the specified `box_format`.

    This function decodes bounding box deltas relative to anchors to obtain the
    final bounding box coordinates. The boxes are encoded in a specific
    `encoded_format` (center_yxhw by default) during the decoding process.
    This allows flexibility in how the deltas are applied to the anchors.

    Args:
        anchors: Can be `Tensors` or `Dict[Tensors]` where keys are level
            indices and values are corresponding anchor boxes.
            The shape of the array/tensor should be `(N, 4)` where N is the
            number of anchors.
        boxes_delta Can be `Tensors` or `Dict[Tensors]` Bounding box deltas
            must have the same type and structure as `anchors`.  The
            shape of the array/tensor can be `(N, 4)` or `(B, N, 4)` where N is
            the number of boxes.
        anchor_format: str. The format of the input `anchors`.
            (e.g., `"xyxy"`, `"xywh"`, etc.)
        box_format: str. The desired format for the output boxes.
            (e.g., `"xyxy"`, `"xywh"`, etc.)
        encoded_format: str. Raw output format from regression head. Defaults
            to `"center_yxhw"`.
        variance: `List[floats]`. A 4-element array/tensor representing
            variance factors to scale the box deltas. If provided, the deltas
            are multiplied by the variance before being applied to the anchors.
            Defaults to None.
        image_shape: `Tuple[int]`. The shape of the image (height, width, 3).
            When using relative bounding box format for `box_format` the
            `image_shape` is used for normalization.

    Returns:
        Decoded box coordinates. The return type matches the `box_format`.

    Raises:
        ValueError: If `variance` is not None and its length is not 4.
        ValueError: If `encoded_format` is not `"center_xywh"` or
            `"center_yxhw"`.

    """
    if variance is not None:
        variance = ops.convert_to_tensor(variance, "float32")
        var_len = variance.shape[-1]

        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")

    if encoded_format not in ["center_xywh", "center_yxhw"]:
        raise ValueError(
            f"`encoded_format` should be 'center_xywh' or 'center_yxhw', "
            f"but got '{encoded_format}'."
        )

    if image_shape is None:
        height, width = None, None
    else:
        height, width, _ = image_shape

    def decode_single_level(anchor, box_delta):
        encoded_anchor = convert_format(
            anchor,
            source=anchor_format,
            target=encoded_format,
            height=height,
            width=width,
        )
        if variance is not None:
            box_delta = box_delta * variance
        # anchors be unbatched, boxes can either be batched or unbatched.
        box = ops.concatenate(
            [
                box_delta[..., :2] * encoded_anchor[..., 2:]
                + encoded_anchor[..., :2],
                ops.exp(box_delta[..., 2:]) * encoded_anchor[..., 2:],
            ],
            axis=-1,
        )
        box = convert_format(
            box,
            source=encoded_format,
            target=box_format,
            height=height,
            width=width,
        )
        return box

    if isinstance(anchors, dict) and isinstance(boxes_delta, dict):
        boxes = {}
        for lvl, anchor in anchors.items():
            boxes[lvl] = decode_single_level(anchor, boxes_delta[lvl])
        return boxes
    else:
        return decode_single_level(anchors, boxes_delta)
