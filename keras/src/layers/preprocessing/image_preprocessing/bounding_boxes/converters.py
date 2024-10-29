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
    and 'left' coordinates by the specified amounts. The boxes are first
    converted to `"xyxy"` format, padded, and then converted back to the
    original format.

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
