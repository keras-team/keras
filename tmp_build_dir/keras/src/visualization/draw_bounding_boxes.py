import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)

try:
    import cv2
except ImportError:
    cv2 = None


@keras_export("keras.visualization.draw_bounding_boxes")
def draw_bounding_boxes(
    images,
    bounding_boxes,
    bounding_box_format,
    class_mapping=None,
    color=(128, 128, 128),
    line_thickness=2,
    text_thickness=1,
    font_scale=1.0,
    data_format=None,
):
    """Draws bounding boxes on images.

    This function draws bounding boxes on a batch of images.  It supports
    different bounding box formats and can optionally display class labels
    and confidences.

    Args:
        images: A batch of images as a 4D tensor or NumPy array. Shape should be
            `(batch_size, height, width, channels)`.
        bounding_boxes: A dictionary containing bounding box data.  Should have
            the following keys:
            - `boxes`: A tensor or array of shape `(batch_size, num_boxes, 4)`
               containing the bounding box coordinates in the specified format.
            - `labels`: A tensor or array of shape `(batch_size, num_boxes)`
              containing the class labels for each bounding box.
            - `confidences` (Optional): A tensor or array of shape
               `(batch_size, num_boxes)` containing the confidence scores for
               each bounding box.
        bounding_box_format: A string specifying the format of the bounding
            boxes. Refer [keras-io](TODO)
        class_mapping: A dictionary mapping class IDs (integers) to class labels
            (strings).  Used to display class labels next to the bounding boxes.
            Defaults to None (no labels displayed).
        color: A tuple or list representing the RGB color of the bounding boxes.
            For example, `(255, 0, 0)` for red. Defaults to `(128, 128, 128)`.
        line_thickness: An integer specifying the thickness of the bounding box
            lines. Defaults to `2`.
        text_thickness: An integer specifying the thickness of the text labels.
            Defaults to `1`.
        font_scale: A float specifying the scale of the font used for text
            labels. Defaults to `1.0`.
        data_format: A string, either `"channels_last"` or `"channels_first"`,
            specifying the order of dimensions in the input images. Defaults to
            the `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            "channels_last".

    Returns:
        A NumPy array of the annotated images with the bounding boxes drawn.
        The array will have the same shape as the input `images`.

    Raises:
        ValueError: If `images` is not a 4D tensor/array, if `bounding_boxes` is
            not a dictionary, or if `bounding_boxes` does not contain `"boxes"`
            and `"labels"` keys.
        TypeError: If `bounding_boxes` is not a dictionary.
        ImportError: If `cv2` (OpenCV) is not installed.
    """

    if cv2 is None:
        raise ImportError(
            "The `draw_bounding_boxes` function requires the `cv2` package "
            " (OpenCV). Please install it with `pip install opencv-python`."
        )

    class_mapping = class_mapping or {}
    text_thickness = (
        text_thickness or line_thickness
    )  # Default text_thickness if not provided.
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if not isinstance(bounding_boxes, dict):
        raise TypeError(
            "`bounding_boxes` should be a dict. "
            f"Received: bounding_boxes={bounding_boxes} of type "
            f"{type(bounding_boxes)}"
        )
    if "boxes" not in bounding_boxes or "labels" not in bounding_boxes:
        raise ValueError(
            "`bounding_boxes` should be a dict containing 'boxes' and "
            f"'labels' keys. Received: bounding_boxes={bounding_boxes}"
        )
    if data_format == "channels_last":
        h_axis = -3
        w_axis = -2
    else:
        h_axis = -2
        w_axis = -1
    height = images_shape[h_axis]
    width = images_shape[w_axis]
    bounding_boxes = bounding_boxes.copy()
    bounding_boxes = convert_format(
        bounding_boxes, bounding_box_format, "xyxy", height, width
    )

    # To numpy array
    images = ops.convert_to_numpy(images).astype("uint8")
    boxes = ops.convert_to_numpy(bounding_boxes["boxes"])
    labels = ops.convert_to_numpy(bounding_boxes["labels"])
    if "confidences" in bounding_boxes:
        confidences = ops.convert_to_numpy(bounding_boxes["confidences"])
    else:
        confidences = None

    result = []
    batch_size = images.shape[0]
    for i in range(batch_size):
        _image = images[i]
        _box = boxes[i]
        _class = labels[i]
        for box_i in range(_box.shape[0]):
            x1, y1, x2, y2 = _box[box_i].astype("int32")
            c = _class[box_i].astype("int32")
            if c == -1:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            c = int(c)
            # Draw bounding box
            cv2.rectangle(_image, (x1, y1), (x2, y2), color, line_thickness)

            if c in class_mapping:
                label = class_mapping[c]
                if confidences is not None:
                    conf = confidences[i][box_i]
                    label = f"{label} | {conf:.2f}"

                font_x1, font_y1 = _find_text_location(
                    x1, y1, font_scale, text_thickness
                )
                cv2.putText(
                    img=_image,
                    text=label,
                    org=(font_x1, font_y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color,
                    thickness=text_thickness,
                )
        result.append(_image)
    return np.stack(result, axis=0)


def _find_text_location(x, y, font_scale, thickness):
    font_height = int(font_scale * 12)
    target_y = y - 8
    if target_y - (2 * font_height) > 0:
        return x, y - 8

    line_offset = thickness
    static_offset = 3

    return (
        x + static_offset,
        y + (2 * font_height) + line_offset + static_offset,
    )
