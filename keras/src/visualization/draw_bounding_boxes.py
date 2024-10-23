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
    color,
    bounding_box_format,
    class_mapping=None,
    line_thickness=2,
    text_thickness=1,
    font_scale=1.0,
    data_format=None,
):
    """Utility to draw bounding boxes on the target image.

    Accepts a batch of images and batch of bounding boxes. The function draws
    the bounding boxes onto the image, and returns a new image tensor with the
    annotated images. This API is intentionally not exported, and is considered
    an implementation detail.

    Args:
        images: a batch Tensor of images to plot bounding boxes onto.
        bounding_boxes: a Tensor of batched bounding boxes to plot onto the
            provided images.
        bounding_box_format: The format of bounding boxes to plot onto the
            images. Refer
            [to the keras.io docs](TODO)
            for more details on supported bounding box formats.
        color: the color in which to plot the bounding boxes
        class_mapping: Dictionary from class ID to class label. Defaults to
            `None`.
        line_thickness: Line thicknes for the box and text labels.
            Defaults to `2`.
        text_thickness: The thickness for the text. Defaults to `1.0`.
        font_scale: Scale of font to draw in. Defaults to `1.0`.

    Returns:
        the input `images` with provided bounding boxes plotted on top of them
    """
    class_mapping = class_mapping or {}
    text_thickness = text_thickness or line_thickness
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
    if "boxes" not in bounding_boxes or "classes" not in bounding_boxes:
        raise ValueError(
            "`bounding_boxes` should be a dict containing 'boxes' and "
            f"'classes' keys. Received: bounding_boxes={bounding_boxes}"
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
    classes = ops.convert_to_numpy(bounding_boxes["labels"])
    if "confidences" in bounding_boxes:
        confidences = ops.convert_to_numpy(bounding_boxes["confidences"])
    else:
        confidences = None

    result = []
    batch_size = images.shape[0]
    for i in range(batch_size):
        _image = images[i]
        _box = boxes[i]
        _class = classes[i]
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
                    _image,
                    label,
                    (font_x1, font_y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    text_thickness,
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
