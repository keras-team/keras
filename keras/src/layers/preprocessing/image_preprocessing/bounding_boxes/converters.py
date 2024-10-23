from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box import (  # noqa: E501
    BoundingBox,
)
from keras.src.utils import backend_utils


@keras_export("keras.utils.bounding_boxes.convert_format")
def convert_format(
    boxes, source, target, height=None, width=None, dtype="float32"
):
    # Switch to tensorflow backend if we are in tf.data pipe
    box_utils = BoundingBox()
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
def clip_to_image_size(bounding_boxes, height=None, width=None, format="xyxy"):
    # Switch to tensorflow backend if we are in tf.data pipe

    box_utils = BoundingBox()
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    bounding_boxes = box_utils.clip_to_image_size(
        bounding_boxes, height=height, width=width, format=format
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
    format="xyxy",
):
    if format != "xyxy":
        raise NotImplementedError
    # Switch to tensorflow backend if we are in tf.data pipe
    box_utils = BoundingBox()
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    outputs = box_utils.affine(
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
    return outputs


@keras_export("keras.utils.bounding_boxes.crop")
def crop(boxes, top, left, height, width, format="xyxy"):
    if format != "xyxy":
        raise NotImplementedError
    box_utils = BoundingBox()
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")
    outputs = box_utils.crop(boxes, top, left, height, width)
    box_utils.backend.reset()
    return outputs


@keras_export("keras.utils.bounding_boxes.pad")
def pad(boxes, top, left, format="xyxy"):
    if format != "xyxy":
        raise NotImplementedError
    box_utils = BoundingBox()
    if backend_utils.in_tf_graph():
        box_utils.backend.set_backend("tensorflow")

    outputs = box_utils.pad(boxes, top, left)
    box_utils.backend.reset()
    return outputs
