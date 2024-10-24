import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export


@keras_export("keras.visualization.draw_segmentation_masks")
def draw_segmentation_masks(
    images,
    segmentation_masks,
    num_classes=None,
    color_mapping=None,
    alpha=0.8,
    ignore_index=-1,
    data_format=None,
):
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    images = ops.convert_to_tensor(images, dtype="float32")
    segmentation_masks = ops.convert_to_tensor(segmentation_masks)

    if not backend.is_int_dtype(segmentation_masks.dtype):
        dtype = backend.standardize_dtype(segmentation_masks.dtype)
        raise TypeError(
            "`segmentation_masks` must be in integer dtype. "
            f"Received: segmentation_masks.dtype={dtype}"
        )

    # Infer num_classes
    if num_classes is None:
        num_classes = int(ops.convert_to_numpy(ops.max(segmentation_masks)))
    if color_mapping is None:
        colors = _generate_color_palette(num_classes)
    else:
        colors = [color_mapping[i] for i in range(num_classes)]
    valid_masks = ops.not_equal(segmentation_masks, ignore_index)
    valid_masks = ops.squeeze(valid_masks, axis=-1)
    segmentation_masks = ops.one_hot(segmentation_masks, num_classes)
    segmentation_masks = segmentation_masks[..., 0, :]
    segmentation_masks = ops.convert_to_numpy(segmentation_masks)

    # Replace class with color
    masks = segmentation_masks
    masks = np.transpose(masks, axes=(3, 0, 1, 2)).astype("bool")
    images_to_draw = ops.convert_to_numpy(images).copy()
    for mask, color in zip(masks, colors):
        color = np.array(color, dtype=images_to_draw.dtype)
        images_to_draw[mask, ...] = color[None, :]
    images_to_draw = ops.convert_to_tensor(images_to_draw)
    images_to_draw = ops.cast(images_to_draw, dtype="float32")

    # Apply blending
    outputs = images * (1 - alpha) + images_to_draw * alpha
    outputs = ops.where(valid_masks[..., None], outputs, images)
    outputs = ops.cast(outputs, dtype="uint8")
    outputs = ops.convert_to_numpy(outputs)
    return outputs


def _generate_color_palette(num_classes: int):
    palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [((i * palette) % 255).tolist() for i in range(num_classes)]
