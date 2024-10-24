import functools

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.visualization.draw_segmentation_masks import (
    draw_segmentation_masks,
)
from keras.src.visualization.plot_image_gallery import plot_image_gallery


@keras_export("keras.visualization.plot_segmentation_mask_gallery")
def plot_segmentation_mask_gallery(
    images,
    value_range,
    num_classes,
    y_true=None,
    y_pred=None,
    rows=None,
    cols=None,
    color_mapping=None,
    alpha=0.8,
    ignore_index=-1,
    data_format=None,
    **kwargs,
):
    """Plots a gallery of images with corresponding segmentation masks.

    Args:
        images: A 4D tensor or NumPy array of images. Shape should be
            `(batch_size, height, width, channels)`.
        value_range: A tuple specifying the value range of the images
            (e.g., `(0, 255)` or `(0, 1)`).
        num_classes: The number of segmentation classes.  Class indices should
            start from `1`.  Class `0` will be treated as background and
            ignored if `ignore_index` is not 0.
        y_true: A 3D/4D tensor or NumPy array representing the ground truth
            segmentation masks. Shape should be `(batch_size, height, width)` or
            `(batch_size, height, width, 1)`. Defaults to `None`.
        y_pred: A 3D/4D tensor or NumPy array representing the predicted
            segmentation masks.  Shape should be the same as `y_true`.
            Defaults to `None`.
        rows: The number of rows in the image gallery. If `None`,
            it's calculated automatically. Defaults to `None`.
        cols: The number of columns in the image gallery. If `None`,
            it's calculated automatically. Defaults to `None`.
        color_mapping: A dictionary mapping class indices to RGB colors.
            If `None`, a default color palette is used. Class indices start
            from `1`. Defaults to `None`.
        alpha: The opacity of the segmentation masks (a float between 0 and 1).
            Defaults to `0.8`.
        ignore_index: The class index to ignore when drawing masks.
            Defaults to `-1`.
        data_format: The image data format `"channels_last"` or
            `"channels_first"`. Defaults to the Keras backend data format.
        kwargs: Additional keyword arguments to be passed to
            `keras.visualization.plot_image_gallery`.

    Returns:
        The output of `keras.visualization.plot_image_gallery`.

    Raises:
        ValueError: If `images` is not a 4D tensor/array.
    """
    data_format = data_format or backend.image_data_format()
    image_shape = ops.shape(images)
    if len(image_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={image_shape}"
        )
    if data_format == "channels_first":
        images = ops.transpose(images, (0, 2, 3, 1))

    images_np = ops.convert_to_numpy(images)
    images_with_masks = [images_np]

    draw_masks_fn = functools.partial(
        draw_segmentation_masks,
        num_classes=num_classes,
        color_mapping=color_mapping,
        alpha=alpha,
        ignore_index=ignore_index,
    )

    if y_true is not None:
        if data_format == "channels_first":
            y_true = ops.transpose(y_true, (0, 2, 3, 1))
        y_true = ops.cast(y_true, "int32")
        true_masks_drawn = draw_masks_fn(images_np, y_true)
        images_with_masks.append(true_masks_drawn)

    if y_pred is not None:
        if data_format == "channels_first":
            y_pred = ops.transpose(y_pred, (0, 2, 3, 1))
        y_pred = ops.cast(y_pred, "int32")
        predicted_masks_drawn = draw_masks_fn(images_np, y_pred)
        images_with_masks.append(predicted_masks_drawn)

    # Concatenate images and masks
    gallery_images = np.concatenate(images_with_masks, axis=2)

    return plot_image_gallery(
        gallery_images, value_range=value_range, rows=rows, cols=cols, **kwargs
    )
