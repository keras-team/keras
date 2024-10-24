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
    data_format=None,
    **kwargs
):
    """Plots a gallery of images with corresponding segmentation masks.

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery. The images should be batched and of shape (B, H, W, C).
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        num_classes: number of segmentation classes.
        y_true: A Tensor or NumPy array representing the ground truth
            segmentation masks. The ground truth segmentation maps should be
            batched.
        y_pred: A Tensor or NumPy array representing the predicted
            segmentation masks. The predicted segmentation masks should be
            batched.
        rows: int. Number of rows in the gallery to shows. Required if inputs
            are unbatched. Defaults to `None`
        cols: int. Number of columns in the gallery to show. Required if inputs
            are unbatched.Defaults to `None`
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        kwargs: keyword arguments to propagate to
            `keras.visualization.plot_image_gallery()`.
    """
    data_format = data_format or backend.image_data_format()
    plotted_images = ops.convert_to_numpy(images)
    masks_to_contatenate = [plotted_images]

    draw_fn = functools.partial(
        draw_segmentation_masks,
        num_classes=num_classes,
        color_mapping=color_mapping,
        data_format=data_format,
    )

    if y_true is not None:
        plotted_y_true = draw_fn(plotted_images, y_true)
        masks_to_contatenate.append(plotted_y_true)

    if y_pred is not None:
        plotted_y_pred = draw_fn(plotted_images, y_pred)
        masks_to_contatenate.append(plotted_y_pred)

    # Concatenate the images and the masks together.
    plotted_images = np.concatenate(masks_to_contatenate, axis=2)

    return plot_image_gallery(
        plotted_images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        data_format=data_format,
    )
