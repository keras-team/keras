import math

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _extract_image_batch(images, num_images, batch_size):
    """Extracts a batch of images for plotting.

    Args:
        images: The 4D tensor or NumPy array of images.
        num_images: The number of images to extract.
        batch_size: The original batch size of the images.

    Returns:
        A 4D tensor or NumPy array containing the extracted images.

    Raises:
        ValueError: If `images` is not a 4D tensor/array.
    """

    if len(ops.shape(images)) != 4:
        raise ValueError(
            "`plot_images_gallery()` requires you to "
            "batch your `np.array` samples together."
        )
    num_samples = min(num_images, batch_size)
    sample = images[:num_samples, ...]

    return sample


@keras_export("keras.visualization.plot_image_gallery")
def plot_image_gallery(
    images,
    rows=None,
    cols=None,
    value_range=(0, 255),
    scale=2,
    path=None,
    show=None,
    transparent=True,
    dpi=60,
    legend_handles=None,
    data_format=None,
):
    """Displays a gallery of images.

    Args:
        images: A 4D tensor or NumPy array of images. Shape should be
           `(batch_size, height, width, channels)`.
        value_range: A tuple specifying the value range of the images
            (e.g., `(0, 255)` or `(0, 1)`). Defaults to `(0, 255)`.
        rows: The number of rows in the gallery. If `None`, it's calculated
            based on the number of images and `cols`. Defaults to `None`.
        cols: The number of columns in the gallery. If `None`, it's calculated
            based on the number of images and `rows`. Defaults to `None`.
        scale: A float controlling the size of the displayed images. The images
            are scaled by this factor. Defaults to `2`.
        path: The path to save the generated gallery image. If `None`, the
            image is displayed using `plt.show()`. Defaults to `None`.
        show: Whether to display the image using `plt.show()`. If `True`, the
            image is displayed. If `False`, the image is not displayed.
            Ignored if `path` is not `None`. Defaults to `True` if `path`
            is `None`, `False` otherwise.
        transparent:  A boolean, whether to save the figure with a transparent
            background. Defaults to `True`.
        dpi: The DPI (dots per inch) for saving the figure. Defaults to 60.
        legend_handles: A list of matplotlib `Patch` objects to use as legend
            handles. Defaults to `None`.
        data_format: The image data format `"channels_last"` or
            `"channels_first"`. Defaults to the Keras backend data format.

    Raises:
        ValueError: If both `path` and `show` are set to non-`None` values or if
            `images` is not a 4D tensor or array.
        ImportError: if matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "The `plot_image_gallery` function requires the `matplotlib` "
            "package. Please install it with `pip install matplotlib`."
        )

    if path is not None and show:
        raise ValueError(
            "plot_gallery() expects either `path` to be set, or `show` "
            "to be true."
        )

    show = show if show is not None else (path is None)
    data_format = data_format or backend.image_data_format()

    batch_size = ops.shape(images)[0] if len(ops.shape(images)) == 4 else 1

    rows = rows or int(math.ceil(math.sqrt(batch_size)))
    cols = cols or int(math.ceil(batch_size // rows))
    num_images = rows * cols

    images = _extract_image_batch(images, num_images, batch_size)
    if (
        data_format == "channels_first"
    ):  # Ensure correct data format for plotting
        images = ops.transpose(images, (0, 2, 3, 1))
    # Generate subplots
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * scale, rows * scale),
        frameon=False,
        layout="tight",
        squeeze=True,
        sharex="row",
        sharey="col",
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    if isinstance(axes, np.ndarray) and len(axes.shape) == 1:
        expand_axis = 0 if rows == 1 else -1
        axes = np.expand_dims(axes, expand_axis)

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="lower center")

    images = BaseImagePreprocessingLayer()._transform_value_range(
        images=images, original_range=value_range, target_range=(0, 255)
    )

    images = ops.convert_to_numpy(images)
    if data_format == "channels_first":
        images = images.transpose(0, 2, 3, 1)

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            current_axis = (
                axes[row, col] if isinstance(axes, np.ndarray) else axes
            )
            current_axis.imshow(images[index].astype("uint8"))
            current_axis.margins(x=0, y=0)
            current_axis.axis("off")

    if path is not None:
        plt.savefig(
            fname=path,
            pad_inches=0,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
        )
        plt.close()
    elif show:
        plt.show()
        plt.close()
