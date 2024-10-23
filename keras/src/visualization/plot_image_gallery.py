import math

import numpy as np

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)

try:
    import matplotlib.pyplot as plt
except:
    plt = None


def _extract_image_batch(images, num_images, batch_size):
    num_batches_required = math.ceil(num_images / batch_size)

    if len(ops.shape(images)) != 4:
        raise ValueError(
            "`plot_images_gallery()` requires you to "
            "batch your `np.array` samples together."
        )
    num_samples = (
        num_images if num_images <= batch_size else num_batches_required
    )
    sample = images[:num_samples, ...]

    return sample


@keras_export("keras.visualization.plot_image_gallery")
def plot_image_gallery(
    images,
    value_range,
    scale=2,
    rows=None,
    cols=None,
    path=None,
    show=None,
    transparent=True,
    dpi=60,
    legend_handles=None,
):
    """Displays a gallery of images.

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        scale: how large to scale the images in the gallery
        rows: (Optional) number of rows in the gallery to show.
            Required if inputs are unbatched.
        cols: (Optional) number of columns in the gallery to show.
            Required if inputs are unbatched.
        path: (Optional) path to save the resulting gallery to.
        show: (Optional) whether to show the gallery of images.
        transparent: (Optional) whether to give the image a transparent
            background, defaults to `True`.
        dpi: (Optional) the dpi to pass to matplotlib.savefig(), defaults to
            `60`.
        legend_handles: (Optional) matplotlib.patches List of legend handles.
            I.e. passing: `[patches.Patch(color='red', label='mylabel')]` will
            produce a legend with a single red patch and the label 'mylabel'.
    """

    if path is not None and show:
        raise ValueError(
            "plot_gallery() expects either `path` to be set, or `show` "
            "to be true."
        )

    batch_size = (
        ops.shape(images)[0] if len(ops.shape(images)) == 4 else 1
    )  # batch_size from np.array or single image

    rows = rows or int(math.ceil(math.sqrt(batch_size)))
    cols = cols or int(math.ceil(batch_size // rows))
    num_images = rows * cols
    images = _extract_image_batch(images, num_images, batch_size)

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
    else:
        return fig
