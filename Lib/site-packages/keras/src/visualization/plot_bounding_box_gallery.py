import functools

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.visualization.draw_bounding_boxes import draw_bounding_boxes
from keras.src.visualization.plot_image_gallery import plot_image_gallery

try:
    from matplotlib import patches  # For legend patches
except ImportError:
    patches = None


@keras_export("keras.visualization.plot_bounding_box_gallery")
def plot_bounding_box_gallery(
    images,
    bounding_box_format,
    y_true=None,
    y_pred=None,
    value_range=(0, 255),
    true_color=(0, 188, 212),
    pred_color=(255, 235, 59),
    line_thickness=2,
    font_scale=1.0,
    text_thickness=None,
    class_mapping=None,
    ground_truth_mapping=None,
    prediction_mapping=None,
    legend=False,
    legend_handles=None,
    rows=None,
    cols=None,
    data_format=None,
    **kwargs,
):
    """Plots a gallery of images with bounding boxes.

    This function can display both ground truth and predicted bounding boxes on
    a set of images.  It supports various bounding box formats and can include
    class labels and a legend.

    Args:
        images: A 4D tensor or NumPy array of images. Shape should be
            `(batch_size, height, width, channels)`.
        bounding_box_format: The format of the bounding boxes.
            Refer [keras-io](TODO)
        y_true: A dictionary containing the ground truth bounding boxes and
            labels. Should have the same structure as the `bounding_boxes`
            argument in `keras.visualization.draw_bounding_boxes`.
            Defaults to `None`.
        y_pred: A dictionary containing the predicted bounding boxes and labels.
            Should have the same structure as `y_true`. Defaults to `None`.
        value_range: A tuple specifying the value range of the images
            (e.g., `(0, 255)` or `(0, 1)`). Defaults to `(0, 255)`.
        true_color: A tuple of three integers representing the RGB color for the
            ground truth bounding boxes. Defaults to `(0, 188, 212)`.
        pred_color: A tuple of three integers representing the RGB color for the
            predicted bounding boxes. Defaults to `(255, 235, 59)`.
        line_thickness: The thickness of the bounding box lines. Defaults to 2.
        font_scale: The scale of the font used for labels. Defaults to 1.0.
        text_thickness: The thickness of the bounding box text. Defaults to
            `line_thickness`.
        class_mapping: A dictionary mapping class IDs to class names.  Used f
            or both ground truth and predicted boxes if `ground_truth_mapping`
            and `prediction_mapping` are not provided. Defaults to `None`.
        ground_truth_mapping:  A dictionary mapping class IDs to class names
            specifically for ground truth boxes. Overrides `class_mapping`
            for ground truth. Defaults to `None`.
        prediction_mapping: A dictionary mapping class IDs to class names
            specifically for predicted boxes. Overrides `class_mapping` for
            predictions. Defaults to `None`.
        legend: A boolean indicating whether to show a legend.
            Defaults to `False`.
        legend_handles: A list of matplotlib `Patch` objects to use for the
            legend. If this is provided, the `legend` argument will be ignored.
            Defaults to `None`.
        rows: The number of rows in the image gallery. Required if the images
            are not batched. Defaults to `None`.
        cols: The number of columns in the image gallery. Required if the images
            are not batched. Defaults to `None`.
        data_format: The image data format `"channels_last"` or
            `"channels_first"`. Defaults to the Keras backend data format.
        kwargs: Additional keyword arguments to be passed to
            `keras.visualization.plot_image_gallery`.

    Returns:
       The output of `keras.visualization.plot_image_gallery`.

    Raises:
        ValueError: If `images` is not a 4D tensor/array or if both `legend` a
        nd `legend_handles` are specified.
        ImportError: if matplotlib is not installed
    """
    if patches is None:
        raise ImportError(
            "The `plot_bounding_box_gallery` function requires the "
            " `matplotlib` package. Please install it with "
            " `pip install matplotlib`."
        )

    prediction_mapping = prediction_mapping or class_mapping
    ground_truth_mapping = ground_truth_mapping or class_mapping
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if data_format == "channels_first":  # Ensure correct data format
        images = ops.transpose(images, (0, 2, 3, 1))
    plotted_images = ops.convert_to_numpy(images)

    draw_fn = functools.partial(
        draw_bounding_boxes,
        bounding_box_format=bounding_box_format,
        line_thickness=line_thickness,
        text_thickness=text_thickness,
        font_scale=font_scale,
    )

    if y_true is not None:
        plotted_images = draw_fn(
            plotted_images,
            y_true,
            color=true_color,
            class_mapping=ground_truth_mapping,
        )

    if y_pred is not None:
        plotted_images = draw_fn(
            plotted_images,
            y_pred,
            color=pred_color,
            class_mapping=prediction_mapping,
        )

    if legend:
        if legend_handles:
            raise ValueError(
                "Only pass `legend` OR `legend_handles` to "
                "`keras.visualization.plot_bounding_box_gallery()`."
            )
        legend_handles = [
            patches.Patch(
                color=np.array(true_color) / 255.0,  # Normalize color
                label="Ground Truth",
            ),
            patches.Patch(
                color=np.array(pred_color) / 255.0,  # Normalize color
                label="Prediction",
            ),
        ]

    return plot_image_gallery(
        plotted_images,
        value_range=value_range,
        legend_handles=legend_handles,
        rows=rows,
        cols=cols,
        **kwargs,
    )
