import functools

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.visualization.draw_bounding_boxes import draw_bounding_boxes
from keras.src.visualization.plot_image_gallery import plot_image_gallery

try:
    from matplotlib import patches
except:
    patches = None


@keras_export("keras.visualization.plot_bounding_box_gallery")
def plot_bounding_box_gallery(
    images,
    value_range,
    bounding_box_format,
    y_true=None,
    y_pred=None,
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
    **kwargs
):
    """
    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery.
        value_range: Value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        bounding_box_format: The bounding_box_format the provided bounding boxes
            are in.
        y_true: Bounding box dictionary representing the
            ground truth bounding boxes and labels. Defaults to `None`
        y_pred: Bounding box dictionary representing the
            ground truth bounding boxes and labels. Defaults to `None`
        pred_color: Three element tuple representing the color to use for
            plotting predicted bounding boxes.
        true_color: three element tuple representing the color to use for
            plotting true bounding boxes.
        class_mapping: Class mapping from class IDs to strings. Defaults to
            `None`.
        ground_truth_mapping: Class mapping from class IDs to
            strings, defaults to `class_mapping`. Defaults to `None`
        prediction_mapping: Class mapping from class IDs to strings.
            Defaults to `class_mapping`.
        line_thickness: Line thickness for the box and text labels.
            Defaults to 2.
        text_thickness: The line thickness for the text, defaults to
            `1.0`.
        font_scale: Font size to draw bounding boxes in.
        legend: Whether to create a legend with the specified colors for
            `y_true` and `y_pred`. Defaults to False.
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

    prediction_mapping = prediction_mapping or class_mapping
    ground_truth_mapping = ground_truth_mapping or class_mapping
    data_format = data_format or backend.image_data_format()

    plotted_images = ops.convert_to_numpy(images)

    draw_fn = functools.partial(
        draw_bounding_boxes,
        bounding_box_format=bounding_box_format,
        line_thickness=line_thickness,
        text_thickness=text_thickness,
        font_scale=font_scale,
        data_format=data_format,
    )

    if y_true is not None:
        plotted_images = draw_fn(
            plotted_images,
            y_true,
            true_color,
            class_mapping=ground_truth_mapping,
        )

    if y_pred is not None:
        plotted_images = draw_fn(
            plotted_images, y_pred, pred_color, class_mapping=prediction_mapping
        )

    if legend:
        if legend_handles:
            raise ValueError(
                "Only pass `legend` OR `legend_handles` to "
                "`luketils.visualization.plot_bounding_box_gallery()`."
            )
        legend_handles = [
            patches.Patch(
                color=np.array(true_color) / 255.0,
                label="Ground Truth",
            ),
            patches.Patch(
                color=np.array(pred_color) / 255.0,
                label="Prediction",
            ),
        ]

    return plot_image_gallery(
        plotted_images,
        value_range,
        legend_handles=legend_handles,
        rows=rows,
        cols=cols,
        data_format=data_format,
        **kwargs
    )
