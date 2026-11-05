"""Adaptive Max Pooling 2D layer."""

from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_adaptive_pooling import (
    BaseAdaptiveMaxPooling,
)


@keras_export("keras.layers.AdaptiveMaxPooling2D")
class AdaptiveMaxPooling2D(BaseAdaptiveMaxPooling):
    """Adaptive max pooling operation for 2D spatial data.

    This layer applies an adaptive max pooling operation, which pools the
    input such that the output has a target spatial size specified by
    `output_size`, regardless of the input spatial size. The kernel size
    and stride are automatically computed to achieve the target output size.

    Args:
        output_size: Integer or tuple of 2 integers specifying the
            target output size.
            If an integer, the same value is used for both height and width.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`.
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, `"channels_last"` is used.

    Input shape:
        - If `data_format="channels_last"`: 4D tensor
            `(batch_size, height, width, channels)`
        - If `data_format="channels_first"`: 4D tensor
            `(batch_size, channels, height, width)`

    Output shape:
        - If `data_format="channels_last"`:
            `(batch_size, output_height, output_width, channels)`
        - If `data_format="channels_first"`:
            `(batch_size, channels, output_height, output_width)`

    Examples:
        >>> import numpy as np
        >>> input_img = np.random.rand(1, 64, 64, 3)
        >>> layer = AdaptiveMaxPooling2D(output_size=32)
        >>> output_img = layer(input_img)
        >>> output_img.shape
        (1, 32, 32, 3)
    """

    def __init__(self, output_size, data_format=None, **kwargs):
        if isinstance(output_size, int):
            output_size_tuple = (output_size, output_size)
        elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
            output_size_tuple = tuple(output_size)
        else:
            raise TypeError(
                f"`output_size` must be an integer or (height, width) tuple. "
                f"Received: {output_size} of type {type(output_size)}"
            )

        super().__init__(output_size_tuple, data_format, **kwargs)
