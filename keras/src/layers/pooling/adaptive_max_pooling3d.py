"""Adaptive Max Pooling 3D layer."""

from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_adaptive_pooling import (
    BaseAdaptiveMaxPooling,
)


@keras_export("keras.layers.AdaptiveMaxPooling3D")
class AdaptiveMaxPooling3D(BaseAdaptiveMaxPooling):
    """Adaptive max pooling operation for 3D volumetric data.

    This layer applies an adaptive max pooling operation, which pools the
    input such that the output has a target spatial size specified by
    `output_size`, regardless of the input spatial size. The kernel size
    and stride are automatically computed to achieve the target output size.

    Args:
        output_size: Integer or tuple of 3 integers specifying the
            target output size.
            If an integer, the same value is used for depth, height, and width.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, depth, height, width, channels)`.
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, depth, height, width)`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, `"channels_last"` is used.

    Input shape:
        - If `data_format="channels_last"`: 5D tensor
            `(batch_size, depth, height, width, channels)`
        - If `data_format="channels_first"`: 5D tensor
            `(batch_size, channels, depth, height, width)`

    Output shape:
        - If `data_format="channels_last"`:
            `(batch_size, output_depth, output_height, output_width, channels)`
        - If `data_format="channels_first"`:
            `(batch_size, channels, output_depth, output_height, output_width)`

    Examples:
        >>> import numpy as np
        >>> input_vol = np.random.rand(1, 32, 32, 32, 3)
        >>> layer = AdaptiveMaxPooling3D(output_size=16)
        >>> output_vol = layer(input_vol)
        >>> output_vol.shape
        (1, 16, 16, 16, 3)
    """

    def __init__(self, output_size, data_format=None, **kwargs):
        if isinstance(output_size, int):
            output_size_tuple = (output_size, output_size, output_size)
        elif isinstance(output_size, (tuple, list)) and len(output_size) == 3:
            output_size_tuple = tuple(output_size)
        else:
            raise TypeError(
                f"`output_size` must be an integer or "
                f"(depth, height, width) tuple. "
                f"Received: {output_size} of type {type(output_size)}"
            )

        super().__init__(output_size_tuple, data_format, **kwargs)
