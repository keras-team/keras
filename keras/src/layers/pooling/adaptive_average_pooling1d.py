"""Adaptive Average Pooling 1D layer."""

from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_adaptive_pooling import (
    BaseAdaptiveAveragePooling,
)


@keras_export("keras.layers.AdaptiveAveragePooling1D")
class AdaptiveAveragePooling1D(BaseAdaptiveAveragePooling):
    """Adaptive average pooling operation for 1D temporal or spatial data.

    This layer applies an adaptive average pooling operation, which pools the
    input such that the output has a target length specified by `output_size`,
    regardless of the input length. The kernel size and stride are automatically
    computed to achieve the target output size.

    Args:
        output_size: Integer specifying the target output length.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, length, channels)`.
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, length)`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, `"channels_last"` is used.

    Input shape:
        - If `data_format="channels_last"`: 3D tensor
            `(batch_size, length, channels)`
        - If `data_format="channels_first"`: 3D tensor
            `(batch_size, channels, length)`

    Output shape:
        - If `data_format="channels_last"`:
            `(batch_size, output_length, channels)`
        - If `data_format="channels_first"`:
            `(batch_size, channels, output_length)`

    Examples:
        >>> import numpy as np
        >>> input_seq = np.random.rand(1, 64, 3)
        >>> layer = AdaptiveAveragePooling1D(output_size=32)
        >>> output_seq = layer(input_seq)
        >>> output_seq.shape
        (1, 32, 3)
    """

    def __init__(self, output_size, data_format=None, **kwargs):
        if isinstance(output_size, int):
            output_size = (output_size,)
        elif isinstance(output_size, (tuple, list)):
            if len(output_size) != 1:
                raise ValueError(
                    f"For 1D input, `output_size` tuple must have length 1. "
                    f"Received: {output_size}"
                )
            output_size = tuple(output_size)
        else:
            raise TypeError(
                f"`output_size` must be an integer or tuple of 1 integer. "
                f"Received: {output_size} of type {type(output_size)}"
            )

        super().__init__(output_size, data_format, **kwargs)
