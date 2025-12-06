"""Adaptive Max Pooling 1D layer."""

from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_adaptive_pooling import (
    BaseAdaptiveMaxPooling,
)


@keras_export("keras.layers.AdaptiveMaxPooling1D")
class AdaptiveMaxPooling1D(BaseAdaptiveMaxPooling):
    """Adaptive max pooling operation for 1D temporal or spatial data.

    This layer applies an adaptive max pooling operation, which pools the
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
        >>> layer = AdaptiveMaxPooling1D(output_size=32)
        >>> output_seq = layer(input_seq)
        >>> output_seq.shape
        (1, 32, 3)
    """

    def __init__(self, output_size, data_format=None, **kwargs):
        if not isinstance(output_size, int):
            raise TypeError(
                "`output_size` must be an integer. Received output_size={} "
                "of type {}".format(output_size, type(output_size))
            )

        output_size_tuple = (output_size,)

        super().__init__(output_size_tuple, data_format, **kwargs)
