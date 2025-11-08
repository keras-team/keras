"""Adaptive Average Pooling 3D layer."""

from keras import config
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.AdaptiveAveragePooling3D")
class AdaptiveAveragePooling3D(Layer):
    """Adaptive average pooling operation for 3D spatial data.

    This layer applies an adaptive average pooling operation, which pools the
    input such that the output has a target shape specified by `output_size`,
    regardless of the input shape. The kernel size and stride are automatically
    computed to achieve the target output size.

    Args:
        output_size: Integer or tuple of 3 integers, specifying the target
            output size `(depth, height, width)`.
            If a single integer is provided, the same value is used for all
            three dimensions.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, depth, height, width, channels)` while
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, depth, height, width)`.
            Defaults to the value found in your Keras config file at
            `~/.keras/keras.json`. If never set, then "channels_last" is used.

    Input shape:
        - If `data_format="channels_last"`:
            5D tensor with shape `(batch_size, depth, height, width, channels)`.
        - If `data_format="channels_first"`:
            5D tensor with shape `(batch_size, channels, depth, height, width)`.

    Output shape:
        - If `data_format="channels_last"`:
            5D tensor with shape
            `(batch_size, output_depth, output_height, output_width, channels)`.
        - If `data_format="channels_first"`:
            5D tensor with shape
            `(batch_size, channels, output_depth, output_height, output_width)`.

    Examples:

    >>> input_vol = np.random.rand(1, 16, 64, 64, 3)
    >>> layer = keras.layers.AdaptiveAveragePooling3D(output_size=(8, 32, 32))
    >>> output_vol = layer(input_vol)
    >>> output_vol.shape
    (1, 8, 32, 32, 3)

    >>> # Single integer for cubic output
    >>> layer = keras.layers.AdaptiveAveragePooling3D(output_size=4)
    >>> output_vol = layer(input_vol)
    >>> output_vol.shape
    (1, 4, 4, 4, 3)
    """

    def __init__(self, output_size, data_format=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        elif isinstance(output_size, (list, tuple)):
            if len(output_size) != 3:
                raise ValueError(
                    "`output_size` must be an integer or tuple of 3 integers. "
                    f"Received output_size={output_size}"
                )
            self.output_size = tuple(output_size)
        else:
            raise TypeError(
                "`output_size` must be an integer or tuple of 3 integers. "
                "Received output_size={} of type {}".format(
                    output_size, type(output_size)
                )
            )

        self.data_format = data_format or config.image_data_format()

        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"Invalid data_format: {self.data_format}. "
                "Must be either 'channels_first' or 'channels_last'."
            )

    def call(self, inputs):
        return ops.adaptive_avg_pool(
            inputs, output_size=self.output_size, data_format=self.data_format
        )

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            return (
                input_shape[0],
                self.output_size[0],
                self.output_size[1],
                self.output_size[2],
                input_shape[4],
            )
        else:  # channels_first
            return (
                input_shape[0],
                input_shape[1],
                self.output_size[0],
                self.output_size[1],
                self.output_size[2],
            )

    def get_config(self):
        config_dict = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config_dict}
