import math

from keras_core import backend
from keras_core import ops
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Flatten")
class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Note: If inputs are shaped `(batch,)` without a feature axis, then
    flattening adds an extra channel dimension and output shape is `(batch, 1)`.

    Args:
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Example:

    >>> x = keras_core.Input(shape=(10, 64))
    >>> y = keras_core.layers.Flatten()(x)
    >>> y.shape
    (None, 640)
    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)
        self._channels_first = self.data_format == "channels_first"

    def call(self, inputs):
        input_shape = inputs.shape
        rank = len(input_shape)

        if self._channels_first and rank > 1:
            # Switch to channels-last format.
            inputs = ops.transpose(inputs, axes=(0, *range(2, rank), 1))

        output_shape = tuple(
            dim if dim is not None else -1
            for dim in self.compute_output_shape(input_shape)
        )
        return ops.reshape(inputs, output_shape)

    def compute_output_shape(self, input_shape):
        non_batch_dims = input_shape[1:]
        if len(non_batch_dims) == 0:
            flattened_dim = 1
        elif None in non_batch_dims:
            flattened_dim = None
        else:
            flattened_dim = math.prod(non_batch_dims)
        return (input_shape[0], flattened_dim)

    def get_config(self):
        config = {"data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
