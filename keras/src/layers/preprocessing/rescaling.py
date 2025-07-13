import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Rescaling")
class Rescaling(TFDataLayer):
    """A preprocessing layer which rescales input values to a new range.

    This layer rescales every value of an input (often an image) by multiplying
    by `scale` and adding `offset`.

    For instance:

    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.

    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
    you would pass `scale=1./127.5, offset=-1`.

    The rescaling is applied both during training and inference. Inputs can be
    of integer or floating point dtype, and by default the layer will output
    floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        scale: Float, int, list, tuple or np.ndarray.
            The scale to apply to the inputs.
            If scalar, the same scale will be applied to
            all features or channels of input. If a list, tuple or
            1D array, the scaling is applied per channel.
        offset: Float, int, list/tuple or numpy ndarray.
            The offset to apply to the inputs.
            If scalar, the same scale will be applied to
            all features or channels of input. If a list, tuple or
            1D array, the scaling is applied per channel.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(self, scale, offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.supports_masking = True

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = self.backend.cast(self.scale, dtype)
        offset = self.backend.cast(self.offset, dtype)
        scale_shape = self.backend.core.shape(scale)
        if (
            len(scale_shape) > 0
            and backend.image_data_format() == "channels_first"
            and len(inputs.shape) > 2
        ):
            scale = self.backend.numpy.reshape(
                scale, scale_shape + (1,) * (3 - len(scale_shape))
            )
        return self.backend.cast(inputs, dtype) * scale + offset

    def compute_output_shape(self, input_shape):
        input_shape = tuple(input_shape)

        if backend.image_data_format() == "channels_last":
            channels_axis = -1
        else:
            channels_axis = 1

        input_channels = input_shape[channels_axis]

        if input_channels is None:
            return input_shape

        scale_len = None
        offset_len = None

        if isinstance(self.scale, (list, tuple)):
            scale_len = len(self.scale)
        elif isinstance(self.scale, np.ndarray) and self.scale.ndim == 1:
            scale_len = self.scale.size
        elif isinstance(self.scale, (int, float)):
            scale_len = 1

        if isinstance(self.offset, (list, tuple)):
            offset_len = len(self.offset)
        elif isinstance(self.offset, np.ndarray) and self.offset.ndim == 1:
            offset_len = self.offset.size
        elif isinstance(self.offset, (int, float)):
            offset_len = 1

        if scale_len == 1 and offset_len == 1:
            return input_shape

        broadcast_len = None
        if scale_len is not None and scale_len != input_channels:
            broadcast_len = scale_len
        if offset_len is not None and offset_len != input_channels:
            if broadcast_len is not None and offset_len != broadcast_len:
                raise ValueError(
                    "Inconsistent `scale` and `offset` lengths "
                    f"for broadcasting."
                    f" Received: `scale` = {self.scale},"
                    f"`offset` = {self.offset}. "
                    f"Ensure both `scale` and `offset` are either scalar "
                    f"or list, tuples, arrays of the same length."
                )
            broadcast_len = offset_len

        if broadcast_len:
            output_shape = list(input_shape)
            output_shape[channels_axis] = broadcast_len
            return tuple(output_shape)

        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # `scale` and `offset` might be numpy array.
                "scale": serialization_lib.serialize_keras_object(self.scale),
                "offset": serialization_lib.serialize_keras_object(self.offset),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        config["scale"] = serialization_lib.deserialize_keras_object(
            config["scale"], custom_objects=custom_objects
        )
        config["offset"] = serialization_lib.deserialize_keras_object(
            config["offset"], custom_objects=custom_objects
        )
        return cls(**config)
