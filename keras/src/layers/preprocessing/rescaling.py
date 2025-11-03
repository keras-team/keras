from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Rescaling")
class Rescaling(DataLayer):
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

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        scale: Float, the scale to apply to the inputs.
        offset: Float, the offset to apply to the inputs.
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
        ):
            scale = self.backend.numpy.reshape(
                scale, scale_shape + (1,) * (3 - len(scale_shape))
            )
        return self.backend.cast(inputs, dtype) * scale + offset

    def compute_output_shape(self, input_shape):
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
