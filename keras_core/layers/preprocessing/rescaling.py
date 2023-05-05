from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Rescaling")
class Rescaling(Layer):
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

    Args:
        scale: Float, the scale to apply to the inputs.
        offset: Float, the offset to apply to the inputs.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype.
    """

    def __init__(self, scale, offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.supports_masking = True

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = ops.cast(self.scale, dtype)
        offset = ops.cast(self.offset, dtype)
        return inputs * scale + offset

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "scale": self.scale,
            "offset": self.offset,
        }
        return {**base_config, **config}
