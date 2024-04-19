from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.regularization.dropout import Dropout


class BaseSpatialDropout(Dropout):
    def __init__(self, rate, seed=None, name=None, dtype=None):
        super().__init__(rate, seed=seed, name=name, dtype=dtype)

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            return backend.random.dropout(
                inputs,
                self.rate,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed_generator,
            )
        return inputs

    def get_config(self):
        return {
            "rate": self.rate,
            "seed": self.seed,
            "name": self.name,
            "dtype": self.dtype,
        }


@keras_export("keras.layers.SpatialDropout1D")
class SpatialDropout1D(BaseSpatialDropout):
    """Spatial 1D version of Dropout.

    This layer performs the same function as Dropout, however, it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, `SpatialDropout1D` will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.

    Call arguments:
        inputs: A 3D tensor.
        training: Python boolean indicating whether the layer
            should behave in training mode (applying dropout)
            or in inference mode (pass-through).

    Input shape:
        3D tensor with shape: `(samples, timesteps, channels)`

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
    """

    def __init__(self, rate, seed=None, name=None, dtype=None):
        super().__init__(rate, seed=seed, name=name, dtype=dtype)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = ops.shape(inputs)
        return (input_shape[0], 1, input_shape[2])


@keras_export("keras.layers.SpatialDropout2D")
class SpatialDropout2D(BaseSpatialDropout):
    """Spatial 2D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, `SpatialDropout2D` will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        data_format: `"channels_first"` or `"channels_last"`.
            In `"channels_first"` mode, the channels dimension (the depth)
            is at index 1, in `"channels_last"` mode is it at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.

    Call arguments:
        inputs: A 4D tensor.
        training: Python boolean indicating whether the layer
            should behave in training mode (applying dropout)
            or in inference mode (pass-through).

    Input shape:
        4D tensor with shape: `(samples, channels, rows, cols)` if
            data_format='channels_first'
        or 4D tensor with shape: `(samples, rows, cols, channels)` if
            data_format='channels_last'.

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
    """

    def __init__(
        self, rate, data_format=None, seed=None, name=None, dtype=None
    ):
        super().__init__(rate, seed=seed, name=name, dtype=dtype)
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = ops.shape(inputs)
        if self.data_format == "channels_first":
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == "channels_last":
            return (input_shape[0], 1, 1, input_shape[3])

    def get_config(self):
        base_config = super().get_config()
        config = {
            "data_format": self.data_format,
        }
        return {**base_config, **config}


@keras_export("keras.layers.SpatialDropout3D")
class SpatialDropout3D(BaseSpatialDropout):
    """Spatial 3D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        data_format: `"channels_first"` or `"channels_last"`.
            In `"channels_first"` mode, the channels dimension (the depth)
            is at index 1, in `"channels_last"` mode is it at index 4.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.

    Call arguments:
        inputs: A 5D tensor.
        training: Python boolean indicating whether the layer
                should behave in training mode (applying dropout)
                or in inference mode (pass-through).

    Input shape:
        5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if
            data_format='channels_first'
        or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if
            data_format='channels_last'.

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
    """

    def __init__(
        self, rate, data_format=None, seed=None, name=None, dtype=None
    ):
        super().__init__(rate, seed=seed, name=name, dtype=dtype)
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def _get_noise_shape(self, inputs):
        input_shape = ops.shape(inputs)
        if self.data_format == "channels_first":
            return (input_shape[0], input_shape[1], 1, 1, 1)
        elif self.data_format == "channels_last":
            return (input_shape[0], 1, 1, 1, input_shape[4])

    def get_config(self):
        base_config = super().get_config()
        config = {
            "data_format": self.data_format,
        }
        return {**base_config, **config}
