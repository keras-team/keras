import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomRotation")
class RandomRotation(TFDataLayer):
    """A preprocessing layer which randomly rotates images during training.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, pass `training=True` when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        factor: a float represented as fraction of 2 Pi, or a tuple of size 2
            representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating
            counter clock-wise,
            while a negative value means clock-wise.
            When represented as a single
            float, this value is used for both the upper and lower bound.
            For instance, `factor=(-0.2, 0.3)`
            results in an output rotation by a random
            amount in the range `[-20% * 2pi, 30% * 2pi]`.
            `factor=0.2` results in an
            output rotating by a random amount
            in the range `[-20% * 2pi, 20% * 2pi]`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by
                filling all values beyond the edge with
                the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
    """

    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        value_range=(0, 255),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._set_factor(factor)
        self._set_value_range(value_range)
        self.data_format = backend.standardize_data_format(data_format)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value

        self.supports_jit = False

        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self.value_range_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self.value_range_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        self.value_range = sorted(value_range)

    def _set_factor(self, factor):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f"Received: factor={factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            self._factor = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            self._factor = [-factor, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR + f"Received: factor={factor}"
            )

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: input_number={input_number}"
            )

    """
    Assume an angle ø, then rotation matrix is defined by
    | cos(ø)   -sin(ø)  x_offset |
    | sin(ø)    cos(ø)  y_offset |
    |   0         0         1    |

    This function is returning the 8 elements barring the final 1 as a 1D array
    """

    def _get_rotation_matrix(self, inputs):
        shape = self.backend.core.shape(inputs)
        if len(shape) == 4:
            if self.data_format == "channels_last":
                batch_size = shape[0]
                image_height = shape[1]
                image_width = shape[2]
            else:
                batch_size = shape[1]
                image_height = shape[2]
                image_width = shape[3]
        else:
            batch_size = 1
            if self.data_format == "channels_last":
                image_height = shape[0]
                image_width = shape[1]
            else:
                image_height = shape[1]
                image_width = shape[2]

        lower = self._factor[0] * 2.0 * self.backend.convert_to_tensor(np.pi)
        upper = self._factor[1] * 2.0 * self.backend.convert_to_tensor(np.pi)

        seed_generator = self._get_seed_generator(self.backend._backend)
        angle = self.backend.random.uniform(
            shape=(batch_size,),
            minval=lower,
            maxval=upper,
            seed=seed_generator,
        )

        cos_theta = self.backend.numpy.cos(angle)
        sin_theta = self.backend.numpy.sin(angle)
        image_height = self.backend.core.cast(image_height, cos_theta.dtype)
        image_width = self.backend.core.cast(image_width, cos_theta.dtype)

        x_offset = (
            (image_width - 1)
            - (cos_theta * (image_width - 1) - sin_theta * (image_height - 1))
        ) / 2.0

        y_offset = (
            (image_height - 1)
            - (sin_theta * (image_width - 1) + cos_theta * (image_height - 1))
        ) / 2.0

        outputs = self.backend.numpy.concatenate(
            [
                self.backend.numpy.cos(angle)[:, None],
                -self.backend.numpy.sin(angle)[:, None],
                x_offset[:, None],
                self.backend.numpy.sin(angle)[:, None],
                self.backend.numpy.cos(angle)[:, None],
                y_offset[:, None],
                self.backend.numpy.zeros((batch_size, 2)),
            ],
            axis=1,
        )
        if len(shape) == 3:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        if training:
            rotation_matrix = self._get_rotation_matrix(inputs)
            transformed_image = self.backend.image.affine_transform(
                image=inputs,
                transform=rotation_matrix,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                data_format=self.data_format,
            )
            return transformed_image
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self._factor,
            "value_range": self.value_range,
            "data_format": self.data_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
