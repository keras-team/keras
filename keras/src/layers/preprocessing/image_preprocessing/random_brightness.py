from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomBrightness")
class RandomBrightness(TFDataLayer):
    """A preprocessing layer which randomly adjusts brightness during training.

    This layer will randomly increase/reduce the brightness for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment. A float value will be chosen randomly between
            the limits. When -1.0 is chosen, the output image will be black, and
            when 1.0 is chosen, the image will be fully white.
            When only one float is provided, eg, 0.2,
            then -0.2 will be used for lower bound and 0.2
            will be used for upper bound.
        value_range: Optional list/tuple of 2 floats
            for the lower and upper limit
            of the values of the input data.
            To make no change, use `[0.0, 1.0]`, e.g., if the image input
            has been scaled before this layer. Defaults to `[0.0, 255.0]`.
            The brightness adjustment will be scaled to this range, and the
            output values will be clipped to this range.
        seed: optional integer, for fixed RNG behavior.

    Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
        values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)

    Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
        `factor`. By default, the layer will output floats.
        The output value will be clipped to the range `[0, 255]`,
        the valid range of RGB colors, and
        rescaled based on the `value_range` if needed.

    Example:

    ```python
    random_bright = keras.layers.RandomBrightness(factor=0.2)

    # An image with shape [2, 2, 3]
    image = [[[1, 2, 3], [4 ,5 ,6]], [[7, 8, 9], [10, 11, 12]]]

    # Assume we randomly select the factor to be 0.1, then it will apply
    # 0.1 * 255 to all the channel
    output = random_bright(image, training=True)

    # output will be int64 with 25.5 added to each channel and round down.
    >>> array([[[26.5, 27.5, 28.5]
                [29.5, 30.5, 31.5]]
               [[32.5, 33.5, 34.5]
                [35.5, 36.5, 37.5]]],
              shape=(2, 2, 3), dtype=int64)
    ```
    """

    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    def __init__(self, factor, value_range=(0, 255), seed=None, **kwargs):
        super().__init__(**kwargs)
        self._set_factor(factor)
        self._set_value_range(value_range)
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
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

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        if training:
            return self._randomly_adjust_brightness(inputs)
        else:
            return inputs

    def _randomly_adjust_brightness(self, images):
        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        if rank == 3:
            rgb_delta_shape = (1, 1, 1)
        elif rank == 4:
            # Keep only the batch dim. This will ensure to have same adjustment
            # with in one image, but different across the images.
            rgb_delta_shape = [images_shape[0], 1, 1, 1]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        seed_generator = self._get_seed_generator(self.backend._backend)
        rgb_delta = self.backend.random.uniform(
            minval=self._factor[0],
            maxval=self._factor[1],
            shape=rgb_delta_shape,
            seed=seed_generator,
        )
        rgb_delta = rgb_delta * (self.value_range[1] - self.value_range[0])
        rgb_delta = self.backend.cast(rgb_delta, images.dtype)
        images += rgb_delta
        return self.backend.numpy.clip(
            images, self.value_range[0], self.value_range[1]
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self._factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
