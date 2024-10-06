"""Legacy Keras 1/2 layers.

AlphaDropout
RandomHeight
RandomWidth
ThresholdedReLU
"""

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras._legacy.layers.AlphaDropout")
class AlphaDropout(Layer):
    """DEPRECATED."""

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.noise_shape = noise_shape
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.supports_masking = True
        self.built = True

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            alpha_p = -alpha * scale

            if self.noise_shape is None:
                noise_shape = tf.shape(inputs)
            else:
                noise_shape = self.noise_shape
            kept_idx = tf.greater_equal(
                backend.random.uniform(noise_shape, seed=self.seed_generator),
                self.rate,
            )
            kept_idx = tf.cast(kept_idx, inputs.dtype)

            # Get affine transformation params
            a = ((1 - self.rate) * (1 + self.rate * alpha_p**2)) ** -0.5
            b = -a * alpha_p * self.rate

            # Apply mask
            x = inputs * kept_idx + alpha_p * (1 - kept_idx)

            # Do affine transformation
            return a * x + b
        return inputs

    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


@keras_export("keras._legacy.layers.RandomHeight")
class RandomHeight(Layer):
    """DEPRECATED."""

    def __init__(self, factor, interpolation="bilinear", seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.height_lower = factor[0]
            self.height_upper = factor[1]
        else:
            self.height_lower = -factor
            self.height_upper = factor

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`factor` argument cannot have an upper bound lesser than the "
                f"lower bound. Received: factor={factor}"
            )
        if self.height_lower < -1.0 or self.height_upper < -1.0:
            raise ValueError(
                "`factor` argument must have values larger than -1. "
                f"Received: factor={factor}"
            )
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)

        def random_height_inputs(inputs):
            """Inputs height-adjusted with random ops."""
            inputs_shape = tf.shape(inputs)
            img_hd = tf.cast(inputs_shape[-3], tf.float32)
            img_wd = inputs_shape[-2]
            height_factor = backend.random.uniform(
                shape=[],
                minval=(1.0 + self.height_lower),
                maxval=(1.0 + self.height_upper),
                seed=self.seed_generator,
            )
            adjusted_height = tf.cast(height_factor * img_hd, tf.int32)
            adjusted_size = tf.stack([adjusted_height, img_wd])
            output = tf.image.resize(
                images=inputs,
                size=adjusted_size,
                method=self.interpolation,
            )
            # tf.resize will output float32 regardless of input type.
            output = tf.cast(output, self.compute_dtype)
            output_shape = inputs.shape.as_list()
            output_shape[-3] = None
            output.set_shape(output_shape)
            return output

        if training:
            return random_height_inputs(inputs)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-3] = None
        return tuple(input_shape)

    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_export("keras._legacy.layers.RandomWidth")
class RandomWidth(Layer):
    """DEPRECATED."""

    def __init__(self, factor, interpolation="bilinear", seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.width_lower = factor[0]
            self.width_upper = factor[1]
        else:
            self.width_lower = -factor
            self.width_upper = factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`factor` argument cannot have an upper bound less than the "
                f"lower bound. Received: factor={factor}"
            )
        if self.width_lower < -1.0 or self.width_upper < -1.0:
            raise ValueError(
                "`factor` argument must have values larger than -1. "
                f"Received: factor={factor}"
            )
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)

        def random_width_inputs(inputs):
            """Inputs width-adjusted with random ops."""
            inputs_shape = tf.shape(inputs)
            img_hd = inputs_shape[-3]
            img_wd = tf.cast(inputs_shape[-2], tf.float32)
            width_factor = backend.random.uniform(
                shape=[],
                minval=(1.0 + self.width_lower),
                maxval=(1.0 + self.width_upper),
                seed=self.seed_generator,
            )
            adjusted_width = tf.cast(width_factor * img_wd, tf.int32)
            adjusted_size = tf.stack([img_hd, adjusted_width])
            output = tf.image.resize(
                images=inputs,
                size=adjusted_size,
                method=self.interpolation,
            )
            # tf.resize will output float32 regardless of input type.
            output = tf.cast(output, self.compute_dtype)
            output_shape = inputs.shape.as_list()
            output_shape[-2] = None
            output.set_shape(output_shape)
            return output

        if training:
            return random_width_inputs(inputs)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-2] = None
        return tuple(input_shape)

    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_export("keras._legacy.layers.ThresholdedReLU")
class ThresholdedReLU(Layer):
    """DEPRECATED."""

    def __init__(self, theta=1.0, **kwargs):
        super().__init__(**kwargs)
        if theta is None:
            raise ValueError(
                "Theta of a Thresholded ReLU layer cannot be None, expecting a "
                f"float. Received: {theta}"
            )
        if theta < 0:
            raise ValueError(
                "The theta value of a Thresholded ReLU layer "
                f"should be >=0. Received: {theta}"
            )
        self.supports_masking = True
        self.theta = tf.convert_to_tensor(theta, dtype=self.compute_dtype)

    def call(self, inputs):
        dtype = self.compute_dtype
        return inputs * tf.cast(tf.greater(inputs, self.theta), dtype)

    def get_config(self):
        config = {"theta": float(self.theta)}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
