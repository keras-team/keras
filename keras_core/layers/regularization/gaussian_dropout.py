import math

from keras_core import backend
from keras_core import layers
from keras_core import operations as ops
from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.layers.GaussianDropout")
class GaussianDropout(layers.Layer):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Args:
        rate: Float, drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(
        self, rate, noise_shape=None, seed=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`rate`. Expected a float value between 0 and 1. "
                f"Received: rate={rate}"
            )
        self.rate = rate
        self.seed = seed
        self.noise_shape = noise_shape
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.supports_masking = True

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            stddev = math.sqrt(self.rate / (1.0 - self.rate))
            return inputs * backend.random.normal(
                shape=ops.shape(inputs),
                mean=1.0,
                stddev=stddev,
                seed=self.seed_generator,
            )
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "rate": self.rate,
            "seed": self.seed,
        }
        return {**base_config, **config}
