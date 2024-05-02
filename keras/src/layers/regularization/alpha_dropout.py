from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.AlphaDropout")
class AlphaDropout(Layer):
    """Applies Alpha Dropout to the input.

    Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
    to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units (SELU) by
    randomly setting activations to the negative saturation value.

    Args:
        rate: Float between 0 and 1. The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        noise_shape: 1D integer tensor representing the shape of the
            binary alpha dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the alpha dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding alpha dropout) or in inference mode
            (doing nothing).
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`rate`. Expected a float value between 0 and 1. "
                f"Received: rate={rate}"
            )
        self.rate = rate
        self.seed = seed
        self.noise_shape = noise_shape
        if rate > 0:
            self.seed_generator = backend.random.SeedGenerator(seed)
        self.supports_masking = True

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            noise_shape = self._get_concrete_noise_shape(
                inputs, self.noise_shape
            )
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            alpha_p = -alpha * scale

            kept_idx = ops.greater_equal(
                ops.random.uniform(noise_shape, seed=self.seed_generator),
                self.rate,
            )
            kept_idx = ops.cast(kept_idx, inputs.dtype)

            # Compute affine transformation parameters
            a = ((1 - self.rate) * (1 + self.rate * alpha_p**2)) ** -0.5
            b = -a * alpha_p * self.rate

            # Apply mask
            x = inputs * kept_idx + alpha_p * (1 - kept_idx)
            return a * x + b

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_concrete_noise_shape(self, inputs, noise_shape):
        if noise_shape is None:
            return ops.shape(inputs)

        concrete_inputs_shape = ops.shape(inputs)
        concrete_noise_shape = []
        for i, value in enumerate(noise_shape):
            concrete_noise_shape.append(
                concrete_inputs_shape[i] if value is None else value
            )
        return concrete_noise_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "rate": self.rate,
            "seed": self.seed,
            "noise_shape": self.noise_shape,
        }
        return {**base_config, **config}
