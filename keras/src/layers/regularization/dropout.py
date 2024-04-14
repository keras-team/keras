from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Dropout")
class Dropout(Layer):
    """Applies dropout to the input.

    The `Dropout` layer randomly sets input units to 0 with a frequency of
    `rate` at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by `1 / (1 - rate)` such that the sum over
    all inputs is unchanged.

    Note that the `Dropout` layer only applies when `training` is set to `True`
    in `call()`, such that no values are dropped during inference.
    When using `model.fit`, `training` will be appropriately set to `True`
    automatically. In other contexts, you can set the argument explicitly
    to `True` when calling the layer.

    (This is in contrast to setting `trainable=False` for a `Dropout` layer.
    `trainable` does not affect the layer's behavior, as `Dropout` does
    not have any variables/weights that can be frozen during training.)

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
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
            return backend.random.dropout(
                inputs,
                self.rate,
                noise_shape=self.noise_shape,
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
            "noise_shape": self.noise_shape,
        }
        return {**base_config, **config}
