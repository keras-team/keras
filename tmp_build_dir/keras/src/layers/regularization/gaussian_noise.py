from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export


@keras_export("keras.layers.GaussianNoise")
class GaussianNoise(layers.Layer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Args:
        stddev: Float, standard deviation of the noise distribution.
        seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding noise) or in inference mode (doing nothing).
    """

    def __init__(self, stddev, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= stddev <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`stddev`. Expected a float value between 0 and 1. "
                f"Received: stddev={stddev}"
            )
        self.stddev = stddev
        self.seed = seed
        if stddev > 0:
            self.seed_generator = backend.random.SeedGenerator(seed)
        self.supports_masking = True

        self._build_at_init()

    def call(self, inputs, training=False):
        if training and self.stddev > 0:
            return inputs + backend.random.normal(
                shape=ops.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=self.compute_dtype,
                seed=self.seed_generator,
            )
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stddev": self.stddev,
            "seed": self.seed,
        }
        return {**base_config, **config}
