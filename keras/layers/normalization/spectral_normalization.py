from keras import initializers
from keras import ops
from keras.api_export import keras_export
from keras.layers import Wrapper
from keras.layers.input_spec import InputSpec
from keras.utils.numerical_utils import normalize


@keras_export("keras.layers.SpectralNormalization")
class SpectralNormalization(Wrapper):
    """Performs spectral normalization on the weights of a target layer.

    This wrapper controls the Lipschitz constant of the weights of a layer by
    constraining their spectral norm, which can stabilize the training of GANs.

    Args:
        layer: A `keras.layers.Layer` instance that
            has either a `kernel` (e.g. `Conv2D`, `Dense`...)
            or an `embeddings` attribute (`Embedding` layer).
        power_iterations: int, the number of iterations during normalization.
        **kwargs: Base wrapper keyword arguments.

    Examples:

    Wrap `keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    (1, 9, 9, 2)

    Wrap `keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    (1, 10, 10, 10)

    Reference:

    - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero. Received: "
                f"`power_iterations={power_iterations}`"
            )
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        self.input_spec = InputSpec(shape=[None] + list(input_shape[1:]))

        if hasattr(self.layer, "kernel"):
            self.kernel = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.kernel = self.layer.embeddings
        else:
            raise ValueError(
                f"{type(self.layer).__name__} object has no attribute 'kernel' "
                "nor 'embeddings'"
            )

        self.kernel_shape = self.kernel.shape

        self.vector_u = self.add_weight(
            shape=(1, self.kernel_shape[-1]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="vector_u",
            dtype=self.kernel.dtype,
        )

    def call(self, inputs, training=False):
        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        weights = ops.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        vector_u = self.vector_u

        # check for zeroes weights
        if not all([w == 0.0 for w in weights]):
            for _ in range(self.power_iterations):
                vector_v = normalize(
                    ops.matmul(vector_u, ops.transpose(weights)), axis=None
                )
                vector_u = normalize(ops.matmul(vector_v, weights), axis=None)
            # vector_u = tf.stop_gradient(vector_u)
            # vector_v = tf.stop_gradient(vector_v)
            sigma = ops.matmul(
                ops.matmul(vector_v, weights), ops.transpose(vector_u)
            )
            self.vector_u.assign(ops.cast(vector_u, self.vector_u.dtype))
            self.kernel.assign(
                ops.cast(
                    ops.reshape(self.kernel / sigma, self.kernel_shape),
                    self.kernel.dtype,
                )
            )

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
