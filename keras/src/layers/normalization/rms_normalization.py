from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.RMSNormalization")
class RMSNormalization(Layer):
    """Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value and applies
    a learned scaling factor.

    Args:
        input_dim: int. The dimensionality of the input tensor.
    """

    def __init__(self, input_dim, axis=-1, epsilon=1e-6):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.scale = self.add_weight(
            name="scale", shape=(input_dim,), initializer="ones"
        )

    def call(self, x):
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            The RMS-normalized tensor of the same shape (batch_size, input_dim),
            scaled by the learned `scale` parameter.
        """
        return ops.rms_norm(
            x, scale=self.scale, axis=self.axis, epsilon=self.epsilon
        )

    def compute_output_shape(self, input_shape):
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = self.axis

        for axis in axes:
            if axis >= len(input_shape) or axis < -len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for "
                    f"input shape {input_shape}. "
                    f"Received: axis={self.axis}"
                )
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "scale": self.scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}
