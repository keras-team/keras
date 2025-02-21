from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.RMSNormalization")
class RMSNormalization(Layer):
    """Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value.

    The Keras layer performs the operation as described in
    [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
    by Biao Zhang et al.


    If `scale` is enabled, the layer will scale the normalized outputs via
    a learnable scaling factor.

    So, with scaling enabled, the normalization equations
    are as follows:

    Let the intermediate activations for a mini-batch to be the `inputs`.

    ```python
    rms_normalization(x) = x * rsqrt(mean(square(x))) * scale
    ```

    For example:

    >>> layer = keras.layers.RMSNormalization()
    >>> layer.build([5, 20, 30, 10])
    >>> print(layer.scale.shape)
    (10,)
    >>> layer(np.random.rand(1, 10)).numpy()
    array([[0.35098287, 1.0495652 , 1.4645109 , 1.2944688 , 0.31124955,
            1.2768592 , 1.184331  , 0.17474432, 0.49955517, 1.2428929 ]],
        dtype=float32)

    Args:
        axis: int. The axis on which to perform the normalization.
        epsilon: float. A small number to add to avoid division by zero.
    """

    def __init__(self, axis=-1, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        if isinstance(self.axis, list):
            shape = tuple([input_shape[dim] for dim in self.axis])
        else:
            shape = (input_shape[self.axis],)
            self.axis = [self.axis]

        self.scale = self.add_weight(
            name="scale", shape=shape, initializer="ones"
        )

        self.built = True

    def call(self, x):
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            The RMS-normalized tensor of the same shape (batch_size, input_dim),
            scaled by the learned `scale` parameter.
        """
        return ops.rms_normalization(
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
        }
        base_config = super().get_config()
        return {**base_config, **config}
