from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.UnitNormalization")
class UnitNormalization(Layer):
    """Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:

    >>> data = np.arange(6).reshape(2, 3)
    >>> normalized_data = keras.layers.UnitNormalization()(data)
    >>> print(np.sum(normalized_data[0, :] ** 2)
    1.0

    Args:
        axis: Integer or list/tuple. The axis or axes to normalize across.
            Typically, this is the features axis or axes. The left-out axes are
            typically the batch axis or axes. `-1` is the last dimension
            in the input. Defaults to `-1`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Invalid value for `axis` argument: "
                "expected an int or a list/tuple of ints. "
                f"Received: axis={axis}"
            )
        self.supports_masking = True

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x = ops.cast(inputs, self.compute_dtype)

        square_sum = ops.sum(ops.square(x), axis=self.axis, keepdims=True)
        x_inv_norm = ops.rsqrt(ops.maximum(square_sum, 1e-12))
        return ops.multiply(x, x_inv_norm)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
