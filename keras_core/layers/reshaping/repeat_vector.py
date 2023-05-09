from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.RepeatVector")
class RepeatVector(Layer):
    """Repeats the input n times.

    Example:

    >>> x = keras_core.Input(shape=(32,))
    >>> y = keras_core.layers.RepeatVector(3)(x)
    >>> y.shape
    (None, 3, 32)

    Args:
        n: Integer, repetition factor.

    Input shape:
        2D tensor with shape `(batch_size, features)`.

    Output shape:
        3D tensor with shape `(batch_size, n, features)`.
    """

    def __init__(self, n, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(
                f"Expected an integer value for `n`, got {type(n)}."
            )
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, inputs):
        input_shape = inputs.shape
        reshaped = ops.reshape(inputs, (input_shape[0], 1, input_shape[1]))
        return ops.repeat(reshaped, self.n, axis=1)

    def get_config(self):
        config = {"n": self.n}
        base_config = super().get_config()
        return {**base_config, **config}
