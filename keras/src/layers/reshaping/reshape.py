from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.ops import operation_utils


@keras_export("keras.layers.Reshape")
class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

    Args:
        target_shape: Target shape. Tuple of integers, does not include the
            samples dimension (batch size).

    Input shape:
        Arbitrary, although all dimensions in the input shape must be
        known/fixed. Use the keyword argument `input_shape` (tuple of integers,
        does not include the samples/batch size axis) when using this layer as
        the first layer in a model.

    Output shape:
        `(batch_size, *target_shape)`

    Example:

    >>> x = keras.Input(shape=(12,))
    >>> y = keras.layers.Reshape((3, 4))(x)
    >>> y.shape
    (None, 3, 4)

    >>> # also supports shape inference using `-1` as dimension
    >>> y = keras.layers.Reshape((-1, 2, 2))(x)
    >>> y.shape
    (None, 3, 2, 2)
    """

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            *operation_utils.compute_reshape_output_shape(
                input_shape[1:], self.target_shape, "target_shape"
            ),
        )

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            shape=output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def build(self, input_shape):
        sample_output_shape = operation_utils.compute_reshape_output_shape(
            input_shape[1:], self.target_shape, "target_shape"
        )
        self._resolved_target_shape = tuple(
            -1 if d is None else d for d in sample_output_shape
        )
        self.built = True

    def call(self, inputs):
        return ops.reshape(
            inputs, (ops.shape(inputs)[0],) + self._resolved_target_shape
        )

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
