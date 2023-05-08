from keras_core import operations as ops
from keras_core.layers.layer import Layer
from keras_core.operations import operation_utils


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

    >>> # as first layer in a Sequential model
    >>> model = keras_core.Sequential()
    >>> model.add(keras_core.layers.Reshape((3, 4), input_shape=(12,)))
    >>> # model.output_shape == (None, 3, 4), `None` is the batch size.
    >>> model.output_shape
    (None, 3, 4)

    >>> # as intermediate layer in a Sequential model
    >>> model.add(keras_core.layers.Reshape((6, 2)))
    >>> model.output_shape
    (None, 6, 2)

    >>> # also supports shape inference using `-1` as dimension
    >>> model.add(keras_core.layers.Reshape((-1, 2, 2)))
    >>> model.output_shape
    (None, 3, 2, 2)
    """

    def __init__(self, target_shape, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            *operation_utils.compute_reshape_output_shape(
                input_shape[1:], self.target_shape, "target_shape"
            ),
        )

    def call(self, inputs):
        return ops.reshape(inputs, (inputs.shape[0],) + self.target_shape)

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
