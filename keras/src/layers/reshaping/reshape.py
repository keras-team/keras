from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.ops import operation_utils
import math

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
        target_shape = tuple(target_shape)
        # test validity of target_shape
        if target_shape.count(-1) > 1:
            raise ValueError(
                "The `target_shape` argument must not contain more than one "
                "`-1` value. Received: target_shape={}".format(target_shape)
            )
        self.target_shape = target_shape
        # precalculate all values that might be required
        self.need_explicit_shape_for_batch_size_None = (target_shape.count(-1) == 1)
        self.new_size_no_minus_one = math.prod(
            d for d in target_shape if d != -1
        )

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

    def call(self, inputs):
        target_shape = self.target_shape
        if self.need_explicit_shape_for_batch_size_None and (inputs.shape[0] is None):
            input_nonbatch_shape = tuple(inputs.shape[1:])
            if input_nonbatch_shape.count(None) == 0:
                # If the input shape is fully defined, we can compute the desired target_shape
                if True:
                    inp_nonbatch_size = math.prod(inputs.shape[1:])
                else:
                    inp_nonbatch_size = ops.prod(ops.shape(inputs)[1:])
                target_shape = tuple(d if d != -1 else (inp_nonbatch_size // self.new_size_no_minus_one) for d in self.target_shape)

        return ops.reshape(
            inputs, (ops.shape(inputs)[0],) + target_shape
        )


    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
