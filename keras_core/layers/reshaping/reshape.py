import math

from keras_core import operations as ops
from keras_core.layers.layer import Layer


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

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.

        Args:
            input_shape: Shape of tensor being reshaped as a tuple of ints.
            output_shape: Desired shape of the tensor as a tuple of ints. It
                contains at most a single `-1` which indicates a dimension that
                should be derived from the input shape.

        Returns:
            The new output shape as a tuple of ints with a -1 replaced with its
            computed value.

        Raises:
            ValueError: If the total tensor size of the output_shape is
                different than the input_shape, or more than one unknown
                dimension is specified.
        """
        msg = (
            "total size of new tensor must be unchanged, "
            f"input_shape={input_shape},output_shape={output_shape}"
        )

        known_output_size, unknown_dim_index = 1, None
        for index, dim in enumerate(output_shape):
            if dim == -1:
                if unknown_dim_index is None:
                    unknown_dim_index = index
                else:
                    raise ValueError(
                        "There must be at most one unknown dimension in "
                        f"output_shape. Received: output_shape={output_shape}."
                    )
            else:
                known_output_size *= dim

        input_size = math.prod(input_shape)
        if unknown_dim_index is not None:
            if known_output_size == 0 or input_size % known_output_size != 0:
                raise ValueError(msg)
            result = list(output_shape)
            result[unknown_dim_index] = input_size // known_output_size
            return tuple(result)
        elif input_size != known_output_size:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],)
        if None in input_shape[1:]:
            # input shape (partially) unknown? replace -1's with None's
            output_shape += tuple(
                s if s != -1 else None for s in self.target_shape
            )
        else:
            output_shape += self._fix_unknown_dimension(
                input_shape[1:], self.target_shape
            )
        return output_shape

    def call(self, inputs):
        return ops.reshape(inputs, (inputs.shape[0],) + self.target_shape)

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
