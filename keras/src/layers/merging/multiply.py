from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.merging.base_merge import Merge


@keras_export("keras.layers.Multiply")
class Multiply(Merge):
    """Performs elementwise multiplication.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Multiply()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.multiply([x1, x2])`
    >>> y = keras.layers.Multiply()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def _merge_function(self, inputs):
        masks = [getattr(x, "_keras_mask", None) for x in inputs]
        has_output_mask = all(mask is not None for mask in masks)
        output = None
        output_mask = None

        for x, mask in zip(inputs, masks):
            if mask is not None:
                mask = ops.broadcast_to(ops.expand_dims(mask, -1), ops.shape(x))
                # Replace 0s with 1s outside of mask.
                x = ops.where(mask, x, ops.cast(1, x.dtype))
                if has_output_mask:
                    output_mask = (
                        mask
                        if output_mask is None
                        else ops.logical_or(output_mask, mask)
                    )
            output = x if output is None else ops.multiply(output, x)

        if has_output_mask:
            # Replace 1s with 0s outside of mask per standard masking rules.
            output = ops.where(output_mask, output, ops.cast(0, output.dtype))
            output_mask = ops.any(output_mask, axis=-1, keepdims=False)
            output._keras_mask = output_mask
        return output


@keras_export("keras.layers.multiply")
def multiply(inputs, **kwargs):
    """Functional interface to the `keras.layers.Multiply` layer.

    Args:
        inputs: A list of input tensors , all of the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the elementwise product of the inputs with the same
        shape as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.multiply([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> y = keras.layers.multiply([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Multiply(**kwargs)(inputs)
