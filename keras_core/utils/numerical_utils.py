import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.utils.normalize")
def normalize(x, axis=-1, order=2):
    """Normalizes an array.

    If the input is a NumPy array, a NumPy array will be returned.
    If it's a backend tensor, a backend tensor will be returned.

    Args:
        x: Array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. `order=2` for L2 norm).

    Returns:
        A normalized copy of the array.
    """
    from keras_core import ops

    if not isinstance(order, int) or not order >= 1:
        raise ValueError(
            "Argument `order` must be an int >= 1. " f"Received: order={order}"
        )
    if isinstance(x, np.ndarray):
        # NumPy input
        norm = np.atleast_1d(np.linalg.norm(x, order, axis))
        norm[norm == 0] = 1

        # axis cannot be `None`
        axis = axis or -1
        return x / np.expand_dims(norm, axis)

    # Backend tensor input
    if len(x.shape) == 0:
        x = ops.expand_dims(x, axis=0)
    epsilon = backend.epsilon()
    if order == 2:
        power_sum = ops.sum(ops.square(x), axis=axis, keepdims=True)
        norm = ops.reciprocal(ops.sqrt(ops.maximum(power_sum, epsilon)))
    else:
        power_sum = ops.sum(ops.power(x, order), axis=axis, keepdims=True)
        norm = ops.reciprocal(
            ops.power(ops.maximum(power_sum, epsilon), 1.0 / order)
        )
    return ops.multiply(x, norm)


@keras_core_export("keras_core.utils.to_categorical")
def to_categorical(x, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = keras_core.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = np.array([.9, .04, .03, .03,
    ...               .3, .45, .15, .13,
    ...               .04, .01, .94, .05,
    ...               .12, .21, .5, .17],
    ...               shape=[4, 4])
    >>> loss = keras_core.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = keras_core.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    if backend.is_tensor(x):
        return backend.nn.one_hot(x, num_classes)
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
