import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export


@keras_export("keras.utils.normalize")
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
    from keras.src import ops

    if isinstance(x, np.ndarray):
        # NumPy input
        norm = np.atleast_1d(np.linalg.norm(x, order, axis))
        norm[norm == 0] = 1

        # axis cannot be `None`
        axis = axis or -1
        return x / np.expand_dims(norm, axis)

    # Backend tensor input
    return ops.nn.normalize(x, axis=axis, order=order)


@keras_export("keras.utils.to_categorical")
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

    >>> a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
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
    >>> loss = keras.ops.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = keras.ops.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    if backend.is_tensor(x):
        input_shape = backend.core.shape(x)
        # Shrink the last dimension if the shape is (..., 1).
        if (
            input_shape is not None
            and len(input_shape) > 1
            and input_shape[-1] == 1
        ):
            newshape = tuple(input_shape[:-1])
            x = backend.numpy.reshape(x, newshape)
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


def encode_categorical_inputs(
    inputs,
    output_mode,
    depth,
    dtype,
    sparse=False,
    count_weights=None,
    backend_module=None,
):
    """Encodes categorical inputs according to output_mode.

    Args:
        inputs: the inputs to encode.
        output_mode: one of `"int"`, `"one_hot"`, `"multi_hot"`, or `"count"`.
        depth: number of classes, this will be the last dimension of the output.
        dtype: the dtype of the output, unless `count_weights` is not `None`.
        sparse: whether the output should be sparse for backends supporting it.
        count_weights: weights to apply if `output_mode` is `"count"`.
        backend_module: the backend to use instead of the curren one.
    Returns: the encoded inputs.
    """
    backend_module = backend_module or backend

    if output_mode == "int":
        return backend_module.cast(inputs, dtype=dtype)

    rank_of_inputs = len(backend_module.shape(inputs))

    # In all cases, we should uprank scalar input to a single sample.
    if rank_of_inputs == 0:
        # We need to update `rank_of_inputs` if necessary.
        inputs = backend_module.numpy.expand_dims(inputs, -1)

    if output_mode == "multi_hot":
        return backend_module.nn.multi_hot(
            inputs, depth, dtype=dtype, sparse=sparse
        )
    elif output_mode == "one_hot":
        input_shape = backend_module.core.shape(inputs)
        # Shrink the last dimension if the shape is (..., 1).
        if (
            input_shape is not None
            and len(input_shape) > 1
            and input_shape[-1] == 1
        ):
            newshape = tuple(input_shape[:-1])
            inputs = backend_module.numpy.reshape(inputs, newshape)
        return backend_module.nn.one_hot(
            inputs, depth, dtype=dtype, sparse=sparse
        )
    elif output_mode == "count":
        # We don't use `ops.bincount` because its output has a dynamic shape
        # (last dimension is the highest value of `inputs`). We implement a
        # narrower use case where `minlength` and `maxlength` (not supported by
        # `ops.bincount`) are the same and static value: `depth`. We also don't
        # need to support indices that are negative or greater than `depth`.
        reduction_axis = 1 if len(inputs.shape) > 1 else 0

        if count_weights is not None:
            dtype = count_weights.dtype
        one_hot_encoding = backend_module.nn.one_hot(
            inputs, depth, dtype=dtype, sparse=sparse
        )
        if count_weights is not None:
            count_weights = backend_module.numpy.expand_dims(count_weights, -1)
            one_hot_encoding = one_hot_encoding * count_weights

        outputs = backend_module.numpy.sum(
            one_hot_encoding,
            axis=reduction_axis,
        )
        return outputs
