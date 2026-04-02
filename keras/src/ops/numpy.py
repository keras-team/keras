import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common.backend_utils import (  # noqa: E501
    canonicalize_axis as standardize_axis,
)
from keras.src.utils.argument_validation import standardize_tuple


@keras_export("keras.ops.abs")
def absolute(x):
    """Return the absolute value of the input element-wise.

    This function is similar to `numpy.abs`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([-1.2, 1.2])
    >>> keras.ops.abs(x)
    array([1.2, 1.2], dtype=float32)
    """
    return backend.numpy.absolute(x)


@keras_export("keras.ops.abs")
def abs(x):
    """Return the absolute value of the input element-wise.

    This function is similar to `numpy.abs`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([-1.2, 1.2])
    >>> keras.ops.abs(x)
    array([1.2, 1.2], dtype=float32)
    """
    return backend.numpy.absolute(x)


@keras_export("keras.ops.all")
def all(x, axis=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to `True`.

    This function is similar to `numpy.all`.

    Args:
        x: A tensor or a Python number.
        axis: Axis or axes along which a logical AND reduction is performed.
            The default is to perform a logical AND over all the dimensions
            of the input array.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.all(x)
    array(True)

    >>> x = keras.ops.convert_to_tensor([0, 1, 2])
    >>> keras.ops.all(x)
    array(False)

    >>> x = keras.ops.convert_to_tensor([[True, True], [True, False]])
    >>> keras.ops.all(x, axis=0)
    array([ True, False])
    """
    return backend.numpy.all(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.any")
def any(x, axis=None, keepdims=False):
    """Test whether any array elements along a given axis evaluate to `True`.

    This function is similar to `numpy.any`.

    Args:
        x: A tensor or a Python number.
        axis: Axis or axes along which a logical OR reduction is performed.
            The default is to perform a logical OR over all the dimensions
            of the input array.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 0, 0])
    >>> keras.ops.any(x)
    array(True)

    >>> x = keras.ops.convert_to_tensor([0, 0, 0])
    >>> keras.ops.any(x)
    array(False)

    >>> x = keras.ops.convert_to_tensor([[True, True], [True, False]])
    >>> keras.ops.any(x, axis=0)
    array([ True, True])
    """
    return backend.numpy.any(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.append")
def append(x1, x2, axis=None):
    """Append values to the end of a tensor.

    This function is similar to `numpy.append`.

    Args:
        x1: A tensor.
        x2: A tensor.
        axis: The axis along which `x2` is appended to `x1`. If `None`,
            both tensors are flattened before use.

    Returns:
        A tensor with `x2` appended to `x1`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.append(x1, x2)
    array([1, 2, 3, 4, 5, 6], dtype=int32)

    >>> x1 = keras.ops.convert_to_tensor([[1, 2], [3, 4]])
    >>> x2 = keras.ops.convert_to_tensor([[5, 6]])
    >>> keras.ops.append(x1, x2, axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)
    """
    return backend.numpy.append(x1, x2, axis=axis)


@keras_export("keras.ops.arange")
def arange(start, stop=None, step=1, dtype=None):
    """Return evenly spaced values within a given interval.

    This function is similar to `numpy.arange`.

    Args:
        start: Start of interval. The interval includes this value.
        stop: End of interval. The interval does not include this value,
            except in some cases where `step` is not an integer and
            floating point round-off affects the length of `out`.
        step: Spacing between values.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `start`, `stop` and `step`.

    Returns:
        A tensor of evenly spaced values.

    Examples:
    >>> keras.ops.arange(3)
    array([0, 1, 2], dtype=int32)
    >>> keras.ops.arange(3.0)
    array([0., 1., 2.], dtype=float32)
    >>> keras.ops.arange(3, 7)
    array([3, 4, 5, 6], dtype=int32)
    >>> keras.ops.arange(3, 7, 2)
    array([3, 5], dtype=int32)
    """
    return backend.numpy.arange(start, stop, step, dtype=dtype)


@keras_export("keras.ops.arccos")
def arccos(x):
    """Return the trigonometric inverse cosine of the input element-wise.

    This function is similar to `numpy.arccos`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the angle of the complex argument.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arccos(x)
    array([0.       , 3.1415927], dtype=float32)
    """
    return backend.numpy.arccos(x)


@keras_export("keras.ops.arccosh")
def arccosh(x):
    """Return the inverse hyperbolic cosine of the input element-wise.

    This function is similar to `numpy.arccosh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the inverse hyperbolic cosine of the input.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.arccosh(x)
    array([0.       , 1.3169579, 1.7627472], dtype=float32)
    """
    return backend.numpy.arccosh(x)


@keras_export("keras.ops.arcsin")
def arcsin(x):
    """Return the trigonometric inverse sine of the input element-wise.

    This function is similar to `numpy.arcsin`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the inverse sine of the input.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arcsin(x)
    array([ 1.5707964, -1.5707964], dtype=float32)
    """
    return backend.numpy.arcsin(x)


@keras_export("keras.ops.arcsinh")
def arcsinh(x):
    """Return the inverse hyperbolic sine of the input element-wise.

    This function is similar to `numpy.arcsinh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the inverse hyperbolic sine of the input.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.arcsinh(x)
    array([0.8813736, 1.4436355, 1.8184465], dtype=float32)
    """
    return backend.numpy.arcsinh(x)


@keras_export("keras.ops.arctan")
def arctan(x):
    """Return the trigonometric inverse tangent of the input element-wise.

    This function is similar to `numpy.arctan`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the inverse tangent of the input.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arctan(x)
    array([ 0.7853982, -0.7853982], dtype=float32)
    """
    return backend.numpy.arctan(x)


@keras_export("keras.ops.arctan2")
def arctan2(y, x):
    """Return the element-wise arc tangent of `y / x` choosing the quadrant correctly.

    This function is similar to `numpy.arctan2`.

    Args:
        y: A tensor or a Python number.
        x: A tensor or a Python number.

    Returns:
        A tensor with the arc tangent of `y / x`.

    Examples:
    >>> y = keras.ops.convert_to_tensor([1, -1])
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arctan2(y, x)
    array([ 0.7853982, -2.3561945], dtype=float32)
    """
    return backend.numpy.arctan2(y, x)


@keras_export("keras.ops.arctanh")
def arctanh(x):
    """Return the inverse hyperbolic tangent of the input element-wise.

    This function is similar to `numpy.arctanh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor with the inverse hyperbolic tangent of the input.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0.5, 0.2, 0.1])
    >>> keras.ops.arctanh(x)
    array([0.54930615, 0.20273255, 0.10033535], dtype=float32)
    """
    return backend.numpy.arctanh(x)


@keras_export("keras.ops.argmax")
def argmax(x, axis=None):
    """Return the indices of the maximum values along an axis.

    This function is similar to `numpy.argmax`.

    Args:
        x: A tensor.
        axis: The axis along which to find the indices. If `None`, the
            array is flattened before use.

    Returns:
        A tensor with the indices of the maximum values.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.argmax(x)
    array(5)
    >>> keras.ops.argmax(x, axis=0)
    array([1, 1, 1])
    >>> keras.ops.argmax(x, axis=1)
    array([2, 2])
    """
    return backend.numpy.argmax(x, axis=axis)


@keras_export("keras.ops.argmin")
def argmin(x, axis=None):
    """Return the indices of the minimum values along an axis.

    This function is similar to `numpy.argmin`.

    Args:
        x: A tensor.
        axis: The axis along which to find the indices. If `None`, the
            array is flattened before use.

    Returns:
        A tensor with the indices of the minimum values.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.argmin(x)
    array(0)
    >>> keras.ops.argmin(x, axis=0)
    array([0, 0, 0])
    >>> keras.ops.argmin(x, axis=1)
    array([0, 0])
    """
    return backend.numpy.argmin(x, axis=axis)


@keras_export("keras.ops.argsort")
def argsort(x, axis=-1, direction="ascending"):
    """Return the indices that would sort a tensor.

    This function is similar to `numpy.argsort`.

    Args:
        x: A tensor.
        axis: The axis along which to sort. If `None`, the array is
            flattened before use.
        direction: The direction to sort. `"ascending"` or `"descending"`.

    Returns:
        A tensor of indices that sort `x` along the specified `axis`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([3, 1, 2])
    >>> keras.ops.argsort(x)
    array([1, 2, 0])

    >>> x = keras.ops.convert_to_tensor([[0, 3], [2, 1]])
    >>> keras.ops.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])
    >>> keras.ops.argsort(x, axis=1)
    array([[0, 1],
           [1, 0]])
    """
    return backend.numpy.argsort(x, axis=axis, direction=direction)


@keras_export("keras.ops.array")
def array(x, dtype=None):
    """Create a tensor.

    This function is similar to `numpy.array`.

    Args:
        x: A tensor, a list, a tuple or a Python number.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `x`.

    Returns:
        A tensor.

    Examples:
    >>> keras.ops.array([1, 2, 3])
    array([1, 2, 3], dtype=int32)
    >>> keras.ops.array([1, 2, 3], dtype="float32")
    array([1., 2., 3.], dtype=float32)
    """
    return backend.numpy.array(x, dtype=dtype)


@keras_export("keras.ops.average")
def average(x, axis=None, weights=None):
    """Compute the weighted average along the specified axis.

    This function is similar to `numpy.average`.

    Args:
        x: A tensor.
        axis: The axis along which to average `x`. If `None`, the array
            is flattened before use.
        weights: A tensor of weights associated with the values in `x`.
            Each value in `x` contributes to the average according to its
            associated weight. The weights array can either be 1-D (in which
            case its length must be the size of `x` along the given axis) or
            of the same shape as `x`. If `weights=None`, then all data in `x`
            are assumed to have a weight equal to one.

    Returns:
        A tensor with the weighted average of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.average(x)
    array(2.5, dtype=float64)
    >>> keras.ops.average(x, axis=0)
    array([1.5, 2.5, 3.5], dtype=float64)
    >>> keras.ops.average(x, weights=[1, 2, 3])
    array(3.1666666666666665, dtype=float64)
    """
    return backend.numpy.average(x, axis=axis, weights=weights)


@keras_export("keras.ops.bincount")
def bincount(x, weights=None, minlength=0):
    """Count number of occurrences of each value in a tensor of non-negative ints.

    This function is similar to `numpy.bincount`.

    Args:
        x: A 1-D tensor of non-negative integers.
        weights: A tensor of weights, an array of the same shape as `x`.
        minlength: A minimum number of bins for the output tensor.

    Returns:
        A tensor of length `max(x) + 1` if `x` is non-empty, or length 0
        otherwise.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 1, 1, 2, 2, 2])
    >>> keras.ops.bincount(x)
    array([1, 2, 3], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([0, 1, 1, 2, 2, 2])
    >>> weights = keras.ops.convert_to_tensor([0.5, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> keras.ops.bincount(x, weights=weights)
    array([0.5, 0.5, 1.5], dtype=float32)
    """
    return backend.numpy.bincount(x, weights=weights, minlength=minlength)


@keras_export("keras.ops.broadcast_to")
def broadcast_to(x, shape):
    """Broadcast a tensor to a new shape.

    This function is similar to `numpy.broadcast_to`.

    Args:
        x: A tensor.
        shape: The shape of the new tensor.

    Returns:
        A tensor with the new shape.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]], dtype=int32)
    """
    return backend.numpy.broadcast_to(x, shape)


@keras_export("keras.ops.ceil")
def ceil(x):
    """Return the ceiling of the input, element-wise.

    This function is similar to `numpy.ceil`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1.1, 2.2, 3.3])
    >>> keras.ops.ceil(x)
    array([2., 3., 4.], dtype=float32)
    """
    return backend.numpy.ceil(x)


@keras_export("keras.ops.clip")
def clip(x, x_min, x_max):
    """Clip (limit) the values in a tensor.

    This function is similar to `numpy.clip`.

    Args:
        x: A tensor or a Python number.
        x_min: The minimum value.
        x_max: The maximum value.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.arange(10)
    >>> keras.ops.clip(x, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8], dtype=int32)
    """
    return backend.numpy.clip(x, x_min, x_max)


@keras_export("keras.ops.concatenate")
def concatenate(x, axis=0):
    """Join a sequence of tensors along an existing axis.

    This function is similar to `numpy.concatenate`.

    Args:
        x: A sequence of tensors.
        axis: The axis along which the tensors will be joined. If `None`,
            tensors are flattened before use.

    Returns:
        A tensor with the concatenated tensors.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([[1, 2], [3, 4]])
    >>> x2 = keras.ops.convert_to_tensor([[5, 6]])
    >>> keras.ops.concatenate([x1, x2], axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)
    """
    return backend.numpy.concatenate(x, axis=axis)


@keras_export("keras.ops.conj")
def conjugate(x):
    """Return the complex conjugate, element-wise.

    This function is similar to `numpy.conjugate`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1+2j, 1-2j])
    >>> keras.ops.conj(x)
    array([1.-2.j, 1.+2.j])
    """
    return backend.numpy.conjugate(x)


@keras_export("keras.ops.copy")
def copy(x):
    """Return a copy of the tensor.

    This function is similar to `numpy.copy`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A copy of `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> y = keras.ops.copy(x)
    >>> x is y
    False
    """
    return backend.numpy.copy(x)


@keras_export("keras.ops.cos")
def cos(x):
    """Return the trigonometric cosine, element-wise.

    This function is similar to `numpy.cos`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 3.14159265])
    >>> keras.ops.cos(x)
    array([ 1., -1.], dtype=float32)
    """
    return backend.numpy.cos(x)


@keras_export("keras.ops.cosh")
def cosh(x):
    """Return the hyperbolic cosine, element-wise.

    This function is similar to `numpy.cosh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 1, -1])
    >>> keras.ops.cosh(x)
    array([1.       , 1.5430806, 1.5430806], dtype=float32)
    """
    return backend.numpy.cosh(x)


@keras_export("keras.ops.count_nonzero")
def count_nonzero(x, axis=None):
    """Count the number of non-zero values in a tensor.

    This function is similar to `numpy.count_nonzero`.

    Args:
        x: A tensor.
        axis: The axis along which to count non-zeros. If `None`, the
            array is flattened before use.

    Returns:
        An integer or a tensor of integers.

    Examples:
    >>> x = keras.ops.convert_to_tensor([[0, 1, 2], [1, 1, 0]])
    >>> keras.ops.count_nonzero(x)
    array(4)
    >>> keras.ops.count_nonzero(x, axis=0)
    array([1, 2, 1])
    """
    return backend.numpy.count_nonzero(x, axis=axis)


@keras_export("keras.ops.cross")
def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Return the cross product of two (arrays of) vectors.

    This function is similar to `numpy.cross`.

    Args:
        x1: First vector(s).
        x2: Second vector(s).
        axisa: Axis of `x1` that defines the vector(s). Defaults to the last axis.
        axisb: Axis of `x2` that defines the vector(s). Defaults to the last axis.
        axisc: Axis of `output` that contains the cross product vector(s).
            Defaults to the last axis.
        axis: If defined, the axis of `x1`, `x2` and `output` that defines the
            vector(s) and cross product(s). Overrides `axisa`, `axisb`
            and `axisc`.

    Returns:
        A tensor with the cross product of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.cross(x1, x2)
    array([-3,  6, -3], dtype=int32)
    """
    return backend.numpy.cross(x1, x2, axisa, axisb, axisc, axis)


@keras_export("keras.ops.cumprod")
def cumprod(x, axis=None):
    """Return the cumulative product of the elements along a given axis.

    This function is similar to `numpy.cumprod`.

    Args:
        x: A tensor.
        axis: The axis along which the cumulative product is computed. If
            `None`, the array is flattened before use.

    Returns:
        A tensor with the cumulative product of `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.cumprod(x)
    array([1, 2, 6], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> keras.ops.cumprod(x, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]], dtype=int32)
    """
    return backend.numpy.cumprod(x, axis=axis)


@keras_export("keras.ops.cumsum")
def cumsum(x, axis=None):
    """Return the cumulative sum of the elements along a given axis.

    This function is similar to `numpy.cumsum`.

    Args:
        x: A tensor.
        axis: The axis along which the cumulative sum is computed. If
            `None`, the array is flattened before use.

    Returns:
        A tensor with the cumulative sum of `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.cumsum(x)
    array([1, 3, 6], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> keras.ops.cumsum(x, axis=0)
    array([[1, 2, 3],
           [5, 7, 9]], dtype=int32)
    """
    return backend.numpy.cumsum(x, axis=axis)


@keras_export("keras.ops.diag")
def diag(x, k=0):
    """Extract a diagonal or construct a diagonal array.

    This function is similar to `numpy.diag`.

    Args:
        x: A tensor. If `x` is a 2-D array, return a 1-D array with the
            diagonal elements. If `x` is a 1-D array, return a 2-D array
            with `x` on the `k`-th diagonal.
        k: The diagonal to extract. Defaults to 0.

    Returns:
        A tensor with the extracted diagonal(s) or a diagonal array.

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> keras.ops.diag(x)
    array([0, 4, 8], dtype=int32)
    >>> keras.ops.diag(x, k=1)
    array([1, 5], dtype=int32)

    >>> x = keras.ops.arange(3)
    >>> keras.ops.diag(x)
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 2]], dtype=int32)
    """
    return backend.numpy.diag(x, k=k)


@keras_export("keras.ops.diagonal")
def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    This function is similar to `numpy.diagonal`.

    If `x` is 2-D, returns the diagonal of `x` with the given offset,
    i.e., the collection of elements of the form `x[i, i+offset]`.
    If `x` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned. The shape of the resulting array can be determined by removing
    `axis1` and `axis2` and appending a dimension to the result.

    Args:
        x: A tensor.
        offset: Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal (0).
        axis1: Axis to be used as the first axis of the 2-D sub-arrays
            from which the diagonals should be taken. Defaults to first axis (0).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays
            from which the diagonals should be taken. Defaults to second axis (1).

    Returns:
        A tensor with the extracted diagonal(s).

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]], dtype=int32)
    >>> keras.ops.diagonal(x)
    array([0, 4, 8], dtype=int32)
    >>> keras.ops.diagonal(x, offset=1)
    array([1, 5], dtype=int32)
    >>> keras.ops.diagonal(x, offset=-1)
    array([3, 7], dtype=int32)

    >>> x = keras.ops.zeros((3, 4, 5))
    >>> keras.ops.diagonal(x).shape
    (3, 3)
    """
    if axis1 == axis2:
        raise ValueError(
            "axis1 and axis2 cannot be the same. "
            f"Received: axis1={axis1}, axis2={axis2}"
        )
    return backend.numpy.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@keras_export("keras.ops.diff")
def diff(x, n=1, axis=-1):
    """Calculate the n-th discrete difference along the given axis.

    This function is similar to `numpy.diff`.

    Args:
        x: A tensor.
        n: The number of times values are differenced. If zero, the input
            is returned as-is.
        axis: The axis along which the difference is taken, default is the
            last axis.

    Returns:
        A tensor with the n-th discrete difference of `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 4, 7, 0])
    >>> keras.ops.diff(x)
    array([ 1,  2,  3, -7], dtype=int32)
    >>> keras.ops.diff(x, n=2)
    array([ 1,  1, -10], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> keras.ops.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]], dtype=int32)
    """
    return backend.numpy.diff(x, n=n, axis=axis)


@keras_export("keras.ops.digitize")
def digitize(x, bins):
    """Return the indices of the bins to which each value in input array belongs.

    This function is similar to `numpy.digitize`.

    Args:
        x: A tensor of values to be digitized.
        bins: A 1-D tensor of bins. It has to be sorted in non-decreasing
            order.

    Returns:
        A tensor of indices of the same shape as `x`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0.2, 6.4, 3.0, 1.6])
    >>> bins = keras.ops.convert_to_tensor([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> keras.ops.digitize(x, bins)
    array([1, 4, 3, 2], dtype=int32)
    """
    return backend.numpy.digitize(x, bins)


@keras_export("keras.ops.dot")
def dot(x, y):
    """Return the dot product of two tensors.

    This function is similar to `numpy.dot`.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        A tensor with the dot product of `x` and `y`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> y = keras.ops.arange(6).reshape((3, 2))
    >>> keras.ops.dot(x, y)
    array([[10, 13],
           [28, 40]], dtype=int32)

    >>> x = keras.ops.arange(3)
    >>> y = keras.ops.arange(3)
    >>> keras.ops.dot(x, y)
    array(5)
    """
    return backend.numpy.dot(x, y)


@keras_export("keras.ops.empty")
def empty(shape, dtype=None):
    """Return a new tensor of given shape and type, without initializing entries.

    This function is similar to `numpy.empty`.

    Args:
        shape: The shape of the new tensor.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A tensor of uninitialized data with the given shape and dtype.

    Examples:
    >>> keras.ops.empty((2, 3))
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    """
    return backend.numpy.empty(shape, dtype=dtype)


@keras_export("keras.ops.equal")
def equal(x1, x2):
    """Return the truth value of `x1 == x2` element-wise.

    This function is similar to `numpy.equal`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 2, 4])
    >>> keras.ops.equal(x1, x2)
    array([ True,  True, False])
    """
    return backend.numpy.equal(x1, x2)


@keras_export("keras.ops.exp")
def exp(x):
    """Return the exponential of the input, element-wise.

    This function is similar to `numpy.exp`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.exp(x)
    array([ 2.7182817,  7.389056 , 20.085537 ], dtype=float32)
    """
    return backend.numpy.exp(x)


@keras_export("keras.ops.expand_dims")
def expand_dims(x, axis):
    """Expand the shape of a tensor.

    This function is similar to `numpy.expand_dims`.

    Args:
        x: A tensor.
        axis: The axis to expand.

    Returns:
        A tensor with the expanded shape.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.expand_dims(x, axis=0)
    array([[1, 2, 3]], dtype=int32)
    >>> keras.ops.expand_dims(x, axis=1)
    array([[1],
           [2],
           [3]], dtype=int32)
    """
    return backend.numpy.expand_dims(x, axis)


@keras_export("keras.ops.expm1")
def expm1(x):
    """Return `exp(x) - 1`, element-wise.

    This function is similar to `numpy.expm1`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.expm1(x)
    array([ 1.7182819,  6.389056 , 19.085537 ], dtype=float32)
    """
    return backend.numpy.expm1(x)


@keras_export("keras.ops.eye")
def eye(N, M=None, k=0, dtype=None):
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    This function is similar to `numpy.eye`.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If `None`, defaults to `N`.
        k: Index of the diagonal. 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal, and a
            negative value to a lower diagonal.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A 2-D tensor with ones on the diagonal and zeros elsewhere.

    Examples:
    >>> keras.ops.eye(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    >>> keras.ops.eye(3, k=1)
    array([[0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.]], dtype=float32)
    """
    return backend.numpy.eye(N, M=M, k=k, dtype=dtype)


@keras_export("keras.ops.flip")
def flip(x, axis=None):
    """Reverse the order of elements in a tensor along the given axis.

    This function is similar to `numpy.flip`.

    Args:
        x: A tensor.
        axis: The axis along which to flip. If `None`, the array is
            flattened before use.

    Returns:
        A tensor with the flipped elements.

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> keras.ops.flip(x, axis=0)
    array([[6, 7, 8],
           [3, 4, 5],
           [0, 1, 2]], dtype=int32)
    >>> keras.ops.flip(x, axis=1)
    array([[2, 1, 0],
           [5, 4, 3],
           [8, 7, 6]], dtype=int32)
    """
    return backend.numpy.flip(x, axis=axis)


@keras_export("keras.ops.floor")
def floor(x):
    """Return the floor of the input, element-wise.

    This function is similar to `numpy.floor`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1.1, 2.2, 3.3])
    >>> keras.ops.floor(x)
    array([1., 2., 3.], dtype=float32)
    """
    return backend.numpy.floor(x)


@keras_export("keras.ops.full")
def full(shape, fill_value, dtype=None):
    """Return a new tensor of given shape and type, filled with `fill_value`.

    This function is similar to `numpy.full`.

    Args:
        shape: The shape of the new tensor.
        fill_value: The value to fill the new tensor with.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `fill_value`.

    Returns:
        A tensor of `fill_value` with the given shape and dtype.

    Examples:
    >>> keras.ops.full((2, 3), 10)
    array([[10, 10, 10],
           [10, 10, 10]], dtype=int32)
    """
    return backend.numpy.full(shape, fill_value, dtype=dtype)


@keras_export("keras.ops.full_like")
def full_like(x, fill_value, dtype=None):
    """Return a new tensor with the same shape and type as a given tensor.

    This function is similar to `numpy.full_like`.

    Args:
        x: A tensor.
        fill_value: The value to fill the new tensor with.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `x`.

    Returns:
        A tensor of `fill_value` with the same shape and type as `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.full_like(x, 10)
    array([[10, 10, 10],
           [10, 10, 10]], dtype=int32)
    """
    return backend.numpy.full_like(x, fill_value, dtype=dtype)


@keras_export("keras.ops.greater")
def greater(x1, x2):
    """Return the truth value of `x1 > x2` element-wise.

    This function is similar to `numpy.greater`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.greater(x1, x2)
    array([False, False,  True])
    """
    return backend.numpy.greater(x1, x2)


@keras_export("keras.ops.greater_equal")
def greater_equal(x1, x2):
    """Return the truth value of `x1 >= x2` element-wise.

    This function is similar to `numpy.greater_equal`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.greater_equal(x1, x2)
    array([ True, False,  True])
    """
    return backend.numpy.greater_equal(x1, x2)


@keras_export("keras.ops.identity")
def identity(n, dtype=None):
    """Return the identity array.

    This function is similar to `numpy.identity`.

    Args:
        n: Number of rows (and columns) in `n x n` output.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A `n x n` 2-D tensor with its main diagonal set to ones, and
        all other elements `0`.

    Examples:
    >>> keras.ops.identity(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    """
    return backend.numpy.identity(n, dtype=dtype)


@keras_export("keras.ops.imag")
def imag(x):
    """Return the imaginary part of the complex argument.

    This function is similar to `numpy.imag`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1+2j, 1-2j])
    >>> keras.ops.imag(x)
    array([ 2., -2.], dtype=float32)
    """
    return backend.numpy.imag(x)


@keras_export("keras.ops.isclose")
def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return whether two tensors are element-wise close.

    This function is similar to `numpy.isclose`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.
        equal_nan: Whether to compare `NaN` as equal.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 2, 4])
    >>> keras.ops.isclose(x1, x2)
    array([ True,  True, False])
    """
    return backend.numpy.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


@keras_export("keras.ops.isfinite")
def isfinite(x):
    """Return whether the input is finite, element-wise.

    This function is similar to `numpy.isfinite`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, float('inf'), 2, float('nan')])
    >>> keras.ops.isfinite(x)
    array([ True, False,  True, False])
    """
    return backend.numpy.isfinite(x)


@keras_export("keras.ops.isinf")
def isinf(x):
    """Return whether the input is an infinity, element-wise.

    This function is similar to `numpy.isinf`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, float('inf'), 2, float('nan')])
    >>> keras.ops.isinf(x)
    array([False,  True, False, False])
    """
    return backend.numpy.isinf(x)


@keras_export("keras.ops.isnan")
def isnan(x):
    """Return whether the input is a `NaN`, element-wise.

    This function is similar to `numpy.isnan`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, float('inf'), 2, float('nan')])
    >>> keras.ops.isnan(x)
    array([False, False, False,  True])
    """
    return backend.numpy.isnan(x)


@keras_export("keras.ops.less")
def less(x1, x2):
    """Return the truth value of `x1 < x2` element-wise.

    This function is similar to `numpy.less`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.less(x1, x2)
    array([False,  True, False])
    """
    return backend.numpy.less(x1, x2)


@keras_export("keras.ops.less_equal")
def less_equal(x1, x2):
    """Return the truth value of `x1 <= x2` element-wise.

    This function is similar to `numpy.less_equal`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.less_equal(x1, x2)
    array([ True,  True, False])
    """
    return backend.numpy.less_equal(x1, x2)


@keras_export("keras.ops.linspace")
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Return evenly spaced numbers over a specified interval.

    This function is similar to `numpy.linspace`.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence, unless `endpoint` is set to
            `False`.
        num: Number of samples to generate. Default is 50.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is
            not included.
        retstep: If `True`, return `(samples, step)`, where `step` is the
            spacing between samples.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `start` and `stop`.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Returns:
        A tensor of `num` evenly-spaced samples. If `retstep` is `True`,
        returns `(samples, step)`.

    Examples:
    >>> keras.ops.linspace(0, 10, 5)
    array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)
    """
    return backend.numpy.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


@keras_export("keras.ops.log")
def log(x):
    """Return the natural logarithm, element-wise.

    This function is similar to `numpy.log`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.log(x)
    array([0.       , 0.6931472, 1.0986123], dtype=float32)
    """
    return backend.numpy.log(x)


@keras_export("keras.ops.log10")
def log10(x):
    """Return the base 10 logarithm, element-wise.

    This function is similar to `numpy.log10`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 10, 100])
    >>> keras.ops.log10(x)
    array([0., 1., 2.], dtype=float32)
    """
    return backend.numpy.log10(x)


@keras_export("keras.ops.log1p")
def log1p(x):
    """Return `log(1 + x)`, element-wise.

    This function is similar to `numpy.log1p`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.log1p(x)
    array([0.6931472, 1.0986123, 1.3862944], dtype=float32)
    """
    return backend.numpy.log1p(x)


@keras_export("keras.ops.log2")
def log2(x):
    """Return the base 2 logarithm, element-wise.

    This function is similar to `numpy.log2`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 4])
    >>> keras.ops.log2(x)
    array([0., 1., 2.], dtype=float32)
    """
    return backend.numpy.log2(x)


@keras_export("keras.ops.logaddexp")
def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs.

    This function is similar to `numpy.logaddexp`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 2, 4])
    >>> keras.ops.logaddexp(x1, x2)
    array([1.6931472, 2.6931472, 4.317488 ], dtype=float32)
    """
    return backend.numpy.logaddexp(x1, x2)


@keras_export("keras.ops.logical_and")
def logical_and(x1, x2):
    """Return the truth value of `x1 AND x2` element-wise.

    This function is similar to `numpy.logical_and`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([True, False, True, False])
    >>> x2 = keras.ops.convert_to_tensor([True, True, False, False])
    >>> keras.ops.logical_and(x1, x2)
    array([ True, False, False, False])
    """
    return backend.numpy.logical_and(x1, x2)


@keras_export("keras.ops.logical_not")
def logical_not(x):
    """Return the truth value of `NOT x` element-wise.

    This function is similar to `numpy.logical_not`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([True, False])
    >>> keras.ops.logical_not(x)
    array([False,  True])
    """
    return backend.numpy.logical_not(x)


@keras_export("keras.ops.logical_or")
def logical_or(x1, x2):
    """Return the truth value of `x1 OR x2` element-wise.

    This function is similar to `numpy.logical_or`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([True, False, True, False])
    >>> x2 = keras.ops.convert_to_tensor([True, True, False, False])
    >>> keras.ops.logical_or(x1, x2)
    array([ True,  True,  True, False])
    """
    return backend.numpy.logical_or(x1, x2)


@keras_export("keras.ops.logical_xor")
def logical_xor(x1, x2):
    """Return the truth value of `x1 XOR x2` element-wise.

    This function is similar to `numpy.logical_xor`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([True, False, True, False])
    >>> x2 = keras.ops.convert_to_tensor([True, True, False, False])
    >>> keras.ops.logical_xor(x1, x2)
    array([False,  True,  True, False])
    """
    return backend.numpy.logical_xor(x1, x2)


@keras_export("keras.ops.logspace")
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale.

    This function is similar to `numpy.logspace`.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence, unless `endpoint` is set to
            `False`.
        num: Number of samples to generate. Default is 50.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is
            not included.
        base: The base of the log space. Default is 10.0.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `start` and `stop`.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Returns:
        A tensor of `num` evenly-spaced samples on a log scale.

    Examples:
    >>> keras.ops.logspace(0, 10, 5)
    array([1.0000000e+00, 3.1622777e+02, 1.0000000e+05, 3.1622776e+07,
           1.0000000e+10], dtype=float32)
    """
    return backend.numpy.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


@keras_export("keras.ops.max")
def max(x, axis=None, keepdims=False, initial=None):
    """Return the maximum of a tensor or maximum along an axis.

    This function is similar to `numpy.max`.

    Args:
        x: A tensor.
        axis: The axis along which to find the maximum. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.
        initial: The minimum value of an output element. Must be present to
            allow computation on empty slice.

    Returns:
        A tensor with the maximum of `x`.

    Examples:
    >>> x = keras.ops.arange(4).reshape((2, 2))
    >>> keras.ops.max(x)
    array(3)
    >>> keras.ops.max(x, axis=0)
    array([2, 3])
    >>> keras.ops.max(x, axis=1)
    array([1, 3])
    """
    return backend.numpy.max(x, axis=axis, keepdims=keepdims, initial=initial)


@keras_export("keras.ops.maximum")
def maximum(x1, x2):
    """Return the element-wise maximum of two tensors.

    This function is similar to `numpy.maximum`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor with the element-wise maximum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.maximum(x1, x2)
    array([1, 3, 3], dtype=int32)
    """
    return backend.numpy.maximum(x1, x2)


@keras_export("keras.ops.mean")
def mean(x, axis=None, keepdims=False):
    """Compute the arithmetic mean along the specified axis.

    This function is similar to `numpy.mean`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the mean. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the mean of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.mean(x)
    array(2.5, dtype=float64)
    >>> keras.ops.mean(x, axis=0)
    array([1.5, 2.5, 3.5], dtype=float64)
    """
    return backend.numpy.mean(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.median")
def median(x, axis=None, keepdims=False):
    """Compute the median along the specified axis.

    This function is similar to `numpy.median`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the median. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the median of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.median(x)
    array(2.5, dtype=float64)
    >>> keras.ops.median(x, axis=0)
    array([1.5, 2.5, 3.5], dtype=float64)
    """
    return backend.numpy.median(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.meshgrid")
def meshgrid(*x, indexing="xy"):
    """Return coordinate matrices from coordinate vectors.

    This function is similar to `numpy.meshgrid`.

    Args:
        x: 1-D tensors representing the coordinate values.
        indexing: The indexing mode, either `"xy"` or `"ij"`.

    Returns:
        A list of tensors with the coordinate matrices.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> y = keras.ops.convert_to_tensor([4, 5, 6])
    >>> xv, yv = keras.ops.meshgrid(x, y)
    >>> xv
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]], dtype=int32)
    >>> yv
    array([[4, 4, 4],
           [5, 5, 5],
           [6, 6, 6]], dtype=int32)
    """
    return backend.numpy.meshgrid(*x, indexing=indexing)


@keras_export("keras.ops.min")
def min(x, axis=None, keepdims=False, initial=None):
    """Return the minimum of a tensor or minimum along an axis.

    This function is similar to `numpy.min`.

    Args:
        x: A tensor.
        axis: The axis along which to find the minimum. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.
        initial: The maximum value of an output element. Must be present to
            allow computation on empty slice.

    Returns:
        A tensor with the minimum of `x`.

    Examples:
    >>> x = keras.ops.arange(4).reshape((2, 2))
    >>> keras.ops.min(x)
    array(0)
    >>> keras.ops.min(x, axis=0)
    array([0, 1])
    >>> keras.ops.min(x, axis=1)
    array([0, 2])
    """
    return backend.numpy.min(x, axis=axis, keepdims=keepdims, initial=initial)


@keras_export("keras.ops.minimum")
def minimum(x1, x2):
    """Return the element-wise minimum of two tensors.

    This function is similar to `numpy.minimum`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor with the element-wise minimum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.minimum(x1, x2)
    array([1, 2, 2], dtype=int32)
    """
    return backend.numpy.minimum(x1, x2)


@keras_export("keras.ops.mod")
def mod(x1, x2):
    """Return element-wise remainder of division.

    This function is similar to `numpy.mod`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 2, 2])
    >>> keras.ops.mod(x1, x2)
    array([0, 0, 1], dtype=int32)
    """
    return backend.numpy.mod(x1, x2)


@keras_export("keras.ops.moveaxis")
def moveaxis(x, source, destination):
    """Move axes of a tensor to new positions.

    This function is similar to `numpy.moveaxis`.

    Args:
        x: A tensor.
        source: Original positions of the axes to move.
        destination: Destination positions for each of the original axes.

    Returns:
        A tensor with moved axes.

    Examples:
    >>> x = keras.ops.zeros((3, 4, 5))
    >>> keras.ops.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> keras.ops.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    """
    return backend.numpy.moveaxis(x, source, destination)


@keras_export("keras.ops.multiply")
def multiply(x1, x2):
    """Multiply arguments element-wise.

    This function is similar to `numpy.multiply`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.multiply(x1, x2)
    array([ 4, 10, 18], dtype=int32)
    """
    return backend.numpy.multiply(x1, x2)


@keras_export("keras.ops.nan_to_num")
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace `NaN`, positive infinity, and negative infinity values.

    This function is similar to `numpy.nan_to_num`.

    Args:
        x: A tensor.
        nan: Value to be used to fill `NaN` values.
        posinf: Value to be used to fill positive infinity values.
        neginf: Value to be used to fill negative infinity values.

    Returns:
        A tensor with `NaN`, positive infinity, and negative infinity values
        replaced.

    Examples:
    >>> x = keras.ops.convert_to_tensor(
    ...     [float('inf'), -float('inf'), float('nan'), 1, -1])
    >>> keras.ops.nan_to_num(x)
    array([ 65504., -65504.,      0.,      1.,     -1.], dtype=float16)
    """
    return backend.numpy.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@keras_export("keras.ops.ndim")
def ndim(x):
    """Return the number of dimensions of a tensor.

    This function is similar to `numpy.ndim`.

    Args:
        x: A tensor.

    Returns:
        An integer.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.ndim(x)
    2
    """
    return backend.numpy.ndim(x)


@keras_export("keras.ops.negative")
def negative(x):
    """Return the negative of the input, element-wise.

    This function is similar to `numpy.negative`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1, 2, -2])
    >>> keras.ops.negative(x)
    array([-1,  1, -2,  2], dtype=int32)
    """
    return backend.numpy.negative(x)


@keras_export("keras.ops.nonzero")
def nonzero(x):
    """Return the indices of the elements that are non-zero.

    This function is similar to `numpy.nonzero`.

    Args:
        x: A tensor.

    Returns:
        A tuple of tensors, one for each dimension of `x`, containing the
        indices of the non-zero elements in that dimension.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 0, 2, 0, 3, 0])
    >>> keras.ops.nonzero(x)
    (array([0, 2, 4]),)

    >>> x = keras.ops.convert_to_tensor([[1, 0, 2], [0, 3, 0]])
    >>> keras.ops.nonzero(x)
    (array([0, 0, 1]), array([0, 2, 1]))
    """
    return backend.numpy.nonzero(x)


@keras_export("keras.ops.not_equal")
def not_equal(x1, x2):
    """Return the truth value of `x1 != x2` element-wise.

    This function is similar to `numpy.not_equal`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A `bool` tensor.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 3, 2])
    >>> keras.ops.not_equal(x1, x2)
    array([False,  True,  True])
    """
    return backend.numpy.not_equal(x1, x2)


@keras_export("keras.ops.ones")
def ones(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with ones.

    This function is similar to `numpy.ones`.

    Args:
        shape: The shape of the new tensor.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A tensor of ones with the given shape and dtype.

    Examples:
    >>> keras.ops.ones((2, 3))
    array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    """
    return backend.numpy.ones(shape, dtype=dtype)


@keras_export("keras.ops.ones_like")
def ones_like(x, dtype=None):
    """Return a new tensor with the same shape and type as a given tensor.

    This function is similar to `numpy.ones_like`.

    Args:
        x: A tensor.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `x`.

    Returns:
        A tensor of ones with the same shape and type as `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    """
    return backend.numpy.ones_like(x, dtype=dtype)


@keras_export("keras.ops.outer")
def outer(x1, x2):
    """Compute the outer product of two vectors.

    This function is similar to `numpy.outer`.

    Args:
        x1: A 1-D tensor.
        x2: A 1-D tensor.

    Returns:
        A 2-D tensor with the outer product of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.outer(x1, x2)
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]], dtype=int32)
    """
    return backend.numpy.outer(x1, x2)


@keras_export("keras.ops.pad")
def pad(x, pad_width, mode="constant", constant_values=0):
    """Pad a tensor.

    This function is similar to `numpy.pad`.

    Args:
        x: A tensor.
        pad_width: Number of values padded to the edges of each axis.
        mode: The mode for padding. One of `"constant"`, `"reflect"`, or
            `"symmetric"`.
        constant_values: The values to set for `mode="constant"`.

    Returns:
        A tensor with the padded values.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.pad(x, ((1, 1), (2, 2)))
    array([[0, 0, 0, 1, 2, 0, 0],
           [0, 0, 3, 4, 5, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    """
    return backend.numpy.pad(x, pad_width, mode=mode, constant_values=constant_values)


@keras_export("keras.ops.positive")
def positive(x):
    """Return the positive of the input, element-wise.

    This function is similar to `numpy.positive`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1, 2, -2])
    >>> keras.ops.positive(x)
    array([ 1, -1,  2, -2], dtype=int32)
    """
    return backend.numpy.positive(x)


@keras_export("keras.ops.power")
def power(x1, x2):
    """Compute `x1` to the power of `x2`, element-wise.

    This function is similar to `numpy.power`.

    Args:
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.power(x1, x2)
    array([ 1,  4, 27], dtype=int32)
    """
    return backend.numpy.power(x1, x2)


@keras_export("keras.ops.prod")
def prod(x, axis=None, keepdims=False, dtype=None):
    """Return the product of tensor elements over a given axis.

    This function is similar to `numpy.prod`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the product. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `x`.

    Returns:
        A tensor with the product of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.prod(x)
    array(0)
    >>> keras.ops.prod(x, axis=0)
    array([0, 4, 10])
    """
    return backend.numpy.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


@keras_export("keras.ops.quantile")
def quantile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th quantile of the data along the specified axis.

    This function is similar to `numpy.quantile`.

    Args:
        x: A tensor.
        q: The quantile to compute.
        axis: The axis along which to compute the quantile. If `None`, the
            array is flattened before use.
        method: The method for interpolation.
            One of `"linear"`, `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the q-th quantile of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.quantile(x, 0.5)
    array(2.5, dtype=float64)
    >>> keras.ops.quantile(x, 0.5, axis=0)
    array([1.5, 2.5, 3.5], dtype=float64)
    """
    return backend.numpy.quantile(
        x, q, axis=axis, method=method, keepdims=keepdims
    )


@keras_export("keras.ops.ravel")
def ravel(x):
    """Return a contiguous flattened tensor.

    This function is similar to `numpy.ravel`.

    Args:
        x: A tensor.

    Returns:
        A flattened tensor.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.ravel(x)
    array([0, 1, 2, 3, 4, 5], dtype=int32)
    """
    return backend.numpy.ravel(x)


@keras_export("keras.ops.real")
def real(x):
    """Return the real part of the complex argument.

    This function is similar to `numpy.real`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1+2j, 1-2j])
    >>> keras.ops.real(x)
    array([1., 1.], dtype=float32)
    """
    return backend.numpy.real(x)


@keras_export("keras.ops.reciprocal")
def reciprocal(x):
    """Return the reciprocal of the input, element-wise.

    This function is similar to `numpy.reciprocal`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 4])
    >>> keras.ops.reciprocal(x)
    array([1.  , 0.5 , 0.25], dtype=float32)
    """
    return backend.numpy.reciprocal(x)


@keras_export("keras.ops.repeat")
def repeat(x, repeats, axis=None):
    """Repeat elements of a tensor.

    This function is similar to `numpy.repeat`.

    Args:
        x: A tensor.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values. By default, use the
            flattened input array, and return a flat output array.

    Returns:
        A tensor with repeated values.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3], dtype=int32)

    >>> x = keras.ops.arange(4).reshape((2, 2))
    >>> keras.ops.repeat(x, 2, axis=0)
    array([[0, 1],
           [0, 1],
           [2, 3],
           [2, 3]], dtype=int32)
    """
    return backend.numpy.repeat(x, repeats, axis=axis)


@keras_export("keras.ops.reshape")
def reshape(x, new_shape):
    """Give a new shape to a tensor without changing its data.

    This function is similar to `numpy.reshape`.

    Args:
        x: A tensor.
        new_shape: The new shape should be compatible with the original
            shape.

    Returns:
        A tensor with the new shape.

    Examples:
    >>> x = keras.ops.arange(6)
    >>> keras.ops.reshape(x, (2, 3))
    array([[0, 1, 2],
           [3, 4, 5]], dtype=int32)
    """
    return backend.numpy.reshape(x, new_shape)


@keras_export("keras.ops.round")
def round(x, decimals=0):
    """Round a tensor to the given number of decimals.

    This function is similar to `numpy.round`.

    Args:
        x: A tensor or a Python number.
        decimals: Number of decimal places to round to. If `decimals` is
            negative, it specifies the number of positions to the left
            of the decimal point. Defaults to 0.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1.23, 1.58, 2.5])
    >>> keras.ops.round(x)
    array([1., 2., 2.], dtype=float32)
    >>> keras.ops.round(x, decimals=1)
    array([1.2, 1.6, 2.5], dtype=float32)
    """
    return backend.numpy.round(x, decimals=decimals)


@keras_export("keras.ops.sign")
def sign(x):
    """Return an element-wise indication of the sign of a number.

    This function is similar to `numpy.sign`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, -1, 0, 2, -2])
    >>> keras.ops.sign(x)
    array([ 1, -1,  0,  1, -1], dtype=int32)
    """
    return backend.numpy.sign(x)


@keras_export("keras.ops.sin")
def sin(x):
    """Return the trigonometric sine, element-wise.

    This function is similar to `numpy.sin`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 3.14159265 / 2])
    >>> keras.ops.sin(x)
    array([0., 1.], dtype=float32)
    """
    return backend.numpy.sin(x)


@keras_export("keras.ops.sinh")
def sinh(x):
    """Return the hyperbolic sine, element-wise.

    This function is similar to `numpy.sinh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 1, -1])
    >>> keras.ops.sinh(x)
    array([ 0.       ,  1.1752012, -1.1752012], dtype=float32)
    """
    return backend.numpy.sinh(x)


@keras_export("keras.ops.size")
def size(x):
    """Return the number of elements in a tensor.

    This function is similar to `numpy.size`.

    Args:
        x: A tensor.

    Returns:
        An integer.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.size(x)
    6
    """
    return backend.numpy.size(x)


@keras_export("keras.ops.sort")
def sort(x, axis=-1):
    """Sort a tensor.

    This function is similar to `numpy.sort`.

    Args:
        x: A tensor.
        axis: The axis along which to sort. If `None`, the array is
            flattened before use.

    Returns:
        A sorted tensor.

    Examples:
    >>> x = keras.ops.convert_to_tensor([3, 1, 2])
    >>> keras.ops.sort(x)
    array([1, 2, 3], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[0, 3], [2, 1]])
    >>> keras.ops.sort(x, axis=0)
    array([[0, 1],
           [2, 3]], dtype=int32)
    """
    return backend.numpy.sort(x, axis=axis)


@keras_export("keras.ops.split")
def split(x, indices_or_sections, axis=0):
    """Split a tensor into multiple sub-tensors.

    This function is similar to `numpy.split`.

    Args:
        x: A tensor.
        indices_or_sections: If an integer, `N`, the tensor will be divided
            into `N` equal arrays along `axis`. If a 1-D array of sorted
            integers, the entries indicate where along `axis` the array is
            split.
        axis: The axis along which to split.

    Returns:
        A list of sub-tensors.

    Examples:
    >>> x = keras.ops.arange(9)
    >>> keras.ops.split(x, 3)
    [array([0, 1, 2], dtype=int32), array([3, 4, 5], dtype=int32),
     array([6, 7, 8], dtype=int32)]

    >>> x = keras.ops.arange(8).reshape((2, 2, 2))
    >>> keras.ops.split(x, [1], axis=0)
    [array([[[0, 1],
            [2, 3]]], dtype=int32),
     array([[[4, 5],
            [6, 7]]], dtype=int32)]
    """
    return backend.numpy.split(x, indices_or_sections, axis=axis)


@keras_export("keras.ops.sqrt")
def sqrt(x):
    """Return the non-negative square-root of a tensor, element-wise.

    This function is similar to `numpy.sqrt`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 4, 9])
    >>> keras.ops.sqrt(x)
    array([1., 2., 3.], dtype=float32)
    """
    return backend.numpy.sqrt(x)


@keras_export("keras.ops.square")
def square(x):
    """Return the element-wise square of the input.

    This function is similar to `numpy.square`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.square(x)
    array([1, 4, 9], dtype=int32)
    """
    return backend.numpy.square(x)


@keras_export("keras.ops.squeeze")
def squeeze(x, axis=None):
    """Remove axes of length one from `x`.

    This function is similar to `numpy.squeeze`.

    Args:
        x: A tensor.
        axis: The axis to squeeze.

    Returns:
        A tensor with the squeezed axes.

    Examples:
    >>> x = keras.ops.zeros((1, 2, 1, 3, 1))
    >>> keras.ops.squeeze(x).shape
    (2, 3)
    >>> keras.ops.squeeze(x, axis=0).shape
    (2, 1, 3, 1)
    """
    return backend.numpy.squeeze(x, axis=axis)


@keras_export("keras.ops.stack")
def stack(x, axis=0):
    """Join a sequence of tensors along a new axis.

    This function is similar to `numpy.stack`.

    Args:
        x: A sequence of tensors.
        axis: The axis in the result tensor along which the input tensors
            are stacked.

    Returns:
        A tensor with the stacked tensors.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.stack([x1, x2], axis=0)
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    """
    return backend.numpy.stack(x, axis=axis)


@keras_export("keras.ops.std")
def std(x, axis=None, keepdims=False):
    """Compute the standard deviation along the specified axis.

    This function is similar to `numpy.std`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the standard deviation. If
            `None`, the array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the standard deviation of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.std(x)
    array(1.7078251, dtype=float32)
    >>> keras.ops.std(x, axis=0)
    array([1.5, 1.5, 1.5], dtype=float32)
    """
    return backend.numpy.std(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.sum")
def sum(x, axis=None, keepdims=False):
    """Sum of tensor elements over a given axis.

    This function is similar to `numpy.sum`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the sum. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the sum of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.sum(x)
    array(15)
    >>> keras.ops.sum(x, axis=0)
    array([3, 5, 7])
    """
    return backend.numpy.sum(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.swapaxes")
def swapaxes(x, axis1, axis2):
    """Interchange two axes of a tensor.

    This function is similar to `numpy.swapaxes`.

    Args:
        x: A tensor.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        A tensor with the swapped axes.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.swapaxes(x, 0, 1)
    array([[0, 3],
           [1, 4],
           [2, 5]], dtype=int32)
    """
    return backend.numpy.swapaxes(x, axis1, axis2)


@keras_export("keras.ops.take")
def take(x, indices, axis=None):
    """Take elements from a tensor along an axis.

    This function is similar to `numpy.take`.

    Args:
        x: A tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the
            flattened input array is used.

    Returns:
        A tensor with the selected values.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.take(x, [0, 2], axis=1)
    array([[0, 2],
           [3, 5]], dtype=int32)
    """
    return backend.numpy.take(x, indices, axis=axis)


@keras_export("keras.ops.take_along_axis")
def take_along_axis(x, indices, axis=None):
    """Take values from the input tensor by matching axes with indices.

    This function is similar to `numpy.take_along_axis`.

    Args:
        x: A tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the
            flattened input array is used.

    Returns:
        A tensor with the selected values.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> indices = keras.ops.convert_to_tensor([[0, 2], [1, 0]])
    >>> keras.ops.take_along_axis(x, indices, axis=1)
    array([[0, 2],
           [4, 3]], dtype=int32)
    """
    return backend.numpy.take_along_axis(x, indices, axis=axis)


@keras_export("keras.ops.tan")
def tan(x):
    """Return the trigonometric tangent, element-wise.

    This function is similar to `numpy.tan`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 3.14159265 / 4])
    >>> keras.ops.tan(x)
    array([0., 1.], dtype=float32)
    """
    return backend.numpy.tan(x)


@keras_export("keras.ops.tanh")
def tanh(x):
    """Return the hyperbolic tangent, element-wise.

    This function is similar to `numpy.tanh`.

    Args:
        x: A tensor or a Python number.

    Returns:
        A tensor or a Python number.

    Examples:
    >>> x = keras.ops.convert_to_tensor([0, 1, -1])
    >>> keras.ops.tanh(x)
    array([ 0.       ,  0.7615942, -0.7615942], dtype=float32)
    """
    return backend.numpy.tanh(x)


@keras_export("keras.ops.tensordot")
def tensordot(x1, x2, axes=2):
    """Compute tensor dot product along specified axes.

    This function is similar to `numpy.tensordot`.

    Args:
        x1: A tensor.
        x2: A tensor.
        axes: The axes to compute the dot product over.

    Returns:
        A tensor with the tensor dot product of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.arange(12).reshape((2, 3, 2))
    >>> x2 = keras.ops.arange(12).reshape((2, 2, 3))
    >>> keras.ops.tensordot(x1, x2, axes=([1, 2], [2, 0]))
    array([[ 94, 112],
           [280, 340]], dtype=int32)
    """
    return backend.numpy.tensordot(x1, x2, axes=axes)


@keras_export("keras.ops.tile")
def tile(x, repeats):
    """Construct a tensor by repeating `x` the number of times given by `repeats`.

    This function is similar to `numpy.tile`.

    Args:
        x: A tensor.
        repeats: The number of repetitions of `x` along each axis.

    Returns:
        A tensor with repeated values.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 2, 3])
    >>> keras.ops.tile(x, 2)
    array([1, 2, 3, 1, 2, 3], dtype=int32)

    >>> x = keras.ops.arange(4).reshape((2, 2))
    >>> keras.ops.tile(x, (2, 1))
    array([[0, 1],
           [2, 3],
           [0, 1],
           [2, 3]], dtype=int32)
    """
    return backend.numpy.tile(x, repeats)


@keras_export("keras.ops.trace")
def trace(x, offset=0, axis1=0, axis2=1):
    """Return the sum along diagonals of the tensor.

    This function is similar to `numpy.trace`.

    Args:
        x: A tensor.
        offset: Offset of the diagonal from the main diagonal.
        axis1: The first axis of the 2-D sub-arrays from which the
            diagonals should be taken.
        axis2: The second axis of the 2-D sub-arrays from which the
            diagonals should be taken.

    Returns:
        A tensor with the sum along diagonals.

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> keras.ops.trace(x)
    array(12)
    >>> keras.ops.trace(x, offset=1)
    array(6)
    """
    return backend.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


@keras_export("keras.ops.transpose")
def transpose(x, axes=None):
    """Reverse or permute the axes of a tensor.

    This function is similar to `numpy.transpose`.

    Args:
        x: A tensor.
        axes: The permutation of the axes.

    Returns:
        A tensor with the permuted axes.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.transpose(x)
    array([[0, 3],
           [1, 4],
           [2, 5]], dtype=int32)

    >>> keras.ops.transpose(x, (1, 0))
    array([[0, 3],
           [1, 4],
           [2, 5]], dtype=int32)
    """
    return backend.numpy.transpose(x, axes=axes)


@keras_export("keras.ops.tri")
def tri(N, M=None, k=0, dtype=None):
    """Return a tensor with ones at and below the given diagonal and zeros elsewhere.

    This function is similar to `numpy.tri`.

    Args:
        N: Number of rows in the array.
        M: Number of columns in the array. By default, `M` is taken equal to `N`.
        k: The sub-diagonal at and below which the array is filled.
            `k=0` is the main diagonal, while `k<0` is below it, and `k>0` is
            above.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A tensor with its lower triangle filled with ones and other elements
        zeros.

    Examples:
    >>> keras.ops.tri(3)
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]], dtype=float32)
    >>> keras.ops.tri(3, k=1)
    array([[1., 1., 0.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    """
    return backend.numpy.tri(N, M=M, k=k, dtype=dtype)


@keras_export("keras.ops.tril")
def tril(x, k=0):
    """Lower triangle of a tensor.

    This function is similar to `numpy.tril`.

    Args:
        x: A tensor.
        k: The sub-diagonal at and below which the array is filled.
            `k=0` is the main diagonal, while `k<0` is below it, and `k>0` is
            above.

    Returns:
        A tensor with its lower triangle filled with the values of `x` and
        other elements zeros.

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> keras.ops.tril(x)
    array([[0, 0, 0],
           [3, 4, 0],
           [6, 7, 8]], dtype=int32)
    >>> keras.ops.tril(x, k=1)
    array([[0, 1, 0],
           [3, 4, 5],
           [6, 7, 8]], dtype=int32)
    """
    return backend.numpy.tril(x, k=k)


@keras_export("keras.ops.triu")
def triu(x, k=0):
    """Upper triangle of a tensor.

    This function is similar to `numpy.triu`.

    Args:
        x: A tensor.
        k: The sub-diagonal at and below which the array is filled.
            `k=0` is the main diagonal, while `k<0` is below it, and `k>0` is
            above.

    Returns:
        A tensor with its upper triangle filled with the values of `x` and
        other elements zeros.

    Examples:
    >>> x = keras.ops.arange(9).reshape((3, 3))
    >>> keras.ops.triu(x)
    array([[0, 1, 2],
           [0, 4, 5],
           [0, 0, 8]], dtype=int32)
    >>> keras.ops.triu(x, k=1)
    array([[0, 1, 2],
           [0, 0, 5],
           [0, 0, 0]], dtype=int32)
    """
    return backend.numpy.triu(x, k=k)


@keras_export("keras.ops.vdot")
def vdot(x1, x2):
    """Return the dot product of two vectors.

    This function is similar to `numpy.vdot`.

    Args:
        x1: A tensor.
        x2: A tensor.

    Returns:
        A tensor with the dot product of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.vdot(x1, x2)
    array(32)
    """
    return backend.numpy.vdot(x1, x2)


@keras_export("keras.ops.var")
def var(x, axis=None, keepdims=False):
    """Compute the variance along the specified axis.

    This function is similar to `numpy.var`.

    Args:
        x: A tensor.
        axis: The axis along which to compute the variance. If `None`, the
            array is flattened before use.
        keepdims: If `True`, the axes which are reduced are left in the result
            as dimensions with size one.

    Returns:
        A tensor with the variance of `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.var(x)
    array(2.9166667, dtype=float32)
    >>> keras.ops.var(x, axis=0)
    array([2.25, 2.25, 2.25], dtype=float32)
    """
    return backend.numpy.var(x, axis=axis, keepdims=keepdims)


@keras_export("keras.ops.where")
def where(condition, x1, x2):
    """Return elements chosen from `x1` or `x2` depending on `condition`.

    This function is similar to `numpy.where`.

    Args:
        condition: A `bool` tensor. Where `True`, yield `x1`, otherwise
            yield `x2`.
        x1: A tensor or a Python number.
        x2: A tensor or a Python number.

    Returns:
        A tensor with elements from `x1` where `condition` is `True`, and
        `x2` otherwise.

    Examples:
    >>> condition = keras.ops.convert_to_tensor([True, False, True])
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([4, 5, 6])
    >>> keras.ops.where(condition, x1, x2)
    array([1, 5, 3], dtype=int32)
    """
    return backend.numpy.where(condition, x1, x2)


@keras_export("keras.ops.zeros")
def zeros(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with zeros.

    This function is similar to `numpy.zeros`.

    Args:
        shape: The shape of the new tensor.
        dtype: The type of the output tensor. If `None`, the dtype is
            `"float32"`.

    Returns:
        A tensor of zeros with the given shape and dtype.

    Examples:
    >>> keras.ops.zeros((2, 3))
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    """
    return backend.numpy.zeros(shape, dtype=dtype)


@keras_export("keras.ops.zeros_like")
def zeros_like(x, dtype=None):
    """Return a new tensor with the same shape and type as a given tensor.

    This function is similar to `numpy.zeros_like`.

    Args:
        x: A tensor.
        dtype: The type of the output tensor. If `None`, the dtype is
            inferred from `x`.

    Returns:
        A tensor of zeros with the same shape and type as `x`.

    Examples:
    >>> x = keras.ops.arange(6).reshape((2, 3))
    >>> keras.ops.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
    """
    return backend.numpy.zeros_like(x, dtype=dtype)
