import builtins
import re

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape


def shape_equal(shape1, shape2, axis=None, allow_none=True):
    """Check if two shapes are equal.

    Args:
        shape1: A list or tuple of integers for first shape to be compared.
        shape2: A list or tuple of integers for second shape to be compared.
        axis: An integer, list, or tuple of integers (optional):
            Axes to ignore during comparison. Defaults to `None`.
        allow_none (bool, optional): If `True`, allows `None` in a shape
            to match any value in the corresponding position of the other shape.
            Defaults to `True`.

    Returns:
        bool: `True` if shapes are considered equal based on the criteria,
        `False` otherwise.

    Examples:

    >>> shape_equal((32, 64, 128), (32, 64, 128))
    True
    >>> shape_equal((32, 64, 128), (32, 64, 127))
    False
    >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=True)
    True
    >>> shape_equal((32, 64, None), (32, 64, 128), allow_none=False)
    False
    >>> shape_equal((32, 64, 128), (32, 63, 128), axis=1)
    True
    >>> shape_equal((32, 64, 128), (32, 63, 127), axis=(1, 2))
    True
    >>> shape_equal((32, 64, 128), (32, 63, 127), axis=[1,2])
    True
    >>> shape_equal((32, 64), (32, 64, 128))
    False
    """
    if len(shape1) != len(shape2):
        return False

    shape1 = list(shape1)
    shape2 = list(shape2)

    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        for ax in axis:
            shape1[ax] = -1
            shape2[ax] = -1

    if allow_none:
        for i in range(len(shape1)):
            if shape1[i] is None:
                shape1[i] = shape2[i]
            if shape2[i] is None:
                shape2[i] = shape1[i]

    return shape1 == shape2


class Absolute(Operation):
    def call(self, x):
        return backend.numpy.absolute(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.absolute", "keras.ops.numpy.absolute"])
def absolute(x):
    """Compute the absolute value element-wise.

    `keras.ops.abs` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        An array containing the absolute value of each element in `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([-1.2, 1.2])
    >>> keras.ops.absolute(x)
    array([1.2, 1.2], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.numpy.absolute(x)


class Abs(Absolute):
    pass


@keras_export(["keras.ops.abs", "keras.ops.numpy.abs"])
def abs(x):
    """Shorthand for `keras.ops.absolute`."""
    return absolute(x)


class Add(Operation):
    def call(self, x1, x2):
        return backend.numpy.add(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.add", "keras.ops.numpy.add"])
def add(x1, x2):
    """Add arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        The tensor containing the element-wise sum of `x1` and `x2`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 4])
    >>> x2 = keras.ops.convert_to_tensor([5, 6])
    >>> keras.ops.add(x1, x2)
    array([6, 10], dtype=int32)

    `keras.ops.add` also broadcasts shapes:
    >>> x1 = keras.ops.convert_to_tensor(
    ...     [[5, 4],
    ...      [5, 6]]
    ... )
    >>> x2 = keras.ops.convert_to_tensor([5, 6])
    >>> keras.ops.add(x1, x2)
    array([[10 10]
           [10 12]], shape=(2, 2), dtype=int32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Add().symbolic_call(x1, x2)
    return backend.numpy.add(x1, x2)


class All(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.all(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_export(["keras.ops.all", "keras.ops.numpy.all"])
def all(x, axis=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical AND reduction is performed. The default
            (`axis=None`) is to perform a logical AND over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Defaults to `False`.

    Returns:
        The tensor containing the logical AND reduction over the `axis`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([True, False])
    >>> keras.ops.all(x)
    array(False, shape=(), dtype=bool)

    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, axis=0)
    array([ True False], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.all(x, axis=axis, keepdims=keepdims)


class Any(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.any(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(
                x.shape,
                axis=self.axis,
                keepdims=self.keepdims,
            ),
            dtype="bool",
        )


@keras_export(["keras.ops.any", "keras.ops.numpy.any"])
def any(x, axis=None, keepdims=False):
    """Test whether any array element along a given axis evaluates to `True`.

    Args:
        x: Input tensor.
        axis: An integer or tuple of integers that represent the axis along
            which a logical OR reduction is performed. The default
            (`axis=None`) is to perform a logical OR over all the dimensions
            of the input array. `axis` may be negative, in which case it counts
            for the last to the first axis.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the input array. Defaults to `False`.

    Returns:
        The tensor containing the logical OR reduction over the `axis`.

    Examples:
    >>> x = keras.ops.convert_to_tensor([True, False])
    >>> keras.ops.any(x)
    array(True, shape=(), dtype=bool)

    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.any(x, axis=0)
    array([ True  True], shape=(2,), dtype=bool)

    `keepdims=True` outputs a tensor with dimensions reduced to one.
    >>> x = keras.ops.convert_to_tensor([[True, False], [True, True]])
    >>> keras.ops.all(x, keepdims=True)
    array([[False]], shape=(1, 1), dtype=bool)
    """
    if any_symbolic_tensors((x,)):
        return Any(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.any(x, axis=axis, keepdims=keepdims)


class Amax(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.amax(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_export(["keras.ops.amax", "keras.ops.numpy.amax"])
def amax(x, axis=None, keepdims=False):
    """Returns the maximum of an array or maximum value along an axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the maximum.
            By default (`axis=None`), find the maximum value in all the
            dimensions of the input array.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions that are broadcast to the size of the original
            input tensor. Defaults to `False`.

    Returns:
        An array with the maximum value. If `axis=None`, the result is a scalar
        value representing the maximum element in the entire array. If `axis` is
        given, the result is an array with the maximum values along
        the specified axis.

    Examples:
    >>> x = keras.ops.convert_to_tensor([[1, 3, 5], [2, 3, 6]])
    >>> keras.ops.amax(x)
    array(6, dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
    >>> keras.ops.amax(x, axis=0)
    array([1, 6, 8], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
    >>> keras.ops.amax(x, axis=1, keepdims=True)
    array([[8], [5]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Amax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.amax(x, axis=axis, keepdims=keepdims)


class Amin(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.amin(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_export(["keras.ops.amin", "keras.ops.numpy.amin"])
def amin(x, axis=None, keepdims=False):
    """Returns the minimum of an array or minimum value along an axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the minimum.
            By default (`axis=None`), find the minimum value in all the
            dimensions of the input array.
        keepdims: If `True`, axes which are reduced are left in the result as
            dimensions that are broadcast to the size of the original
            input tensor. Defaults to `False`.

    Returns:
        An array with the minimum value. If `axis=None`, the result is a scalar
        value representing the minimum element in the entire array. If `axis` is
        given, the result is an array with the minimum values along
        the specified axis.

    Examples:
    >>> x = keras.ops.convert_to_tensor([1, 3, 5, 2, 3, 6])
    >>> keras.ops.amin(x)
    array(1, dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
    >>> keras.ops.amin(x, axis=0)
    array([1,5,3], dtype=int32)

    >>> x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
    >>> keras.ops.amin(x, axis=1, keepdims=True)
    array([[1],[3]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Amin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.amin(x, axis=axis, keepdims=keepdims)


class Append(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x1, x2):
        return backend.numpy.append(x1, x2, axis=self.axis)

    def compute_output_spec(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if self.axis is None:
            if None in x1_shape or None in x2_shape:
                output_shape = [None]
            else:
                output_shape = [int(np.prod(x1_shape) + np.prod(x2_shape))]
            return KerasTensor(output_shape, dtype=dtype)

        if not shape_equal(x1_shape, x2_shape, [self.axis]):
            raise ValueError(
                "`append` requires inputs to have the same shape except the "
                f"`axis={self.axis}`, but received shape {x1_shape} and "
                f"{x2_shape}."
            )

        output_shape = list(x1_shape)
        output_shape[self.axis] = x1_shape[self.axis] + x2_shape[self.axis]
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.append", "keras.ops.numpy.append"])
def append(
    x1,
    x2,
    axis=None,
):
    """Append tensor `x2` to the end of tensor `x1`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        axis: Axis along which tensor `x2` is appended to tensor `x1`.
            If `None`, both tensors are flattened before use.

    Returns:
        A tensor with the values of `x2` appended to `x1`.

    Examples:
    >>> x1 = keras.ops.convert_to_tensor([1, 2, 3])
    >>> x2 = keras.ops.convert_to_tensor([[4, 5, 6], [7, 8, 9]])
    >>> keras.ops.append(x1, x2)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

    When `axis` is specified, `x1` and `x2` must have compatible shapes.
    >>> x1 = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    >>> x2 = keras.ops.convert_to_tensor([[7, 8, 9]])
    >>> keras.ops.append(x1, x2, axis=0)
    array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=int32)
    >>> x3 = keras.ops.convert_to_tensor([7, 8, 9])
    >>> keras.ops.append(x1, x3, axis=0)
    Traceback (most recent call last):
        ...
    TypeError: Cannot concatenate arrays with different numbers of
    dimensions: got (2, 3), (3,).
    """
    if any_symbolic_tensors((x1, x2)):
        return Append(axis=axis).symbolic_call(x1, x2)
    return backend.numpy.append(x1, x2, axis=axis)


class Arange(Operation):
    def call(self, start, stop=None, step=1, dtype=None):
        return backend.numpy.arange(start, stop, step=step, dtype=dtype)

    def compute_output_spec(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            start, stop = 0, start
        output_shape = [int(np.ceil((stop - start) / step))]
        if dtype is None:
            dtypes_to_resolve = [
                getattr(start, "dtype", type(start)),
                getattr(step, "dtype", type(step)),
            ]
            if stop is not None:
                dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
            dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.arange", "keras.ops.numpy.arange"])
def arange(start, stop=None, step=1, dtype=None):
    """Return evenly spaced values within a given interval.

    `arange` can be called with a varying number of positional arguments:
    * `arange(stop)`: Values are generated within the half-open interval
        `[0, stop)` (in other words, the interval including start but excluding
        stop).
    * `arange(start, stop)`: Values are generated within the half-open interval
        `[start, stop)`.
    * `arange(start, stop, step)`: Values are generated within the half-open
        interval `[start, stop)`, with spacing between values given by step.

    Args:
        start: Integer or real, representing the start of the interval. The
            interval includes this value.
        stop: Integer or real, representing the end of the interval. The
            interval does not include this value, except in some cases where
            `step` is not an integer and floating point round-off affects the
            length of `out`. Defaults to `None`.
        step: Integer or real, represent the spacing between values. For any
            output `out`, this is the distance between two adjacent values,
            `out[i+1] - out[i]`. The default step size is 1. If `step` is
            specified as a position argument, `start` must also be given.
        dtype: The type of the output array. If `dtype` is not given, infer the
            data type from the other input arguments.

    Returns:
        Tensor of evenly spaced values.
        For floating point arguments, the length of the result is
        `ceil((stop - start)/step)`. Because of floating point overflow, this
        rule may result in the last element of out being greater than stop.

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
    return backend.numpy.arange(start, stop, step=step, dtype=dtype)


class Arccos(Operation):
    def call(self, x):
        return backend.numpy.arccos(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.arccos", "keras.ops.numpy.arccos"])
def arccos(x):
    """Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the angle of the ray intersecting the unit circle at the given
        x-coordinate in radians `[0, pi]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1])
    >>> keras.ops.arccos(x)
    array([0.0, 3.1415927], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccos().symbolic_call(x)
    return backend.numpy.arccos(x)


class Arccosh(Operation):
    def call(self, x):
        return backend.numpy.arccosh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.arccosh", "keras.ops.numpy.arccosh"])
def arccosh(x):
    """Inverse hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as x.

    Example:
    >>> x = keras.ops.convert_to_tensor([10, 100])
    >>> keras.ops.arccosh(x)
    array([2.993223, 5.298292], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arccosh().symbolic_call(x)
    return backend.numpy.arccosh(x)


class Arcsin(Operation):
    def call(self, x):
        return backend.numpy.arcsin(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arcsin", "keras.ops.numpy.arcsin"])
def arcsin(x):
    """Inverse sine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse sine of each element in `x`, in radians and in
        the closed interval `[-pi/2, pi/2]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1, 0])
    >>> keras.ops.arcsin(x)
    array([ 1.5707964, -1.5707964,  0.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsin().symbolic_call(x)
    return backend.numpy.arcsin(x)


class Arcsinh(Operation):
    def call(self, x):
        return backend.numpy.arcsinh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arcsinh", "keras.ops.numpy.arcsinh"])
def arcsinh(x):
    """Inverse hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.

    Example:
    >>> x = keras.ops.convert_to_tensor([1, -1, 0])
    >>> keras.ops.arcsinh(x)
    array([0.88137364, -0.88137364, 0.0], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arcsinh().symbolic_call(x)
    return backend.numpy.arcsinh(x)


class Arctan(Operation):
    def call(self, x):
        return backend.numpy.arctan(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arctan", "keras.ops.numpy.arctan"])
def arctan(x):
    """Trigonometric inverse tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor of the inverse tangent of each element in `x`, in the interval
        `[-pi/2, pi/2]`.

    Example:
    >>> x = keras.ops.convert_to_tensor([0, 1])
    >>> keras.ops.arctan(x)
    array([0., 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Arctan().symbolic_call(x)
    return backend.numpy.arctan(x)


class Arctan2(Operation):
    def call(self, x1, x2):
        return backend.numpy.arctan2(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        outputs_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_dtype = backend.standardize_dtype(
            getattr(x1, "dtype", backend.floatx())
        )
        x2_dtype = backend.standardize_dtype(
            getattr(x2, "dtype", backend.floatx())
        )
        dtype = dtypes.result_type(x1_dtype, x2_dtype, float)
        return KerasTensor(outputs_shape, dtype=dtype)


@keras_export(["keras.ops.arctan2", "keras.ops.numpy.arctan2"])
def arctan2(x1, x2):
    """Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the
    signed angle in radians between the ray ending at the origin and passing
    through the point `(1, 0)`, and the ray ending at the origin and passing
    through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate"
    is the first function parameter, the "x-coordinate" is the second.) By IEEE
    convention, this function is defined for `x2 = +/-0` and for either or both
    of `x1` and `x2` `= +/-inf`.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Tensor of angles in radians, in the range `[-pi, pi]`.

    Examples:
    Consider four points in different quadrants:
    >>> x = keras.ops.convert_to_tensor([-1, +1, +1, -1])
    >>> y = keras.ops.convert_to_tensor([-1, -1, +1, +1])
    >>> keras.ops.arctan2(y, x) * 180 / numpy.pi
    array([-135., -45., 45., 135.], dtype=float32)

    Note the order of the parameters. `arctan2` is defined also when x2=0 and
    at several other points, obtaining values in the range `[-pi, pi]`:
    >>> keras.ops.arctan2(
    ...     keras.ops.array([1., -1.]),
    ...     keras.ops.array([0., 0.]),
    ... )
    array([ 1.5707964, -1.5707964], dtype=float32)
    >>> keras.ops.arctan2(
    ...     keras.ops.array([0., 0., numpy.inf]),
    ...     keras.ops.array([+0., -0., numpy.inf]),
    ... )
    array([0., 3.1415925, 0.7853982], dtype=float32)
    """
    if any_symbolic_tensors((x1, x2)):
        return Arctan2().symbolic_call(x1, x2)
    return backend.numpy.arctan2(x1, x2)


class Arctanh(Operation):
    def call(self, x):
        return backend.numpy.arctanh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.arctanh", "keras.ops.numpy.arctanh"])
def arctanh(x):
    """Inverse hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Arctanh().symbolic_call(x)
    return backend.numpy.arctanh(x)


class Argmax(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.argmax(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        if self.keepdims:
            return KerasTensor(x.shape, dtype="int32")
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


@keras_export(["keras.ops.argmax", "keras.ops.numpy.argmax"])
def argmax(x, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras.ops.argmax(x)
    array(5, dtype=int32)
    >>> keras.ops.argmax(x, axis=0)
    array([1, 1, 1], dtype=int32)
    >>> keras.ops.argmax(x, axis=1)
    array([2, 2], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmax(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.argmax(x, axis=axis, keepdims=keepdims)


class Argmin(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.argmin(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        if self.keepdims:
            return KerasTensor(x.shape, dtype="int32")
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


@keras_export(["keras.ops.argmin", "keras.ops.numpy.argmin"])
def argmin(x, axis=None, keepdims=False):
    """Returns the indices of the minium values along an axis.

    Args:
        x: Input tensor.
        axis: By default, the index is into the flattened tensor, otherwise
            along the specified axis.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.

    Returns:
        Tensor of indices. It has the same shape as `x`, with the dimension
        along `axis` removed.

    Example:
    >>> x = keras.ops.arange(6).reshape(2, 3) + 10
    >>> x
    array([[10, 11, 12],
           [13, 14, 15]], dtype=int32)
    >>> keras.ops.argmin(x)
    array(0, dtype=int32)
    >>> keras.ops.argmin(x, axis=0)
    array([0, 0, 0], dtype=int32)
    >>> keras.ops.argmin(x, axis=1)
    array([0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argmin(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.argmin(x, axis=axis, keepdims=keepdims)


class Argsort(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argsort(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([int(np.prod(x.shape))], dtype="int32")
        return KerasTensor(x.shape, dtype="int32")


@keras_export(["keras.ops.argsort", "keras.ops.numpy.argsort"])
def argsort(x, axis=-1):
    """Returns the indices that would sort a tensor.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. Defaults to `-1` (the last axis). If
            `None`, the flattened tensor is used.

    Returns:
        Tensor of indices that sort `x` along the specified `axis`.

    Examples:
    One dimensional array:
    >>> x = keras.ops.array([3, 1, 2])
    >>> keras.ops.argsort(x)
    array([1, 2, 0], dtype=int32)

    Two-dimensional array:
    >>> x = keras.ops.array([[0, 3], [3, 2], [4, 5]])
    >>> x
    array([[0, 3],
           [3, 2],
           [4, 5]], dtype=int32)
    >>> keras.ops.argsort(x, axis=0)
    array([[0, 1],
           [1, 0],
           [2, 2]], dtype=int32)
    >>> keras.ops.argsort(x, axis=1)
    array([[0, 1],
           [1, 0],
           [0, 1]], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Argsort(axis=axis).symbolic_call(x)
    return backend.numpy.argsort(x, axis=axis)


class Array(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.array(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.array", "keras.ops.numpy.array"])
def array(x, dtype=None):
    """Create a tensor.

    Args:
        x: Input tensor.
        dtype: The desired data-type for the tensor.

    Returns:
        A tensor.

    Examples:
    >>> keras.ops.array([1, 2, 3])
    array([1, 2, 3], dtype=int32)

    >>> keras.ops.array([1, 2, 3], dtype="float32")
    array([1., 2., 3.], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Array().symbolic_call(x, dtype=dtype)
    return backend.numpy.array(x, dtype=dtype)


class Average(Operation):
    def __init__(self, axis=None):
        super().__init__()
        # np.average() does not support axis as tuple as declared by the
        # docstring, it only supports int or None.
        self.axis = axis

    def call(self, x, weights=None):
        return backend.numpy.average(x, weights=weights, axis=self.axis)

    def compute_output_spec(self, x, weights=None):
        dtypes_to_resolve = [getattr(x, "dtype", type(x)), float]
        if weights is not None:
            shape_match = shape_equal(x.shape, weights.shape, allow_none=True)
            if self.axis is not None:
                shape_match_on_axis = shape_equal(
                    [x.shape[self.axis]], weights.shape, allow_none=True
                )
            dtypes_to_resolve.append(getattr(weights, "dtype", type(weights)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
        if self.axis is None:
            if weights is None or shape_match:
                return KerasTensor([], dtype=dtype)
            else:
                raise ValueError(
                    "`weights` must have the same shape as `x` when "
                    f"`axis=None`, but received `weights.shape={weights.shape}`"
                    f" and `x.shape={x.shape}`."
                )

        if weights is None or shape_match_on_axis or shape_match:
            return KerasTensor(
                reduce_shape(x.shape, axis=[self.axis]), dtype=dtype
            )
        else:
            # `weights` can either be a 1D array of length `x.shape[axis]` or
            # of the same shape as `x`.
            raise ValueError(
                "`weights` must have the same size as `x` at "
                f"`axis={self.axis}` but received "
                f"`weights.shape={weights.shape}` while x.shape at "
                f"`{self.axis}` is `{x.shape[self.axis]}`."
            )


@keras_export(["keras.ops.average", "keras.ops.numpy.average"])
def average(x, axis=None, weights=None):
    """Compute the weighted average along the specified axis.

    Args:
        x: Input tensor.
        axis: Integer along which to average `x`. The default, `axis=None`,
            will average over all of the elements of the input tensor. If axis
            is negative it counts from the last to the first axis.
        weights: Tensor of wieghts associated with the values in `x`. Each
            value in `x` contributes to the average according to its
            associated weight. The weights array can either be 1-D (in which
            case its length must be the size of a along the given axis) or of
            the same shape as `x`. If `weights=None` (default), then all data
            in `x` are assumed to have a weight equal to one.

            The 1-D calculation is: `avg = sum(a * weights) / sum(weights)`.
            The only constraint on weights is that `sum(weights)` must not be 0.

    Returns:
        Return the average along the specified axis.

    Examples:
    >>> data = keras.ops.arange(1, 5)
    >>> data
    array([1, 2, 3, 4], dtype=int32)
    >>> keras.ops.average(data)
    array(2.5, dtype=float32)
    >>> keras.ops.average(
    ...     keras.ops.arange(1, 11),
    ...     weights=keras.ops.arange(10, 0, -1)
    ... )
    array(4., dtype=float32)

    >>> data = keras.ops.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]], dtype=int32)
    >>> keras.ops.average(
    ...     data,
    ...     axis=1,
    ...     weights=keras.ops.array([1./4, 3./4])
    ... )
    array([0.75, 2.75, 4.75], dtype=float32)
    >>> keras.ops.average(
    ...     data,
    ...     weights=keras.ops.array([1./4, 3./4])
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Axis must be specified when shapes of a and weights differ.
    """
    if any_symbolic_tensors((x,)):
        return Average(axis=axis).symbolic_call(x, weights=weights)
    return backend.numpy.average(x, weights=weights, axis=axis)


class Bincount(Operation):
    def __init__(self, weights=None, minlength=0, sparse=False):
        super().__init__()
        self.weights = weights
        self.minlength = minlength
        self.sparse = sparse

    def call(self, x):
        return backend.numpy.bincount(
            x,
            weights=self.weights,
            minlength=self.minlength,
            sparse=self.sparse,
        )

    def compute_output_spec(self, x):
        dtypes_to_resolve = [x.dtype]
        if self.weights is not None:
            weights = backend.convert_to_tensor(self.weights)
            dtypes_to_resolve.append(weights.dtype)
            dtype = dtypes.result_type(*dtypes_to_resolve)
        else:
            dtype = "int32"
        x_sparse = getattr(x, "sparse", False)
        return KerasTensor(
            list(x.shape[:-1]) + [None],
            dtype=dtype,
            sparse=x_sparse or self.sparse,
        )


@keras_export(["keras.ops.bincount", "keras.ops.numpy.bincount"])
def bincount(x, weights=None, minlength=0, sparse=False):
    """Count the number of occurrences of each value in a tensor of integers.

    Args:
        x: Input tensor.
            It must be of dimension 1, and it must only contain non-negative
            integer(s).
        weights: Weight tensor.
            It must have the same length as `x`. The default value is `None`.
            If specified, `x` is weighted by it, i.e. if `n = x[i]`,
            `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
        minlength: An integer.
            The default value is 0. If specified, there will be at least
            this number of bins in the output tensor. If greater than
            `max(x) + 1`, each value of the output at an index higher than
            `max(x)` is set to 0.
        sparse: Whether to return a sparse tensor; for backends that support
            sparse tensors.

    Returns:
        1D tensor where each element gives the number of occurrence(s) of its
        index value in x. Its length is the maximum between `max(x) + 1` and
        minlength.

    Examples:
    >>> x = keras.ops.array([1, 2, 2, 3], dtype="uint8")
    >>> keras.ops.bincount(x)
    array([0, 1, 2, 1], dtype=int32)
    >>> weights = x / 2
    >>> weights
    array([0.5, 1., 1., 1.5], dtype=float64)
    >>> keras.ops.bincount(x, weights=weights)
    array([0., 0.5, 2., 1.5], dtype=float64)
    >>> minlength = (keras.ops.max(x).numpy() + 1) + 2 # 6
    >>> keras.ops.bincount(x, minlength=minlength)
    array([0, 1, 2, 1, 0, 0], dtype=int32)
    """
    if any_symbolic_tensors((x,)):
        return Bincount(
            weights=weights, minlength=minlength, sparse=sparse
        ).symbolic_call(x)
    return backend.numpy.bincount(
        x, weights=weights, minlength=minlength, sparse=sparse
    )


class BitwiseAnd(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.bitwise_and(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.bitwise_and", "keras.ops.numpy.bitwise_and"])
def bitwise_and(x, y):
    """Compute the bit-wise AND of two arrays element-wise.

    Computes the bit-wise AND of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `&`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseAnd().symbolic_call(x, y)
    return backend.numpy.bitwise_and(x, y)


class BitwiseInvert(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return backend.numpy.bitwise_invert(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.bitwise_invert", "keras.ops.numpy.bitwise_invert"])
def bitwise_invert(x):
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `~`.

    Args:
        x: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x,)):
        return BitwiseInvert().symbolic_call(x)
    return backend.numpy.bitwise_invert(x)


class BitwiseNot(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return backend.numpy.bitwise_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.bitwise_not", "keras.ops.numpy.bitwise_not"])
def bitwise_not(x):
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `~`.

    Args:
        x: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x,)):
        return BitwiseNot().symbolic_call(x)
    return backend.numpy.bitwise_not(x)


class BitwiseOr(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.bitwise_or(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.bitwise_or", "keras.ops.numpy.bitwise_or"])
def bitwise_or(x, y):
    """Compute the bit-wise OR of two arrays element-wise.

    Computes the bit-wise OR of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `|`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseOr().symbolic_call(x, y)
    return backend.numpy.bitwise_or(x, y)


class BitwiseXor(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.bitwise_xor(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.bitwise_xor", "keras.ops.numpy.bitwise_xor"])
def bitwise_xor(x, y):
    """Compute the bit-wise XOR of two arrays element-wise.

    Computes the bit-wise XOR of the underlying binary representation of the
    integers in the input arrays. This ufunc implements the C/Python operator
    `^`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseXor().symbolic_call(x, y)
    return backend.numpy.bitwise_xor(x, y)


class BitwiseLeftShift(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.bitwise_left_shift(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(
    ["keras.ops.bitwise_left_shift", "keras.ops.numpy.bitwise_left_shift"]
)
def bitwise_left_shift(x, y):
    """Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `y` 0s at the right of `x`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x` by `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseLeftShift().symbolic_call(x, y)
    return backend.numpy.bitwise_left_shift(x, y)


class LeftShift(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.left_shift(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.left_shift", "keras.ops.numpy.left_shift"])
def left_shift(x, y):
    """Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `y` 0s at the right of `x`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x` by `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return LeftShift().symbolic_call(x, y)
    return backend.numpy.left_shift(x, y)


class BitwiseRightShift(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.bitwise_right_shift(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(
    ["keras.ops.bitwise_right_shift", "keras.ops.numpy.bitwise_right_shift"]
)
def bitwise_right_shift(x, y):
    """Shift the bits of an integer to the right.

    Bits are shifted to the right `y`. Because the internal representation of
    numbers is in binary format, this operation is equivalent to dividing `x` by
    `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return BitwiseRightShift().symbolic_call(x, y)
    return backend.numpy.bitwise_right_shift(x, y)


class RightShift(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x, y):
        return backend.numpy.right_shift(x, y)

    def compute_output_spec(self, x, y):
        dtype = dtypes.result_type(x.dtype, y.dtype)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.right_shift", "keras.ops.numpy.right_shift"])
def right_shift(x, y):
    """Shift the bits of an integer to the right.

    Bits are shifted to the right `y`. Because the internal representation of
    numbers is in binary format, this operation is equivalent to dividing `x` by
    `2**y`.

    Args:
        x: Input integer tensor.
        y: Input integer tensor.

    Returns:
        Result tensor.
    """
    if any_symbolic_tensors((x, y)):
        return RightShift().symbolic_call(x, y)
    return backend.numpy.right_shift(x, y)


class BroadcastTo(Operation):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def call(self, x):
        return backend.numpy.broadcast_to(x, self.shape)

    def compute_output_spec(self, x):
        # Catch broadcasting errors for clear error messages.
        broadcast_shapes(x.shape, self.shape)
        return KerasTensor(self.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.broadcast_to",
        "keras.ops.numpy.broadcast_to",
    ]
)
def broadcast_to(x, shape):
    """Broadcast a tensor to a new shape.

    Args:
        x: The tensor to broadcast.
        shape: The shape of the desired tensor. A single integer `i` is
            interpreted as `(i,)`.

    Returns:
        A tensor with the desired shape.

    Examples:
    >>> x = keras.ops.array([1, 2, 3])
    >>> keras.ops.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    if any_symbolic_tensors((x,)):
        return BroadcastTo(shape=shape).symbolic_call(x)
    return backend.numpy.broadcast_to(x, shape)


class Ceil(Operation):
    def call(self, x):
        return backend.numpy.ceil(x)

    def compute_output_spec(self, x):
        if backend.standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.ceil", "keras.ops.numpy.ceil"])
def ceil(x):
    """Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.

    Args:
        x: Input tensor.

    Returns:
        The ceiling of each element in `x`, with float dtype.
    """
    if any_symbolic_tensors((x,)):
        return Ceil().symbolic_call(x)
    return backend.numpy.ceil(x)


class Clip(Operation):
    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return backend.numpy.clip(x, self.x_min, self.x_max)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.clip", "keras.ops.numpy.clip"])
def clip(x, x_min, x_max):
    """Clip (limit) the values in a tensor.

    Given an interval, values outside the interval are clipped to the
    interval edges. For example, if an interval of `[0, 1]` is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.

    Args:
        x: Input tensor.
        x_min: Minimum value.
        x_max: Maximum value.
    Returns:
        The clipped tensor.
    """
    if any_symbolic_tensors((x,)):
        return Clip(x_min, x_max).symbolic_call(x)
    return backend.numpy.clip(x, x_min, x_max)


class Concatenate(Operation):
    def __init__(self, axis=0):
        super().__init__()
        if axis is None:
            raise ValueError("`axis` cannot be None for `concatenate`.")
        self.axis = axis

    def call(self, xs):
        return backend.numpy.concatenate(xs, axis=self.axis)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        all_sparse = True
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(
                x.shape, first_shape, axis=[self.axis], allow_none=True
            ):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[self.axis] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[self.axis]
            all_sparse = all_sparse and getattr(x, "sparse", False)
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[self.axis] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype, sparse=all_sparse)


@keras_export(
    [
        "keras.ops.concatenate",
        "keras.ops.numpy.concatenate",
    ]
)
def concatenate(xs, axis=0):
    """Join a sequence of tensors along an existing axis.

    Args:
        xs: The sequence of tensors to concatenate.
        axis: The axis along which the tensors will be joined. Defaults to `0`.

    Returns:
        The concatenated tensor.
    """
    if any_symbolic_tensors(xs):
        return Concatenate(axis=axis).symbolic_call(xs)
    return backend.numpy.concatenate(xs, axis=axis)


class Conjugate(Operation):
    def call(self, x):
        return backend.numpy.conjugate(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.conjugate", "keras.ops.numpy.conjugate"])
def conjugate(x):
    """Returns the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the sign
    of its imaginary part.

    `keras.ops.conj` is a shorthand for this function.

    Args:
        x: Input tensor.

    Returns:
        The complex conjugate of each element in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Conjugate().symbolic_call(x)
    return backend.numpy.conjugate(x)


class Conj(Conjugate):
    pass


@keras_export(["keras.ops.conj", "keras.ops.numpy.conj"])
def conj(x):
    """Shorthand for `keras.ops.conjugate`."""
    return conjugate(x)


class Copy(Operation):
    def call(self, x):
        return backend.numpy.copy(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.copy", "keras.ops.numpy.copy"])
def copy(x):
    """Returns a copy of `x`.

    Args:
        x: Input tensor.

    Returns:
        A copy of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Copy().symbolic_call(x)
    return backend.numpy.copy(x)


class Cos(Operation):
    def call(self, x):
        return backend.numpy.cos(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.cos", "keras.ops.numpy.cos"])
def cos(x):
    """Cosine, element-wise.

    Args:
        x: Input tensor.

    Returns:
        The corresponding cosine values.
    """
    if any_symbolic_tensors((x,)):
        return Cos().symbolic_call(x)
    return backend.numpy.cos(x)


class Cosh(Operation):
    def call(self, x):
        return backend.numpy.cosh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.cosh", "keras.ops.numpy.cosh"])
def cosh(x):
    """Hyperbolic cosine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Cosh().symbolic_call(x)
    return backend.numpy.cosh(x)


class CountNonzero(Operation):
    def __init__(self, axis=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = axis

    def call(self, x):
        return backend.numpy.count_nonzero(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis),
            dtype="int32",
        )


@keras_export(
    [
        "keras.ops.count_nonzero",
        "keras.ops.numpy.count_nonzero",
    ]
)
def count_nonzero(x, axis=None):
    """Counts the number of non-zero values in `x` along the given `axis`.

    If no axis is specified then all non-zeros in the tensor are counted.

    Args:
        x: Input tensor.
        axis: Axis or tuple of axes along which to count the number of
            non-zeros. Defaults to `None`.

    Returns:
        int or tensor of ints.

    Examples:
    >>> x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
    >>> keras.ops.count_nonzero(x)
    5
    >>> keras.ops.count_nonzero(x, axis=0)
    array([1, 1, 2, 1], dtype=int64)
    >>> keras.ops.count_nonzero(x, axis=1)
    array([2, 3], dtype=int64)
    """
    if any_symbolic_tensors((x,)):
        return CountNonzero(axis=axis).symbolic_call(x)
    return backend.numpy.count_nonzero(x, axis=axis)


class Cross(Operation):
    def __init__(self, axisa=-1, axisb=-1, axisc=-1, axis=None):
        super().__init__()
        if axis is not None:
            self.axisa = axis
            self.axisb = axis
            self.axisc = axis
        else:
            self.axisa = axisa
            self.axisb = axisb
            self.axisc = axisc

    def call(self, x1, x2):
        return backend.numpy.cross(x1, x2, self.axisa, self.axisb, self.axisc)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)

        x1_value_size = x1_shape[self.axisa]
        x2_value_size = x2_shape[self.axisa]
        del x1_shape[self.axisa]
        del x2_shape[self.axisb]
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        if x1_value_size is not None and x1_value_size not in (2, 3):
            raise ValueError(
                "`x1`'s dim on `axis={axisa}` must be either 2 or 3, but "
                f"received: {x1_value_size}"
            )
        if x2_value_size is not None and x2_value_size not in (2, 3):
            raise ValueError(
                "`x2`'s dim on `axis={axisb}` must be either 2 or 3, but "
                f"received: {x2_value_size}"
            )

        if x1_value_size == 3 or x2_value_size == 3:
            value_size = [3]
        else:
            value_size = []

        output_shape = (
            output_shape[: self.axisc] + value_size + output_shape[self.axisc :]
        )

        dtype = dtypes.result_type(x1.dtype, x2.dtype)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.cross", "keras.ops.numpy.cross"])
def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Returns the cross product of two (arrays of) vectors.

    The cross product of `x1` and `x2` in R^3 is a vector
    perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of
    vectors, the vectors are defined by the last axis of `x1` and `x2`
    by default, and these axes can have dimensions 2 or 3.

    Where the dimension of either `x1` or `x2` is 2, the third component of
    the input vector is assumed to be zero and the cross product calculated
    accordingly.

    In cases where both input vectors have dimension 2, the z-component of
    the cross product is returned.

    Args:
        x1: Components of the first vector(s).
        x2: Components of the second vector(s).
        axisa: Axis of `x1` that defines the vector(s). Defaults to `-1`.
        axisb: Axis of `x2` that defines the vector(s). Defaults to `-1`.
        axisc: Axis of the result containing the cross product vector(s).
            Ignored if both input vectors have dimension 2, as the return is
            scalar. By default, the last axis.
        axis: If defined, the axis of `x1`, `x2` and the result that
            defines the vector(s) and cross product(s). Overrides `axisa`,
            `axisb` and `axisc`.

    Note:
        Torch backend does not support two dimensional vectors, or the
        arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.

    Returns:
        Vector cross product(s).
    """
    if any_symbolic_tensors((x1, x2)):
        return Cross(
            axisa=axisa, axisb=axisb, axisc=axisc, axis=axis
        ).symbolic_call(x1, x2)
    return backend.numpy.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


class Cumprod(Operation):
    def __init__(self, axis=None, dtype=None):
        super().__init__()
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.cumprod(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
        else:
            output_shape = x.shape
        output_dtype = backend.standardize_dtype(self.dtype or x.dtype)
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.cumprod", "keras.ops.numpy.cumprod"])
def cumprod(x, axis=None, dtype=None):
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative product is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.
    """
    return Cumprod(axis=axis, dtype=dtype)(x)


class Cumsum(Operation):
    def __init__(self, axis=None, dtype=None):
        super().__init__()
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.cumsum(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
        else:
            output_shape = x.shape
        output_dtype = backend.standardize_dtype(self.dtype or x.dtype)
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.cumsum", "keras.ops.numpy.cumsum"])
def cumsum(x, axis=None, dtype=None):
    """Returns the cumulative sum of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.
            By default the input is flattened.
        dtype: dtype of returned tensor. Defaults to x.dtype.

    Returns:
        Output tensor.
    """
    return Cumsum(axis=axis, dtype=dtype)(x)


class Diag(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.diag(x, k=self.k)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if len(x_shape) == 1:
            if x_shape[0] is None:
                output_shape = [None, None]
            else:
                output_shape = [
                    x_shape[0] + int(np.abs(self.k)),
                    x_shape[0] + int(np.abs(self.k)),
                ]
        elif len(x_shape) == 2:
            if None in x_shape:
                output_shape = [None]
            else:
                shorter_side = np.minimum(x_shape[0], x_shape[1])
                if self.k > 0:
                    remaining = x_shape[1] - self.k
                else:
                    remaining = x_shape[0] + self.k
                output_shape = [
                    int(np.maximum(0, np.minimum(remaining, shorter_side)))
                ]
        else:
            raise ValueError(
                f"`x` must be 1-D or 2-D, but received shape {x.shape}."
            )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.diag", "keras.ops.numpy.diag"])
def diag(x, k=0):
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
            If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
        k: The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals
            above the main diagonal, and `k < 0` for diagonals below
            the main diagonal.

    Returns:
        The extracted diagonal or constructed diagonal tensor.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> ops.diag(x)
    array([0, 4, 8])
    >>> ops.diag(x, k=1)
    array([1, 5])
    >>> ops.diag(x, k=-1)
    array([3, 7])

    >>> ops.diag(ops.diag(x)))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    if any_symbolic_tensors((x,)):
        return Diag(k=k).symbolic_call(x)
    return backend.numpy.diag(x, k=k)


class Diagonal(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.diagonal(
            x,
            offset=self.offset,
            axis1=self.axis1,
            axis2=self.axis2,
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if len(x_shape) < 2:
            raise ValueError(
                "`diagonal` requires an array of at least two dimensions, but "
                "`x` is of shape {x.shape}."
            )

        shape_2d = [x_shape[self.axis1], x_shape[self.axis2]]
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = list(filter((-1).__ne__, x_shape))
        if None in shape_2d:
            diag_shape = [None]
        else:
            shorter_side = np.minimum(shape_2d[0], shape_2d[1])
            if self.offset > 0:
                remaining = shape_2d[1] - self.offset
            else:
                remaining = shape_2d[0] + self.offset
            diag_shape = [
                int(np.maximum(0, np.minimum(remaining, shorter_side)))
            ]
        output_shape = output_shape + diag_shape
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.diagonal", "keras.ops.numpy.diagonal"])
def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
    collection of elements of the form `x[i, i+offset]`.

    If `x` has more than two dimensions, the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal
    is returned.

    The shape of the resulting array can be determined by removing `axis1`
    and `axis2` and appending an index to the right equal to the size of
    the resulting diagonals.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to `0`.(main diagonal).
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to `0`.(first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to `1` (second axis).

    Returns:
        Tensor of diagonals.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.arange(4).reshape((2, 2))
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.diagonal()
    array([0, 3])
    >>> x.diagonal(1)
    array([1])

    >>> x = ops.arange(8).reshape((2, 2, 2))
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> x.diagonal(0, 0, 1)
    array([[0, 6],
           [1, 7]])
    """
    if any_symbolic_tensors((x,)):
        return Diagonal(
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        ).symbolic_call(x)
    return backend.numpy.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


class Diff(Operation):
    def __init__(self, n=1, axis=-1):
        super().__init__()
        self.n = n
        self.axis = axis

    def call(self, a):
        return backend.numpy.diff(a, n=self.n, axis=self.axis)

    def compute_output_spec(self, a):
        shape = list(a.shape)
        size = shape[self.axis]
        if size is not None:
            shape[self.axis] = builtins.max(size - self.n, 0)
        return KerasTensor(shape, dtype=a.dtype)


@keras_export(["keras.ops.diff", "keras.ops.numpy.diff"])
def diff(a, n=1, axis=-1):
    """Calculate the n-th discrete difference along the given axis.

    The first difference is given by `out[i] = a[i+1] - a[i]` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Args:
        a: Input tensor.
        n: The number of times values are differenced. Defaults to `1`.
        axis: Axis to compute discrete difference(s) along.
            Defaults to `-1`.(last axis).

    Returns:
        Tensor of diagonals.

    Examples:
    >>> from keras.src import ops
    >>> x = ops.convert_to_tensor([1, 2, 4, 7, 0])
    >>> ops.diff(x)
    array([ 1,  2,  3, -7])
    >>> ops.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = ops.convert_to_tensor([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> ops.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> ops.diff(x, axis=0)
    array([[-1,  2,  0, -2]])
    """
    return Diff(n=n, axis=axis)(a)


class Digitize(Operation):
    def call(self, x, bins):
        return backend.numpy.digitize(x, bins)

    def compute_output_spec(self, x, bins):
        bins_shape = bins.shape
        if len(bins_shape) > 1:
            raise ValueError(
                f"`bins` must be a 1D array. Received: bins={bins} "
                f"with shape bins.shape={bins_shape}"
            )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype="int32", sparse=sparse)


@keras_export(["keras.ops.digitize", "keras.ops.numpy.digitize"])
def digitize(x, bins):
    """Returns the indices of the bins to which each value in `x` belongs.

    Args:
        x: Input array to be binned.
        bins: Array of bins. It has to be one-dimensional and monotonically
            increasing.

    Returns:
        Output array of indices, of same shape as `x`.

    Example:
    >>> x = np.array([0.0, 1.0, 3.0, 1.6])
    >>> bins = np.array([0.0, 3.0, 4.5, 7.0])
    >>> keras.ops.digitize(x, bins)
    array([1, 1, 2, 1])
    """
    if any_symbolic_tensors((x, bins)):
        return Digitize().symbolic_call(x, bins)
    return backend.numpy.digitize(x, bins)


class Dot(Operation):
    def call(self, x1, x2):
        return backend.numpy.dot(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if x1_shape == [] or x2_shape == []:
            return multiply(x1, x2)
        if len(x1_shape) == 1 and len(x2_shape) == 1:
            return KerasTensor([], dtype=dtype)
        if len(x2_shape) == 1:
            if x1_shape[-1] != x2_shape[0]:
                raise ValueError(
                    "Shape must match on the last axis of `x1` and `x2` when "
                    "`x1` is N-d array while `x2` is 1-D, but receive shape "
                    f"`x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
                )
            return KerasTensor(x1_shape[:-1], dtype=dtype)

        if (
            x1_shape[-1] is None
            or x2_shape[-2] is None
            or x1_shape[-1] == x2_shape[-2]
        ):
            del x1_shape[-1]
            del x2_shape[-2]
            return KerasTensor(x1_shape + x2_shape, dtype=dtype)

        raise ValueError(
            "Shape must match on the last axis of `x1` and second last "
            "axis of `x2` when `x1` is N-d array while `x2` is M-D, but "
            f"received `x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
        )


@keras_export(["keras.ops.dot", "keras.ops.numpy.dot"])
def dot(x1, x2):
    """Dot product of two tensors.

    - If both `x1` and `x2` are 1-D tensors, it is inner product of vectors
      (without complex conjugation).
    - If both `x1` and `x2` are 2-D tensors, it is matrix multiplication.
    - If either `x1` or `x2` is 0-D (scalar), it is equivalent to `x1 * x2`.
    - If `x1` is an N-D tensor and `x2` is a 1-D tensor, it is a sum product
      over the last axis of `x1` and `x2`.
    - If `x1` is an N-D tensor and `x2` is an M-D tensor (where `M>=2`),
      it is a sum product over the last axis of `x1` and the second-to-last
      axis of `x2`: `dot(x1, x2)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`.

    Args:
        x1: First argument.
        x2: Second argument.

    Note:
        Torch backend does not accept 0-D tensors as arguments.

    Returns:
        Dot product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Dot().symbolic_call(x1, x2)
    return backend.numpy.dot(x1, x2)


class Einsum(Operation):
    def __init__(self, subscripts):
        super().__init__()
        self.subscripts = subscripts

    def call(self, *operands):
        return backend.numpy.einsum(self.subscripts, *operands)

    def compute_output_spec(self, *operands):
        """Compute the output shape of `einsum`.

        The shape computation follows the steps below:
        1. Find all letters in the input specs (left part of "->"), and
            break them into two categories: letters appearing more than once
            go to `reduced_dims`, otherwise go to `kept_dims`.
        2. Adjust `reduced_dims` and `kept_dims` based on the output spec
            (right part of "->"). The rule is if the letter appears in the
            output spec, then move it to `kept_dims`, otherwise move it to
            `reduced_dims`.
        3. Compute the target output shape. If no output spec is set, then
            the target output shape will be "...{kept_dims}", e.g., "...ijk",
            else it will be the same as output spec. "..." is a wildcard that
            could map shape of arbitrary length.
        4. For each operand in `operands`, map the shape specified in the input
            spec to the output target, e.g, if operand is of shape [2,3,4],
            input spec is "i..." and output target is "i...jk", then 2 will go
            the index 0. For dims not represented by any letter, insert to the
            wildcard part. For each letter in output target not appearing in
            input spec, the dim will be 1 for broadcasting. After 4, each
            operand should have a target shape containing only number and
            `None`.
        5. Broadcast all shapes computed from 4, and the result is the output
            shape.

        Let's take an example to illustrate the steps above. Let's define:
        ```python
        x = KerasTensor([None, 3, 4])
        y = KerasTensor(2, 4, 3)
        z = knp.einsum("...ij, kji->...k", x, y)
        ```

        1. `reduced_dims` is {"i", "j"}, `kept_dims` is {"k"}.
        2. `reduced_dims` is still {"i", "j"}, and `kept_dims` is {"k"}.
        3. Output target is "...k".
        4. For `x`, the input spec is "...ij", and the output target is "...k".
            "i" and "j" do not appear in the output target, so no replacement
            happens, and [None] goes to wildcard. Afterwards, "k" is replaced
            by 1, so we get shape [None, 1]. Applying the same logic to `y`, we
            get shape [2].
        5. Broadcast [None, 1] and [2], and we get [None, 2], which is the
            output shape.
        """
        split_subscripts = self.subscripts.split("->")
        if len(split_subscripts) > 2:
            raise ValueError(
                "At most one '->' is supported in `einsum` subscripts, but "
                f"received {self.subscripts}."
            )
        if len(split_subscripts) == 2:
            subscripts = split_subscripts[0]
            output_spec = split_subscripts[1]
        else:
            subscripts = self.subscripts
            output_spec = None
        input_specs = subscripts.split(",")
        if len(input_specs) != len(operands):
            raise ValueError(
                f"Number of operands ({len(operands)}) does not match the "
                f"number of input specs ({len(input_specs)}) in `einsum`, "
                f"received subscripts={self.subscripts}."
            )
        reduced_dims = set()
        kept_dims = set()
        for s in subscripts:
            if not s.isalpha():
                continue
            if s not in reduced_dims and s not in kept_dims:
                kept_dims.add(s)
            elif s in kept_dims:
                kept_dims.remove(s)
                reduced_dims.add(s)

        if output_spec is not None:
            # The output spec changes the rule of kept_dims and reduced_dims.
            # In short, dims appearing in the output spec will be kept, and
            # dims not appearing in the output spec will be reduced.
            kept_dims_copy = kept_dims.copy()
            reduced_dims_copy = reduced_dims.copy()
            for dim in kept_dims:
                if dim not in output_spec:
                    kept_dims_copy.remove(dim)
                    reduced_dims_copy.add(dim)
            for dim in reduced_dims:
                if dim in output_spec:
                    reduced_dims_copy.remove(dim)
                    kept_dims_copy.add(dim)
            kept_dims = kept_dims_copy
            reduced_dims = reduced_dims_copy

        reduced_dims = sorted(reduced_dims)
        kept_dims = sorted(kept_dims)

        if output_spec is None:
            target_broadcast_spec = "..." + "".join(kept_dims)
        else:
            target_broadcast_spec = output_spec

        expanded_operands_shapes = []
        for x, spec in zip(operands, input_specs):
            x_shape = getattr(x, "shape", [])
            x_shape = [-1 if size is None else size for size in x_shape]
            split_spec = spec.split("...")
            expanded_shape = target_broadcast_spec
            if len(split_spec) == 1:
                # In this case, the input spec is just a string of letters,
                # e.g., "ijk".
                if len(x_shape) != len(split_spec[0]):
                    raise ValueError(
                        "Number of dimensions in the subscript does not "
                        "match the number of dimensions in the operand, "
                        f"received subscript `{spec}` and operand of shape "
                        f"{x_shape}."
                    )
                for size, s in zip(x_shape, split_spec[0]):
                    # Replace the letter with the right shape.
                    expanded_shape = expanded_shape.replace(s, str(size) + " ")
                expanded_shape = expanded_shape.replace("...", "")
            else:
                # In this case, the input spec has "...", e.g., "i...j", "i...",
                # or "...j".
                for i in range(len(split_spec[0])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[0][i], str(x_shape[i]) + " "
                    )
                for i in range(len(split_spec[1])):
                    expanded_shape = expanded_shape.replace(
                        split_spec[1][-i - 1], str(x_shape[-i - 1]) + " "
                    )
                # Shape matched by "..." will be inserted to the position of
                # "...".
                wildcard_shape_start_index = len(split_spec[0])
                wildcard_shape_end_index = (
                    len(x_shape)
                    if len(split_spec[1]) == 0
                    else -len(split_spec[1])
                )
                wildcard_shape = x_shape[
                    wildcard_shape_start_index:wildcard_shape_end_index
                ]
                wildcard_shape_str = (
                    " ".join([str(size) for size in wildcard_shape]) + " "
                )
                expanded_shape = expanded_shape.replace(
                    "...", wildcard_shape_str
                )
            # Replace all letters not yet handled with "1" for broadcasting.
            expanded_shape = re.sub("[a-z]", "1 ", expanded_shape)
            expanded_shape = expanded_shape.split()
            expanded_shape = [
                None if size == "-1" else int(size) for size in expanded_shape
            ]
            expanded_operands_shapes.append(expanded_shape)

        output_shape = expanded_operands_shapes[0]
        for shape in expanded_operands_shapes[1:]:
            output_shape = broadcast_shapes(output_shape, shape)
        dtypes_to_resolve = list(
            set(
                backend.standardize_dtype(getattr(x, "dtype", type(x)))
                for x in operands
            )
        )
        if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
            dtype = "int32"
        else:
            dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.einsum", "keras.ops.numpy.einsum"])
def einsum(subscripts, *operands):
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts: Specifies the subscripts for summation as comma separated
            list of subscript labels. An implicit (classical Einstein
            summation) calculation is performed unless the explicit indicator
            `->` is included as well as subscript labels of the precise
            output form.
        operands: The operands to compute the Einstein sum of.

    Returns:
        The calculation based on the Einstein summation convention.

    Example:
    >>> from keras.src import ops
    >>> a = ops.arange(25).reshape(5, 5)
    >>> b = ops.arange(5)
    >>> c = ops.arange(6).reshape(2, 3)

    Trace of a matrix:

    >>> ops.einsum("ii", a)
    60
    >>> ops.einsum(a, [0, 0])
    60
    >>> ops.trace(a)
    60

    Extract the diagonal:

    >>> ops.einsum("ii -> i", a)
    array([ 0,  6, 12, 18, 24])
    >>> ops.einsum(a, [0, 0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> ops.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis:

    >>> ops.einsum("ij -> i", a)
    array([ 10,  35,  60,  85, 110])
    >>> ops.einsum(a, [0, 1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> ops.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional tensors summing a single axis can be done
    with ellipsis:

    >>> ops.einsum("...j -> ...", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [..., 1], [...])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose or reorder any number of axes:

    >>> ops.einsum("ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum("ij -> ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.einsum(c, [1, 0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> ops.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Matrix vector multiplication:

    >>> ops.einsum("ij, j", a, b)
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum(a, [0, 1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> ops.einsum("...j, j", a, b)
    array([ 30,  80, 130, 180, 230])
    """
    if any_symbolic_tensors(operands):
        return Einsum(subscripts).symbolic_call(*operands)
    return backend.numpy.einsum(subscripts, *operands)


class Empty(Operation):
    def call(self, shape, dtype=None):
        return backend.numpy.empty(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype=None):
        dtype = dtype or backend.floatx()
        return KerasTensor(shape, dtype=dtype)


@keras_export(["keras.ops.empty", "keras.ops.numpy.empty"])
def empty(shape, dtype=None):
    """Return a tensor of given shape and type filled with uninitialized data.

    Args:
        shape: Shape of the empty tensor.
        dtype: Desired data type of the empty tensor.

    Returns:
        The empty tensor.
    """
    return backend.numpy.empty(shape, dtype=dtype)


class Equal(Operation):
    def call(self, x1, x2):
        return backend.numpy.equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.equal", "keras.ops.numpy.equal"])
def equal(x1, x2):
    """Returns `(x1 == x2)` element-wise.

    Args:
        x1: Tensor to compare.
        x2: Tensor to compare.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Equal().symbolic_call(x1, x2)
    return backend.numpy.equal(x1, x2)


class Exp(Operation):
    def call(self, x):
        return backend.numpy.exp(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(x.dtype)
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.exp", "keras.ops.numpy.exp"])
def exp(x):
    """Calculate the exponential of all elements in the input tensor.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise exponential of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Exp().symbolic_call(x)
    return backend.numpy.exp(x)


class ExpandDims(Operation):
    def __init__(self, axis):
        super().__init__()
        if not isinstance(axis, (int, tuple, list)):
            raise ValueError(
                "The `axis` argument to `expand_dims` should be an integer, "
                f"tuple or list. Received axis={axis}"
            )
        self.axis = axis

    def call(self, x):
        return backend.numpy.expand_dims(x, self.axis)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_expand_dims_output_shape(
            x.shape, self.axis
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(
    [
        "keras.ops.expand_dims",
        "keras.ops.numpy.expand_dims",
    ]
)
def expand_dims(x, axis):
    """Expand the shape of a tensor.

    Insert a new axis at the `axis` position in the expanded tensor shape.

    Args:
        x: Input tensor.
        axis: Position in the expanded axes where the new axis
            (or axes) is placed.

    Returns:
        Output tensor with the number of dimensions increased.
    """
    if any_symbolic_tensors((x,)):
        return ExpandDims(axis=axis).symbolic_call(x)
    return backend.numpy.expand_dims(x, axis)


class Expm1(Operation):
    def call(self, x):
        return backend.numpy.expm1(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(x.dtype)
        if "int" in dtype or dtype == "bool":
            dtype = backend.floatx()
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.expm1", "keras.ops.numpy.expm1"])
def expm1(x):
    """Calculate `exp(x) - 1` for all elements in the tensor.

    Args:
        x: Input values.

    Returns:
        Output tensor, element-wise exponential minus one.
    """
    if any_symbolic_tensors((x,)):
        return Expm1().symbolic_call(x)
    return backend.numpy.expm1(x)


class Flip(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.flip(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.flip", "keras.ops.numpy.flip"])
def flip(x, axis=None):
    """Reverse the order of elements in the tensor along the given axis.

    The shape of the tensor is preserved, but the elements are reordered.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to flip the tensor. The default,
            `axis=None`, will flip over all of the axes of the input tensor.

    Returns:
        Output tensor with entries of `axis` reversed.
    """
    if any_symbolic_tensors((x,)):
        return Flip(axis=axis).symbolic_call(x)
    return backend.numpy.flip(x, axis=axis)


class Floor(Operation):
    def call(self, x):
        return backend.numpy.floor(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.floor", "keras.ops.numpy.floor"])
def floor(x):
    """Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise floor of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Floor().symbolic_call(x)
    return backend.numpy.floor(x)


class Full(Operation):
    def call(self, shape, fill_value, dtype=None):
        return backend.numpy.full(shape, fill_value, dtype=dtype)

    def compute_output_spec(self, shape, fill_value, dtype=None):
        dtype = dtype or backend.floatx()
        return KerasTensor(shape, dtype=dtype)


@keras_export(["keras.ops.full", "keras.ops.numpy.full"])
def full(shape, fill_value, dtype=None):
    """Return a new tensor of given shape and type, filled with `fill_value`.

    Args:
        shape: Shape of the new tensor.
        fill_value: Fill value.
        dtype: Desired data type of the tensor.

    Returns:
        Output tensor.
    """
    return backend.numpy.full(shape, fill_value, dtype=dtype)


class FullLike(Operation):
    def call(self, x, fill_value, dtype=None):
        return backend.numpy.full_like(x, fill_value, dtype=dtype)

    def compute_output_spec(self, x, fill_value, dtype=None):
        dtype = dtype or x.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.full_like", "keras.ops.numpy.full_like"])
def full_like(x, fill_value, dtype=None):
    """Return a full tensor with the same shape and type as the given tensor.

    Args:
        x: Input tensor.
        fill_value: Fill value.
        dtype: Overrides data type of the result.

    Returns:
        Tensor of `fill_value` with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return FullLike().symbolic_call(x, fill_value, dtype=dtype)
    return backend.numpy.full_like(x, fill_value, dtype=dtype)


class GetItem(Operation):
    def call(self, x, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[key]

    def compute_output_spec(self, x, key):
        remaining_shape = list(x.shape)
        new_shape = []
        if isinstance(key, int):
            remaining_key = [key]
        elif isinstance(key, tuple):
            remaining_key = list(key)
        elif isinstance(key, list):
            remaining_key = key.copy()
        else:
            raise ValueError(
                f"Unsupported key type for array slice. Recieved: `{key}`"
            )
        num_ellipses = remaining_key.count(Ellipsis)
        if num_ellipses > 1:
            raise ValueError(
                f"Slice should only have one ellipsis. Recieved: `{key}`"
            )
        elif num_ellipses == 0:
            # Add an implicit final ellipsis.
            remaining_key.append(Ellipsis)
        # Consume slice key element by element.
        while True:
            if not remaining_key:
                break
            subkey = remaining_key.pop(0)
            # Check for `newaxis` and `Ellipsis`.
            if subkey == Ellipsis:
                # Keep as many slices remain in our key, omitting `newaxis`.
                needed = len(remaining_key) - remaining_key.count(np.newaxis)
                consumed = len(remaining_shape) - needed
                new_shape += remaining_shape[:consumed]
                remaining_shape = remaining_shape[consumed:]
                continue
            # All frameworks follow numpy for newaxis. `np.newaxis == None`.
            if subkey == np.newaxis:
                new_shape.append(1)
                continue
            # At this point, we need to consume a new axis from the shape.
            if not remaining_shape:
                raise ValueError(
                    f"Array has shape {x.shape} but slice "
                    f"has to many indices. Received: `{key}`"
                )
            length = remaining_shape.pop(0)
            if isinstance(subkey, int):
                if length is not None:
                    index = subkey if subkey >= 0 else subkey + length
                    if index < 0 or index >= length:
                        raise ValueError(
                            f"Array has shape {x.shape} but out-of-bounds "
                            f"index {key} was requested."
                        )
            elif isinstance(subkey, slice):
                if length is not None:
                    # python3 friendly way to compute a slice length.
                    new_length = len(range(*subkey.indices(length)))
                    new_shape.append(new_length)
                else:
                    new_shape.append(length)
            else:
                raise ValueError(
                    f"Unsupported key type for array slice. Received: `{key}`"
                )
        return KerasTensor(tuple(new_shape), dtype=x.dtype)


@keras_export(["keras.ops.get_item", "keras.ops.numpy.get_item"])
def get_item(x, key):
    """Return `x[key]`."""
    if any_symbolic_tensors((x,)):
        return GetItem().symbolic_call(x, key)
    return x[key]


class Greater(Operation):
    def call(self, x1, x2):
        return backend.numpy.greater(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.greater", "keras.ops.numpy.greater"])
def greater(x1, x2):
    """Return the truth value of `x1 > x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Greater().symbolic_call(x1, x2)
    return backend.numpy.greater(x1, x2)


class GreaterEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.greater_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.greater_equal",
        "keras.ops.numpy.greater_equal",
    ]
)
def greater_equal(x1, x2):
    """Return the truth value of `x1 >= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return GreaterEqual().symbolic_call(x1, x2)
    return backend.numpy.greater_equal(x1, x2)


class Hstack(Operation):
    def call(self, xs):
        return backend.numpy.hstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[1], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[1] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[1]
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[1] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.hstack", "keras.ops.numpy.hstack"])
def hstack(xs):
    """Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the first axis for 1-D tensors,
    and along the second axis for all other tensors.

    Args:
        xs: Sequence of tensors.

    Returns:
        The tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Hstack().symbolic_call(xs)
    return backend.numpy.hstack(xs)


class Identity(Operation):
    def call(self, n, dtype=None):
        return backend.numpy.identity(n, dtype=dtype)

    def compute_output_spec(self, n, dtype=None):
        dtype = dtype or backend.floatx()
        return KerasTensor([n, n], dtype=dtype)


@keras_export(["keras.ops.identity", "keras.ops.numpy.identity"])
def identity(n, dtype=None):
    """Return the identity tensor.

    The identity tensor is a square tensor with ones on the main diagonal and
    zeros elsewhere.

    Args:
        n: Number of rows (and columns) in the `n x n` output tensor.
        dtype: Data type of the output tensor.

    Returns:
        The identity tensor.
    """
    return backend.numpy.identity(n, dtype=dtype)


class Imag(Operation):
    def call(self, x):
        return backend.numpy.imag(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.imag", "keras.ops.numpy.imag"])
def imag(x):
    """Return the imaginary part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The imaginary component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Imag().symbolic_call(x)
    return backend.numpy.imag(x)


class Isclose(Operation):
    def call(self, x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
        return backend.numpy.isclose(x1, x2, rtol, atol, equal_nan)

    def compute_output_spec(
        self, x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False
    ):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.isclose", "keras.ops.numpy.isclose"])
def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return whether two tensors are element-wise almost equal.

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If `True`, element-wise NaNs are considered equal.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Isclose().symbolic_call(x1, x2, rtol, atol, equal_nan)
    return backend.numpy.isclose(x1, x2, rtol, atol, equal_nan)


class Isfinite(Operation):
    def call(self, x):
        return backend.numpy.isfinite(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isfinite", "keras.ops.numpy.isfinite"])
def isfinite(x):
    """Return whether a tensor is finite, element-wise.

    Real values are finite when they are not NaN, not positive infinity, and
    not negative infinity. Complex values are finite when both their real
    and imaginary parts are finite.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isfinite().symbolic_call(x)
    return backend.numpy.isfinite(x)


class Isinf(Operation):
    def call(self, x):
        return backend.numpy.isinf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isinf", "keras.ops.numpy.isinf"])
def isinf(x):
    """Test element-wise for positive or negative infinity.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isinf().symbolic_call(x)
    return backend.numpy.isinf(x)


class Isnan(Operation):
    def call(self, x):
        return backend.numpy.isnan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(["keras.ops.isnan", "keras.ops.numpy.isnan"])
def isnan(x):
    """Test element-wise for NaN and return result as a boolean tensor.

    Args:
        x: Input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x,)):
        return Isnan().symbolic_call(x)
    return backend.numpy.isnan(x)


class Less(Operation):
    def call(self, x1, x2):
        return backend.numpy.less(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.less", "keras.ops.numpy.less"])
def less(x1, x2):
    """Return the truth value of `x1 < x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Less().symbolic_call(x1, x2)
    return backend.numpy.less(x1, x2)


class LessEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.less_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.less_equal",
        "keras.ops.numpy.less_equal",
    ]
)
def less_equal(x1, x2):
    """Return the truth value of `x1 <= x2` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparison of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return LessEqual().symbolic_call(x1, x2)
    return backend.numpy.less_equal(x1, x2)


class Linspace(Operation):
    def __init__(
        self, num=50, endpoint=True, retstep=False, dtype=float, axis=0
    ):
        super().__init__()
        self.num = num
        self.endpoint = endpoint
        self.retstep = retstep
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.numpy.linspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            retstep=self.retstep,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        if self.axis == -1:
            output_shape = output_shape + [self.num]
        elif self.axis >= 0:
            output_shape = (
                output_shape[: self.axis]
                + [self.num]
                + output_shape[self.axis :]
            )
        else:
            output_shape = (
                output_shape[: self.axis + 1]
                + [self.num]
                + output_shape[self.axis + 1 :]
            )

        dtype = (
            self.dtype
            if self.dtype is not None
            else getattr(start, "dtype", type(start))
        )
        dtype = backend.result_type(dtype, float)
        if self.retstep:
            return (KerasTensor(output_shape, dtype=dtype), None)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.linspace", "keras.ops.numpy.linspace"])
def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    """Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the interval
    `[start, stop]`.

    The endpoint of the interval can optionally be excluded.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence, unless `endpoint` is set to
            `False`. In that case, the sequence consists of all but the last
            of `num + 1` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is `False`.
        num: Number of samples to generate. Defaults to `50`. Must be
            non-negative.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is
            not included. Defaults to `True`.
        retstep: If `True`, return `(samples, step)`, where `step` is the
            spacing between samples.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only if
            start or stop are array-like. Defaults to `0`.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced numbers.
        If `retstep` is `True`, returns `(samples, step)`
    """
    if any_symbolic_tensors((start, stop)):
        return Linspace(num, endpoint, retstep, dtype, axis)(start, stop)
    return backend.numpy.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


class Log(Operation):
    def call(self, x):
        return backend.numpy.log(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log", "keras.ops.numpy.log"])
def log(x):
    """Natural logarithm, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log().symbolic_call(x)
    return backend.numpy.log(x)


class Log10(Operation):
    def call(self, x):
        return backend.numpy.log10(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log10", "keras.ops.numpy.log10"])
def log10(x):
    """Return the base 10 logarithm of the input tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base 10 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log10().symbolic_call(x)
    return backend.numpy.log10(x)


class Log1p(Operation):
    def call(self, x):
        return backend.numpy.log1p(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.log1p", "keras.ops.numpy.log1p"])
def log1p(x):
    """Returns the natural logarithm of one plus the `x`, element-wise.

    Calculates `log(1 + x)`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise natural logarithm of `1 + x`.
    """
    if any_symbolic_tensors((x,)):
        return Log1p().symbolic_call(x)
    return backend.numpy.log1p(x)


class Log2(Operation):
    def call(self, x):
        return backend.numpy.log2(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.log2", "keras.ops.numpy.log2"])
def log2(x):
    """Base-2 logarithm of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise base-2 logarithm of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Log2().symbolic_call(x)
    return backend.numpy.log2(x)


class Logaddexp(Operation):
    def call(self, x1, x2):
        return backend.numpy.logaddexp(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.logaddexp", "keras.ops.numpy.logaddexp"])
def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs.

    Calculates `log(exp(x1) + exp(x2))`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logarithm of the sum of exponentiations
        of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Logaddexp().symbolic_call(x1, x2)
    return backend.numpy.logaddexp(x1, x2)


class LogicalAnd(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_and(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_and",
        "keras.ops.numpy.logical_and",
    ]
)
def logical_and(x1, x2):
    """Computes the element-wise logical AND of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical AND of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalAnd().symbolic_call(x1, x2)
    return backend.numpy.logical_and(x1, x2)


class LogicalNot(Operation):
    def call(self, x):
        return backend.numpy.logical_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_not",
        "keras.ops.numpy.logical_not",
    ]
)
def logical_not(x):
    """Computes the element-wise NOT of the given input tensor.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise logical NOT of the input.
    """
    if any_symbolic_tensors((x,)):
        return LogicalNot().symbolic_call(x)
    return backend.numpy.logical_not(x)


class LogicalOr(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_or(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(
    [
        "keras.ops.logical_or",
        "keras.ops.numpy.logical_or",
    ]
)
def logical_or(x1, x2):
    """Computes the element-wise logical OR of the given input tensors.

    Zeros are treated as `False` and non-zeros are treated as `True`.

    Args:
        x1: Input tensor.
        x2: Input tensor.

    Returns:
        Output tensor, element-wise logical OR of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalOr().symbolic_call(x1, x2)
    return backend.numpy.logical_or(x1, x2)


class Logspace(Operation):
    def __init__(self, num=50, endpoint=True, base=10, dtype=float, axis=0):
        super().__init__()
        self.num = num
        self.endpoint = endpoint
        self.base = base
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.numpy.logspace(
            start,
            stop,
            num=self.num,
            endpoint=self.endpoint,
            base=self.base,
            dtype=self.dtype,
            axis=self.axis,
        )

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, "shape", [])
        stop_shape = getattr(stop, "shape", [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        if self.axis == -1:
            output_shape = output_shape + [self.num]
        elif self.axis >= 0:
            output_shape = (
                output_shape[: self.axis]
                + [self.num]
                + output_shape[self.axis :]
            )
        else:
            output_shape = (
                output_shape[: self.axis + 1]
                + [self.num]
                + output_shape[self.axis + 1 :]
            )
        dtype = (
            self.dtype
            if self.dtype is not None
            else getattr(start, "dtype", type(start))
        )
        dtype = backend.result_type(dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.logspace", "keras.ops.numpy.logspace"])
def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    """Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at `base ** start` and ends with
    `base ** stop` (see `endpoint` below).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless `endpoint` is `False`.
            In that case, `num + 1` values are spaced over the interval in
            log-space, of which all but the last (a sequence of length `num`)
            are returned.
        num: Number of samples to generate. Defaults to `50`.
        endpoint: If `True`, `stop` is the last sample. Otherwise, it is not
            included. Defaults to `True`.
        base: The base of the log space. Defaults to `10`.
        dtype: The type of the output tensor.
        axis: The axis in the result to store the samples. Relevant only
            if start or stop are array-like.

    Note:
        Torch backend does not support `axis` argument.

    Returns:
        A tensor of evenly spaced samples on a log scale.
    """
    if any_symbolic_tensors((start, stop)):
        return Logspace(num, endpoint, base, dtype, axis)(start, stop)
    return backend.numpy.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.numpy.matmul(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = operation_utils.compute_matmul_output_shape(
            x1_shape, x2_shape
        )
        x1_sparse = getattr(x1, "sparse", True)
        x2_sparse = getattr(x2, "sparse", True)
        output_sparse = x1_sparse and x2_sparse
        x1_dtype = backend.standardize_dtype(getattr(x1, "dtype", type(x1)))
        x2_dtype = backend.standardize_dtype(getattr(x2, "dtype", type(x2)))
        if x1_dtype == "int8" and x2_dtype == "int8":
            dtype = "int32"
        else:
            dtype = dtypes.result_type(x1_dtype, x2_dtype)
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.matmul", "keras.ops.numpy.matmul"])
def matmul(x1, x2):
    """Matrix product of two tensors.

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If either tensor is N-D, N > 2, it is treated as a stack of matrices
      residing in the last two indexes and broadcast accordingly.
    - If the first tensor is 1-D, it is promoted to a matrix by prepending
      a 1 to its dimensions. After matrix multiplication the prepended
      1 is removed.
    - If the second tensor is 1-D, it is promoted to a matrix by appending a 1
      to its dimensions. After matrix multiplication the appended 1 is removed.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, matrix product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Matmul().symbolic_call(x1, x2)
    return backend.numpy.matmul(x1, x2)


class Max(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.numpy.max(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_export(["keras.ops.max", "keras.ops.numpy.max"])
def max(x, axis=None, keepdims=False, initial=None):
    """Return the maximum of a tensor or maximum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.
        initial: The minimum value of an output element. Defaults to `None`.

    Returns:
        Maximum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Max(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.numpy.max(x, axis=axis, keepdims=keepdims, initial=initial)


class Maximum(Operation):
    def call(self, x1, x2):
        return backend.numpy.maximum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.maximum", "keras.ops.numpy.maximum"])
def maximum(x1, x2):
    """Element-wise maximum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise maximum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Maximum().symbolic_call(x1, x2)
    return backend.numpy.maximum(x1, x2)


class Median(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.median(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )
        if backend.standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.median", "keras.ops.numpy.median"])
def median(x, axis=None, keepdims=False):
    """Compute the median along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the medians are computed. Defaults to
            `axis=None` which is to compute the median(s) along a flattened
            version of the array.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.

    Returns:
        The output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Median(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.median(x, axis=axis, keepdims=keepdims)


class Meshgrid(Operation):
    def __init__(self, indexing="xy"):
        super().__init__()
        if indexing not in ("xy", "ij"):
            raise ValueError(
                "Valid values for `indexing` are 'xy' and 'ij', "
                "but received {index}."
            )
        self.indexing = indexing

    def call(self, *x):
        return backend.numpy.meshgrid(*x, indexing=self.indexing)

    def compute_output_spec(self, *x):
        output_shape = []
        for xi in x:
            if len(xi.shape) == 0:
                size = 1
            else:
                if None in xi.shape:
                    size = None
                else:
                    size = int(np.prod(xi.shape))
            output_shape.append(size)
        if self.indexing == "ij":
            return [KerasTensor(output_shape) for _ in range(len(x))]
        tmp = output_shape[0]
        output_shape[0] = output_shape[1]
        output_shape[1] = tmp
        return [
            KerasTensor(output_shape, dtype=xi.dtype) for _ in range(len(x))
        ]


@keras_export(["keras.ops.meshgrid", "keras.ops.numpy.meshgrid"])
def meshgrid(*x, indexing="xy"):
    """Creates grids of coordinates from coordinate vectors.

    Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
    lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
    `G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
    `Gi` is constructed by expanding `Ti` to the result shape.

    Args:
        x: 1-D tensors representing the coordinates of a grid.
        indexing: `"xy"` or `"ij"`. "xy" is cartesian; `"ij"` is matrix
            indexing of output. Defaults to `"xy"`.

    Returns:
        Sequence of N tensors.

    Example:
    >>> from keras.src import ops
    >>> x = ops.array([1, 2, 3])
    >>> y = ops.array([4, 5, 6])

    >>> grid_x, grid_y = ops.meshgrid(x, y, indexing="ij")
    >>> grid_x
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> grid_y
    array([[4, 5, 6],
           [4, 5, 6],
           [4, 5, 6]])
    """
    if any_symbolic_tensors(x):
        return Meshgrid(indexing=indexing).symbolic_call(*x)
    return backend.numpy.meshgrid(*x, indexing=indexing)


class Min(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.initial = initial

    def call(self, x):
        return backend.numpy.min(
            x, axis=self.axis, keepdims=self.keepdims, initial=self.initial
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


@keras_export(["keras.ops.min", "keras.ops.numpy.min"])
def min(x, axis=None, keepdims=False, initial=None):
    """Return the minimum of a tensor or minimum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate. By default, flattened input
            is used.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one. Defaults to `False`.
        initial: The maximum value of an output element. Defaults to `None`.

    Returns:
        Minimum of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Min(axis=axis, keepdims=keepdims, initial=initial).symbolic_call(
            x
        )
    return backend.numpy.min(x, axis=axis, keepdims=keepdims, initial=initial)


class Minimum(Operation):
    def call(self, x1, x2):
        return backend.numpy.minimum(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.minimum", "keras.ops.numpy.minimum"])
def minimum(x1, x2):
    """Element-wise minimum of `x1` and `x2`.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise minimum of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Minimum().symbolic_call(x1, x2)
    return backend.numpy.minimum(x1, x2)


class Mod(Operation):
    def call(self, x1, x2):
        return backend.numpy.mod(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if output_dtype == "bool":
            output_dtype = "int32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.mod", "keras.ops.numpy.mod"])
def mod(x1, x2):
    """Returns the element-wise remainder of division.

    Args:
        x1: First tensor.
        x2: Second tensor.

    Returns:
        Output tensor, element-wise remainder of division.
    """
    if any_symbolic_tensors((x1, x2)):
        return Mod().symbolic_call(x1, x2)
    return backend.numpy.mod(x1, x2)


class Moveaxis(Operation):
    def __init__(self, source, destination):
        super().__init__()
        if isinstance(source, int):
            self.source = [source]
        else:
            self.source = source
        if isinstance(destination, int):
            self.destination = [destination]
        else:
            self.destination = destination

        if len(self.source) != len(self.destination):
            raise ValueError(
                "`source` and `destination` arguments must have the same "
                f"number of elements, but received `source={source}` and "
                f"`destination={destination}`."
            )

    def call(self, x):
        return backend.numpy.moveaxis(x, self.source, self.destination)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        output_shape = [-1 for _ in range(len(x.shape))]
        for sc, dst in zip(self.source, self.destination):
            output_shape[dst] = x_shape[sc]
            x_shape[sc] = -1
        i, j = 0, 0
        while i < len(output_shape):
            while i < len(output_shape) and output_shape[i] != -1:
                # Find the first dim unset.
                i += 1
            while j < len(output_shape) and x_shape[j] == -1:
                # Find the first dim not being passed.
                j += 1
            if i == len(output_shape):
                break
            output_shape[i] = x_shape[j]
            i += 1
            j += 1
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.moveaxis", "keras.ops.numpy.moveaxis"])
def moveaxis(x, source, destination):
    """Move axes of a tensor to new positions.

    Other axes remain in their original order.

    Args:
        x: Tensor whose axes should be reordered.
        source: Original positions of the axes to move. These must be unique.
        destination: Destinations positions for each of the original axes.
            These must also be unique.

    Returns:
        Tensor with moved axes.
    """
    if any_symbolic_tensors((x,)):
        return Moveaxis(source, destination).symbolic_call(x)
    return backend.numpy.moveaxis(x, source=source, destination=destination)


class NanToNum(Operation):
    def __init__(self, nan=0.0, posinf=None, neginf=None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def call(self, x):
        return backend.numpy.nan_to_num(
            x, nan=self.nan, posinf=self.posinf, neginf=self.neginf
        )

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.nan_to_num",
        "keras.ops.numpy.nan_to_num",
    ]
)
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers.

    Args:
        x: Input data.
        nan: Optional float or int. Value to replace `NaN` entries with.
        posinf: Optional float or int.
            Value to replace positive infinity with.
        neginf: Optional float or int.
            Value to replace negative infinity with.

    Returns:
        `x`, with non-finite values replaced.
    """
    if any_symbolic_tensors((x,)):
        return NanToNum(nan=nan, posinf=posinf, neginf=neginf).symbolic_call(x)
    return backend.numpy.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


class Ndim(Operation):
    def call(self, x):
        return backend.numpy.ndim(
            x,
        )

    def compute_output_spec(self, x):
        return KerasTensor([len(x.shape)])


@keras_export(["keras.ops.ndim", "keras.ops.numpy.ndim"])
def ndim(x):
    """Return the number of dimensions of a tensor.

    Args:
        x: Input tensor.

    Returns:
        The number of dimensions in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Ndim().symbolic_call(x)
    return backend.numpy.ndim(x)


class Nonzero(Operation):
    def call(self, x):
        return backend.numpy.nonzero(x)

    def compute_output_spec(self, x):
        return tuple(
            [KerasTensor((None,), dtype="int32") for _ in range(len(x.shape))]
        )


@keras_export(["keras.ops.nonzero", "keras.ops.numpy.nonzero"])
def nonzero(x):
    """Return the indices of the elements that are non-zero.

    Args:
        x: Input tensor.

    Returns:
        Indices of elements that are non-zero.
    """
    if any_symbolic_tensors((x,)):
        return Nonzero().symbolic_call(x)
    return backend.numpy.nonzero(x)


class NotEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.not_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.not_equal", "keras.ops.numpy.not_equal"])
def not_equal(x1, x2):
    """Return `(x1 != x2)` element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise comparsion of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return NotEqual().symbolic_call(x1, x2)
    return backend.numpy.not_equal(x1, x2)


class OnesLike(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.ones_like(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(["keras.ops.ones_like", "keras.ops.numpy.ones_like"])
def ones_like(x, dtype=None):
    """Return a tensor of ones with the same shape and type of `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of ones with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return OnesLike().symbolic_call(x, dtype=dtype)
    return backend.numpy.ones_like(x, dtype=dtype)


class ZerosLike(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.zeros_like(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return KerasTensor(x.shape, dtype=dtype)


@keras_export(
    [
        "keras.ops.zeros_like",
        "keras.ops.numpy.zeros_like",
    ]
)
def zeros_like(x, dtype=None):
    """Return a tensor of zeros with the same shape and type as `x`.

    Args:
        x: Input tensor.
        dtype: Overrides the data type of the result.

    Returns:
        A tensor of zeros with the same shape and type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return ZerosLike().symbolic_call(x, dtype=dtype)
    return backend.numpy.zeros_like(x, dtype=dtype)


class Outer(Operation):
    def call(self, x1, x2):
        return backend.numpy.outer(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [1])
        x2_shape = getattr(x2, "shape", [1])
        if None in x1_shape:
            x1_flatten_shape = None
        else:
            x1_flatten_shape = int(np.prod(x1_shape))
        if None in x2_shape:
            x2_flatten_shape = None
        else:
            x2_flatten_shape = int(np.prod(x2_shape))
        output_shape = [x1_flatten_shape, x2_flatten_shape]
        output_dtype = backend.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.outer", "keras.ops.numpy.outer"])
def outer(x1, x2):
    """Compute the outer product of two vectors.

    Given two vectors `x1` and `x2`, the outer product is:

    ```
    out[i, j] = x1[i] * x2[j]
    ```

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Outer product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Outer().symbolic_call(x1, x2)
    return backend.numpy.outer(x1, x2)


class Pad(Operation):
    def __init__(self, pad_width, mode="constant"):
        super().__init__()
        self.pad_width = self._process_pad_width(pad_width)
        self.mode = mode

    def _process_pad_width(self, pad_width):
        if isinstance(pad_width, int):
            return ((pad_width, pad_width),)
        if isinstance(pad_width, (tuple, list)) and isinstance(
            pad_width[0], int
        ):
            return (pad_width,)
        first_len = len(pad_width[0])
        for i, pw in enumerate(pad_width):
            if len(pw) != first_len:
                raise ValueError(
                    "`pad_width` should be a list of tuples of length "
                    f"1 or 2. Received: pad_width={pad_width}"
                )
            if len(pw) == 1:
                pad_width[i] = (pw[0], pw[0])
        return pad_width

    def call(self, x, constant_values=None):
        if len(self.pad_width) > 1 and len(self.pad_width) != len(x.shape):
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`. "
                f"Received: pad_width={self.pad_width} "
                f"(of length {len(self.pad_width)}) and x.shape={x.shape} "
                f"(of length {len(x.shape)})"
            )
        return backend.numpy.pad(
            x,
            pad_width=self.pad_width,
            mode=self.mode,
            constant_values=constant_values,
        )

    def compute_output_spec(self, x, constant_values=None):
        output_shape = list(x.shape)
        if len(self.pad_width) == 1:
            pad_width = [self.pad_width[0] for _ in range(len(output_shape))]
        elif len(self.pad_width) == len(output_shape):
            pad_width = self.pad_width
        else:
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`. "
                f"Received: pad_width={self.pad_width} "
                f"(of length {len(self.pad_width)}) and x.shape={x.shape} "
                f"(of length {len(x.shape)})"
            )

        for i in range(len(output_shape)):
            if output_shape[i] is None:
                output_shape[i] = None
            else:
                output_shape[i] += pad_width[i][0] + pad_width[i][1]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.pad", "keras.ops.numpy.pad"])
def pad(x, pad_width, mode="constant", constant_values=None):
    """Pad a tensor.

    Args:
        x: Tensor to pad.
        pad_width: Number of values padded to the edges of each axis.
            `((before_1, after_1), ...(before_N, after_N))` unique pad
            widths for each axis.
            `((before, after),)` yields same before and after pad for
            each axis.
            `(pad,)` or `int` is a shortcut for `before = after = pad`
            width for all axes.
        mode: One of `"constant"`, `"edge"`, `"linear_ramp"`,
            `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
            `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
            `"circular"`. Defaults to `"constant"`.
        constant_values: value to pad with if `mode == "constant"`.
            Defaults to `0`. A `ValueError` is raised if not None and
            `mode != "constant"`.

    Note:
        Torch backend only supports modes `"constant"`, `"reflect"`,
        `"symmetric"` and `"circular"`.
        Only Torch backend supports `"circular"` mode.

    Note:
        Tensorflow backend only supports modes `"constant"`, `"reflect"`
        and `"symmetric"`.

    Returns:
        Padded tensor.
    """
    return Pad(pad_width, mode=mode)(x, constant_values=constant_values)


class Prod(Operation):
    def __init__(self, axis=None, keepdims=False, dtype=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.prod(
            x,
            axis=self.axis,
            keepdims=self.keepdims,
            dtype=self.dtype,
        )

    def compute_output_spec(self, x):
        if self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = backend.result_type(x.dtype)
            if dtype == "bool":
                dtype = "int32"
            elif dtype in ("int8", "int16"):
                dtype = "int32"
            elif dtype in ("uint8", "uint16"):
                dtype = "uint32"
        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
        )


@keras_export(["keras.ops.prod", "keras.ops.numpy.prod"])
def prod(x, axis=None, keepdims=False, dtype=None):
    """Return the product of tensor elements over a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which a product is performed. The default,
            `axis=None`, will compute the product of all elements
            in the input tensor.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.
        dtype: Data type of the returned tensor.

    Returns:
        Product of elements of `x` over the given axis or axes.
    """
    if any_symbolic_tensors((x,)):
        return Prod(axis=axis, keepdims=keepdims, dtype=dtype).symbolic_call(x)
    return backend.numpy.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


class Quantile(Operation):
    def __init__(self, axis=None, method="linear", keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.method = method
        self.keepdims = keepdims

    def call(self, x, q):
        return backend.numpy.quantile(
            x, q, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x, q):
        output_shape = reduce_shape(
            x.shape, axis=self.axis, keepdims=self.keepdims
        )
        if hasattr(q, "shape"):
            if len(q.shape) > 0:
                output_shape = (q.shape[0],) + output_shape
        if backend.standardize_dtype(x.dtype) == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.quantile", "keras.ops.numpy.quantile"])
def quantile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th quantile(s) of the data along the specified axis.

    Args:
        x: Input tensor.
        q: Probability or sequence of probabilities for the quantiles to
            compute. Values must be between 0 and 1 inclusive.
        axis: Axis or axes along which the quantiles are computed. Defaults to
            `axis=None` which is to compute the quantile(s) along a flattened
            version of the array.
        method: A string specifies the method to use for estimating the
            quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,
            `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
            If the desired quantile lies between two data points `i < j`:
            - `"linear"`: `i + (j - i) * fraction`, where fraction is the
                fractional part of the index surrounded by `i` and `j`.
            - `"lower"`: `i`.
            - `"higher"`: `j`.
            - `"midpoint"`: `(i + j) / 2`
            - `"nearest"`: `i` or `j`, whichever is nearest.
        keepdims: If this is set to `True`, the axes which are reduce
            are left in the result as dimensions with size one.

    Returns:
        The quantile(s). If `q` is a single probability and `axis=None`, then
        the result is a scalar. If multiple probabilies levels are given, first
        axis of the result corresponds to the quantiles. The other axes are the
        axes that remain after the reduction of `x`.
    """
    if any_symbolic_tensors((x, q)):
        return Quantile(
            axis=axis, method=method, keepdims=keepdims
        ).symbolic_call(x, q)
    return backend.numpy.quantile(
        x, q, axis=axis, method=method, keepdims=keepdims
    )


class Ravel(Operation):
    def call(self, x):
        return backend.numpy.ravel(x)

    def compute_output_spec(self, x):
        if None in x.shape:
            output_shape = [
                None,
            ]
        else:
            output_shape = [int(np.prod(x.shape))]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.ravel", "keras.ops.numpy.ravel"])
def ravel(x):
    """Return a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Ravel().symbolic_call(x)
    return backend.numpy.ravel(x)


class Real(Operation):
    def call(self, x):
        return backend.numpy.real(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.real", "keras.ops.numpy.real"])
def real(x):
    """Return the real part of the complex argument.

    Args:
        x: Input tensor.

    Returns:
        The real component of the complex argument.
    """
    if any_symbolic_tensors((x,)):
        return Real().symbolic_call(x)
    return backend.numpy.real(x)


class Reciprocal(Operation):
    def call(self, x):
        return backend.numpy.reciprocal(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


@keras_export(
    [
        "keras.ops.reciprocal",
        "keras.ops.numpy.reciprocal",
    ]
)
def reciprocal(x):
    """Return the reciprocal of the argument, element-wise.

    Calculates `1/x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, element-wise reciprocal of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Reciprocal().symbolic_call(x)
    return backend.numpy.reciprocal(x)


class Repeat(Operation):
    def __init__(self, repeats, axis=None):
        super().__init__()
        self.axis = axis
        self.repeats = repeats

    def call(self, x):
        return backend.numpy.repeat(x, self.repeats, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        repeats = self.repeats
        if isinstance(repeats, int):
            repeats = [repeats]
        repeats_size = len(repeats)
        broadcast = repeats_size == 1

        if self.axis is None:
            if None in x_shape:
                return KerasTensor([None], dtype=x.dtype)

            x_flatten_size = int(np.prod(x_shape))
            if broadcast:
                output_shape = [x_flatten_size * repeats[0]]
            elif repeats_size != x_flatten_size:
                raise ValueError(
                    "Size of `repeats` and "
                    "dimensions of `x` after flattening should be compatible. "
                    f"Received: {repeats_size} and {x_flatten_size}"
                )
            else:
                output_shape = [int(np.sum(repeats))]
            return KerasTensor(output_shape, dtype=x.dtype)

        size_on_ax = x_shape[self.axis]
        if size_on_ax is None:
            return KerasTensor(x_shape, dtype=x.dtype)

        output_shape = x_shape
        if broadcast:
            output_shape[self.axis] = size_on_ax * repeats[0]
        elif size_on_ax != repeats_size:
            raise ValueError(
                "Size of `repeats` and "
                f"dimensions of `axis {self.axis} of x` should be compatible. "
                f"Received: {repeats_size} and {x_shape}"
            )
        else:
            output_shape[self.axis] = int(np.sum(repeats))
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.repeat", "keras.ops.numpy.repeat"])
def repeat(x, repeats, axis=None):
    """Repeat each element of a tensor after themselves.

    Args:
        x: Input tensor.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values. By default, use
            the flattened input array, and return a flat output array.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Repeat(repeats, axis=axis).symbolic_call(x)
    return backend.numpy.repeat(x, repeats, axis=axis)


class Reshape(Operation):
    def __init__(self, newshape):
        super().__init__()
        self.newshape = newshape

    def call(self, x):
        return backend.numpy.reshape(x, self.newshape)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_reshape_output_shape(
            x.shape, self.newshape, "newshape"
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.reshape", "keras.ops.numpy.reshape"])
def reshape(x, newshape):
    """Gives a new shape to a tensor without changing its data.

    Args:
        x: Input tensor.
        newshape: The new shape should be compatible with the original shape.
            One shape dimension can be -1 in which case the value is
            inferred from the length of the array and remaining dimensions.

    Returns:
        The reshaped tensor.
    """
    if any_symbolic_tensors((x,)):
        return Reshape(newshape).symbolic_call(x)
    return backend.numpy.reshape(x, newshape)


class Roll(Operation):
    def __init__(self, shift, axis=None):
        super().__init__()
        self.shift = shift
        self.axis = axis

    def call(self, x):
        return backend.numpy.roll(x, self.shift, self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.roll", "keras.ops.numpy.roll"])
def roll(x, shift, axis=None):
    """Roll tensor elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        x: Input tensor.
        shift: The number of places by which elements are shifted.
        axis: The axis along which elements are shifted. By default, the
            array is flattened before shifting, after which the original
            shape is restored.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Roll(shift, axis=axis).symbolic_call(x)
    return backend.numpy.roll(x, shift, axis=axis)


class Round(Operation):
    def __init__(self, decimals=0):
        super().__init__()
        self.decimals = decimals

    def call(self, x):
        return backend.numpy.round(x, self.decimals)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.round", "keras.ops.numpy.round"])
def round(x, decimals=0):
    """Evenly round to the given number of decimals.

    Args:
        x: Input tensor.
        decimals: Number of decimal places to round to. Defaults to `0`.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Round(decimals).symbolic_call(x)
    return backend.numpy.round(x, decimals)


class SearchSorted(Operation):
    def call(self, sorted_sequence, values, side="left"):
        sorted_sequence = backend.convert_to_tensor(sorted_sequence)
        values = backend.convert_to_tensor(values)
        return backend.numpy.searchsorted(sorted_sequence, values, side=side)

    def compute_output_spec(self, sorted_sequence, values, side="left"):
        if len(sorted_sequence.shape) != 1:
            raise ValueError(
                "searchsorted only supports 1-D sorted sequences. Use"
                "keras.ops.vectorized_map to extend to N-D sequences."
            )
        out_type = (
            "int32"
            if sorted_sequence.shape[0] <= np.iinfo(np.int32).max
            else "int64"
        )
        return KerasTensor(values.shape, dtype=out_type)


@keras_export(["keras.ops.searchsorted"])
def searchsorted(sorted_sequence, values, side="left"):
    """Perform a binary search, returning indices for insertion of `values`
    into `sorted_sequence` that maintain the sorting order.

    Args:
        sorted_sequence: 1-D input tensor, sorted along the innermost
            dimension.
        values: N-D tensor of query insertion values.
        side: 'left' or 'right', specifying the direction in which to insert
            for the equality case (tie-breaker).

    Returns:
        Tensor of insertion indices of same shape as `values`.
    """
    if any_symbolic_tensors((sorted_sequence, values)):
        return SearchSorted().symbolic_call(sorted_sequence, values, side=side)

    sorted_sequence = backend.convert_to_tensor(sorted_sequence)
    values = backend.convert_to_tensor(values)
    return backend.numpy.searchsorted(sorted_sequence, values, side=side)


class Sign(Operation):
    def call(self, x):
        return backend.numpy.sign(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.sign", "keras.ops.numpy.sign"])
def sign(x):
    """Returns a tensor with the signs of the elements of `x`.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sign().symbolic_call(x)
    return backend.numpy.sign(x)


class Sin(Operation):
    def call(self, x):
        return backend.numpy.sin(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sin", "keras.ops.numpy.sin"])
def sin(x):
    """Trigonometric sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sin().symbolic_call(x)
    return backend.numpy.sin(x)


class Sinh(Operation):
    def call(self, x):
        return backend.numpy.sinh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sinh", "keras.ops.numpy.sinh"])
def sinh(x):
    """Hyperbolic sine, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sinh().symbolic_call(x)
    return backend.numpy.sinh(x)


class Size(Operation):
    def call(self, x):
        return backend.numpy.size(x)

    def compute_output_spec(self, x):
        return KerasTensor([], dtype="int32")


@keras_export(["keras.ops.size", "keras.ops.numpy.size"])
def size(x):
    """Return the number of elements in a tensor.

    Args:
        x: Input tensor.

    Returns:
        Number of elements in `x`.
    """
    if any_symbolic_tensors((x,)):
        return Size().symbolic_call(x)
    return backend.numpy.size(x)


class Sort(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.sort(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


@keras_export(["keras.ops.sort", "keras.ops.numpy.sort"])
def sort(x, axis=-1):
    """Sorts the elements of `x` along a given axis in ascending order.

    Args:
        x: Input tensor.
        axis: Axis along which to sort. If `None`, the tensor is flattened
            before sorting. Defaults to `-1`; the last axis.

    Returns:
        Sorted tensor.
    """
    if any_symbolic_tensors((x,)):
        return Sort(axis=axis).symbolic_call(x)
    return backend.numpy.sort(x, axis=axis)


class Split(Operation):
    def __init__(self, indices_or_sections, axis=0):
        super().__init__()
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def call(self, x):
        return backend.numpy.split(x, self.indices_or_sections, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        x_size_on_axis = x_shape[self.axis]
        if isinstance(self.indices_or_sections, int):
            if x_size_on_axis is None:
                x_shape[self.axis] = None
                return [
                    KerasTensor(x_shape, dtype=x.dtype)
                    for _ in range(self.indices_or_sections)
                ]
            if np.mod(x_size_on_axis, self.indices_or_sections) != 0:
                raise ValueError(
                    "`x` size on given `axis` must be dividible by "
                    "`indices_or_sections` when `indices_or_sections` is an "
                    f"int. But received {x_size_on_axis} and "
                    f"{self.indices_or_sections}."
                )
            size = x_size_on_axis // self.indices_or_sections
            x_shape[self.axis] = size
            return [
                KerasTensor(x_shape, dtype=x.dtype)
                for _ in range(self.indices_or_sections)
            ]

        indices_or_sections = (0, *self.indices_or_sections, x_size_on_axis)
        output_size = np.diff(indices_or_sections)
        outputs = []
        for i in range(len(output_size)):
            output_shape = list(x_shape)
            output_shape[self.axis] = int(output_size[i])
            outputs.append(KerasTensor(output_shape, dtype=x.dtype))
        return outputs


@keras_export(["keras.ops.split", "keras.ops.numpy.split"])
def split(x, indices_or_sections, axis=0):
    """Split a tensor into chunks.

    Args:
        x: Input tensor.
        indices_or_sections: If an integer, N, the tensor will be split into N
            equal sections along `axis`. If a 1-D array of sorted integers,
            the entries indicate indices at which the tensor will be split
            along `axis`.
        axis: Axis along which to split. Defaults to `0`.

    Note:
        A split does not have to result in equal division when using
        Torch backend.

    Returns:
        A list of tensors.
    """
    if any_symbolic_tensors((x,)):
        return Split(indices_or_sections, axis=axis).symbolic_call(x)
    return backend.numpy.split(x, indices_or_sections, axis=axis)


class Stack(Operation):
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def call(self, xs):
        return backend.numpy.stack(xs, axis=self.axis)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape. But found "
                    f"element of shape {x.shape},  which is different from the "
                    f"first element's shape {first_shape}."
                )
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))

        size_on_axis = len(xs)
        output_shape = list(first_shape)
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)
        output_dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.stack", "keras.ops.numpy.stack"])
def stack(x, axis=0):
    """Join a sequence of tensors along a new axis.

    The `axis` parameter specifies the index of the new axis in the
    dimensions of the result.

    Args:
        x: A sequence of tensors.
        axis: Axis along which to stack. Defaults to `0`.

    Returns:
        The stacked tensor.
    """
    if any_symbolic_tensors((x,)):
        return Stack(axis=axis).symbolic_call(x)
    return backend.numpy.stack(x, axis=axis)


class Std(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.std(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = backend.standardize_dtype(x.dtype)
        if "int" in output_dtype or output_dtype == "bool":
            output_dtype = backend.floatx()
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.std", "keras.ops.numpy.std"])
def std(x, axis=None, keepdims=False):
    """Compute the standard deviation along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis along which to compute standard deviation.
            Default is to compute the standard deviation of the
            flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the standard deviation values.
    """
    if any_symbolic_tensors((x,)):
        return Std(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.std(x, axis=axis, keepdims=keepdims)


class Swapaxes(Operation):
    def __init__(self, axis1, axis2):
        super().__init__()

        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.swapaxes(x, self.axis1, self.axis2)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        tmp = x_shape[self.axis1]
        x_shape[self.axis1] = x_shape[self.axis2]
        x_shape[self.axis2] = tmp
        return KerasTensor(x_shape, dtype=x.dtype)


@keras_export(["keras.ops.swapaxes", "keras.ops.numpy.swapaxes"])
def swapaxes(x, axis1, axis2):
    """Interchange two axes of a tensor.

    Args:
        x: Input tensor.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        A tensor with the axes swapped.
    """
    if any_symbolic_tensors((x,)):
        return Swapaxes(axis1, axis2).symbolic_call(x)
    return backend.numpy.swapaxes(x, axis1=axis1, axis2=axis2)


class Take(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        if isinstance(indices, KerasTensor):
            indices_shape = list(indices.shape)
        else:
            indices_shape = list(getattr(np.array(indices), "shape", []))
        if self.axis is None:
            return KerasTensor(indices_shape, dtype=x.dtype)

        # make sure axis is non-negative
        axis = len(x_shape) + self.axis if self.axis < 0 else self.axis
        output_shape = x_shape[:axis] + indices_shape + x_shape[axis + 1 :]
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.take", "keras.ops.numpy.take"])
def take(x, indices, axis=None):
    """Take elements from a tensor along an axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the
            flattened input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return Take(axis=axis).symbolic_call(x, indices)
    return backend.numpy.take(x, indices, axis=axis)


class TakeAlongAxis(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take_along_axis(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        output_shape = operation_utils.compute_take_along_axis_output_shape(
            x.shape, indices.shape, self.axis
        )
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(
    [
        "keras.ops.take_along_axis",
        "keras.ops.numpy.take_along_axis",
    ]
)
def take_along_axis(x, indices, axis=None):
    """Select values from `x` at the 1-D `indices` along the given axis.

    Args:
        x: Source tensor.
        indices: The indices of the values to extract.
        axis: The axis over which to select values. By default, the flattened
            input tensor is used.

    Returns:
        The corresponding tensor of values.
    """
    if any_symbolic_tensors((x, indices)):
        return TakeAlongAxis(axis=axis).symbolic_call(x, indices)
    return backend.numpy.take_along_axis(x, indices, axis=axis)


class Tan(Operation):
    def call(self, x):
        return backend.numpy.tan(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.tan", "keras.ops.numpy.tan"])
def tan(x):
    """Compute tangent, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tan().symbolic_call(x)
    return backend.numpy.tan(x)


class Tanh(Operation):
    def call(self, x):
        return backend.numpy.tanh(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, "dtype", backend.floatx()))
        if dtype == "int64":
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.tanh", "keras.ops.numpy.tanh"])
def tanh(x):
    """Hyperbolic tangent, element-wise.

    Arguments:
        x: Input tensor.

    Returns:
        Output tensor of same shape as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tanh().symbolic_call(x)
    return backend.numpy.tanh(x)


class Tensordot(Operation):
    def __init__(self, axes=2):
        super().__init__()
        self.axes = axes

    def call(self, x1, x2):
        return backend.numpy.tensordot(x1, x2, axes=self.axes)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if not isinstance(self.axes, int):
            x1_select_shape = [x1_shape[ax] for ax in self.axes[0]]
            x2_select_shape = [x2_shape[ax] for ax in self.axes[1]]
            if not shape_equal(
                x1_select_shape, x2_select_shape, allow_none=True
            ):
                raise ValueError(
                    "Shape mismatch on `x1[axes[0]]` and `x2[axes[1]]`, "
                    f"received {x1_select_shape} and {x2_select_shape}."
                )

            for ax in self.axes[0]:
                x1_shape[ax] = -1
            for ax in self.axes[1]:
                x2_shape[ax] = -1

            x1_shape = list(filter((-1).__ne__, x1_shape))
            x2_shape = list(filter((-1).__ne__, x2_shape))

            output_shape = x1_shape + x2_shape
            return KerasTensor(output_shape, dtype=dtype)

        if self.axes <= 0:
            output_shape = x1_shape + x2_shape
        else:
            output_shape = x1_shape[: -self.axes] + x2_shape[self.axes :]

        return KerasTensor(output_shape, dtype=dtype)


@keras_export(["keras.ops.tensordot", "keras.ops.numpy.tensordot"])
def tensordot(x1, x2, axes=2):
    """Compute the tensor dot product along specified axes.

    Args:
        x1: First tensor.
        x2: Second tensor.
        axes: - If an integer, N, sum over the last N axes of `x1` and the
                first N axes of `x2` in order. The sizes of the corresponding
                axes must match.
              - Or, a list of axes to be summed over, first sequence applying
                to `x1`, second to `x2`. Both sequences must be of the
                same length.

    Returns:
        The tensor dot product of the inputs.
    """
    if any_symbolic_tensors((x1, x2)):
        return Tensordot(axes=axes).symbolic_call(x1, x2)
    return backend.numpy.tensordot(x1, x2, axes=axes)


class Tile(Operation):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def call(self, x):
        return backend.numpy.tile(x, self.repeats)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        repeats = self.repeats
        if isinstance(repeats, int):
            repeats = [repeats]
        if len(x_shape) > len(repeats):
            repeats = [1] * (len(x_shape) - len(repeats)) + repeats
        else:
            x_shape = [1] * (len(repeats) - len(x_shape)) + x_shape

        output_shape = []
        for x_size, repeat in zip(x_shape, repeats):
            if x_size is None:
                output_shape.append(None)
            else:
                output_shape.append(x_size * repeat)
        return KerasTensor(output_shape, dtype=x.dtype)


@keras_export(["keras.ops.tile", "keras.ops.numpy.tile"])
def tile(x, repeats):
    """Repeat `x` the number of times given by `repeats`.

    If `repeats` has length `d`, the result will have dimension of
    `max(d, x.ndim)`.

    If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending
    new axes.

    If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.

    Args:
        x: Input tensor.
        repeats: The number of repetitions of `x` along each axis.

    Returns:
        The tiled output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Tile(
            repeats,
        ).symbolic_call(x)
    return backend.numpy.tile(x, repeats)


class Trace(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.trace(
            x, offset=self.offset, axis1=self.axis1, axis2=self.axis2
        )

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = list(filter((-1).__ne__, x_shape))
        output_dtype = backend.standardize_dtype(x.dtype)
        if output_dtype not in ("int64", "uint32", "uint64"):
            output_dtype = dtypes.result_type(output_dtype, "int32")
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.trace", "keras.ops.numpy.trace"])
def trace(x, offset=0, axis1=0, axis2=1):
    """Return the sum along diagonals of the tensor.

    If `x` is 2-D, the sum along its diagonal with the given offset is
    returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.

    If a has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-arrays whose traces are
    returned.

    The shape of the resulting tensor is the same as that of `x` with `axis1`
    and `axis2` removed.

    Args:
        x: Input tensor.
        offset: Offset of the diagonal from the main diagonal. Can be
            both positive and negative. Defaults to `0`.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
            Defaults to `0`.(first axis).
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.
            Defaults to `1` (second axis).

    Returns:
        If `x` is 2-D, the sum of the diagonal is returned. If `x` has
        larger dimensions, then a tensor of sums along diagonals is
        returned.
    """
    if any_symbolic_tensors((x,)):
        return Trace(offset, axis1, axis2).symbolic_call(x)
    return backend.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


class Tri(Operation):
    def __init__(self, k=0, dtype=None):
        super().__init__()
        self.k = k
        self.dtype = dtype or backend.floatx()

    def call(self, N, M=None):
        return backend.numpy.tri(N=N, M=M, k=self.k, dtype=self.dtype)

    def compute_output_spec(self, N, M=None):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=self.dtype)


@keras_export(["keras.ops.tri", "keras.ops.numpy.tri"])
def tri(N, M=None, k=0, dtype=None):
    """Return a tensor with ones at and below a diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the tensor.
        M: Number of columns in the tensor.
        k: The sub-diagonal at and below which the array is filled.
            `k = 0` is the main diagonal, while `k < 0` is below it, and
            `k > 0` is above. The default is 0.
        dtype: Data type of the returned tensor. The default is "float32".

    Returns:
        Tensor with its lower triangle filled with ones and zeros elsewhere.
        `T[i, j] == 1` for `j <= i + k`, 0 otherwise.
    """
    return backend.numpy.tri(N, M=M, k=k, dtype=dtype)


class Tril(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.tril(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.tril", "keras.ops.numpy.tril"])
def tril(x, k=0):
    """Return lower triangle of a tensor.

    For tensors with `ndim` exceeding 2, `tril` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal above which to zero elements. Defaults to `0`. the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Lower triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Tril(k=k).symbolic_call(x)
    return backend.numpy.tril(x, k=k)


class Triu(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.triu(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.triu", "keras.ops.numpy.triu"])
def triu(x, k=0):
    """Return upper triangle of a tensor.

    For tensors with `ndim` exceeding 2, `triu` will apply to the
    final two axes.

    Args:
        x: Input tensor.
        k: Diagonal below which to zero elements. Defaults to `0`. the
            main diagonal. `k < 0` is below it, and `k > 0` is above it.

    Returns:
        Upper triangle of `x`, of same shape and data type as `x`.
    """
    if any_symbolic_tensors((x,)):
        return Triu(k=k).symbolic_call(x)
    return backend.numpy.triu(x, k=k)


class Trunc(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return backend.numpy.trunc(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export(["keras.ops.trunc", "keras.ops.numpy.trunc"])
def trunc(x):
    """Return the truncated value of the input, element-wise.

    The truncated value of the scalar `x` is the nearest integer `i` which is
    closer to zero than `x` is. In short, the fractional part of the signed
    number `x` is discarded.

    Args:
        x: Input tensor.

    Returns:
        The truncated value of each element in `x`.

    Example:
    >>> x = ops.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> ops.trunc(x)
    array([-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0])
    """
    if any_symbolic_tensors((x,)):
        return Trunc().symbolic_call(x)
    return backend.numpy.trunc(x)


class Vdot(Operation):
    def call(self, x1, x2):
        return backend.numpy.vdot(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor([], dtype=dtype)


@keras_export(["keras.ops.vdot", "keras.ops.numpy.vdot"])
def vdot(x1, x2):
    """Return the dot product of two vectors.

    If the first argument is complex, the complex conjugate of the first
    argument is used for the calculation of the dot product.

    Multidimensional tensors are flattened before the dot product is taken.

    Args:
        x1: First input tensor. If complex, its complex conjugate is taken
            before calculation of the dot product.
        x2: Second input tensor.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return Vdot().symbolic_call(x1, x2)
    return backend.numpy.vdot(x1, x2)


@keras_export(["keras.ops.vectorize", "keras.ops.numpy.vectorize"])
def vectorize(pyfunc, *, excluded=None, signature=None):
    """Turn a function into a vectorized function.

    Example:

    ```python
    def myfunc(a, b):
        return a + b

    vfunc = keras.ops.vectorize(myfunc)
    y = vfunc([1, 2, 3, 4], 2)  # Returns Tensor([3, 4, 5, 6])
    ```

    Args:
        pyfunc: Callable of a single tensor argument.
        excluded: Optional set of integers representing
            positional arguments for which the function
            will not be vectorized.
            These will be passed directly to `pyfunc` unmodified.
        signature: Optional generalized universal function signature,
            e.g., `"(m,n),(n)->(m)"` for vectorized
            matrix-vector multiplication. If provided,
            `pyfunc` will be called with (and expected to return)
            arrays with shapes given by the size of corresponding
            core dimensions. By default, `pyfunc` is assumed
            to take scalars tensors as input and output.

    Returns:
        A new function that applies `pyfunc` to every element
        of its input along axis 0 (the batch axis).
    """
    if not callable(pyfunc):
        raise ValueError(
            "Expected argument `pyfunc` to be a callable. "
            f"Received: pyfunc={pyfunc}"
        )
    return backend.numpy.vectorize(
        pyfunc, excluded=excluded, signature=signature
    )


class Vstack(Operation):
    def call(self, xs):
        return backend.numpy.vstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[0], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape except on "
                    f"the `axis` dim. But found element of shape {x.shape}, "
                    f"which is different from the first element's "
                    f"shape {first_shape}."
                )
            if total_size_on_axis is None or x.shape[0] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[0]
            dtypes_to_resolve.append(getattr(x, "dtype", type(x)))
        output_shape = list(first_shape)
        output_shape[0] = total_size_on_axis
        output_dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, output_dtype)


@keras_export(["keras.ops.vstack", "keras.ops.numpy.vstack"])
def vstack(xs):
    """Stack tensors in sequence vertically (row wise).

    Args:
        xs: Sequence of tensors.

    Returns:
        Tensor formed by stacking the given tensors.
    """
    if any_symbolic_tensors((xs,)):
        return Vstack().symbolic_call(xs)
    return backend.numpy.vstack(xs)


class Where(Operation):
    def call(self, condition, x1=None, x2=None):
        return backend.numpy.where(condition, x1, x2)

    def compute_output_spec(self, condition, x1, x2):
        condition_shape = getattr(condition, "shape", [])
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(condition_shape, x1_shape)
        output_shape = broadcast_shapes(output_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1) if x1 is not None else "int"),
            getattr(x2, "dtype", type(x2) if x2 is not None else "int"),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.where", "keras.ops.numpy.where"])
def where(condition, x1=None, x2=None):
    """Return elements chosen from `x1` or `x2` depending on `condition`.

    Args:
        condition: Where `True`, yield `x1`, otherwise yield `x2`.
        x1: Values from which to choose when `condition` is `True`.
        x2: Values from which to choose when `condition` is `False`.

    Returns:
        A tensor with elements from `x1` where `condition` is `True`, and
        elements from `x2` where `condition` is `False`.
    """
    if (x1 is None and x2 is not None) or (x1 is not None and x2 is None):
        raise ValueError(
            "`x1` and `x2` either both should be `None`"
            " or both should have non-None value."
        )
    if any_symbolic_tensors((condition, x1, x2)):
        return Where().symbolic_call(condition, x1, x2)
    return backend.numpy.where(condition, x1, x2)


class Subtract(Operation):
    def call(self, x1, x2):
        return backend.numpy.subtract(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and x2_sparse
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.subtract", "keras.ops.numpy.subtract"])
def subtract(x1, x2):
    """Subtract arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise difference of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.numpy.subtract(x1, x2)


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.numpy.multiply(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_sparse = getattr(x1, "sparse", True)
        x2_sparse = getattr(x2, "sparse", True)
        output_sparse = x1_sparse or x2_sparse
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=dtype, sparse=output_sparse)


@keras_export(["keras.ops.multiply", "keras.ops.numpy.multiply"])
def multiply(x1, x2):
    """Multiply arguments element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, element-wise product of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.numpy.multiply(x1, x2)


class Divide(Operation):
    def call(self, x1, x2):
        return backend.numpy.divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.divide", "keras.ops.numpy.divide"])
def divide(x1, x2):
    """Divide arguments element-wise.

    `keras.ops.true_divide` is an alias for this function.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output tensor, the quotient `x1/x2`, element-wise.
    """
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.numpy.divide(x1, x2)


class DivideNoNan(Operation):
    def call(self, x1, x2):
        return backend.numpy.divide_no_nan(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(["keras.ops.divide_no_nan", "keras.ops.numpy.divide_no_nan"])
def divide_no_nan(x1, x2):
    """Safe element-wise division which returns 0 where the denominator is 0.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        The quotient `x1/x2`, element-wise, with zero where x2 is zero.
    """
    if any_symbolic_tensors((x1, x2)):
        return DivideNoNan().symbolic_call(x1, x2)
    return backend.numpy.divide_no_nan(x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.numpy.true_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
            float,
        )
        x1_sparse = getattr(x1, "sparse", False)
        x2_sparse = getattr(x2, "sparse", False)
        output_sparse = x1_sparse and not x2_sparse
        return KerasTensor(
            output_shape, dtype=output_dtype, sparse=output_sparse
        )


@keras_export(
    [
        "keras.ops.true_divide",
        "keras.ops.numpy.true_divide",
    ]
)
def true_divide(x1, x2):
    """Alias for `keras.ops.divide`."""
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.numpy.true_divide(x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.numpy.power(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.power", "keras.ops.numpy.power"])
def power(x1, x2):
    """First tensor elements raised to powers from second tensor, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        Output tensor, the bases in `x1` raised to the exponents in `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.numpy.power(x1, x2)


class Negative(Operation):
    def call(self, x):
        return backend.numpy.negative(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.negative", "keras.ops.numpy.negative"])
def negative(x):
    """Numerical negative, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, `y = -x`.
    """
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.numpy.negative(x)


class Square(Operation):
    def call(self, x):
        return backend.numpy.square(x)

    def compute_output_spec(self, x):
        sparse = getattr(x, "sparse", False)
        dtype = backend.standardize_dtype(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.square", "keras.ops.numpy.square"])
def square(x):
    """Return the element-wise square of the input.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the square of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.numpy.square(x)


class Sqrt(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.numpy.sqrt(x)

    def compute_output_spec(self, x):
        dtype = (
            backend.floatx()
            if backend.standardize_dtype(x.dtype) == "int64"
            else dtypes.result_type(x.dtype, float)
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)


@keras_export(["keras.ops.sqrt", "keras.ops.numpy.sqrt"])
def sqrt(x):
    """Return the non-negative square root of a tensor, element-wise.

    Args:
        x: Input tensor.

    Returns:
        Output tensor, the non-negative square root of `x`.
    """
    if any_symbolic_tensors((x,)):
        return Sqrt().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.numpy.sqrt(x)


class Squeeze(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.squeeze(x, axis=self.axis)

    def compute_output_spec(self, x):
        input_shape = list(x.shape)
        sparse = getattr(x, "sparse", False)
        axis = to_tuple_or_list(self.axis)
        if axis is None:
            output_shape = list(filter((1).__ne__, input_shape))
            return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)
        else:
            for a in axis:
                if input_shape[a] != 1:
                    raise ValueError(
                        f"Cannot squeeze axis {a}, because the dimension "
                        "is not 1."
                    )
            axis = [canonicalize_axis(a, len(input_shape)) for a in axis]
            for a in sorted(axis, reverse=True):
                del input_shape[a]
            return KerasTensor(input_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.squeeze", "keras.ops.numpy.squeeze"])
def squeeze(x, axis=None):
    """Remove axes of length one from `x`.

    Args:
        x: Input tensor.
        axis: Select a subset of the entries of length one in the shape.

    Returns:
        The input tensor with all or a subset of the dimensions of
        length 1 removed.
    """
    if any_symbolic_tensors((x,)):
        return Squeeze(axis=axis).symbolic_call(x)
    return backend.numpy.squeeze(x, axis=axis)


class Transpose(Operation):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes

    def call(self, x):
        return backend.numpy.transpose(x, axes=self.axes)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_transpose_output_shape(
            x.shape, self.axes
        )
        sparse = getattr(x, "sparse", False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)


@keras_export(["keras.ops.transpose", "keras.ops.numpy.transpose"])
def transpose(x, axes=None):
    """Returns a tensor with `axes` transposed.

    Args:
        x: Input tensor.
        axes: Sequence of integers. Permutation of the dimensions of `x`.
            By default, the order of the axes are reversed.

    Returns:
        `x` with its axes permuted.
    """
    if any_symbolic_tensors((x,)):
        return Transpose(axes=axes).symbolic_call(x)
    return backend.numpy.transpose(x, axes=axes)


class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.mean(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        ori_dtype = backend.standardize_dtype(x.dtype)
        compute_dtype = dtypes.result_type(x.dtype, "float32")
        if "int" in ori_dtype or ori_dtype == "bool":
            result_dtype = compute_dtype
        else:
            result_dtype = ori_dtype
        sparse = getattr(x, "sparse", False)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=result_dtype,
            sparse=sparse,
        )


@keras_export(["keras.ops.mean", "keras.ops.numpy.mean"])
def mean(x, axis=None, keepdims=False):
    """Compute the arithmetic mean along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the means are computed. The default
            is to compute the mean of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the mean values.
    """
    if any_symbolic_tensors((x,)):
        return Mean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.mean(x, axis=axis, keepdims=keepdims)


class Var(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.var(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_dtype = backend.result_type(getattr(x, "dtype", type(x)), float)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=output_dtype,
        )


@keras_export(["keras.ops.var", "keras.ops.numpy.var"])
def var(x, axis=None, keepdims=False):
    """Compute the variance along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed. The default
            is to compute the variance of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the variance.
    """
    if any_symbolic_tensors((x,)):
        return Var(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.var(x, axis=axis, keepdims=keepdims)


class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.sum(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        dtype = dtypes.result_type(getattr(x, "dtype", backend.floatx()))
        # follow jax's rule
        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
        # TODO: torch doesn't support uint32
        if backend.backend() == "torch" and dtype == "uint32":
            dtype = "int32"
        sparse = getattr(x, "sparse", False)
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=dtype,
            sparse=sparse,
        )


@keras_export(["keras.ops.sum", "keras.ops.numpy.sum"])
def sum(x, axis=None, keepdims=False):
    """Sum of a tensor over the given axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the sum is computed. The default is to
            compute the sum of the flattened tensor.
        keepdims: If this is set to `True`, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        Output tensor containing the sum.
    """
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.sum(x, axis=axis, keepdims=keepdims)


class Zeros(Operation):
    def call(self, shape, dtype=None):
        return backend.numpy.zeros(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype=None):
        dtype = dtype or backend.floatx()
        return KerasTensor(shape, dtype=dtype)


@keras_export(["keras.ops.zeros", "keras.ops.numpy.zeros"])
def zeros(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with zeros.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of zeros with the given shape and dtype.
    """
    return backend.numpy.zeros(shape, dtype=dtype)


class Ones(Operation):
    def call(self, shape, dtype=None):
        return backend.numpy.ones(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype=None):
        dtype = dtype or backend.floatx()
        return KerasTensor(shape, dtype=dtype)


@keras_export(["keras.ops.ones", "keras.ops.numpy.ones"])
def ones(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with ones.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.

    Returns:
        Tensor of ones with the given shape and dtype.
    """
    return backend.numpy.ones(shape, dtype=dtype)


class Eye(Operation):
    def __init__(self, k=0, dtype=None):
        super().__init__()
        self.k = k
        self.dtype = dtype or backend.floatx()

    def call(self, N, M=None):
        return backend.numpy.eye(N, M=M, k=self.k, dtype=self.dtype)

    def compute_output_spec(self, N, M=None):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=self.dtype)


@keras_export(["keras.ops.eye", "keras.ops.numpy.eye"])
def eye(N, M=None, k=0, dtype=None):
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output.
        M: Number of columns in the output. If `None`, defaults to `N`.
        k: Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal,
            and a negative value to a lower diagonal.
        dtype: Data type of the returned tensor.

    Returns:
        Tensor with ones on the k-th diagonal and zeros elsewhere.
    """
    return backend.numpy.eye(N, M=M, k=k, dtype=dtype)


class FloorDivide(Operation):
    def call(self, x1, x2):
        return backend.numpy.floor_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.floor_divide", "keras.ops.numpy.floor_divide"])
def floor_divide(x1, x2):
    """Returns the largest integer smaller or equal to the division of inputs.

    Args:
        x1: Numerator.
        x2: Denominator.

    Returns:
        Output tensor, `y = floor(x1/x2)`
    """
    if any_symbolic_tensors((x1, x2)):
        return FloorDivide().symbolic_call(x1, x2)
    return backend.numpy.floor_divide(x1, x2)


class LogicalXor(Operation):
    def call(self, x1, x2):
        return backend.numpy.logical_xor(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype="bool")


@keras_export(["keras.ops.logical_xor", "keras.ops.numpy.logical_xor"])
def logical_xor(x1, x2):
    """Compute the truth value of `x1 XOR x2`, element-wise.

    Args:
        x1: First input tensor.
        x2: Second input tensor.

    Returns:
        Output boolean tensor.
    """
    if any_symbolic_tensors((x1, x2)):
        return LogicalXor().symbolic_call(x1, x2)
    return backend.numpy.logical_xor(x1, x2)


class Correlate(Operation):
    def __init__(self, mode="valid"):
        super().__init__()
        self.mode = mode

    def call(self, x1, x2):
        return backend.numpy.correlate(x1, x2, mode=self.mode)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        if len(x1_shape) != 1:
            raise ValueError(
                "`x1` must be a 1-dimensional tensor, but received"
                + f"shape {x1_shape}"
            )
        if len(x2_shape) != 1:
            raise ValueError(
                "`x2` must be a 1-dimensional tensor, but received"
                + f"shape {x2_shape}"
            )
        x1_len, x2_len = x1_shape[0], x2_shape[0]
        output_shape = (
            np.maximum(x1_len, x2_len) - np.minimum(x1_len, x2_len) + 1,
        )
        if self.mode == "same":
            output_shape = (np.maximum(x1_len, x2_len),)
        elif self.mode == "full":
            output_shape = (x1_len + x2_len - 1,)
        if self.mode not in ("valid", "same", "full"):
            raise ValueError(
                "`mode` must be either `valid`, `same`, or `full`, but"
                f"received: {self.mode}"
            )
        output_dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        if output_dtype == "int64":
            output_dtype = "float64"
        elif output_dtype not in ["bfloat16", "float16", "float64"]:
            output_dtype = "float32"
        return KerasTensor(output_shape, dtype=output_dtype)


@keras_export(["keras.ops.correlate", "keras.ops.numpy.correlate"])
def correlate(x1, x2, mode="valid"):
    """Compute the cross-correlation of two 1-dimensional tensors.

    Args:
        x1: First 1-dimensional input tensor of length M.
        x2: Second 1-dimensional input tensor of length N.
        mode: Either `valid`, `same` or `full`.
            By default the mode is set to `valid`, which returns
            an output of length max(M, N) - min(M, N) + 1.
            `same` returns an output of length max(M, N).
            `full` mode returns the convolution at each point of
            overlap, with an output length of N+M-1

    Returns:
        Output tensor, cross-correlation of `x1` and `x2`.
    """
    if any_symbolic_tensors((x1, x2)):
        return Correlate(mode=mode).symbolic_call(x1, x2)
    return backend.numpy.correlate(x1, x2, mode=mode)


class Select(Operation):
    def __init__(self):
        super().__init__()

    def call(self, condlist, choicelist, default=0):
        return backend.numpy.select(condlist, choicelist, default)

    def compute_output_spec(self, condlist, choicelist, default=0):
        first_element = choicelist[0]
        return KerasTensor(first_element.shape, dtype=first_element.dtype)


@keras_export(["keras.ops.select", "keras.ops.numpy.select"])
def select(condlist, choicelist, default=0):
    """Return elements from `choicelist`, based on conditions in `condlist`.

    Args:
        condlist: List of boolean tensors.
            The list of conditions which determine from which array
            in choicelist the output elements are taken.
            When multiple conditions are satisfied,
            the first one encountered in condlist is used.
        choicelist: List of tensors.
            The list of tensors from which the output elements are taken.
            This list has to be of the same length as `condlist`.
        defaults: Optional scalar value.
            The element inserted in the output
            when all conditions evaluate to `False`.

    Returns:
        Tensor where the output at position `m` is the `m`-th element
        of the tensor in `choicelist` where the `m`-th element of the
        corresponding tensor in `condlist` is `True`.

    Example:

    ```python
    from keras import ops

    x = ops.arange(6)
    condlist = [x<3, x>3]
    choicelist = [x, x**2]
    ops.select(condlist, choicelist, 42)
    # Returns: tensor([0,  1,  2, 42, 16, 25])
    ```
    """
    if not isinstance(condlist, (list, tuple)) or not isinstance(
        choicelist, (list, tuple)
    ):
        raise ValueError(
            "condlist and choicelist must be lists. Received: "
            f"type(condlist) = {type(condlist)}, "
            f"type(choicelist) = {type(choicelist)}"
        )
    condlist = list(condlist)
    choicelist = list(choicelist)
    if not condlist or not choicelist:
        raise ValueError(
            "condlist and choicelist must not be empty. Received: "
            f"condlist = {condlist}, "
            f"choicelist = {choicelist}"
        )
    if any_symbolic_tensors(condlist + choicelist + [default]):
        return Select().symbolic_call(condlist, choicelist, default)
    return backend.numpy.select(condlist, choicelist, default)


class Slogdet(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return backend.numpy.slogdet(x)

    def compute_output_spec(self, x):
        sign = KerasTensor((), dtype=x.dtype)
        logabsdet = KerasTensor(x.shape[:-2], dtype=x.dtype)
        return (sign, logabsdet)


@keras_export(["keras.ops.slogdet", "keras.ops.numpy.slogdet"])
def slogdet(x):
    """Compute the sign and natural logarithm of the determinant of a matrix.

    Args:
        x: Input matrix. It must 2D and square.

    Returns:
        A tuple `(sign, logabsdet)`. `sign` is a number representing
        the sign of the determinant. For a real matrix, this is 1, 0, or -1.
        For a complex matrix, this is a complex number with absolute value 1
        (i.e., it is on the unit circle), or else 0.
        `logabsdet` is the natural log of the absolute value of the determinant.
    """
    if any_symbolic_tensors((x,)):
        return Slogdet().symbolic_call(x)
    return backend.numpy.slogdet(x)


class Argpartition(Operation):
    def __init__(self, kth, axis=-1):
        super().__init__()
        if not isinstance(kth, int):
            raise ValueError("kth must be an integer. Received:" f"kth = {kth}")
        self.kth = kth
        self.axis = axis

    def call(self, x):
        return backend.numpy.argpartition(x, kth=self.kth, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="int32")


@keras_export(["keras.ops.argpartition", "keras.ops.numpy.argpartition"])
def argpartition(x, kth, axis=-1):
    """Performs an indirect partition along the given axis.

    It returns an array
    of indices of the same shape as `x` that index data along the given axis
    in partitioned order.

    Args:
        a: Array to sort.
        kth: Element index to partition by.
            The k-th element will be in its final sorted position and all
            smaller elements will be moved before it and all larger elements
            behind it. The order of all elements in the partitions is undefined.
            If provided with a sequence of k-th it will partition all of them
            into their sorted position at once.
        axis: Axis along which to sort. The default is -1 (the last axis).
            If `None`, the flattened array is used.

    Returns:
        Array of indices that partition `x` along the specified `axis`.
    """
    if any_symbolic_tensors((x,)):
        return Argpartition(kth, axis).symbolic_call(x)
    return backend.numpy.argpartition(x, kth, axis)


class Histogram(Operation):
    def __init__(self, bins=10, range=None):
        super().__init__()

        if not isinstance(bins, int):
            raise TypeError("bins must be of type `int`")
        if bins < 0:
            raise ValueError("`bins` should be a non-negative integer")

        if range:
            if len(range) < 2 or not isinstance(range, tuple):
                raise ValueError("range must be a tuple of two elements")

            if range[1] < range[0]:
                raise ValueError(
                    "The second element of range must be greater than the first"
                )

        self.bins = bins
        self.range = range

    def call(self, x):
        x = backend.convert_to_tensor(x)
        if len(x.shape) > 1:
            raise ValueError("Input tensor must be 1-dimensional")
        return backend.math.histogram(x, bins=self.bins, range=self.range)

    def compute_output_spec(self, x):
        return (
            KerasTensor(shape=(self.bins,), dtype=x.dtype),
            KerasTensor(shape=(self.bins + 1,), dtype=x.dtype),
        )


@keras_export(["keras.ops.histogram", "keras.ops.numpy.histogram"])
def histogram(x, bins=10, range=None):
    """Computes a histogram of the data tensor `x`.

    Args:
        x: Input tensor.
        bins: An integer representing the number of histogram bins.
            Defaults to 10.
        range: A tuple representing the lower and upper range of the bins.
            If not specified, it will use the min and max of `x`.

    Returns:
        A tuple containing:
        - A tensor representing the counts of elements in each bin.
        - A tensor representing the bin edges.

    Example:

    ```
    >>> input_tensor = np.random.rand(8)
    >>> keras.ops.histogram(input_tensor)
    (array([1, 1, 1, 0, 0, 1, 2, 1, 0, 1], dtype=int32),
    array([0.0189519 , 0.10294958, 0.18694726, 0.27094494, 0.35494262,
        0.43894029, 0.52293797, 0.60693565, 0.69093333, 0.77493101,
        0.85892869]))
    ```
    """
    if not isinstance(bins, int):
        raise TypeError(
            f"Argument `bins` must be of type `int`. Received: bins={bins}"
        )
    if bins < 0:
        raise ValueError(
            "Argument `bins` should be a non-negative integer. "
            f"Received: bins={bins}"
        )

    if range:
        if len(range) < 2 or not isinstance(range, tuple):
            raise ValueError(
                "Argument `range` must be a tuple of two elements. "
                f"Received: range={range}"
            )

        if range[1] < range[0]:
            raise ValueError(
                "The second element of `range` must be greater than the first. "
                f"Received: range={range}"
            )

    if any_symbolic_tensors((x,)):
        return Histogram(bins=bins, range=range).symbolic_call(x)

    x = backend.convert_to_tensor(x)
    if len(x.shape) > 1:
        raise ValueError(
            "Input tensor must be 1-dimensional. "
            f"Received: input.shape={x.shape}"
        )
    return backend.numpy.histogram(x, bins=bins, range=range)
