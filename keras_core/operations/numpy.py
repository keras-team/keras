"""
MANIFEST:

abs
absolute
add
all
amax
amin
append
arange
arccos
arcsin
arctan
arctan2
argmax
argmin
argsort
array
average
bincount
broadcast_to
ceil
clip
concatenate
conj
conjugate
copy
cos
count_nonzero
cross
cumprod
cumsum
diag
diagonal
diff
divide
dot
dtype
einsum
empty
equal
exp
expand_dims
expm1
eye
flip
floor
full
full_like
greater
greater_equal
hstack
identity
imag
interp
isclose
isfinite
isinf
isnan
less
less_equal
linspace
log
log10
log1p
log2
logaddexp
logical_and
logical_not
logical_or
logspace
matmul
max
maximum
mean
median
meshgrid
mgrid
min
minimum
mod
moveaxis
multiply
nan_to_num
ndim
nonzero
not_equal
ones
ones_like
outer
pad
percentile
power
prod
ravel
real
reciprocal
repeat
reshape
roll
round
sign
sin
size
sort
split
sqrt
square
squeeze
stack
std
subtract
sum
swapaxes
take
take_along_axis
tan
tensordot
tile
trace
transpose
tri
tril
triu
true_divide
vdot
vstack
where
zeros
zeros_like


"""
import re

import numpy as np

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations import operation_utils
from keras_core.operations.operation import Operation


def broadcast_shapes(shape1, shape2):
    # Broadcast input shapes to a unified shape.
    # Convert to list for mutability.
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2

    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
                output_shape[i] = shape1[i]
            else:
                raise ValueError(
                    "Cannot broadcast shape, the failure dim has value "
                    f"{shape1[i]}, which cannot be broadcasted to {shape2[i]}. "
                    f"Input shapes are: {origin_shape1} and {origin_shape2}."
                )

    return output_shape


def reduce_shape(shape, axis=None, keepdims=False):
    shape = list(shape)
    if axis is None:
        if keepdims:
            output_shape = [1 for _ in range(shape)]
        else:
            output_shape = []
        return output_shape

    if keepdims:
        for ax in axis:
            shape[ax] = 1
        return shape
    else:
        for ax in axis:
            shape[ax] = -1
        output_shape = list(filter((-1).__ne__, shape))
        return output_shape


def shape_equal(shape1, shape2, axis=None, allow_none=True):
    """Check if two shapes are equal.

    Args:
        shape1: A tuple or list of integers.
        shape2: A tuple or list of integers.
        axis: int or list/tuple of ints, defaults to None. If specified, the
            shape check will ignore the axes specified by `axis`.
        allow_none: bool, defaults to True. If True, None in the shape will
            match any value.
    """
    if len(shape1) != len(shape2):
        return False
    shape1 = list(shape1)
    shape2 = list(shape2)
    if axis is not None:
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
        return KerasTensor(x.shape, dtype=x.dtype)


def absolute(x):
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.numpy.absolute(x)


class Abs(Absolute):
    pass


def abs(x):
    return absolute(x)


class Add(Operation):
    def call(self, x1, x2):
        return backend.numpy.add(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def add(x1, x2):
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


def all(x, axis=None, keepdims=False):
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


def any(x, axis=None, keepdims=False):
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


def amax(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
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


def amin(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
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
        if self.axis is None:
            if None in x1_shape or None in x2_shape:
                output_shape = [None]
            else:
                output_shape = [int(np.prod(x1_shape) + np.prod(x2_shape))]
            return KerasTensor(output_shape, dtype=x1.dtype)

        if not shape_equal(x1_shape, x2_shape, [self.axis]):
            raise ValueError(
                "`append` requires inputs to have the same shape except the "
                f"`axis={self.axis}`, but received shape {x1_shape} and "
                f"{x2_shape}."
            )

        output_shape = list(x1_shape)
        output_shape[self.axis] = x1_shape[self.axis] + x2_shape[self.axis]
        return KerasTensor(output_shape, dtype=x1.dtype)


def append(
    x1,
    x2,
    axis=None,
):
    if any_symbolic_tensors((x1, x2)):
        return Append(axis=axis).symbolic_call(x1, x2)
    return backend.numpy.append(x1, x2, axis=axis)


class Arange(Operation):
    def call(self, start, stop=None, step=None, dtype=None):
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        return backend.numpy.arange(start, stop, step=step, dtype=dtype)

    def compute_output_spec(self, start, stop=None, step=None, dtype=None):
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        output_shape = [np.ceil((stop - start) / step).astype(int)]
        return KerasTensor(output_shape, dtype=dtype)


def arange(start, stop=None, step=None, dtype=None):
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    return backend.numpy.arange(start, stop, step=step, dtype=dtype)


class Arccos(Operation):
    def call(self, x):
        return backend.numpy.arccos(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arccos(x):
    if any_symbolic_tensors((x,)):
        return Arccos().symbolic_call(x)
    return backend.numpy.arccos(x)


class Arcsin(Operation):
    def call(self, x):
        return backend.numpy.arcsin(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arcsin(x):
    if any_symbolic_tensors((x,)):
        return Arcsin().symbolic_call(x)
    return backend.numpy.arcsin(x)


class Arctan(Operation):
    def call(self, x):
        return backend.numpy.arctan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arctan(x):
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
        return KerasTensor(outputs_shape, dtype=x1.dtype)


def arctan2(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Arctan2().symbolic_call(x1, x2)
    return backend.numpy.arctan2(x1, x2)


class Argmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


def argmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Argmax(axis=axis).symbolic_call(x)
    return backend.numpy.argmax(x, axis=axis)


class Argmin(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.argmin(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


def argmin(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Argmin(axis=axis).symbolic_call(x)
    return backend.numpy.argmin(x, axis=axis)


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


def argsort(x, axis=-1):
    if any_symbolic_tensors((x,)):
        return Argsort(axis=axis).symbolic_call(x)
    return backend.numpy.argsort(x, axis=axis)


class Array(Operation):
    def call(self, x, dtype=None):
        return backend.numpy.array(x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


def array(x, dtype=None):
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
        if weights is not None:
            shape_match = shape_equal(x.shape, weights.shape, allow_none=True)
            if self.axis is not None:
                shape_match_on_axis = shape_equal(
                    [x.shape[self.axis]], weights.shape, allow_none=True
                )
        if self.axis is None:
            if weights is None or shape_match:
                return KerasTensor(
                    [],
                    dtype=x.dtype,
                )
            else:
                raise ValueError(
                    "`weights` must have the same shape as `x` when "
                    f"`axis=None`, but received `weights.shape={weights.shape}`"
                    f" and `x.shape={x.shape}`."
                )

        if weights is None or shape_match_on_axis or shape_match:
            return KerasTensor(
                reduce_shape(x.shape, axis=[self.axis]),
                dtype=x.dtype,
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


def average(x, axis=None, weights=None):
    if any_symbolic_tensors((x,)):
        return Average(axis=axis).symbolic_call(x, weights=weights)
    return backend.numpy.average(x, weights=weights, axis=axis)


class Bincount(Operation):
    def __init__(self, weights=None, minlength=0):
        super().__init__()
        self.weights = weights
        self.minlength = minlength

    def call(self, x):
        return backend.numpy.bincount(
            x, weights=self.weights, minlength=self.minlength
        )

    def compute_output_spec(self, x):
        out_shape = backend.numpy.amax(x) + 1
        return KerasTensor(out_shape, dtype=x.dtype)


def bincount(x, weights=None, minlength=0):
    if any_symbolic_tensors((x,)):
        return Bincount(weights=weights, minlength=minlength).symbolic_call(x)
    return backend.numpy.bincount(x, weights=weights, minlength=minlength)


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


def broadcast_to(x, shape):
    if any_symbolic_tensors((x,)):
        return BroadcastTo(shape=shape).symbolic_call(x)
    return backend.numpy.broadcast_to(x, shape)


class Ceil(Operation):
    def call(self, x):
        return backend.numpy.ceil(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def ceil(x):
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
        return KerasTensor(x.shape, dtype=x.dtype)


def clip(x, x_min, x_max):
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
        output_shape = list(first_shape)
        output_shape[self.axis] = total_size_on_axis
        return KerasTensor(output_shape, dtype=x.dtype)


def concatenate(xs, axis=0):
    if any_symbolic_tensors(xs):
        return Concatenate(axis=axis).symbolic_call(xs)
    return backend.numpy.concatenate(xs, axis=axis)


class Conjugate(Operation):
    def call(self, x):
        return backend.numpy.conjugate(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def conjugate(x):
    if any_symbolic_tensors((x,)):
        return Conjugate().symbolic_call(x)
    return backend.numpy.conjugate(x)


class Conj(Conjugate):
    pass


def conj(x):
    return conjugate(x)


class Copy(Operation):
    def call(self, x):
        return backend.numpy.copy(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def copy(x):
    if any_symbolic_tensors((x,)):
        return Copy().symbolic_call(x)
    return backend.numpy.copy(x)


class Cos(Operation):
    def call(self, x):
        return backend.numpy.cos(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def cos(x):
    if any_symbolic_tensors((x,)):
        return Cos().symbolic_call(x)
    return backend.numpy.cos(x)


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


def count_nonzero(x, axis=None):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
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
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.cumprod(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
            return KerasTensor(output_shape, dtype="int32")
        return KerasTensor(x.shape, dtype=x.dtype)


def cumprod(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Cumprod(axis=axis).symbolic_call(x)
    return backend.numpy.cumprod(x, axis=axis)


class Cumsum(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.cumsum(x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
            return KerasTensor(output_shape, dtype="int32")
        return KerasTensor(x.shape, dtype=x.dtype)


def cumsum(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Cumsum(axis=axis).symbolic_call(x)
    return backend.numpy.cumsum(x, axis=axis)


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


def diag(x, k=0):
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


def diagonal(x, offset=0, axis1=0, axis2=1):
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


class Dot(Operation):
    def call(self, x1, x2):
        return backend.numpy.dot(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
        if x1_shape == [] or x2_shape == []:
            return multiply(x1, x2)
        if len(x1_shape) == 1 and len(x2_shape) == 1:
            return KerasTensor([], dtype=x1.dtype)
        if len(x2_shape) == 1:
            if x1_shape[-1] != x2_shape[0]:
                raise ValueError(
                    "Shape must match on the last axis of `x1` and `x2` when "
                    "`x1` is N-d array while `x2` is 1-D, but receive shape "
                    f"`x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
                )
            return KerasTensor(x1_shape[:-1], dtype=x1.dtype)

        if (
            x1_shape[-1] is None
            or x2_shape[-2] is None
            or x1_shape[-1] == x2_shape[-2]
        ):
            del x1_shape[-1]
            del x2_shape[-2]
            return KerasTensor(x1_shape + x2_shape, dtype=x1.dtype)

        raise ValueError(
            "Shape must match on the last axis of `x1` and second last "
            "axis of `x2` when `x1` is N-d array while `x2` is M-D, but "
            f"received `x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
        )


def dot(x1, x2):
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
            operand should have a target shape containing only number and None.
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
        dtype = None
        for x in operands:
            if hasattr(x, "dtype"):
                dtype = x.dtype
                break
        return KerasTensor(output_shape, dtype=dtype)


def einsum(subscripts, *operands):
    if any_symbolic_tensors(operands):
        return Einsum(subscripts).symbolic_call(*operands)
    return backend.numpy.einsum(subscripts, *operands)


class Empty(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.empty(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def empty(shape, dtype="float32"):
    return backend.numpy.empty(shape, dtype=dtype)


class Equal(Operation):
    def call(self, x1, x2):
        return backend.numpy.equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def equal(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Equal().symbolic_call(x1, x2)
    return backend.numpy.equal(x1, x2)


class Exp(Operation):
    def call(self, x):
        return backend.numpy.exp(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def exp(x):
    if any_symbolic_tensors((x,)):
        return Exp().symbolic_call(x)
    return backend.numpy.exp(x)


class ExpandDims(Operation):
    def __init__(self, axis):
        super().__init__()
        if isinstance(axis, list):
            raise ValueError(
                "The `axis` argument to `expand_dims` should be an integer, "
                f"but received a list: {axis}."
            )
        self.axis = axis

    def call(self, x):
        return backend.numpy.expand_dims(x, self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if self.axis < 0:
            axis = len(x.shape) + 1 + self.axis
        else:
            axis = self.axis
        output_shape = x_shape[:axis] + [1] + x_shape[axis:]
        return KerasTensor(output_shape, dtype=x.dtype)


def expand_dims(x, axis):
    if any_symbolic_tensors((x,)):
        return ExpandDims(axis=axis).symbolic_call(x)
    return backend.numpy.expand_dims(x, axis)


class Expm1(Operation):
    def call(self, x):
        return backend.numpy.expm1(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def expm1(x):
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


def flip(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Flip(axis=axis).symbolic_call(x)
    return backend.numpy.flip(x, axis=axis)


class Floor(Operation):
    def call(self, x):
        return backend.numpy.floor(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def floor(x):
    if any_symbolic_tensors((x,)):
        return Floor().symbolic_call(x)
    return backend.numpy.floor(x)


class Full(Operation):
    def call(self, shape, fill_value, dtype=None):
        return backend.numpy.full(shape, fill_value, dtype=dtype)

    def compute_output_spec(self, shape, fill_value, dtype=None):
        return KerasTensor(shape, dtype=dtype)


def full(shape, fill_value, dtype=None):
    return backend.numpy.full(shape, fill_value, dtype=dtype)


class FullLike(Operation):
    def call(self, x, fill_value, dtype=None):
        return backend.numpy.full_like(x, fill_value, dtype=dtype)

    def compute_output_spec(self, x, fill_value, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    if any_symbolic_tensors((x,)):
        return FullLike().symbolic_call(x, fill_value, dtype=dtype)
    return backend.numpy.full_like(x, fill_value, dtype=dtype)


class GetItem(Operation):
    def call(self, x, key):
        if not isinstance(key, int):
            # TODO: support slicing.
            raise ValueError(
                "Only scalar int keys are supported at this time. Cannot "
                f"process key {key}"
            )
        return x[key]

    def compute_output_spec(self, x, key):
        if not isinstance(key, int):
            # TODO: support slicing.
            raise ValueError(
                "Only scalar int keys are supported at this time. Cannot "
                f"process key {key}"
            )
        if len(x.shape) == 0:
            raise ValueError(
                f"Too many indices for array: array is scalar "
                f"but index {key} was requested. A scalar array "
                "cannot be indexed."
            )
        if x.shape[0] is not None and key >= x.shape[0]:
            raise ValueError(
                f"Array has shape {x.shape} "
                f"but out-of-bound index {key} was requested."
            )
        return KerasTensor(x.shape[1:], dtype=x.dtype)


def get_item(x, key):
    if any_symbolic_tensors((x,)):
        return GetItem().symbolic_call(x, key)
    if not isinstance(key, int):
        # TODO: support slicing.
        raise ValueError(
            "Only scalar int keys are supported at this time. Cannot "
            f"process key {key}"
        )
    return x[key]


class Greater(Operation):
    def call(self, x1, x2):
        return backend.numpy.greater(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def greater(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def greater_equal(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return GreaterEqual().symbolic_call(x1, x2)
    return backend.numpy.greater_equal(x1, x2)


class Hstack(Operation):
    def call(self, xs):
        return backend.numpy.hstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
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
        output_shape = list(first_shape)
        output_shape[1] = total_size_on_axis
        return KerasTensor(output_shape)


def hstack(xs):
    if any_symbolic_tensors((xs,)):
        return Hstack().symbolic_call(xs)
    return backend.numpy.hstack(xs)


class Identity(Operation):
    def call(self, n, dtype="float32"):
        return backend.numpy.identity(n, dtype=dtype)

    def compute_output_spec(self, n, dtype="float32"):
        return KerasTensor([n, n], dtype=dtype)


def identity(n, dtype="float32"):
    return backend.numpy.identity(n, dtype=dtype)


class Imag(Operation):
    def call(self, x):
        return backend.numpy.imag(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def imag(x):
    if any_symbolic_tensors((x,)):
        return Imag().symbolic_call(x)
    return backend.numpy.imag(x)


class Isclose(Operation):
    def call(self, x1, x2):
        return backend.numpy.isclose(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def isclose(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Isclose().symbolic_call(x1, x2)
    return backend.numpy.isclose(x1, x2)


class Isfinite(Operation):
    def call(self, x):
        return backend.numpy.isfinite(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isfinite(x):
    if any_symbolic_tensors((x,)):
        return Isfinite().symbolic_call(x)
    return backend.numpy.isfinite(x)


class Isinf(Operation):
    def call(self, x):
        return backend.numpy.isinf(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isinf(x):
    if any_symbolic_tensors((x,)):
        return Isinf().symbolic_call(x)
    return backend.numpy.isinf(x)


class Isnan(Operation):
    def call(self, x):
        return backend.numpy.isnan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isnan(x):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def less(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def less_equal(x1, x2):
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

        dtype = self.dtype if self.dtype is not None else start.dtype
        if self.retstep:
            return (KerasTensor(output_shape, dtype=dtype), None)
        return KerasTensor(output_shape, dtype=dtype)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
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
        return KerasTensor(x.shape, dtype=x.dtype)


def log(x):
    if any_symbolic_tensors((x,)):
        return Log().symbolic_call(x)
    return backend.numpy.log(x)


class Log10(Operation):
    def call(self, x):
        return backend.numpy.log10(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def log10(x):
    if any_symbolic_tensors((x,)):
        return Log10().symbolic_call(x)
    return backend.numpy.log10(x)


class Log1p(Operation):
    def call(self, x):
        return backend.numpy.log1p(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def log1p(x):
    if any_symbolic_tensors((x,)):
        return Log1p().symbolic_call(x)
    return backend.numpy.log1p(x)


class Log2(Operation):
    def call(self, x):
        return backend.numpy.log2(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def log2(x):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def logaddexp(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def logical_and(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return LogicalAnd().symbolic_call(x1, x2)
    return backend.numpy.logical_and(x1, x2)


class LogicalNot(Operation):
    def call(self, x):
        return backend.numpy.logical_not(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def logical_not(x):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def logical_or(x1, x2):
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

        dtype = self.dtype if self.dtype is not None else start.dtype
        return KerasTensor(output_shape, dtype=dtype)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
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
        if len(x1_shape) == 1:
            x1_shape = (1, x1_shape[0])
        if len(x2_shape) == 1:
            x2_shape = (x2_shape[0], 1)
        if (
            x1_shape[-1] is not None
            and x2_shape[-2] is not None
            and x1_shape[-1] != x2_shape[-2]
        ):
            raise ValueError(
                "Inner dimensions (`x1.shape[-1]` and `x2.shape[-2]`) must be "
                f"equal, but received `x1.shape={x1.shape}` and "
                f"`x2.shape={x2.shape}`."
            )

        leading_shape = broadcast_shapes(x1_shape[:-2], x2_shape[:-2])
        last_2_dims_shape = [x1_shape[-2], x2_shape[-1]]
        output_shape = leading_shape + last_2_dims_shape
        if len(x1.shape) == 1:
            del output_shape[-2]
        if len(x2.shape) == 1:
            del output_shape[-1]
        return KerasTensor(output_shape, dtype=x1.dtype)


def matmul(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Matmul().symbolic_call(x1, x2)
    # The below conversion works around an outstanding JAX bug.
    x1 = backend.convert_to_tensor(x1)
    x2 = backend.convert_to_tensor(x2)
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


def max(x, axis=None, keepdims=False, initial=None):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def maximum(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Maximum().symbolic_call(x1, x2)
    return backend.numpy.maximum(x1, x2)


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
        return [KerasTensor(output_shape) for _ in range(len(x))]


def meshgrid(*x, indexing="xy"):
    if any_symbolic_tensors(x):
        return Meshgrid(indexing=indexing).symbolic_call(*x)
    return backend.numpy.meshgrid(*x, indexing=indexing)


class Min(Operation):
    def __init__(self, axis=None, keepdims=False, initial=None):
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


def min(x, axis=None, keepdims=False, initial=None):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def minimum(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def mod(x1, x2):
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


def moveaxis(x, source, destination):
    if any_symbolic_tensors((x,)):
        return Moveaxis(source, destination).symbolic_call(x)
    return backend.numpy.moveaxis(x, source=source, destination=destination)


class NanToNum(Operation):
    def call(self, x):
        return backend.numpy.nan_to_num(x)


def nan_to_num(x):
    return backend.numpy.nan_to_num(x)


class Ndim(Operation):
    def call(self, x):
        return backend.numpy.ndim(
            x,
        )

    def compute_output_spec(self, x):
        return KerasTensor([len(x.shape)])


def ndim(x):
    if any_symbolic_tensors((x,)):
        return Ndim().symbolic_call(x)
    return backend.numpy.ndim(x)


class Nonzero(Operation):
    def call(self, x):
        return backend.numpy.nonzero(x)


def nonzero(x):
    return backend.numpy.nonzero(x)


class NotEqual(Operation):
    def call(self, x1, x2):
        return backend.numpy.not_equal(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def not_equal(x1, x2):
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


def ones_like(x, dtype=None):
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


def zeros_like(x, dtype=None):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def outer(x1, x2):
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
                    "`pad_width` should be a list of tuples of length 2 or "
                    f"1, but received {pad_width}."
                )
            if len(pw) == 1:
                pad_width[i] = (pw[0], pw[0])
        return pad_width

    def call(self, x):
        return backend.numpy.pad(x, pad_width=self.pad_width, mode=self.mode)

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        if len(self.pad_width) == 1:
            pad_width = [self.pad_width[0] for _ in range(len(output_shape))]
        elif len(self.pad_width) == len(output_shape):
            pad_width = self.pad_width
        else:
            raise ValueError(
                "`pad_width` must have the same length as `x.shape`, but "
                f"received {len(self.pad_width)} and {len(x.shape)}."
            )

        for i in range(len(output_shape)):
            if output_shape[i] is None:
                output_shape[i] = None
            else:
                output_shape[i] += pad_width[i][0] + pad_width[i][1]
        return KerasTensor(output_shape, dtype=x.dtype)


def pad(x, pad_width, mode="constant"):
    if any_symbolic_tensors((x,)):
        return Pad(pad_width, mode=mode).symbolic_call(x)
    return backend.numpy.pad(x, pad_width, mode=mode)


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
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=self.dtype,
        )


def prod(x, axis=None, keepdims=False, dtype=None):
    if any_symbolic_tensors((x,)):
        return Prod(axis=axis, keepdims=keepdims, dtype=dtype).symbolic_call(x)
    return backend.numpy.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


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


def ravel(x):
    if any_symbolic_tensors((x,)):
        return Ravel().symbolic_call(x)
    return backend.numpy.ravel(x)


class Real(Operation):
    def call(self, x):
        return backend.numpy.real(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def real(x):
    if any_symbolic_tensors((x,)):
        return Real().symbolic_call(x)
    return backend.numpy.real(x)


class Reciprocal(Operation):
    def call(self, x):
        return backend.numpy.reciprocal(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def reciprocal(x):
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
        if self.axis is None:
            if None in x_shape:
                return KerasTensor([None], dtype=x.dtype)

            x_flatten_size = int(np.prod(x_shape))
            if isinstance(self.repeats, int):
                output_shape = [x_flatten_size * self.repeats]
            else:
                output_shape = [int(np.sum(self.repeats))]
            return KerasTensor(output_shape, dtype=x.dtype)

        size_on_ax = x_shape[self.axis]
        output_shape = x_shape
        if isinstance(self.repeats, int):
            output_shape[self.axis] = size_on_ax * self.repeats
        else:
            output_shape[self.axis] = int(np.sum(self.repeats))
        return KerasTensor(output_shape, dtype=x.dtype)


def repeat(x, repeats, axis=None):
    if any_symbolic_tensors((x,)):
        return Repeat(repeats, axis=axis).symbolic_call(x)
    return backend.numpy.repeat(x, repeats, axis=axis)


class Reshape(Operation):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def call(self, x):
        return backend.numpy.reshape(x, self.new_shape)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_reshape_output_shape(
            x.shape, self.new_shape, "new_shape"
        )
        return KerasTensor(output_shape, dtype=x.dtype)


def reshape(x, new_shape):
    if any_symbolic_tensors((x,)):
        return Reshape(new_shape).symbolic_call(x)
    return backend.numpy.reshape(x, new_shape)


class Roll(Operation):
    def __init__(self, shift, axis=None):
        super().__init__()
        self.shift = shift
        self.axis = axis

    def call(self, x):
        return backend.numpy.roll(x, self.shift, self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def roll(x, shift, axis=None):
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
        return KerasTensor(x.shape, dtype=x.dtype)


def round(x, decimals=0):
    if any_symbolic_tensors((x,)):
        return Round(decimals).symbolic_call(x)
    return backend.numpy.round(x, decimals)


class Sign(Operation):
    def call(self, x):
        return backend.numpy.sign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="int32")


def sign(x):
    if any_symbolic_tensors((x,)):
        return Sign().symbolic_call(x)
    return backend.numpy.sign(x)


class Sin(Operation):
    def call(self, x):
        return backend.numpy.sin(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def sin(x):
    if any_symbolic_tensors((x,)):
        return Sin().symbolic_call(x)
    return backend.numpy.sin(x)


class Size(Operation):
    def call(self, x):
        return backend.numpy.size(x)

    def compute_output_spec(self, x):
        return KerasTensor([], dtype="int32")


def size(x):
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


def sort(x, axis=-1):
    if any_symbolic_tensors((x,)):
        return Sort(axis=axis).symbolic_call(x)
    return backend.numpy.sort(x, axis=axis)


class Split(Operation):
    def __init__(self, indices_or_sections, axis=0):
        super().__init__()
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

        indices_or_sections = [0] + self.indices_or_sections
        output_size = np.diff(indices_or_sections)
        outputs = []
        for i in range(len(output_size)):
            output_shape = list(x_shape)
            output_shape[self.axis] = int(output_size[i])
            outputs.append(KerasTensor(output_shape, dtype=x.dtype))
        return outputs


def split(x, indices_or_sections, axis=0):
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
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[], allow_none=True):
                raise ValueError(
                    "Every value in `xs` must have the same shape. But found "
                    f"element of shape {x.shape},  which is different from the "
                    f"first element's shape {first_shape}."
                )

        size_on_axis = len(xs)
        output_shape = list(first_shape)
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)
        return KerasTensor(output_shape, dtype=x.dtype)


def stack(x, axis=0):
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
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
        )


def std(x, axis=None, keepdims=False):
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


def swapaxes(x, axis1, axis2):
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
        indices_shape = list(getattr(np.array(indices), "shape", []))
        if self.axis is None:
            return KerasTensor(indices_shape, dtype=x.dtype)

        if self.axis == -1:
            output_shape = x_shape[:-1] + indices_shape
        else:
            output_shape = (
                x_shape[: self.axis] + indices_shape + x_shape[self.axis + 1 :]
            )
        return KerasTensor(output_shape, dtype=x.dtype)


def take(x, indices, axis=None):
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
        x_shape = list(x.shape)
        indices_shape = list(indices.shape)
        if self.axis is None:
            x_shape = [None] if None in x_shape else [int(np.prod(x_shape))]

        if len(x_shape) != len(indices_shape):
            raise ValueError(
                "`x` and `indices` must have the same number of dimensions, "
                f"but receive shape {x_shape} and {indices_shape}."
            )

        del x_shape[self.axis]
        del indices_shape[self.axis]
        output_shape = broadcast_shapes(x_shape, indices_shape)
        size_on_axis = indices.shape[self.axis]
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)

        return KerasTensor(output_shape, dtype=x.dtype)


def take_along_axis(x, indices, axis=None):
    if any_symbolic_tensors((x, indices)):
        return TakeAlongAxis(axis=axis).symbolic_call(x, indices)
    return backend.numpy.take_along_axis(x, indices, axis=axis)


class Tan(Operation):
    def call(self, x):
        return backend.numpy.tan(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def tan(x):
    if any_symbolic_tensors((x,)):
        return Tan().symbolic_call(x)
    return backend.numpy.tan(x)


class Tensordot(Operation):
    def __init__(self, axes=2):
        super().__init__()
        self.axes = axes

    def call(self, x1, x2):
        return backend.numpy.tensordot(x1, x2, axes=self.axes)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(getattr(x1, "shape", []))
        x2_shape = list(getattr(x2, "shape", []))
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
            return KerasTensor(output_shape, dtype=x1.dtype)

        if self.axes <= 0:
            output_shape = x1_shape + x2_shape
        else:
            output_shape = x1_shape[: -self.axes] + x2_shape[self.axes :]

        return KerasTensor(output_shape, dtype=x1.dtype)


def tensordot(x1, x2, axes=2):
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


def tile(x, repeats):
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
        return KerasTensor(output_shape, dtype=x.dtype)


def trace(x, offset=0, axis1=0, axis2=1):
    if any_symbolic_tensors((x,)):
        return Trace(offset, axis1, axis2).symbolic_call(x)
    return backend.numpy.trace(x, offset=offset, axis1=axis1, axis2=axis2)


class Tri(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.numpy.tri(N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


def tri(N, M=None, k=0, dtype="float32"):
    return backend.numpy.tri(N, M=M, k=k, dtype=dtype)


class Tril(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.tril(x, k=self.k)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def tril(x, k=0):
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


def triu(x, k=0):
    if any_symbolic_tensors((x,)):
        return Triu(k=k).symbolic_call(x)
    return backend.numpy.triu(x, k=k)


class Vdot(Operation):
    def call(self, x1, x2):
        return backend.numpy.vdot(x1, x2)

    def compute_output_spec(self, x1, x2):
        return KerasTensor([], dtype=x1.dtype)


def vdot(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Vdot().symbolic_call(x1, x2)
    return backend.numpy.vdot(x1, x2)


class Vstack(Operation):
    def call(self, xs):
        return backend.numpy.vstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
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
        output_shape = list(first_shape)
        output_shape[0] = total_size_on_axis
        return KerasTensor(output_shape)


def vstack(xs):
    if any_symbolic_tensors((xs,)):
        return Vstack().symbolic_call(xs)
    return backend.numpy.vstack(xs)


class Where(Operation):
    def call(self, condition, x1, x2):
        return backend.numpy.where(condition, x1, x2)

    def compute_output_spec(self, condition, x1, x2):
        condition_shape = getattr(condition, "shape", [])
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(condition_shape, x1_shape)
        output_shape = broadcast_shapes(output_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def where(condition, x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def subtract(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def multiply(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.numpy.divide(x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.numpy.true_divide(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def true_divide(x1, x2):
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
        return KerasTensor(output_shape, dtype=x1.dtype)


def power(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.numpy.power(x1, x2)


class Negative(Operation):
    def call(self, x):
        return backend.numpy.negative(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def negative(x):
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.numpy.negative(x)


class Square(Operation):
    def call(self, x):
        return backend.numpy.square(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def square(x):
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.numpy.square(x)


class Sqrt(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.numpy.sqrt(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def sqrt(x):
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

    def compute_output_spec(self, x, axis=None):
        input_shape = list(x.shape)
        if axis is None:
            output_shape = list(filter((1).__ne__, input_shape))
            return KerasTensor(output_shape)
        else:
            if input_shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {axis}, because the dimension is not "
                    "1."
                )
            del input_shape[axis]
            return KerasTensor(input_shape, dtype=x.dtype)


def squeeze(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Squeeze().symbolic_call(x, axis=axis)
    return backend.numpy.squeeze(x, axis=axis)


class Transpose(Operation):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes

    def call(self, x):
        return backend.numpy.transpose(x, axes=self.axes)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if self.axes is None:
            return KerasTensor(x_shape[::-1])

        if len(self.axes) != len(x_shape):
            raise ValueError(
                "axis must be a list of the same length as the input shape, "
                f"expected {len(x_shape)}, but received {len(self.axes)}."
            )
        output_shape = []
        for ax in self.axes:
            output_shape.append(x_shape[ax])
        return KerasTensor(output_shape, dtype=x.dtype)


def transpose(x, axes=None):
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
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def mean(x, axis=None, keepdims=False):
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
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def var(x, axis=None, keepdims=False):
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
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def sum(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.numpy.sum(x, axis=axis, keepdims=keepdims)


class Zeros(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.zeros(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    return backend.numpy.zeros(shape, dtype=dtype)


class Ones(Operation):
    def call(self, shape, dtype="float32"):
        return backend.numpy.ones(shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def ones(shape, dtype="float32"):
    return backend.numpy.ones(shape, dtype=dtype)


class Eye(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.numpy.eye(N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


def eye(N, M=None, k=0, dtype="float32"):
    return backend.numpy.eye(N, M=M, k=k, dtype=dtype)
