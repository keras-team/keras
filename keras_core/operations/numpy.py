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
shape
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
unique
unrival_index
vdot
vectorize
vstack
where
zeros
zeros_like


"""
import numpy as np

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


def broadcast_shapes(shape1, shape2):
    # Broadcast input shapes to a unified shape.
    # Convert to list for mutability.
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2

    if len(shape1) > len(shape2):
        shape2 = [None] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [None] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = shape2[i]
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
                shape1[i] = -1
            if shape2[i] is None:
                shape2[i] = -1

    return shape1 == shape2


class Absolute(Operation):
    def call(self, x):
        return backend.execute("absolute", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def absolute(x):
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.execute("absolute", x)


class Abs(Absolute):
    pass


def abs(x):
    return absolute(x)


class Add(Operation):
    def call(self, x1, x2):
        return backend.execute("add", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def add(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Add().symbolic_call(x1, x2)
    return backend.execute("add", x1, x2)


class All(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute(
            "all",
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
            dtype=x.dtype,
        )


def all(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("all", x, axis=axis, keepdims=keepdims)


class Amax(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute(
            "amax",
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
    return backend.execute("amax", x, axis=axis, keepdims=keepdims)


class Amin(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute(
            "amin", x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def amin(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return All(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("amin", x, axis=axis, keepdims=keepdims)


class Append(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x1, x2):
        return backend.execute("append", x1, x2, axis=self.axis)

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
    return backend.execute("append", x1, x2, axis=axis)


class Arange(Operation):
    def call(self, start, stop=None, step=None, dtype=None):
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        return backend.execute("arange", start, stop, step=step, dtype=dtype)

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
    return backend.execute("arange", start, stop, step=step, dtype=dtype)


class Arccos(Operation):
    def call(self, x):
        return backend.execute("arccos", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arccos(x):
    if any_symbolic_tensors((x,)):
        return Arccos().symbolic_call(x)
    return backend.execute("arccos", x)


class Arcsin(Operation):
    def call(self, x):
        return backend.execute("arcsin", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arcsin(x):
    if any_symbolic_tensors((x,)):
        return Arcsin().symbolic_call(x)
    return backend.execute("arcsin", x)


class Arctan(Operation):
    def call(self, x):
        return backend.execute("arctan", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def arctan(x):
    if any_symbolic_tensors((x,)):
        return Arctan().symbolic_call(x)
    return backend.execute("arctan", x)


class Arctan2(Operation):
    def call(self, x1, x2):
        return backend.execute("arctan2", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        outputs_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(outputs_shape, dtype=x1.dtype)


def arctan2(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Arctan2().symbolic_call(x1, x2)
    return backend.execute("arctan2", x1, x2)


class Argmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("argmax", x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


def argmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Argmax(axis=axis).symbolic_call(x)
    return backend.execute("argmax", x, axis=axis)


class Argmin(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("argmin", x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([], dtype="int32")
        return KerasTensor(
            reduce_shape(x.shape, axis=[self.axis]), dtype="int32"
        )


def argmin(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Argmin(axis=axis).symbolic_call(x)
    return backend.execute("argmin", x, axis=axis)


class Argsort(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("argsort", x, axis=self.axis)

    def compute_output_spec(self, x):
        if self.axis is None:
            return KerasTensor([int(np.prod(x.shape))], dtype="int32")
        return KerasTensor(x.shape, dtype="int32")


def argsort(x, axis=-1):
    if any_symbolic_tensors((x,)):
        return Argsort(axis=axis).symbolic_call(x)
    return backend.execute("argsort", x, axis=axis)


class Array(Operation):
    def call(self, x, dtype=None):
        return backend.execute("array", x, dtype=dtype)

    def compute_output_spec(self, x, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


def array(x, dtype=None):
    if any_symbolic_tensors((x,)):
        return Array().symbolic_call(x, dtype=dtype)
    return backend.execute("array", x, dtype=dtype)


class Average(Operation):
    def __init__(self, axis=None):
        super().__init__()
        # np.average() does not support axis as tuple as declared by the
        # docstring, it only supports int or None.
        self.axis = axis

    def call(self, x, weights=None):
        return backend.execute("average", x, weights=weights, axis=self.axis)

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
    return backend.execute("average", x, weights=weights, axis=axis)


class BroadcastTo(Operation):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def call(self, x):
        return backend.execute("broadcast_to", x, self.shape)

    def compute_output_spec(self, x):
        # Catch broadcasting errors for clear error messages.
        broadcast_shapes(x.shape, self.shape)
        return KerasTensor(self.shape, dtype=x.dtype)


def broadcast_to(x, shape):
    if any_symbolic_tensors((x,)):
        return BroadcastTo(shape=shape).symbolic_call(x)
    return backend.execute("broadcast_to", x, shape)


class Ceil(Operation):
    def call(self, x):
        return backend.execute("ceil", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def ceil(x):
    if any_symbolic_tensors((x,)):
        return Ceil().symbolic_call(x)
    return backend.execute("ceil", x)


class Clip(Operation):
    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return backend.execute("clip", x, self.x_min, self.x_max)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def clip(x, x_min, x_max):
    if any_symbolic_tensors((x,)):
        return Clip(x_min, x_max).symbolic_call(x)
    return backend.execute("clip", x, x_min, x_max)


class Concatenate(Operation):
    def __init__(self, axis=0):
        super().__init__()
        if axis is None:
            raise ValueError("`axis` cannot be None for `concatenate`.")
        self.axis = axis

    def call(self, xs):
        return backend.execute("concatenate", xs, axis=self.axis)

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
    return backend.execute("concatenate", xs, axis=axis)


class Conjugate(Operation):
    def call(self, x):
        return backend.execute("conjugate", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def conjugate(x):
    if any_symbolic_tensors((x,)):
        return Conjugate().symbolic_call(x)
    return backend.execute("conjugate", x)


class Conj(Conjugate):
    pass


def conj(x):
    return conjugate(x)


class Copy(Operation):
    def call(self, x):
        return backend.execute("copy", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def copy(x):
    if any_symbolic_tensors((x,)):
        return Copy().symbolic_call(x)
    return backend.execute("copy", x)


class Cos(Operation):
    def call(self, x):
        return backend.execute("cos", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def cos(x):
    if any_symbolic_tensors((x,)):
        return Cos().symbolic_call(x)
    return backend.execute("cos", x)


class CountNonzero(Operation):
    def __init__(self, axis=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis

    def call(self, x):
        return backend.execute(
            "count_nonzero",
            x,
            axis=self.axis,
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis),
            dtype="int32",
        )


def count_nonzero(x, axis=None):
    if any_symbolic_tensors((x,)):
        return CountNonzero(axis=axis).symbolic_call(x)
    return backend.execute("count_nonzero", x, axis=axis)


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
        return backend.execute(
            "cross", x1, x2, self.axisa, self.axisb, self.axisc
        )

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
    return backend.execute(
        "cross",
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
        return backend.execute("cumprod", x, axis=self.axis)

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
    return backend.execute("cumprod", x, axis=axis)


class Cumsum(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("cumsum", x, axis=self.axis)

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
    return backend.execute("cumsum", x, axis=axis)


class Diag(Operation):
    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.execute("diag", x, k=self.k)

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
                shorter_side = min(x_shape[0], x_shape[1])
                if self.k > 0:
                    remaining = x_shape[1] - self.k
                else:
                    remaining = x_shape[0] + self.k
                output_shape = [max(0, min(remaining, shorter_side))]
        else:
            raise ValueError(
                f"`x` must be 1-D or 2-D, but received shape {x.shape}."
            )
        return KerasTensor(output_shape, dtype=x.dtype)


def diag(x, k=0):
    if any_symbolic_tensors((x,)):
        return Diag(k=k).symbolic_call(x)
    return backend.execute("diag", x, k=k)


class Diagonal(Operation):
    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.execute(
            "diagonal",
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
            shorter_side = min(shape_2d[0], shape_2d[1])
            if self.offset > 0:
                remaining = shape_2d[1] - self.offset
            else:
                remaining = shape_2d[0] + self.offset
            diag_shape = [max(0, min(remaining, shorter_side))]
        output_shape = output_shape + diag_shape
        return KerasTensor(output_shape, dtype=x.dtype)


def diagonal(x, offset=0, axis1=0, axis2=1):
    if any_symbolic_tensors((x,)):
        return Diagonal(
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        ).symbolic_call(x)
    return backend.execute(
        "diagonal",
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


class Dot(Operation):
    def call(self, x1, x2):
        return backend.execute("dot", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
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

        if x1_shape[-1] != x2_shape[-2]:
            raise ValueError(
                "Shape must match on the last axis of `x1` and second last "
                "axis of `x2` when `x1` is N-d array while `x2` is M-D, but "
                f"received `x1.shape={x1.shape}` and x2.shape=`{x2.shape}`."
            )
        del x1_shape[-1]
        del x2_shape[-2]
        return KerasTensor(x1_shape + x2_shape, dtype=x1.dtype)


class Empty(Operation):
    def call(self, shape, dtype="float32"):
        return backend.execute("empty", shape, dtype=dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def empty(shape, dtype="float32"):
    return backend.execute("empty", shape, dtype=dtype)


class Equal(Operation):
    def call(self, x1, x2):
        return backend.execute("equal", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def equal(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Equal().symbolic_call(x1, x2)
    return backend.execute("equal", x1, x2)


class Exp(Operation):
    def call(self, x):
        return backend.execute("exp", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def exp(x):
    if any_symbolic_tensors((x,)):
        return Exp().symbolic_call(x)
    return backend.execute("exp", x)


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
        return backend.execute("expand_dims", x, self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if self.axis < 0:
            output_shape = (
                x_shape[: self.axis + 1] + [1] + x_shape[self.axis + 1 :]
            )
        else:
            output_shape = x_shape[: self.axis] + [1] + x_shape[self.axis :]
        return KerasTensor(output_shape, dtype=x.dtype)


def expand_dims(x, axis):
    if any_symbolic_tensors((x,)):
        return ExpandDims(axis=axis).symbolic_call(x)
    return backend.execute("expand_dims", x, axis)


class Expm1(Operation):
    def call(self, x):
        return backend.execute("expm1", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def expm1(x):
    if any_symbolic_tensors((x,)):
        return Expm1().symbolic_call(x)
    return backend.execute("expm1", x)


class Flip(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("flip", x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def flip(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Flip(axis=axis).symbolic_call(x)
    return backend.execute("flip", x, axis=axis)


class Floor(Operation):
    def call(self, x):
        return backend.execute("floor", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def floor(x):
    if any_symbolic_tensors((x,)):
        return Floor().symbolic_call(x)
    return backend.execute("floor", x)


class Full(Operation):
    def call(self, shape, fill_value, dtype=None):
        return backend.execute("full", shape, fill_value, dtype=dtype)

    def compute_output_spec(self, shape, fill_value, dtype=None):
        return KerasTensor(shape, dtype=dtype)


def full(shape, fill_value, dtype=None):
    return backend.execute("full", shape, fill_value, dtype=dtype)


class FullLike(Operation):
    def call(self, x, fill_value, dtype=None):
        return backend.execute("full_like", x, fill_value, dtype=dtype)

    def compute_output_spec(self, x, fill_value, dtype=None):
        return KerasTensor(x.shape, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    if any_symbolic_tensors((x,)):
        return FullLike().symbolic_call(x, fill_value, dtype=dtype)
    return backend.execute("full_like", x, fill_value, dtype=dtype)


class Greater(Operation):
    def call(self, x1, x2):
        return backend.execute("greater", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def greater(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Greater().symbolic_call(x1, x2)
    return backend.execute("greater", x1, x2)


class GreaterEqual(Operation):
    def call(self, x1, x2):
        return backend.execute("greater_equal", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def greater_equal(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return GreaterEqual().symbolic_call(x1, x2)
    return backend.execute("greater_equal", x1, x2)


class Hstack(Operation):
    def call(self, xs):
        return backend.execute("hstack", xs)

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
    return backend.execute("hstack", xs)


class Identity(Operation):
    def call(self, n, dtype="float32"):
        return backend.execute("identity", n, dtype=dtype)

    def compute_output_spec(self, n, dtype="float32"):
        return KerasTensor([n, n], dtype=dtype)


def identity(n, dtype="float32"):
    return backend.execute("identity", n, dtype=dtype)


class Imag(Operation):
    def call(self, x):
        return backend.execute("imag", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def imag(x):
    if any_symbolic_tensors((x,)):
        return Imag().symbolic_call(x)
    return backend.execute("imag", x)


class Isclose(Operation):
    def call(self, x1, x2):
        return backend.execute("isclose", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def isclose(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Isclose().symbolic_call(x1, x2)
    return backend.execute("isclose", x1, x2)


class Isfinite(Operation):
    def call(self, x):
        return backend.execute("isfinite", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isfinite(x):
    if any_symbolic_tensors((x,)):
        return Isfinite().symbolic_call(x)
    return backend.execute("isfinite", x)


class Isinf(Operation):
    def call(self, x):
        return backend.execute("isinf", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isinf(x):
    if any_symbolic_tensors((x,)):
        return Isinf().symbolic_call(x)
    return backend.execute("isinf", x)


class Isnan(Operation):
    def call(self, x):
        return backend.execute("isnan", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="bool")


def isnan(x):
    if any_symbolic_tensors((x,)):
        return Isnan().symbolic_call(x)
    return backend.execute("isnan", x)


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.execute("matmul", x1, x2)

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
    return backend.execute("matmul", x1, x2)


class Subtract(Operation):
    def call(self, x1, x2):
        return backend.execute("subtract", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def subtract(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.execute("subtract", x1, x2)


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.execute("multiply", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def multiply(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.execute("multiply", x1, x2)


class Divide(Operation):
    def call(self, x1, x2):
        return backend.execute("divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.execute("divide", x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.execute("true_divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def true_divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.execute("true_divide", x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.execute("power", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, "shape", [])
        x2_shape = getattr(x2, "shape", [])
        output_shape = broadcast_shapes(x1_shape, x2_shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def power(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.execute("power", x1, x2)


class Negative(Operation):
    def call(self, x):
        return backend.execute("negative", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def negative(x):
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.execute("negative", x)


class Square(Operation):
    def call(self, x):
        return backend.execute("square", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def square(x):
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.execute("square", x)


class Sqrt(Operation):
    def call(self, x):
        return backend.execute("sqrt", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def sqrt(x):
    if any_symbolic_tensors((x,)):
        return Sqrt().symbolic_call(x)
    return backend.execute("sqrt", x)


class Squeeze(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.execute("squeeze", x, axis=self.axis)

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
    return backend.execute("squeeze", x, axis=axis)


class Transpose(Operation):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes

    def call(self, x):
        return backend.execute("transpose", x, axes=self.axes)

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
    return backend.execute("transpose", x, axes=axes)


class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute(
            "mean", x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def mean(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Mean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("mean", x, axis=axis, keepdims=keepdims)


class Var(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute("var", x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def var(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Var(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("var", x, axis=axis, keepdims=keepdims)


class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute("sum", x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def sum(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("sum", x, axis=axis, keepdims=keepdims)


class Zeros(Operation):
    def call(self, shape, dtype="float32"):
        return backend.execute("zeros", shape, dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    return backend.execute("zeros", shape, dtype)


class Ones(Operation):
    def call(self, shape, dtype="float32"):
        return backend.execute("ones", shape, dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def ones(shape, dtype="float32"):
    return backend.execute("ones", shape, dtype)


class Eye(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.execute("eye", N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


def eye(N, M=None, k=0, dtype="float32"):
    return backend.execute("eye", N, M=M, k=k, dtype=dtype)
