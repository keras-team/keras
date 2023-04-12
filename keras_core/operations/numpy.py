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
arctanh
argmax
argmin
argsort
array
array_equal
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
cov
cross
cumprod
cumsum
diag
diag_indices
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
indices
interp
isclose
isfinite
isin
isinf
isnan
isscalar
issubdtype
issubctype
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
        elif shape1[i] == None:
            output_shape[i] = shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] == None or shape2[i] == shape1[i]:
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


def shape_equal(shape1, shape2, axis=None):
    if len(shape1) != len(shape2):
        return False
    if axis is not None:
        shape1 = list(shape1)
        shape2 = list(shape2)
        for ax in axis:
            shape1[ax] = -1
            shape2[ax] = -1
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
        output_shape = broadcast_shapes(x1.shape, x2.shape)
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


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.execute("matmul", x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
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
        output_shape = broadcast_shapes(x1.shape, x2.shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def subtract(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.execute("subtract", x1, x2)


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.execute("multiply", x1, x2)

    def compute_output_spec(self, x1, x2):
        output_shape = broadcast_shapes(x1.shape, x2.shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def multiply(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.execute("multiply", x1, x2)


class Divide(Operation):
    def call(self, x1, x2):
        return backend.execute("divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        output_shape = broadcast_shapes(x1.shape, x2.shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.execute("divide", x1, x2)


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.execute("true_divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        output_shape = broadcast_shapes(x1.shape, x2.shape)
        return KerasTensor(output_shape, dtype=x1.dtype)


def true_divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.execute("true_divide", x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.execute("power", x1, x2)

    def compute_output_spec(self, x1, x2):
        output_shape = broadcast_shapes(x1.shape, x2.shape)
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


class Absolute(Operation):
    def call(self, x):
        return backend.execute("absolute", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def absolute(x):
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.execute("absolute", x)


class Square(Operation):
    def call(self, x):
        return backend.execute("square", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def square(x):
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.execute("square", x)


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
