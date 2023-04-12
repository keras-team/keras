"""
MANIFEST:

matmul
add
subtract
multiply
divide
true_divide
power
negative
absolute
mean
var
zeros
ones


"""
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.backend import convert_to_tensor
from keras_core.operations.symbolic_arguments import SymbolicArguments
from keras_core.operations.operation import Operation
from keras_core import backend

from tensorflow import nest
import jax


# TODO: replace this function with one that can handle
# dynamic shapes.
def compute_np_output_spec(op_name, *args, **kwargs):
    op = getattr(jax.numpy, op_name)

    def convert_keras_tensor_to_jax_array(x):
        if isinstance(x, KerasTensor):
            return jax.numpy.zeros(x.shape, dtype=x.dtype)
        return x

    args, kwargs = SymbolicArguments(*args, **kwargs).convert(
        convert_keras_tensor_to_jax_array
    )
    jax_out = jax.eval_shape(op, *args, **kwargs)

    def convert_jax_spec_to_keras_tensor(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return KerasTensor(x.shape, x.dtype)
        return x

    return nest.map_structure(convert_jax_spec_to_keras_tensor, jax_out)


#####################
### Two-input ops ###
#####################


### matmul ###


class Matmul(Operation):
    def call(self, x1, x2):
        return backend.execute("matmul", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("matmul", x1, x2)


def matmul(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Matmul().symbolic_call(x1, x2)
    x1 = convert_to_tensor(x1, x1.dtype)
    x2 = convert_to_tensor(x2, x2.dtype)
    return backend.execute("matmul", x1, x2)


### add ###


class Add(Operation):
    def call(self, x1, x2):
        return backend.execute("add", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("add", x1, x2)


def add(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Add().symbolic_call(x1, x2)
    return backend.execute("add", x1, x2)


### subtract ###


class Subtract(Operation):
    def call(self, x1, x2):
        return backend.execute("subtract", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("subtract", x1, x2)


def subtract(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Subtract().symbolic_call(x1, x2)
    return backend.execute("subtract", x1, x2)


### multiply ###


class Multiply(Operation):
    def call(self, x1, x2):
        return backend.execute("multiply", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("multiply", x1, x2)


def multiply(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Multiply().symbolic_call(x1, x2)
    return backend.execute("multiply", x1, x2)


### divide ###


class Divide(Operation):
    def call(self, x1, x2):
        return backend.execute("divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("divide", x1, x2)


def divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Divide().symbolic_call(x1, x2)
    return backend.execute("divide", x1, x2)


### true_divide ###


class TrueDivide(Operation):
    def call(self, x1, x2):
        return backend.execute("true_divide", x1, x2)

    def compute_output_spec(self, x1, x2):
        return compute_np_output_spec("true_divide", x1, x2)


def true_divide(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return TrueDivide().symbolic_call(x1, x2)
    return backend.execute("true_divide", x1, x2)


class Power(Operation):
    def call(self, x1, x2):
        return backend.execute("power", x1, x2)

    def compute_output_spec(self, x1, x2):
        return KerasTensor(x1.shape, dtype=x1.dtype)


def power(x1, x2):
    if any_symbolic_tensors((x1, x2)):
        return Power().symbolic_call(x1, x2)
    return backend.execute("power", x1, x2)


########################
### Single-input ops ###
########################

### negative ###


class Negative(Operation):
    def call(self, x):
        return backend.execute("negative", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def negative(x):
    if any_symbolic_tensors((x,)):
        return Negative().symbolic_call(x)
    return backend.execute("negative", x)


### absolute ###


class Absolute(Operation):
    def call(self, x):
        return backend.execute("absolute", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def absolute(x):
    if any_symbolic_tensors((x,)):
        return Absolute().symbolic_call(x)
    return backend.execute("absolute", x)


### square ###


class Square(Operation):
    def call(self, x):
        return backend.execute("square", x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


def square(x):
    if any_symbolic_tensors((x,)):
        return Square().symbolic_call(x)
    return backend.execute("square", x)


#####################
### Reshaping ops ###
#####################


### squeeze ###


class Squeeze(Operation):
    def __init__(self, axis=None):
        self.axis = axis

    def call(self, a):
        return backend.execute("squeeze", a, axis=self.axis)

    def compute_output_spec(self, a):
        return compute_np_output_spec("squeeze", a, axis=self.axis)


def squeeze(a, axis=None):
    if any_symbolic_tensors((a,)):
        return Squeeze().symbolic_call(a, axis=axis)
    return backend.execute("squeeze", a, axis=axis)


### transpose ###


class Transpose(Operation):
    def __init__(self, axes=None):
        self.axes = axes

    def call(self, a):
        return backend.execute("transpose", a, axes=self.axes)

    def compute_output_spec(self, a):
        return compute_np_output_spec("transpose", a, axes=self.axes)


def transpose(a, axes=None):
    if any_symbolic_tensors((a,)):
        return Transpose().symbolic_call(a, axes=axes)
    return backend.execute("transpose", a, axes=axes)


#####################
### Reduction ops ###
#####################


class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute(
            "mean", x, axis=self.axis, keepdims=self.keepdims
        )

    def compute_output_spec(self, x):
        return compute_np_output_spec(
            "mean", x, axis=self.axis, keepdims=self.keepdims
        )


def mean(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Mean(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("mean", x, axis=axis, keepdims=keepdims)


class Var(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute("var", x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return compute_np_output_spec(
            "var", x, axis=self.axis, keepdims=self.keepdims
        )


def var(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Var(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("var", x, axis=axis, keepdims=keepdims)


class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.execute("sum", x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return compute_np_output_spec(
            "sum", x, axis=self.axis, keepdims=self.keepdims
        )


def sum(x, axis=None, keepdims=False):
    if any_symbolic_tensors((x,)):
        return Sum(axis=axis, keepdims=keepdims).symbolic_call(x)
    return backend.execute("sum", x, axis=axis, keepdims=keepdims)


##########################
### Array creation ops ###
##########################


### zeros ###


class Zeros(Operation):
    def call(self, shape, dtype="float32"):
        return backend.execute("zeros", shape, dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    return backend.execute("zeros", shape, dtype)


### ones ###


class Ones(Operation):
    def call(self, shape, dtype="float32"):
        return backend.execute("ones", shape, dtype)

    def compute_output_spec(self, shape, dtype="float32"):
        return KerasTensor(shape, dtype=dtype)


def ones(shape, dtype="float32"):
    return backend.execute("ones", shape, dtype)


### eye ###


class Eye(Operation):
    def call(self, N, M=None, k=0, dtype="float32"):
        return backend.execute("eye", N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype="float32"):
        if M is None:
            M = N
        return KerasTensor((N, M), dtype=dtype)


def eye(N, M=None, k=0, dtype="float32"):
    return backend.execute("eye", N, M=M, k=k, dtype=dtype)
