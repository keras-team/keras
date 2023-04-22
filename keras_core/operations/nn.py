"""
relu
relu6
sigmoid
softplus
softsign
silu
swish
log_sigmoid
leaky_relu
hard_sigmoid
elu
selu
gelu
softmax
log_softmax

max_pooling
average_pooling
conv
depthwise_conv
separable_conv
conv_transpose

ctc ??
"""

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class Relu(Operation):
    def call(self, x):
        return backend.nn.relu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu(x):
    if any_symbolic_tensors((x,)):
        return Relu().symbolic_call(x)
    return backend.nn.relu(x)


class Relu6(Operation):
    def call(self, x):
        return backend.nn.relu6(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu6(x):
    if any_symbolic_tensors((x,)):
        return Relu6().symbolic_call(x)
    return backend.nn.relu6(x)


class Sigmoid(Operation):
    def call(self, x):
        return backend.nn.sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def sigmoid(x):
    if any_symbolic_tensors((x,)):
        return Sigmoid().symbolic_call(x)
    return backend.nn.sigmoid(x)


class Softplus(Operation):
    def call(self, x):
        return backend.nn.softplus(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softplus(x):
    if any_symbolic_tensors((x,)):
        return Softplus().symbolic_call(x)
    return backend.nn.softplus(x)


class Softsign(Operation):
    def call(self, x):
        return backend.nn.softsign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softsign(x):
    if any_symbolic_tensors((x,)):
        return Softsign().symbolic_call(x)
    return backend.nn.softsign(x)


class Silu(Operation):
    def call(self, x):
        return backend.nn.silu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def silu(x):
    if any_symbolic_tensors((x,)):
        return Silu().symbolic_call(x)
    return backend.nn.silu(x)


class Swish(Operation):
    def call(self, x):
        return backend.nn.swish(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def swish(x):
    if any_symbolic_tensors((x,)):
        return Swish().symbolic_call(x)
    return backend.nn.swish(x)


class LogSigmoid(Operation):
    def call(self, x):
        return backend.nn.log_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def log_sigmoid(x):
    if any_symbolic_tensors((x,)):
        return LogSigmoid().symbolic_call(x)
    return backend.nn.log_sigmoid(x)


class LeakyRelu(Operation):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def call(self, x):
        return backend.nn.leaky_relu(x, self.negative_slope)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def leaky_relu(x, negative_slope=0.2):
    if any_symbolic_tensors((x,)):
        return LeakyRelu(negative_slope).symbolic_call(x)
    return backend.nn.leaky_relu(x, negative_slope=negative_slope)


class HardSigmoid(Operation):
    def call(self, x):
        return backend.nn.hard_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def hard_sigmoid(x):
    if any_symbolic_tensors((x,)):
        return HardSigmoid().symbolic_call(x)
    return backend.nn.hard_sigmoid(x)


class Elu(Operation):
    def call(self, x):
        return backend.nn.elu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def elu(x):
    if any_symbolic_tensors((x,)):
        return Elu().symbolic_call(x)
    return backend.nn.elu(x)


class Selu(Operation):
    def call(self, x):
        return backend.nn.selu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def selu(x):
    if any_symbolic_tensors((x,)):
        return Selu().symbolic_call(x)
    return backend.nn.selu(x)


class Gelu(Operation):
    def __init__(self, approximate=True):
        super().__init__()
        self.approximate = approximate

    def call(self, x):
        return backend.nn.gelu(x, self.approximate)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def gelu(x, approximate=True):
    if any_symbolic_tensors((x,)):
        return Gelu(approximate).symbolic_call(x)
    return backend.nn.gelu(x, approximate)


class Softmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis if axis is not None else -1

    def call(self, x):
        return backend.nn.softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Softmax(axis).symbolic_call(x)
    return backend.nn.softmax(x)


class LogSoftmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis if axis is not None else -1

    def call(self, x):
        return backend.nn.log_softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def log_softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return LogSoftmax(axis).symbolic_call(x)
    return backend.nn.log_softmax(x, axis=axis)
