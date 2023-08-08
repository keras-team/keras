"""Commonly-used neural network operations not included in NumPy."""

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.backend import standardize_data_format
from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras_core.ops import operation_utils
from keras_core.ops.operation import Operation


class Relu(Operation):
    def call(self, x):
        return backend.nn.relu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.relu", "keras_core.ops.nn.relu"])
def relu(x):
    """Rectified linear unit activation function.

    It is defined as `f(x) = max(0, x)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_relu = keras_core.ops.relu(x)
    >>> print(x_relu)
    array([0., 0., 1.], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Relu().symbolic_call(x)
    return backend.nn.relu(x)


class Relu6(Operation):
    def call(self, x):
        return backend.nn.relu6(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.relu6", "keras_core.ops.nn.relu6"])
def relu6(x):
    """Rectified linear unit activation function with upper bound of 6.

    It is defined as `f(x) = np.clip(x, 0, 6)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1., 6., 7.])
    >>> x_relu6 = keras_core.ops.relu6(x)
    >>> print(x_relu6)
    array([0., 0., 1., 6., 6.], shape=(5,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Relu6().symbolic_call(x)
    return backend.nn.relu6(x)


class Sigmoid(Operation):
    def call(self, x):
        return backend.nn.sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.sigmoid", "keras_core.ops.nn.sigmoid"])
def sigmoid(x):
    """Sigmoid activation function.

    It is defined as `f(x) = 1 / (1 + exp(-x))`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_sigmoid = keras_core.ops.sigmoid(x)
    >>> print(x_sigmoid)
    array([0.26894143, 0.5, 0.7310586 ], dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Sigmoid().symbolic_call(x)
    return backend.nn.sigmoid(x)


class Softplus(Operation):
    def call(self, x):
        return backend.nn.softplus(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.softplus", "keras_core.ops.nn.softplus"])
def softplus(x):
    """Softplus activation function.

    It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
    logarithm and `exp` is the exponential function.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_softplus = keras_core.ops.softplus(x)
    >>> print(x_softplus)
    array([0.31326166, 0.6931472 , 1.3132616 ], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Softplus().symbolic_call(x)
    return backend.nn.softplus(x)


class Softsign(Operation):
    def call(self, x):
        return backend.nn.softsign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.softsign", "keras_core.ops.nn.softsign"])
def softsign(x):
    """Softsign activation function.

    It is defined as `f(x) = x / (abs(x) + 1)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_softsign = keras_core.ops.softsign(x)
    >>> print(x_softsign)
    array([-0.5, 0., 0.5], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Softsign().symbolic_call(x)
    return backend.nn.softsign(x)


class Silu(Operation):
    def call(self, x):
        return backend.nn.silu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.silu", "keras_core.ops.nn.silu"])
def silu(x):
    """Sigmoid-weighted linear unit activation function.

    It is defined as `f(x) = x * sigmoid(x)`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_silu = keras_core.ops.silu(x)
    >>> print(x_silu)
    array([-0.26894143, 0., 0.7310586], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Silu().symbolic_call(x)
    return backend.nn.silu(x)


class Swish(Operation):
    def call(self, x):
        return backend.nn.swish(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.swish", "keras_core.ops.nn.swish"])
def swish(x):
    if any_symbolic_tensors((x,)):
        return Swish().symbolic_call(x)
    return backend.nn.swish(x)


class LogSigmoid(Operation):
    def call(self, x):
        return backend.nn.log_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.log_sigmoid",
        "keras_core.ops.nn.log_sigmoid",
    ]
)
def log_sigmoid(x):
    """Logarithm of the sigmoid activation function.

    It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_log_sigmoid = keras_core.ops.log_sigmoid(x)
    >>> print(x_log_sigmoid)
    array([-1.3132616, -0.6931472, -0.3132616], shape=(3,), dtype=float64)

    """
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
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    ["keras_core.ops.leaky_relu", "keras_core.ops.nn.leaky_relu"]
)
def leaky_relu(x, negative_slope=0.2):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active, it is defined as:

    `f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.

    Args:
        x: Input tensor.
        negative_slope: Slope of the activation function at x < 0.
            Defaults to `0.2`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_leaky_relu = keras_core.ops.leaky_relu(x)
    >>> print(x_leaky_relu)
    array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return LeakyRelu(negative_slope).symbolic_call(x)
    return backend.nn.leaky_relu(x, negative_slope=negative_slope)


class HardSigmoid(Operation):
    def call(self, x):
        return backend.nn.hard_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.hard_sigmoid",
        "keras_core.ops.nn.hard_sigmoid",
    ]
)
def hard_sigmoid(x):
    """Hard sigmoid activation function.

    It is defined as:

    `0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_hard_sigmoid = keras_core.ops.hard_sigmoid(x)
    >>> print(x_hard_sigmoid)
    array([0.3, 0.5, 0.7], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return HardSigmoid().symbolic_call(x)
    return backend.nn.hard_sigmoid(x)


class Elu(Operation):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def call(self, x):
        return backend.nn.elu(x, alpha=self.alpha)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.elu", "keras_core.ops.nn.elu"])
def elu(x, alpha=1.0):
    """Exponential Linear Unit.

    It is defined as:

    `f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.

    Args:
        x: Input tensor.
        alpha: A scalar, slope of positive section. Defaults to `1.0`.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_elu = keras_core.ops.elu(x)
    >>> print(x_elu)
    array([-0.63212055, 0., 1.], shape=(3,), dtype=float64)

    """
    if any_symbolic_tensors((x,)):
        return Elu(alpha).symbolic_call(x)
    return backend.nn.elu(x, alpha=alpha)


class Selu(Operation):
    def call(self, x):
        return backend.nn.selu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.selu", "keras_core.ops.nn.selu"])
def selu(x):
    """Scaled Exponential Linear Unit (SELU).

    It is defined as:

    `f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = scale * x for x >= 0`.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same shape as `x`.

    Example:

    >>> x = np.array([-1., 0., 1.])
    >>> x_selu = keras_core.ops.selu(x)
    >>> print(x_selu)
    array([-1.11133055, 0., 1.05070098], shape=(3,), dtype=float64)

    """
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
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.gelu", "keras_core.ops.nn.gelu"])
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
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(["keras_core.ops.softmax", "keras_core.ops.nn.softmax"])
def softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Softmax(axis).symbolic_call(x)
    return backend.nn.softmax(x, axis=axis)


class LogSoftmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis if axis is not None else -1

    def call(self, x):
        return backend.nn.log_softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_core_export(
    [
        "keras_core.ops.log_softmax",
        "keras_core.ops.nn.log_softmax",
    ]
)
def log_softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return LogSoftmax(axis).symbolic_call(x)
    return backend.nn.log_softmax(x, axis=axis)


class MaxPool(Operation):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def call(self, inputs):
        return backend.nn.max_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        output_shape = operation_utils.compute_pooling_output_shape(
            inputs.shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(["keras_core.ops.max_pool", "keras_core.ops.nn.max_pool"])
def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Max pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        pool_size: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`, specifying the size of the pooling
            window for each spatial dimension of the input tensor. If
            `pool_size` is int, then every spatial dimension shares the same
            `pool_size`.
        strides: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`. The stride of the sliding window for
            each spatial dimension of the input tensor. If `strides` is int,
            then every spatial dimension shares the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.

    Returns:
        A tensor of rank N+2, the result of the max pooling operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return MaxPool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(inputs)
    return backend.nn.max_pool(inputs, pool_size, strides, padding, data_format)


class AveragePool(Operation):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def call(self, inputs):
        return backend.nn.average_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        output_shape = operation_utils.compute_pooling_output_shape(
            inputs.shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(
    [
        "keras_core.ops.average_pool",
        "keras_core.ops.nn.average_pool",
    ]
)
def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Average pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,)` + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        pool_size: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`, specifying the size of the pooling
            window for each spatial dimension of the input tensor. If
            `pool_size` is int, then every spatial dimension shares the same
            `pool_size`.
        strides: int or tuple/list of integers of size
            `len(inputs_spatial_shape)`. The stride of the sliding window for
            each spatial dimension of the input tensor. If `strides` is int,
            then every spatial dimension shares the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.

    Returns:
        A tensor of rank N+2, the result of the average pooling operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return AveragePool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(inputs)
    return backend.nn.average_pool(
        inputs, pool_size, strides, padding, data_format
    )


class Conv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, kernel):
        return backend.nn.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        output_shape = operation_utils.compute_conv_output_shape(
            inputs.shape,
            kernel.shape[-1],
            kernel.shape[:-2],
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(["keras_core.ops.conv", "keras_core.ops.nn.conv"])
def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D convolution.

    This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            `(kernel_spatial_shape, num_input_channels, num_output_channels)`.
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return Conv(strides, padding, data_format, dilation_rate).symbolic_call(
            inputs, kernel
        )
    return backend.nn.conv(
        inputs, kernel, strides, padding, data_format, dilation_rate
    )


class DepthwiseConv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, kernel):
        return backend.nn.depthwise_conv(
            inputs,
            kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        output_shape = operation_utils.compute_conv_output_shape(
            inputs.shape,
            kernel.shape[-1] * kernel.shape[-2],
            kernel.shape[:-2],
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(
    [
        "keras_core.ops.depthwise_conv",
        "keras_core.ops.nn.depthwise_conv",
    ]
)
def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D depthwise convolution.

    This ops supports 1D and 2D depthwise convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,)` + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return DepthwiseConv(
            strides, padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return backend.nn.depthwise_conv(
        inputs,
        kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )


class SeparableConv(Operation):
    def __init__(
        self,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, depthwise_kernel, pointwise_kernel):
        return backend.nn.separable_conv(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, depthwise_kernel, pointwise_kernel):
        output_shape = list(
            depthwise_conv(
                inputs,
                depthwise_kernel,
                self.strides,
                self.padding,
                self.data_format,
                self.dilation_rate,
            ).shape
        )
        if self.data_format == "channels_last":
            output_shape[-1] = pointwise_kernel.shape[-1]
        else:
            output_shape[1] = pointwise_kernel.shape[-1]
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(
    [
        "keras_core.ops.separable_conv",
        "keras_core.ops.nn.separable_conv",
    ]
)
def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """General N-D separable convolution.

    This ops supports 1D and 2D separable convolution. `separable_conv` is
    a depthwise conv followed by a pointwise conv.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,)` + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        depthwise_kernel: Tensor of rank N+2. `depthwise_kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        pointwise_kernel: Tensor of rank N+2. `pointwise_kernel` has shape
            `(*ones_like(kernel_spatial_shape),
            num_input_channels * num_channels_multiplier, num_output_channels)`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return SeparableConv(
            strides,
            padding,
            data_format,
            dilation_rate,
        ).symbolic_call(inputs, depthwise_kernel, pointwise_kernel)
    return backend.nn.separable_conv(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )


class ConvTranspose(Operation):
    def __init__(
        self,
        strides,
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=1,
    ):
        super().__init__()
        self.strides = strides
        self.output_padding = output_padding
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(
        self,
        inputs,
        kernel,
    ):
        return backend.nn.conv_transpose(
            inputs,
            kernel,
            self.strides,
            self.output_padding,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        kernel_size = kernel.shape[:-2]
        filters = kernel.shape[-2]
        output_shape = compute_conv_transpose_output_shape(
            inputs.shape,
            kernel_size,
            filters,
            self.strides,
            self.padding,
            self.output_padding,
            self.data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


@keras_core_export(
    [
        "keras_core.ops.conv_transpose",
        "keras_core.ops.nn.conv_transpose",
    ]
)
def conv_transpose(
    inputs,
    kernel,
    strides,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    """General N-D convolution transpose.

    Also known as de-convolution. This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,)` + inputs_spatial_shape + (num_channels,)` if
            `data_format="channels_last"`, or
            `(batch_size, num_channels) + inputs_spatial_shape` if
            `data_format="channels_first"`.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_output_channels, num_input_channels],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        output_padding: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the amount of padding along the height and width of
            the output tensor. Can be a single integer to specify the same
            value for all spatial dimensions. The amount of output padding
            along a given dimension must be lower than the stride along that
            same dimension. If set to `None` (default), the output shape is
            inferred.
        data_format: A string, either `"channels_last"` or `"channels_first"`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, `inputs` is of shape
            `(batch_size, ..., channels)` while if
            `data_format="channels_first"`, `inputs` is of shape
            `(batch_size, channels, ...)`.
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
    data_format = standardize_data_format(data_format)
    if any_symbolic_tensors((inputs,)):
        return ConvTranspose(
            strides, padding, output_padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return backend.nn.conv_transpose(
        inputs,
        kernel,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )


class OneHot(Operation):
    def __init__(self, num_classes, axis=-1, dtype=None):
        super().__init__()
        self.num_classes = num_classes
        self.axis = axis
        self.dtype = dtype or backend.floatx()

    def call(self, x):
        return backend.nn.one_hot(
            x, self.num_classes, axis=self.axis, dtype=self.dtype
        )

    def compute_output_spec(self, x):
        x_shape = list(getattr(x, "shape", []))
        if self.axis == -1:
            x_shape.append(self.num_classes)
        elif self.axis >= 0 and self.axis < len(x_shape):
            x_shape.insert(self.axis, self.num_classes)
        else:
            raise ValueError(
                f"axis must be -1 or between [0, {len(x.shape)}), but "
                f"received {self.axis}."
            )
        return KerasTensor(x_shape, dtype=self.dtype)


@keras_core_export(["keras_core.ops.one_hot", "keras_core.ops.nn.one_hot"])
def one_hot(x, num_classes, axis=-1, dtype=None):
    """Converts integer tensor `x` into a one-hot tensor.

    The one-hot encoding is a representation where each integer value is
    converted into a binary vector with a length equal to `num_classes`,
    and the index corresponding to the integer value is marked as 1, while
    all other indices are marked as 0.

    Args:
        x : Integer tensor to be encoded. The shape can be
            arbitrary, but the dtype should be integer.
        num_classes: Number of classes for the one-hot encoding.
        axis: Axis along which the encoding is performed. Default is
            -1, which represents the last axis.
        dtype: (Optional) Data type of the output tensor. If not
            provided, it defaults to the default data type of the backend.

    Returns:
        Integer tensor: One-hot encoded tensor with the same shape as `x`
        except for the specified `axis` dimension, which will have
        a length of `num_classes`. The dtype of the output tensor
        is determined by `dtype` or the default data type of the backend.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([1, 3, 2, 0])
    >>> one_hot(x, num_classes=4)
    array([[0. 1. 0. 0.]
           [0. 0. 0. 1.]
           [0. 0. 1. 0.]
           [1. 0. 0. 0.]], shape=(4, 4), dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return OneHot(num_classes, axis=axis, dtype=dtype).symbolic_call(x)
    return backend.nn.one_hot(
        x, num_classes, axis=axis, dtype=dtype or backend.floatx()
    )


class BinaryCrossentropy(Operation):
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def call(self, target, output):
        return backend.nn.binary_crossentropy(
            target, output, from_logits=self.from_logits
        )

    def compute_output_spec(self, target, output):
        if target.shape != output.shape:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        return KerasTensor(output.shape, dtype=output.dtype)


@keras_core_export(
    [
        "keras_core.ops.binary_crossentropy",
        "keras_core.ops.nn.binary_crossentropy",
    ]
)
def binary_crossentropy(target, output, from_logits=False):
    """Computes binary cross-entropy loss between target and output tensor.

    The binary cross-entropy loss is commonly used in binary
    classification tasks where each input sample belongs to one
    of the two classes. It measures the dissimilarity between the
    target and output probabilities or logits.

    Args:
        target: The target tensor representing the true binary labels.
            Its shape should match the shape of the `output` tensor.
        output: The output tensor representing the predicted probabilities
            or logits. Its shape should match the shape of the
            `target` tensor.
        from_logits: (optional) Whether `output` is a tensor of logits or
            probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Default is `False`.

    Returns:
        Integer tensor: The computed binary cross-entropy loss between
        `target` and `output`.

    Example:

    >>> target = keras_core.ops.convert_to_tensor([0, 1, 1, 0])
    >>> output = keras_core.ops.convert_to_tensor([0.1, 0.9, 0.8, 0.2])
    >>> binary_crossentropy(target, output)
    array([0.10536054 0.10536054 0.22314355 0.22314355],
          shape=(4,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return BinaryCrossentropy(from_logits=from_logits).symbolic_call(
            target, output
        )
    return backend.nn.binary_crossentropy(
        target, output, from_logits=from_logits
    )


class CategoricalCrossentropy(Operation):
    def __init__(self, from_logits=False, axis=-1):
        super().__init__()
        self.from_logits = from_logits
        self.axis = axis

    def call(self, target, output):
        return backend.nn.categorical_crossentropy(
            target, output, from_logits=self.from_logits, axis=self.axis
        )

    def compute_output_spec(self, target, output):
        if target.shape != output.shape:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        if len(target.shape) < 1:
            raise ValueError(
                "Arguments `target` and `output` must be at least rank 1. "
                "Received: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        return KerasTensor(output.shape[:-1], dtype=output.dtype)


@keras_core_export(
    [
        "keras_core.ops.categorical_crossentropy",
        "keras_core.ops.nn.categorical_crossentropy",
    ]
)
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Computes categorical cross-entropy loss between target and output tensor.

    The categorical cross-entropy loss is commonly used in multi-class
    classification tasks where each input sample can belong to one of
    multiple classes. It measures the dissimilarity
    between the target and output probabilities or logits.

    Args:
        target: The target tensor representing the true categorical labels.
            Its shape should match the shape of the `output` tensor
            except for the last dimension.
        output: The output tensor representing the predicted probabilities
            or logits. Its shape should match the shape of the `target`
            tensor except for the last dimension.
        from_logits: (optional) Whether `output` is a tensor of logits or
            probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Default is `False`.
        axis: (optional) The axis along which the categorical cross-entropy
            is computed.
            Default is -1, which corresponds to the last dimension of
            the tensors.

    Returns:
        Integer tensor: The computed categorical cross-entropy loss between
        `target` and `output`.

    Example:

    >>> target = keras_core.ops.convert_to_tensor(
    ... [[1, 0, 0],
    ...  [0, 1, 0],
    ...  [0, 0, 1]])
    >>> output = keras_core.ops.convert_to_tensor(
    ... [[0.9, 0.05, 0.05],
    ...  [0.1, 0.8, 0.1],
    ...  [0.2, 0.3, 0.5]])
    >>> categorical_crossentropy(target, output)
    array([0.10536054 0.22314355 0.6931472 ], shape=(3,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return CategoricalCrossentropy(
            from_logits=from_logits, axis=axis
        ).symbolic_call(target, output)
    return backend.nn.categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis
    )


class SparseCategoricalCrossentropy(Operation):
    def __init__(self, from_logits=False, axis=-1):
        super().__init__()
        self.from_logits = from_logits
        self.axis = axis

    def call(self, target, output):
        return backend.nn.sparse_categorical_crossentropy(
            target, output, from_logits=self.from_logits, axis=self.axis
        )

    def compute_output_spec(self, target, output):
        if len(output.shape) < 1:
            raise ValueError(
                "Argument `output` must be at least rank 1. "
                "Received: "
                f"output.shape={output.shape}"
            )
        target_shape = target.shape
        if len(target_shape) == len(output.shape) and target_shape[-1] == 1:
            target_shape = target_shape[:-1]
        if target_shape != output.shape[:-1]:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape "
                "up until the last dimension: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        return KerasTensor(output.shape[:-1], dtype=output.dtype)


@keras_core_export(
    [
        "keras_core.ops.sparse_categorical_crossentropy",
        "keras_core.ops.nn.sparse_categorical_crossentropy",
    ]
)
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Computes sparse categorical cross-entropy loss.

    The sparse categorical cross-entropy loss is similar to categorical
    cross-entropy, but it is used when the target tensor contains integer
    class labels instead of one-hot encoded vectors. It measures the
    dissimilarity between the target and output probabilities or logits.

    Args:
        target: The target tensor representing the true class labels as
            integers. Its shape should match the shape of the `output`
            tensor except for the last dimension.
        output: The output tensor representing the predicted probabilities
            or logits.
            Its shape should match the shape of the `target` tensor except
            for the last dimension.
        from_logits: (optional) Whether `output` is a tensor of logits
            or probabilities.
            Set it to `True` if `output` represents logits; otherwise,
            set it to `False` if `output` represents probabilities.
            Default is `False`.
        axis: (optional) The axis along which the sparse categorical
            cross-entropy is computed.
            Default is -1, which corresponds to the last dimension
            of the tensors.

    Returns:
        Integer tensor: The computed sparse categorical cross-entropy
        loss between `target` and `output`.

    Example:

    >>> target = keras_core.ops.convert_to_tensor([0, 1, 2], dtype=int32)
    >>> output = keras_core.ops.convert_to_tensor(
    ... [[0.9, 0.05, 0.05],
    ...  [0.1, 0.8, 0.1],
    ...  [0.2, 0.3, 0.5]])
    >>> sparse_categorical_crossentropy(target, output)
    array([0.10536056 0.22314355 0.6931472 ], shape=(3,), dtype=float32)
    """
    if any_symbolic_tensors((target, output)):
        return SparseCategoricalCrossentropy(
            from_logits=from_logits, axis=axis
        ).symbolic_call(target, output)
    return backend.nn.sparse_categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis
    )


class MultiHot(Operation):
    def __init__(self, num_tokens=None, axis=-1, dtype=None, name=None):
        super().__init__(name)
        self.num_tokens = num_tokens
        self.axis = axis
        self.dtype = dtype or backend.floatx()

    def call(self, inputs):
        return backend.nn.multi_hot(
            inputs,
            num_classes=self.num_tokens,
            axis=self.axis,
            dtype=self.dtype,
        )

    def compute_output_spec(self, inputs):
        x_shape = list(getattr(inputs, "shape", []))
        if self.axis == -1:
            x_shape.append(self.num_tokens)
        elif self.axis >= 0 and self.axis < len(x_shape):
            x_shape.insert(self.axis, self.num_tokens)
        else:
            raise ValueError(
                f"axis must be -1 or between [0, {len(inputs.shape)}), but "
                f"received {self.axis}."
            )

        if len(x_shape) == 2:
            x_shape = [x_shape[-1]]
        else:
            x_shape = [x_shape[0]] + x_shape[2:]

        return KerasTensor(x_shape, dtype=inputs.dtype)


@keras_core_export(
    [
        "keras_core.ops.multi_hot",
        "keras_core.ops.nn.multi_hot",
    ]
)
def multi_hot(inputs, num_tokens, axis=-1, dtype=None):
    if any_symbolic_tensors((inputs,)):
        return MultiHot(num_tokens, axis, dtype).symbolic_call(inputs)

    return backend.nn.multi_hot(inputs, num_tokens, axis, dtype)
