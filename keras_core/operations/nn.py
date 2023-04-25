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

import numpy as np

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
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


class MaxPool(Operation):
    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format="channels_last",
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
        strides = self.pool_size if self.strides is None else self.strides
        input_shape = np.array(inputs.shape)
        if self.data_format == "channels_last":
            spatial_shape = input_shape[1:-1]
        else:
            spatial_shape = input_shape[2:]
        pool_size = np.array(self.pool_size)
        if self.padding == "valid":
            output_spatial_shape = (
                np.floor((spatial_shape - self.pool_size) / strides) + 1
            )
            negative_in_shape = np.all(output_spatial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
                )
        elif self.padding == "same":
            output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
        output_spatial_shape = [int(i) for i in output_spatial_shape]
        if self.data_format == "channels_last":
            output_shape = (
                [inputs.shape[0]]
                + list(output_spatial_shape)
                + [inputs.shape[-1]]
            )
        else:
            output_shape = inputs.shape[:2] + list(output_spatial_shape)
        return KerasTensor(output_shape, dtype=inputs.dtype)


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    """Max pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
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
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).

    Returns:
        A tensor of rank N+2, the result of the max pooling operation.
    """
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
        data_format="channels_last",
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
        strides = self.pool_size if self.strides is None else self.strides
        input_shape = np.array(inputs.shape)
        if self.data_format == "channels_last":
            spatial_shape = input_shape[1:-1]
        else:
            spatial_shape = input_shape[2:]
        pool_size = np.array(self.pool_size)
        if self.padding == "valid":
            output_spatial_shape = (
                np.floor((spatial_shape - self.pool_size) / strides) + 1
            )
            negative_in_shape = np.all(output_spatial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
                )
        elif self.padding == "same":
            output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
        output_spatial_shape = [int(i) for i in output_spatial_shape]
        if self.data_format == "channels_last":
            output_shape = (
                [inputs.shape[0]] + output_spatial_shape + [inputs.shape[-1]]
            )
        else:
            output_shape = inputs.shape[:2] + output_spatial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    """Average pooling operation.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
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
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).

    Returns:
        A tensor of rank N+2, the result of the average pooling operation.
    """
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
        data_format="channel_last",
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
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
        )

    def compute_output_spec(self, inputs, kernel):
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            spatial_shape = input_shape[1:-1]
        else:
            spatial_shape = input_shape[2:]
        if len(kernel.shape) != len(input_shape):
            raise ValueError(
                "Kernel shape must have the same length as input, but received "
                f"kernel of shape {kernel.shape} and "
                f"input of shape {input_shape}."
            )
        if isinstance(self.dilation_rate, int):
            dilation_rate = (self.dilation_rate,) * len(spatial_shape)
        else:
            dilation_rate = self.dilation_rate
        if len(dilation_rate) != len(spatial_shape):
            raise ValueError(
                "Dilation must be None, scalar or tuple/list of length of "
                "inputs' spatial shape, but received "
                f"`dilation_rate={self.dilation_rate}` and "
                f"input of shape {input_shape}."
            )
        spatial_shape = np.array(spatial_shape)
        kernel_spatial_shape = np.array(kernel.shape[:-2])
        dilation_rate = np.array(dilation_rate)
        if self.padding == "valid":
            output_spatial_shape = (
                np.floor(
                    (
                        spatial_shape
                        - dilation_rate * (kernel_spatial_shape - 1)
                        - 1
                    )
                    / self.strides
                )
                + 1
            )
            negative_in_shape = np.all(output_spatial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={inputs.shape}`, "
                    f"`kernel spatial size={kernel.size}`, "
                    f"`dilation_rate={self.dilation_rate}`."
                )
        elif self.padding == "same":
            output_spatial_shape = (
                np.floor((spatial_shape - 1) / self.strides) + 1
            )
        output_spatial_shape = [int(i) for i in output_spatial_shape]
        if self.data_format == "channels_last":
            output_shape = (
                [inputs.shape[0]] + output_spatial_shape + [kernel.shape[-1]]
            )
        else:
            output_shape = [
                inputs.shape[0],
                kernel.shape[-1],
            ] + output_spatial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """General N-D convolution.

    This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_output_channels],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
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
        data_format="channels_last",
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
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            spatial_shape = input_shape[1:-1]
        else:
            spatial_shape = input_shape[2:]
        if len(kernel.shape) != len(inputs.shape):
            raise ValueError(
                "Kernel shape must have the same length as input, but received "
                f"kernel of shape {kernel.shape} and "
                f"input of shape {input_shape}."
            )
        if isinstance(self.dilation_rate, int):
            dilation_rate = (self.dilation_rate,) * len(spatial_shape)
        else:
            dilation_rate = self.dilation_rate
        if len(dilation_rate) != len(spatial_shape):
            raise ValueError(
                "Dilation must be None, scalar or tuple/list of length of "
                "inputs' spatial shape, but received "
                f"`dilation_rate={self.dilation_rate}` and input of "
                f"shape {input_shape}."
            )
        spatial_shape = np.array(spatial_shape)
        kernel_spatial_shape = np.array(kernel.shape[:-2])
        dilation_rate = np.array(self.dilation_rate)
        if self.padding == "valid":
            output_spatial_shape = (
                np.floor(
                    (
                        spatial_shape
                        - dilation_rate * (kernel_spatial_shape - 1)
                        - 1
                    )
                    / self.strides
                )
                + 1
            )
            negative_in_shape = np.all(output_spatial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={inputs.shape}`, "
                    f"`kernel spatial size={kernel.size}`, "
                    f"`dilation_rate={self.dilation_rate}`."
                )
        elif self.padding == "same":
            output_spatial_shape = (
                np.floor((spatial_shape - 1) / self.strides) + 1
            )

        output_spatial_shape = [int(i) for i in output_spatial_shape]
        output_channels = kernel.shape[-1] * kernel.shape[-2]
        if self.data_format == "channels_last":
            output_shape = (
                [input_shape[0]] + output_spatial_shape + [output_channels]
            )
        else:
            output_shape = [
                input_shape[0],
                output_channels,
            ] + output_spatial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """General N-D depthwise convolution.

    This ops supports 1D and 2D depthwise convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
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
        data_format="channels_last",
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


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """General N-D separable convolution.

    This ops supports 1D and 2D separable convolution. `separable_conv` is
    a depthwise conv followed by a pointwise conv.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        depthwise_kernel: Tensor of rank N+2. `depthwise_kernel` has shape
            [kernel_spatial_shape, num_input_channels, num_channels_multiplier],
            `num_input_channels` should match the number of channels in
            `inputs`.
        pointwise_kernel: Tensor of rank N+2. `pointwise_kernel` has shape
            [ones_like(kernel_spatial_shape),
            num_input_channels * num_channels_multiplier, num_output_channels].
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the depthwise conv operation.
    """
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
        data_format="channels_last",
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
        output_shape = compute_conv_transpose_output_shape(
            inputs,
            kernel,
            self.strides,
            self.padding,
            self.output_padding,
            self.data_format,
            self.dilation_rate,
        )
        return KerasTensor(output_shape, dtype=inputs.dtype)


def conv_transpose(
    inputs,
    kernel,
    strides,
    padding="valid",
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    """General N-D convolution transpose.

    Also known as de-convolution. This ops supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            [batch_size] + inputs_spatial_shape + [num_channels] if
            `data_format="channels_last"`, or
            [batch_size, num_channels] + inputs_spatial_shape if
            `data_format="channels_first"`. Pooling happens over the spatial
            dimensions only.
        kernel: Tensor of rank N+2. `kernel` has shape
            [kernel_spatial_shape, num_output_channels, num_input_channels],
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and "same" results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        output_padding: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the amount of padding along the height and width of
            the output tensor. Can be a single integer to specify the same
            value for all spatial dimensions. The amount of output padding
            along a given dimension must be lower than the stride along that
            same dimension. If set to None (default), the output shape is
            inferred.
        data_format: A string, either "channels_last" or `channels_first`.
            `data_format` determines the ordering of the dimensions in the
            inputs. If `data_format="channels_last"`, inputs is of shape
            (batch_size, spatial_shape, channels) while if
            `data_format="channels_first"`, inputs is of shape
            (batch_size, channels, spatial_shape).
        dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the dilation rate to use for dilated convolution. If
            `dilation_rate` is int, then every spatial dimension shares
            the same `dilation_rate`.

    Returns:
        A tensor of rank N+2, the result of the conv operation.
    """
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
