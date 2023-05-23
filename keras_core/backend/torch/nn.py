import numpy as np
import torch
import torch.nn.functional as tnn

from keras_core.backend import standardize_data_format
from keras_core.backend.config import epsilon
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.utils.argument_validation import standardize_tuple


def relu(x):
    x = convert_to_tensor(x)
    return tnn.relu(x)


def relu6(x):
    x = convert_to_tensor(x)
    return tnn.relu6(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.sigmoid(x)


def tanh(x):
    x = convert_to_tensor(x)
    return tnn.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return tnn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return tnn.softsign(x)


def silu(x, beta=1.0):
    x = convert_to_tensor(x)
    return x * sigmoid(beta * x)


def swish(x):
    x = convert_to_tensor(x)
    return silu(x, beta=1)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.logsigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return tnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.hardsigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return tnn.elu(x, alpha)


def selu(x):
    x = convert_to_tensor(x)
    return tnn.selu(x)


def gelu(x, approximate=True):
    # TODO: torch.nn.gelu expects string approximate of `"none"` or `"tanh"`
    x = convert_to_tensor(x)
    if approximate:
        return tnn.gelu(x, approximate="tanh")
    return tnn.gelu(x)


def softmax(x, axis=None):
    logits = convert_to_tensor(x)
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        logits_exp = torch.exp(logits)
        return logits_exp / torch.sum(logits_exp)
    return tnn.softmax(logits, dim=axis)


def log_softmax(x, axis=None):
    logits = convert_to_tensor(x)
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        logits_exp = torch.exp(logits)
        return logits - torch.log(torch.sum(logits_exp))
    return tnn.log_softmax(logits, dim=axis)


def _compute_padding_length(kernel_length):
    """Compute padding length along one dimension."""
    total_padding_length = kernel_length - 1
    left_padding = int(np.floor(total_padding_length / 2))
    right_padding = int(np.ceil(total_padding_length / 2))
    return (left_padding, right_padding)


def _apply_same_padding(inputs, kernel_size):
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = ()
    for i in range(num_spatial_dims):
        padding_size = _compute_padding_length(kernel_size[i])
        padding = padding_size + padding

    return tnn.pad(inputs, padding, mode="replicate")


def _transpose_spatial_inputs(inputs):
    num_spatial_dims = inputs.ndim - 2
    # Torch pooling does not support `channels_last` format, so
    # we need to transpose to `channels_first` format.
    if num_spatial_dims == 1:
        inputs = torch.permute(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = torch.permute(inputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        inputs = torch.permute(inputs, (0, 4, 1, 2, 3))
    else:
        raise ValueError(
            "Pooling inputs's shape must be 3, 4 or 5, corresponding to 1D, 2D "
            f"and 3D inputs. But received shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the tranpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = torch.permute(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = torch.permute(outputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        outputs = torch.permute(outputs, (0, 2, 3, 4, 1))
    return outputs


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    if strides is None:
        strides = pool_size
    else:
        strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)

    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs = _apply_same_padding(inputs, pool_size)

    if num_spatial_dims == 1:
        outputs = tnn.max_pool1d(inputs, kernel_size=pool_size, stride=strides)
    elif num_spatial_dims == 2:
        outputs = tnn.max_pool2d(inputs, kernel_size=pool_size, stride=strides)
    elif num_spatial_dims == 3:
        outputs = tnn.max_pool3d(inputs, kernel_size=pool_size, stride=strides)
    else:
        raise ValueError(
            "Inputs to pooling operation must have ndim=3, 4, or 5 "
            "corresponding to 1D, 2D and 3D inputs. Received: "
            f"inputs.shape={inputs.shape}."
        )
    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    pool_size = standardize_tuple(pool_size, num_spatial_dims, "pool_size")
    if strides is None:
        strides = pool_size
    else:
        strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)

    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs = _apply_same_padding(inputs, pool_size)

    if num_spatial_dims == 1:
        outputs = tnn.avg_pool1d(inputs, kernel_size=pool_size, stride=strides)
    elif num_spatial_dims == 2:
        outputs = tnn.avg_pool2d(inputs, kernel_size=pool_size, stride=strides)
    elif num_spatial_dims == 3:
        outputs = tnn.avg_pool3d(inputs, kernel_size=pool_size, stride=strides)
    else:
        raise ValueError(
            "Inputs to pooling operation must have ndim=3, 4, or 5 "
            "corresponding to 1D, 2D and 3D inputs. Received: "
            f"inputs.shape={inputs.shape}."
        )
    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError("`conv` not yet implemented for PyTorch Backend")


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError(
        "`depthwise_conv` not yet implemented for PyTorch Backend"
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError(
        "`conv_transpose` not yet implemented for PyTorch backend"
    )


def one_hot(x, num_classes, axis=-1):
    # Axis is the output axis. By default, PyTorch, outputs to last axis.
    # If axis is not last, change output to axis and shift remaining elements.
    x = convert_to_tensor(x, dtype=torch.long)
    output = tnn.one_hot(x, num_classes)
    dims = output.dim()
    if axis != -1 and axis != dims:
        new_axes_order = list(range(dims))
        new_axes_order[axis] = -1  # Shifts output to axis positon
        # Shift remaining axes with offset by 1 since output moved to `axis`.
        for ax in range(axis + 1, dims):
            new_axes_order[ax] -= 1
        output = output.permute(new_axes_order)
    return output


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

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

    if from_logits:
        log_prob = tnn.log_softmax(output, dim=axis)
    else:
        output = output / torch.sum(output, dim=axis, keepdim=True)
        output = torch.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = torch.log(output)
    return -torch.sum(target * log_prob, dim=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype=torch.long)
    output = convert_to_tensor(output)

    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = torch.squeeze(target, dim=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = tnn.log_softmax(output, dim=axis)
    else:
        output = output / torch.sum(output, dim=axis, keepdim=True)
        output = torch.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = torch.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -torch.sum(target * log_prob, dim=axis)


def binary_crossentropy(target, output, from_logits=False):
    # TODO: `torch.as_tensor` has device arg. Need to think how to pass it.
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    # By default, PyTorch, does reduction of `sum` over all rows,
    # change reduction to `none` to keep dim
    if from_logits:
        return tnn.binary_cross_entropy_with_logits(
            output, target, reduction="none"
        )
    else:
        return tnn.binary_cross_entropy(output, target, reduction="none")
