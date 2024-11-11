import torch
import torch.nn.functional as tnn

from keras.src import backend
from keras.src import tree
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_torch,
)
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.numpy import expand_dims
from keras.src.backend.torch.numpy import maximum
from keras.src.backend.torch.numpy import where
from keras.src.utils.argument_validation import standardize_tuple


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


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return tnn.tanhshrink(x)


def softplus(x):
    x = convert_to_tensor(x)
    return tnn.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return tnn.softsign(x)


def silu(x):
    x = convert_to_tensor(x)
    return tnn.silu(x)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.logsigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return tnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return tnn.hardsigmoid(x)


def hard_silu(x):
    x = convert_to_tensor(x)
    return tnn.hardswish(x)


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


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return tnn.celu(x, alpha=alpha)


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    return tnn.glu(x, dim=axis)


def hard_tanh(x):
    x = convert_to_tensor(x)
    return tnn.hardtanh(x, min_val=-1.0, max_val=1.0)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return tnn.hardshrink(x, lambd=threshold)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = backend.standardize_dtype(x.dtype)
    # TODO: tnn.softmax doesn't support float16 using cpu
    if (
        get_device() == "cpu"
        and backend.standardize_dtype(x.dtype) == "float16"
    ):
        x = cast(x, "float32")
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = torch.reshape(x, [-1])
        output = tnn.softmax(output, dim=-1)
        output = torch.reshape(output, x.shape)
    else:
        output = tnn.softmax(x, dim=axis)
    return cast(output, dtype)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = backend.standardize_dtype(x.dtype)
    # TODO: tnn.log_softmax doesn't support float16 using cpu
    if (
        get_device() == "cpu"
        and backend.standardize_dtype(x.dtype) == "float16"
    ):
        x = cast(x, "float32")
    if axis is None:
        # Unlike numpy, PyTorch will handle axis=None as axis=-1.
        # We need this workaround for the reduction on every dim.
        output = torch.reshape(x, [-1])
        output = tnn.log_softmax(output, dim=-1)
        output = torch.reshape(output, x.shape)
    else:
        output = tnn.log_softmax(x, dim=axis)
    return cast(output, dtype)


def _compute_padding_length(
    input_length, kernel_length, stride, dilation_rate=1
):
    """Compute padding length along one dimension."""
    total_padding_length = (
        dilation_rate * (kernel_length - 1) - (input_length - 1) % stride
    )
    left_padding = total_padding_length // 2
    right_padding = (total_padding_length + 1) // 2
    return (left_padding, right_padding)


def _apply_same_padding(
    inputs, kernel_size, strides, operation_type, dilation_rate=1
):
    """Apply same padding to the input tensor.

    This function will evaluate if the padding value is compatible with torch
    functions. To avoid calling `pad()` as much as possible, which may cause
    performance or memory issues, when compatible, it does not apply the padding
    to the tensor, but returns the input tensor and the padding value to pass to
    the torch functions. If not compatible, it returns the padded tensor and 0
    as the padding value.

    Returns:
        tensor: A padded tensor or the inputs.
        padding: The padding value, ready to pass to the torch functions.
    """
    spatial_shape = inputs.shape[2:]
    num_spatial_dims = len(spatial_shape)
    padding = ()

    for i in range(num_spatial_dims):
        if operation_type == "pooling":
            padding_size = _compute_padding_length(
                spatial_shape[i], kernel_size[i], strides[i]
            )
            mode = "replicate"
        else:
            dilation_rate = standardize_tuple(
                dilation_rate, num_spatial_dims, "dilation_rate"
            )
            padding_size = _compute_padding_length(
                spatial_shape[i], kernel_size[i], strides[i], dilation_rate[i]
            )
            mode = "constant"
        padding = (padding_size,) + padding

    if all([left == right for left, right in padding]):
        return inputs, [left for left, _ in padding]

    flattened_padding = tuple(
        value for left_and_right in padding for value in left_and_right
    )
    return tnn.pad(inputs, pad=flattened_padding, mode=mode), 0


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
            "Inputs must have ndim=3, 4 or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the transpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = torch.permute(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = torch.permute(outputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        outputs = torch.permute(outputs, (0, 2, 3, 4, 1))
    return outputs


def _transpose_conv_kernel(kernel):
    # Torch requires conv kernel of format
    # `(out_channels, in_channels, spatial_dims)`, we need to transpose.
    num_spatial_dims = len(kernel.shape) - 2
    if num_spatial_dims == 1:
        kernel = torch.permute(kernel, (2, 1, 0))
    elif num_spatial_dims == 2:
        kernel = torch.permute(kernel, (3, 2, 0, 1))
    elif num_spatial_dims == 3:
        kernel = torch.permute(kernel, (4, 3, 0, 1, 2))
    return kernel


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

    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)

    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs, padding = _apply_same_padding(
            inputs, pool_size, strides, operation_type="pooling"
        )
    else:
        padding = 0

    device = get_device()
    # Torch max pooling ops do not support symbolic tensors.
    # Create a real tensor to execute the ops.
    if device == "meta":
        inputs = torch.empty(
            size=inputs.shape, dtype=inputs.dtype, device="cpu"
        )

    if num_spatial_dims == 1:
        outputs = tnn.max_pool1d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    elif num_spatial_dims == 2:
        outputs = tnn.max_pool2d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    elif num_spatial_dims == 3:
        outputs = tnn.max_pool3d(
            inputs, kernel_size=pool_size, stride=strides, padding=padding
        )
    else:
        raise ValueError(
            "Inputs to pooling op must have ndim=3, 4 or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )

    outputs = outputs.to(device)
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

    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)
    if padding == "same":
        # Torch does not natively support `"same"` padding, we need to manually
        # apply the right amount of padding to `inputs`.
        inputs, padding = _apply_same_padding(
            inputs, pool_size, strides, operation_type="pooling"
        )
    else:
        padding = 0

    if num_spatial_dims == 1:
        outputs = tnn.avg_pool1d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
    elif num_spatial_dims == 2:
        outputs = tnn.avg_pool2d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
    elif num_spatial_dims == 3:
        outputs = tnn.avg_pool3d(
            inputs,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            count_include_pad=False,
        )
    else:
        raise ValueError(
            "Inputs to pooling op must have ndim=3, 4 or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )
    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)
    # Transpose kernel from keras format to torch format.
    kernel = _transpose_conv_kernel(kernel)
    if padding == "same" and any(d != 1 for d in tree.flatten(strides)):
        # Torch does not support this case in conv2d().
        # Manually pad the tensor.
        inputs, padding = _apply_same_padding(
            inputs,
            kernel.shape[2:],
            strides,
            operation_type="conv",
            dilation_rate=dilation_rate,
        )
    channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[1]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel.shape[1]. Received: inputs.shape={inputs.shape}, "
            f"kernel.shape={kernel.shape}"
        )
    groups = channels // kernel_in_channels
    if num_spatial_dims == 1:
        outputs = tnn.conv1d(
            inputs,
            kernel,
            stride=strides,
            dilation=dilation_rate,
            groups=groups,
            padding=padding,
        )
    elif num_spatial_dims == 2:
        outputs = tnn.conv2d(
            inputs,
            kernel,
            stride=strides,
            dilation=dilation_rate,
            groups=groups,
            padding=padding,
        )
    elif num_spatial_dims == 3:
        outputs = tnn.conv3d(
            inputs,
            kernel,
            stride=strides,
            dilation=dilation_rate,
            groups=groups,
            padding=padding,
        )
    else:
        raise ValueError(
            "Inputs to conv operation should have ndim=3, 4, or 5,"
            "corresponding to 1D, 2D and 3D inputs. Received input "
            f"shape: {inputs.shape}."
        )

    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    kernel = convert_to_tensor(kernel)
    kernel = torch.reshape(
        kernel, kernel.shape[:-2] + (1, kernel.shape[-2] * kernel.shape[-1])
    )
    return conv(inputs, kernel, strides, padding, data_format, dilation_rate)


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
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
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2
    strides = standardize_tuple(strides, num_spatial_dims, "strides")

    data_format = backend.standardize_data_format(data_format)
    (
        torch_padding,
        torch_output_padding,
    ) = compute_conv_transpose_padding_args_for_torch(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )
    if data_format == "channels_last":
        inputs = _transpose_spatial_inputs(inputs)
    # Transpose kernel from keras format to torch format.
    kernel = _transpose_conv_kernel(kernel)
    kernel_spatial_shape = kernel.shape[2:]
    if isinstance(dilation_rate, int):
        dilation_rate = [dilation_rate] * len(kernel_spatial_shape)

    if num_spatial_dims == 1:
        outputs = tnn.conv_transpose1d(
            inputs,
            kernel,
            stride=strides,
            padding=torch_padding,
            output_padding=torch_output_padding,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 2:
        outputs = tnn.conv_transpose2d(
            inputs,
            kernel,
            stride=strides,
            padding=torch_padding,
            output_padding=torch_output_padding,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 3:
        outputs = tnn.conv_transpose3d(
            inputs,
            kernel,
            stride=strides,
            padding=torch_padding,
            output_padding=torch_output_padding,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(
            "Inputs to conv transpose operation should have ndim=3, 4, or 5,"
            "corresponding to 1D, 2D and 3D inputs. Received input "
            f"shape: {inputs.shape}."
        )
    if data_format == "channels_last":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    # Axis is the output axis. By default, PyTorch, outputs to last axis.
    # If axis is not last, change output to axis and shift remaining elements.
    x = convert_to_tensor(x, dtype=torch.long)
    zero = convert_to_tensor(0, dtype=torch.long)

    # Torch one_hot does not natively handle negative values, so we add some
    # manual handling for negatives in the input to one_hot by using max(x, 0).
    # The output will have some invalid results, so we set them back to 0 using
    # `where` afterwards.
    output = tnn.one_hot(maximum(x, 0), num_classes)
    output = where(expand_dims(x, axis=-1) >= 0, output, zero)
    output = convert_to_tensor(output, dtype=dtype)
    dims = output.dim()
    if axis != -1 and axis != dims:
        new_axes_order = list(range(dims))
        new_axes_order[axis] = -1  # Shifts output to axis position
        # Shift remaining axes with offset by 1 since output moved to `axis`.
        for ax in range(axis + 1, dims):
            new_axes_order[ax] -= 1
        output = output.permute(new_axes_order)
    return output


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = torch.amax(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        dim=reduction_axis,
    )
    return outputs


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
        output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
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
        output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = torch.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -torch.sum(target * log_prob, dim=axis)


def binary_crossentropy(target, output, from_logits=False):
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
        output = torch.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        return tnn.binary_cross_entropy(output, target, reduction="none")


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with PyTorch."
        )
    x = convert_to_tensor(x)
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = torch.mean(x, dim=axes, keepdim=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    # Note: stop_gradient does not change the gradient to the mean, because that
    # gradient is zero.
    variance = torch.mean(
        torch.square(x), dim=axes, keepdim=True
    ) - torch.square(mean)

    if not keepdims:
        mean = torch.squeeze(mean, axes)
        variance = torch.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = torch.clip(
            mean,
            torch.finfo(torch.float16).min,
            torch.finfo(torch.float16).max,
        )
        variance = torch.clip(
            variance,
            torch.finfo(torch.float16).min,
            torch.finfo(torch.float16).max,
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)

    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = torch.reshape(mean, shape)
    variance = torch.reshape(variance, shape)

    if offset is not None:
        offset = convert_to_tensor(offset)
        offset = torch.reshape(offset, shape)
    else:
        offset = torch.zeros_like(mean)
    if scale is not None:
        scale = convert_to_tensor(scale)
        scale = torch.reshape(scale, shape)
    else:
        scale = torch.ones_like(variance)

    return (
        x.subtract(mean)
        .mul_(variance.add(epsilon).rsqrt_().mul(scale))
        .add_(offset)
    )


def ctc_loss(
    target,
    output,
    target_length,
    output_length,
    mask_index=0,
):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length)
    output_length = convert_to_tensor(output_length)

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(output.dtype, "float32")
    output = cast(output, dtype)

    output = torch.transpose(output, 1, 0)
    logits = tnn.log_softmax(output, dim=-1)
    loss = tnn.ctc_loss(
        logits,
        target,
        output_length,
        target_length,
        blank=mask_index,
        reduction="none",
    )
    return loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = torch.argmax(inputs, axis=-1)
    indices = cast(indices, "int32")
    scores = torch.max(inputs, axis=-1)[0]

    seqlen_mask = torch.arange(max_length, device=indices.device)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = torch.where(seqlen_mask, mask_index, indices)
    scores = torch.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat = indices[:, 1:] == indices[:, :-1]
        repeat = tnn.pad(repeat, (1, 0, 0, 0))
        indices = torch.where(repeat, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = torch.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = torch.unsqueeze(
        torch.arange(max_length, device=indices.device), dim=0
    )  # [1, N]
    order = torch.tile(order, (batch_size, 1))  # [B, N]
    order = torch.where(invalid_mask, max_length, order)
    order = torch.argsort(order, dim=-1)
    indices = torch.take_along_dim(indices, order, dim=-1)

    scores = -torch.sum(scores, axis=1)[:, None]
    indices = torch.unsqueeze(indices, dim=0)
    return indices, scores


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        raise NotImplementedError(
            "Torch backend doesn't yet support the beam search strategy for CTC"
            "decoding."
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )


def psnr(x1, x2, max_val):
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    x1, x2 = (
        convert_to_tensor(x1),
        convert_to_tensor(x2),
    )
    max_val = convert_to_tensor(max_val, dtype=x1.dtype)
    mse = torch.mean((x1 - x2) ** 2)
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = backend.standardize_dtype(dtype)
    if dtype == "float16":
        val = 65500.0
    else:
        val = 3.38953e38
    return convert_to_tensor(val * -0.7, dtype=dtype)


def is_flash_attention_enabled(query, key, value, mask=None, is_causal=False):
    params = torch.backends.cuda.SDPAParams(
        query,
        key,
        value,
        mask,
        0.0,
        is_causal,
    )
    is_enabled = torch.backends.cuda.can_use_flash_attention(params, False)
    return is_enabled


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=False,
):
    if bias is not None:
        raise ValueError(
            "torch's `dot_product_attention` doesn't support `bias`."
        )
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 3D and 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    bias = bias if bias is None else convert_to_tensor(bias)
    mask = mask if mask is None else convert_to_tensor(mask, dtype="bool")
    if mask is not None:
        # Explicit set `is_causal` to `False` when `mask` is not `None`.
        is_causal = False
        mask = torch.where(mask, 0.0, _get_large_negative(query.dtype))

    axis0, axis1 = 1, 2
    query = torch.transpose(query, axis0, axis1)
    key = torch.transpose(key, axis0, axis1)
    value = torch.transpose(value, axis0, axis1)

    if flash_attention:
        is_enabled = is_flash_attention_enabled(
            query=query,
            key=key,
            value=value,
            mask=mask,
            is_causal=is_causal,
        )
        if not is_enabled:
            raise ValueError(
                "Flash attention is not enabled in `torch` backend. "
                "The dtype of the inputs should be float16/bfloat16 "
                "and your GPU should support flash attention implementation."
            )

        with torch.nn.attention.sdpa_kernel(
            backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION],
        ):
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=is_causal,
                scale=scale,
            )
    else:
        if mask is not None:
            mask = mask.contiguous()
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=mask,
            is_causal=is_causal,
            scale=scale,
        )
    return torch.transpose(attention_output, axis1, axis0)
