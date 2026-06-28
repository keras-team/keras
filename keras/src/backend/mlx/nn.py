import builtins
import math

import mlx.core as mx
import numpy as np

from keras.src import backend
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import check_conv_input_channels
from keras.src.backend.common.backend_utils import (
    check_conv_transpose_input_channels,
)
from keras.src.backend.common.backend_utils import (
    compute_adaptive_pooling_window_sizes,
)
from keras.src.backend.mlx.core import _cast
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_numpy
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.utils.module_utils import scipy


def cast(x, dtype):
    return _cast(x, dtype)


# ======================================================================
# Group: activations
# ======================================================================

def relu(x):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(0.0, x.dtype))


def relu6(x):
    x = convert_to_tensor(x)
    # np.clip incorrectly promotes bfloat16 to float32, so we replace it with
    # mx.minimum and mx.maximum here
    return mx.minimum(
        mx.maximum(x, mx.array(0.0, x.dtype)), mx.array(6.0, x.dtype)
    )


def sigmoid(x):
    x = convert_to_tensor(x)
    return mx.array(1.0, x.dtype) / (mx.array(1.0, x.dtype) + mx.exp(-x))


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return mx.where(
        x <= -1,
        mx.array(0.0, x.dtype),
        mx.where(
            x >= 1, mx.array(1.0, x.dtype), 0.5 * (x + 1)
        ),
    )


def tanh(x):
    return mx.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return x - mx.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return mx.logaddexp(x, mx.array(0.0, x.dtype))


def softsign(x):
    x = convert_to_tensor(x)
    return x / (mx.array(1.0, x.dtype) + mx.abs(x))


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return mx.where(
        x > threshold,
        (x - threshold).astype(x.dtype),
        mx.where(
            x < -threshold,
            (x + threshold).astype(x.dtype),
            mx.array(0.0, dtype=x.dtype),
        ),
    )


def sparse_plus(x):
    x = convert_to_tensor(x)
    return mx.where(
        x <= -1,
        mx.zeros_like(x),
        mx.where(
            x < 1, (0.25 * (x + 1) ** 2), x
        ),
    )


def silu(x):
    x = convert_to_tensor(x)
    return x * sigmoid(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + mx.sqrt(x**2 + b)
    return y / 2


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return -softplus(-x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(negative_slope, x.dtype) * x)


def hard_sigmoid(x):
    # python numbers would be promoted to float64, so first convert the python
    # numbers to mlx scalars of the input dtype
    x = convert_to_tensor(x)
    x = x / mx.array(6.0, x.dtype) + mx.array(0.5, x.dtype)
    return mx.where(
        x <= 0.0,
        mx.array(0.0, x.dtype),
        mx.where(x >= 1.0, mx.array(1.0, x.dtype), x),
    )


def hard_silu(x):
    x = convert_to_tensor(x)
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return mx.where(
        x >= mx.array(0.0, x.dtype), x, mx.array(alpha, x.dtype) * mx.expm1(x)
    )


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = convert_to_tensor(x)
    return mx.array(scale, x.dtype) * elu(x, alpha)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    # followed by JAX's implementation
    if approximate:
        sqrt_2_over_pi = mx.sqrt(mx.array(2.0 / np.pi, x.dtype))
        cdf = mx.array(0.5, x.dtype) * (
            mx.array(1.0, x.dtype)
            + mx.tanh(
                sqrt_2_over_pi
                * (x + mx.array(0.044715, x.dtype) * (x**3).astype(x.dtype))
            )
        )
        return x * cdf
    else:
        sqrt_2 = mx.sqrt(mx.array(2.0, x.dtype))
        return (
            x
            * (mx.array(scipy.special.erf(convert_to_numpy(x / sqrt_2))).astype(x.dtype) + 1).astype(
                x.dtype
            )
            / mx.array(2, x.dtype)
        )


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    alpha = mx.array(alpha, x.dtype)
    return mx.maximum(x, mx.array(0.0, dtype=x.dtype)) + alpha * mx.expm1(
        mx.minimum(x, mx.array(0.0, dtype=x.dtype)) / alpha
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = x.dtype
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = mx.split(x, 2, axis)
    return (x1 * sigmoid(x2)).astype(dtype)


def hard_tanh(x):
    x = convert_to_tensor(x)
    min_val = mx.array(-1.0, x.dtype)
    max_val = mx.array(1.0, x.dtype)
    return mx.clip(x, min_val, max_val)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    threshold = mx.array(threshold, x.dtype)
    return mx.where(
        mx.abs(x) > threshold, x, mx.array(0.0, dtype=x.dtype)
    )


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return mx.where(
        x > threshold, x, mx.array(default_value, dtype=x.dtype)
    )


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    exp_x = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    max_x = mx.max(x, axis=axis, keepdims=True)
    logsumexp = mx.log(mx.exp(x - max_x).sum(axis=axis, keepdims=True))
    return x - max_x - logsumexp


def sparsemax(x, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(x)
    logits_sorted = -1.0 * mx.sort(-1.0 * logits, axis=axis)
    logits_cumsum = mx.cumsum(logits_sorted, axis=axis)
    r = mx.arange(1, logits.shape[axis] + 1)
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = mx.reshape(r, r_shape)
    support = (logits_sorted - (logits_cumsum - 1) / r) > 0
    # Find the threshold
    k = mx.sum(support, axis=axis, keepdims=True)
    logits_cumsum_safe = mx.where(support, logits_cumsum, mx.array(0.0))
    tau = (mx.sum(logits_cumsum_safe, axis=axis, keepdims=True) - 1) / k
    output = mx.maximum(logits - tau, mx.array(0.0))
    return output


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    if dtype is None:
        dtype = "float32"
    x = convert_to_tensor(x)
    input_shape = x.shape

    x = x.reshape(-1)
    if not num_classes:
        num_classes = int(mx.max(x).item()) + 1

    # Vectorized one-hot via broadcasting: (N, 1) == (1, num_classes).
    indices = x.astype(mx.int32)
    classes = mx.arange(num_classes)
    categorical = mx.where(
        mx.reshape(indices, (-1, 1)) == mx.reshape(classes, (1, -1)),
        mx.array(1.0),
        mx.array(0.0),
    )
    # Mask out negative indices (treated as no class).
    categorical = mx.where(
        mx.reshape(x, (-1, 1)) >= 0, categorical, mx.array(0.0)
    )
    categorical = _cast(categorical, dtype)

    # First, reshape the array with the extra dimension at the end
    output_shape = input_shape + (num_classes,)
    categorical = mx.reshape(categorical, output_shape)

    # Then, move this new dimension to the right place (according to axis)
    if axis != -1 and axis != len(output_shape) - 1:
        categorical = mx.moveaxis(categorical, -1, axis)

    return categorical


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = mx.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs



# ======================================================================
# Group: conv
# ======================================================================


def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _to_spatial_list(value, num_spatial_dims):
    """Normalize an int / tuple / list to a python list of N spatial entries."""
    if isinstance(value, int):
        return [value] * num_spatial_dims
    return [int(v) for v in value]


def _same_pad_for_dim(in_size, kernel, stride, dilation):
    """Replicate jax/tf "SAME" padding for one spatial dim.

    Returns (left, right) explicit padding amounts.
    """
    eff_kernel = (kernel - 1) * dilation + 1
    out = (in_size + stride - 1) // stride
    total = max((out - 1) * stride + eff_kernel - in_size, 0)
    left = total // 2
    right = total - left
    return left, right


def _compute_forward_padding(
    input_shape, kernel_shape, strides, dilation_rate, padding, data_format
):
    """Compute explicit per-side integer padding for a forward convolution.

    MLX's ``mx.conv_general`` only accepts integer / list padding (no
    ``"same"`` / ``"valid"`` strings), so we port the jax padding math here.

    Returns a tuple ``(before, after)`` where each is a list of length
    ``num_spatial_dims`` (suitable for ``mx.conv_general(padding=(before, after))``).
    """
    num_spatial_dims = len(input_shape) - 2
    if data_format == "channels_last":
        spatial_sizes = input_shape[1:-1]
    else:
        spatial_sizes = input_shape[2:]
    kernel_spatial = kernel_shape[:-2]

    strides = _to_spatial_list(strides, num_spatial_dims)
    dilation_rate = _to_spatial_list(dilation_rate, num_spatial_dims)

    before = []
    after = []
    pad_lower = str(padding).lower()
    for i in range(num_spatial_dims):
        if pad_lower == "valid":
            l, r = 0, 0
        elif pad_lower == "same":
            l, r = _same_pad_for_dim(
                int(spatial_sizes[i]),
                int(kernel_spatial[i]),
                strides[i],
                dilation_rate[i],
            )
        else:
            raise ValueError(
                "The `padding` argument must be one of 'valid', 'same'. "
                f"Received: padding={padding}"
            )
        before.append(l)
        after.append(r)
    return before, after


def _channels_last_perm(ndim):
    """Permutation moving channels (axis 1) to the end: NCHW -> NHWC."""
    return (0,) + tuple(range(2, ndim)) + (1,)


def _channels_first_perm(ndim):
    """Inverse of ``_channels_last_perm``: NHWC -> NCHW."""
    return (0, ndim - 1) + tuple(range(1, ndim - 1))


def _to_channels_last(x, data_format):
    if data_format == "channels_last":
        return x
    return x.transpose(_channels_last_perm(x.ndim))


def _from_channels_last(x, data_format):
    if data_format == "channels_last":
        return x
    return x.transpose(_channels_first_perm(x.ndim))


def _transpose_conv_kernel_forward(kernel, num_spatial_dims):
    """Keras conv kernel ``(spatial..., C_in, C_out)`` -> MLX ``(C_out, spatial..., C_in)``."""
    if num_spatial_dims == 1:
        perm = (2, 0, 1)
    elif num_spatial_dims == 2:
        perm = (3, 0, 1, 2)
    elif num_spatial_dims == 3:
        perm = (4, 0, 1, 2, 3)
    else:
        raise ValueError(
            "Inputs to conv operation should have ndim=3, 4, or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input with {num_spatial_dims} spatial dims."
        )
    return kernel.transpose(perm)


def _transpose_conv_kernel_transpose(kernel, num_spatial_dims):
    """Keras transpose-conv kernel ``(spatial..., C_out, C_in)`` -> MLX
    ``(C_out, spatial..., C_in)`` for ``mx.conv_transpose{1,2,3}d``."""
    if num_spatial_dims == 1:
        perm = (1, 0, 2)
    elif num_spatial_dims == 2:
        perm = (2, 0, 1, 3)
    elif num_spatial_dims == 3:
        perm = (3, 0, 1, 2, 4)
    else:
        raise ValueError(
            "Inputs to conv transpose operation should have ndim=3, 4, or 5, "
            "corresponding to 1D, 2D and 3D inputs."
        )
    return kernel.transpose(perm)


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    from keras.src import backend
    from keras.src.backend.mlx.core import is_tensor

    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial_dims = inputs.ndim - 2

    strides = _to_spatial_list(strides, num_spatial_dims)
    dilation_rate = _to_spatial_list(dilation_rate, num_spatial_dims)

    # Validate input channels against the kernel.
    channels = inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
    kernel_in_channels = kernel.shape[-2]
    if (
        isinstance(channels, int)
        and isinstance(kernel_in_channels, int)
        and channels % kernel_in_channels > 0
    ):
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            "kernel's in_channels. Received input channels "
            f"{channels} and kernel in_channels {kernel_in_channels}."
        )
    feature_group_count = (
        channels // kernel_in_channels
        if isinstance(channels, int) and isinstance(kernel_in_channels, int)
        else 1
    )

    inputs_cl = _to_channels_last(inputs, data_format)
    kernel_mlx = _transpose_conv_kernel_forward(kernel, num_spatial_dims)

    before, after = _compute_forward_padding(
        tuple(inputs.shape),
        tuple(kernel.shape),
        strides,
        dilation_rate,
        padding,
        data_format,
    )

    if feature_group_count > 1 and num_spatial_dims == 3:
        # MLX's `conv_general` supports grouped convolution only for 1D/2D
        # inputs (it raises "Can only handle groups != 1 in 1D or 2D" for 3D).
        # Fall back to `feature_group_count` independent groups=1 convolutions:
        # split the input along its channel axis and the kernel along its
        # output-channel axis, convolve each pair, then concatenate. This is
        # numerically identical to a single grouped conv (verified against the
        # native groups path for 1D/2D).
        groups = feature_group_count
        in_splits = mx.split(inputs_cl, groups, axis=-1)
        w_splits = mx.split(kernel_mlx, groups, axis=0)
        outputs = mx.concatenate(
            [
                mx.conv_general(
                    in_splits[i],
                    w_splits[i],
                    stride=strides,
                    padding=(before, after),
                    kernel_dilation=dilation_rate,
                    groups=1,
                )
                for i in range(groups)
            ],
            axis=-1,
        )
    else:
        outputs = mx.conv_general(
            inputs_cl,
            kernel_mlx,
            stride=strides,
            padding=(before, after),
            kernel_dilation=dilation_rate,
            groups=feature_group_count,
        )

    if outputs.size == 0 and inputs.size != 0:
        raise ValueError(
            "The convolution operation resulted in an empty output. "
            "This can happen if the input is too small for the given "
            "kernel size, strides, dilation rate, and padding mode. "
            "Please check the input shape and convolution parameters."
        )

    return _from_channels_last(outputs, data_format)


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    from keras.src import backend
    from keras.src.backend.common.backend_utils import check_conv_input_channels

    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    check_conv_input_channels(inputs, kernel, data_format)
    num_spatial_dims = inputs.ndim - 2

    feature_group_count = (
        inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
    )
    depth_multiplier = kernel.shape[-1]

    # Keras depthwise kernel is (spatial..., C_in, depth_multiplier). Reshape to
    # a regular conv kernel (spatial..., C_in_new=1, C_out=C_in*depth_multiplier)
    # so that one convolution with groups=C_in reproduces depthwise behaviour.
    new_shape = kernel.shape[:-2] + (1, feature_group_count * depth_multiplier)
    kernel = mx.reshape(kernel, new_shape)

    strides = _to_spatial_list(strides, num_spatial_dims)
    dilation_rate = _to_spatial_list(dilation_rate, num_spatial_dims)

    inputs_cl = _to_channels_last(inputs, data_format)
    kernel_mlx = _transpose_conv_kernel_forward(kernel, num_spatial_dims)

    before, after = _compute_forward_padding(
        tuple(inputs.shape),
        tuple(kernel.shape),
        strides,
        dilation_rate,
        padding,
        data_format,
    )

    outputs = mx.conv_general(
        inputs_cl,
        kernel_mlx,
        stride=strides,
        padding=(before, after),
        kernel_dilation=dilation_rate,
        groups=feature_group_count,
    )

    return _from_channels_last(outputs, data_format)


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    from keras.src import backend
    from keras.src.backend.common.backend_utils import check_conv_input_channels

    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    depthwise_kernel = convert_to_tensor(depthwise_kernel)
    pointwise_kernel = convert_to_tensor(pointwise_kernel)
    check_conv_input_channels(inputs, depthwise_kernel, data_format)

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
    from keras.src import backend
    from keras.src.backend.common.backend_utils import (
        check_conv_transpose_input_channels,
    )
    from keras.src.backend.common.backend_utils import (
        compute_conv_transpose_output_crops_for_torch,
    )

    data_format = backend.standardize_data_format(data_format)
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    check_conv_transpose_input_channels(inputs, kernel, data_format)
    num_spatial_dims = inputs.ndim - 2

    # MLX's conv_transpose{1,2,3}d only supports symmetric padding + a right-side
    # output_padding, which cannot express the asymmetric padding Keras "same"
    # semantics require. Mirror the torch backend's strategy: run with
    # padding=0, output_padding=0 (the largest "natural" output) and then
    # asymmetrically crop / zero-pad the spatial dims to the window JAX would
    # compute.
    crops = compute_conv_transpose_output_crops_for_torch(
        input_shape=tuple(inputs.shape),
        kernel_shape=tuple(kernel.shape),
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )

    strides = _to_spatial_list(strides, num_spatial_dims)
    dilation_rate = _to_spatial_list(dilation_rate, num_spatial_dims)

    inputs_cl = _to_channels_last(inputs, data_format)
    kernel_mlx = _transpose_conv_kernel_transpose(kernel, num_spatial_dims)

    if num_spatial_dims == 1:
        outputs = mx.conv_transpose1d(
            inputs_cl,
            kernel_mlx,
            stride=strides[0],
            padding=0,
            output_padding=0,
            dilation=dilation_rate[0],
        )
    elif num_spatial_dims == 2:
        outputs = mx.conv_transpose2d(
            inputs_cl,
            kernel_mlx,
            stride=strides,
            padding=0,
            output_padding=0,
            dilation=dilation_rate,
        )
    elif num_spatial_dims == 3:
        outputs = mx.conv_transpose3d(
            inputs_cl,
            kernel_mlx,
            stride=strides,
            padding=0,
            output_padding=0,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(
            "Inputs to conv transpose operation should have ndim=3, 4, or 5, "
            "corresponding to 1D, 2D and 3D inputs. "
            f"Received input shape: {inputs.shape}."
        )

    # Apply the asymmetric crop to each spatial dim. Negative crops mean we
    # instead need to zero-pad that side. spatial dims start at axis 1 in the
    # channels-last output (N, *spatial, C).
    slices = [slice(None)]
    for crop_left, crop_right in crops:
        start = max(0, crop_left)
        end = -crop_right if crop_right > 0 else None
        slices.append(slice(start, end))
    outputs = outputs[tuple(slices)]

    needs_zero_pad = any(cl < 0 or cr < 0 for cl, cr in crops)
    if needs_zero_pad:
        pad_widths = []
        pad_widths.append((0, 0))  # batch
        for crop_left, crop_right in crops:
            pad_widths.append(
                (
                    -crop_left if crop_left < 0 else 0,
                    -crop_right if crop_right < 0 else 0,
                )
            )
        pad_widths.append((0, 0))  # channels
        outputs = mx.pad(outputs, pad_widths)

    return _from_channels_last(outputs, data_format)



# ======================================================================
# Group: pool
# ======================================================================

def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    x = (x,) * num_spatial_dims if isinstance(x, int) else tuple(x)
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _same_pad_amount(in_size, window, stride):
    """jax-style SAME padding amounts for one spatial axis.

    Returns (pad_before, pad_after) so that the output size is
    ceil(in_size / stride) and each output window is full.
    """
    out_size = (in_size + stride - 1) // stride
    pad_needed = max((out_size - 1) * stride + window - in_size, 0)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    return pad_before, pad_after


def _pool(
    inputs,
    initial_value,
    reduce_fn,
    pool_size,
    strides=None,
    padding="valid",
):
    """Helper that performs a windowed reduction (max or sum) over spatial axes.

    `pool_size` and `strides` are full N+2 tuples (with 1 on the batch and
    channel axes, so only the spatial axes are actually reduced).
    `reduce_fn` is either "max" or "sum" (string, to avoid passing callables
    that call numpy/lax).
    """
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid padding '{padding}', must be 'same' or 'valid'."
        )

    inputs = convert_to_tensor(inputs)
    nd = inputs.ndim
    pool_size = tuple(int(s) for s in pool_size)
    strides = tuple(int(s) for s in strides)
    # Process any axis that has a non-unit window OR a non-unit stride. A
    # stride>1 with window==1 is a pure strided subsample (e.g. ResNetV2's
    # `MaxPooling2D(1, strides=stride)` shortcut); it must NOT be skipped, or
    # the spatial size stays unchanged and downstream residual adds mismatch.
    spatial_axes = [i for i in range(nd) if pool_size[i] != 1 or strides[i] != 1]

    if len(spatial_axes) == 0:
        # Nothing to reduce over.
        return inputs

    # Compute explicit integer padding for "same".
    pads = [(0, 0)] * nd
    if padding == "same":
        for ax in spatial_axes:
            in_size = inputs.shape[ax]
            window = pool_size[ax]
            stride = strides[ax]
            pads[ax] = _same_pad_amount(in_size, window, stride)

    # Apply padding.
    pad_width_arg = [(p[0], p[1]) for p in pads]
    needs_pad = any(p != (0, 0) for p in pad_width_arg)
    if needs_pad:
        if reduce_fn == "max":
            pad_val = float("-inf")
        else:  # sum
            pad_val = 0.0
        # `mx.pad` takes the fill value via the `constant_values=` kwarg; the
        # third positional arg is `mode` (a string), not the fill value.
        x = mx.pad(inputs, pad_width_arg, constant_values=pad_val)
    else:
        x = inputs

    # Output spatial sizes.
    out_shape = []
    for ax in spatial_axes:
        in_size = x.shape[ax]
        window = pool_size[ax]
        stride = strides[ax]
        out_shape.append((in_size - window) // stride + 1)

    # Build per-axis start-index grids via mx.arange and gather windows.
    # Gather axis-by-axis to keep memory bounded; we use advanced indexing.
    def _gather_reduce(x, axes, windows, strides_, out_sizes, is_max):
        # Recursive: process one spatial axis at a time.
        ax = axes[0]
        w = windows[0]
        s = strides_[0]
        o = out_sizes[0]
        starts = mx.arange(0, o) * s  # (o,)
        # window offsets
        offs = mx.arange(0, w)  # (w,)
        # index grid (o, w)
        idx = mx.reshape(starts, (o, 1)) + mx.reshape(offs, (1, w))
        # gather along axis ax -> shape with extra window axis inserted after ax
        gathered = mx.take(x, idx, axis=ax)
        # gathered has shape: x[:ax], (o, w), x[ax+1:]
        # reduce the window axis (which is axis ax+1 now)
        if is_max:
            reduced = mx.max(gathered, axis=ax + 1)
        else:
            reduced = mx.sum(gathered, axis=ax + 1)
        if len(axes) == 1:
            return reduced
        remaining_axes = [a if a < ax else a - 0 for a in axes[1:]]
        # The remaining axes keep their positions relative to the new tensor,
        # but since we reduced a window axis (didn't remove the spatial axis
        # itself, just merged), axis numbering is unchanged.
        return _gather_reduce(
            reduced,
            axes[1:],
            windows[1:],
            strides_[1:],
            out_sizes[1:],
            is_max,
        )

    result = _gather_reduce(
        x,
        spatial_axes,
        [pool_size[ax] for ax in spatial_axes],
        [strides[ax] for ax in spatial_axes],
        out_shape,
        reduce_fn == "max",
    )
    return result


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )
    return _pool(inputs, None, "max", pool_size, strides, padding)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )

    pooled = _pool(inputs, None, "sum", pool_size, strides, padding)
    if padding == "valid":
        # Avoid the extra reduction.
        return pooled / math.prod(pool_size)
    else:
        # Count the number of valid entries at each input point, then use that
        # for computing average. Assumes that any two arrays of same shape will
        # be padded the same. Avoid broadcasting on axis where pooling is
        # skipped.
        shape = [
            (a if b != 1 else 1) for (a, b) in zip(inputs.shape, pool_size)
        ]
        ones = mx.ones(shape, dtype=convert_to_tensor(inputs).dtype)
        window_counts = _pool(ones, None, "sum", pool_size, strides, padding)
        return pooled / window_counts


# ---------------------------------------------------------------------------
# Adaptive pooling.
#
# The reference implementation is purely shape/index driven and relies on
# numpy stride tricks (as_strided). Reproducing that exactly in mlx without an
# equivalent stride primitive is brittle, so we materialize the input to numpy
# and reuse the verified numpy algorithm, then convert back. Adaptive pooling
# is not autograd-critical in the keras pipeline (it is used for inference
# paths such as adaptive pooling layers).
# ---------------------------------------------------------------------------


def _to_np(x):
    if hasattr(x, "dtype") and "mlx" in str(type(x)):
        return np.asarray(convert_to_numpy(x))
    return np.asarray(x)


def _compute_adaptive_pooling_gather_indices(
    input_dim, output_size, big_window
):
    window_starts = np.floor(
        (np.arange(output_size) * input_dim) / output_size
    ).astype(np.int32)

    window_ends = np.ceil(
        (np.arange(1, output_size + 1) * input_dim) / output_size
    ).astype(np.int32)

    window_sizes = window_ends - window_starts
    is_big = window_sizes == big_window

    small_window = big_window - 1
    small_pool_len = input_dim - small_window + 1

    small_indices = window_starts
    big_indices = window_starts + small_pool_len

    gather = np.where(is_big, big_indices, small_indices)
    return gather.astype(np.int32)


def _strided_view_1d(x, window_size):
    n, l, c = x.shape
    out = l - window_size + 1

    strides = x.strides
    shape = (n, out, window_size, c)
    new_strides = (strides[0], strides[1], strides[1], strides[2])

    return np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=new_strides
    )


def _adaptive_pool1d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size,)

    inputs = _to_np(inputs)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 1))

    n, l, c = inputs.shape
    out_l = output_size[0]

    small, big = compute_adaptive_pooling_window_sizes(l, out_l)
    gather = _compute_adaptive_pooling_gather_indices(l, out_l, big)

    sv_small = _strided_view_1d(inputs, small)
    small_pool = (
        np.mean(sv_small, axis=2)
        if mode == "average"
        else np.max(sv_small, axis=2)
    )

    sv_big = _strided_view_1d(inputs, big)
    big_pool = (
        np.mean(sv_big, axis=2)
        if mode == "average"
        else np.max(sv_big, axis=2)
    )

    combined = np.concatenate([small_pool, big_pool], axis=1)
    out = combined[:, gather, :]

    if data_format == "channels_first":
        out = np.transpose(out, (0, 2, 1))

    return convert_to_tensor(out)


def _adaptive_pool2d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    inputs = _to_np(inputs)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 3, 1))

    n, h, w, c = inputs.shape
    out_h, out_w = output_size

    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = np.transpose(inputs, (0, 2, 1, 3)).reshape(n * w, h, c)

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        np.mean(sv_small_h, axis=2)
        if mode == "average"
        else np.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        np.mean(sv_big_h, axis=2)
        if mode == "average"
        else np.max(sv_big_h, axis=2)
    )

    combined_h = np.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :]

    pooled_h = pooled_h.reshape(n, w, out_h, c)
    pooled_h = np.transpose(pooled_h, (0, 2, 1, 3))

    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        np.mean(sv_small_w, axis=2)
        if mode == "average"
        else np.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        np.mean(sv_big_w, axis=2)
        if mode == "average"
        else np.max(sv_big_w, axis=2)
    )

    combined_w = np.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_h, out_w, c)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 3, 1, 2))

    return convert_to_tensor(out)


def _adaptive_pool3d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    inputs = _to_np(inputs)

    if data_format == "channels_first":
        inputs = np.transpose(inputs, (0, 2, 3, 4, 1))

    n, d, h, w, c = inputs.shape
    out_d, out_h, out_w = output_size

    small_d, big_d = compute_adaptive_pooling_window_sizes(d, out_d)
    gather_d = _compute_adaptive_pooling_gather_indices(d, out_d, big_d)

    x_d = np.transpose(inputs, (0, 2, 3, 1, 4)).reshape(n * h * w, d, c)

    sv_small_d = _strided_view_1d(x_d, small_d)
    small_pool_d = (
        np.mean(sv_small_d, axis=2)
        if mode == "average"
        else np.max(sv_small_d, axis=2)
    )

    sv_big_d = _strided_view_1d(x_d, big_d)
    big_pool_d = (
        np.mean(sv_big_d, axis=2)
        if mode == "average"
        else np.max(sv_big_d, axis=2)
    )

    combined_d = np.concatenate([small_pool_d, big_pool_d], axis=1)
    pooled_d = combined_d[:, gather_d, :].reshape(n, h, w, out_d, c)
    pooled_d = np.transpose(pooled_d, (0, 3, 1, 2, 4))

    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = np.transpose(pooled_d, (0, 1, 3, 2, 4)).reshape(
        n * out_d * w, h, c
    )

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        np.mean(sv_small_h, axis=2)
        if mode == "average"
        else np.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        np.mean(sv_big_h, axis=2)
        if mode == "average"
        else np.max(sv_big_h, axis=2)
    )

    combined_h = np.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :].reshape(n, out_d, w, out_h, c)
    pooled_h = np.transpose(pooled_h, (0, 1, 3, 2, 4))

    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_d * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        np.mean(sv_small_w, axis=2)
        if mode == "average"
        else np.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        np.mean(sv_big_w, axis=2)
        if mode == "average"
        else np.max(sv_big_w, axis=2)
    )

    combined_w = np.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_d, out_h, out_w, c)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 4, 1, 2, 3))

    return convert_to_tensor(out)


def adaptive_average_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 2:
        return _adaptive_pool2d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 3:
        return _adaptive_pool3d_impl(
            inputs, output_size, "average", data_format
        )
    raise ValueError("adaptive_average_pool supports only 1D/2D/3D")


def adaptive_max_pool(inputs, output_size, data_format=None):
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(inputs, output_size, "max", data_format)
    if dims == 2:
        return _adaptive_pool2d_impl(inputs, output_size, "max", data_format)
    if dims == 3:
        return _adaptive_pool3d_impl(inputs, output_size, "max", data_format)
    raise ValueError("adaptive_max_pool supports only 1D/2D/3D")



# ======================================================================
# Group: attn_norm
# ======================================================================

def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with the MLX backend."
        )

    x = convert_to_tensor(x)
    # Normalize axes to a tuple (support negative axes / single int).
    ndim = len(x.shape)
    if isinstance(axes, int):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    axes = tuple((a % ndim) for a in axes)

    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16.
    need_cast = False
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = _cast(x, "float32")

    # mean = E[x], keepdims=True
    mean = mx.mean(x, axis=axes, keepdims=True)

    # Variance via Var = E[|x|^2] - |E[x]|^2 (faster, less stable).
    mean_of_squares = mx.mean(mx.square(x), axis=axes, keepdims=True)
    variance = mean_of_squares - mx.square(mean)

    if not keepdims:
        # mx.squeeze removes the reduced axes.
        mean = mx.squeeze(mean, axes)
        variance = mx.squeeze(variance, axes)

    if need_cast:
        # Avoid overflow/underflow when casting back float32 -> float16.
        finfo_min = np.finfo(np.float16).min
        finfo_max = np.finfo(np.float16).max
        mean = mx.clip(mean, finfo_min, finfo_max)
        variance = mx.clip(variance, finfo_min, finfo_max)
        mean = _cast(mean, ori_dtype)
        variance = _cast(variance, ori_dtype)

    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)
    if scale is not None:
        scale = convert_to_tensor(scale)
    if offset is not None:
        offset = convert_to_tensor(offset)

    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mx.reshape(mean, shape)
    variance = mx.reshape(variance, shape)

    inv = 1.0 / mx.sqrt(variance + epsilon)
    if scale is not None:
        scale = mx.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = mx.reshape(offset, shape)
        res = res + offset

    return x * inv + res


def _build_additive_attention_mask(mask, is_causal, query_len, key_len, dtype):
    """Combine an optional boolean mask and causal mask into an additive
    (float) mask broadcast-compatible with [B, N, T_q, T_kv].

    Returns None when nothing needs masking.
    """
    # Use a large-negative python float for masked positions (mlx broadcasts it).
    # This is added to the attention logits before softmax, so masked positions
    # get ~zero weight (exp(-1e9) ~= 0, same as -inf to machine precision).
    neg_inf = -1e9

    parts = []

    if mask is not None:
        mask = convert_to_tensor(mask)
        # MLX SDPA additive mask must be float and broadcastable to
        # [B, N, T_q, T_kv]. Convert a boolean mask to additive (0 / -1e9):
        # True (keep) -> 0, False (mask out) -> -1e9.
        if standardize_dtype(mask.dtype) == "bool":
            mask = mx.where(mask, 0.0, neg_inf).astype(mx.float32)
        else:
            mask = mask.astype(mx.float32)
        parts.append(mask)

    if is_causal:
        # Lower-triangular causal mask of shape [T_q, T_kv] (broadcastable).
        eye = mx.tril(mx.ones((query_len, key_len), dtype=mx.bool_))
        causal = mx.where(eye, 0.0, neg_inf).astype(mx.float32)
        parts.append(causal)

    if not parts:
        return None

    combined = parts[0]
    for p in parts[1:]:
        # Two additive masks combine by addition: -inf + finite = -inf.
        combined = combined + p
    # SDPA requires the mask to promote to the output dtype (e.g. float16 when
    # flash_attention forces half precision), so cast to the compute dtype.
    return _cast(combined, dtype)


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    # The keras ops layer always forwards `flash_attention` and
    # `attn_logits_soft_cap`. `attn_logits_soft_cap` is guarded by the ops layer
    # (only valid for JAX-TPU, else it raises), so for MLX it is always None.
    # MLX has no flash / soft-cap control of its own; accept and ignore both.
    del flash_attention, attn_logits_soft_cap

    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )

    from keras.src.backend.common import dtypes as _dtypes

    compute_dtype = _dtypes.result_type(query.dtype, key.dtype, value.dtype)
    query = _cast(query, compute_dtype)
    key = _cast(key, compute_dtype)
    value = _cast(value, compute_dtype)
    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)

    # Keras layout is (B, T, N, D). mx.fast.scaled_dot_product_attention uses
    # (B, N, T, D), so transpose the head/sequence axes.
    query_t = mx.transpose(query, (0, 2, 1, 3))  # B, N, T_q, D
    key_t = mx.transpose(key, (0, 2, 1, 3))  # B, N_kv, T_kv, D
    value_t = mx.transpose(value, (0, 2, 1, 3))  # B, N_kv, T_kv, D

    H = key.shape[3]
    if scale is None:
        scale = 1.0 / math.sqrt(H)
    scale = float(scale)

    # Build an additive attention mask that also folds in the additive `bias`.
    # mx.fast.scaled_dot_product_attention accepts an additive float mask
    # broadcast-compatible with [B, N, T_q, T_kv].
    query_len = query.shape[1]
    key_len = key.shape[1]
    attn_mask = _build_additive_attention_mask(
        mask, is_causal, query_len, key_len, compute_dtype
    )
    if bias is not None:
        # `bias` is already cast to `compute_dtype`; keep the additive mask in
        # that dtype so it promotes to the SDPA output type.
        if attn_mask is None:
            attn_mask = bias
        else:
            attn_mask = attn_mask + bias

    out = mx.fast.scaled_dot_product_attention(
        query_t, key_t, value_t, scale=scale, mask=attn_mask
    )

    # Back to keras layout (B, T, N, D).
    out = mx.transpose(out, (0, 2, 1, 3))
    return _cast(out, compute_dtype)



# ======================================================================
# Group: losses
# ======================================================================

def _log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    # Canonicalize negative axis.
    ndim = len(x.shape)
    ax = axis if axis >= 0 else axis + ndim
    max_x = mx.max(x, axis=ax, keepdims=True)
    shifted = x - max_x
    sum_exp = mx.sum(mx.exp(shifted), axis=ax, keepdims=True)
    return shifted - mx.log(sum_exp)


def _one_hot(x, num_classes, axis=-1):
    """One-hot encode integer tensor `x`.

    Matches keras numpy backend `one_hot`: negative indices yield an all-zero
    row, the new class dimension is placed at `axis`, and the result is float32.
    """
    x = convert_to_tensor(x)
    if num_classes is None or num_classes == 0:
        num_classes = int(mx.max(x).item()) + 1
    x_int = _cast(x, "int32")
    # Build the one-hot mask over a flattened (prod(input_shape), num_classes).
    input_shape = tuple(x_int.shape)
    flat = mx.reshape(x_int, (-1,))
    idx = mx.arange(num_classes)
    # mask[b, c] = (flat[b] == c) & (flat[b] >= 0)
    eq = flat[:, None] == idx[None, :]
    ge0 = flat[:, None] >= 0
    categorical = _cast(mx.logical_and(eq, ge0), "float32")
    out_shape = input_shape + (num_classes,)
    categorical = mx.reshape(categorical, out_shape)
    if axis != -1:
        ndim = len(out_shape)
        ax = axis if axis >= 0 else axis + ndim
        # Move the trailing class axis to `ax`.
        perm = list(range(ndim))
        perm.insert(ax, perm.pop(ndim - 1))
        categorical = mx.transpose(categorical, perm)
    return categorical


def _sigmoid(x):
    return mx.sigmoid(convert_to_tensor(x))


def _log10(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    return mx.log10(_cast(x, dtype))


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
        log_prob = _log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis=axis, keepdims=True)
        eps = backend.epsilon()
        output = mx.clip(output, eps, 1.0 - eps)
        log_prob = mx.log(output)
    return -mx.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = _cast(convert_to_tensor(target), "int32")
    output = convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = mx.squeeze(target, axis=-1)

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
        log_prob = _log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis=axis, keepdims=True)
        eps = backend.epsilon()
        output = mx.clip(output, eps, 1.0 - eps)
        log_prob = mx.log(output)
    num_classes = output.shape[axis]
    target = _one_hot(target, num_classes, axis=axis)
    return -mx.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = _sigmoid(output)

    eps = backend.epsilon()
    output = mx.clip(output, eps, 1.0 - eps)
    bce = target * mx.log(output)
    bce = bce + (1.0 - target) * mx.log(1.0 - output)
    return -bce


def psnr(x1, x2, max_val):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    # mean over the squared difference; promote integer images to float.
    diff = x1 - x2
    sq = mx.square(diff)
    dtype = standardize_dtype(sq.dtype)
    if "int" in dtype or dtype == "bool":
        sq = _cast(sq, "float32")
    mse = mx.mean(sq)
    # PSNR = 20*log10(max_val) - 10*log10(mse). Use python scalars so mlx
    # broadcasts them (do NOT pass python floats through `_cast`, which calls
    # `.astype` on them and fails).
    return 20.0 * _log10(max_val) - 10.0 * _log10(mse)



# ======================================================================
# Group: ctc
# ======================================================================

def ctc_loss(target, output, target_length, output_length, mask_index=0):
    # Ref: https://github.com/google-deepmind/optax
    # optax.ctc_loss_with_forward_probs
    #
    # CTC loss is a dynamic-programming algorithm that relies on numpy-only
    # utilities (logaddexp, einsum, boolean masks, padded scans). It is almost
    # never on the autograd path, so we materialize the inputs to numpy and run
    # the exact reference computation, then return the result as an mlx tensor.
    # This keeps behaviour and numerical stability identical to the numpy backend.
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, "int32")
    output_length = convert_to_tensor(output_length, "int32")

    # Materialize to numpy for the dynamic-programming computation.
    output_np = np.asarray(convert_to_numpy(output))
    target_np = np.asarray(convert_to_numpy(target))
    target_length_np = np.asarray(convert_to_numpy(target_length))
    output_length_np = np.asarray(convert_to_numpy(output_length))

    batch_size, max_input_length, num_classes = output_np.shape
    batch_size, max_label_length = target_np.shape
    log_epsilon = -1e5

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(str(output_np.dtype), "float32")
    output_np = output_np.astype(dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = np.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = np.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return np.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length_np, max_label_length)
    output_paddings = _lengths_to_paddings(output_length_np, max_input_length)
    target_paddings = target_paddings.astype(output_np.dtype)
    output_paddings = output_paddings.astype(output_np.dtype)

    # log_softmax computed inline (numerically stable) to avoid depending on a
    # sibling function that may not be in scope when this module is assembled.
    def _np_log_softmax(x, axis=-1):
        max_x = np.max(x, axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
        return x - max_x - logsumexp

    # one_hot computed inline.
    def _np_one_hot(x, num_classes, dtype=None):
        if dtype is None:
            dtype = "float32"
        x = np.asarray(x)
        input_shape = x.shape
        x_flat = x.reshape(-1)
        if not num_classes:
            num_classes = int(np.max(x_flat)) + 1
        bsz = x_flat.shape[0]
        categorical = np.zeros((bsz, num_classes), dtype=dtype)
        valid = x_flat >= 0
        categorical[np.arange(bsz)[valid], x_flat[valid]] = 1
        return np.reshape(categorical, input_shape + (num_classes,))

    logprobs = _np_log_softmax(output_np, axis=-1)
    label_lengths = max_label_length - np.sum(target_paddings, axis=1).astype(
        np.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target_np[:, :-1] == target_np[:, 1:]).astype(np.float32)
    repeat = np.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = np.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = _np_one_hot(target_np, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = np.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = np.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        np.ones((batch_size, max_label_length + 1), dtype=output_np.dtype)
        * log_epsilon
    )
    logalpha_phi_init[:, 0] = 0.0
    logalpha_emit_init = (
        np.ones((batch_size, max_label_length), dtype=output_np.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return np.concatenate(
            [phi[:, :1], np.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = np.logaddexp(
            prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label is repetition
        next_phi = update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
        )

        pad = pad.reshape((batch_size, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    def np_scan(f, init, xs):
        carry = init
        ys = []
        for x in zip(*xs):
            carry, y = f(carry, x)
            ys.append(y)
        result = []
        for i in range(len(ys[0])):
            result.append(np.stack([y[i] for y in ys]))
        return carry, result

    xs = (logprobs_emit, logprobs_phi, output_paddings.transpose((1, 0)))
    _, (logalpha_phi, logalpha_emit) = np_scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi[-1] = logalpha_phi_last

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = _np_one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -np.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
    return convert_to_tensor(per_seq_loss)


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")

    inputs_np = np.asarray(convert_to_numpy(inputs))
    seq_np = np.asarray(convert_to_numpy(sequence_lengths))

    batch_size, max_length, num_classes = inputs_np.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = np.argmax(inputs_np, axis=-1).astype("int32")
    scores = np.max(inputs_np, axis=-1)

    seqlen_mask = np.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= seq_np[:, None]

    indices = np.where(seqlen_mask, mask_index, indices)
    scores = np.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = np.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = np.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = np.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = np.expand_dims(np.arange(max_length), axis=0)  # [1, N]
    order = np.tile(order, (batch_size, 1))  # [B, N]
    order = np.where(invalid_mask, max_length, order)
    order = np.argsort(order, axis=-1)
    indices = np.take_along_axis(indices, order, axis=-1)

    scores = -np.sum(scores, axis=1)[:, None]
    indices = np.expand_dims(indices, axis=0)
    return convert_to_tensor(indices), convert_to_tensor(scores)


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths)

    inputs_np = np.asarray(convert_to_numpy(inputs))
    seq_np = np.asarray(convert_to_numpy(sequence_lengths))

    batch_size, max_seq_len, num_classes = inputs_np.shape

    def _np_log_softmax(x, axis=-1):
        max_x = np.max(x, axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
        return x - max_x - logsumexp

    inputs = _np_log_softmax(inputs_np, axis=-1)
    seqlen_mask = np.arange(max_seq_len)[None, :] >= seq_np[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that np.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = np.flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = np.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=np.int32
    )

    num_init_paths = np.min(np.array([num_classes, beam_width]))
    max_classes = np.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = np.where(max_classes == mask_index, _pad, max_classes)
    init_paths[:, :num_init_paths, 0] = init_classes

    init_scores = np.full(
        (batch_size, 2 * beam_width), -np.inf, dtype=inputs.dtype
    )
    init_scores[:, :num_init_paths] = np.take_along_axis(
        inputs[:, 0], max_classes, axis=1
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = np.repeat(paths, num_classes, axis=0)
        scores = np.repeat(scores, num_classes)
        masked = np.repeat(masked, num_classes)

        path_tail_index = np.argmax(paths == _pad, axis=1)
        paths_arange = np.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = np.where(path_tail_index == 0, _pad, path_tails)

        classes = np.arange(num_classes)
        classes[mask_index] = _pad
        classes = np.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = np.where(masked_repeat, _pad, classes)
        paths[paths_arange, path_tail_index] = classes

        x = np.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = np.max(scores)
        scores_exp = np.exp(scores - scores_max)
        new_scores = np.zeros_like(scores)
        np.add.at(new_scores, unique_inverse, scores_exp)
        return np.log(new_scores) + scores_max

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)

        emit_scores = np.where(masked, -np.inf, scores)
        mask_scores = np.where(masked, scores, -np.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = np.logaddexp(emit_scores, mask_scores)
        top_indices = np.argsort(total_scores, kind="stable")[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = np.tile(paths, (2, 1))
        scores = np.concatenate([emit_scores, mask_scores])
        masked = np.concatenate(
            [np.zeros(beam_width, bool), np.ones(beam_width, bool)]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x
        if not seqlen_mask:
            paths, scores, masked = _decode_step(paths, scores, masked, x)
        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        def np_scan_only_carry(f, init, xs):
            carry = init
            for x in zip(*xs):
                carry, y = f(carry, x)
            return carry, None

        (paths, scores, masked), _ = np_scan_only_carry(
            _step,
            (init_paths, init_scores, init_masked),
            (inputs[1:], seqlen_mask[1:]),
        )

        paths, unique_inverse = np.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = np.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = np.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = np.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    results = [
        _decode_batch(p, s, m, i, sm)
        for p, s, m, i, sm in zip(
            init_paths, init_scores, init_masked, inputs, seqlen_mask
        )
    ]
    paths = np.stack([r[0] for r in results])
    scores = np.stack([r[1] for r in results])

    # convert classes back to the correct indices
    paths = np.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = np.transpose(paths, [1, 0, 2])
    return convert_to_tensor(paths), convert_to_tensor(scores)


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
    dtype = backend.result_type(str(np.asarray(convert_to_numpy(inputs)).dtype), "float32")
    inputs = _cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )



# ======================================================================
# Group: misc
# ======================================================================

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """MLX implementation of Unfold.

    Extract sliding local blocks from a **NCHW** batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W)  **required**.
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    k = _pair(kernel_size)
    d = _pair(dilation)
    p = _pair(padding)
    s = _pair(stride)

    N, C, H, W = input.shape

    # ---- padding ----
    if any(_ > 0 for _ in p):
        input = mx.pad(
            input, [(0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])], mode="constant"
        )

    # ---- spatial size ----
    oH = (input.shape[2] - (k[0] - 1) * d[0] - 1) // s[0] + 1
    oW = (input.shape[3] - (k[1] - 1) * d[1] - 1) // s[1] + 1

    i0 = mx.arange(0, oH) * s[0]
    j0 = mx.arange(0, oW) * s[1]

    L = oH * oW

    # ---- gather each (kH, kW) window slice, then stack ----
    # patches[:, :, idx, jdx, :] = input[:, :, i + idx*d[0], j + jdx*d[1]]
    # each gathered slice has shape (N, C, L); we collect kH*kW of them and
    # reshape to (N, C, kH, kW, L) then (N, C*kH*kW, L).
    row_blocks = []
    for idx in range(k[0]):
        h_idx = i0 + idx * d[0]  # (oH,)
        col_blocks = []
        for jdx in range(k[1]):
            w_idx = j0 + jdx * d[1]  # (oW,)
            # input[:, :, h_idx[:, None], w_idx[None, :]] -> (N, C, oH, oW)
            slice_ = input[:, :, h_idx[:, None], w_idx[None, :]]
            col_blocks.append(mx.reshape(slice_, (N, C, L)))
        row_blocks.append(mx.stack(col_blocks, axis=2))  # (N, C, kW, L)
    patches = mx.stack(row_blocks, axis=2)  # (N, C, kH, kW, L)

    # ---- reshape -> (N, C*kH*kW, L) ----
    return mx.reshape(patches, (N, C * k[0] * k[1], L))


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """MLX implementation of Fold (col2im).

    Combine an array of sliding local blocks into a large tensor.

    Args:
        x: 3-D tensor, shape (N, C*kH*kW, L)  **required**.
        output_size: int or (oH, oW)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 1
        stride: int or (sH, sW), default 1

    Returns:
        4-D tensor, shape (N, C, oH, oW)
    """

    def _pair(val):
        return (val, val) if isinstance(val, int) else val

    oH, oW = _pair(output_size)
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    pH, pW = _pair(padding)
    sH, sW = _pair(stride)

    x = convert_to_tensor(x)
    N, CKK, L = x.shape
    C = CKK // (kH * kW)

    # Number of output patches along each dimension
    nH = (oH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    nW = (oW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # Reshape: (N, C*kH*kW, L) -> (N, C, kH, kW, nH, nW)
    x = mx.reshape(x, (N, C, kH, kW, nH, nW))

    # Padded output size
    oH_pad = oH + 2 * pH
    oW_pad = oW + 2 * pW

    output = mx.zeros((N, C, oH_pad, oW_pad), dtype=x.dtype)

    # Scatter-add each kernel tap onto the padded output grid. Overlapping
    # windows must accumulate (col2im sum), and `arr.at[idx].add(upd)` is the
    # MLX primitive that correctly handles duplicate-index updates.
    for i in range(kH):
        for j in range(kW):
            h_indices = i * dH + mx.arange(0, nH) * sH  # (nH,)
            w_indices = j * dW + mx.arange(0, nW) * sW  # (nW,)
            hi = h_indices[:, None]  # (nH, 1)
            wj = w_indices[None, :]  # (1, nW)
            x_patch = x[:, :, i, j, :, :]  # (N, C, nH, nW)
            output = output.at[:, :, hi, wj].add(x_patch)

    # Remove padding
    if pH > 0 or pW > 0:
        output = output[:, :, pH : oH_pad - pH, pW : oW_pad - pW]

    return output



def depth_to_space(x, block_size, data_format="channels_last"):
    """MLX implementation of depth_to_space (pixel shuffle).

    Rearranges data from depth into blocks of spatial data.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H*block_size, W*block_size, C/block_size**2)
        for channels_last or (N, C/block_size**2, H*block_size, W*block_size)
        for channels_first.
    """
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, H, W, C) -> (N, H, W, block_size, block_size, new_C)
        x = mx.reshape(x, (n, h, w, block_size, block_size, new_c))
        # Transpose to (N, H, bH, W, bW, new_C) to interleave spatial blocks.
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        # Reshape to the final spatial dimensions.
        x = mx.reshape(x, (n, h * block_size, w * block_size, new_c))
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_c = c // (block_size**2)
        # Reshape: (N, C, H, W) -> (N, new_C, block_size, block_size, H, W)
        x = mx.reshape(x, (n, new_c, block_size, block_size, h, w))
        # Transpose: (N, C, bH, bW, H, W) -> (N, C, H, bH, W, bW)
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        # Reshape: (N, C, H, bH, W, bW) -> (N, C, H*bH, W*bW)
        x = mx.reshape(x, (n, new_c, h * block_size, w * block_size))
    return x


def space_to_depth(x, block_size, data_format="channels_last"):
    """MLX implementation of space_to_depth (pixel unshuffle).

    Rearranges blocks of spatial data into depth.

    Args:
        x: 4-D tensor with shape (N, H, W, C) for channels_last or
            (N, C, H, W) for channels_first.
        block_size: An integer specifying the block size.
        data_format: "channels_last" or "channels_first".

    Returns:
        A tensor with shape (N, H/block_size, W/block_size, C*block_size**2)
        for channels_last or (N, C*block_size**2, H/block_size, W/block_size)
        for channels_first.
    """
    if data_format == "channels_last":
        # NHWC format
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, H, W, C) -> (N, new_H, bH, new_W, bW, C)
        x = mx.reshape(x, (n, new_h, block_size, new_w, block_size, c))
        # Transpose: -> (N, new_H, new_W, bH, bW, C)
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        # Reshape: -> (N, new_H, new_W, C*bH*bW)
        x = mx.reshape(x, (n, new_h, new_w, c * block_size**2))
    else:
        # NCHW format
        n, c, h, w = x.shape
        new_h = h // block_size
        new_w = w // block_size
        # Reshape: (N, C, H, W) -> (N, C, new_H, bH, new_W, bW)
        x = mx.reshape(x, (n, c, new_h, block_size, new_w, block_size))
        # Transpose: -> (N, C, bH, bW, new_H, new_W)
        x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
        # Reshape: -> (N, C*bH*bW, new_H, new_W)
        x = mx.reshape(x, (n, c * block_size**2, new_h, new_w))
    return x



