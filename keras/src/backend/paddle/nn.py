import paddle
import paddle.nn.functional as F

from keras.src.backend.paddle.core import convert_to_tensor


def relu(x):
    return F.relu(convert_to_tensor(x))


def relu6(x):
    return F.relu6(convert_to_tensor(x))


def sigmoid(x):
    return F.sigmoid(convert_to_tensor(x))


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        shape = x.shape
        x = x.flatten()
        x = F.softmax(x, axis=0)
        return x.reshape(shape)
    return F.softmax(x, axis=axis)


def softplus(x):
    return F.softplus(convert_to_tensor(x))


def softsign(x):
    return F.softsign(convert_to_tensor(x))


def silu(x):
    return F.silu(convert_to_tensor(x))


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return paddle.log(F.sigmoid(x))


def leaky_relu(x, negative_slope=0.2):
    return F.leaky_relu(convert_to_tensor(x), negative_slope=negative_slope)


def prelu(x, alpha):
    return F.prelu(convert_to_tensor(x), convert_to_tensor(alpha))


def elu(x, alpha=1.0):
    return F.elu(convert_to_tensor(x), alpha=alpha)


def selu(x):
    return F.selu(convert_to_tensor(x))


def gelu(x, approximate=False):
    return F.gelu(convert_to_tensor(x), approximate=approximate)


def celu(x, alpha=1.0):
    return F.celu(convert_to_tensor(x), alpha=alpha)


def tanh(x):
    return paddle.tanh(convert_to_tensor(x))


def hard_sigmoid(x):
    return F.hardsigmoid(convert_to_tensor(x))


def hard_silu(x):
    return F.hardswish(convert_to_tensor(x))


def hard_tanh(x):
    return F.hardtanh(convert_to_tensor(x))


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    x = convert_to_tensor(x, dtype="int64")
    out = F.one_hot(x, num_classes)
    if axis != -1 and axis != out.ndim - 1:
        out = paddle.moveaxis(out, -1, axis)
    if sparse:
        import scipy.sparse as sp

        out_np = out.numpy()
        return sp.csr_matrix(out_np.reshape(-1, num_classes)).reshape(
            out_np.shape
        )
    return paddle.cast(out, dtype)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        shape = x.shape
        x = x.flatten()
        r = F.log_softmax(x, axis=0)
        return r.reshape(shape)
    return F.log_softmax(x, axis=axis)


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return F.softshrink(x, threshold=threshold)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return paddle.where(paddle.abs(x) > threshold, x, paddle.zeros_like(x))


def tanh_shrink(x):
    return convert_to_tensor(x) - paddle.tanh(convert_to_tensor(x))


def sparsemax(x, axis=-1):
    logits = convert_to_tensor(x)
    logits_sorted = paddle.sort(logits, axis=axis, descending=True)
    logits_cumsum = paddle.cumsum(logits_sorted, axis=axis)
    r = paddle.arange(1, paddle.shape(logits)[axis] + 1, dtype=logits.dtype)
    r_shape = [1] * len(logits.shape)
    r_shape[axis] = -1
    r = paddle.reshape(r, r_shape)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    k = paddle.sum(support.cast("int32"), axis=axis, keepdim=True)
    sum_selected = paddle.take_along_axis(logits_cumsum, k - 1, axis=axis)
    tau = (sum_selected - 1) / k.cast(logits.dtype)
    output = paddle.clip(logits - tau, min=0.0)
    return output


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    return 0.5 * (x + paddle.sqrt(x * x + b))


def sparse_plus(x):
    x = convert_to_tensor(x)
    return paddle.where(
        x < -1,
        paddle.zeros_like(x),
        paddle.where(x > 1, x, 0.25 * (x + 1) ** 2),
    )


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return paddle.where(
        x < -1,
        paddle.zeros_like(x),
        paddle.where(x > 1, paddle.ones_like(x), 0.5 * x + 0.5),
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    a, b = paddle.chunk(x, 2, axis=axis)
    return a * F.sigmoid(b)


def threshold(x, threshold_value, value):
    x = convert_to_tensor(x)
    return paddle.where(x > threshold_value, x, paddle.full_like(x, value))


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    x = convert_to_tensor(x)
    reduction_axis = [i for i in range(x.ndim) if i != (axis % x.ndim)]
    if not reduction_axis:
        reduction_axis = 0
    outputs = paddle.amax(
        one_hot(x.cast("int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs


def _standardize_tuple(x, n, name):
    if isinstance(x, int):
        return (x,) * n
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (x[0],) * n
        if len(x) == n:
            return tuple(x)
        raise ValueError(f"`{name}` should have length 1 or {n}. Received: {x}")
    raise ValueError(f"`{name}` should be int or tuple. Received: {x}")


def _to_channels_first(x, data_format):
    """Convert channels_last to channels_first."""
    if data_format == "channels_last":
        ndim = x.ndim
        if ndim == 3:  # (B, L, C) -> (B, C, L)
            return paddle.transpose(x, [0, 2, 1])
        elif ndim == 4:  # (B, H, W, C) -> (B, C, H, W)
            return paddle.transpose(x, [0, 3, 1, 2])
        elif ndim == 5:  # (B, D, H, W, C) -> (B, C, D, H, W)
            return paddle.transpose(x, [0, 4, 1, 2, 3])
    return x


def _to_channels_last(x, data_format):
    """Convert channels_first to channels_last."""
    if data_format == "channels_last":
        ndim = x.ndim
        if ndim == 3:
            return paddle.transpose(x, [0, 2, 1])
        elif ndim == 4:
            return paddle.transpose(x, [0, 2, 3, 1])
        elif ndim == 5:
            return paddle.transpose(x, [0, 2, 3, 4, 1])
    return x


def _conv_padding(padding, kernel_size, strides, dilation_rate):
    """Compute padding values for paddle conv."""
    if isinstance(padding, str):
        if padding == "valid":
            return (
                [0] * len(kernel_size)
                if isinstance(kernel_size, (list, tuple))
                else [0]
            )
        elif padding == "same":
            # Paddle supports "same" padding directly in some cases
            return "same"
    if isinstance(padding, int):
        return (
            [padding] * len(kernel_size)
            if isinstance(kernel_size, (list, tuple))
            else [padding]
        )
    return list(padding)


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
    orig_dtype = inputs.dtype
    # Cast unsupported dtypes for CPU
    _unsupported = {
        paddle.float16,
        paddle.bfloat16,
        paddle.int64,
        paddle.int32,
        paddle.int16,
        paddle.int8,
        paddle.uint8,
        paddle.bool,
    }
    if inputs.dtype in _unsupported:
        inputs = inputs.cast("float32")
        kernel = kernel.cast("float32")
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    # Convert to channels_first for paddle
    inputs = _to_channels_first(inputs, data_format)

    # Kernel: Keras [*kernel_size, in_channels, out_channels]
    # Paddle: [out_channels, in_channels, *kernel_size]
    perm = [num_spatial + 1, num_spatial] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)

    if padding == "same":
        pad_mode = "same"
    else:
        pad_mode = _conv_padding(
            padding, kernel.shape[2:], strides, dilation_rate
        )

    if num_spatial == 1:
        out = F.conv1d(
            inputs,
            kernel,
            stride=strides[0],
            padding=pad_mode,
            dilation=dilation_rate[0],
        )
    elif num_spatial == 2:
        out = F.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
        )
    elif num_spatial == 3:
        out = F.conv3d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(f"Unsupported number of spatial dims: {num_spatial}")

    out = _to_channels_last(out, data_format)
    if out.dtype != orig_dtype:
        out = out.cast(orig_dtype)
    return out


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    # Kernel: Keras [kernel_size..., in_channels, depth_multiplier]
    # Paddle depthwise: [in_channels * depth_multiplier, 1, *kernel_size]
    perm = [num_spatial, num_spatial + 1] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)
    kernel_shape = paddle.shape(kernel)
    new_kernel_shape = paddle.concat(
        [
            (kernel_shape[0] * kernel_shape[1]).reshape([1]),
            paddle.to_tensor([1], dtype=kernel_shape.dtype),
            kernel_shape[2:],
        ]
    )
    kernel = paddle.reshape(kernel, new_kernel_shape)

    in_channels = inputs.shape[1]
    groups = in_channels

    if padding == "same":
        pad_mode = "same"
    else:
        pad_mode = _conv_padding(
            padding, kernel.shape[2:], strides, dilation_rate
        )

    if num_spatial == 1:
        out = F.conv1d(
            inputs,
            kernel,
            stride=strides[0],
            padding=pad_mode,
            dilation=dilation_rate[0],
            groups=groups,
        )
    elif num_spatial == 2:
        out = F.conv2d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    elif num_spatial == 3:
        out = F.conv3d(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            dilation=dilation_rate,
            groups=groups,
        )
    else:
        raise ValueError(f"Unsupported number of spatial dims: {num_spatial}")

    return _to_channels_last(out, data_format)


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    # Depthwise convolution
    x = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )
    # Pointwise convolution (1x1)
    x = conv(
        x,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=1,
    )
    return x


def conv_transpose(
    inputs,
    kernel,
    strides,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    _unsupported = {
        paddle.float16,
        paddle.bfloat16,
        paddle.int64,
        paddle.int32,
        paddle.int16,
        paddle.int8,
        paddle.uint8,
        paddle.bool,
    }
    if inputs.dtype in _unsupported:
        inputs = inputs.cast("float32")
        kernel = kernel.cast("float32")
    num_spatial = inputs.ndim - 2

    strides = _standardize_tuple(strides, num_spatial, "strides")
    dilation_rate = _standardize_tuple(
        dilation_rate, num_spatial, "dilation_rate"
    )

    if output_padding is None:
        output_padding = 0
    output_padding = _standardize_tuple(
        output_padding, num_spatial, "output_padding"
    )

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    # Kernel transpose: Keras [*kernel_size, out_channels, in_channels]
    # Paddle: [in_channels, out_channels, *kernel_size]
    perm = [num_spatial + 1, num_spatial] + list(range(num_spatial))
    kernel = paddle.transpose(kernel, perm)

    if padding == "same":
        pad_mode = "same"
    else:
        pad_mode = _conv_padding(
            padding, kernel.shape[2:], strides, dilation_rate
        )

    if num_spatial == 1:
        out = F.conv1d_transpose(
            inputs,
            kernel,
            stride=strides[0],
            padding=pad_mode,
            output_padding=output_padding[0],
            dilation=dilation_rate[0],
        )
    elif num_spatial == 2:
        out = F.conv2d_transpose(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            output_padding=output_padding,
            dilation=dilation_rate,
        )
    elif num_spatial == 3:
        out = F.conv3d_transpose(
            inputs,
            kernel,
            stride=strides,
            padding=pad_mode,
            output_padding=output_padding,
            dilation=dilation_rate,
        )
    else:
        raise ValueError(f"Unsupported number of spatial dims: {num_spatial}")

    return _to_channels_last(out, data_format)


def _pool(inputs, pool_size, strides, padding, data_format, pool_type):
    inputs = convert_to_tensor(inputs)
    num_spatial = inputs.ndim - 2

    pool_size = _standardize_tuple(pool_size, num_spatial, "pool_size")
    strides = _standardize_tuple(strides, num_spatial, "strides")

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    if padding == "same":
        pad_mode = "same"
    else:
        pad_mode = 0

    if pool_type == "max":
        pool_fn = [F.max_pool1d, F.max_pool2d, F.max_pool3d][num_spatial - 1]
    else:
        pool_fn = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][num_spatial - 1]

    if num_spatial == 1:
        out = pool_fn(
            inputs,
            kernel_size=pool_size[0],
            stride=strides[0],
            padding=pad_mode,
        )
    else:
        out = pool_fn(
            inputs, kernel_size=pool_size, stride=strides, padding=pad_mode
        )

    return _to_channels_last(out, data_format)


def avg_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    return _pool(inputs, pool_size, strides, padding, data_format, "avg")


def max_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    return _pool(inputs, pool_size, strides, padding, data_format, "max")


def average_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    return avg_pool(inputs, pool_size, strides, padding, data_format)


def adaptive_pool(inputs, output_size, data_format, pool_type):
    inputs = convert_to_tensor(inputs)
    num_spatial = inputs.ndim - 2

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    inputs = _to_channels_first(inputs, data_format)

    if isinstance(output_size, int):
        output_size = [output_size] * num_spatial

    if pool_type == "max":
        pool_fn = [
            F.adaptive_max_pool1d,
            F.adaptive_max_pool2d,
            F.adaptive_max_pool3d,
        ][num_spatial - 1]
    else:
        pool_fn = [
            F.adaptive_avg_pool1d,
            F.adaptive_avg_pool2d,
            F.adaptive_avg_pool3d,
        ][num_spatial - 1]

    if num_spatial == 1:
        out = pool_fn(inputs, output_size=output_size[0])
    else:
        out = pool_fn(inputs, output_size=output_size)

    return _to_channels_last(out, data_format)


def adaptive_avg_pool(inputs, output_size, data_format=None):
    return adaptive_pool(inputs, output_size, data_format, "avg")


def adaptive_average_pool(inputs, output_size, data_format=None):
    return adaptive_avg_pool(inputs, output_size, data_format)


def adaptive_max_pool(inputs, output_size, data_format=None):
    return adaptive_pool(inputs, output_size, data_format, "max")


def global_average_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        # Reduce over all spatial dims (everything except batch and channel)
        axes = list(range(1, inputs.ndim - 1))
    else:
        # Reduce over spatial dims (everything except batch and channel)
        axes = list(range(2, inputs.ndim))

    return paddle.mean(inputs, axis=axes)


def global_max_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)

    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        axes = list(range(1, inputs.ndim - 1))
    else:
        axes = list(range(2, inputs.ndim))

    return paddle.max(inputs, axis=axes)


def moments(inputs, axes, keepdims=False, synchronized=False):
    inputs = convert_to_tensor(inputs)
    mean = paddle.mean(inputs, axis=axes, keepdim=keepdims)
    variance = paddle.var(inputs, axis=axes, keepdim=keepdims, unbiased=False)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)

    ndim = x.ndim
    axis = axis + ndim if axis < 0 else axis
    shape = [1] * ndim
    shape[axis] = -1

    mean = paddle.reshape(mean, shape)
    variance = paddle.reshape(variance, shape)
    x_norm = (x - mean) / paddle.sqrt(variance + epsilon)
    if scale is not None:
        x_norm = x_norm * paddle.reshape(convert_to_tensor(scale), shape)
    if offset is not None:
        x_norm = x_norm + paddle.reshape(convert_to_tensor(offset), shape)
    return x_norm


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
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    inputs_shape = paddle.shape(inputs)
    batch_size = inputs_shape[0]
    max_length = inputs_shape[1]
    num_classes = inputs.shape[2]

    if strategy == "greedy":
        indices = paddle.argmax(inputs, axis=-1).cast("int32")
        scores = paddle.max(inputs, axis=-1)

        seqlen_mask = paddle.arange(max_length).unsqueeze(0)
        seqlen_mask = seqlen_mask >= sequence_lengths.unsqueeze(1)

        blank_idx = num_classes - 1 if mask_index == -1 else mask_index
        indices = paddle.where(
            seqlen_mask, paddle.to_tensor(blank_idx, dtype="int32"), indices
        )
        scores = paddle.where(seqlen_mask, paddle.zeros_like(scores), scores)

        if merge_repeated:
            repeat = indices[:, 1:] == indices[:, :-1]
            zeros = paddle.zeros([batch_size, 1], dtype="bool")
            repeat = paddle.concat([zeros, repeat], axis=1)
            indices = paddle.where(
                repeat, paddle.to_tensor(blank_idx, dtype="int32"), indices
            )

        invalid_mask = indices == blank_idx
        indices = paddle.where(
            invalid_mask, paddle.to_tensor(-1, dtype="int32"), indices
        )

        order = paddle.arange(max_length).unsqueeze(0)
        order = paddle.broadcast_to(order, [batch_size, max_length])
        order = paddle.where(
            invalid_mask, paddle.to_tensor(max_length, dtype="int32"), order
        )
        order = paddle.argsort(order, axis=-1)
        indices = paddle.take_along_axis(indices, order, axis=-1)

        scores = -paddle.sum(scores, axis=1).unsqueeze(1)
        indices = indices.unsqueeze(0)
        return indices, scores
    raise NotImplementedError(
        f"CTC decode strategy '{strategy}' is not supported"
    )


def psnr(x1, x2, max_val):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    mse = paddle.mean((x1 - x2) ** 2)
    return 10.0 * paddle.log10(max_val**2 / (mse + 1e-10))


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
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)

    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    scores = paddle.matmul(query, key, transpose_y=True) * scale

    if attn_logits_soft_cap is not None:
        scores = attn_logits_soft_cap * paddle.tanh(
            scores / attn_logits_soft_cap
        )

    if bias is not None:
        scores = scores + convert_to_tensor(bias)

    if mask is not None:
        mask = convert_to_tensor(mask)
        scores = paddle.where(
            mask == 0, paddle.full_like(scores, float("-inf")), scores
        )

    if is_causal:
        seq_len = paddle.shape(query)[-2]
        causal_mask = paddle.tril(
            paddle.ones([seq_len, seq_len], dtype="int32")
        ).cast("bool")
        scores = paddle.where(
            causal_mask, scores, paddle.full_like(scores, float("-inf"))
        )

    weights = F.softmax(scores, axis=-1)
    return paddle.matmul(weights, value)


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    if from_logits:
        output = F.sigmoid(output)
    output = paddle.clip(output, min=1e-7, max=1 - 1e-7)
    return F.binary_cross_entropy(output, target)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    if from_logits:
        return F.cross_entropy(
            output, target, soft_label=True, reduction="none", axis=axis
        )
    return -paddle.sum(target * paddle.log(output + 1e-7), axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype="int64")
    output = convert_to_tensor(output)
    if from_logits:
        return F.cross_entropy(
            output, target, soft_label=False, reduction="none", axis=axis
        )
    target = target.unsqueeze(axis)
    probs = paddle.take_along_axis(output, target, axis=axis).squeeze(axis=axis)
    return -paddle.log(probs + 1e-7)


def ctc_loss(target, output, target_length, output_length, mask_value=0):
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output, dtype="float32")
    target_length = convert_to_tensor(target_length, dtype="int64")
    output_length = convert_to_tensor(output_length, dtype="int64")
    loss = paddle.nn.CTCLoss(blank=mask_value, reduction="none")
    return loss(output, target, output_length, target_length)


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Paddle implementation of Fold.
    Combine an array of sliding local blocks into a large tensor (col2im).

    Args:
        x: 3-D tensor, shape (N, C*kH*kW, L)
        output_size: int or (oH, oW)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        4-D tensor, shape (N, C, oH, oW)
    """
    x = convert_to_tensor(x)
    return F.fold(
        x,
        output_sizes=output_size,
        kernel_sizes=kernel_size,
        dilations=dilation,
        paddings=padding,
        strides=stride,
    )


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Paddle implementation of Unfold.
    Extract sliding local blocks from a **NCHW** batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """
    return F.unfold(
        input,
        kernel_sizes=kernel_size,
        dilations=dilation,
        paddings=padding,
        strides=stride,
    )


def depth_to_space(inputs, block_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        inputs = paddle.transpose(inputs, [0, 3, 1, 2])

    b, c, h, w = inputs.shape
    new_c = c // (block_size**2)
    inputs = paddle.reshape(inputs, [b, new_c, block_size, block_size, h, w])
    inputs = paddle.transpose(inputs, [0, 1, 4, 2, 5, 3])
    out = paddle.reshape(inputs, [b, new_c, h * block_size, w * block_size])

    if data_format == "channels_last":
        return paddle.transpose(out, [0, 2, 3, 1])
    return out


def space_to_depth(inputs, block_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    from keras.src.backend.config import standardize_data_format

    data_format = standardize_data_format(data_format)

    if data_format == "channels_last":
        inputs = paddle.transpose(inputs, [0, 3, 1, 2])

    b, c, h, w = inputs.shape
    new_h = h // block_size
    new_w = w // block_size
    new_c = c * (block_size**2)
    inputs = paddle.reshape(
        inputs, [b, c, new_h, block_size, new_w, block_size]
    )
    inputs = paddle.transpose(inputs, [0, 3, 5, 1, 2, 4])
    out = paddle.reshape(inputs, [b, new_c, new_h, new_w])

    if data_format == "channels_last":
        return paddle.transpose(out, [0, 2, 3, 1])
    return out
