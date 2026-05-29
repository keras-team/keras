import paddle
import paddle.nn.functional as F

from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import is_tensor
from keras.src.backend.paddle.core import shape
from keras.src.backend.paddle.core import to_paddle_dtype


def relu(x):
    return F.relu(convert_to_tensor(x))


def relu6(x):
    return F.relu6(convert_to_tensor(x))


def sigmoid(x):
    return F.sigmoid(convert_to_tensor(x))


def softmax(x, axis=-1):
    return F.softmax(convert_to_tensor(x), axis=axis)


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


def one_hot(x, num_classes, axis=-1, dtype="float32"):
    return F.one_hot(convert_to_tensor(x, dtype="int64"), num_classes)


def log_softmax(x, axis=-1):
    return F.log_softmax(convert_to_tensor(x), axis=axis)


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return F.softshrink(x, threshold=threshold)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return paddle.where(
        paddle.abs(x) > threshold, x, paddle.zeros_like(x)
    )


def tanh_shrink(x):
    return convert_to_tensor(x) - paddle.tanh(convert_to_tensor(x))


def sparsemax(x, axis=-1):
    raise NotImplementedError(
        "`sparsemax` is not supported with paddle backend"
    )


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    return 0.5 * (x + paddle.sqrt(x * x + b))


def sparse_plus(x):
    x = convert_to_tensor(x)
    return paddle.where(x < -1, paddle.zeros_like(x), paddle.where(x > 1, x, 0.25 * (x + 1) ** 2))


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return paddle.where(x < -1, paddle.zeros_like(x), paddle.where(x > 1, paddle.ones_like(x), 0.5 * x + 0.5))


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    a, b = paddle.chunk(x, 2, axis=axis)
    return a * F.sigmoid(b)


def threshold(x, threshold_value, value):
    x = convert_to_tensor(x)
    return paddle.where(x > threshold_value, x, paddle.full_like(x, value))


def multi_hot(x, num_classes, axis=-1, dtype="float32"):
    raise NotImplementedError(
        "`multi_hot` is not supported with paddle backend"
    )


def conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`conv` is not supported with paddle backend"
    )


def depthwise_conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`depthwise_conv` is not supported with paddle backend"
    )


def separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`separable_conv` is not supported with paddle backend"
    )


def conv_transpose(inputs, kernel, strides, padding="valid", output_padding=None, data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`conv_transpose` is not supported with paddle backend"
    )


def avg_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`avg_pool` is not supported with paddle backend"
    )


def max_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`max_pool` is not supported with paddle backend"
    )


def adaptive_avg_pool(inputs, output_size, data_format=None):
    raise NotImplementedError(
        "`adaptive_avg_pool` is not supported with paddle backend"
    )


def adaptive_max_pool(inputs, output_size, data_format=None):
    raise NotImplementedError(
        "`adaptive_max_pool` is not supported with paddle backend"
    )


def average_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`average_pool` is not supported with paddle backend"
    )


def global_average_pool(inputs, data_format=None):
    raise NotImplementedError(
        "`global_average_pool` is not supported with paddle backend"
    )


def global_max_pool(inputs, data_format=None):
    raise NotImplementedError(
        "`global_max_pool` is not supported with paddle backend"
    )


def moments(inputs, axes, keepdims=False, synchronized=False):
    inputs = convert_to_tensor(inputs)
    mean = paddle.mean(inputs, axis=axes, keepdim=keepdims)
    variance = paddle.var(inputs, axis=axes, keepdim=keepdims, unbiased=False)
    return mean, variance


def batch_normalization(x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)
    x_norm = (x - mean) / paddle.sqrt(variance + epsilon)
    if scale is not None:
        x_norm = x_norm * convert_to_tensor(scale)
    if offset is not None:
        x_norm = x_norm + convert_to_tensor(offset)
    return x_norm


def ctc_decode(inputs, input_lengths, strategy="greedy", beam_width=100, top_paths=1, merge_repeated=False, mask_value=-1):
    raise NotImplementedError(
        "`ctc_decode` is not supported with paddle backend"
    )


def psnr(x1, x2, max_val):
    raise NotImplementedError(
        "`psnr` is not supported with paddle backend"
    )


def dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, flash_attention=None, attn_logits_soft_cap=None):
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)

    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    scores = paddle.matmul(query, key, transpose_y=True) * scale

    if attn_logits_soft_cap is not None:
        scores = attn_logits_soft_cap * paddle.tanh(scores / attn_logits_soft_cap)

    if bias is not None:
        scores = scores + convert_to_tensor(bias)

    if mask is not None:
        mask = convert_to_tensor(mask)
        scores = paddle.where(mask == 0, paddle.full_like(scores, float('-inf')), scores)

    if is_causal:
        seq_len = query.shape[-2]
        causal_mask = paddle.tril(paddle.ones([seq_len, seq_len], dtype="bool"))
        scores = paddle.where(causal_mask, scores, paddle.full_like(scores, float('-inf')))

    weights = F.softmax(scores, axis=-1)
    return paddle.matmul(weights, value)


def conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    ndim = len(inputs.shape) - 2
    if isinstance(strides, int):
        strides = (strides,) * ndim
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * ndim

    if padding == "valid":
        pad = "valid"
    elif padding == "same":
        pad = "same"
    else:
        pad = padding

    if ndim == 1:
        return F.conv1d(inputs, kernel, stride=strides[0], padding=pad, dilation=dilation_rate[0])
    elif ndim == 2:
        return F.conv2d(inputs, kernel, stride=strides, padding=pad, dilation=dilation_rate)
    elif ndim == 3:
        return F.conv3d(inputs, kernel, stride=strides, padding=pad, dilation=dilation_rate)
    raise ValueError(f"Unsupported ndim: {ndim}")


def depthwise_conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate)
    if padding == "valid":
        pad = "valid"
    elif padding == "same":
        pad = "same"
    else:
        pad = padding
    groups = inputs.shape[1]
    return F.conv2d(inputs, kernel, stride=strides, padding=pad, dilation=dilation_rate, groups=groups)


def separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    x = depthwise_conv(inputs, depthwise_kernel, strides=strides, padding=padding, dilation_rate=dilation_rate)
    return conv(x, pointwise_kernel, strides=1, padding="valid")


def conv_transpose(inputs, kernel, strides, padding="valid", output_padding=None, data_format=None, dilation_rate=1):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    ndim = len(inputs.shape) - 2
    if isinstance(strides, int):
        strides = (strides,) * ndim

    if ndim == 1:
        return F.conv_transpose1d(inputs, kernel, stride=strides[0], padding=padding)
    elif ndim == 2:
        return F.conv_transpose2d(inputs, kernel, stride=strides, padding=padding)
    elif ndim == 3:
        return F.conv_transpose3d(inputs, kernel, stride=strides, padding=padding)
    raise ValueError(f"Unsupported ndim: {ndim}")


def avg_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    inputs = convert_to_tensor(inputs)
    ndim = len(inputs.shape) - 2
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndim
    if isinstance(strides, int):
        strides = (strides,) * ndim
    if ndim == 1:
        return F.avg_pool1d(inputs, pool_size[0], stride=strides[0], padding=padding)
    elif ndim == 2:
        return F.avg_pool2d(inputs, pool_size, stride=strides, padding=padding)
    elif ndim == 3:
        return F.avg_pool3d(inputs, pool_size, stride=strides, padding=padding)
    raise ValueError(f"Unsupported ndim: {ndim}")


def max_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    inputs = convert_to_tensor(inputs)
    ndim = len(inputs.shape) - 2
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndim
    if isinstance(strides, int):
        strides = (strides,) * ndim
    if ndim == 1:
        return F.max_pool1d(inputs, pool_size[0], stride=strides[0], padding=padding)
    elif ndim == 2:
        return F.max_pool2d(inputs, pool_size, stride=strides, padding=padding)
    elif ndim == 3:
        return F.max_pool3d(inputs, pool_size, stride=strides, padding=padding)
    raise ValueError(f"Unsupported ndim: {ndim}")


def adaptive_avg_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    ndim = len(inputs.shape) - 2
    if ndim == 1:
        return F.adaptive_avg_pool1d(inputs, output_size)
    elif ndim == 2:
        return F.adaptive_avg_pool2d(inputs, output_size)
    elif ndim == 3:
        return F.adaptive_avg_pool3d(inputs, output_size)
    raise ValueError(f"Unsupported ndim: {ndim}")


def adaptive_average_pool(inputs, output_size, data_format=None):
    return adaptive_avg_pool(inputs, output_size, data_format=data_format)


def adaptive_max_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    ndim = len(inputs.shape) - 2
    if ndim == 1:
        return F.adaptive_max_pool1d(inputs, output_size)
    elif ndim == 2:
        return F.adaptive_max_pool2d(inputs, output_size)
    elif ndim == 3:
        return F.adaptive_max_pool3d(inputs, output_size)
    raise ValueError(f"Unsupported ndim: {ndim}")


def average_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    return avg_pool(inputs, pool_size, strides, padding=padding, data_format=data_format)


def global_average_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)
    axes = list(range(2, len(inputs.shape)))
    return paddle.mean(inputs, axis=axes)


def global_max_pool(inputs, data_format=None):
    inputs = convert_to_tensor(inputs)
    axes = list(range(2, len(inputs.shape)))
    return paddle.max(inputs, axis=axes)


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    if from_logits:
        output = F.sigmoid(output)
    return F.binary_cross_entropy(output, target)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)
    if from_logits:
        output = F.softmax(output, axis=axis)
    return -paddle.sum(target * paddle.log(output + 1e-7), axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype="int64")
    output = convert_to_tensor(output)
    if from_logits:
        output = F.softmax(output, axis=axis)
    # Gather the probabilities at the target indices
    target_one_hot = F.one_hot(target, output.shape[axis])
    return -paddle.sum(target_one_hot * paddle.log(output + 1e-7), axis=axis)


def ctc_loss(target, output, target_length, output_length, mask_value=0):
    raise NotImplementedError("`ctc_loss` is not supported with paddle backend")


def ctc_decode(inputs, input_lengths, strategy="greedy", beam_width=100, top_paths=1, merge_repeated=False, mask_value=-1):
    raise NotImplementedError(
        "`ctc_decode` is not supported with paddle backend"
    )


def psnr(x1, x2, max_val):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    mse = paddle.mean((x1 - x2) ** 2)
    return 10.0 * paddle.log(max_val ** 2 / mse) / paddle.log(paddle.to_tensor(10.0))


def fold(x, output_shape, kernel_size, strides=1, padding="valid", dilation_rate=1, data_format=None):
    raise NotImplementedError("`fold` is not supported with paddle backend")


def unfold(x, kernel_size, strides=1, padding="valid", dilation_rate=1, data_format=None):
    raise NotImplementedError("`unfold` is not supported with paddle backend")


def depth_to_space(inputs, block_size, data_format=None):
    raise NotImplementedError(
        "`depth_to_space` is not supported with paddle backend"
    )


def space_to_depth(inputs, block_size, data_format=None):
    raise NotImplementedError(
        "`space_to_depth` is not supported with paddle backend"
    )
