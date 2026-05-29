import paddle
import paddle.nn.functional as F

from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import shape


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
    raise NotImplementedError(
        "`moments` is not supported with paddle backend"
    )


def batch_normalization(x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3):
    raise NotImplementedError(
        "`batch_normalization` is not supported with paddle backend"
    )


def ctc_decode(inputs, input_lengths, strategy="greedy", beam_width=100, top_paths=1, merge_repeated=False, mask_value=-1):
    raise NotImplementedError(
        "`ctc_decode` is not supported with paddle backend"
    )


def psnr(x1, x2, max_val):
    raise NotImplementedError(
        "`psnr` is not supported with paddle backend"
    )


def dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, flash_attention=None):
    raise NotImplementedError(
        "`dot_product_attention` is not supported with paddle backend"
    )
