import numpy as np


def relu(x):
    from openvino.runtime.opset14 import relu
    return relu(x)


def relu6(x):
    raise NotImplementedError(
        "`relu6` is not supported with openvino backend"
    )


def sigmoid(x):
    from openvino.runtime.opset14 import sigmoid
    return sigmoid(x)


def tanh(x):
    from openvino.runtime.opset14 import tanh
    return tanh(x)


def softplus(x):
    from openvino.runtime.opset14 import softplus
    return softplus(x)


def softsign(x):
    from openvino.runtime.opset14 import softsign
    return softsign(x)


def silu(x):
    from openvino.runtime.opset14 import sigmoid
    from openvino.runtime.opset14 import multiply
    return multiply(x, sigmoid(x))


def log_sigmoid(x):
    raise NotImplementedError(
        "`log_sigmoid` is not supported with openvino backend"
    )


def leaky_relu(x, negative_slope=0.2):
    raise NotImplementedError(
        "`leaky_relu` is not supported with openvino backend"
    )


def hard_sigmoid(x):
    from openvino.runtime.opset14 import hard_sigmoid
    alpha = 1 / np.array(6.0, dtype=np.float32)
    beta = np.array(0.5, dtype=np.float32)
    return hard_sigmoid(x, alpha, beta)


def hard_silu(x):
    from openvino.runtime.opset14 import multiply
    return multiply(x, hard_sigmoid(x))


def elu(x, alpha=1.0):
    from openvino.runtime.opset14 import elu
    return elu(x, alpha)


def selu(
        x,
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946,
):
    from openvino.runtime.opset14 import selu
    return selu(x, alpha, scale)


def gelu(x, approximate=True):
    from openvino.runtime.opset14 import gelu
    approximate_mode = "erf"
    if approximate:
        approximate_mode = "tanh"
    return gelu(x, approximate_mode)


def softmax(x, axis=None):
    from openvino.runtime.opset14 import softmax
    return softmax(x, axis)


def log_softmax(x, axis=None):
    from openvino.runtime.opset14 import log_softmax
    return log_softmax(x, axis)


def max_pool(
        inputs,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
):
    raise NotImplementedError(
        "`max_pool` is not supported with openvino backend"
    )


def average_pool(
        inputs,
        pool_size,
        strides,
        padding,
        data_format=None,
):
    raise NotImplementedError(
        "`average_pool` is not supported with openvino backend"
    )


def conv(
        inputs,
        kernel,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
):
    raise NotImplementedError(
        "`conv` is not supported with openvino backend"
    )


def depthwise_conv(
        inputs,
        kernel,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
):
    raise NotImplementedError(
        "`depthwise_conv` is not supported with openvino backend"
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
    raise NotImplementedError(
        "`separable_conv` is not supported with openvino backend"
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
    raise NotImplementedError(
        "`conv_transpose` is not supported with openvino backend"
    )


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    raise NotImplementedError(
        "`one_hot` is not supported with openvino backend"
    )


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    raise NotImplementedError(
        "`multi_hot` is not supported with openvino backend"
    )


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`categorical_crossentropy` is not supported with openvino backend"
    )


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`sparse_categorical_crossentropy` is not supported with openvino backend"
    )


def binary_crossentropy(target, output, from_logits=False):
    raise NotImplementedError(
        "`binary_crossentropy` is not supported with openvino backend"
    )


def moments(x, axes, keepdims=False, synchronized=False):
    raise NotImplementedError(
        "`moments` is not supported with openvino backend"
    )


def batch_normalization(
        x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    raise NotImplementedError(
        "`batch_normalization` is not supported with openvino backend"
    )


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    raise NotImplementedError(
        "`ctc_loss` is not supported with openvino backend"
    )


def ctc_decode(
        inputs,
        sequence_lengths,
        strategy="greedy",
        beam_width=100,
        top_paths=1,
        merge_repeated=True,
        mask_index=0,
):
    raise NotImplementedError(
        "`ctc_decode` is not supported with openvino backend"
    )


def psnr(x1, x2, max_val):
    raise NotImplementedError(
        "`psnr` is not supported with openvino backend"
    )
