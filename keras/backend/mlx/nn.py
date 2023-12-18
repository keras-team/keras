import mlx.core as mx
import mlx.nn as nn

from keras.backend.config import epsilon
from keras.backend.mlx.core import convert_to_tensor
from keras.backend.mlx.numpy import clip


def relu(x):
    x = convert_to_tensor(x)
    return nn.relu(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return mx.sigmoid(x)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return mx.softmax(x, axis=axis)


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
        log_prob = output - mx.logsumexp(output, axis=axis)
    else:
        output = output / output.sum(axis=axis, keepdims=True)
        output = clip(output, epsilon(), 1-epsilon())
        log_prob = mx.log(output)

    return -(target * log_prob).sum(axis=axis)
