import numpy as np

from .control_flow import learning_phase


def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)


def relu(x, alpha=0., max_value=None, threshold=0.):
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def softplus(x):
    return np.log(1. + np.exp(x))


def categorical_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = softmax(output)
    else:
        output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.sum(target * -np.log(output), axis=-1, keepdims=False)


def binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        output = np.clip(output, 1e-7, 1 - 1e-7)
        output = np.log(output / (1 - output))
    return (target * -np.log(sigmoid(output)) +
            (1 - target) * -np.log(1 - sigmoid(output)))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)


def tanh(x):
    return np.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is None:
        noise_shape = x.shape
    if learning_phase():
        noise = np.random.choice([0, 1],
                                 noise_shape,
                                 replace=True,
                                 p=[level, 1 - level])
        return x * noise / (1 - level)
    else:
        return x


def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)
