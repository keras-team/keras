import tensorflow as tf
from tensorflow import nn as tfnn


def relu(x):
    return tfnn.relu(x)


def relu6(x):
    return tfnn.relu6(x)


def sigmoid(x):
    return tfnn.sigmoid(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tfnn.softsign(x)


def silu(x, beta=1.0):
    return tfnn.silu(x, beta=beta)


def swish(x):
    return x * sigmoid(x)


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tfnn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)


def elu(x):
    return tfnn.elu(x)


def selu(x):
    return tfnn.selu(x)


def gelu(x, approximate=True):
    return tfnn.gelu(x, approximate)


def softmax(x, axis=None):
    return tfnn.softmax(x, axis=axis)


def log_softmax(x, axis=None):
    return tfnn.log_softmax(x, axis=axis)
