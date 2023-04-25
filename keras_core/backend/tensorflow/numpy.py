import tensorflow as tf
from tensorflow.experimental import numpy as tfnp


def add(x1, x2):
    return tfnp.add(x1, x2)


def subtract(x1, x2):
    return tfnp.subtract(x1, x2)


def matmul(x1, x2):
    return tfnp.matmul(x1, x2)


def multiply(x1, x2):
    return tfnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    return tfnp.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return tfnp.max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype="float32"):
    with tf.init_scope():
        return tf.ones(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    with tf.init_scope():
        return tf.zeros(shape, dtype=dtype)
