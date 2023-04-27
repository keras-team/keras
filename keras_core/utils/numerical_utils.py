from keras_core import backend
from keras_core import operations as ops


def l2_normalize(x, axis=0):
    epsilon = backend.epsilon()
    square_sum = ops.sum(ops.square(x), axis=axis, keepdims=True)
    l2_norm = ops.reciprocal(ops.sqrt(ops.maximum(square_sum, epsilon)))
    return ops.multiply(x, l2_norm)
