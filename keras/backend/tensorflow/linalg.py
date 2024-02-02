import tensorflow as tf


def cholesky(a):
    out = tf.linalg.cholesky(a)
    # tf.linalg.cholesky simply returns NaNs for non-positive definite matrices
    return tf.debugging.check_numerics(out, "Cholesky")


def det(a):
    return tf.linalg.det(a)


def inv(a):
    return tf.linalg.inv(a)


def solve(a, b):
    # tensorflow.linalg.solve only supports same rank inputs
    if tf.rank(b) == tf.rank(a) - 1:
        b = tf.expand_dims(b, axis=-1)
        return tf.squeeze(tf.linalg.solve(a, b), axis=-1)
    return tf.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    return tf.linalg.triangular_solve(a, b, lower=lower)
