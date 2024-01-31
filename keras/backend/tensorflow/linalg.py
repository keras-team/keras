import tensorflow as tf


def cholesky(a):
    return tf.linalg.cholesky(a)


def det(a):
    return tf.linalg.det(a)


def inv(a):
    return tf.linalg.inv(a)


def solve(a, b):
    return tf.linalg.solve(a, b)
