import tensorflow as tf


def scatter(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)
