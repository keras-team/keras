"""Class to specify an input's shape/dtype/value range.
"""

import tensorflow as tf


class InputSpec:
    def __init__(self, shape, dtype="float32", range=None):
        self.shape = shape
        self.dtype = dtype
        self.range = range


def spec_to_value(spec):
    shape = spec.shape
    dtype = spec.dtype
    rg = spec.range or [0, 1]
    if dtype == "string":
        return tf.constant(
            ["some string" for _ in range(shape[0])], dtype="string"
        )
    return tf.random.stateless_uniform(
        shape, seed=[123, 1], minval=rg[0], maxval=rg[1], dtype=dtype
    )
