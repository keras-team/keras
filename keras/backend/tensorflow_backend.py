import tensorflow as tf
import numpy as np
from .common import _FLOATX, _EPSILON

# INTERNAL UTILS

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = tf.Session('')
    return _SESSION


def _set_session(session):
    global _SESSION
    _SESSION = session


# VARIABLE MANIPULATION

def variable(value, dtype=_FLOATX, name=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    _get_session().run(v.initializer)
    return v


def shape(x):
    return x.get_shape()


def eval(x):
    '''Run a graph.
    '''
    return x.eval(session=_get_session())


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    return variable(np.ones(shape), dtype, name)


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    return tf.placeholder(dtype, shape=shape, name=name)


# LINEAR ALGEBRA

def dot(x, y):
    return tf.matmul(x, y)


def transpose(x):
    return tf.transpose(x)


# ELEMENT-WISE OPERATIONS

def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).

    Return array of int8 (0s and 1s).
    '''
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.int8)


def argmax(x, axis=-1):
    axis = axis % len(x.get_shape())
    return tf.argmax(x, axis)


def argmin(x, axis=-1):
    axis = axis % len(x.get_shape())
    return tf.argmin(x, axis)


def square(x):
    return tf.square(x)


def abs(x):
    return tf.abs(x)


def sqrt(x):
    x = tf.clip_by_value(x, 0.0, np.inf)
    return tf.sqrt(x)


def exp(x):
    return tf.exp(x)


def log(x):
    return tf.log(x)


def round(x):
    return tf.round(x)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return tf.clip_by_value(x, min_value, max_value)


def equal(x, y):
    return tf.equal(x, y)


def maximum(x, y):
    return tf.maximum(x, y)


def minimum(x, y):
    return tf.maximum(x, y)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    axis = axis % len(tensors[0].get_shape())
    return tf.concat(axis, tensors)


def reshape(x, shape):
    return tf.reshape(x, shape)


def permute_dimensions(x, pattern):
    '''Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    '''
    return tf.transpose(x, perm=pattern)


def repeat(x, n, axis=-1):
    tiling_pattern = [1 for _ in range(len(x.get_shape()))]
    tiling_pattern[axis] = n
    return tf.tile(x, tiling_pattern)


def flatten(x):
    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])
    return x


# VALUE MANIPULATION

def get_value(x):
    '''Technically the same as eval() for TF.
    '''
    return x.eval(session=_get_session())


def set_value(x, value):
    tf.assign(x, value).op.run(session=_get_session())


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[]):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            self.updates = [tf.assign(p, new_p) for (p, new_p) in updates]

    def __call__(self, inputs):
        names = [v.name for v in self.inputs]
        feed_dict = dict(zip(names, inputs))
        session = _get_session()
        updated = session.run(self.outputs + self.updates, feed_dict=feed_dict)
        return updated[:len(self.outputs)]


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    return tf.gradients(loss, variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states, go_backwards=False):
    '''TODO
    '''
    pass


def switch(condition, then_expression, else_expression):
    '''condition: scalar tensor.
    '''
    return tf.control_flow_ops.cond(condition,
                                    lambda: then_expression,
                                    lambda: else_expression)


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    '''ReLU. 

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, 0., max_value)
    x -= alpha * negative_part
    return x


def softmax(x):
    return tf.nn.softmax(x)


def softplus(x):
    return tf.nn.softplus(x)


def categorical_crossentropy(output, target, from_logits=False):
    '''Note: tf.nn.softmax_cross_entropy_with_logits
    expects logits, Keras expects probabilities.
    '''
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape())-1,
                                keep_dims=True)
        # manual computation of crossentropy
        output = tf.clip_by_value(output, _EPSILON, 1.-_EPSILON)
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape())-1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)


def binary_crossentropy(output, target, from_logits=False):
    '''Note: tf.nn.sigmoid_cross_entropy_with_logits
    expects logits, Keras expects probabilities.
    '''
    if not from_logits:
        # transform back to logits
        output = tf.clip_by_value(output, _EPSILON, 1.-_EPSILON)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    x = tf.clip_by_value(x, 0., 1.)
    return x


def tanh(x):
    return tf.nn.tanh(x)


def dropout(x, level, seed=None):
    retain_prob = 1. - level
    if seed is None:
        seed = np.random.randint(10e6)
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incomptability)
    return tf.nn.dropout(x * 1., retain_prob, seed=seed)


# CONVOLUTIONS


def conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th'):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    dim_ordering: whether to use Theano or TensorFlow dimension ordering
    in inputs/kernels/ouputs.
    '''
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    strides = (1,) + strides + (1,)

    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))
    elif dim_ordering == 'tf':
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))
    return x


def maxpool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th'):
    '''
    pool_size: tuple of 2 integers.
    strides: tuple of 2 integers.
    border_mode: one of "valid", "same".
    dim_ordering: one of "th", "tf".
    '''
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension, instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = tf.transpose(x, (0, 2, 3, 1))
        x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))
    elif dim_ordering == 'tf':
        x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))
    return x


# RANDOMNESS

def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype, seed=seed)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype, seed=seed)
