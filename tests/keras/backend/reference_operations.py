"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal
import scipy as sp
from keras.backend import floatx


def normalize_conv(func):
    def wrapper(*args, **kwargs):
        x = args[0]
        w = args[1]
        if x.ndim == 3:
            w = np.flipud(w)
            w = np.transpose(w, (1, 2, 0))
            if kwargs['data_format'] == 'channels_last':
                x = np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            w = np.fliplr(np.flipud(w))
            w = np.transpose(w, (2, 3, 0, 1))
            if kwargs['data_format'] == 'channels_last':
                x = np.transpose(x, (0, 3, 1, 2))
        else:
            w = np.flip(np.fliplr(np.flipud(w)), axis=2)
            w = np.transpose(w, (3, 4, 0, 1, 2))
            if kwargs['data_format'] == 'channels_last':
                x = np.transpose(x, (0, 4, 1, 2, 3))

        dilation_rate = kwargs.pop('dilation_rate', 1)
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate,) * (x.ndim - 2)
        for (i, d) in enumerate(dilation_rate):
            if d > 1:
                for j in range(w.shape[2 + i] - 1):
                    w = np.insert(w, 2 * j + 1, 0, axis=2 + i)

        y = func(x, w, **kwargs)

        if kwargs['data_format'] == 'channels_last':
            if y.ndim == 3:
                y = np.transpose(y, (0, 2, 1))
            elif y.ndim == 4:
                y = np.transpose(y, (0, 2, 3, 1))
            else:
                y = np.transpose(y, (0, 2, 3, 4, 1))

        return y

    return wrapper


@normalize_conv
def conv(x, w, padding, data_format):
    y = []
    for i in range(x.shape[0]):
        _y = []
        for j in range(w.shape[1]):
            __y = []
            for k in range(w.shape[0]):
                __y.append(signal.convolve(x[i, k], w[k, j], mode=padding))
            _y.append(np.sum(np.stack(__y, axis=-1), axis=-1))
        y.append(_y)
    y = np.array(y)
    return y


@normalize_conv
def depthwise_conv(x, w, padding, data_format):
    y = []
    for i in range(x.shape[0]):
        _y = []
        for j in range(w.shape[0]):
            __y = []
            for k in range(w.shape[1]):
                __y.append(signal.convolve(x[i, j], w[j, k], mode=padding))
            _y.append(np.stack(__y, axis=0))
        y.append(np.concatenate(_y, axis=0))
    y = np.array(y)
    return y


def separable_conv(x, w1, w2, padding, data_format):
    x2 = depthwise_conv(x, w1, padding=padding, data_format=data_format)
    return conv(x2, w2, padding=padding, data_format=data_format)


def conv_transpose(x, w, output_shape, padding, data_format, dilation_rate=1):
    if x.ndim == 4:
        w = np.fliplr(np.flipud(w))
        w = np.transpose(w, (0, 1, 3, 2))
    else:
        w = np.flip(np.fliplr(np.flipud(w)), axis=2)
        w = np.transpose(w, (0, 1, 2, 4, 3))

    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * (x.ndim - 2)
    for (i, d) in enumerate(dilation_rate):
        if d > 1:
            for j in range(w.shape[i] - 1):
                w = np.insert(w, 2 * j + 1, 0, axis=i)

    return conv(x, w, padding=padding, data_format=data_format)


conv1d = conv
conv2d = conv
conv3d = conv
depthwise_conv2d = depthwise_conv
separable_conv1d = separable_conv
separable_conv2d = separable_conv
conv2d_transpose = conv_transpose
conv3d_transpose = conv_transpose


def pool(x, pool_size, strides, padding, data_format, pool_mode):
    if data_format == 'channels_last':
        if x.ndim == 3:
            x = np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            x = np.transpose(x, (0, 3, 1, 2))
        else:
            x = np.transpose(x, (0, 4, 1, 2, 3))

    if padding == 'same':
        pad = [(0, 0), (0, 0)] + [(s // 2, s // 2) for s in pool_size]
        x = np.pad(x, pad, 'constant', constant_values=-np.inf)

    # indexing trick
    x = np.pad(x, [(0, 0), (0, 0)] + [(0, 1) for _ in pool_size],
               'constant', constant_values=0)

    if x.ndim == 3:
        y = [x[:, :, k:k1:strides[0]]
             for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0))]
    elif x.ndim == 4:
        y = []
        for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0)):
            for (l, l1) in zip(range(pool_size[1]), range(-pool_size[1], 0)):
                y.append(x[:, :, k:k1:strides[0], l:l1:strides[1]])
    else:
        y = []
        for (k, k1) in zip(range(pool_size[0]), range(-pool_size[0], 0)):
            for (l, l1) in zip(range(pool_size[1]), range(-pool_size[1], 0)):
                for (m, m1) in zip(range(pool_size[2]), range(-pool_size[2], 0)):
                    y.append(x[:,
                               :,
                               k:k1:strides[0],
                               l:l1:strides[1],
                               m:m1:strides[2]])
    y = np.stack(y, axis=-1)
    if pool_mode == 'avg':
        y = np.mean(np.ma.masked_invalid(y), axis=-1).data
    elif pool_mode == 'max':
        y = np.max(y, axis=-1)

    if data_format == 'channels_last':
        if y.ndim == 3:
            y = np.transpose(y, (0, 2, 1))
        elif y.ndim == 4:
            y = np.transpose(y, (0, 2, 3, 1))
        else:
            y = np.transpose(y, (0, 2, 3, 4, 1))

    return y


pool2d = pool
pool3d = pool


def bias_add(x, y, data_format):
    if data_format == 'channels_first':
        if y.ndim > 1:
            y = np.reshape(y, y.shape[::-1])
        for _ in range(x.ndim - y.ndim - 1):
            y = np.expand_dims(y, -1)
    else:
        for _ in range(x.ndim - y.ndim - 1):
            y = np.expand_dims(y, 0)
    return x + y


def rnn(x, w, init, go_backwards=False, mask=None, unroll=False, input_length=None):
    w_i, w_h, w_o = w
    h = []
    o = []

    if go_backwards:
        t_list = range(x.shape[1] - 1, -1, -1)
    else:
        t_list = range(x.shape[1])

    if mask is not None:
        from keras import backend as K
        np_mask = K.eval(mask)
    else:
        np_mask = None

    for (i, t) in enumerate(t_list):
        h_t = np.dot(x[:, t], w_i)

        if w_h is not None:
            prev = h[i - 1] if i > 0 else init
            h_t1 = np.dot(prev, w_h)
            if np_mask is not None:
                h_t1[np_mask[:, t] == 0] = prev[np_mask[:, t] == 0]
        else:
            h_t1 = 0

        o_t = h_t + h_t1
        if w_o is not None:
            o_t = np.dot(o_t, w_o)
        o.append(o_t)

        if np_mask is not None:
            h_t = h_t * np_mask[:, t].reshape(-1, 1)
        h.append(h_t + h_t1)

    return o[-1], np.stack(o, axis=1), np.stack(h, axis=1)


_LEARNING_PHASE = True


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    _LEARNING_PHASE = value


def in_train_phase(x, alt, training=None):
    if training is None:
        training = learning_phase()

    if training is 1 or training is True:
        if callable(x):
            return x()
        else:
            return x
    else:
        if callable(alt):
            return alt()
        else:
            return alt


def in_test_phase(x, alt, training=None):
    return in_train_phase(alt, x, training=training)


def relu(x, alpha=0., max_value=None, threshold=0.):
    y = x * (x >= threshold)
    if max_value is not None:
        y = np.clip(y, 0.0, max_value)
    y += alpha * (x - threshold) * (x < threshold)
    return y


def switch(condition, then_expression, else_expression):
    cond_float = condition.astype(floatx())
    while cond_float.ndim < then_expression.ndim:
        cond_float = cond_float[..., None]
    return cond_float * then_expression + (1 - cond_float) * else_expression


def softplus(x):
    return np.log(1. + np.exp(x))


def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    y = np.minimum(y, 1.)
    y = np.maximum(y, 0.)
    return y


def tanh(x):
    return np.tanh(x)


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)


def binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        output = np.clip(output, 1e-7, 1 - 1e-7)
        output = np.log(output / (1 - output))
    return (target * -np.log(sigmoid(output)) +
            (1 - target) * -np.log(1 - sigmoid(output)))


def categorical_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = softmax(output)
    else:
        output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.sum(target * -np.log(output), axis=-1, keepdims=False)


def max(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.min(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.mean(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.var(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.std(x, axis=axis, keepdims=keepdims)


def logsumexp(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return sp.misc.logsumexp(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    return np.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    return np.cumprod(x, axis=axis)


def any(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.all(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return np.argmax(x, axis=axis)


def argmin(x, axis=-1):
    return np.argmin(x, axis=axis)


def sqrt(x):
    y = np.sqrt(x)
    y[np.isnan(y)] = 0.
    return y


def pow(x, a=1.):
    return np.power(x, a)


def clip(x, min_value, max_value):
    return np.clip(x, min_value, max_value)


def concatenate(tensors, axis=-1):
    return np.concatenate(tensors, axis)


def permute_dimensions(x, pattern):
    return np.transpose(x, pattern)


def reshape(x, shape):
    return np.reshape(x, shape)


def repeat_elements(x, rep, axis):
    return np.repeat(x, rep, axis=axis)


def repeat(x, n):
    y = np.expand_dims(x, 1)
    y = np.repeat(y, n, axis=1)
    return y


def tile(x, n):
    return np.tile(x, n)


def arange(start, stop=None, step=1, dtype='int32'):
    return np.arange(start, stop, step, dtype)


def flatten(x):
    return np.reshape(x, (-1,))


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def eval(x):
    return x


def dtype(x):
    return x.dtype.name


def constant(value, dtype=None, shape=None, name=None):
    if dtype is None:
        dtype = floatx()
    if shape is None:
        shape = ()
    np_value = value * np.ones(shape)
    np_value.astype(dtype)
    return np_value


def print_tensor(x, message=''):
    print(x, message)
    return x


def dot(x, y):
    return np.dot(x, y)


def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x.ndim - 1, y.ndim - 2]
    if any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if x.ndim > y.ndim:
        diff = x.ndim - y.ndim
        y = np.reshape(y, np.concatenate([np.shape(y), [1] * diff], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = np.sum(np.multiply(x, y), axes[0])
        else:
            out = np.sum(np.multiply(np.transpose(x, [1, 0]), y), axes[1])
    else:
        out = np.tensordot(x, y, axes=axes)
        for axis in [axes[0]]:
            axis_list = np.arange(len(out.shape) - 1).tolist()
            axis_list.insert(0, axis_list.pop(axis))
            out = np.transpose(np.diagonal(out, axis1=0, axis2=axis),
                               tuple(axis_list))
    if diff:
        if x.ndim > y.ndim:
            idx = x.ndim + y.ndim - 3
        else:
            idx = x.ndim - 1
        out = np.squeeze(out, tuple(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def transpose(x):
    return np.transpose(x)


def reverse(x, axes):
    if isinstance(axes, int):
        axes = [axes]
    for a in axes:
        x = np.flip(x, a)
    return x


def variable(value, dtype=None, name=None, constraint=None):
    if constraint is not None:
        raise TypeError("Constraint must be None when "
                        "using the NumPy backend.")
    return np.array(value, dtype)


def equal(x, y):
    return x == y


def not_equal(x, y):
    return x != y


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def less(x, y):
    return x < y


def less_equal(x, y):
    return x <= y


def maximum(x, y):
    return np.maximum(x, y)


def minimum(x, y):
    return np.minimum(x, y)


def ndim(x):
    return x.ndim


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    return (high - low) * np.random.random(shape).astype(dtype) + low


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return scale * np.random.randn(*shape).astype(dtype) + mean


def zeros(shape, dtype=floatx(), name=None):
    return np.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=floatx(), name=None):
    return np.zeros_like(x, dtype=dtype)


def ones(shape, dtype=floatx(), name=None):
    return np.ones(shape, dtype=dtype)


def ones_like(x, dtype=floatx(), name=None):
    return np.ones_like(x, dtype=dtype)


def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)


def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        x = repeat_elements(x, height_factor, axis=2)
        x = repeat_elements(x, width_factor, axis=3)
    elif data_format == 'channels_last':
        x = repeat_elements(x, height_factor, axis=1)
        x = repeat_elements(x, width_factor, axis=2)
    return x


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        x = repeat_elements(x, depth_factor, axis=2)
        x = repeat_elements(x, height_factor, axis=3)
        x = repeat_elements(x, width_factor, axis=4)
    elif data_format == 'channels_last':
        x = repeat_elements(x, depth_factor, axis=1)
        x = repeat_elements(x, height_factor, axis=2)
        x = repeat_elements(x, width_factor, axis=3)
    return x


square = np.square
abs = np.abs
exp = np.exp
log = np.log
round = np.round
sign = np.sign
expand_dims = np.expand_dims
squeeze = np.squeeze
cos = np.cos
sin = np.sin
