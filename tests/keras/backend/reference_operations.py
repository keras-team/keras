"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal


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


conv1d = conv
conv2d = conv
conv3d = conv
depthwise_conv2d = depthwise_conv
separable_conv1d = separable_conv
separable_conv2d = separable_conv


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


def relu(x, alpha=0., max_value=None):
    y = x * (x > 0) + alpha * x * (x < 0)
    if max_value is not None:
        y = np.minimum(y, max_value)
    return y


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
        for a in axis:
            x = np.max(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        for a in axis:
            x = np.min(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.min(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        for a in axis:
            x = np.mean(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        for a in axis:
            x = np.std(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.std(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        for a in axis:
            x = np.sum(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        for a in axis:
            x = np.prod(x, axis=a, keepdims=keepdims)
        return x
    else:
        return np.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    return np.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    return np.cumprod(x, axis=axis)


def any(x, axis=None, keepdims=False):
    return np.any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
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


def reshape(x, shape):
    return np.reshape(x, shape)


def repeat_elements(x, rep, axis):
    return np.repeat(x, rep, axis=axis)


def eval(x):
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


square = np.square
abs = np.abs
exp = np.exp
log = np.log
round = np.round
sign = np.sign
