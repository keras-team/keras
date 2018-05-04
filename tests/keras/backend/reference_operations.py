"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal


def normalize_conv(func):
    def wrapper(*args):
        x = args[0]
        w = args[1]
        if x.ndim == 3:
            w = np.flipud(w)
            w = np.transpose(w, (1, 2, 0))
            if args[3] == 'channels_last':
                x = np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            w = np.fliplr(np.flipud(w))
            w = np.transpose(w, (2, 3, 0, 1))
            if args[3] == 'channels_last':
                x = np.transpose(x, (0, 3, 1, 2))
        else:
            w = np.flip(np.fliplr(np.flipud(w)), axis=2)
            w = np.transpose(w, (3, 4, 0, 1, 2))
            if args[3] == 'channels_last':
                x = np.transpose(x, (0, 4, 1, 2, 3))

        y = func(x, w, args[2], args[3])

        if args[3] == 'channels_last':
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
    x2 = depthwise_conv(x, w1, padding, data_format)
    return conv(x2, w2, padding, data_format)


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
                    y.append(x[:, :, k:k1:strides[0], l:l1:strides[1], m:m1:strides[2]])
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
