from __future__ import absolute_import
from __future__ import division

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
