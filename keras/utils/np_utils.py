"""Numpy-related utilities."""
from __future__ import absolute_import

import numpy as np
from six.moves import range
from six.moves import zip
from .. import backend as K


def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def binary_logloss(p, y):
    epsilon = 1e-15
    p = np.maximum(epsilon, p)
    p = np.minimum(1 - epsilon, p)
    res = sum(y * np.log(p) + np.subtract(1, y) * np.log(np.subtract(1, p)))
    res *= -1.0 / len(y)
    return res


def multiclass_logloss(p, y):
    npreds = [p[i][y[i] - 1] for i in range(len(y))]
    score = -(1. / len(y)) * np.sum(np.log(npreds))
    return score


def accuracy(p, y):
    return np.mean([a == b for a, b in zip(p, y)])


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def convert_kernel(kernel, dim_ordering=None):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (4D or 5D).
        dim_ordering: the data format.

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid dim_ordering.
    """
    if dim_ordering is None:
        dim_ordering = K.image_dim_ordering()
    if not 4 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)

    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    if dim_ordering == 'th':  # (out_depth, input_depth, ...)
        slices[:2] = no_flip
    elif dim_ordering == 'tf':  # (..., input_depth, out_depth)
        slices[-2:] = no_flip
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    return np.copy(kernel[slices])


def conv_output_length(input_length, filter_size,
                       border_mode, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        border_mode: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, border_mode, stride):
    """Determines input length of a convolution given output length.

    # Arguments
        output_length: integer.
        filter_size: integer.
        border_mode: one of "same", "valid", "full".
        stride: integer.

    # Returns
        The input length (integer).
    """
    if output_length is None:
        return None
    assert border_mode in {'same', 'valid', 'full'}
    if border_mode == 'same':
        pad = filter_size // 2
    elif border_mode == 'valid':
        pad = 0
    elif border_mode == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size
