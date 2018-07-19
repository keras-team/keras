"""Utilities used in convolutional layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import numpy as np
from .. import backend as K


def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.

    # Arguments
        value: The value to validate and convert. Could an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
          "kernel_size". This is only used to format error messages.

    # Returns
        A tuple of n integers.

    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
    return value_tuple


def normalize_padding(value):
    padding = value.lower()
    allowed = {'valid', 'same', 'causal'}
    if K.backend() == 'theano':
        allowed.add('full')
    if padding not in allowed:
        raise ValueError('The `padding` argument must be one of "valid", "same" (or "causal" for Conv1D). '
                         'Received: ' + str(padding))
    return padding


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (3D, 4D or 5D).

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[slices])


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
    """Determines input length of a convolution given output length.

    # Arguments
        output_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.

    # Returns
        The input length (integer).
    """
    if output_length is None:
        return None
    assert padding in {'same', 'valid', 'full'}
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    elif padding == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size


def deconv_length(dim_size, stride_size, kernel_size, padding, output_padding):
    """Determines output length of a transposed convolution given input length.

    # Arguments
        dim_size: Integer, the input length.
        stride_size: Integer, the stride along the dimension of `dim_size`.
        kernel_size: Integer, the kernel size along the dimension of
            `dim_size`.
        padding: One of `"same"`, `"valid"`, `"full"`.
        output_padding: Integer, amount of padding along the output dimension,
            Can be set to `None` in which case the output length is inferred.

    # Returns
        The output length (integer).
    """
    assert padding in {'same', 'valid', 'full'}
    if dim_size is None:
        return None

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'valid':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'full':
            dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
        elif padding == 'same':
            dim_size = dim_size * stride_size
    else:
        if padding == 'same':
            pad = kernel_size // 2
        elif padding == 'valid':
            pad = 0
        elif padding == 'full':
            pad = kernel_size - 1

        dim_size = ((dim_size - 1) * stride_size + kernel_size - 2 * pad +
                    output_padding)

    return dim_size
