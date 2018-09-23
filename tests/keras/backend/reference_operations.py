"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal
import scipy as sp

from keras.backend.common import normalize_data_format
from keras.backend.common import floatx


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


def relu(x, alpha=0., max_value=None):
    y = x * (x > 0) + alpha * x * (x < 0)
    if max_value is not None:
        y = np.minimum(y, max_value)
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


def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)


def dot(x, y):
    return np.dot(x, y)


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


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    return (high - low) * np.random.random(shape).astype(dtype) + low


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return scale * np.random.randn(*shape).astype(dtype) + mean


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


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.shape
    if dims is not None:
        return len(dims)
    return None


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17], [53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = np.reshape(y, np.concat([np.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = np.reshape(x, np.concat([np.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = np.reduce_sum(np.multiply(x, y), axes[0])
        else:
            out = np.reduce_sum(np.multiply(np.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = np.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = np.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    """Apply 1D conv with un-shared weights.

    # Arguments
        inputs: 3D tensor with shape: (batch_size, steps, input_dim)
        kernel: the unshared weight for convolution,
                with shape (output_length, feature_dim, filters)
        kernel_size: a tuple of a single integer,
                     specifying the length of the 1D convolution window
        strides: a tuple of a single integer,
                 specifying the stride length of the convolution
        data_format: the data format, channels_first or channels_last

    # Returns
        the tensor after 1d conv with un-shared weights, with shape (batch_size, output_length, filters)

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    stride = strides[0]
    kernel_shape = int_shape(kernel)
    output_length, feature_dim, filters = kernel_shape

    xs = []
    for i in range(output_length):
        slice_length = py_slice(i * stride,
                                i * stride + kernel_size[0])
        xs.append(reshape(inputs[:, slice_length, :],
                          (1, -1, feature_dim)))
    x_aggregate = concatenate(xs, axis=0)
    # Shape: `(output_length, batch_size, filters)`.
    output = batch_dot(x_aggregate, kernel)
    return permute_dimensions(output, (1, 0, 2))


def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    dims = x.shape
    if dims is not None:
        return dims
    return None


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    """Apply 2D conv with un-shared weights.

    # Arguments
        inputs: 4D tensor with shape:
                (batch_size, filters, new_rows, new_cols)
                if data_format='channels_first'
                or 4D tensor with shape:
                (batch_size, new_rows, new_cols, filters)
                if data_format='channels_last'.
        kernel: the unshared weight for convolution,
                with shape (output_items, feature_dim, filters)
        kernel_size: a tuple of 2 integers, specifying the
                     width and height of the 2D convolution window.
        strides: a tuple of 2 integers, specifying the strides
                 of the convolution along the width and height.
        output_shape: a tuple with (output_row, output_col)
        data_format: the data format, channels_first or channels_last

    # Returns
        A 4d tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
        or 4D tensor with shape:
        (batch_size, new_rows, new_cols, filters)
        if data_format='channels_last'.

    # Raises
        ValueError: if `data_format` is neither
                    `channels_last` or `channels_first`.
    """
    data_format = normalize_data_format(data_format)

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = int_shape(kernel)
    _, feature_dim, filters = kernel_shape

    xs = []
    for i in range(output_row):
        for j in range(output_col):
            slice_row = py_slice(i * stride_row,
                                 i * stride_row + kernel_size[0])
            slice_col = py_slice(j * stride_col,
                                 j * stride_col + kernel_size[1])
            if data_format == 'channels_first':
                xs.append(reshape(inputs[:, :, slice_row, slice_col],
                                  (1, -1, feature_dim)))
            else:
                xs.append(reshape(inputs[:, slice_row, slice_col, :],
                                  (1, -1, feature_dim)))

    x_aggregate = concatenate(xs, axis=0)
    output = batch_dot(x_aggregate, kernel)
    output = reshape(output,
                     (output_row, output_col, -1, filters))

    if data_format == 'channels_first':
        output = permute_dimensions(output, (2, 3, 0, 1))
    else:
        output = permute_dimensions(output, (2, 0, 1, 3))
    return output


def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    if isinstance(n, int):
        n = [n]
    return np.tile(x, n)


def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
    ```
    """
    return x.astype(dtype)

def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    v = np.zeros(shape=shape, dtype=dtype)
    if py_all(list(v.shape)):
        return variable(v, dtype=dtype, name=name)
    return v

def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, filled with `1.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.ones((3,4))
        >>> K.eval(kvar)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    v = np.ones(shape=shape, dtype=dtype)
    if py_all(list(v.shape)):
        return variable(v, dtype=dtype, name=name)
    return v

def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or Keras tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with zeros.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_zeros = K.zeros_like(kvar)
        >>> K.eval(kvar_zeros)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    return np.zeros_like(x, dtype=dtype)


def ones_like(x, dtype=None, name=None):
    """Instantiates an all-ones variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with ones.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_ones = K.ones_like(kvar)
        >>> K.eval(kvar_ones)
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
    ```
    """
    return np.ones_like(x, dtype=dtype)

def identity(x, name=None):
    """Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.
        name: String, name for the variable to create.

    # Returns
        A tensor of the same shape, type and content.
    """
    return np.identity(x)

def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```python
        # TensorFlow example
        >>> from keras import backend as K
        >>> tf_session = K.get_session()
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
        >>> K.shape(inputs)
        <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
        # To get integer shape (Instead, you can use K.int_shape(x))
        >>> K.shape(kvar).eval(session=tf_session)
        array([2, 2], dtype=int32)
        >>> K.shape(inputs).eval(session=tf_session)
        array([2, 4, 5], dtype=int32)
    ```
    """
    return np.array(np.shape(x), dtype='int32')

py_all = all
py_any = any
py_sum = sum
py_slice = slice

square = np.square
abs = np.abs
exp = np.exp
log = np.log
round = np.round
sign = np.sign
expand_dims = np.expand_dims
squeeze = np.squeeze
sin = np.sin
cos = np.cos
