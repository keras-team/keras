"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal
import scipy as sp
from .common import floatx
from keras.utils.generic_utils import transpose_shape
from keras.utils import to_categorical


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


def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    if constants is None:
        constants = []

    output_sample, _ = step_function(inputs[:, 0], initial_states + constants)
    if mask is not None:
        if mask.dtype != np.bool:
            mask = mask.astype(np.bool)
        if mask.shape != inputs.shape[:2]:
            raise ValueError(
                'mask should have `shape=(samples, time)`, '
                'got {}'.format(mask.shape))

        def expand_mask(mask_, x):
            # expand mask so that `mask[:, t].ndim == x.ndim`
            while mask_.ndim < x.ndim + 1:
                mask_ = np.expand_dims(mask_, axis=-1)
            return mask_
        output_mask = expand_mask(mask, output_sample)
        states_masks = [expand_mask(mask, state) for state in initial_states]

    if input_length is None:
        input_length = inputs.shape[1]
    assert input_length == inputs.shape[1]
    time_index = range(input_length)
    if go_backwards:
        time_index = time_index[::-1]

    outputs = []
    states_tm1 = initial_states  # tm1 means "t minus one" as in "previous timestep"
    output_tm1 = np.zeros(output_sample.shape)
    for t in time_index:
        output_t, states_t = step_function(inputs[:, t], states_tm1 + constants)
        if mask is not None:
            output_t = np.where(output_mask[:, t], output_t, output_tm1)
            states_t = [np.where(state_mask[:, t], state_t, state_tm1)
                        for state_mask, state_t, state_tm1
                        in zip(states_masks, states_t, states_tm1)]
        outputs.append(output_t)
        states_tm1 = states_t
        output_tm1 = output_t

    return outputs[-1], np.stack(outputs, axis=1), states_tm1


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
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold


def switch(condition, then_expression, else_expression):
    cond_float = condition.astype(floatx())
    while cond_float.ndim < then_expression.ndim:
        cond_float = cond_float[..., np.newaxis]
    return cond_float * then_expression + (1 - cond_float) * else_expression


def softplus(x):
    return np.log(1. + np.exp(x))


def softsign(x):
    return x / (1 + np.abs(x))


def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)


def tanh(x):
    return np.tanh(x)


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)


def in_top_k(predictions, targets, k):
    top_k = np.argsort(-predictions)[:, :k]
    targets = targets.reshape(-1, 1)
    return np.any(targets == top_k, axis=-1)


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
    return sp.special.logsumexp(x, axis=axis, keepdims=keepdims)


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


def temporal_padding(x, padding=(1, 1)):
    return np.pad(x, [(0, 0), padding, (0, 0)], mode='constant')


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    all_dims_padding = ((0, 0),) + padding + ((0, 0),)
    all_dims_padding = transpose_shape(all_dims_padding, data_format,
                                       spatial_axes=(1, 2))
    return np.pad(x, all_dims_padding, mode='constant')


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    all_dims_padding = ((0, 0),) + padding + ((0, 0),)
    all_dims_padding = transpose_shape(all_dims_padding, data_format,
                                       spatial_axes=(1, 2, 3))
    return np.pad(x, all_dims_padding, mode='constant')


def tile(x, n):
    return np.tile(x, n)


def arange(start, stop=None, step=1, dtype='int32'):
    return np.arange(start, stop, step, dtype)


def flatten(x):
    return np.reshape(x, (-1,))


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def gather(reference, indices):
    return reference[indices]


def eval(x):
    return x


def get_value(x):
    return x


def count_params(x):
    return x.size


def int_shape(x):
    return x.shape


def get_variable_shape(x):
    return int_shape(x)


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
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError('Batch dot requires inputs of rank 2 or more.')

    if isinstance(axes, int):
        axes = [axes, axes]
    elif isinstance(axes, tuple):
        axes = list(axes)

    if axes is None:
        if y.ndim == 2:
            axes = [x.ndim - 1, y.ndim - 1]
        else:
            axes = [x.ndim - 1, y.ndim - 2]

    if any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))

    # Handle negative axes
    if axes[0] < 0:
        axes[0] += x.ndim
    if axes[1] < 0:
        axes[1] += y.ndim

    if 0 in axes:
        raise ValueError('Can not perform batch dot over axis 0.')

    if x.shape[0] != y.shape[0]:
        raise ValueError('Can not perform batch dot on inputs'
                         ' with different batch sizes.')

    d1 = x.shape[axes[0]]
    d2 = y.shape[axes[1]]
    if d1 != d2:
        raise ValueError('Can not do batch_dot on inputs with shapes ' +
                         str(x.shape) + ' and ' + str(y.shape) +
                         ' with axes=' + str(axes) + '. x.shape[%d] != '
                         'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))

    result = []
    axes = [axes[0] - 1, axes[1] - 1]  # ignore batch dimension
    for xi, yi in zip(x, y):
        result.append(np.tensordot(xi, yi, axes))
    result = np.array(result)

    if result.ndim == 1:
        result = np.expand_dims(result, -1)

    return result


def transpose(x):
    return np.transpose(x)


def reverse(x, axes):
    if isinstance(axes, list):
        axes = tuple(axes)
    return np.flip(x, axes)


py_slice = slice


def slice(x, start, size):
    slices = [py_slice(i, i + j) for i, j in zip(start, size)]
    return x[tuple(slices)]


def variable(value, dtype=None, name=None, constraint=None):
    if constraint is not None:
        raise TypeError("Constraint must be None when "
                        "using the NumPy backend.")
    return np.array(value, dtype)


def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is None:
        noise_shape = x.shape
    if learning_phase():
        noise = np.random.choice([0, 1],
                                 noise_shape,
                                 replace=True,
                                 p=[level, 1 - level])
        return x * noise / (1 - level)
    else:
        return x


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
    if isinstance(size, (list, tuple)):
        n, m = size
    else:
        n, m = size, size
    return np.eye(n, m, dtype=dtype)


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


def one_hot(indices, num_classes):
    return to_categorical(indices, num_classes)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1,
               merge_repeated=False):
    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[-1]
    log_prob = np.zeros((num_samples, 1))
    decoded_dense = -np.ones_like(y_pred[..., 0])
    decoded_length = np.zeros((num_samples,), dtype=np.int)
    if greedy:
        for i in range(num_samples):
            prob = y_pred[i]
            length = input_length[i]
            decoded = np.argmax(prob[:length], axis=-1)
            log_prob[i] = -np.sum(np.log(prob[np.arange(length), decoded]))
            decoded = _remove_repeats(decoded)
            decoded = _remove_blanks(decoded, num_classes)
            decoded_length[i] = len(decoded)
            decoded_dense[i, :len(decoded)] = decoded
        return decoded_dense[:, :np.max(decoded_length)], log_prob
    else:
        raise NotImplementedError


def _remove_repeats(inds):
    is_not_repeat = np.insert(np.diff(inds).astype(np.bool), 0, True)
    return inds[is_not_repeat]


def _remove_blanks(inds, num_classes):
    return inds[inds < (num_classes - 1)]


def stack(x, axis=0):
    return np.stack(x, axis=axis)


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
