import tensorflow as tf
import numpy as np
import os
import warnings
from .common import _FLOATX, _EPSILON

# INTERNAL UTILS

_SESSION = None


def get_session():
    global _SESSION
    if _SESSION is None:
        if not os.environ.get('OMP_NUM_THREADS'):
            _SESSION = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
            _SESSION = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread, allow_soft_placement=True))
    return _SESSION


def set_session(session):
    global _SESSION
    _SESSION = session


# VARIABLE MANIPULATION

def variable(value, dtype=_FLOATX, name=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    get_session().run(v.initializer)
    return v


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    if not shape:
        if ndim:
            shape = [None for _ in range(ndim)]
    return tf.placeholder(dtype, shape=shape, name=name)


def shape(x):
    # symbolic shape
    return tf.shape(x)


def int_shape(x):
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):
    return len(x.get_shape())


def eval(x):
    '''Run a graph.
    '''
    return x.eval(session=get_session())


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    return variable(np.ones(shape), dtype, name)


def ones_like(x, name=None):
    return tf.ones_like(x, name=name)


def zeros_like(x, name=None):
    return tf.zeros_like(x, name=name)


def count_params(x):
    '''Return number of scalars in a tensor.
    '''
    shape = x.get_shape()
    return np.prod([shape[i]._value for i in range(len(shape))])


def cast(x, dtype):
    return tf.cast(x, dtype)


# LINEAR ALGEBRA

def dot(x, y):
    return tf.matmul(x, y)


def batch_dot(x, y, axes=None):
    if axes:
        adj_x = None if axes[0][0] == ndim(x)-1 else True
        adj_y = True if axes[1][0] == ndim(y)-1 else None
    else:
        adj_x = None
        adj_y = None
    return tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)


def transpose(x):
    return tf.transpose(x)


def gather(reference, indices):
    '''
    # Arguments
        reference: a tensor.
        indices: an int tensor of indices.

    # Returns
        a tensor of same type as `reference`.
    '''
    return tf.gather(reference, indices)


# ELEMENT-WISE OPERATIONS

def normalize_axis(axis, ndim):
    if type(axis) is tuple:
        axis = list(axis)
    if type(axis) is list:
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis


def max(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.sqrt(tf.reduce_mean(devs_squared,
                                  reduction_indices=axis,
                                  keep_dims=keepdims))


def mean(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).

    Return array of uint8 (0s and 1s).
    '''
    axis = normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def argmax(x, axis=-1):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmax(x, axis)


def argmin(x, axis=-1):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmin(x, axis)


def square(x):
    return tf.square(x)


def abs(x):
    return tf.abs(x)


def sqrt(x):
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(np.inf, dtype=_FLOATX))
    return tf.sqrt(x)


def exp(x):
    return tf.exp(x)


def log(x):
    return tf.log(x)


def round(x):
    return tf.round(x)


def sign(x):
    return tf.sign(x)


def pow(x, a):
    return tf.pow(x, a)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return tf.clip_by_value(x, tf.cast(min_value, dtype=_FLOATX),
                            tf.cast(max_value, dtype=_FLOATX))


def equal(x, y):
    return tf.equal(x, y)


def not_equal(x, y):
    return tf.not_equal(x, y)


def maximum(x, y):
    return tf.maximum(x, y)


def minimum(x, y):
    return tf.minimum(x, y)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    if axis < 0:
        if len(tensors[0].get_shape()):
            axis = axis % len(tensors[0].get_shape())
        else:
            axis = 0
    return tf.concat(axis, tensors)


def reshape(x, shape):
    return tf.reshape(x, shape)


def permute_dimensions(x, pattern):
    '''Transpose dimensions.

    # Arguments
        pattern: should be a tuple or list of
            dimension indices, e.g. [0, 2, 1].
    '''
    return tf.transpose(x, perm=pattern)


def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        new_shape = tf.shape(X)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_nearest_neighbor(X, new_shape)
        return permute_dimensions(X, [0, 3, 1, 2])
    elif dim_ordering == 'tf':
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        return tf.image.resize_nearest_neighbor(X, new_shape)
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    '''
    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    splits = tf.split(axis, x_shape[axis], x)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)


def repeat(x, n):
    '''Repeat a 2D tensor:

    if x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim)
    '''
    assert ndim(x) == 2
    tensors = [x] * n
    stacked = tf.pack(tensors)
    return tf.transpose(stacked, (1, 0, 2))


def tile(x, n):
    return tf.tile(x, n)


def flatten(x):
    return tf.reshape(x, [-1])


def batch_flatten(x):
    '''Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    '''
    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])
    return x


def expand_dims(x, dim=-1):
    '''Add a 1-sized dimension at index "dim".
    '''
    return tf.expand_dims(x, dim)


def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    return tf.squeeze(x, [axis])


def temporal_padding(x, padding=1):
    '''Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.
    '''
    pattern = [[0, 0], [padding, padding], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    if dim_ordering == 'th':
        pattern = [[0, 0], [0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]]]
    else:
        pattern = [[0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]],
                   [0, 0]]
    return tf.pad(x, pattern)


def pack(x):
    return tf.pack(x)


# VALUE MANIPULATION


def get_value(x):
    '''Technically the same as eval() for TF.
    '''
    return x.eval(session=get_session())


def set_value(x, value):
    tf.assign(x, np.asarray(value)).op.run(session=get_session())


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[]):
        assert type(inputs) in {list, tuple}, 'Input to a TensorFlow backend function should be a list or tuple.'
        assert type(outputs) in {list, tuple}, 'Output to a TensorFlow backend function should be a list or tuple.'
        assert type(updates) in {list, tuple}, 'Updates in a TensorFlow backend function should be a list or tuple.'
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            self.updates = [tf.assign(p, new_p) for (p, new_p) in updates]

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        names = [v.name for v in self.inputs]
        feed_dict = dict(zip(names, inputs))
        session = get_session()
        updated = session.run(self.outputs + self.updates, feed_dict=feed_dict)
        return updated[:len(self.outputs)]


def function(inputs, outputs, updates=[], **kwargs):
    if len(kwargs) > 0:
        msg = [
            "Expected no kwargs, you passed %s" % len(kwargs),
            "kwargs passed to function are ignored with Tensorflow backend"
        ]
        warnings.warn('\n'.join(msg))
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    return tf.gradients(loss, variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None):
    '''Iterates over the time dimension of a tensor.

    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, ...) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'.
        initial_states: tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time, 1),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.

    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    ndim = len(inputs.get_shape())
    assert ndim >= 3, "Input should be at least 3D."
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))
    input_list = tf.unpack(inputs)
    if constants is None:
        constants = []

    states = initial_states
    successive_states = []
    successive_outputs = []
    if go_backwards:
        input_list.reverse()

    if mask is not None:
        # Transpose not supported by bool tensor types, hence round-trip to uint8.
        mask = tf.cast(mask, tf.uint8)
        if len(mask.get_shape()) == ndim-1:
            mask = expand_dims(mask)
        mask = tf.cast(tf.transpose(mask, axes), tf.bool)
        mask_list = tf.unpack(mask)

        if go_backwards:
            mask_list.reverse()

        for input, mask_t in zip(input_list, mask_list):
            output, new_states = step_function(input, states + constants)

            # tf.select needs its condition tensor to be the same shape as its two
            # result tensors, but in our case the condition (mask) tensor is
            # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
            # broadcast the mask to match the shape of A and B. That's what the
            # tile call does, is just repeat the mask along its second dimension
            # ndimensions times.
            tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(output)[1]]))

            if len(successive_outputs) == 0:
                prev_output = zeros_like(output)
            else:
                prev_output = successive_outputs[-1]

            output = tf.select(tiled_mask_t, output, prev_output)

            return_states = []
            for state, new_state in zip(states, new_states):
                # (see earlier comment for tile explanation)
                tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(new_state)[1]]))
                return_states.append(tf.select(tiled_mask_t, new_state, state))

            states = return_states
            successive_outputs.append(output)
            successive_states.append(states)
    else:
        for input in input_list:
            output, states = step_function(input, states + constants)
            successive_outputs.append(output)
            successive_states.append(states)

    last_output = successive_outputs[-1]
    outputs = tf.pack(successive_outputs)
    new_states = successive_states[-1]

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def switch(condition, then_expression, else_expression):
    '''Switch between two operations depending on a scalar value.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    '''
    return tf.python.control_flow_ops.cond(condition,
                                           lambda: then_expression,
                                           lambda: else_expression)


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    '''ReLU.

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    if isinstance(alpha, (tuple, list, np.ndarray)) or np.isscalar(alpha):
        alpha = tf.constant(alpha, dtype=_FLOATX)
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
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1. - _EPSILON, dtype=_FLOATX))
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)


def binary_crossentropy(output, target, from_logits=False):
    '''Note: tf.nn.sigmoid_cross_entropy_with_logits
    expects logits, Keras expects probabilities.
    '''
    if not from_logits:
        # transform back to logits
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(1., dtype=_FLOATX))
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


def l2_normalize(x, axis):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


# CONVOLUTIONS


def conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''Runs on cuDNN if available.

    # Arguments
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

    if _FLOATX == 'float64':
        # tf conv2d only supports float32
        x = tf.cast(x, 'float32')
        kernel = tf.cast(kernel, 'float32')

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

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
    return x


def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='th', pool_mode='max'):
    '''
    # Arguments
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

    if _FLOATX == 'float64':
        # tf max_pool only supports float32
        x = tf.cast(x, 'float32')

    if dim_ordering in {'tf', 'th'}:
        if dim_ordering == 'th':
            # TF uses the last dimension as channel dimension,
            # instead of the 2nd one.
            # TH input shape: (samples, input_depth, rows, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            # TH kernel shape: (depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, depth)
            x = tf.transpose(x, (0, 2, 3, 1))
        if pool_mode == 'max':
            x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        elif pool_mode == 'avg':
            x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
        else:
            raise Exception('Invalid pooling mode: ' + str(pool_mode))
        if dim_ordering == 'th':
            x = tf.transpose(x, (0, 3, 1, 2))
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
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


def random_binomial(shape, p=0.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.select(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                     tf.ones(shape), tf.zeros(shape))
