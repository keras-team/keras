import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import downsample
import numpy as np
from .common import _FLOATX, _EPSILON


# INTERNAL UTILS
theano.config.floatX = _FLOATX


def _on_gpu():
    '''Return whether the session is set to
    run on GPU or not (i.e. on CPU).
    '''
    return theano.config.device[:3] == 'gpu' or theano.sandbox.cuda.cuda_enabled


if _on_gpu():
    '''Import cuDNN only if running on GPU:
    not having Cuda installed should not
    prevent from running the present code.
    '''
    from theano.sandbox.cuda import dnn


# VARIABLE MANIPULATION

def variable(value, dtype=_FLOATX, name=None):
    '''Instantiate a tensor variable.
    '''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    '''Instantiate an input data placeholder variable.
    '''
    if shape is None and ndim is None:
        raise Exception('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    broadcast = (False,) * ndim
    return T.TensorType(dtype, broadcast)(name)


def shape(x):
    '''Return the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    '''
    return x.shape


def ndim(x):
    return x.ndim


def eval(x):
    '''Run a graph.
    '''
    return x.eval()


def zeros(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-zeros variable.
    '''
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-ones variable.
    '''
    return variable(np.ones(shape), dtype, name)


def ones_like(x):
    return T.ones_like(x)


def zeros_like(x):
    return T.zeros_like(x)


def count_params(x):
    '''Return number of scalars in a tensor.

    Return: numpy integer.
    '''
    return np.prod(x.shape.eval())


def cast(x, dtype):
    return T.cast(x, dtype)


# LINEAR ALGEBRA

'''
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
'''


def dot(x, y):
    return T.dot(x, y)


def transpose(x):
    return T.transpose(x)


def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]


# ELEMENT-WISE OPERATIONS


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    return T.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    return T.prod(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return T.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).
    '''
    return T.any(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return T.argmax(x, axis=axis, keepdims=False)


def argmin(x, axis=-1):
    return T.argmin(x, axis=axis, keepdims=False)


def square(x):
    return T.sqr(x)


def abs(x):
    return T.abs_(x)


def sqrt(x):
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)


def exp(x):
    return T.exp(x)


def log(x):
    return T.log(x)


def round(x):
    return T.round(x)


def pow(x, a):
    return T.pow(x, a)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return T.clip(x, min_value, max_value)


def equal(x, y):
    return T.eq(x, y)


def not_equal(x, y):
    return T.neq(x, y)


def maximum(x, y):
    return T.maximum(x, y)


def minimum(x, y):
    return T.minimum(x, y)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    return T.concatenate(tensors, axis=axis)


def reshape(x, shape):
    return T.reshape(x, shape)


def permute_dimensions(x, pattern):
    '''Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    '''
    pattern = tuple(pattern)
    return x.dimshuffle(pattern)


def repeat_elements(x, rep, axis):
    '''Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    '''
    return T.repeat(x, rep, axis=axis)


def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        output = repeat_elements(X, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, height_factor, axis=1)
        output = repeat_elements(output, width_factor, axis=2)
        return output
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat(x, n):
    '''Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    '''
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return T.extra_ops.repeat(x, n, axis=1)


def tile(x, n):
    return T.tile(x, n)


def flatten(x):
    return T.flatten(x)


def batch_flatten(x):
    '''Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    '''
    x = T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))
    return x


def expand_dims(x, dim=-1):
    '''Add a 1-sized dimension at index "dim".
    '''
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return x.dimshuffle(pattern)


def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    x = T.addbroadcast(x, axis)
    return T.squeeze(x)


def temporal_padding(x, padding=1):
    '''Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Appologies for the inane API, but Theano makes this
    really hard.
    '''
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + 2 * padding,
                    input_shape[2])
    output = T.zeros(output_shape)
    return T.set_subtensor(output[:, padding:x.shape[1] + padding, :], x)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[indices], x)

# VALUE MANIPULATION


def get_value(x):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value()


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.function = theano.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True, **kwargs)

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        return self.function(*inputs)


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    return T.grad(loss, variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None):
    '''Iterates over the time dimension of a tensor.

    Parameters
    ----------
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
    mask: binary tensor with shape (samples, time),
        with a zero for every element that is masked.

    Returns
    -------
    A tuple (last_output, outputs, new_states).
        last_output: the latest output of the rnn, of shape (samples, ...)
        outputs: tensor with shape (samples, time, ...) where each
            entry outputs[s, t] is the output of the step function
            at time t for sample s.
        new_states: list of tensors, latest states returned by
            the step function, of shape (samples, ...).
    '''
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        # build an all-zero tensor of shape (samples, output_dim)
        initial_output = step_function(inputs[0 if not go_backwards else -1], initial_states)[0] * 0
        # Theano gets confused by broadcasting patterns in the scan op
        initial_output = T.unbroadcast(initial_output, 0, 1)

        def _step(input, mask, output_tm1, *states):
            output, new_states = step_function(input, states)
            # output previous output if masked.
            output = T.switch(mask, output, output_tm1)
            return_states = []
            for state, new_state in zip(states, new_states):
                return_states.append(T.switch(mask, new_state, state))
            return [output] + return_states

        results, _ = theano.scan(
            _step,
            sequences=[inputs, mask],
            outputs_info=[initial_output] + initial_states,
            go_backwards=go_backwards)
    else:
        def _step(input, *states):
            output, new_states = step_function(input, states)
            return [output] + new_states

        results, _ = theano.scan(
            _step,
            sequences=inputs,
            outputs_info=[None] + initial_states,
            go_backwards=go_backwards)

    # deal with Theano API inconsistency
    if type(results) is list:
        outputs = results[0]
        states = results[1:]
    else:
        outputs = results
        states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]
    
    if go_backwards:
        outputs = outputs[-1::-1]
    
    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    '''condition: scalar tensor.
    '''
    return T.switch(condition, then_expression, else_expression)


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


def dropout(x, level, seed=None):
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams(seed=seed)
    retain_prob = 1. - level
    x *= rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    x /= retain_prob
    return x


def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm


# CONVOLUTIONS

def conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    if _on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            np_kernel = kernel.eval()
            assert strides[0] <= np_kernel.shape[2], 'strides should be smaller than the convolution window.'
            assert strides[1] <= np_kernel.shape[3], 'strides should be smaller than the convolution window.'
            conv_out = dnn.dnn_conv(img=x,
                                    kerns=kernel,
                                    border_mode='full')
            shift_x = (np_kernel.shape[2] - strides[0]) // 2
            shift_y = (np_kernel.shape[3] - strides[1]) // 2
            expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
            expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

            conv_out = conv_out[:, :,
                                shift_x: shift_x + expected_width,
                                shift_y: shift_y + expected_height]
        else:
            conv_out = dnn.dnn_conv(img=x,
                                    kerns=kernel,
                                    border_mode=border_mode,
                                    subsample=strides)
    else:
        if border_mode == 'same':
            th_border_mode = 'full'
            np_kernel = kernel.eval()
            assert strides[0] <= np_kernel.shape[2], 'strides should be smaller than the convolution window.'
            assert strides[1] <= np_kernel.shape[3], 'strides should be smaller than the convolution window.'
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        conv_out = T.nnet.conv.conv2d(x, kernel,
                                      border_mode=th_border_mode,
                                      subsample=strides,
                                      image_shape=image_shape,
                                      filter_shape=filter_shape)
        if border_mode == 'same':
            shift_x = (np_kernel.shape[2] - strides[0]) // 2
            shift_y = (np_kernel.shape[3] - strides[1]) // 2
            expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
            expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

            conv_out = conv_out[:, :,
                                shift_x: shift_x + expected_width,
                                shift_y: shift_y + expected_height]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    if border_mode == 'same':
        # TODO: add implementation for border_mode="same"
        raise Exception('border_mode="same" not supported with Theano.')
    elif border_mode == 'valid':
        ignore_border = True
        padding = (0, 0)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = downsample.max_pool_2d(x, ds=pool_size, st=strides,
                                          ignore_border=ignore_border,
                                          padding=padding,
                                          mode='max')
    elif pool_mode == 'avg':
        pool_out = downsample.max_pool_2d(x, ds=pool_size, st=strides,
                                          ignore_border=ignore_border,
                                          padding=padding,
                                          mode='average_exc_pad')
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


# RANDOMNESS


def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)

'''
more TODO:

tensordot -> soon to be introduced in TF
batched_tensordot -> reimplement
'''
