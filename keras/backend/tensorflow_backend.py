import tensorflow as tf
import numpy as np
import os
import copy
import warnings
from .common import _FLOATX, _EPSILON, _IMAGE_DIM_ORDERING

# INTERNAL UTILS

_SESSION = None
_LEARNING_PHASE = tf.placeholder(dtype='uint8', name='keras_learning_phase')  # 0 = test, 1 = train


def learning_phase():
    '''Returns the learning phase flag.

    The learning phase flag is an integer tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.
    '''
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    _LEARNING_PHASE = tf.constant(value, name='keras_learning_phase')


def get_session():
    '''Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.
    '''
    global _SESSION
    if tf.get_default_session() is not None:
        return tf.get_default_session()
    if _SESSION is None:
        if not os.environ.get('OMP_NUM_THREADS'):
            _SESSION = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
            _SESSION = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                                        allow_soft_placement=True))
    return _SESSION


def set_session(session):
    '''Sets the global TF session.
    '''
    global _SESSION
    _SESSION = session


# VARIABLE MANIPULATION

def variable(value, dtype=_FLOATX, name=None):
    '''Instantiates a tensor.

    # Arguments
        value: numpy array, initial value of the tensor.
        dtype: tensor type.
        name: optional name string for the tensor.

    # Returns
        Tensor variable instance.
    '''
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    if tf.get_default_graph() is get_session().graph:
        try:
            get_session().run(v.initializer)
        except tf.errors.InvalidArgumentError:
            warnings.warn('Could not automatically initialize variable, '
                          'make sure you do it manually (e.g. via '
                          '`tf.initialize_all_variables()`).')
    else:
        warnings.warn('The default TensorFlow graph is not the graph '
                      'associated with the TensorFlow session currently '
                      'registered with Keras, and as such Keras '
                      'was not able to automatically initialize a variable. '
                      'You should consider registering the proper session '
                      'with Keras via `K.set_session(sess)`.')
    return v


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    '''Instantiates a placeholder.

    # Arguments
        shape: shape of the placeholder
            (integer tuple, may include None entries).
        ndim: number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: placeholder type.
        name: optional name string for the placeholder.

    # Returns
        Placeholder tensor instance.
    '''
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    x = tf.placeholder(dtype, shape=shape, name=name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def shape(x):
    '''Returns the symbolic shape of a tensor.
    '''
    return tf.shape(x)


def int_shape(x):
    '''Returns the shape of a tensor as a tuple of
    integers or None entries.
    Note that this function only works with TensorFlow.
    '''
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):
    '''Returns the number of axes in a tensor, as an integer.
    '''
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dtype(x):
    '''Returns the dtype of a tensor, as a string.
    '''
    return x.dtype.name


def eval(x):
    '''Evaluates the value of a tensor.
    Returns a Numpy array.
    '''
    return x.eval(session=get_session())


def zeros(shape, dtype=_FLOATX, name=None):
    '''Instantiates an all-zeros tensor variable.
    '''
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    '''Instantiates an all-ones tensor variable.
    '''
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=_FLOATX, name=None):
    '''Instantiate an identity matrix.
    '''
    return variable(np.eye(size), dtype, name)


def zeros_like(x, name=None):
    '''Instantiates an all-zeros tensor
    of the same shape as another tensor.
    '''
    return tf.zeros_like(x, name=name)


def ones_like(x, name=None):
    '''Instantiates an all-ones tensor
    of the same shape as another tensor.
    '''
    return tf.ones_like(x, name=name)


def count_params(x):
    '''Returns the number of scalars in a tensor.
    '''
    shape = x.get_shape()
    return np.prod([shape[i]._value for i in range(len(shape))])


def cast(x, dtype):
    '''Casts a tensor to a different dtype.
    '''
    return tf.cast(x, dtype)


# LINEAR ALGEBRA

def dot(x, y):
    '''Multiplies 2 tensors.
    When attempting to multiply a ND tensor
    with a ND tensor, reproduces the Theano behavior
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
    '''
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = (-1,) + int_shape(x)[1:]
        y_shape = int_shape(y)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes=None):
    '''Batchwise dot product.

    batch_dot results in a tensor with less dimensions than the input.
    If the number of dimensions is reduced to 1, we use `expand_dims` to
    make sure that ndim is at least 2.

    # Example
        Assume x = [[1, 2], [3, 4]]   and y = [[5, 6], [7, 8]]
        batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
        of x.dot(y.T), although we never have to calculate the off-diagonal
        elements.


    # Arguments
        x, y: tensors with ndim >= 2
        axes: list (or single) int with target dimensions

    # Returns
        Tensor with ndim >= 2
    '''
    if type(axes) == int:
        axes = (axes, axes)
    if axes is not None:
        adj_x = None if axes[0] == ndim(x) - 1 else True
        adj_y = True if axes[1] == ndim(y) - 1 else None
    else:
        adj_x = None
        adj_y = None
    out = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def transpose(x):
    '''Transposes a matrix.
    '''
    return tf.transpose(x)


def gather(reference, indices):
    '''Retrieves the vectors of indices `indices`
    in the 2D tensor `reference`.

    # Arguments
        reference: a 2D tensor.
        indices: an int tensor of indices.

    # Returns
        A 3D tensor of same type as `reference`.
    '''
    return tf.gather(reference, indices)


# ELEMENT-WISE OPERATIONS

def _normalize_axis(axis, ndim):
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
    '''Maximum value in a tensor.
    '''
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    '''Minimum value in a tensor.
    '''
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def prod(x, axis=None, keepdims=False):
    '''Multiplies the values in a tensor, alongside the specified axis.
    '''
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    '''Variance of a tensor, alongside the specified axis.
    '''
    axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          reduction_indices=axis,
                          keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
    '''Standard deviation of a tensor, alongside the specified axis.
    '''
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def mean(x, axis=None, keepdims=False):
    '''Mean of a tensor, alongside the specified axis.
    '''
    axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).

    Returns an uint8 tensor (0s and 1s).
    '''
    axis = _normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def all(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical AND).

    Returns an uint8 tensor
    '''
    axis = _normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_all(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def argmax(x, axis=-1):
    '''Returns the index of the maximum value
    along a tensor axis.
    '''
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmax(x, axis)


def argmin(x, axis=-1):
    '''Returns the index of the minimum value
    along a tensor axis.
    '''
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmin(x, axis)


def square(x):
    '''Element-wise square.
    '''
    return tf.square(x)


def abs(x):
    '''Element-wise absolute value.
    '''
    return tf.abs(x)


def sqrt(x):
    '''Element-wise square root.
    '''
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(np.inf, dtype=_FLOATX))
    return tf.sqrt(x)


def exp(x):
    '''Element-wise exponential.
    '''
    return tf.exp(x)


def log(x):
    '''Element-wise log.
    '''
    return tf.log(x)


def round(x):
    '''Element-wise rounding to the closest integer.
    '''
    return tf.round(x)


def sign(x):
    '''Element-wise sign.
    '''
    return tf.sign(x)


def pow(x, a):
    '''Element-wise exponentiation.
    '''
    return tf.pow(x, a)


def clip(x, min_value, max_value):
    '''Element-wise value clipping.
    '''
    if max_value < min_value:
        max_value = min_value
    return tf.clip_by_value(x, tf.cast(min_value, dtype=_FLOATX),
                            tf.cast(max_value, dtype=_FLOATX))


def equal(x, y):
    '''Element-wise equality between two tensors.
    Returns a bool tensor.
    '''
    return tf.equal(x, y)


def not_equal(x, y):
    '''Element-wise inequality between two tensors.
    Returns a bool tensor.
    '''
    return tf.not_equal(x, y)


def maximum(x, y):
    '''Element-wise maximum of two tensors.
    '''
    return tf.maximum(x, y)


def minimum(x, y):
    '''Element-wise minimum of two tensors.
    '''
    return tf.minimum(x, y)


def sin(x):
    '''Computes sin of x element-wise.
    '''
    return tf.sin(x)


def cos(x):
    '''Computes cos of x element-wise.
    '''
    return tf.cos(x)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    '''Concantes a list of tensors alongside the specified axis.
    '''
    if axis < 0:
        if len(tensors[0].get_shape()):
            axis = axis % len(tensors[0].get_shape())
        else:
            axis = 0
    return tf.concat(axis, tensors)


def reshape(x, shape):
    '''Reshapes a tensor to the specified shape.
    '''
    return tf.reshape(x, shape)


def permute_dimensions(x, pattern):
    '''Permutes axes in a tensor.

    # Arguments
        pattern: should be a tuple of
            dimension indices, e.g. (0, 2, 1).
    '''
    return tf.transpose(x, perm=pattern)


def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        original_shape = int_shape(X)
        new_shape = tf.shape(X)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_nearest_neighbor(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif dim_ordering == 'tf':
        original_shape = int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_nearest_neighbor(X, new_shape)
        X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
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
    '''Repeats a 2D tensor:

    if x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim)
    '''
    assert ndim(x) == 2
    tensors = [x] * n
    stacked = tf.pack(tensors)
    return tf.transpose(stacked, (1, 0, 2))


def tile(x, n):
    if not hasattr(n, 'shape') and not hasattr(n, '__len__'):
        n = [n]
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
    '''Adds a 1-sized dimension at index "dim".
    '''
    return tf.expand_dims(x, dim)


def squeeze(x, axis):
    '''Removes a 1-dimension from the tensor at index "axis".
    '''
    return tf.squeeze(x, [axis])


def temporal_padding(x, padding=1):
    '''Pads the middle dimension of a 3D tensor
    with "padding" zeros left and right.
    '''
    pattern = [[0, 0], [padding, padding], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pads the 2nd and 3rd dimensions of a 4D tensor
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
    '''Returns the value of a tensor variable,
    as a Numpy array.
    '''
    return x.eval(session=get_session())


def batch_get_value(xs):
    '''Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    '''
    if xs:
        return get_session().run(xs)
    else:
        return []


def set_value(x, value):
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    tf.assign(x, np.asarray(value)).op.run(session=get_session())


def batch_set_value(tuples):
    '''Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    '''
    if tuples:
        ops = [tf.assign(x, np.asarray(value)) for x, value in tuples]
        get_session().run(ops)

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
    '''Instantiates a Keras function.

    # Arguments
        inputs: list of placeholder/variable tensors.
        outputs: list of output tensors.
        updates: list of update tuples (old_tensor, new_tensor).
    '''
    if len(kwargs) > 0:
        msg = [
            "Expected no kwargs, you passed %s" % len(kwargs),
            "kwargs passed to function are ignored with Tensorflow backend"
        ]
        warnings.warn('\n'.join(msg))
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    '''Returns the gradients of `variables` (list of tensor variables)
    with regard to `loss`.
    '''
    return tf.gradients(loss, variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
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
        unroll: with TensorFlow the RNN is always unrolled, but with Theano you
            can use this boolean flag to unroll the RNN.
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

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
    '''Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    '''
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.python.control_flow_ops.cond(tf.cast(condition, 'bool'),
                                        lambda: then_expression,
                                        lambda: else_expression)
    x.set_shape(x_shape)
    return x


def in_train_phase(x, alt):
    '''Selects `x` in train phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    '''
    x_shape = copy.copy(x.get_shape())
    x = tf.python.control_flow_ops.cond(tf.cast(_LEARNING_PHASE, 'bool'),
                                        lambda: x,
                                        lambda: alt)
    x._uses_learning_phase = True
    x.set_shape(x_shape)
    return x


def in_test_phase(x, alt):
    '''Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    '''
    x_shape = copy.copy(x.get_shape())
    x = tf.python.control_flow_ops.cond(tf.cast(_LEARNING_PHASE, 'bool'),
                                        lambda: alt,
                                        lambda: x)
    x._uses_learning_phase = True
    x.set_shape(x_shape)
    return x


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    '''Rectified linear unit

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
    '''Softmax of a tensor.
    '''
    return tf.nn.softmax(x)


def softplus(x):
    '''Softplus of a tensor.
    '''
    return tf.nn.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def categorical_crossentropy(output, target, from_logits=False):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
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


def sparse_categorical_crossentropy(output, target, from_logits=False):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is an integer tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        output = tf.log(output)

    output_shape = output.get_shape()
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        tf.reshape(output, [-1, int(output_shape[-1])]),
        cast(flatten(target), 'int64'))
    if len(output_shape) == 3:
        # if our output includes timesteps we need to reshape
        return tf.reshape(res, [-1, int(output_shape[-2])])
    else:
        return res


def binary_crossentropy(output, target, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def sigmoid(x):
    '''Element-wise sigmoid.
    '''
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    '''Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    '''
    x = (0.2 * x) + 0.5
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(1., dtype=_FLOATX))
    return x


def tanh(x):
    '''Element-wise tanh.
    '''
    return tf.nn.tanh(x)


def dropout(x, level, seed=None):
    '''Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0
        seed: random seed to ensure determinism.
    '''
    retain_prob = 1. - level
    if seed is None:
        seed = np.random.randint(10e6)
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incomptability)
    return tf.nn.dropout(x * 1., retain_prob, seed=seed)


def l2_normalize(x, axis):
    '''Normalizes a tensor wrt the L2 norm alongside the specified axis.
    '''
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


# CONVOLUTIONS

def _preprocess_conv2d_input(x, dim_ordering):
    if _FLOATX == 'float64':
        x = tf.cast(x, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x


def _preprocess_conv3d_input(x, dim_ordering):
    if _FLOATX == 'float64':
        x = tf.cast(x, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if _FLOATX == 'float64':
        kernel = tf.cast(kernel, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
    return kernel


def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if _FLOATX == 'float64':
        kernel = tf.cast(kernel, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        kernel = tf.transpose(kernel, (2, 3, 4, 1, 0))
    return kernel


def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    return padding


def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 3, 1, 2))

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
    return x


def _postprocess_conv3d_output(x, dim_ordering):
    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 4, 1, 2, 3))

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
    return x


def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering=_IMAGE_DIM_ORDERING,
           image_shape=None, filter_shape=None):
    '''2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv2d(x, kernel, strides, padding=padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering=_IMAGE_DIM_ORDERING,
             image_shape=None, filter_shape=None):
    '''2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    # TODO: pre-process output_shape if dim_ordering == th
    x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def atrous_conv2d(x, kernel, rate=1,
                  border_mode='valid',
                  dim_ordering=_IMAGE_DIM_ORDERING,
                  image_shape=None, filter_shape=None):
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if rate == 1:
        return conv2d(x, kernel, strides=(1, 1), border_mode=border_mode,
                      dim_ordering=dim_ordering)

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)

    x = tf.nn.atrous_conv2d(x, kernel, rate, padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     border_mode='valid', dim_ordering=_IMAGE_DIM_ORDERING):
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    depthwise_kernel = _preprocess_conv2d_kernel(depthwise_kernel,
                                                 dim_ordering)
    pointwise_kernel = _preprocess_conv2d_kernel(pointwise_kernel,
                                                 dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                           strides, padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering=_IMAGE_DIM_ORDERING,
           volume_shape=None, filter_shape=None):
    '''3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d(x, kernel, strides, padding)
    return _postprocess_conv3d_output(x, dim_ordering)


def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering=_IMAGE_DIM_ORDERING,
           pool_mode='max'):
    '''2D Pooling.

    # Arguments
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        pool_mode: one of "max", "avg".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = _preprocess_conv2d_input(x, dim_ordering)

    if pool_mode == 'max':
        x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    return _postprocess_conv2d_output(x, dim_ordering)


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering=_IMAGE_DIM_ORDERING, pool_mode='max'):
    '''3D Pooling.

    # Arguments
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        pool_mode: one of "max", "avg".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = _preprocess_conv3d_input(x, dim_ordering)

    if pool_mode == 'max':
        x = tf.nn.max_pool3d(x, pool_size, strides, padding=padding)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool3d(x, pool_size, strides, padding=padding)
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    return _postprocess_conv3d_output(x, dim_ordering)


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
