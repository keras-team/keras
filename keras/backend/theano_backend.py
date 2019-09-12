from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from contextlib import contextmanager
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.printing import Print
from theano.ifelse import ifelse
try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

import numpy as np
from .common import floatx
from .common import epsilon
from .common import normalize_data_format
from ..utils.generic_utils import transpose_shape
from ..utils.generic_utils import has_arg
# Legacy functions
from .common import set_image_dim_ordering, image_dim_ordering

py_all = all
py_any = any
py_sum = sum
py_slice = slice


# INTERNAL UTILS
theano.config.floatX = floatx()
# 0 = test, 1 = train
_LEARNING_PHASE = T.scalar(dtype='uint8', name='keras_learning_phase')
_UID_PREFIXES = defaultdict(int)


def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```python
        >>> keras.backend.get_uid('dense')
        1
        >>> keras.backend.get_uid('dense')
        2
    ```

    """
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


# VARIABLE MANIPULATION


def _assert_sparse_module():
    if not th_sparse_module:
        raise ImportError("Failed to import theano.sparse\n"
                          "You probably need to pip install nose-parameterized")


def is_sparse(tensor):
    return th_sparse_module and isinstance(tensor.type, th_sparse_module.SparseType)


def to_dense(tensor):
    if is_sparse(tensor):
        return th_sparse_module.dense_from_sparse(tensor)
    else:
        return tensor


NAME_SCOPE_STACK = []


@contextmanager
def name_scope(name):
    global NAME_SCOPE_STACK
    NAME_SCOPE_STACK.append(name)
    yield
    NAME_SCOPE_STACK.pop()


def _prepare_name(name, default):
    prefix = '/'.join(NAME_SCOPE_STACK)
    if name is None:
        return prefix + '/' + default
    return prefix + '/' + name


def variable(value, dtype=None, name=None, constraint=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.

    # Returns
        A variable instance (with Keras metadata included).
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        _assert_sparse_module()
        variable = th_sparse_module.as_sparse_variable(
            value, name=_prepare_name(name, 'variable'))
    else:
        if isinstance(value, (theano.tensor.TensorVariable,
                              theano.tensor.sharedvar.TensorSharedVariable,
                              theano.tensor.TensorConstant)):
            # Support for RandomStreams().normal(), .uniform().
            value = value.eval()
        value = np.asarray(value, dtype=dtype)
        variable = theano.shared(value=value,
                                 name=_prepare_name(name, 'variable'),
                                 strict=False)
    variable._keras_shape = value.shape
    variable._uses_learning_phase = False
    variable.constraint = constraint
    return variable


def is_variable(x):
    return isinstance(x, theano.tensor.sharedvar.TensorSharedVariable)


def constant(value, dtype=None, shape=None, name=None):
    if dtype is None:
        dtype = floatx()
    if shape is None:
        shape = ()
    if not is_tensor(value):
        value = np.array(value)
        if len(value.shape) == 0:
            value = value * np.ones(shape)
        if shape and value.shape != shape:
            value = np.reshape(value, shape)
    const = T.constant(value,
                       dtype=dtype,
                       name=_prepare_name(name, 'constant'))
    const._keras_shape = shape
    const._uses_learning_phase = False
    return const


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> # A variable indirectly created outside of keras is not a Keras tensor.
        >>> K.is_keras_tensor(k_var)
        False
        >>> keras_var = K.variable(np_var)
        >>> # A variable created with the keras backend is not a Keras tensor.
        >>> K.is_keras_tensor(keras_var)
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> # A placeholder is not a Keras tensor.
        >>> K.is_keras_tensor(keras_placeholder)
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> # Any Keras layer output is a Keras tensor.
        >>> K.is_keras_tensor(keras_layer_output)
        True
    ```
    """
    if not is_tensor(x):
        raise ValueError('Unexpectedly found an instance of type `' +
                         str(type(x)) + '`. '
                         'Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


def is_tensor(x):
    return isinstance(x, (T.TensorVariable,
                          T.sharedvar.TensorSharedVariable,
                          T.TensorConstant))


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiate an input data placeholder variable.
    """
    if dtype is None:
        dtype = floatx()
    if shape is None and ndim is None:
        raise ValueError('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    else:
        shape = tuple([None for _ in range(ndim)])

    name = _prepare_name(name, 'placeholder')
    broadcast = (False,) * ndim
    if sparse:
        _assert_sparse_module()
        x = th_sparse_module.csr_matrix(name=name, dtype=dtype)
    else:
        x = T.TensorType(dtype, broadcast)(name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    x._theano_placeholder = True
    return x


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    """
    return hasattr(x, '_theano_placeholder') and x._theano_placeholder


def shape(x):
    """Returns the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    """
    return x.shape


def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        return None


def ndim(x):
    return x.ndim


def dtype(x):
    return x.dtype


def eval(x):
    """Returns the value of a tensor.
    """
    return to_dense(x).eval()


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable.
    """
    if dtype is None:
        dtype = floatx()
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones variable.
    """
    if dtype is None:
        dtype = floatx()
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=None, name=None):
    """Instantiates an identity matrix.
    """
    if dtype is None:
        dtype = floatx()
    if isinstance(size, (list, tuple)):
        n, m = size
    else:
        n, m = size, size
    return variable(np.eye(n, m), dtype, name)


def ones_like(x, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return T.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return T.zeros_like(x, dtype=dtype)


def identity(x, name=None):
    """Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.
        name: String, name for the variable to create.

    # Returns
        A tensor of the same shape, type and content.
    """
    return x.copy(name=name)


def random_uniform_variable(shape, low, high, dtype=None, name=None):
    return variable(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype, name=name)


def count_params(x):
    """Returns the number of scalars in a tensor.

    Return: numpy integer.
    """
    # We don't want those compilation to show up in Theano profiler.
    f = theano.function([], x.shape, profile=False)
    return np.prod(f())


def cast(x, dtype):
    return T.cast(x, dtype)


def size(x, name=None):
    """Returns the size of a tensor.
    # Arguments
        x: The input tensor.
        name: A name for the operation (optional).
    # Returns
        Size of the tensor.
    ```
    """
    return sum(ones_like(x, name=name))


# UPDATES OPS


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def moving_average_update(variable, value, momentum):
    return (variable, variable * momentum + value * (1. - momentum))


# LINEAR ALGEBRA

"""
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
"""


def dot(x, y):
    if is_sparse(x):
        out = th_sparse_module.basic.structured_dot(x, y)
    else:
        out = T.dot(x, y)
    if hasattr(x, '_keras_shape') and hasattr(y, '_keras_shape'):
        x_shape = list(x._keras_shape)
        y_shape = list(y._keras_shape)
        if len(x_shape) > 0:
            x_shape.pop()
        if len(y_shape) == 1:
            y_shape.pop()
        elif len(y_shape) > 1:
            y_shape.pop(-2)
        out._keras_shape = tuple(x_shape + y_shape)
    return out


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    batch_dot results in a tensor with less dimensions than the input.
    If the number of dimensions is reduced to 1, we use `expand_dims` to
    make sure that ndim is at least 2.

    # Arguments
        x, y: tensors with ndim >= 2
        axes: list (or single) int with target dimensions

    # Returns
        A tensor with shape equal to the concatenation of x's shape
        (less the dimension that was summed over) and y's shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to (batch_size, 1).

    # Examples
        Assume x = [[1, 2], [3, 4]]   and y = [[5, 6], [7, 8]]
        batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
        of x.dot(y.T), although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let x's shape be (100, 20) and y's shape be (100, 30, 20).
        If dot_axes is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in x's shape and y's shape:
        x.shape[0] : 100 : append to output shape
        x.shape[1] : 20 : do not append to output shape,
            dimension 1 of x has been summed over. (dot_axes[0] = 1)
        y.shape[0] : 100 : do not append to output shape,
            always ignore first dimension of y
        y.shape[1] : 30 : append to output shape
        y.shape[2] : 20 : do not append to output shape,
            dimension 2 of y has been summed over. (dot_axes[1] = 2)

        output_shape = (100, 30)
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        if y.ndim == 2:
            axes = [x.ndim - 1, y.ndim - 1]
        else:
            axes = [x.ndim - 1, y.ndim - 2]
    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if isinstance(axes, tuple):
        axes = list(axes)

    if 0 in axes:
        raise ValueError('Can not perform batch_dot over axis 0.'
                         'If your inputs are not batched,'
                         ' add a dummy batch dimension to your '
                         'inputs using K.expand_dims(x, 0)')

    out = T.batched_tensordot(x, y, axes=axes)
    if ndim(out) == 1:
        out = expand_dims(out, 1)

    if hasattr(x, '_keras_shape') and hasattr(y, '_keras_shape'):
        shape = []
        for axis in range(len(x._keras_shape)):
            if axis != axes[0]:
                shape.append(x._keras_shape[axis])
        for axis in range(1, len(y._keras_shape)):
            if axis != axes[1]:
                shape.append(y._keras_shape[axis])
        if len(shape) == 1:
            shape.append(1)     # Expand dims if ndim == 1
        out._keras_shape = tuple(shape)
    return out


def transpose(x):
    y = T.transpose(x)
    if hasattr(x, '_keras_shape'):
        y._keras_shape = tuple(reversed(x._keras_shape))
    return y


def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    y = reference[indices]
    if hasattr(reference, '_keras_shape') and hasattr(indices, '_keras_shape'):
        y._keras_shape = indices._keras_shape + reference._keras_shape[1:]
    return y


# ELEMENT-WISE OPERATIONS


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.
    """
    return T.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis.
    """
    return T.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return T.extra_ops.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return T.extra_ops.cumprod(x, axis=axis)


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.
    """
    dtype = None
    # bool is available since theano v0.9dev
    if 'int' in x.dtype or x.dtype == 'bool':
        dtype = floatx()
    return T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    return T.var(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).
    """
    y = T.any(x, axis=axis, keepdims=keepdims)
    y = _set_keras_shape_for_reduction(x, y, axis, keepdims)
    return y


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).
    """
    y = T.all(x, axis=axis, keepdims=keepdims)
    y = _set_keras_shape_for_reduction(x, y, axis, keepdims)
    return y


def _set_keras_shape_for_reduction(x, y, axis, keepdims):
    if hasattr(x, '_keras_shape'):
        if axis is None:
            y._keras_shape = (1,) * len(x._keras_shape) if keepdims else (1,)
        else:
            if isinstance(axis, int):
                axis_list = [axis]
            else:
                axis_list = list(set(int(a) for a in axis))
            keras_shape_list = list(x._keras_shape)
            if keepdims:
                for a in axis_list:
                    keras_shape_list[a] = 1
            else:
                for a in axis_list[::-1]:
                    keras_shape_list.pop(a)
                if not keras_shape_list:
                    keras_shape_list = (1,)
            y._keras_shape = tuple(keras_shape_list)
    return y


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


def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to reduce over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    """
    # Theano has a built-in optimization for logsumexp
    # (see https://github.com/Theano/Theano/pull/4736)
    # so we can just write the expression directly:
    return T.log(T.sum(T.exp(x), axis=axis, keepdims=keepdims))


def round(x):
    return T.round(x, mode='half_to_even')


def sign(x):
    return T.sgn(x)


def pow(x, a):
    return T.pow(x, a)


def clip(x, min_value, max_value):
    if (isinstance(min_value, (int, float)) and
            isinstance(max_value, (int, float))):
        if max_value < min_value:
            max_value = min_value
    if min_value is None:
        min_value = -np.inf
    if max_value is None:
        max_value = np.inf
    return T.clip(x, min_value, max_value)


def equal(x, y):
    return T.eq(x, y)


def not_equal(x, y):
    z = T.neq(x, y)
    if hasattr(x, '_keras_shape'):
        z._keras_shape = x._keras_shape
    elif hasattr(y, '_keras_shape'):
        z._keras_shape = y._keras_shape
    return z


def greater(x, y):
    return T.gt(x, y)


def greater_equal(x, y):
    return T.ge(x, y)


def less(x, y):
    return T.lt(x, y)


def less_equal(x, y):
    return T.le(x, y)


def maximum(x, y):
    return T.maximum(x, y)


def minimum(x, y):
    return T.minimum(x, y)


def sin(x):
    return T.sin(x)


def cos(x):
    return T.cos(x)


def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.
    """
    # TODO remove this if statement when Theano without
    # T.nnet.bn.batch_normalization_train is deprecated
    if not hasattr(T.nnet.bn, 'batch_normalization_train'):
        return _old_normalize_batch_in_training(
            x, gamma, beta, reduction_axes, epsilon)

    if gamma is None:
        if beta is None:
            gamma = ones_like(x)
        else:
            gamma = ones_like(beta)
    if beta is None:
        if gamma is None:
            beta = zeros_like(x)
        beta = zeros_like(gamma)

    normed, mean, stdinv = T.nnet.bn.batch_normalization_train(
        x, gamma, beta, reduction_axes, epsilon)

    return normed, mean, T.inv(stdinv ** 2)


def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Apply batch normalization on x given mean, var, beta and gamma.
    """
    # TODO remove this if statement when Theano without
    # T.nnet.bn.batch_normalization_test is deprecated
    if not hasattr(T.nnet.bn, 'batch_normalization_test'):
        return _old_batch_normalization(x, mean, var, beta, gamma, epsilon)

    if gamma is None:
        gamma = ones_like(var)
    if beta is None:
        beta = zeros_like(mean)

    if mean.ndim == 1:
        # based on TensorFlow's default: normalize along rightmost dimension
        reduction_axes = list(range(x.ndim - 1))
    else:
        reduction_axes = [i for i in range(x.ndim) if mean.broadcastable[i]]

    return T.nnet.bn.batch_normalization_test(
        x, gamma, beta, mean, var, reduction_axes, epsilon)


# TODO remove this function when Theano without
# T.nnet.bn.batch_normalization_train is deprecated
def _old_normalize_batch_in_training(x, gamma, beta, reduction_axes,
                                     epsilon=1e-3):  # pragma: no cover
    """Computes mean and std for batch then apply batch_normalization on batch.
    """
    if gamma is None:
        gamma = ones_like(x)
    if beta is None:
        beta = zeros_like(x)

    dev = theano.config.device
    use_cudnn = (ndim(x) < 5 and
                 reduction_axes == [0, 2, 3] and
                 (dev.startswith('cuda') or dev.startswith('gpu')))
    if use_cudnn:
        broadcast_beta = beta.dimshuffle('x', 0, 'x', 'x')
        broadcast_gamma = gamma.dimshuffle('x', 0, 'x', 'x')
        try:
            trained = theano.sandbox.cuda.dnn.dnn_batch_normalization_train(
                x, broadcast_gamma, broadcast_beta, 'spatial', epsilon)
            normed, mean, stdinv = trained
            normed = theano.tensor.as_tensor_variable(normed)
            mean = theano.tensor.as_tensor_variable(mean)
            stdinv = theano.tensor.as_tensor_variable(stdinv)
            var = T.inv(stdinv ** 2)
            return normed, T.flatten(mean), T.flatten(var)
        except AttributeError:
            pass

    var = x.var(reduction_axes)
    mean = x.mean(reduction_axes)

    target_shape = []
    for axis in range(ndim(x)):
        if axis in reduction_axes:
            target_shape.append(1)
        else:
            target_shape.append(x.shape[axis])
    target_shape = T.stack(*target_shape)

    broadcast_mean = T.reshape(mean, target_shape)
    broadcast_var = T.reshape(var, target_shape)
    broadcast_beta = T.reshape(beta, target_shape)
    broadcast_gamma = T.reshape(gamma, target_shape)
    normed = batch_normalization(x, broadcast_mean, broadcast_var,
                                 broadcast_beta, broadcast_gamma,
                                 epsilon)
    return normed, mean, var


# TODO remove this if statement when Theano without
# T.nnet.bn.batch_normalization_test is deprecated
def _old_batch_normalization(x, mean, var, beta, gamma,
                             epsilon=1e-3):  # pragma: no cover
    """Apply batch normalization on x given mean, var, beta and gamma.
    """
    if gamma is None:
        gamma = ones_like(var)
    if beta is None:
        beta = zeros_like(mean)

    if mean.ndim == 1 and x.ndim > 1:
        # in TensorFlow's batch_normalization, if the parameters are vectors
        # the batch normalization should be applied along the rightmost axis.
        # Theano expects the parameters to always have x.ndim dimensions.
        shuffle_pattern = ['x'] * (x.ndim - 1) + [0]
        mean = mean.dimshuffle(shuffle_pattern)
        var = var.dimshuffle(shuffle_pattern)
        beta = beta.dimshuffle(shuffle_pattern)
        gamma = gamma.dimshuffle(shuffle_pattern)

    ndim = x.ndim
    dev = theano.config.device
    use_cudnn = ndim < 5 and (dev.startswith('cuda') or dev.startswith('gpu'))
    if use_cudnn:
        try:
            axis = mean.broadcastable.index(False)
            if axis != 1:
                shuffle_pattern = list(range(ndim))
                shuffle_pattern[1] = shuffle_pattern[axis]
                shuffle_pattern[axis] = 1
                result = theano.sandbox.cuda.dnn.dnn_batch_normalization_test(
                    x.dimshuffle(shuffle_pattern),
                    gamma.dimshuffle(shuffle_pattern),
                    beta.dimshuffle(shuffle_pattern),
                    mean.dimshuffle(shuffle_pattern),
                    var.dimshuffle(shuffle_pattern),
                    'spatial', epsilon).dimshuffle(shuffle_pattern)
            else:
                result = theano.sandbox.cuda.dnn.dnn_batch_normalization_test(
                    x, gamma, beta, mean, var, 'spatial', epsilon)
            return theano.tensor.as_tensor_variable(result)
        except AttributeError:
            pass
        except ValueError:
            pass
    return T.nnet.bn.batch_normalization(x, gamma, beta, mean, sqrt(var + epsilon),
                                         mode='high_mem')


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    if py_all([is_sparse(x) for x in tensors]):
        axis = axis % ndim(tensors[0])
        if axis == 0:
            output = th_sparse_module.basic.vstack(tensors, format='csr')
        elif axis == 1:
            output = th_sparse_module.basic.hstack(tensors, format='csr')
        else:
            raise ValueError('Invalid concat axis for sparse matrix:', axis)
    else:
        output = T.concatenate([to_dense(x) for x in tensors], axis=axis)

    if py_all([hasattr(tensor, '_keras_shape') for tensor in tensors]):
        input_shapes = [tensor._keras_shape for tensor in tensors]
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[axis] is None or shape[axis] is None:
                output_shape[axis] = None
                break
            output_shape[axis] += shape[axis]
        output._keras_shape = tuple(output_shape)

    return output


def reshape(x, shape):
    y = T.reshape(x, shape)
    shape = tuple(x if isinstance(x, int) and x > 0 else None for x in shape)
    y._keras_shape = shape
    if hasattr(x, '_uses_learning_phase'):
        y._uses_learning_phase = x._uses_learning_phase
    else:
        y._uses_learning_phase = False
    return y


def permute_dimensions(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    y = x.dimshuffle(pattern)
    if hasattr(x, '_keras_shape'):
        y._keras_shape = tuple(np.asarray(x._keras_shape)[list(pattern)])
    return y


def repeat_elements(x, rep, axis):
    """Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    """
    y = T.repeat(x, rep, axis=axis)
    if hasattr(x, '_keras_shape'):
        y._keras_shape = list(x._keras_shape)
        repeat_dim = x._keras_shape[axis]
        if repeat_dim is not None:
                y._keras_shape[axis] = repeat_dim * rep
        y._keras_shape = tuple(y._keras_shape)
    return y


def resize_images(x,
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
    """Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    """
    if data_format == 'channels_first':
        axis_1 = 2
        axis_2 = 3
    elif data_format == 'channels_last':
        axis_1 = 1
        axis_2 = 2
    else:
        raise ValueError('Invalid data_format:', data_format)

    if interpolation == 'nearest':
        output = repeat_elements(x, height_factor, axis=axis_1)
        output = repeat_elements(output, width_factor, axis=axis_2)
    elif interpolation == 'bilinear':
        if not (height_factor == width_factor == 2):
            raise NotImplementedError(
                'Bilinear upscaling with factors other than (2, 2)'
                'is not available when using the Theano backend.')
        if data_format == 'channels_last':
            output = permute_dimensions(x, [0, 3, 1, 2])
        else:
            output = x
        output = T.nnet.abstract_conv.bilinear_upsampling(output,
                                                          ratio=height_factor)
        if data_format == 'channels_last':
            output = permute_dimensions(output, [0, 2, 3, 1])
        if hasattr(x, '_keras_shape'):
            output._keras_shape = list(x._keras_shape)
            output._keras_shape[axis_1] *= height_factor
            output._keras_shape[axis_2] *= width_factor
            output._keras_shape = tuple(output._keras_shape)
    else:
        raise ValueError('interpolation should be one of "nearest" or "bilinear".')

    return output


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    """Resize the volume contained in a 5D tensor of shape
    - [batch, channels, depth, height, width] (for 'channels_first' data_format)
    - [batch, depth, height, width, channels] (for 'channels_last' data_format)
    by a factor of (depth_factor, height_factor, width_factor).
    Both factors should be positive integers.
    """
    if data_format == 'channels_first':
        output = repeat_elements(x, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif data_format == 'channels_last':
        output = repeat_elements(x, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise ValueError('Invalid data_format:', data_format)


def repeat(x, n):
    """Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    """
    assert x.ndim == 2
    y = x.dimshuffle((0, 'x', 1))
    y = T.extra_ops.repeat(y, n, axis=1)
    if hasattr(x, '_keras_shape'):
        shape = list(x._keras_shape)
        shape.insert(1, n)
        y._keras_shape = tuple(shape)

    return y


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

    The default type of the returned tensor is 'int32' to
    match TensorFlow's default.
    """
    return T.arange(start, stop=stop, step=step, dtype=dtype)


def tile(x, n):
    if isinstance(n, int):
        n = (n,)
    elif isinstance(n, list):
        n = tuple(n)

    y = T.tile(x, n, ndim=x.ndim)
    shape = int_shape(x)
    if shape is None:
        return y
    elif isinstance(n, tuple) and len(n) < len(shape):  # Padding the axis
        n = tuple([1 for _ in range(len(shape) - len(n))]) + n
    elif isinstance(n, tuple) and len(n) != len(shape):
        raise NotImplementedError

    if isinstance(n, tuple):
        y._keras_shape = tuple([None if a is None else a * b
                                for (a, b) in zip(shape, n)])
    return y


def flatten(x):
    y = T.flatten(x)
    if hasattr(x, '_keras_shape'):
        if None in x._keras_shape:
            y._keras_shape = (None,)
        else:
            y._keras_shape = (np.prod(x._keras_shape), )
    return y


def batch_flatten(x):
    """Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    """
    y = T.reshape(x, (x.shape[0], T.prod(x.shape[1:])))
    if hasattr(x, '_keras_shape'):
        if None in x._keras_shape[1:]:
            y._keras_shape = (x._keras_shape[0], None)
        else:
            y._keras_shape = (x._keras_shape[0], np.prod(x._keras_shape[1:]))
    return y


def expand_dims(x, axis=-1):
    """Add a 1-sized dimension at index "dim".
    """
    pattern = [i for i in range(x.type.ndim)]
    if axis < 0:
        if x.type.ndim == 0:
            axis = 0
        else:
            axis = axis % x.type.ndim + 1
    pattern.insert(axis, 'x')
    y = x.dimshuffle(pattern)
    if hasattr(x, '_keras_shape'):
        shape = list(x._keras_shape)
        shape.insert(axis, 1)
        y._keras_shape = tuple(shape)
    return y


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    shape = list(x.shape)
    shape.pop(axis)
    y = T.reshape(x, tuple(shape))
    if hasattr(x, '_keras_shape'):
        kshape = list(x._keras_shape)
        kshape.pop(axis)
        y._keras_shape = tuple(kshape)
    return y


def temporal_padding(x, padding=(1, 1)):
    """Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    """
    assert len(padding) == 2
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + padding[0] + padding[1],
                    input_shape[2])
    output = T.zeros(output_shape)
    result = T.set_subtensor(output[:, padding[0]:x.shape[1] + padding[0], :], x)
    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0],
                               x._keras_shape[1] + py_sum(padding),
                               x._keras_shape[2])
    return result


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    data_format = normalize_data_format(data_format)

    input_shape = x.shape
    if data_format == 'channels_first':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + top_pad + bottom_pad,
                        input_shape[3] + left_pad + right_pad)
        output = T.zeros(output_shape)
        indices = (py_slice(None),
                   py_slice(None),
                   py_slice(top_pad, input_shape[2] + top_pad),
                   py_slice(left_pad, input_shape[3] + left_pad))

    else:
        output_shape = (input_shape[0],
                        input_shape[1] + top_pad + bottom_pad,
                        input_shape[2] + left_pad + right_pad,
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (py_slice(None),
                   py_slice(top_pad, input_shape[1] + top_pad),
                   py_slice(left_pad, input_shape[2] + left_pad),
                   py_slice(None))
    y = T.set_subtensor(output[indices], x)
    if hasattr(x, '_keras_shape'):
        if data_format == 'channels_first':
            if x._keras_shape[2] is not None:
                h = x._keras_shape[2] + top_pad + bottom_pad
            else:
                h = None
            if x._keras_shape[3] is not None:
                w = x._keras_shape[3] + left_pad + right_pad
            else:
                w = None
            output_keras_shape = (x._keras_shape[0],
                                  x._keras_shape[1],
                                  h,
                                  w)
        else:
            if x._keras_shape[1] is not None:
                h = x._keras_shape[1] + top_pad + bottom_pad
            else:
                h = None
            if x._keras_shape[2] is not None:
                w = x._keras_shape[2] + left_pad + right_pad
            else:
                w = None
            output_keras_shape = (x._keras_shape[0],
                                  h,
                                  w,
                                  x._keras_shape[3])
        y._keras_shape = output_keras_shape
    return y


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """Pad the 2nd, 3rd and 4th dimensions of a 5D tensor
    with "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right.
    """
    data_format = normalize_data_format(data_format)

    input_shape = x.shape
    if data_format == 'channels_first':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + padding[0][0] + padding[0][1],
                        input_shape[3] + padding[1][0] + padding[1][1],
                        input_shape[4] + padding[2][0] + padding[2][1])
        output = T.zeros(output_shape)
        indices = (py_slice(None),
                   py_slice(None),
                   py_slice(padding[0][0], input_shape[2] + padding[0][0]),
                   py_slice(padding[1][0], input_shape[3] + padding[1][0]),
                   py_slice(padding[2][0], input_shape[4] + padding[2][0]))

    else:
        output_shape = (input_shape[0],
                        input_shape[1] + padding[0][0] + padding[0][1],
                        input_shape[2] + padding[1][0] + padding[1][1],
                        input_shape[3] + padding[2][0] + padding[2][1],
                        input_shape[4])
        output = T.zeros(output_shape)
        indices = (py_slice(None),
                   py_slice(padding[0][0], input_shape[1] + padding[0][0]),
                   py_slice(padding[1][0], input_shape[2] + padding[1][0]),
                   py_slice(padding[2][0], input_shape[3] + padding[2][0]),
                   py_slice(None))
    y = T.set_subtensor(output[indices], x)
    if hasattr(x, '_keras_shape'):
        if data_format == 'channels_first':
            if x._keras_shape[2] is not None:
                h = x._keras_shape[2] + padding[0][0] + padding[0][1]
            else:
                h = None
            if x._keras_shape[3] is not None:
                w = x._keras_shape[3] + padding[1][0] + padding[1][1]
            else:
                w = None
            if x._keras_shape[4] is not None:
                d = x._keras_shape[4] + padding[2][0] + padding[2][1]
            else:
                d = None
            output_keras_shape = (x._keras_shape[0],
                                  x._keras_shape[1],
                                  h,
                                  w,
                                  d)
        else:
            if x._keras_shape[1] is not None:
                h = x._keras_shape[1] + padding[0][0] + padding[0][1]
            else:
                h = None
            if x._keras_shape[2] is not None:
                w = x._keras_shape[2] + padding[1][0] + padding[1][1]
            else:
                w = None
            if x._keras_shape[3] is not None:
                d = x._keras_shape[3] + padding[2][0] + padding[2][1]
            else:
                d = None
            output_keras_shape = (x._keras_shape[0],
                                  h,
                                  w,
                                  d,
                                  x._keras_shape[4])
        y._keras_shape = output_keras_shape
    return y


def stack(x, axis=0):
    return T.stack(x, axis=axis)


def one_hot(indices, num_classes):
    """Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
    Output: (n + 1)D one hot representation of the input
    with shape (batch_size, dim1, dim2, ... dim(n-1), num_classes)
    """
    input_shape = tuple((indices.shape[i] for i in range(indices.ndim)))
    indices = T.flatten(indices)
    oh = T.extra_ops.to_one_hot(indices, num_classes)
    oh = T.reshape(oh, input_shape + (num_classes,))
    return oh


def reverse(x, axes):
    """Reverse a tensor along the specified axes
    """
    if isinstance(axes, int):
        axes = [axes]
    elif isinstance(axes, tuple):
        axes = list(axes)
    for i in range(len(axes)):
        if axes[i] == -1:
            axes[i] = x.ndim - 1
    slices = []
    for i in range(x.ndim):
        if i in axes:
            slices.append(py_slice(None, None, -1))
        else:
            slices.append(py_slice(None, None, None))
    return x[slices]


def slice(x, start, size):
    if not (len(int_shape(x)) == len(start) == len(size)):
        raise ValueError('The dimension and the size of indices should match.')
    out = x[tuple([py_slice(i, i + j) for (i, j) in zip(start, size)])]
    out._keras_shape = tuple(size)
    return out


def pattern_broadcast(x, broadcastable):
    return T.patternbroadcast(x, broadcastable)

# VALUE MANIPULATION


def get_value(x):
    if not hasattr(x, 'get_value'):
        raise TypeError('`get_value` can only be called on a variable. '
                        'If you have an expression instead, use `eval()`.')
    return x.get_value()


def batch_get_value(xs):
    """Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    """
    return [get_value(x) for x in xs]


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def batch_set_value(tuples):
    for x, value in tuples:
        x.set_value(np.asarray(value, dtype=x.dtype))


def get_variable_shape(x):
    return x.get_value(borrow=True, return_internal_type=True).shape


def print_tensor(x, message=''):
    """Print the message & the tensor when evaluated & return the same tensor.
    """
    p_op = Print(message)
    return p_op(x)


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[], name=None, **kwargs):
        unique_variables_to_update = {}
        for v, nv in updates:
            if v not in unique_variables_to_update:
                unique_variables_to_update[v] = nv
        updates = unique_variables_to_update.items()
        self.outputs = outputs
        self.function = theano.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        name=name,
                                        **kwargs)
        self._metrics = [x for x in outputs if hasattr(x, '_is_metric')]
        self._metrics_function = theano.function(
            [], self._metrics,
            name=name + '_metrics' if name else None)
        self.name = name

    def __call__(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outputs = self.function(*inputs)
        if self._metrics:
            metrics = self._metrics_function()
        i = 0
        j = 0
        for x in self.outputs:
            if hasattr(x, '_is_metric'):
                v = metrics[j]
                outputs[i] = v
                j += 1
            i += 1
        return outputs


def _raise_invalid_arg(key):
    msg = 'Invalid argument "%s" passed to K.function with Theano backend' % key
    raise ValueError(msg)


def function(inputs, outputs, updates=[], **kwargs):
    if len(kwargs) > 0:
        for key in kwargs.keys():
            if not has_arg(theano.function, key, True):
                _raise_invalid_arg(key)
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    return T.grad(loss, variables)


def stop_gradient(variables):
    """Returns `variables` but with zero gradient w.r.t. every other variable.

    # Arguments
        variables: tensor or list of tensors to consider constant with respect
            to any other variable.

    # Returns
        A single tensor or a list of tensors (depending on the passed argument)
            that has constant gradient with respect to any other variable.
    """
    if isinstance(variables, (list, tuple)):
        return map(theano.gradient.disconnected_grad, variables)
    else:
        return theano.gradient.disconnected_grad(variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        step_function:
            Parameters:
                inputs: Tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: List of tensors.
            Returns:
                outputs: Tensor with shape (samples, ...) (no time dimension),
                new_states: List of tensors, same length and shapes
                    as 'states'.
        inputs: Tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        initial_states: Tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: Boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: Binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: A list of constant values passed at each step.
        unroll: Whether to unroll the RNN or to use a symbolic loop
            (`while_loop` or `scan` depending on backend).
        input_length: Static number of timesteps in the input.
            Must be specified if using `unroll`.

    # Returns
        A tuple (last_output, outputs, new_states).

        last_output: The latest output of the rnn, of shape `(samples, ...)`
        outputs: Tensor with shape `(samples, time, ...)` where each
            entry `outputs[s, t]` is the output of the step function
            at time `t` for sample `s`.
        new_states: List of tensors, latest states returned by
            the step function, of shape `(samples, ...)`.
    """
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise ValueError('When specifying `unroll=True`, '
                             'an `input_length` '
                             'must be provided to `rnn`.')
        if input_length == 1:
            raise ValueError('`input_length=1` is not'
                             ' supported when `unroll=True`.')

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if constants is None:
        constants = []

    global uses_learning_phase
    uses_learning_phase = False

    if mask is not None:
        if mask.ndim != 2:
            raise ValueError(
                'mask should have `shape=(samples, time)`, '
                'got {}'.format(mask.shape))
        mask = mask.dimshuffle([1, 0])

        def get_matching_mask(mask_t, ref_tensor_t):
            # tf.where needs its condition tensor
            # to be the same shape as its two
            # result tensors
            ndim = ref_tensor_t.ndim
            for _ in range(ndim - 1):
                mask_t = expand_dims(mask_t)
            add_shape = ref_tensor_t.shape[1:]
            reps = T.concatenate([[1], add_shape], 0)
            return T.tile(mask_t, reps, ndim=ndim)

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, new_states = step_function(inputs[i], states + constants)
                if getattr(output, '_uses_learning_phase', False):
                    uses_learning_phase = True

                if len(successive_outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output_mask = get_matching_mask(mask[i], output)
                output = T.switch(output_mask, output, prev_output)
                kept_states = []
                for state, new_state in zip(states, new_states):
                    state_mask = get_matching_mask(mask[i], state)
                    kept_states.append(T.switch(state_mask, new_state, state))
                states = kept_states

                successive_outputs.append(output)
                successive_states.append(states)

            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                new_states = []
                for states_at_step in successive_states:
                    new_states.append(states_at_step[i])
                states.append(T.stack(*new_states))
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = step_function(inputs[0], initial_states + constants)
            initial_output = initial_output[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 0, 1)

            def _step(inputs, mask, output_tm1, *states):
                outputs, new_states = step_function(inputs, states)
                if getattr(outputs, '_uses_learning_phase', False):
                    global uses_learning_phase
                    uses_learning_phase = True
                # output previous output if masked.
                output_mask = get_matching_mask(mask, outputs)
                outputs = T.switch(output_mask, outputs, output_tm1)
                return_states = []
                for state, new_state in zip(states, new_states):
                    state_mask = get_matching_mask(mask, state)
                    return_states.append(T.switch(state_mask, new_state, state))
                return [outputs] + return_states

            results, _ = theano.scan(
                _step,
                sequences=[inputs, mask],
                outputs_info=[initial_output] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if isinstance(results, list):
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []
    else:
        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                outputs, states = step_function(inputs[i], states + constants)
                if getattr(outputs, '_uses_learning_phase', False):
                    uses_learning_phase = True
                successive_outputs.append(outputs)
                successive_states.append(states)
            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(
                    *[states_at_step[i] for states_at_step in successive_states]))

        else:
            def _step(inputs, *states):
                outputs, new_states = step_function(inputs, states)
                if getattr(outputs, '_uses_learning_phase', False):
                    global uses_learning_phase
                    uses_learning_phase = True
                return [outputs] + new_states

            # Theano likes to make shape==1 dimensions
            # in the initial states (outputs_info) broadcastable
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 0, 1)

            results, _ = theano.scan(
                _step,
                sequences=inputs,
                outputs_info=[None] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if isinstance(results, list):
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    last_output._uses_learning_phase = uses_learning_phase
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value.

    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.
    """
    if callable(then_expression):
        then_expression = then_expression()
    if callable(else_expression):
        else_expression = else_expression()
    cond_ndim = ndim(condition)
    expr_ndim = ndim(then_expression)
    if cond_ndim < expr_ndim:
        ndim_diff = expr_ndim - cond_ndim
        for _ in range(ndim_diff):
            condition = expand_dims(condition)
    return T.switch(condition, then_expression, else_expression)


def in_train_phase(x, alt, training=None):
    """Selects `x` in train phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Returns
        Either `x` or `alt` based on the `training` flag.
        the `training` flag defaults to `K.learning_phase()`.
    """
    if training is None:
        training = learning_phase()
        uses_learning_phase = True
    else:
        uses_learning_phase = False

    if training is 1 or training is True:
        if callable(x):
            return x()
        else:
            return x

    elif training is 0 or training is False:
        if callable(alt):
            return alt()
        else:
            return alt

    if callable(x):
        x = x()
    if callable(alt):
        alt = alt()

    # else: assume learning phase is a placeholder tensor.
    x = ifelse(training, x, alt)
    if uses_learning_phase:
        x._uses_learning_phase = True
    return x


def in_test_phase(x, alt, training=None):
    """Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    """
    return in_train_phase(alt, x, training=training)


# NN OPERATIONS

def _assert_has_capability(module, func):
    if not hasattr(module, func):
        raise EnvironmentError(
            'It looks like like your version of '
            'Theano is out of date. '
            'Install the latest version with:\n'
            'pip install git+git://github.com/Theano/Theano.git '
            '--upgrade --no-deps')


def elu(x, alpha=1.0):
    """ Exponential linear unit

    # Arguments
        x: Tensor to compute the activation function for.
        alpha: scalar
    """
    _assert_has_capability(T.nnet, 'elu')
    return T.nnet.elu(x, alpha)


def relu(x, alpha=0., max_value=None, threshold=0.):
    _assert_has_capability(T.nnet, 'relu')

    if alpha != 0.:
        if threshold != 0.:
            negative_part = T.nnet.relu(-x + threshold)
        else:
            negative_part = T.nnet.relu(-x)

    if threshold != 0.:
        x = x * T.cast(T.gt(x, threshold), floatx())
    else:
        x = T.nnet.relu(x)

    if max_value is not None:
        x = T.clip(x, 0.0, max_value)

    if alpha != 0.:
        x -= alpha * negative_part

    return x


def softmax(x, axis=-1):
    if (axis == -1 or axis == x.ndim - 1) and x.ndim == 2:
        return T.nnet.softmax(x)
    xm = x.max(axis=axis, keepdims=True)
    return T.exp(x - xm) / T.exp(
        x - xm).sum(axis=axis, keepdims=True)


def softplus(x):
    return T.nnet.softplus(x)


def softsign(x):
    return T_softsign(x)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(int_shape(output))))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(int_shape(output)))))
    # If the channels are not in the last axis, move them to be there:
    if axis != -1 and axis != output_dimensions[-1]:
        permutation = output_dimensions[:axis]
        permutation += output_dimensions[axis + 1:] + [axis]
        output = permute_dimensions(output, permutation)
        target = permute_dimensions(target, permutation)
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, epsilon(), 1.0 - epsilon())
    return T.nnet.categorical_crossentropy(output, target)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(int_shape(output))))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(int_shape(output)))))
    # If the channels are not in the last axis, move them to be there:
    if axis != -1 and axis != output_dimensions[-1]:
        permutation = output_dimensions[:axis]
        permutation += output_dimensions[axis + 1:] + [axis]
        output = permute_dimensions(output, permutation)
        target = permute_dimensions(target, permutation)
    target = T.cast(T.flatten(target), 'int32')
    target = T.extra_ops.to_one_hot(target, nb_class=output.shape[-1])
    target = reshape(target, shape(output))
    return categorical_crossentropy(target, output, from_logits, axis=-1)


def binary_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, epsilon(), 1.0 - epsilon())
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    """
    if level < 0. or level >= 1:
        raise ValueError('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(1, 10e6)
    if isinstance(noise_shape, list):
        noise_shape = tuple(noise_shape)

    rng = RandomStreams(seed=seed)
    retain_prob = 1. - level

    if noise_shape is None:
        random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    else:
        random_tensor = rng.binomial(noise_shape, p=retain_prob, dtype=x.dtype)
        random_tensor = T.patternbroadcast(random_tensor,
                                           [dim == 1 for dim in noise_shape])
    x *= random_tensor
    x /= retain_prob
    return x


def l2_normalize(x, axis=None):
    square_sum = T.sum(T.square(x), axis=axis, keepdims=True)
    norm = T.sqrt(T.maximum(square_sum, epsilon()))
    return x / norm


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`.

    # Arguments
        predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
        targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A 1D tensor of length `batch_size` and type `bool`.
        `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
        values of `predictions[i]`.
    """
    # handle k < 1 and k >= predictions.shape[1] cases to match TF behavior
    if k < 1:
        # dtype='bool' is only available since Theano 0.9.0
        try:
            return T.zeros_like(targets, dtype='bool')
        except TypeError:
            return T.zeros_like(targets, dtype='int8')

    if k >= int_shape(predictions)[1]:
        try:
            return T.ones_like(targets, dtype='bool')
        except TypeError:
            return T.ones_like(targets, dtype='int8')

    predictions_k = T.sort(predictions)[:, -k]
    targets_values = predictions[T.arange(targets.shape[0]), targets]
    return T.ge(targets_values, predictions_k)


# CONVOLUTIONS

def _preprocess_conv2d_input(x, data_format):
    if data_format == 'channels_last':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = x.dimshuffle((0, 3, 1, 2))
    return x


def _preprocess_conv3d_input(x, data_format):
    if data_format == 'channels_last':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, slices)
        # TF input shape: (samples, rows, cols, slices, input_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
    return x


def _preprocess_conv2d_kernel(kernel, data_format):
    # As of Keras 2.0.0, all kernels are normalized
    # on the format `(rows, cols, input_depth, depth)`,
    # independently of `data_format`.
    # Theano expects `(depth, input_depth, rows, cols)`.
    kernel = kernel.dimshuffle((3, 2, 0, 1))
    return kernel


def _preprocess_conv2d_depthwise_kernel(kernel, kernel_shape, data_format):
    # As of Keras 2.0.0, all kernels are normalized
    # on the format `(rows, cols, input_depth, depth)`,
    # independently of `data_format`.
    # Theano expects `(input_depth * depth, 1, rows, cols)`
    # for depthwise convolution.
    kernel = kernel[::-1, ::-1, :, :]
    kernel = kernel.dimshuffle((2, 3, 0, 1))
    kernel = reshape(kernel, kernel_shape)
    return kernel


def _preprocess_conv3d_kernel(kernel, data_format):
    # As of Keras 2.0.0, all kernels are normalized
    # on the format `(space, input_depth, depth)`,
    # independently of `data_format`.
    # Theano expects `(depth, input_depth, space)`.
    kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
    return kernel


def _preprocess_padding(padding):
    if padding == 'same':
        th_padding = 'half'
    elif padding == 'valid':
        th_padding = 'valid'
    elif padding == 'full':
        th_padding = 'full'
    else:
        raise ValueError('Border mode not supported:', str(padding))
    return th_padding


def _preprocess_conv2d_image_shape(image_shape, data_format):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if data_format == 'channels_last':
        if image_shape:
            image_shape = transpose_shape(image_shape, 'channels_first',
                                          spatial_axes=(1, 2))
    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)
    return image_shape


def _preprocess_conv3d_volume_shape(volume_shape, data_format):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if data_format == 'channels_last':
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
    if volume_shape is not None:
        volume_shape = tuple(int_or_none(v) for v in volume_shape)
    return volume_shape


def _preprocess_conv2d_filter_shape(filter_shape, data_format):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if filter_shape:
        filter_shape = (filter_shape[3], filter_shape[2],
                        filter_shape[0], filter_shape[1])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _preprocess_conv2d_depthwise_filter_shape(filter_shape, data_format):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if filter_shape:
        filter_shape = (filter_shape[3] * filter_shape[2], 1,
                        filter_shape[0], filter_shape[1])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _preprocess_conv3d_filter_shape(filter_shape, data_format):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if filter_shape:
        filter_shape = (filter_shape[4], filter_shape[3],
                        filter_shape[0], filter_shape[1], filter_shape[2])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _postprocess_conv2d_output(conv_out, x,
                               padding, kernel_shape,
                               strides, data_format):
    if padding == 'same':
        if kernel_shape[2] % 2 == 0:
            i = (x.shape[2] + strides[0] - 1) // strides[0]
            conv_out = conv_out[:, :, :i, :]
        if kernel_shape[3] % 2 == 0:
            i = (x.shape[3] + strides[1] - 1) // strides[1]
            conv_out = conv_out[:, :, :, :i]
    if data_format == 'channels_last':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def _postprocess_conv3d_output(conv_out, x,
                               padding, kernel_shape,
                               strides, data_format):
    if padding == 'same':
        if kernel_shape[2] % 2 == 0:
            i = (x.shape[2] + strides[0] - 1) // strides[0]
            conv_out = conv_out[:, :, :i, :, :]
        if kernel_shape[3] % 2 == 0:
            i = (x.shape[3] + strides[1] - 1) // strides[1]
            conv_out = conv_out[:, :, :, :i, :]
        if kernel_shape[4] % 2 == 0:
            i = (x.shape[4] + strides[2] - 1) // strides[2]
            conv_out = conv_out[:, :, :, :, :i]
    if data_format == 'channels_last':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))
    return conv_out


def conv1d(x, kernel, strides=1, padding='valid',
           data_format=None, dilation_rate=1):
    """1D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, one of "channels_last", "channels_first"
        dilation_rate: integer.
    """
    data_format = normalize_data_format(data_format)

    kernel_shape = int_shape(kernel)
    if padding == 'causal':
        # causal (dilated) convolution:
        if not kernel_shape:
            raise AttributeError('Causal padding requires kernel._keras_shape set.')
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    shape = int_shape(x)
    if data_format == 'channels_last':
        # original shape: (batch, length, input_dim)
        # add dim to x to have (batch, length, 1, input_dim)
        x = expand_dims(x, 2)
        # update x._keras_shape
        if shape is not None:
            x._keras_shape = (shape[0], shape[1], 1, shape[2])
    else:
        # original shape: (batch, input_dim, length)
        # add dim to x to have (batch, input_dim, length, 1)
        x = expand_dims(x, 3)
        # update x._keras_shape
        if shape is not None:
            x._keras_shape = (shape[0], shape[1], shape[2], 1)
    # update dilation rate, strides
    dilation_rate = (dilation_rate, 1)
    strides = (strides, 1)
    # add dim to kernel (always same format independently of data_format)
    # i.e. (rows, 1, input_depth, depth)
    kernel = expand_dims(kernel, 1)
    output = conv2d(x, kernel,
                    strides=strides, padding=padding,
                    data_format=data_format, dilation_rate=dilation_rate)
    # remove added dim
    if data_format == 'channels_last':
        output = squeeze(output, 2)
    else:
        output = squeeze(output, 3)
    return output


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow data format
        in inputs/kernels/outputs.
    """
    data_format = normalize_data_format(data_format)

    image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
    kernel_shape = int_shape(kernel)
    if kernel_shape is None:
        kernel_shape = kernel.eval().shape  # in case of a shared variable
    kernel_shape = _preprocess_conv2d_filter_shape(kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    kernel = _preprocess_conv2d_kernel(kernel, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv2d(x, kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=kernel_shape,
                             filter_dilation=dilation_rate)
    conv_out = _postprocess_conv2d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
    return conv_out


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D deconvolution (transposed convolution).

    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow data format
            in inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Raises
        ValueError: if using an even kernel size with padding 'same'.
    """
    flip_filters = False
    data_format = normalize_data_format(data_format)

    if data_format == 'channels_last':
        output_shape = (output_shape[0],
                        output_shape[3],
                        output_shape[1],
                        output_shape[2])

    kernel_shape = int_shape(kernel)
    if kernel_shape is None:
        kernel_shape = kernel.eval().shape  # in case of a shared variable

    if padding == 'same' and kernel_shape[0] % 2 == 0:
        raise ValueError('In `Conv2DTranspose`, with padding mode `same`, '
                         'even kernel sizes are not supported with Theano. '
                         'You can set `kernel_size` to an odd number.')

    kernel_shape = _preprocess_conv2d_filter_shape(kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    kernel = _preprocess_conv2d_kernel(kernel, data_format)

    th_padding = _preprocess_padding(padding)
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
        imshp=None,
        kshp=kernel_shape,
        subsample=strides,
        border_mode=th_padding,
        filter_flip=not flip_filters,
        filter_dilation=dilation_rate)
    conv_out = op(kernel, x, output_shape[2:])
    conv_out = _postprocess_conv2d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
    return conv_out


def separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1,
                     padding='valid', data_format=None, dilation_rate=1):
    """1D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides integer.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilation rate.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or
        `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,)

    if data_format == 'channels_last':
        spatial_start_dim = 2
    else:
        spatial_start_dim = 3
    x = expand_dims(x, spatial_start_dim)
    depthwise_kernel = expand_dims(depthwise_kernel, 1)
    pointwise_kernel = expand_dims(pointwise_kernel, 1)
    strides = strides + (1,)
    dilation_rate = dilation_rate + (1,)

    image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
    depthwise_kernel_shape = int_shape(depthwise_kernel)
    if depthwise_kernel_shape is None:
        # in case of a shared variable
        depthwise_kernel_shape = depthwise_kernel.eval().shape
    depthwise_kernel_shape = _preprocess_conv2d_depthwise_filter_shape(
        depthwise_kernel_shape, data_format)
    pointwise_kernel_shape = int_shape(pointwise_kernel)
    if pointwise_kernel_shape is None:
        # in case of a shared variable
        pointwise_kernel_shape = pointwise_kernel.eval().shape
    pointwise_kernel_shape = _preprocess_conv2d_filter_shape(
        pointwise_kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    depthwise_kernel = _preprocess_conv2d_depthwise_kernel(
        depthwise_kernel, depthwise_kernel_shape, data_format)
    pointwise_kernel = _preprocess_conv2d_kernel(pointwise_kernel, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv2d(x, depthwise_kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=depthwise_kernel_shape,
                             filter_dilation=dilation_rate,
                             num_groups=image_shape[1])
    conv_out = T.nnet.conv2d(conv_out, pointwise_kernel,
                             border_mode=th_padding,
                             subsample=(1, 1),
                             input_shape=None,
                             filter_shape=pointwise_kernel_shape,
                             filter_dilation=dilation_rate)
    conv_out = _postprocess_conv2d_output(conv_out, x, padding,
                                          pointwise_kernel_shape,
                                          strides, data_format)
    conv_out = squeeze(conv_out, spatial_start_dim)
    return conv_out


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides tuple (length 2).
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or
        `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
    depthwise_kernel_shape = int_shape(depthwise_kernel)
    if depthwise_kernel_shape is None:
        # in case of a shared variable
        depthwise_kernel_shape = depthwise_kernel.eval().shape
    depthwise_kernel_shape = _preprocess_conv2d_depthwise_filter_shape(
        depthwise_kernel_shape, data_format)
    pointwise_kernel_shape = int_shape(pointwise_kernel)
    if pointwise_kernel_shape is None:
        # in case of a shared variable
        pointwise_kernel_shape = pointwise_kernel.eval().shape
    pointwise_kernel_shape = _preprocess_conv2d_filter_shape(
        pointwise_kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    depthwise_kernel = _preprocess_conv2d_depthwise_kernel(
        depthwise_kernel, depthwise_kernel_shape, data_format)
    pointwise_kernel = _preprocess_conv2d_kernel(pointwise_kernel, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv2d(x, depthwise_kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=depthwise_kernel_shape,
                             filter_dilation=dilation_rate,
                             num_groups=image_shape[1])
    conv_out = T.nnet.conv2d(conv_out, pointwise_kernel,
                             border_mode=th_padding,
                             subsample=(1, 1),
                             input_shape=None,
                             filter_shape=pointwise_kernel_shape,
                             filter_dilation=dilation_rate)
    conv_out = _postprocess_conv2d_output(conv_out, x, padding,
                                          pointwise_kernel_shape,
                                          strides, data_format)
    return conv_out


def depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid',
                     data_format=None, dilation_rate=(1, 1)):
    """2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        strides: strides tuple (length 2).
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or
        `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
    depthwise_kernel_shape = int_shape(depthwise_kernel)
    if depthwise_kernel_shape is None:
        # in case of a shared variable
        depthwise_kernel_shape = depthwise_kernel.eval().shape
    depthwise_kernel_shape = _preprocess_conv2d_depthwise_filter_shape(
        depthwise_kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    depthwise_kernel = _preprocess_conv2d_depthwise_kernel(
        depthwise_kernel, depthwise_kernel_shape, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv2d(x, depthwise_kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=depthwise_kernel_shape,
                             filter_dilation=dilation_rate,
                             num_groups=image_shape[1])
    conv_out = _postprocess_conv2d_output(
        conv_out, x, padding, depthwise_kernel_shape, strides, data_format)
    return conv_out


def conv3d(x, kernel, strides=(1, 1, 1),
           padding='valid', data_format=None,
           dilation_rate=(1, 1, 1)):
    """3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow data format
        in inputs/kernels/outputs.
    """
    data_format = normalize_data_format(data_format)

    volume_shape = _preprocess_conv3d_volume_shape(int_shape(x), data_format)
    kernel_shape = int_shape(kernel)
    if kernel_shape is None:
        kernel_shape = kernel.eval().shape  # in case of a shared variable
    kernel_shape = _preprocess_conv3d_filter_shape(kernel_shape, data_format)

    x = _preprocess_conv3d_input(x, data_format)
    kernel = _preprocess_conv3d_kernel(kernel, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv3d(x, kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=volume_shape,
                             filter_shape=kernel_shape,
                             filter_dilation=dilation_rate)
    conv_out = _postprocess_conv3d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
    return conv_out


def conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1),
                     padding='valid', data_format=None):
    """3D deconvolution (transposed convolution).

    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow data format
        in inputs/kernels/outputs.

    # Raises
        ValueError: if using an even kernel size with padding 'same'.
    """
    flip_filters = False
    data_format = normalize_data_format(data_format)

    if data_format == 'channels_last':
        output_shape = (output_shape[0],
                        output_shape[4],
                        output_shape[1],
                        output_shape[2],
                        output_shape[3])

    kernel_shape = int_shape(kernel)
    if kernel_shape is None:
        kernel_shape = kernel.eval().shape  # in case of a shared variable

    if padding == 'same' and kernel_shape[0] % 2 == 0:
        raise ValueError('In `Conv3DTranspose`, with padding mode `same`, '
                         'even kernel sizes are not supported with Theano. '
                         'You can set `kernel_size` to an odd number.')

    kernel_shape = _preprocess_conv3d_filter_shape(kernel_shape, data_format)

    x = _preprocess_conv3d_input(x, data_format)
    kernel = _preprocess_conv3d_kernel(kernel, data_format)

    th_padding = _preprocess_padding(padding)
    op = T.nnet.abstract_conv.AbstractConv3d_gradInputs(imshp=None,
                                                        kshp=kernel_shape,
                                                        subsample=strides,
                                                        border_mode=th_padding,
                                                        filter_flip=not flip_filters)
    conv_out = op(kernel, x, output_shape[2:])
    conv_out = _postprocess_conv3d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
    return conv_out


def pool2d(x, pool_size, strides=(1, 1), padding='valid',
           data_format=None, pool_mode='max'):
    data_format = normalize_data_format(data_format)

    assert pool_size[0] >= 1 and pool_size[1] >= 1

    if padding == 'same':
        odd_pad_w = pool_size[0] > 2 and pool_size[0] % 2 == 1
        w_pad = pool_size[0] - 2 if odd_pad_w else pool_size[0] - 1
        odd_pad_h = pool_size[1] > 2 and pool_size[1] % 2 == 1
        h_pad = pool_size[1] - 2 if odd_pad_h else pool_size[1] - 1
        pad = (w_pad, h_pad)
    elif padding == 'valid':
        pad = (0, 0)
    else:
        raise ValueError('Invalid border mode:', padding)

    if data_format == 'channels_last':
        x = x.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = pool.pool_2d(x, ws=pool_size, stride=strides,
                                ignore_border=True,
                                pad=pad,
                                mode='max')
    elif pool_mode == 'avg':
        pool_out = pool.pool_2d(x, ws=pool_size, stride=strides,
                                ignore_border=True,
                                pad=pad,
                                mode='average_exc_pad')
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)
    if padding == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
        pool_out = pool_out[:, :,
                            : expected_width,
                            : expected_height]

    if data_format == 'channels_last':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid',
           data_format=None, pool_mode='max'):
    data_format = normalize_data_format(data_format)

    if padding == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        d_pad = pool_size[2] - 2 if pool_size[2] % 2 == 1 else pool_size[2] - 1
        pad = (w_pad, h_pad, d_pad)
    elif padding == 'valid':
        pad = (0, 0, 0)
    else:
        raise ValueError('Invalid padding:', padding)

    if data_format == 'channels_last':
        x = x.dimshuffle((0, 4, 1, 2, 3))

    if pool_mode == 'max':
        pool_out = pool.pool_3d(x, ws=pool_size, stride=strides,
                                ignore_border=True,
                                pad=pad,
                                mode='max')
    elif pool_mode == 'avg':
        pool_out = pool.pool_3d(x, ws=pool_size, stride=strides,
                                ignore_border=True,
                                pad=pad,
                                mode='average_exc_pad')
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    if padding == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
        expected_depth = (x.shape[4] + strides[2] - 1) // strides[2]

        pool_out = pool_out[:, :,
                            : expected_width,
                            : expected_height,
                            : expected_depth]

    if data_format == 'channels_last':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out


def bias_add(x, bias, data_format=None):
    data_format = normalize_data_format(data_format)
    if ndim(bias) != 1 and ndim(bias) != ndim(x) - 1:
        raise ValueError('Unexpected bias dimensions %d, '
                         'expect to be 1 or %d dimensions'
                         % (ndim(bias), ndim(x) - 1))
    bias_shape = tuple(bias.shape)
    if ndim(x) == 5:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    else:
        x += bias
    return x


# RANDOMNESS


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=stddev, dtype=dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=minval, high=maxval, dtype=dtype)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.binomial(shape, p=p, dtype=dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)

    try:
        return rng.normal(size=shape, avg=mean, std=stddev, dtype=dtype,
                          truncate=True)
    except TypeError:
        normal_t = rng.normal(size=shape, avg=mean, std=stddev, dtype=dtype)
        # Poor man's truncated normal: we literally clip the tensor
        return T.clip(normal_t, mean - 2 * stddev, mean + 2 * stddev)


# Theano implementation of CTC
# Used with permission from Shawn Tan
# https://github.com/shawntan/
# Note that TensorFlow's native CTC code is significantly
# faster than this


def ctc_interleave_blanks(Y):
    Y_ = T.alloc(-1, Y.shape[0] * 2 + 1)
    Y_ = T.set_subtensor(Y_[T.arange(Y.shape[0]) * 2 + 1], Y)
    return Y_


def ctc_create_skip_idxs(Y):
    skip_idxs = T.arange((Y.shape[0] - 3) // 2) * 2 + 1
    non_repeats = T.neq(Y[skip_idxs], Y[skip_idxs + 2])
    return skip_idxs[non_repeats.nonzero()]


def ctc_update_log_p(skip_idxs, zeros, active, log_p_curr, log_p_prev):
    active_skip_idxs = skip_idxs[(skip_idxs < active).nonzero()]
    active_next = T.cast(T.minimum(
        T.maximum(
            active + 1,
            T.max(T.concatenate([active_skip_idxs, [-1]])) + 2 + 1
        ), log_p_curr.shape[0]), 'int32')

    common_factor = T.max(log_p_prev[:active])
    p_prev = T.exp(log_p_prev[:active] - common_factor)
    _p_prev = zeros[:active_next]
    # copy over
    _p_prev = T.set_subtensor(_p_prev[:active], p_prev)
    # previous transitions
    _p_prev = T.inc_subtensor(_p_prev[1:], _p_prev[:-1])
    # skip transitions
    _p_prev = T.inc_subtensor(
        _p_prev[active_skip_idxs + 2], p_prev[active_skip_idxs])
    updated_log_p_prev = T.log(_p_prev) + common_factor

    log_p_next = T.set_subtensor(
        zeros[:active_next],
        log_p_curr[:active_next] + updated_log_p_prev
    )
    return active_next, log_p_next


def ctc_path_probs(predict, Y, alpha=1e-4):
    smoothed = (1 - alpha) * predict[:, Y] + alpha * np.float32(1.) / Y.shape[0]
    L = T.log(smoothed)
    zeros = T.zeros_like(L[0])
    log_first = zeros

    f_skip_idxs = ctc_create_skip_idxs(Y)
    # there should be a shortcut to calculating this
    b_skip_idxs = ctc_create_skip_idxs(Y[::-1])

    def step(log_f_curr, log_b_curr, f_active, log_f_prev, b_active, log_b_prev):
        f_active_next, log_f_next = ctc_update_log_p(
            f_skip_idxs, zeros, f_active, log_f_curr, log_f_prev)
        b_active_next, log_b_next = ctc_update_log_p(
            b_skip_idxs, zeros, b_active, log_b_curr, log_b_prev)
        return f_active_next, log_f_next, b_active_next, log_b_next

    [f_active, log_f_probs, b_active, log_b_probs], _ = theano.scan(
        step,
        sequences=[L, L[::-1, ::-1]],
        outputs_info=[np.int32(1), log_first, np.int32(1), log_first])

    idxs = T.arange(L.shape[1]).dimshuffle('x', 0)
    mask = ((idxs < f_active.dimshuffle(0, 'x')) &
            (idxs < b_active.dimshuffle(0, 'x'))[::-1, ::-1])
    log_probs = log_f_probs + log_b_probs[::-1, ::-1] - L
    return log_probs, mask


def ctc_cost(predict, Y):
    log_probs, mask = ctc_path_probs(predict, ctc_interleave_blanks(Y))
    common_factor = T.max(log_probs)
    total_log_prob = T.log(T.sum(T.exp(log_probs - common_factor)[mask.nonzero()]))
    total_log_prob = total_log_prob + common_factor
    return -total_log_prob


# batchifies original CTC code
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor (samples, max_string_length) containing the truth labels
        y_pred: tensor (samples, time_steps, num_categories) containing the
                prediction, or output of the softmax
        input_length: tensor (samples,1) containing the sequence length for
                each batch item in y_pred
        label_length: tensor (samples,1) containing the sequence length for
                each batch item in y_true

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element
    """

    def ctc_step(y_true_step, y_pred_step, input_length_step, label_length_step):
        y_pred_step = y_pred_step[0: input_length_step[0]]
        y_true_step = y_true_step[0:label_length_step[0]]
        return ctc_cost(y_pred_step, y_true_step)

    ret, _ = theano.scan(
        fn=ctc_step,
        outputs_info=None,
        sequences=[y_true, y_pred, input_length, label_length]
    )

    ret = ret.dimshuffle('x', 0)
    return ret


# HIGH ORDER FUNCTIONS

def map_fn(fn, elems, name=None, dtype=None):
    """Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor, at least 2 dimensional
        name: A string name for the map node in the graph

    # Returns
        Tensor with first dimension equal to the elems and second depending on
        fn
    """
    return theano.map(fn, elems, name=name)[0]


def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance lambda acc, x: acc + x
        elems: tensor
        initializer: The first value used (elems[0] in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Same type and shape as initializer
    """
    if initializer is None:
        initializer = elems[0]
        elems = elems[1:]

    # We need to change the order of the arguments because theano accepts x as
    # first parameter and accumulator as second
    return theano.foldl(lambda x, acc: fn(acc, x),
                        elems, initializer, name=name)[0]


def foldr(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance lambda acc, x: acc + x
        elems: tensor
        initializer: The first value used (elems[-1] in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Same type and shape as initializer
    """
    if initializer is None:
        initializer = elems[-1]
        elems = elems[:-1]

    # We need to change the order of the arguments because theano accepts x as
    # first parameter and accumulator as second
    return theano.foldr(lambda x, acc: fn(acc, x),
                        elems, initializer, name=name)[0]


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
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


def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format=None):
    data_format = normalize_data_format(data_format)

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = int_shape(kernel)
    _, feature_dim, filters = kernel_shape

    if data_format == 'channels_first':
        output = []
        for i in range(output_row):
            for j in range(output_col):
                slice_row = py_slice(i * stride_row,
                                     i * stride_row + kernel_size[0])
                slice_col = py_slice(j * stride_col,
                                     j * stride_col + kernel_size[1])
                x_flatten = reshape(inputs[:, :, slice_row, slice_col],
                                    (1, -1, feature_dim))
                output.append(dot(x_flatten,
                                  kernel[i * output_col + j, :, :]))
        output = concatenate(output, axis=0)
        output = reshape(output,
                         (output_row, output_col, -1, filters))
        output = permute_dimensions(output, (2, 3, 0, 1))
    else:
        xs = []
        for i in range(output_row):
            for j in range(output_col):
                slice_row = py_slice(i * stride_row,
                                     i * stride_row + kernel_size[0])
                slice_col = py_slice(j * stride_col,
                                     j * stride_col + kernel_size[1])
                xs.append(reshape(inputs[:, slice_row, slice_col, :],
                                  (1, -1, feature_dim)))

        x_aggregate = concatenate(xs, axis=0)
        output = batch_dot(x_aggregate, kernel)
        output = reshape(output,
                         (output_row, output_col, -1, filters))
        output = permute_dimensions(output, (2, 0, 1, 3))
    return output


def ctc_label_dense_to_sparse(labels, label_lengths):
    raise NotImplementedError


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1,
               merge_repeated=False):
    raise NotImplementedError


def control_dependencies(control_inputs):
    @contextmanager
    def nullcontextmanager():
        yield

    return nullcontextmanager()
