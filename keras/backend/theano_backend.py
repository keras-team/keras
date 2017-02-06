import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
from theano.printing import Print
try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign
import inspect
import numpy as np
from .common import _FLOATX, floatx, _EPSILON, image_dim_ordering
py_all = all


# INTERNAL UTILS
theano.config.floatX = _FLOATX
_LEARNING_PHASE = T.scalar(dtype='uint8', name='keras_learning_phase')  # 0 = test, 1 = train


def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


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


def is_explicit_shape(shape):
    if hasattr(shape, '__iter__'):
        for x in shape:
            if x is not None:
                if not isinstance(x, int):
                    return False
        return True
    return False


def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance (with Keras metadata included).
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        _assert_sparse_module()
        variable = th_sparse_module.as_sparse_variable(value)
    else:
        value = np.asarray(value, dtype=dtype)
        variable = theano.shared(value=value, name=name, strict=False)
    variable._keras_shape = value.shape
    variable._uses_learning_phase = False
    return variable


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

    broadcast = (False,) * ndim
    if sparse:
        _assert_sparse_module()
        x = th_sparse_module.csr_matrix(name=name, dtype=dtype)
    else:
        x = T.TensorType(dtype, broadcast)(name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


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
        raise Exception('Not a Keras tensor:', x)


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
    return variable(np.eye(size), dtype, name)


def ones_like(x, dtype=None, name=None):
    return T.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None, name=None):
    return T.zeros_like(x, dtype=dtype)


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
    return np.prod(x.shape.eval())


def cast(x, dtype):
    return T.cast(x, dtype)


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
    # TODO: `keras_shape` inference.
    if is_sparse(x):
        return th_sparse_module.basic.structured_dot(x, y)
    else:
        return T.dot(x, y)


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
    # TODO: `keras_shape` inference.
    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x.ndim - 1, y.ndim - 2]
    out = T.batched_tensordot(x, y, axes=axes)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def transpose(x):
    # TODO: `keras_shape` inference.
    return T.transpose(x)


def gather(reference, indices):
    """reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    """
    # TODO: `keras_shape` inference.
    return reference[indices]


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
    return T.any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).
    """
    return T.all(x, axis=axis, keepdims=keepdims)


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
    return T.round(x, mode='half_to_even')


def sign(x):
    return T.sgn(x)


def pow(x, a):
    return T.pow(x, a)


def clip(x, min_value, max_value):
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return T.clip(x, min_value, max_value)


def equal(x, y):
    return T.eq(x, y)


def not_equal(x, y):
    return T.neq(x, y)


def greater(x, y):
    return T.gt(x, y)


def greater_equal(x, y):
    return T.ge(x, y)


def lesser(x, y):
    return T.lt(x, y)


def lesser_equal(x, y):
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
        return _old_normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon)

    normed, mean, stdinv = T.nnet.bn.batch_normalization_train(
        x, gamma, beta, reduction_axes, epsilon)

    return normed, mean, T.inv(stdinv ** 2)


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Apply batch normalization on x given mean, var, beta and gamma.
    """
    # TODO remove this if statement when Theano without
    # T.nnet.bn.batch_normalization_test is deprecated
    if not hasattr(T.nnet.bn, 'batch_normalization_test'):
        return _old_batch_normalization(x, mean, var, beta, gamma, epsilon)

    if mean.ndim == 1:
        # based on TensorFlow's default: normalize along rightmost dimension
        reduction_axes = range(x.ndim - 1)
    else:
        reduction_axes = [i for i in range(x.ndim) if mean.broadcastable[i]]

    return T.nnet.bn.batch_normalization_test(
        x, gamma, beta, mean, var, reduction_axes, epsilon)


# TODO remove this function when Theano without
# T.nnet.bn.batch_normalization_train is deprecated
def _old_normalize_batch_in_training(x, gamma, beta,
                                     reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.
    """
    dev = theano.config.device
    use_cudnn = ndim(x) < 5 and reduction_axes == [0, 2, 3] and (dev.startswith('cuda') or dev.startswith('gpu'))
    if use_cudnn:
        broadcast_beta = beta.dimshuffle('x', 0, 'x', 'x')
        broadcast_gamma = gamma.dimshuffle('x', 0, 'x', 'x')
        try:
            normed, mean, stdinv = theano.sandbox.cuda.dnn.dnn_batch_normalization_train(
                x, broadcast_gamma, broadcast_beta, 'spatial', epsilon)
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
def _old_batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Apply batch normalization on x given mean, var, beta and gamma.
    """
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
            return th_sparse_module.basic.vstack(tensors, format='csr')
        elif axis == 1:
            return th_sparse_module.basic.hstack(tensors, format='csr')
        else:
            raise ValueError('Invalid concat axis for sparse matrix:', axis)
    else:
        return T.concatenate([to_dense(x) for x in tensors], axis=axis)


def reshape(x, shape):
    y = T.reshape(x, shape)
    if is_explicit_shape(shape):
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
    # TODO: `keras_shape` inference.
    pattern = tuple(pattern)
    return x.dimshuffle(pattern)


def repeat_elements(x, rep, axis):
    """Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    """
    # TODO: `keras_shape` inference.
    return T.repeat(x, rep, axis=axis)


def resize_images(X, height_factor, width_factor, dim_ordering):
    """Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    """
    # TODO: `keras_shape` inference.
    if dim_ordering == 'th':
        output = repeat_elements(X, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, height_factor, axis=1)
        output = repeat_elements(output, width_factor, axis=2)
        return output
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)


def resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering):
    """Resize the volume contained in a 5D tensor of shape
    - [batch, channels, depth, height, width] (for 'th' dim_ordering)
    - [batch, depth, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (depth_factor, height_factor, width_factor).
    Both factors should be positive integers.
    """
    # TODO: `keras_shape` inference.
    if dim_ordering == 'th':
        output = repeat_elements(X, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)


def repeat(x, n):
    """Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    """
    # TODO: `keras_shape` inference.
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return T.extra_ops.repeat(x, n, axis=1)


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
    # TODO: `keras_shape` inference.
    return T.tile(x, n)


def flatten(x):
    # TODO: `keras_shape` inference.
    return T.flatten(x)


def batch_flatten(x):
    """Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    """
    # TODO: `keras_shape` inference.
    x = T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))
    return x


def expand_dims(x, dim=-1):
    """Add a 1-sized dimension at index "dim".
    """
    # TODO: `keras_shape` inference.
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return x.dimshuffle(pattern)


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    # TODO: `keras_shape` inference.
    shape = list(x.shape)
    shape.pop(axis)
    return T.reshape(x, tuple(shape))


def temporal_padding(x, padding=1):
    """Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    """
    # TODO: `keras_shape` inference.
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + 2 * padding,
                    input_shape[2])
    output = T.zeros(output_shape)
    return T.set_subtensor(output[:, padding:x.shape[1] + padding, :], x)


def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    """Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.

    Apologies for the inane API, but Theano makes this
    really hard.
    """
    # TODO: `keras_shape` inference.
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + left_pad + right_pad,
                    input_shape[2])
    output = T.zeros(output_shape)
    return T.set_subtensor(output[:, left_pad:x.shape[1] + left_pad, :], x)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='default'):
    """Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    """
    # TODO: `keras_shape` inference.
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

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
        raise ValueError('Invalid dim_ordering:', dim_ordering)
    return T.set_subtensor(output[indices], x)


def asymmetric_spatial_2d_padding(x, top_pad=1, bottom_pad=1,
                                  left_pad=1, right_pad=1,
                                  dim_ordering='default'):
    """Pad the rows and columns of a 4D tensor
    with "top_pad", "bottom_pad", "left_pad", "right_pad" (resp.) zeros
    rows on top, bottom; cols on left, right.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + top_pad + bottom_pad,
                        input_shape[3] + left_pad + right_pad)
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(top_pad, input_shape[2] + top_pad),
                   slice(left_pad, input_shape[3] + left_pad))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + top_pad + bottom_pad,
                        input_shape[2] + left_pad + right_pad,
                        input_shape[3])
        print(output_shape)
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(top_pad, input_shape[1] + top_pad),
                   slice(left_pad, input_shape[2] + left_pad),
                   slice(None))
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)
    return T.set_subtensor(output[indices], x)


def spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='default'):
    """Pad the 2nd, 3rd and 4th dimensions of a 5D tensor
    with "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1],
                        input_shape[4] + 2 * padding[2])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]),
                   slice(padding[2], input_shape[4] + padding[2]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3] + 2 * padding[2],
                        input_shape[4])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(padding[2], input_shape[3] + padding[2]),
                   slice(None))
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)
    return T.set_subtensor(output[indices], x)


def stack(x):
    return T.stack(*x)


def one_hot(indices, nb_classes):
    """Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
    Output: (n + 1)D one hot representation of the input
    with shape (batch_size, dim1, dim2, ... dim(n-1), nb_classes)
    """
    input_shape = tuple((indices.shape[i] for i in range(indices.ndim)))
    indices = T.flatten(indices)
    oh = T.extra_ops.to_one_hot(indices, nb_classes)
    oh = T.reshape(oh, input_shape + (nb_classes,))
    return oh


def reverse(x, axes):
    """Reverse a tensor along the the specified axes
    """
    if isinstance(axes, int):
        axes = [axes]
    slices = [slice(None, None, -1) if i in axes else slice(None, None, None) for i in range(x.ndim)]
    return x[slices]


def pattern_broadcast(x, broatcastable):
    return T.patternbroadcast(x, broatcastable)

# VALUE MANIPULATION


def get_value(x):
    if not hasattr(x, 'get_value'):
        raise TypeError('get_value() can only be called on a variable. '
                        'If you have an expression instead, use eval().')
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
    """Print the message and the tensor when evaluated and return the same
    tensor.
    """
    p_op = Print(message)
    return p_op(x)


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        unique_variables_to_update = {}
        for v, nv in updates:
            if v not in unique_variables_to_update:
                unique_variables_to_update[v] = nv
        updates = unique_variables_to_update.items()
        self.function = theano.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        **kwargs)

    def __call__(self, inputs):
        assert isinstance(inputs, (list, tuple))
        return self.function(*inputs)


def function(inputs, outputs, updates=[], **kwargs):
    if len(kwargs) > 0:
        function_args = inspect.getargspec(theano.function)[0]
        for key in kwargs.keys():
            if key not in function_args:
                msg = 'Invalid argument "%s" passed to K.function' % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    return T.grad(loss, variables)


def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    return theano.gradient.disconnected_grad(variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

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
        mask: binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: must be specified if using `unroll`.

    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    """
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise ValueError('When specifying `unroll=True`, '
                             'an `input_length` '
                             'must be provided to `rnn`.')

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim - 1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, new_states = step_function(inputs[i], states + constants)

                if len(successive_outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = T.switch(mask[i], output, prev_output)
                kept_states = []
                for state, new_state in zip(states, new_states):
                    kept_states.append(T.switch(mask[i], new_state, state))
                states = kept_states

                successive_outputs.append(output)
                successive_states.append(states)

            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 0, 1)

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
                output, states = step_function(inputs[i], states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

        else:
            def _step(input, *states):
                output, new_states = step_function(input, states)
                return [output] + new_states

            # Theano likes to make shape==1 dimensions in the initial states (outputs_info) broadcastable
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 1)

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
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    """condition: scalar tensor.
    """
    if callable(then_expression):
        then_expression = then_expression()
    if callable(else_expression):
        else_expression = else_expression()
    return T.switch(condition, then_expression, else_expression)


def in_train_phase(x, alt):
    if callable(x):
        x = x()
    if callable(alt):
        alt = alt()
    if _LEARNING_PHASE is 1:
        return x
    elif _LEARNING_PHASE is 0:
        return alt
    x = theano.ifelse.ifelse(_LEARNING_PHASE, x, alt)
    x._uses_learning_phase = True
    return x


def in_test_phase(x, alt):
    if callable(x):
        x = x()
    if callable(alt):
        alt = alt()
    if _LEARNING_PHASE is 1:
        return alt
    elif _LEARNING_PHASE is 0:
        return x
    x = theano.ifelse.ifelse(_LEARNING_PHASE, alt, x)
    x._uses_learning_phase = True
    return x


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


def relu(x, alpha=0., max_value=None):
    _assert_has_capability(T.nnet, 'relu')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def softsign(x):
    return T_softsign(x)


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def sparse_categorical_crossentropy(output, target, from_logits=False):
    target = T.cast(T.flatten(target), 'int32')
    target = T.extra_ops.to_one_hot(target, nb_class=output.shape[-1])
    target = reshape(target, shape(output))
    return categorical_crossentropy(output, target, from_logits)


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


def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`

    # Arguments
        predictions: A tensor of shape batch_size x classess and type float32.
        targets: A tensor of shape batch_size and type int32 or int64.
        k: An int, number of top elements to consider.

    # Returns
        A tensor of shape batch_size and type int. output_i is 1 if
        targets_i is within top-k values of predictions_i
    """
    predictions_top_k = T.argsort(predictions)[:, -k:]
    result, _ = theano.map(lambda prediction, target: any(equal(prediction, target)), sequences=[predictions_top_k, targets])
    return result


# CONVOLUTIONS

def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = x.dimshuffle((0, 3, 1, 2))
    return x


def _preprocess_conv3d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, slices)
        # TF input shape: (samples, rows, cols, slices, input_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = kernel.dimshuffle((3, 2, 0, 1))
    return kernel


def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols, slices)
        # TF kernel shape: (rows, cols, slices, input_depth, depth)
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
    return kernel


def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        th_border_mode = 'half'
    elif border_mode == 'valid':
        th_border_mode = 'valid'
    elif border_mode == 'full':
        th_border_mode = 'full'
    else:
        raise ValueError('Border mode not supported:', str(border_mode))
    return th_border_mode


def _preprocess_conv2d_image_shape(dim_ordering, image_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)
    return image_shape


def _preprocess_conv3d_volume_shape(dim_ordering, volume_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
    if volume_shape is not None:
        volume_shape = tuple(int_or_none(v) for v in volume_shape)
    return volume_shape


def _preprocess_conv2d_filter_shape(dim_ordering, filter_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _preprocess_conv3d_filter_shape(dim_ordering, filter_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1], filter_shape[2])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _postprocess_conv2d_output(conv_out, x, border_mode, kernel_shape, strides, dim_ordering):
    if border_mode == 'same':
        if kernel_shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if kernel_shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def _postprocess_conv3d_output(conv_out, x, border_mode, kernel_shape, strides, dim_ordering):
    if border_mode == 'same':
        if kernel_shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :, :]
        if kernel_shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1], :]
        if kernel_shape[4] % 2 == 0:
            conv_out = conv_out[:, :, :, :, :(x.shape[4] + strides[2] - 1) // strides[2]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))
    return conv_out


def conv1d(x, kernel, stride=1, border_mode='valid',
           image_shape=None, filter_shape=None):
    """1D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: stride integer.
        border_mode: string, "same" or "valid".
    """
    raise NotImplementedError


def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering='default', image_shape=None,
           filter_shape=None, filter_dilation=(1, 1)):
    """2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ', dim_ordering)

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    th_border_mode = _preprocess_border_mode(border_mode)

    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape

    image_shape = _preprocess_conv2d_image_shape(dim_ordering, image_shape)
    filter_shape = _preprocess_conv2d_filter_shape(dim_ordering, filter_shape)

    # TODO: remove the if statement when theano with no filter dilation is deprecated.
    if filter_dilation == (1, 1):
        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape)
    else:
        # T.nnet.conv2d uses **kwargs, so the filter_dilation parameter will be
        # ignored by versions that do not support it
        if 'filter_dilation' not in inspect.getargspec(T.nnet.conv2d).args:
            raise ValueError('conv2d with filter dilation requires Theano '
                             '0.9.0dev2 or newer.')
        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape,
                                 filter_dilation=filter_dilation)

    conv_out = _postprocess_conv2d_output(conv_out, x, border_mode,
                                          kernel_shape, strides, dim_ordering)
    return conv_out


def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    """2D deconvolution (transposed convolution).

    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    """
    flip_filters = False
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + dim_ordering)

    if dim_ordering == 'tf':
        output_shape = (output_shape[0], output_shape[3], output_shape[1], output_shape[2])

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    kernel = kernel.dimshuffle((1, 0, 2, 3))
    th_border_mode = _preprocess_border_mode(border_mode)

    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape

    filter_shape = _preprocess_conv2d_filter_shape(dim_ordering, filter_shape)
    filter_shape = tuple(filter_shape[i] for i in (1, 0, 2, 3))

    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=output_shape,
                                                        kshp=filter_shape,
                                                        subsample=strides,
                                                        border_mode=th_border_mode,
                                                        filter_flip=not flip_filters)
    conv_out = op(kernel, x, output_shape[2:])

    conv_out = _postprocess_conv2d_output(conv_out, x, border_mode,
                                          kernel_shape, strides, dim_ordering)
    return conv_out


def atrous_conv2d(x, kernel, rate=1,
                  border_mode='valid',
                  dim_ordering='default',
                  image_shape=None, filter_shape=None):
    raise NotImplementedError


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     border_mode='valid', dim_ordering='default'):
    raise NotImplementedError


def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='default',
           volume_shape=None, filter_shape=None,
           filter_dilation=(1, 1, 1)):
    """3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    # TODO: remove this if statement when Theano without AbstractConv3d is deprecated
    if not hasattr(T.nnet, 'conv3d'):
        if filter_dilation != (1, 1, 1):
            raise ValueError('conv3d with filter dilation requires Theano '
                             '0.9.0dev3 or newer.')

        return _old_theano_conv3d(x, kernel, strides, border_mode,
                                  dim_ordering, volume_shape, filter_shape)

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    th_border_mode = _preprocess_border_mode(border_mode)

    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape

    volume_shape = _preprocess_conv3d_volume_shape(dim_ordering, volume_shape)
    filter_shape = _preprocess_conv3d_filter_shape(dim_ordering, filter_shape)

    conv_out = T.nnet.conv3d(x, kernel,
                             border_mode=th_border_mode,
                             subsample=strides,
                             input_shape=volume_shape,
                             filter_shape=filter_shape,
                             filter_dilation=filter_dilation)

    conv_out = _postprocess_conv3d_output(conv_out, x, border_mode,
                                          kernel_shape, strides, dim_ordering)
    return conv_out


# TODO: remove this function when theano without AbstractConv3d is deprecated
def _old_theano_conv3d(x, kernel, strides=(1, 1, 1),
                       border_mode='valid', dim_ordering='default',
                       volume_shape=None, filter_shape=None):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)
    if border_mode not in {'same', 'valid'}:
        raise ValueError('Invalid border mode:', border_mode)

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1], filter_shape[2])

    if border_mode == 'same':
        assert(strides == (1, 1, 1))
        pad_dim1 = (kernel.shape[2] - 1)
        pad_dim2 = (kernel.shape[3] - 1)
        pad_dim3 = (kernel.shape[4] - 1)
        output_shape = (x.shape[0], x.shape[1],
                        x.shape[2] + pad_dim1,
                        x.shape[3] + pad_dim2,
                        x.shape[4] + pad_dim3)
        output = T.zeros(output_shape)
        indices = (slice(None), slice(None),
                   slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                   slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                   slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
        x = T.set_subtensor(output[indices], x)
        border_mode = 'valid'

    border_mode_3d = (border_mode, border_mode, border_mode)
    conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                               filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                               border_mode=border_mode_3d)
    conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

    # support strides by manually slicing the output
    if strides != (1, 1, 1):
        conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]

    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))

    return conv_out


def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='default', pool_mode='max'):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    assert pool_size[0] >= 1 and pool_size[1] >= 1

    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] > 2 and pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] > 2 and pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    else:
        raise ValueError('Invalid border mode:', border_mode)

    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        # TODO remove the old call once Theano older than 0.9.0dev4 is deprecated
        try:
            # new interface (introduced in 0.9.0dev4)
            pool_out = pool.pool_2d(x, ws=pool_size, stride=strides,
                                    ignore_border=True,
                                    pad=padding,
                                    mode='max')
        except TypeError:
            # old interface
            pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                    ignore_border=True,
                                    padding=padding,
                                    mode='max')
    elif pool_mode == 'avg':
        # TODO remove the old call once Theano older than 0.9.0dev4 is deprecated
        try:
            # new interface (introduced in 0.9.0dev4)
            pool_out = pool.pool_2d(x, ws=pool_size, stride=strides,
                                    ignore_border=True,
                                    pad=padding,
                                    mode='average_exc_pad')
        except TypeError:
            # old interface
            pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                    ignore_border=True,
                                    padding=padding,
                                    mode='average_exc_pad')
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    if border_mode == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

        pool_out = pool_out[:, :,
                            : expected_width,
                            : expected_height]

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='default', pool_mode='max'):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    # TODO: remove this if statement when Theano without pool_3d is deprecated
    #       (pool_3d was introduced after 0.9.0dev3)
    if not hasattr(T.signal.pool, 'pool_3d'):
        return _old_theano_pool3d(x, pool_size, strides, border_mode,
                                  dim_ordering, pool_mode)

    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        d_pad = pool_size[2] - 2 if pool_size[2] % 2 == 1 else pool_size[2] - 1
        padding = (w_pad, h_pad, d_pad)
    elif border_mode == 'valid':
        padding = (0, 0, 0)
    else:
        raise ValueError('Invalid border mode:', border_mode)
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))

    if pool_mode == 'max':
        # TODO remove the old call once Theano older than 0.9.0dev4 is deprecated
        try:
            # new interface (introduced in 0.9.0dev4)
            pool_out = pool.pool_3d(x, ws=pool_size, stride=strides,
                                    ignore_border=True,
                                    pad=padding,
                                    mode='max')
        except TypeError:
            # old interface
            pool_out = pool.pool_3d(x, ds=pool_size, st=strides,
                                    ignore_border=True,
                                    padding=padding,
                                    mode='max')
    elif pool_mode == 'avg':
        # TODO remove the old call once Theano older than 0.9.0dev4 is deprecated
        try:
            # new interface (introduced in 0.9.0dev4)
            pool_out = pool.pool_3d(x, ws=pool_size, stride=strides,
                                    ignore_border=True,
                                    pad=padding,
                                    mode='average_exc_pad')
        except TypeError:
            # old interface
            pool_out = pool.pool_3d(x, ds=pool_size, st=strides,
                                    ignore_border=True,
                                    padding=padding,
                                    mode='average_exc_pad')
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    if border_mode == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
        expected_depth = (x.shape[4] + strides[2] - 1) // strides[2]

        pool_out = pool_out[:, :,
                            : expected_width,
                            : expected_height,
                            : expected_depth]

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out


# TODO: remove this function when Theano without pool_3d is deprecated
#       (pool_3d was introduced after 0.9.0dev3)
def _old_theano_pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
                       dim_ordering='default', pool_mode='max'):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    if border_mode == 'same':
        # TODO: add implementation for border_mode="same"
        raise ValueError('border_mode="same" not supported with Theano.')
    elif border_mode == 'valid':
        ignore_border = True
        padding = (0, 0)
    else:
        raise ValueError('Invalid border mode:', border_mode)

    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))

    if pool_mode == 'max':
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode='max')

        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode='max')

    elif pool_mode == 'avg':
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode='average_exc_pad')

        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode='average_exc_pad')
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out


# RANDOMNESS


def random_normal(shape, mean=0.0, std=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.binomial(shape, p=p, dtype=dtype)

# Theano implementation of CTC
# Used with permission from Shawn Tan
# https://github.com/shawntan/
# Note that tensorflow's native CTC code is significantly
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
    _p_prev = T.inc_subtensor(_p_prev[active_skip_idxs + 2], p_prev[active_skip_idxs])
    updated_log_p_prev = T.log(_p_prev) + common_factor

    log_p_next = T.set_subtensor(
        zeros[:active_next],
        log_p_curr[:active_next] + updated_log_p_prev
    )
    return active_next, log_p_next


def ctc_path_probs(predict, Y, alpha=1e-4):
    smoothed_predict = (1 - alpha) * predict[:, Y] + alpha * np.float32(1.) / Y.shape[0]
    L = T.log(smoothed_predict)
    zeros = T.zeros_like(L[0])
    log_first = zeros

    f_skip_idxs = ctc_create_skip_idxs(Y)
    b_skip_idxs = ctc_create_skip_idxs(Y[::-1])  # there should be a shortcut to calculating this

    def step(log_f_curr, log_b_curr, f_active, log_f_prev, b_active, log_b_prev):
        f_active_next, log_f_next = ctc_update_log_p(f_skip_idxs, zeros, f_active, log_f_curr, log_f_prev)
        b_active_next, log_b_next = ctc_update_log_p(b_skip_idxs, zeros, b_active, log_b_curr, log_b_prev)
        return f_active_next, log_f_next, b_active_next, log_b_next

    [f_active, log_f_probs, b_active, log_b_probs], _ = theano.scan(
        step, sequences=[L, L[::-1, ::-1]], outputs_info=[np.int32(1), log_first, np.int32(1), log_first])

    idxs = T.arange(L.shape[1]).dimshuffle('x', 0)
    mask = (idxs < f_active.dimshuffle(0, 'x')) & (idxs < b_active.dimshuffle(0, 'x'))[::-1, ::-1]
    log_probs = log_f_probs + log_b_probs[::-1, ::-1] - L
    return log_probs, mask


def ctc_cost(predict, Y):
    log_probs, mask = ctc_path_probs(predict, ctc_interleave_blanks(Y))
    common_factor = T.max(log_probs)
    total_log_prob = T.log(T.sum(T.exp(log_probs - common_factor)[mask.nonzero()])) + common_factor
    return -total_log_prob


# batchifies original CTC code
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor (samples, max_string_length) containing the truth labels
        y_pred: tensor (samples, time_steps, num_categories) containing the prediction,
                or output of the softmax
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

def map_fn(fn, elems, name=None):
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
    fn2 = lambda x, acc: fn(acc, x)

    return theano.foldl(fn2, elems, initializer, name=name)[0]


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
    fn2 = lambda x, acc: fn(acc, x)

    return theano.foldr(fn2, elems, initializer, name=name)[0]
