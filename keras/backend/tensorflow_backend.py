import tensorflow as tf

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc

import numpy as np
import os
import warnings
from .common import floatx, _EPSILON, image_dim_ordering, reset_uids
py_all = all

# INTERNAL UTILS

# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None
# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}
# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# These two integers contain the tensorflow version for coping with API breaks.
tf_major_version = int(tf.__version__.split('.')[0])
tf_minor_version = int(tf.__version__.split('.')[1])


def clear_session():
    """Destroys the current TF graph and creates a new one.

    Useful to avoid clutter from old models / layers.
    """
    global _SESSION
    global _GRAPH_LEARNING_PHASES
    tf.reset_default_graph()
    reset_uids()
    _SESSION = None
    phase = tf.placeholder(dtype='bool', name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES[tf.get_default_graph()] = phase


def manual_variable_initialization(value):
    """Sets the manual variable initialization flag.

    This boolean flag determines whether
    variables should be initialized
    as they are instantiated (default), or if
    the user should handle the initialization
    (e.g. via `tf.initialize_all_variables()`).

    # Arguments
        value: Python boolean.
    """
    global _MANUAL_VAR_INIT
    _MANUAL_VAR_INIT = value


def learning_phase():
    """Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.
    """
    graph = tf.get_default_graph()
    if graph not in _GRAPH_LEARNING_PHASES:
        phase = tf.placeholder(dtype='bool',
                               name='keras_learning_phase')
        _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]


def set_learning_phase(value):
    """Sets the learning phase to a fixed value,
    either 0 or 1 (integers).

    # Raises
        ValueError: if `value` is neither `0` nor `1`.
    """
    global _GRAPH_LEARNING_PHASES
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _GRAPH_LEARNING_PHASES[tf.get_default_graph()] = value


def get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.

    # Returns
        A TensorFlow session.
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        _initialize_variables()
    return session


def set_session(session):
    """Sets the global TF session.
    """
    global _SESSION
    _SESSION = session


# VARIABLE MANIPULATION

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.

    # Arguments
        tensor: A tensor instance.

    # Returns
        A boolean.

    # Example
    ```python
        >>> from keras import backend as K
        >>> a = K.placeholder((2, 2), sparse=False)
        >>> print(K.is_sparse(a))
        False
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
    ```
    """
    return isinstance(tensor, tf.SparseTensor)


def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor
    and returns it.

    # Arguments
        tensor: A tensor instance (potentially sparse).

    # Returns
        A dense tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
        >>> c = K.to_dense(b)
        >>> print(K.is_sparse(c))
        False
    ```
    """
    if is_sparse(tensor):
        return tf.sparse_tensor_to_dense(tensor)
    else:
        return tensor


def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> kvar.eval()
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        if tf_major_version >= 1:
            v = tf.SparseTensor(indices=indices,
                                values=sparse_coo.data,
                                dense_shape=sparse_coo.shape)
        else:
            v = tf.SparseTensor(indices=indices,
                                values=sparse_coo.data,
                                shape=sparse_coo.shape)
        v._dims = len(sparse_coo.shape)
        v._keras_shape = sparse_coo.shape
        v._uses_learning_phase = False
        return v
    v = tf.Variable(value, dtype=_convert_string_dtype(dtype), name=name)
    if isinstance(value, np.ndarray):
        v._keras_shape = value.shape
    elif hasattr(value, 'get_shape'):
        v._keras_shape = tuple(map(int, value.get_shape()))
    v._uses_learning_phase = False
    return v


def _initialize_variables():
    if hasattr(tf, 'global_variables'):
        variables = tf.global_variables()
    else:
        variables = tf.all_variables()

    uninitialized_variables = []
    for v in variables:
        if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
            uninitialized_variables.append(v)
            v._keras_initialized = True
    if uninitialized_variables:
        sess = get_session()
        if hasattr(tf, 'variables_initializer'):
            sess.run(tf.variables_initializer(uninitialized_variables))
        else:
            sess.run(tf.initialize_variables(uninitialized_variables))


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    """
    if dtype is None:
        dtype = floatx()
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    if sparse:
        x = tf.sparse_placeholder(dtype, name=name)
        x._dims = len(shape)
    else:
        x = tf.placeholder(dtype, shape=shape, name=name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```
        # TensorFlow example
        >>> from keras import backend as K
        >>> tf_session = K.get_session()
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> input = keras.backend.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
        >>> K.shape(input)
        <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
        # To get integer shape (Instead, you can use K.int_shape(x))
        >>> K.shape(kvar).eval(session=tf_session)
        array([2, 2], dtype=int32)
        >>> K.shape(input).eval(session=tf_session)
        array([2, 4, 5], dtype=int32)
    ```
    """
    return tf.shape(x)


def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(input)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(input)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    if is_sparse(x):
        return x._dims

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    """
    return x.dtype.name


def eval(x):
    """Evaluates the value of a variable.
    Returns a Numpy array.

    # Arguments
        x: A variable.

    # Returns
        A Numpy array.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    """
    return to_dense(x).eval(session=get_session())


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.

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
    shape = tuple(map(int, shape))
    tf_dtype = _convert_string_dtype(dtype)
    return variable(tf.constant_initializer(0., dtype=tf_dtype)(shape),
                    dtype, name)


def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones tensor variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, filled with `1.0`.

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
    shape = tuple(map(int, shape))
    tf_dtype = _convert_string_dtype(dtype)
    return variable(tf.constant_initializer(1., dtype=tf_dtype)(shape),
                    dtype, name)


def eye(size, dtype=None, name=None):
    """Instantiate an identity matrix and returns it.

    # Arguments
        size: Integer, number of rows/columns.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, an identity matrix.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.eye(3)
        >>> K.eval(kvar)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
    ```

    """
    return variable(np.eye(size), dtype, name)


def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros Keras variable
    of the same shape as another Keras variable or tensor and returns it.

    # Arguments
        x: Keras variable or Keras tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.

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
    return tf.zeros_like(x, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
    """Instantiates an all-ones Keras variable
    of the same shape as another Keras variable or tensor and returns it.

    # Arguments
        x: Keras variable or tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.

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
    return tf.ones_like(x, dtype=dtype, name=name)


def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    """Instantiates an Keras variable filled with
    samples drawn from a uniform distribution and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output inteval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_uniform_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
        >>> K.eval(kvar)
        array([[ 0.10940075,  0.10047495,  0.476143  ],
               [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    shape = tuple(map(int, shape))
    tf_dtype = _convert_string_dtype(dtype)
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.random_uniform_initializer(
        low, high, dtype=tf_dtype, seed=seed)(shape)
    return variable(value, dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    """Instantiates an Keras variable filled with
    samples drawn from a normal distribution and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        mean: Float, mean of the normal distribution.
        scale: Float, standard deviation of the normal distribution.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        # TensorFlow example
        >>> kvar = K.random_normal_variable((2,3), 0, 1)
        >>> kvar
        <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
        >>> K.eval(kvar)
        array([[ 1.19591331,  0.68685907, -0.63814116],
               [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    shape = tuple(map(int, shape))
    tf_dtype = _convert_string_dtype(dtype)
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.random_normal_initializer(
        mean, scale, dtype=tf_dtype, seed=seed)(shape)
    return variable(value, dtype=dtype, name=name)


def count_params(x):
    """Returns the number of scalars in a Keras variable.

    # Arguments
        x: Keras variable.

    # Returns
        Integer, the number of scalars in `x`.

    # Example
    ```python
        >>> kvar = K.zeros((2,3))
        >>> K.count_params(kvar)
        6
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    shape = x.get_shape()
    return np.prod([shape[i]._value for i in range(len(shape))])


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
    return tf.cast(x, dtype)


# UPDATES OPS


def update(x, new_x):
    return tf.assign(x, new_x)


def update_add(x, increment):
    return tf.assign_add(x, increment)


def update_sub(x, decrement):
    return tf.assign_sub(x, decrement)


def moving_average_update(variable, value, momentum):
    try:
        return moving_averages.assign_moving_average(
            variable, value, momentum, zero_debias=False)
    except TypeError:
        return moving_averages.assign_moving_average(
            variable, value, momentum)


# LINEAR ALGEBRA

def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a ND tensor
    with a ND tensor, it reproduces the Theano behavior.
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    if hasattr(tf, 'unstack'):
        unstack = tf.unstack
    else:
        unstack = tf.unpack
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x, y: Keras tensors or variables with `ndim >= 2`
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
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
    if ndim(x) == 2 and ndim(y) == 2:
        if tf_major_version >= 1:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.multiply(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
        else:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.mul(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.mul(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        # TODO: remove later.
        if hasattr(tf, 'batch_matmul'):
            try:
                out = tf.batch_matmul(x, y, adj_a=adj_x, adj_b=adj_y)
            except TypeError:
                out = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
        else:
            out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def transpose(x):
    """Transposes a tensor and returns it.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.

    # Examples
    ```python
        >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
        >>> K.eval(var)
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> var_transposed = K.transpose(var)
        >>> K.eval(var_transposed)
        array([[ 1.,  4.],
               [ 2.,  5.],
               [ 3.,  6.]], dtype=float32)
    ```

    ```python
        >>> input = K.placeholder((2, 3))
        >>> input
        <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
        >>> input_transposed = K.transpose(input)
        >>> input_transposed
        <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

    ```
    """
    return tf.transpose(x)


def gather(reference, indices):
    """Retrieves the elements of indices `indices`
    in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    return tf.gather(reference, indices)


# ELEMENT-WISE OPERATIONS

def _normalize_axis(axis, ndim):
    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list):
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis


def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find minimum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          reduction_indices=axis,
                          keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_all(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmax(x, axis)


def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmin(x, axis)


def square(x):
    """Element-wise square.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.square(x)


def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.abs(x)


def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    zero = _to_tensor(0., x.dtype.base_dtype)
    inf = _to_tensor(np.inf, x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, inf)
    return tf.sqrt(x)


def exp(x):
    """Element-wise exponential.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.exp(x)


def log(x):
    """Element-wise log.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.log(x)


def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.round(x)


def sign(x):
    """Element-wise sign.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.sign(x)


def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.pow(x, a)


def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Returns
        A tensor.
    """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    min_value = _to_tensor(min_value, x.dtype.base_dtype)
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    return tf.clip_by_value(x, min_value, max_value)


def equal(x, y):
    """Element-wise equality between two tensors.

    # Returns
        A bool tensor.
    """
    return tf.equal(x, y)


def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Returns
        A bool tensor.
    """
    return tf.not_equal(x, y)


def greater(x, y):
    """Element-wise truth value of (x > y).

    # Returns
        A bool tensor.
    """
    return tf.greater(x, y)


def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Returns
        A bool tensor.
    """
    return tf.greater_equal(x, y)


def lesser(x, y):
    """Element-wise truth value of (x < y).

    # Returns
        A bool tensor.
    """
    return tf.less(x, y)


def lesser_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Returns
        A bool tensor.
    """
    return tf.less_equal(x, y)


def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Returns
        A tensor.
    """
    return tf.maximum(x, y)


def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Returns
        A tensor.
    """
    return tf.minimum(x, y)


def sin(x):
    """Computes sin of x element-wise.

    # Returns
        A tensor.
    """
    return tf.sin(x)


def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    return tf.cos(x)


def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    mean, var = tf.nn.moments(x, reduction_axes,
                              shift=None, name=None, keep_dims=False)
    if sorted(reduction_axes) == range(ndim(x))[:-1]:
        normed = tf.nn.batch_normalization(x, mean, var,
                                           beta, gamma,
                                           epsilon)
    else:
        # need broadcasting
        target_shape = []
        for axis in range(ndim(x)):
            if axis in reduction_axes:
                target_shape.append(1)
            else:
                target_shape.append(tf.shape(x)[axis])
        target_shape = stack(target_shape)

        broadcast_mean = tf.reshape(mean, target_shape)
        broadcast_var = tf.reshape(var, target_shape)
        broadcast_gamma = tf.reshape(gamma, target_shape)
        broadcast_beta = tf.reshape(beta, target_shape)
        normed = tf.nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                           broadcast_beta, broadcast_gamma,
                                           epsilon)
    return normed, mean, var


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma:

    output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta

    # Returns
        A tensor.
    """
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.

    # Returns
        A tensor.
    """
    if axis < 0:
        dims = ndim(tensors[0])
        if dims:
            axis = axis % dims
        else:
            axis = 0

    if py_all([is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        if tf_major_version >= 1:
            return tf.concat([to_dense(x) for x in tensors], axis)
        else:
            try:
                return tf.concat_v2([to_dense(x) for x in tensors], axis)
            except AttributeError:
                return tf.concat(axis, [to_dense(x) for x in tensors])


def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Returns
        A tensor.
    """
    return tf.reshape(x, shape)


def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        pattern: should be a tuple of
            dimension indices, e.g. (0, 2, 1).

    # Returns
        A tensor.
    """
    return tf.transpose(x, perm=pattern)


def resize_images(X, height_factor, width_factor, dim_ordering):
    """Resizes the images contained in a 4D tensor of shape
    - `[batch, channels, height, width]` (for 'th' dim_ordering)
    - `[batch, height, width, channels]` (for 'tf' dim_ordering)
    by a factor of `(height_factor, width_factor)`. Both factors should be
    positive integers.

    # Returns
        A tensor.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'th':
        original_shape = int_shape(X)
        new_shape = tf.shape(X)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_nearest_neighbor(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        X.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return X
    elif dim_ordering == 'tf':
        original_shape = int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_nearest_neighbor(X, new_shape)
        X.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return X
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)


def resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering):
    """Resizes the volume contained in a 5D tensor of shape
    - `[batch, channels, depth, height, width]` (for 'th' dim_ordering)
    - `[batch, depth, height, width, channels]` (for 'tf' dim_ordering)
    by a factor of `(depth_factor, height_factor, width_factor)`.
    All three factors should be positive integers.

    # Returns
        A tensor.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
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


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Returns
        A tensor.
    """
    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    try:
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
    except TypeError:
        splits = tf.split(value=x, num_split=x_shape[axis], split_dim=axis)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for i in range(rep)]
    return concatenate(x_rep, axis)


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Returns
        A tensor.
    """
    assert ndim(x) == 2
    x = tf.expand_dims(x, 1)
    pattern = stack([1, n, 1])
    return tf.tile(x, pattern)


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.
    """
    # Match the behavior of numpy and Theano by returning an empty seqence.
    if stop is None and start < 0:
        start = 0
    result = tf.range(start, limit=stop, delta=step, name='arange')
    if dtype != 'int32':
        result = cast(result, dtype)
    return result


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
    return tf.tile(x, n)


def flatten(x):
    """Flatten a tensor.

    # Returns
        A tensor, reshaped into 1-D
    """
    return tf.reshape(x, [-1])


def batch_flatten(x):
    """Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.

    In other words, it flattens each data samples of a batch.

    # Returns
        A tensor.
    """
    x = tf.reshape(x, stack([-1, prod(shape(x)[1:])]))
    return x


def expand_dims(x, dim=-1):
    """Adds a 1-sized dimension at index "dim".

    # Returns
        A tensor with expended dimensions.
    """
    return tf.expand_dims(x, dim)


def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    return tf.squeeze(x, [axis])


def temporal_padding(x, padding=1):
    """Pads the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    # Returns
        A padded 3D tensor.
    """
    pattern = [[0, 0], [padding, padding], [0, 0]]
    return tf.pad(x, pattern)


def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    """Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.

    # Returns
        A padded 3D tensor.
    """
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='default'):
    """Pads the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = [[0, 0], [0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]]]
    else:
        pattern = [[0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]],
                   [0, 0]]
    return tf.pad(x, pattern)


def asymmetric_spatial_2d_padding(x, top_pad=1, bottom_pad=1,
                                  left_pad=1, right_pad=1,
                                  dim_ordering='default'):
    """Pad the rows and columns of a 4D tensor
    with "top_pad", "bottom_pad", "left_pad", "right_pad" (resp.) zeros
    rows on top, bottom; cols on left, right.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = [[0, 0],
                   [0, 0],
                   [top_pad, bottom_pad],
                   [left_pad, right_pad]]
    else:
        pattern = [[0, 0],
                   [top_pad, bottom_pad],
                   [left_pad, right_pad],
                   [0, 0]]
    return tf.pad(x, pattern)


def spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='default'):
    """Pads 5D tensor with zeros for the depth, height, width dimension with
    "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right

    For 'tf' dim_ordering, the 2nd, 3rd and 4th dimension will be padded.
    For 'th' dim_ordering, the 3rd, 4th and 5th dimension will be padded.

    # Returns
        A padded 5D tensor.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.

    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'th':
        pattern = [
            [0, 0],
            [0, 0],
            [padding[0], padding[0]],
            [padding[1], padding[1]],
            [padding[2], padding[2]]
        ]
    else:
        pattern = [
            [0, 0],
            [padding[0], padding[0]],
            [padding[1], padding[1]],
            [padding[2], padding[2]],
            [0, 0]
        ]
    return tf.pad(x, pattern)


def stack(x):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: input tensor.

    # Returns
        A tensor.
    """
    try:
        return tf.stack(x)
    except AttributeError:
        return tf.pack(x)


def one_hot(indices, nb_classes):
    """Input: nD integer tensor of shape `(batch_size, dim1, dim2, ... dim(n-1))`
    Output: (n + 1)D one hot representation of the input
    with shape `(batch_size, dim1, dim2, ... dim(n-1), nb_classes)`

    # Returns
        The one-hot tensor.
    """
    return tf.one_hot(indices, depth=nb_classes, axis=-1)


def reverse(x, axes):
    """Reverse a tensor along the the specified axes

    # Returns
        A tensor.
    """
    if isinstance(axes, int):
        axes = [axes]
    try:
        return tf.reverse_v2(x, axes)
    except AttributeError:
        # Older TF versions.
        dims = [True if i in axes else False for i in range(len(x.get_shape()._dims))]
        return tf.reverse(x, dims)


# VALUE MANIPULATION


def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    return x.eval(session=get_session())


def batch_get_value(xs):
    """Returns the value of more than one tensor variable.

    # Arguments
        x: list of variables.

    # Returns
        A list of Numpy arrays.
    """
    if xs:
        return get_session().run(xs)
    else:
        return []


def set_value(x, value):
    """Sets the value of a variable,
    from a Numpy array. It returns `None`.
    """
    value = np.asarray(value)
    tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
    if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
    else:
        assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
    get_session().run(assign_op, feed_dict={assign_placeholder: value})


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.
    It returns `None`.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    if tuples:
        assign_ops = []
        feed_dict = {}
        for x, value in tuples:
            value = np.asarray(value)
            tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
            if hasattr(x, '_assign_placeholder'):
                assign_placeholder = x._assign_placeholder
                assign_op = x._assign_op
            else:
                assign_placeholder = tf.placeholder(tf_dtype,
                                                    shape=value.shape)
                assign_op = x.assign(assign_placeholder)
                x._assign_placeholder = assign_placeholder
                x._assign_op = assign_op
            assign_ops.append(assign_op)
            feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)


def get_variable_shape(x):
    """Returns shape of a variable.

    # Arguments
        A variable.

    # Returns
        A tuple of integers.
    """
    return int_shape(x)


def print_tensor(x, message=''):
    """Print the message and the tensor when evaluated and return the same
    tensor.
    """
    return tf.Print(x, [x], message)


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[]):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            if is_sparse(tensor):
                sparse_coo = value.tocoo()
                indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                          np.expand_dims(sparse_coo.col, 1)), 1)
                value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value
        session = get_session()
        updated = session.run(self.outputs + [self.updates_op],
                              feed_dict=feed_dict)
        return updated[:len(self.outputs)]


def function(inputs, outputs, updates=[], **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: list of placeholder/variable tensors.
        outputs: list of output tensors.
        updates: list of update tuples (old_tensor, new_tensor).
    """
    if len(kwargs) > 0:
        msg = [
            'Expected no kwargs, you passed %s' % len(kwargs),
            'kwargs passed to function are ignored with Tensorflow backend'
        ]
        warnings.warn('\n'.join(msg))
    return Function(inputs, outputs, updates=updates)


def gradients(loss, variables):
    """Returns the gradients of `variables` (list of tensor variables)
    with regard to `loss`.
    """
    return tf.gradients(loss, variables, colocate_gradients_with_ops=True)


def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    return tf.stop_gradient(variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                entry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
    """
    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.get_shape()) == ndim - 1:
            mask = expand_dims(mask)
        mask = tf.transpose(mask, axes)

    if constants is None:
        constants = []

    # TODO: remove later.
    if hasattr(tf, 'select'):
        tf.where = tf.select
    if hasattr(tf, 'stack'):
        stack = tf.stack
        unstack = tf.unstack
    else:
        stack = tf.pack
        unstack = tf.unpack

    if unroll:
        if not inputs.get_shape()[0]:
            raise ValueError('Unrolling requires a '
                             'fixed number of timesteps.')
        states = initial_states
        successive_states = []
        successive_outputs = []

        input_list = unstack(inputs)
        if go_backwards:
            input_list.reverse()

        if mask is not None:
            mask_list = unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for input, mask_t in zip(input_list, mask_list):
                output, new_states = step_function(input, states + constants)

                # tf.where needs its condition tensor
                # to be the same shape as its two
                # result tensors, but in our case
                # the condition (mask) tensor is
                # (nsamples, 1), and A and B are (nsamples, ndimensions).
                # So we need to
                # broadcast the mask to match the shape of A and B.
                # That's what the tile call does,
                # it just repeats the mask along its second dimension
                # n times.
                tiled_mask_t = tf.tile(mask_t,
                                       stack([1, tf.shape(output)[1]]))

                if len(successive_outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t,
                                           stack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.where(tiled_mask_t,
                                                  new_state,
                                                  state))
                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = stack(successive_outputs)
        else:
            for input in input_list:
                output, states = step_function(input, states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = stack(successive_outputs)

    else:
        if go_backwards:
            inputs = reverse(inputs, 0)

        states = tuple(initial_states)

        time_steps = tf.shape(inputs)[0]
        output_ta = tensor_array_ops.TensorArray(
            dtype=inputs.dtype,
            size=time_steps,
            tensor_array_name='output_ta')
        input_ta = tensor_array_ops.TensorArray(
            dtype=inputs.dtype,
            size=time_steps,
            tensor_array_name='input_ta')
        if hasattr(input_ta, 'unstack'):
            input_ta = input_ta.unstack(inputs)
        else:
            input_ta = input_ta.unpack(inputs)
        time = tf.constant(0, dtype='int32', name='time')

        if mask is not None:
            if len(states) == 0:
                raise ValueError('No initial states provided! '
                                 'When using masking in an RNN, you should '
                                 'provide initial states '
                                 '(and your step function should return '
                                 'as its first state at time `t` '
                                 'the output at time `t-1`).')
            if go_backwards:
                mask = reverse(mask, 0)

            mask_ta = tensor_array_ops.TensorArray(
                dtype=tf.bool,
                size=time_steps,
                tensor_array_name='mask_ta')
            if hasattr(mask_ta, 'unstack'):
                mask_ta = mask_ta.unstack(mask)
            else:
                mask_ta = mask_ta.unpack(mask)

            def _step(time, output_ta_t, *states):
                current_input = input_ta.read(time)
                mask_t = mask_ta.read(time)
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                tiled_mask_t = tf.tile(mask_t,
                                       stack([1, tf.shape(output)[1]]))
                output = tf.where(tiled_mask_t, output, states[0])
                new_states = [tf.where(tiled_mask_t, new_states[i], states[i]) for i in range(len(states))]
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)
        else:
            def _step(time, output_ta_t, *states):
                current_input = input_ta.read(time)
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)

        final_outputs = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_step,
            loop_vars=(time, output_ta) + states,
            parallel_iterations=32,
            swap_memory=True)
        last_time = final_outputs[0]
        output_ta = final_outputs[1]
        new_states = final_outputs[2:]

        if hasattr(output_ta, 'stack'):
            outputs = output_ta.stack()
        else:
            outputs = output_ta.pack()
        last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def _cond(condition, then_lambda, else_lambda):
    """Backwards compatible interface to tf.cond prior to public introduction.
    """
    try:
        cond_fn = tf.cond
    except AttributeError:
        from tensorflow.python.ops import control_flow_ops
        cond_fn = control_flow_ops.cond
    return cond_fn(condition, then_lambda, else_lambda)


def switch(condition, then_expression, else_expression):
    """Switches between two operations
    depending on a scalar value (`int` or `bool`).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.
    """
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, 'bool')
    if not callable(then_expression):
        def then_expression_fn():
            return then_expression
    else:
        then_expression_fn = then_expression
    if not callable(else_expression):
        def else_expression_fn():
            return else_expression
    else:
        else_expression_fn = else_expression
    x = _cond(condition,
              then_expression_fn,
              else_expression_fn)
    return x


def in_train_phase(x, alt):
    """Selects `x` in train phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    """
    if learning_phase() is 1:
        if callable(x):
            return x()
        else:
            return x
    elif learning_phase() is 0:
        if callable(alt):
            return alt()
        else:
            return alt
    # else: assume learning phase is a placeholder tensor.
    x = switch(learning_phase(), x, alt)
    x._uses_learning_phase = True
    return x


def in_test_phase(x, alt):
    """Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    """
    if learning_phase() is 1:
        if callable(alt):
            return alt()
        else:
            return alt
    elif learning_phase() is 0:
        if callable(x):
            return x()
        else:
            return x
    # else: assume learning phase is a placeholder tensor.
    x = switch(learning_phase(), alt, x)
    x._uses_learning_phase = True
    return x


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    """Rectified linear unit.
    With default values, it returns element-wise `max(x, 0)`.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: Saturation threshold.

    # Returns
        A tensor.
    """
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    # Returns
        A tensor.
    """
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)


def softmax(x):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return tf.nn.softmax(x)


def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return tf.nn.softplus(x)


def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return tf.nn.softsign(x)


def categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        try:
            return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)
        except TypeError:
            return tf.nn.softmax_cross_entropy_with_logits(output, target)


def sparse_categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy between an output tensor
    and a target tensor, where the target is an integer tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    try:
        res = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits)
    except TypeError:
        res = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, targets)
    if len(output_shape) == 3:
        # if our output includes timesteps we need to reshape
        return tf.reshape(res, tf.shape(output)[:-1])
    else:
        return res


def binary_crossentropy(output, target, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        output: A tensor.
        target: A tensor with the same shape as `output`.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                       logits=output)
    except TypeError:
        return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    x = (0.2 * x) + 0.5
    zero = _to_tensor(0., x.dtype.base_dtype)
    one = _to_tensor(1., x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, one)
    return x


def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return tf.nn.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random, while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.

    # Returns
        A tensor.
    """
    retain_prob = 1. - level
    if seed is None:
        seed = np.random.randint(10e6)
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incomptability)
    return tf.nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)


def l2_normalize(x, axis):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: input tensor.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`

    # Arguments
        predictions: A tensor of shape `batch_size` x classes and type `float32`.
        targets: A tensor of shape batch_size and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A tensor of shape `batch_size` and type `bool`. `output_i` is `True` if
        `targets_i` is within top-k values of `predictions_i`
    """
    return tf.nn.in_top_k(predictions, targets, k)


# CONVOLUTIONS

def _preprocess_deconv_output_shape(x, shape, dim_ordering):
    if dim_ordering == 'th':
        shape = (shape[0], shape[2], shape[3], shape[1])

    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape


def _preprocess_conv2d_input(x, dim_ordering):
    if dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x


def _preprocess_conv3d_input(x, dim_ordering):
    if dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dtype(kernel) == 'float64':
        kernel = tf.cast(kernel, 'float32')
    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
    return kernel


def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if dtype(kernel) == 'float64':
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
        raise ValueError('Invalid border mode:', border_mode)
    return padding


def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 3, 1, 2))

    if floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x


def _postprocess_conv3d_output(x, dim_ordering):
    if dim_ordering == 'th':
        x = tf.transpose(x, (0, 4, 1, 2, 3))

    if floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x


def conv1d(x, kernel, stride=1, border_mode='valid',
           image_shape=None, filter_shape=None):
    """1D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: stride integer.
        border_mode: string, `"same"` or `"valid"`.

    # Returns
        A tensor, result of 1D convolution.
    """
    # pre-process dtype
    x_dtype = dtype(x)
    if x_dtype == 'float64':
        x = tf.cast(x, 'float32')
        kernel = tf.cast(kernel, 'float32')
    padding = _preprocess_border_mode(border_mode)
    x = tf.nn.conv1d(x, kernel, stride, padding=padding)
    # post-process dtype
    if x_dtype == 'float64':
        x = tf.cast(x, 'float64')
    return x


def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering='default',
           image_shape=None, filter_shape=None, filter_dilation=(1, 1)):
    """2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    if filter_dilation == (1, 1):
        strides = (1,) + strides + (1,)
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
    else:
        assert filter_dilation[0] == filter_dilation[1]
        assert strides == (1, 1), 'Invalid strides for dilated convolution'
        x = tf.nn.atrous_conv2d(x, kernel, filter_dilation[0], padding=padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def deconv2d(x, kernel, output_shape, strides=(1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    """2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of transposed 2D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    output_shape = _preprocess_deconv_output_shape(x, output_shape, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 3, 2))
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def atrous_conv2d(x, kernel, rate=1,
                  border_mode='valid',
                  dim_ordering='default',
                  image_shape=None, filter_shape=None):
    """Atrous 2D convolution. Also as known as dilated convolution.

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        rate: integer > 0, the sample stride.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of atrous transposed 2D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))
    if rate == 1:
        return conv2d(x, kernel, strides=(1, 1), border_mode=border_mode,
                      dim_ordering=dim_ordering)

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)

    x = tf.nn.atrous_conv2d(x, kernel, rate, padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     border_mode='valid', dim_ordering='default'):
    """2-D convolution with separable filters.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    depthwise_kernel = _preprocess_conv2d_kernel(depthwise_kernel,
                                                 dim_ordering)
    pointwise_kernel = _preprocess_conv2d_kernel(pointwise_kernel,
                                                 dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                               strides, padding)
    return _postprocess_conv2d_output(x, dim_ordering)


def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='default',
           volume_shape=None, filter_shape=None):
    """3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, `"same"` or `"valid"`.
        dim_ordering: `"tf"` or `"th"`.
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of 3D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d(x, kernel, strides, padding)
    return _postprocess_conv3d_output(x, dim_ordering)


def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='default',
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of `"valid"`, `"same"`.
        dim_ordering: one of `"th"`, `"tf"`.
        pool_mode: one of `"max"`, `"avg"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
        ValueError: if `pool_mode` is neither `max` or `avg`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = _preprocess_conv2d_input(x, dim_ordering)

    if pool_mode == 'max':
        x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    return _postprocess_conv2d_output(x, dim_ordering)


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='default', pool_mode='max'):
    """3D Pooling.

    # Arguments
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        border_mode: one of `"valid"`, `"same"`.
        dim_ordering: one of `"th"`, `"tf"`.
        pool_mode: one of `"max"`, `"avg"`.

    # Returns
        A tensor, result of 3D pooling.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
        ValueError: if `pool_mode` is neither `max` or `avg`.
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = _preprocess_conv3d_input(x, dim_ordering)

    if pool_mode == 'max':
        x = tf.nn.max_pool3d(x, pool_size, strides, padding=padding)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool3d(x, pool_size, strides, padding=padding)
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)

    return _postprocess_conv3d_output(x, dim_ordering)


# RANDOMNESS

def random_normal(shape, mean=0.0, std=1.0, dtype=None, seed=None):
    """Returns a tensor with normal distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: A float, mean of the normal distribution to draw samples.
        std: A float, standard deviation of the normal distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype, seed=seed)


def random_uniform(shape, low=0.0, high=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        low: A float, lower boundary of the uniform distribution
            to draw samples.
        high: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype, seed=seed)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with binomlai distribution

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomlai distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                    tf.ones(shape, dtype=dtype),
                    tf.zeros(shape, dtype=dtype))


# CTC
# tensorflow has a native implemenation, but it uses sparse tensors
# and therefore requires a wrapper for Keras. The functions below convert
# dense to sparse tensors and also wraps up the beam search code that is
# in tensorflow's CTC implementation


def ctc_label_dense_to_sparse(labels, label_lengths):
    # undocumented feature soon to be made public
    from tensorflow.python.ops import functional_ops
    label_shape = tf.shape(labels)
    num_batches_tns = stack([label_shape[0]])
    max_num_labels_tns = stack([label_shape[1]])

    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < tf.fill(max_num_labels_tns, current_input)

    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    dense_mask = functional_ops.scan(range_less_than, label_lengths,
                                     initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
                             label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]),
                                                  max_num_labels_tns), reverse(label_shape, 0)))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(tf.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

    vals_sparse = tf.gather_nd(labels, indices)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)` containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)` containing the prediction,
                or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
                each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
                each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element
    """
    label_length = tf.to_int32(tf.squeeze(label_length))
    input_length = tf.to_int32(tf.squeeze(input_length))
    sparse_labels = tf.to_int32(ctc_label_dense_to_sparse(y_true, label_length))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length), 1)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1):
    """Decodes the output of a softmax using either
       greedy (also known as best path) or a constrained dictionary
       search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)` containing the prediction,
                or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
                each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`. This does
                not use a dictionary
        beam_width: if `greedy` is `false`: a beam search decoder will be used
                with a beam of this width
        top_paths: if `greedy` is `false`: how many of the most probable paths will be returned

    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that contains
                the decoded sequence. If `false`, returns the `top_paths` most probable
                decoded sequences. Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains the log probability of each decoded sequence
    """
    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
    input_length = tf.to_int32(input_length)

    if greedy:
        (decoded, log_prob) = ctc.ctc_greedy_decoder(
            inputs=y_pred,
            sequence_length=input_length)
    else:
        (decoded, log_prob) = ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length, beam_width=beam_width,
            top_paths=top_paths)

    if tf_major_version >= 1:
        decoded_dense = [tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=-1)
                         for st in decoded]
    else:
        decoded_dense = [tf.sparse_to_dense(st.indices, st.shape, st.values, default_value=-1)
                         for st in decoded]

    return (decoded_dense, log_prob)


# HIGH ORDER FUNCTIONS

def map_fn(fn, elems, name=None):
    """Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph

    # Returns
        Tensor with first dimension equal to the elems and second depending on
        fn
    """
    return tf.map_fn(fn, elems, name=name)


def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Same type and shape as initializer
    """
    return tf.foldl(fn, elems, initializer=initializer, name=name)


def foldr(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[-1]` in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Same type and shape as initializer
    """
    return tf.foldr(fn, elems, initializer=initializer, name=name)
