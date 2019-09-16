from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tfdev
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import image_ops as tf_image_ops
from tensorflow.python.ops import math_ops as tf_math_ops
from tensorflow.python.ops import state_ops as tf_state_ops
from tensorflow.python.keras import backend as tf_keras_backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from .common import floatx, epsilon, image_data_format

import sys
import functools
import threading

import numpy as np
from distutils.version import StrictVersion

from ..utils.generic_utils import transpose_shape

py_all = all
py_any = any
py_sum = sum
py_slice = slice

# INTERNAL UTILS

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change during our lifetime.
_LOCAL_DEVICES = None

_SYMBOLIC_SCOPE = threading.local()
_SYMBOLIC_SCOPE.value = True
_LEARNING_PHASE_CACHE = {}


def _is_tf_1():
    return tf.__version__.startswith('1.')

# Set initial config
tf_keras_backend.set_floatx(floatx())
tf_keras_backend.set_epsilon(epsilon())
tf_keras_backend.set_image_data_format(image_data_format())


# Private TF Keras utils
get_graph = tf_keras_backend.get_graph
# learning_phase_scope = tf_keras_backend.learning_phase_scope  # TODO
name_scope = tf.name_scope


def symbolic(func):
    """Decorator used in TensorFlow 2.0 to enter the Keras graph.

    # Arguments
        func: Function to decorate.

    # Returns
        Decorated function.
    """
    if _is_tf_1():
        return func

    @functools.wraps(func)
    def symbolic_fn_wrapper(*args, **kwargs):
        if _SYMBOLIC_SCOPE.value:
            with get_graph().as_default():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return symbolic_fn_wrapper


def is_symbolic(x):
    return isinstance(x, tf.Tensor) and hasattr(x, 'op')


def eager(func):
    """Decorator used in TensorFlow 2.0 to exit the Keras graph.

    # Arguments
        func: Function to decorate.

    # Returns
        Decorated function.
    """
    if _is_tf_1():
        return func

    global _SYMBOLIC_SCOPE

    @functools.wraps(func)
    def eager_fn_wrapper(*args, **kwargs):
        prev_value = _SYMBOLIC_SCOPE.value
        try:
            _SYMBOLIC_SCOPE.value = False
            with context.eager_mode():
                out = func(*args, **kwargs)
        finally:
            _SYMBOLIC_SCOPE.value = prev_value
        return out
    return eager_fn_wrapper


def _has_compat_v1():
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        return True
    return False


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
    return tf_keras_backend.get_uid(prefix)


def manual_variable_initialization(value):
    """Sets the manual variable initialization flag.

    This boolean flag determines whether
    variables should be initialized
    as they are instantiated (default), or if
    the user should handle the initialization.

    # Arguments
        value: Python boolean.
    """
    tf_keras_backend.manual_variable_initialization(value)


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return tf_keras_backend.epsilon()


def reset_uids():
    """Resets graph identifiers."""
    tf_keras_backend.reset_uids()


def set_epsilon(e):
    """Sets the value of the fuzz factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    """
    tf_keras_backend.set_epsilon(e)


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    """
    return tf_keras_backend.floatx()


def set_floatx(floatx):
    """Sets the default float type.

    # Arguments
        floatx: String, 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    """
    tf_keras_backend.set_floatx(floatx)


def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    """
    return tf_keras_backend.cast_to_floatx(x)


def image_data_format():
    """Returns the default image data format convention.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return tf_keras_backend.image_data_format()


def set_image_data_format(data_format):
    """Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    """
    tf_keras_backend.set_image_data_format(data_format)


def normalize_data_format(value):
    """Checks that the value correspond to a valid data format.

    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if value is None:
        value = image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


@symbolic
def learning_phase():
    """Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.

    # Returns
        Learning phase (scalar integer tensor or Python integer).
    """
    lp = tf_keras_backend.learning_phase()
    if _is_tf_1():
        return lp
    else:
        if isinstance(lp, int):
            return lp
        if id(lp) in _LEARNING_PHASE_CACHE:
            return _LEARNING_PHASE_CACHE[id(lp)]
        with name_scope(''):
            int_lp = tf.cast(lp, 'int32', name='learning_phase')
        _LEARNING_PHASE_CACHE[id(lp)] = int_lp
        return int_lp


@symbolic
def set_learning_phase(value):
    """Sets the learning phase to a fixed value.

    # Arguments
        value: Learning phase value, either 0 or 1 (integers).

    # Raises
        ValueError: if `value` is neither `0` nor `1`.
    """
    tf_keras_backend.set_learning_phase(value)


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

    # Raises
        RuntimeError: if no session is available
            (e.g. when using TensorFlow 2.0).
    """
    if not _is_tf_1():
        raise RuntimeError(
            '`get_session` is not available '
            'when using TensorFlow 2.0.')
    if tf.executing_eagerly():
        raise RuntimeError(
            '`get_session` is not available when '
            'TensorFlow is executing eagerly.')
    return tf_keras_backend.get_session()


def set_session(session):
    """Sets the global TensorFlow session.

    # Arguments
        session: A TF Session.

    # Raises
        RuntimeError: if no session is available
            (e.g. when using TensorFlow 2.0).
    """
    if not _is_tf_1():
        raise RuntimeError(
            '`set_session` is not available '
            'when using TensorFlow 2.0.')
    if tf.executing_eagerly():
        raise RuntimeError(
            '`set_session` is not available when '
            'TensorFlow is executing eagerly.')
    tf_keras_backend.set_session(session)


def clear_session():
    """Destroys the current Keras graph and creates a new one.

    Useful to avoid clutter from old models / layers.
    """
    tf_keras_backend.clear_session()
    global _LEARNING_PHASE_CACHE
    _LEARNING_PHASE_CACHE = {}


def v1_variable_initialization():
    session = get_session()
    with session.graph.as_default():
        variables = tf.global_variables()
        candidate_vars = []
        for v in variables:
            if not getattr(v, '_keras_initialized', False):
                candidate_vars.append(v)
        if candidate_vars:
            # This step is expensive, so we only run it on variables
            # not already marked as initialized.
            is_initialized = session.run(
                [tf.is_variable_initialized(v) for v in candidate_vars])
            uninitialized_vars = []
            for flag, v in zip(is_initialized, candidate_vars):
                if not flag:
                    uninitialized_vars.append(v)
                v._keras_initialized = True
            if uninitialized_vars:
                session.run(tf.variables_initializer(uninitialized_vars))


# DEVICE MANIPULATION AND PROBING

class _TfDeviceCaptureOp(object):
    """Class for capturing the TF device scope."""

    def __init__(self):
        # NOTE(robieta): This differs from tf.keras in that self.device is a
        # DeviceSpec rather than a string. This is done for compatibility
        # with a range of TensorFlow versions.
        self.device = None

    def _set_device(self, device):
        """This method captures TF's explicit device scope setting."""
        self.device = device

    def _set_device_from_string(self, device_str):
        self.device = tfdev.DeviceSpec.from_string(device_str)


def _get_current_tf_device():
    """Return explicit device of current context, otherwise returns `None`.

    # Returns
        If the current device scope is explicitly set, it returns a string with
        the device (`CPU` or `GPU`). If the scope is not explicitly set, it will
        return `None`.
    """
    g = get_graph()
    op = _TfDeviceCaptureOp()
    g._apply_device_functions(op)
    return op.device


def _is_current_explicit_device(device_type):
    """Check if the current device is explicitly set on the device type specified.

    # Arguments
        device_type: A string containing `GPU` or `CPU` (case-insensitive).

    # Returns
        A boolean indicating if the current device
        scope is explicitly set on the device type.

    # Raises
        ValueError: If the `device_type` string indicates an unsupported device.
    """
    device_type = device_type.lower()
    if device_type not in ['cpu', 'gpu']:
        raise ValueError('`device_type` should be either "cpu" or "gpu".')
    device = _get_current_tf_device()
    return (device is not None and device.device_type.lower() == device_type)


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        if _is_tf_1():
            devices = get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            _LOCAL_DEVICES = tf.config.experimental_list_devices()
    return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]


def _has_nchw_support():
    """Check whether the current scope supports NCHW ops.

    TensorFlow does not support NCHW on CPU.
    Therefore we check if we are not explicitly put on
    CPU, and have GPUs available.
    In this case there will be soft-placing on the GPU device.

    # Returns
        bool: if the current scope device placement would support nchw
    """
    explicitly_on_cpu = _is_current_explicit_device('cpu')
    gpus_available = len(_get_available_gpus()) > 0
    return (not explicitly_on_cpu and gpus_available)


# VARIABLE MANIPULATION

@symbolic
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


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


@symbolic
def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor and returns it.

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
        return tf.sparse.to_dense(tensor)
    else:
        return tensor


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

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    v = tf_keras_backend.variable(
        value, dtype=dtype, name=name, constraint=constraint)
    if hasattr(value, 'tocoo'):
        v._keras_shape = value.tocoo().shape
    elif isinstance(value, np.ndarray):
        v._keras_shape = value.shape
    elif hasattr(value, 'shape'):
        v._keras_shape = int_shape(value)
    v._uses_learning_phase = False
    return v


def is_variable(x):
    return isinstance(x, tf.Variable)


def constant(value, dtype=None, shape=None, name=None):
    """Creates a constant tensor.

    # Arguments
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    # Returns
        A Constant Tensor.
    """
    with tf_ops.init_scope():
        return tf_keras_backend.constant(
            value, dtype=dtype, shape=shape, name=name)


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
    return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)


@symbolic
def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
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
    x = tf_keras_backend.placeholder(
        shape=shape, ndim=ndim, dtype=dtype, sparse=sparse, name=name)
    if shape is None:
        if ndim is not None:
            shape = tuple(None for _ in range(ndim))
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


@symbolic
def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    """
    try:
        return x.op.type == 'Placeholder'
    except AttributeError:
        return False


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
    return tf.shape(x)


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

    {{np_implementation}}
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        if isinstance(x.shape, tuple):
            return x.shape
        return tuple(x.shape.as_list())
    except ValueError:
        return None


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

    {{np_implementation}}
    """
    return x.shape.rank


def size(x, name=None):
    """Returns the size of a tensor.

    # Arguments
        x: Tensor or variable.
        name: A name for the operation (optional).

    # Returns
        Size of the tensor.

    # Examples
    ```python
    >>> from keras import backend as K
    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = K.variable(value=val)
    >>> K.size(inputs)
    <tf.Tensor: id=9, shape=(), dtype=int32, numpy=4>
    ```

    """
    if is_symbolic(x):
        with get_graph().as_default():
            return tf.size(x)
    return tf.size(x, name=name)


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
    {{np_implementation}}
    """
    return x.dtype.base_dtype.name


def eval(x):
    """Evaluates the value of a tensor.

    # Arguments
        x: A tensor.

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
    {{np_implementation}}
    """
    if _is_tf_1():
        return to_dense(x).eval(session=get_session())
    if hasattr(x, 'numpy'):
        with context.eager_mode():
            return x.numpy()
    eval_fn = function([], [x])
    return eval_fn([])[0]


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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    with tf_ops.init_scope():
        v = tf.zeros(shape=shape, dtype=dtype, name=name)
        if py_all(v.shape.as_list()):
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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    with tf_ops.init_scope():
        v = tf.ones(shape=shape, dtype=dtype, name=name)
        if py_all(v.shape.as_list()):
            return variable(v, dtype=dtype, name=name)
        return v


def eye(size, dtype=None, name=None):
    """Instantiate an identity matrix and returns it.

    # Arguments
        size: Tuple, number of rows and columns. If Integer, number of rows.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, an identity matrix.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.eval(K.eye(3))
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
        >>> K.eval(K.eye((2, 3)))
        array([[1., 0., 0.],
               [0., 1., 0.]], dtype=float32)
    ```
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    if isinstance(size, (list, tuple)):
        n, m = size
    else:
        n, m = size, size
    with tf_ops.init_scope():
        return tf.eye(n, m, dtype=dtype, name=name)


@symbolic
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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    return tf.zeros_like(x, dtype=dtype, name=name)


@symbolic
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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    return tf.ones_like(x, dtype=dtype, name=name)


@symbolic
def identity(x, name=None):
    """Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.
        name: String, name for the variable to create.

    # Returns
        A tensor of the same shape, type and content.
    """
    return tf.identity(x, name)


def random_uniform_variable(shape, low, high,
                            dtype=None,
                            name=None,
                            seed=None):
    """Instantiates a variable with values drawn from a uniform distribution.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output interval.
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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    with tf_ops.init_scope():
        value = tf.random_uniform_initializer(
            low, high, seed=seed)(shape, dtype=dtype)
        return variable(value, dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    """Instantiates a variable with values drawn from a normal distribution.

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
    {{np_implementation}}
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    with tf_ops.init_scope():
        value = tf.random_normal_initializer(
            mean, scale, seed=seed)(shape, dtype=dtype)
        return variable(value, dtype=dtype, name=name)


def count_params(x):
    """Returns the static number of elements in a Keras variable or tensor.

    # Arguments
        x: Keras variable or tensor.

    # Returns
        Integer, the number of elements in `x`, i.e., the product of the
        array's static dimensions.

    # Example
    ```python
        >>> kvar = K.zeros((2,3))
        >>> K.count_params(kvar)
        6
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    {{np_implementation}}
    """
    return np.prod(int_shape(x))


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
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return tf_state_ops.assign(x, new_x)


def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return tf_state_ops.assign_add(x, increment)


def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return tf_state_ops.assign_sub(x, decrement)


@symbolic
def moving_average_update(x, value, momentum):
    """Compute the moving average of a variable.

    # Arguments
        x: A `Variable`.
        value: A tensor with the same shape as `x`.
        momentum: The moving average momentum.

    # Returns
        An operation to update the variable.
    """
    with tf_ops.colocate_with(x):
        decay = tf_ops.convert_to_tensor(1.0 - momentum)
        if decay.dtype != x.dtype.base_dtype:
            decay = tf_math_ops.cast(decay, x.dtype.base_dtype)
        update_delta = (x - tf_math_ops.cast(value, x.dtype)) * decay
        return tf_state_ops.assign_sub(x, update_delta)


# LINEAR ALGEBRA

def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

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
    {{np_implementation}}
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
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
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batches, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: int or tuple(int, int). Target dimensions to be reduced.

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

        Pseudocode:
        ```
        inner_products = []
        for xi, yi in zip(x, y):
            inner_products.append(xi.dot(yi))
        result = stack(inner_products)
        ```

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
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=(1, 2))
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```

    {{np_implementation}}
    """
    x_shape = int_shape(x)
    y_shape = int_shape(y)

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim < 2 or y_ndim < 2:
        raise ValueError('Can not do batch_dot on inputs '
                         'with rank < 2. '
                         'Received inputs with shapes ' +
                         str(x_shape) + ' and ' +
                         str(y_shape) + '.')

    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError('Can not do batch_dot on inputs '
                             'with different batch sizes. '
                             'Received inputs with shapes ' +
                             str(x_shape) + ' and ' +
                             str(y_shape) + '.')

    if isinstance(axes, int):
        axes = [axes, axes]

    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]

    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))

    # if tuple, convert to list.
    axes = list(axes)

    # convert negative indices.
    if axes[0] < 0:
        axes[0] += x_ndim
    if axes[1] < 0:
        axes[1] += y_ndim

    # sanity checks
    if 0 in axes:
        raise ValueError('Can not perform batch_dot over axis 0.'
                         'If your inputs are not batched,'
                         ' add a dummy batch dimension to your '
                         'inputs using K.expand_dims(x, 0)')

    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]

    if d1 is not None and d2 is not None and d1 != d2:
        raise ValueError('Can not do batch_dot on inputs with shapes ' +
                         str(x_shape) + ' and ' + str(y_shape) +
                         ' with axes=' + str(axes) + '. x.shape[%d] != '
                         'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))

    # backup ndims. Need them later.
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    # if rank is 2, expand to 3.
    if x_ndim == 2:
        x = tf.expand_dims(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = tf.expand_dims(y, 2)
        y_ndim += 1

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = tf.transpose(x, pattern)

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y = tf.transpose(y, pattern)

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape = shape(x)
        x_mid_dims = x_shape[1:-1]
        x_squashed_dim = tf.reduce_prod(x_mid_dims)
        x_squashed_shape = tf.stack([x_shape[0], x_squashed_dim, x_shape[-1]])
        x = tf.reshape(x, x_squashed_shape)
        x_squashed = True
    else:
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape = shape(y)
        y_trail_dims = y_shape[2:]
        y_squashed_dim = tf.reduce_prod(y_trail_dims)
        y_squashed_shape = tf.stack([y_shape[0], y_shape[1], y_squashed_dim])
        y = tf.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False

    result = tf.matmul(x, y)

    # if inputs were squashed, we have to reshape the matmul output.
    output_shape = tf.shape(result)
    do_reshape = False

    if x_squashed:
        output_shape = tf.concat([output_shape[:1],
                                  x_mid_dims,
                                  output_shape[-1:]], 0)
        do_reshape = True

    if y_squashed:
        output_shape = tf.concat([output_shape[:-1], y_trail_dims], 0)
        do_reshape = True

    if do_reshape:
        result = tf.reshape(result, output_shape)

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        result = tf.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = tf.squeeze(result, -1)

    return result


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
        >>> inputs = K.placeholder((2, 3))
        >>> inputs
        <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
        >>> input_transposed = K.transpose(inputs)
        >>> input_transposed
        <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

    ```
    {{np_implementation}}
    """
    return tf.transpose(x)


def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.

    {{np_implementation}}
    """
    return tf.nn.embedding_lookup(reference, indices)


# ELEMENT-WISE OPERATIONS


def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to find maximum values. If `None` (default), finds the
            maximum over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.

    {{np_implementation}}
    """
    return tf.reduce_max(x, axis, keepdims)


def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to find minimum values. If `None` (default), finds the
            minimum over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.

    {{np_implementation}}
    """
    return tf.reduce_min(x, axis, keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to sum over. If `None` (default), sums over all
            dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.

    {{np_implementation}}
    """
    return tf.reduce_sum(x, axis, keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the product. If `None` (default), computes
            the product over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.

    {{np_implementation}}
    """
    return tf.reduce_prod(x, axis, keepdims)


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    {{np_implementation}}
    """
    return tf_math_ops.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    {{np_implementation}}
    """
    return tf_math_ops.cumprod(x, axis=axis)


def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the variance. If `None` (default), computes
            the variance over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    {{np_implementation}}
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    m = tf.reduce_mean(x, axis, True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          axis,
                          keepdims)


def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the standard deviation. If `None` (default),
            computes the standard deviation over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    {{np_implementation}}
    """
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the mean. If `None` (default), computes
            the mean over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    {{np_implementation}}
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, axis, keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logical or. If `None` (default), computes
            the logical or over all dimensions.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    {{np_implementation}}
    """
    x = tf.cast(x, tf.bool)
    return tf.reduce_any(x, axis, keepdims)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logical and. If `None` (default), computes
            the logical and over all dimensions.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    {{np_implementation}}
    """
    x = tf.cast(x, tf.bool)
    return tf.reduce_all(x, axis, keepdims)


def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    {{np_implementation}}
    """
    return tf.argmax(x, axis)


def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    {{np_implementation}}
    """
    return tf.argmin(x, axis)


def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.square(x)


def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.abs(x)


def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    {{np_implementation}}
    """
    zero = _to_tensor(0., x.dtype.base_dtype)
    inf = _to_tensor(np.inf, x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, inf)
    return tf.sqrt(x)


def exp(x):
    """Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.exp(x)


def log(x):
    """Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf_math_ops.log(x)


def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the logsumexp. If `None` (default), computes
            the logsumexp over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    {{np_implementation}}
    """
    return tf.reduce_logsumexp(x, axis, keepdims)


def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.round(x)


def sign(x):
    """Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.sign(x)


def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    {{np_implementation}}
    """
    return tf.pow(x, a)


def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float, integer or tensor.
        max_value: Python float, integer or tensor.

    # Returns
        A tensor.
    {{np_implementation}}
    """
    if (isinstance(min_value, (int, float)) and
            isinstance(max_value, (int, float))):
        if max_value < min_value:
            max_value = min_value
    if min_value is None:
        min_value = -np.inf
    if max_value is None:
        max_value = np.inf
    return tf.clip_by_value(x, min_value, max_value)


def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.equal(x, y)


def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.not_equal(x, y)


def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.greater(x, y)


def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.greater_equal(x, y)


def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.less(x, y)


def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    {{np_implementation}}
    """
    return tf.less_equal(x, y)


def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.maximum(x, y)


def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.minimum(x, y)


def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.sin(x)


def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.cos(x)


def _regular_normalize_batch_in_training(x, gamma, beta,
                                         reduction_axes, epsilon=1e-3):
    """Non-fused version of `normalize_batch_in_training`.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    mean, var = tf.nn.moments(x, reduction_axes,
                              None, None, False)
    normed = tf.nn.batch_normalization(x, mean, var,
                                       beta, gamma,
                                       epsilon)
    return normed, mean, var


def _broadcast_normalize_batch_in_training(x, gamma, beta,
                                           reduction_axes, epsilon=1e-3):
    """Non-fused, broadcast version of `normalize_batch_in_training`.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    mean, var = tf.nn.moments(x, reduction_axes,
                              None, None, False)
    target_shape = []
    for axis in range(ndim(x)):
        if axis in reduction_axes:
            target_shape.append(1)
        else:
            target_shape.append(tf.shape(x)[axis])
    target_shape = tf.stack(target_shape)

    broadcast_mean = tf.reshape(mean, target_shape)
    broadcast_var = tf.reshape(var, target_shape)
    if gamma is None:
        broadcast_gamma = None
    else:
        broadcast_gamma = tf.reshape(gamma, target_shape)
    if beta is None:
        broadcast_beta = None
    else:
        broadcast_beta = tf.reshape(beta, target_shape)

    normed = tf.nn.batch_normalization(
        x,
        broadcast_mean,
        broadcast_var,
        broadcast_beta,
        broadcast_gamma,
        epsilon)
    return normed, mean, var


def _fused_normalize_batch_in_training(x, gamma, beta, reduction_axes,
                                       epsilon=1e-3):
    """Fused version of `normalize_batch_in_training`.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    if list(reduction_axes) == [0, 1, 2]:
        normalization_axis = 3
        tf_data_format = 'NHWC'
    else:
        normalization_axis = 1
        tf_data_format = 'NCHW'

    if gamma is None:
        gamma = tf.constant(1.0,
                            dtype=x.dtype,
                            shape=[x.shape[normalization_axis]])
    if beta is None:
        beta = tf.constant(0.0,
                           dtype=x.dtype,
                           shape=[x.shape[normalization_axis]])

    if gamma.dtype != tf.float32:
        gamma = tf.cast(gamma, tf.float32)
    if beta.dtype != tf.float32:
        beta = tf.cast(beta, tf.float32)

    if _has_compat_v1:
        fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
    else:
        fused_batch_norm = tf.nn.fused_batch_norm
    return fused_batch_norm(
        x,
        gamma,
        beta,
        epsilon=epsilon,
        data_format=tf_data_format)


def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    if (ndim(x) == 4 and
            list(reduction_axes) in [[0, 1, 2], [0, 2, 3]] and
            _is_tf_1()):
        if not _has_nchw_support() and list(reduction_axes) == [0, 2, 3]:
            return _broadcast_normalize_batch_in_training(x, gamma, beta,
                                                          reduction_axes,
                                                          epsilon=epsilon)
        return _fused_normalize_batch_in_training(
            x, gamma, beta, reduction_axes,
            epsilon=epsilon)
    else:
        if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
            return _regular_normalize_batch_in_training(x, gamma, beta,
                                                        reduction_axes,
                                                        epsilon=epsilon)
        else:
            return _broadcast_normalize_batch_in_training(x, gamma, beta,
                                                          reduction_axes,
                                                          epsilon=epsilon)


def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    if ndim(x) == 4:
        # The CPU implementation of FusedBatchNorm only support NHWC
        if axis == 1 or axis == -3:
            tf_data_format = 'NCHW'
        elif axis == 3 or axis == -1:
            tf_data_format = 'NHWC'
        else:
            tf_data_format = None

        if ((tf_data_format == 'NHWC' or
                (tf_data_format == 'NCHW' and
                 _has_nchw_support())) and
                _is_tf_1()):
            # The mean / var / beta / gamma may be processed by broadcast
            # so it may have extra axes with 1,
            # it is not needed and should be removed
            if ndim(mean) > 1:
                mean = tf.reshape(mean, [-1])
            if ndim(var) > 1:
                var = tf.reshape(var, [-1])
            if beta is None:
                beta = zeros_like(mean)
            elif ndim(beta) > 1:
                beta = tf.reshape(beta, [-1])
            if gamma is None:
                gamma = ones_like(mean)
            elif ndim(gamma) > 1:
                gamma = tf.reshape(gamma, [-1])

            if gamma.dtype != tf.float32:
                gamma = tf.cast(gamma, tf.float32)
            if beta.dtype != tf.float32:
                beta = tf.cast(beta, tf.float32)
            if mean.dtype != tf.float32:
                mean = tf.cast(mean, tf.float32)
            if var.dtype != tf.float32:
                var = tf.cast(var, tf.float32)

            if _has_compat_v1:
                fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
            else:
                fused_batch_norm = tf.nn.fused_batch_norm

            y, _, _ = fused_batch_norm(
                x,
                gamma,
                beta,
                epsilon=epsilon,
                mean=mean,
                variance=var,
                data_format=tf_data_format,
                is_training=False
            )
            return y
    # default
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.

    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.

    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0
    if py_all([is_sparse(x) for x in tensors]):
        return tf.sparse.concat(axis, tensors)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)


def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    """
    return tf.reshape(x, shape)


def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    """
    return tf.transpose(x, perm=pattern)


def resize_images(x,
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.
        interpolation: A string, one of `nearest` or `bilinear`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        rows, cols = 2, 3
    else:
        rows, cols = 1, 2

    original_shape = int_shape(x)
    new_shape = tf.shape(x)[rows:cols + 1]
    new_shape *= tf.constant(np.array([height_factor, width_factor],
                             dtype='int32'))
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf_image_ops.resize_nearest_neighbor(x, new_shape)
    elif interpolation == 'bilinear':
        x = tf_image_ops.resize_bilinear(x, new_shape)
    else:
        raise ValueError('interpolation should be one '
                         'of "nearest" or "bilinear".')
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])

    if original_shape[rows] is None:
        new_height = None
    else:
        new_height = original_shape[rows] * height_factor

    if original_shape[cols] is None:
        new_width = None
    else:
        new_width = original_shape[cols] * width_factor

    output_shape = (None, new_height, new_width, None)
    x.set_shape(transpose_shape(output_shape, data_format,
                                spatial_axes=(1, 2)))
    return x


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    """Resizes the volume contained in a 5D tensor.

    # Arguments
        x: Tensor or variable to resize.
        depth_factor: Positive integer.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
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
        raise ValueError('Unknown data_format: ' + str(data_format))


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.

    # Returns
        A tensor.
    """
    x_shape = x.shape.as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)

    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.

    # Repeating
    auxiliary_axis = axis + 1
    x_shape = tf.shape(x)
    x_rep = tf.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.shape) + 1)
    reps[auxiliary_axis] = rep
    x_rep = tf.tile(x_rep, reps)

    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = tf.constant(reps, dtype='int32')
    x_shape = x_shape * reps
    x_rep = tf.reshape(x_rep, x_shape)

    # Fix shape representation
    x_shape = x.shape.as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)
    return x_rep


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    assert ndim(x) == 2
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument and "start" is 0.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.

    # Arguments
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.

    # Returns
        An integer tensor.

    """
    # Match the behavior of numpy and Theano by returning an empty sequence.
    if stop is None:
        try:
            if start < 0:
                start = 0
        except TypeError:
            # Handle case where start is a tensor
            start = tf.cond(start < 0,
                            true_fn=lambda: tf.constant(0, dtype=start.dtype),
                            false_fn=lambda: start)

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

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2, 3)))
        >>> kvar_tile = K.tile(K.eye(2), (2, 3))
        >>> K.eval(kvar_tile)
        array([[1., 0., 1., 0., 1., 0.],
               [0., 1., 0., 1., 0., 1.],
               [1., 0., 1., 0., 1., 0.],
               [0., 1., 0., 1., 0., 1.]], dtype=float32)
    ```
    {{np_implementation}}
    """
    if isinstance(n, int):
        n = (n,)
    elif isinstance(n, list):
        n = tuple(n)

    shape = int_shape(x)
    if not is_tensor(n):
        if len(n) < len(shape):  # Padding the axis
            n = tuple([1 for _ in range(len(shape) - len(n))]) + n
        elif len(n) != len(shape):
            raise NotImplementedError

    return tf.tile(x, n)


def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    return tf.reshape(x, [-1])


def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.

    In other words, it flattens each data samples of a batch.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    x = tf.reshape(
        x, tf.stack([-1, prod(shape(x)[1:])],
                    name='stack_' + str(np.random.randint(1e4))))
    return x


def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    return tf.expand_dims(x, axis)


def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    return tf.squeeze(x, [axis])


def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.

    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               [0, 0]]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2))
    return tf.pad(x, pattern)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """Pads 5D tensor with zeros along the depth, height, width dimensions.

    Pads these dimensions with respectively
    "padding[0]", "padding[1]" and "padding[2]" zeros left and right.

    For 'channels_last' data_format,
    the 2nd, 3rd and 4th dimension will be padded.
    For 'channels_first' data_format,
    the 3rd, 4th and 5th dimension will be padded.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 5D tensor.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.

    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [
        [0, 0],
        [padding[0][0], padding[0][1]],
        [padding[1][0], padding[1][1]],
        [padding[2][0], padding[2][1]],
        [0, 0]
    ]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2, 3))

    return tf.pad(x, pattern)


def stack(x, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.stack(x, axis=axis)


def one_hot(indices, num_classes):
    """Computes the one-hot representation of an integer tensor.

    # Arguments
        indices: nD integer tensor of shape
            `(batch_size, dim1, dim2, ... dim(n-1))`
        num_classes: Integer, number of classes to consider.

    # Returns
        (n + 1)D one hot representation of the input
        with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    """
    return tf.one_hot(indices, depth=num_classes, axis=-1)


def reverse(x, axes):
    """Reverses a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    if isinstance(axes, int):
        axes = [axes]
    return tf.reverse(x, axes)


def slice(x, start, size):
    """Extracts a slice from a tensor.

    # Arguments
        x: Input tensor.
        start: Integer list/tuple or tensor
            indicating the start indices of the slice
            along each axis.
        size: Integer list/tuple or tensor
            indicating how many dimensions to slice
            along each axis.

    # Returns
        A sliced tensor:
        ```python
        new_x = x[start[0]: start[0] + size[0], ..., start[-1]: start[-1] + size[-1]]
        ```

    # Raises
        ValueError: if the dimension and the size of indices mismatches.

    {{np_implementation}}
    """
    x_shape = int_shape(x)
    if (x_shape is not None) and (x_shape[0] is not None):
        len_start = int_shape(start)[0] if is_tensor(start) else len(start)
        len_size = int_shape(size)[0] if is_tensor(size) else len(size)
        if not (len(int_shape(x)) == len_start == len_size):
            raise ValueError('The dimension and the size of indices should match.')
    return tf.slice(x, start, size)


# VALUE MANIPULATION


def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    if _is_tf_1():
        return x.eval(session=get_session())
    else:
        return x.numpy()


def batch_get_value(ops):
    """Returns the value of more than one tensor variable.

    # Arguments
        ops: list of ops to run.

    # Returns
        A list of Numpy arrays.
    """
    return tf_keras_backend.batch_get_value(ops)


def set_value(x, value):
    """Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Variable to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    """
    tf_keras_backend.set_value(x, value)


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    tf_keras_backend.batch_set_value(tuples)


def get_variable_shape(x):
    """Returns the shape of a variable.

    # Arguments
        x: A variable.

    # Returns
        A tuple of integers.
    """
    return int_shape(x)


@symbolic
def print_tensor(x, message=''):
    """Prints `message` and the tensor value when evaluated.

    Note that `print_tensor` returns a new tensor identical to `x`
    which should be used in the following code. Otherwise the
    print operation is not taken into account during evaluation.

    # Example

    ```python
        >>> x = K.print_tensor(x, message="x is: ")
    ```

    # Arguments
        x: Tensor to print.
        message: Message to print jointly with the tensor.

    # Returns
        The same tensor `x`, unchanged.
    """
    op = tf.print(message, x, output_stream=sys.stdout)
    with tf.control_dependencies([op]):
        return tf.identity(x)


# GRAPH MANIPULATION


def function(inputs, outputs, updates=None, **kwargs):
    if _is_tf_1():
        v1_variable_initialization()
    return tf_keras_backend.function(inputs, outputs,
                                     updates=updates,
                                     **kwargs)


@symbolic
def gradients(loss, variables):
    """Returns the gradients of `loss` w.r.t. `variables`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    """
    if _is_tf_1():
        return tf.gradients(loss, variables, colocate_gradients_with_ops=True)
    return tf.gradients(loss, variables)


@symbolic
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
        return map(tf.stop_gradient, variables)
    else:
        return tf.stop_gradient(variables)


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

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

        last_output: The latest output of the rnn, of shape `(samples, ...)`
        outputs: Tensor with shape `(samples, time, ...)` where each
            entry `outputs[s, t]` is the output of the step function
            at time `t` for sample `s`.
        new_states: List of tensors, latest states returned by
            the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: If input dimension is less than 3.
        ValueError: If `unroll` is `True`
            but input timestep is not a fixed number.
        ValueError: If `mask` is provided (not `None`)
            but states is not provided (`len(states)` == 0).

    {{np_implementation}}
    """
    last_output, outputs, new_states = tf_keras_backend.rnn(
        step_function, inputs, initial_states,
        go_backwards=go_backwards,
        mask=mask,
        constants=constants,
        unroll=unroll,
        input_length=input_length)
    reachable = tf_utils.get_reachable_from_inputs([learning_phase()],
                                                   targets=[last_output])
    if last_output in reachable:
        last_output._uses_learning_phase = True
    return last_output, outputs, new_states


@symbolic
def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value.

    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.

    # Raises
        ValueError: If rank of `condition` is greater than rank of expressions.

    {{np_implementation}}
    """
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, 'bool')
    cond_ndim = ndim(condition)
    if not cond_ndim:
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
        x = tf.cond(condition,
                    then_expression_fn,
                    else_expression_fn)
    else:
        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        expr_ndim = ndim(then_expression)
        if cond_ndim > expr_ndim:
            raise ValueError('Rank of `condition` should be less than or'
                             ' equal to rank of `then_expression` and '
                             '`else_expression`. ndim(condition)=' +
                             str(cond_ndim) + ', ndim(then_expression)'
                             '=' + str(expr_ndim))
        if cond_ndim > 1:
            ndim_diff = expr_ndim - cond_ndim
            cond_shape = tf.concat([tf.shape(condition), [1] * ndim_diff], axis=0)
            condition = tf.reshape(condition, cond_shape)
            expr_shape = tf.shape(then_expression)
            shape_diff = expr_shape - cond_shape
            zero_expr_shape = tf.ones_like(expr_shape)
            tile_shape = tf.where(shape_diff > 0, expr_shape, zero_expr_shape)
            condition = tf.tile(condition, tile_shape)
        x = tf.where(condition, then_expression, else_expression)
    return x


@symbolic
def in_train_phase(x, alt, training=None):
    """Selects `x` in train phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in train phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

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

    # else: assume learning phase is a placeholder tensor.
    x = switch(training, x, alt)
    if uses_learning_phase:
        x._uses_learning_phase = True
    return x


@symbolic
def in_test_phase(x, alt, training=None):
    """Selects `x` in test phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in test phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    """
    return in_train_phase(alt, x, training=training)


# NN OPERATIONS

def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified linear unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    # Returns
        A tensor.

    {{np_implementation}}
    """

    if alpha != 0.:
        if max_value is None and threshold == 0.:
            return tf.nn.leaky_relu(x, alpha=alpha)

        if threshold != 0.:
            negative_part = tf.nn.relu(-x + threshold)
        else:
            negative_part = tf.nn.relu(-x)

    clip_max = max_value is not None

    if threshold != 0:
        # computes x for x > threshold else 0
        x = x * tf.cast(tf.greater(x, threshold), floatx())
    elif max_value == 6:
        # if no threshold, then can use nn.relu6 native TF op for performance
        x = tf.nn.relu6(x)
        clip_max = False
    else:
        x = tf.nn.relu(x)

    if clip_max:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)

    if alpha != 0:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tensor or variable to compute the activation function for.
        alpha: A scalar, slope of negative section.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)


def softmax(x, axis=-1):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.
        axis: The dimension softmax would be performed on.
            The default is -1 which indicates the last dimension.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.nn.softmax(x, axis=axis)


def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.nn.softplus(x)


def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.nn.softsign(x)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    return tf_keras_backend.categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    return tf_keras_backend.sparse_categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    return tf_keras_backend.binary_crossentropy(
        target, output, from_logits=from_logits)


def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
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

    {{np_implementation}}
    """
    return tf_keras_backend.hard_sigmoid(x)


def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.

    {{np_implementation}}
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
    {{np_implementation}}
    """
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.nn.dropout(x, rate=level, noise_shape=noise_shape, seed=seed)


def l2_normalize(x, axis=None):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.

    {{np_implementation}}
    """
    return tf.nn.l2_normalize(x, axis=axis)


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
    # Note that the order of the 2 first positional arguments
    # has been inverted in TF 2.
    return tf.nn.in_top_k(predictions=predictions,
                          targets=targets,
                          k=k)


# CONVOLUTIONS


def _preprocess_conv1d_input(x, data_format):
    """Transpose and cast the input before the conv1d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NWC'  # to pass TF Conv2dNative operations
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 1))  # NCW -> NWC
        else:
            tf_data_format = 'NCW'
    return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
        force_transpose: boolean, whether force to transpose input from NCHW to NHWC
                        if the `data_format` is `"channels_first"`.

    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support() or force_transpose:
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = 'NCHW'
    return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
    """Transpose and cast the input before the conv3d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NDHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 3, 4, 1))
        else:
            tf_data_format = 'NCDHW'
    return x, tf_data_format


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.

    # Arguments
        padding: string, `"same"` or `"valid"`.

    # Returns
        a string, `"SAME"` or `"VALID"`.

    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding


def conv1d(x, kernel, strides=1, padding='valid',
           data_format=None, dilation_rate=1):
    """1D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilate rate.

    # Returns
        A tensor, result of 1D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    kernel_shape = kernel.shape.as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    padding = _preprocess_padding(padding)
    x, tf_data_format = _preprocess_conv1d_input(x, data_format)

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['dilation_rate'] = (dilation_rate,)
    else:
        kwargs['dilations'] = (dilation_rate,)

    x = tf.nn.convolution(
        x, kernel,
        strides=(strides,),
        padding=padding,
        data_format=tf_data_format,
        **kwargs)

    if data_format == 'channels_first' and tf_data_format == 'NWC':
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW
    return x


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)

    padding = _preprocess_padding(padding)

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['dilation_rate'] = dilation_rate
    else:
        kwargs['dilations'] = dilation_rate

    x = tf.nn.convolution(
        x, kernel,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
        **kwargs)
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of transposed 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    # tf.nn.atrous_conv2d_transpose input only supports NHWC format
    if data_format == 'channels_first' and dilation_rate != (1, 1):
        force_transpose = True
    else:
        force_transpose = False

    x, tf_data_format = _preprocess_conv2d_input(x, data_format, force_transpose)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        output_shape = (output_shape[0],
                        output_shape[2],
                        output_shape[3],
                        output_shape[1])
    if output_shape[0] is None:
        output_shape = (shape(x)[0],) + tuple(output_shape[1:])

    output_shape = tf.stack(list(output_shape))

    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    if dilation_rate == (1, 1):
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides,
                                   padding=padding,
                                   data_format=tf_data_format)
    else:
        assert dilation_rate[0] == dilation_rate[1]
        x = tf.nn.atrous_conv2d_transpose(
            x, kernel, output_shape, dilation_rate[0], padding)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1,
                     padding='valid', data_format=None, dilation_rate=1):
    """1D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: stride integer.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilation rate.

    # Returns
        Output tensor.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,)

    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    if tf_data_format == 'NWC':
        tf_data_format = 'NHWC'
    else:
        tf_data_format = 'NCHW'
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        spatial_start_dim = 1
        strides = (1,) + strides * 2 + (1,)
    else:
        spatial_start_dim = 2
        strides = (1, 1) + strides * 2
    x = tf.expand_dims(x, spatial_start_dim)
    depthwise_kernel = tf.expand_dims(depthwise_kernel, 0)
    pointwise_kernel = tf.expand_dims(pointwise_kernel, 0)
    dilation_rate = (1,) + dilation_rate

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['rate'] = dilation_rate
    else:
        kwargs['dilations'] = dilation_rate

    x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                               strides=strides,
                               padding=padding,
                               data_format=tf_data_format,
                               **kwargs)

    x = tf.squeeze(x, [spatial_start_dim])

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW

    return x


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
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['rate'] = dilation_rate
    else:
        kwargs['dilations'] = dilation_rate

    x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                               strides=strides,
                               padding=padding,
                               data_format=tf_data_format,
                               **kwargs)
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


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
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['rate'] = dilation_rate
    else:
        kwargs['dilations'] = dilation_rate

    x = tf.nn.depthwise_conv2d(x, depthwise_kernel,
                               strides=strides,
                               padding=padding,
                               data_format=tf_data_format,
                               **kwargs)
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1, 1)):
    """3D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.

    # Returns
        A tensor, result of 3D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)

    # TF 2 arg conversion
    kwargs = {}
    if _is_tf_1():
        kwargs['dilation_rate'] = dilation_rate
    else:
        kwargs['dilations'] = dilation_rate

    x = tf.nn.convolution(
        x, kernel,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
        **kwargs)
    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


def conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1),
                     padding='valid', data_format=None):
    """3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    if isinstance(output_shape, (tuple, list)):
        output_shape = tf.stack(output_shape)

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)

    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        output_shape = (output_shape[0],
                        output_shape[2],
                        output_shape[3],
                        output_shape[4],
                        output_shape[1])
    if output_shape[0] is None:
        output_shape = (tf.shape(x)[0],) + tuple(output_shape[1:])
        output_shape = tf.stack(list(output_shape))

    padding = _preprocess_padding(padding)
    if tf_data_format == 'NDHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding,
                               data_format=tf_data_format)
    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


def pool2d(x, pool_size, strides=(1, 1),
           padding='valid', data_format=None,
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == 'max':
        x = tf.nn.max_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid',
           data_format=None, pool_mode='max'):
    """3D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 3D pooling.

    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NDHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == 'max':
        x = tf.nn.max_pool3d(x, pool_size, strides,
                             padding=padding,
                             data_format=tf_data_format)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool3d(x, pool_size, strides,
                             padding=padding,
                             data_format=tf_data_format)
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


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
        the tensor after 1d conv with un-shared weights,
        with shape (batch_size, output_length, filters)

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


def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format=None):
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


def bias_add(x, bias, data_format=None):
    """Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    {{np_implementation}}
    """
    data_format = normalize_data_format(data_format)
    bias_shape = int_shape(bias)
    if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
        raise ValueError('Unexpected bias dimensions %d, '
                         'expect to be 1 or %d dimensions'
                         % (len(bias_shape), ndim(x)))
    if ndim(x) == 5:
        if len(bias_shape) == 1:
            new_shape = (1, 1, 1, 1, bias_shape[0])
        else:
            new_shape = (1,) + bias_shape
        new_shape = transpose_shape(new_shape, data_format,
                                    spatial_axes=(1, 2, 3))
        x = x + reshape(bias, new_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                if _has_nchw_support():
                    x = tf.nn.bias_add(x, bias,
                                       data_format='NCHW')
                else:
                    x = x + reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x = x + reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x = tf.nn.bias_add(x, bias,
                                   data_format='NHWC')
            else:
                x = x + reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if len(bias_shape) == 1:
            new_shape = (1, 1, bias_shape[0])
        else:
            new_shape = (1,) + bias_shape
        new_shape = transpose_shape(new_shape, data_format,
                                    spatial_axes=(1,))
        x = x + reshape(bias, new_shape)
    else:
        x = tf.nn.bias_add(x, bias)
    return x


# RANDOMNESS


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Returns a tensor with normal distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: A float, mean of the normal distribution to draw samples.
        stddev: A float, standard deviation of the normal distribution
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
    if py_any(list(is_symbolic(x) for x in (shape, mean, stddev))):
        with get_graph().as_default():
            return tf_keras_backend.random_normal(
                shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    with tf_ops.init_scope():
        return tf_keras_backend.random_normal(
            shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
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
    if py_any(list(is_symbolic(x) for x in (shape, minval, maxval))):
        with get_graph().as_default():
            return tf_keras_backend.random_uniform(
                shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
    with tf_ops.init_scope():
        return tf_keras_backend.random_uniform(
            shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    if py_any(list(is_symbolic(x) for x in (shape, p))):
        with get_graph().as_default():
            return tf_keras_backend.random_binomial(
                shape, p=p, dtype=dtype, seed=seed)
    with tf_ops.init_scope():
        return tf_keras_backend.random_binomial(
            shape, p=p, dtype=dtype, seed=seed)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Returns a tensor with truncated random normal distribution of values.

    The generated values follow a normal distribution
    with specified mean and standard deviation,
    except that values whose magnitude is more than
    two standard deviations from the mean are dropped and re-picked.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: Mean of the values.
        stddev: Standard deviation of the values.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    if py_any(list(is_symbolic(x) for x in (shape, mean, stddev))):
        with get_graph().as_default():
            return tf_keras_backend.truncated_normal(
                shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    with tf_ops.init_scope():
        return tf_keras_backend.truncated_normal(
            shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


# CTC
# TensorFlow has a native implementation, but it uses sparse tensors
# and therefore requires a wrapper for Keras. The functions below convert
# dense to sparse tensors and also wraps up the beam search code that is
# in TensorFlow's CTC implementation


def ctc_label_dense_to_sparse(labels, label_lengths):
    """Converts CTC labels from dense to sparse.

    # Arguments
        labels: dense CTC labels.
        label_lengths: length of the labels.

    # Returns
        A sparse tensor representation of the labels.
    """
    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(_, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < tf.fill(
            max_num_labels_tns, current_input)

    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    dense_mask = functional_ops.scan(range_less_than, label_lengths,
                                     initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(label_shape[1]), num_batches_tns),
                             label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    tmp = tf.tile(tf.range(label_shape[0]), max_num_labels_tns)
    batch_array = tf.transpose(tf.reshape(tmp, reverse(label_shape, 0)))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)

    indices = concatenate([batch_ind, label_ind], axis=0)
    indices = tf.transpose(tf.reshape(indices, [2, -1]))

    vals_sparse = tf.gather_nd(labels, indices)

    indices = tf.cast(indices, tf.int64)
    label_shape = tf.cast(label_shape, tf.int64)
    return tf.SparseTensor(indices, vals_sparse, label_shape)


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    y_pred = tf_math_ops.log(tf.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length), 1)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1, merge_repeated=False):
    """Decodes the output of a softmax.

    Can use either greedy search (also known as best path)
    or a constrained dictionary search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `True`.
            This does not use a dictionary.
        beam_width: if `greedy` is `False`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `False`,
            how many of the most probable paths will be returned.
        merge_repeated: if `greedy` is `False`,
            merge repeated classes in the output beams.

    # Returns
        Tuple:
            List: if `greedy` is `True`, returns a list of one element that
                contains the decoded sequence.
                If `False`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    y_pred = tf_math_ops.log(tf.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        (decoded, log_prob) = ctc.ctc_greedy_decoder(
            inputs=y_pred,
            sequence_length=input_length)
    else:
        (decoded, log_prob) = ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length, beam_width=beam_width,
            top_paths=top_paths, merge_repeated=merge_repeated)

    decoded_dense = []
    for st in decoded:
        dense_tensor = tf.sparse.to_dense(st, default_value=-1)
        decoded_dense.append(dense_tensor)
    return decoded_dense, log_prob


def control_dependencies(control_inputs):
    """A context manager that specifies control dependencies.

    # Arguments
        control_inputs: A list of Operation or Tensor objects
            which must be executed
            or computed before running the operations defined in the context.
            Can also be None to clear the control dependencies.

    # Returns
        A context manager.
    """
    return tf.control_dependencies(control_inputs)


# HIGH ORDER FUNCTIONS

def map_fn(fn, elems, name=None, dtype=None):
    """Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph
        dtype: Output data type.

    # Returns
        Tensor with dtype `dtype`.
    """
    return tf.map_fn(fn, elems, name=name, dtype=dtype)


def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Tensor with same type and shape as `initializer`.
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
        Tensor with same type and shape as `initializer`.
    """
    return tf.foldr(fn, elems, initializer=initializer, name=name)
