from __future__ import print_function

import mxnet as mx
import numpy as np
from numbers import Number
from functools import wraps
from collections import defaultdict


from .common import floatx, epsilon, set_epsilon, set_floatx, set_image_data_format, image_data_format


_UID_PREFIXES = defaultdict(int)
_LEARNING_PHASE = 1  # The learning phase flag: 0 = test, 1 = train
_MODEL = None
_REENTRY = False
NAME_SCOPE_STACK = []

set_image_data_format('channels_first')


class name_scope(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global NAME_SCOPE_STACK
        NAME_SCOPE_STACK.append(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        NAME_SCOPE_STACK.pop()


def _prepare_name(name, default):
    prefix = '/'.join(NAME_SCOPE_STACK)
    if name is None:
        name = prefix + '/' + default
    else:
        name = prefix + '/' + name
    name += "%d" % get_uid(name)
    return name


def is_reentry():
    return _REENTRY


def set_reentry(value):
    global _REENTRY
    assert type(value) == bool, "Please set to a boolean value."
    _REENTRY = value


def set_model(model):
    global _MODEL
    _MODEL = model


def clear_session():
    global _MODEL
    reset_uids()
    _MODEL = None


def learning_phase():
    return _LEARNING_PHASE  # 0 = test, 1 = train


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


# VARIABLE MANIPULATION
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
    x = np.asarray(x, dtype=floatx())
    if x.shape:
        return x
    else:
        return x.tolist()


def is_sparse(tensor):
    """
    MXNet backend do not yet support sparse tensor operations.
    """
    return False


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
    raise NotImplementedError("MXNet Backend: Sparse operations are not supported.")


def keras_symbol_child(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        global _REENTRY
        reset = False
        try:
            if _REENTRY:
                train_ret = func(*args, **kwargs)
                test_ret = train_ret
            else:
                _REENTRY = True
                reset = True
                old = learning_phase()
                set_learning_phase(1)
                train_ret = func(*args, **kwargs)
                set_learning_phase(0)
                test_ret = func(*args, **kwargs)
                set_learning_phase(old)
                assert type(train_ret) == type(test_ret)

            train_rets = []
            test_rets = []
            if isinstance(train_ret, tuple):
                train_rets = list(train_ret)
                test_rets = list(test_ret)
            if isinstance(train_ret, KerasSymbol):
                train_rets = [train_ret]
                test_rets = [test_ret]
            assert len(train_rets) == len(test_rets)
            for train_r, test_r in zip(train_rets, test_rets):
                assert type(train_r) == type(test_r)
                if isinstance(train_r, KerasSymbol):
                    train_r = [train_r]
                    test_r = [test_r]
                for train_i, test_i in zip(train_r, test_r):
                    if isinstance(train_i, KerasSymbol):
                        for arg in list(args) + list(kwargs.values()) + list(test_i.get_neighbor()):
                            train_i.add_neighbor(arg)
                            if isinstance(arg, (list, tuple)):
                                for t in arg:
                                    train_i.add_neighbor(t)
                        if reset:
                            assert isinstance(train_i._train_sym, mx.sym.Symbol)
                            assert isinstance(test_i._pred_sym, mx.sym.Symbol)
                            assert train_i._name == test_i._name
                            train_i._pred_sym = test_i._pred_sym
                            assert train_i._train_sym is not None and train_i._pred_sym is not None
                    else:
                        assert (train_i == test_i) is True

            if reset:
                _REENTRY = False
            return train_ret
        finally:
            if reset:
                _REENTRY = False
    return func_wrapper


class KerasSymbol(object):
    """Wraps on top of MXNet symbol that helps generate multiple static computation graph and binding values.
    """
    def __init__(self, mx_symbol, neighbors=None, is_var=False):
        if not isinstance(mx_symbol, mx.sym.Symbol):
            raise TypeError
        if is_var:
            self._train_sym = mx_symbol
            self._pred_sym = mx_symbol
        else:
            self._train_sym = mx_symbol if learning_phase() else None
            self._pred_sym = None if learning_phase() else mx_symbol
        self._name = None
        self._neighbors = []
        if neighbors:
            for node in neighbors:
                self.add_neighbor(node)
        self._bind_values = {}  # Map for storing op.name : op.tensor
        self.tensor = None  # This will be MXNet NDArray

    def bind(self, data):
        if not hasattr(self, 'tensor'):
            self.tensor[:] = data
        else:
            self.tensor = data
        if self.name in self._bind_values:
            assert self._bind_values[self.name].shape == data.shape, \
                "Redefinition of variable %s" % self.name
            assert self._bind_values[self.name].dtype == data.dtype, \
                "Redefinition of variable %s" % self.name
            if _MODEL is not None and self.name in _MODEL._args:
                _MODEL._set_weights({self.name: data}, {})
            if _MODEL is not None and self.name in _MODEL._auxs:
                _MODEL._set_weights({}, {self.name: data})
            else:
                self._bind_values[self.name][:] = data
        else:
            self._bind_values[self.name] = data

    def add_neighbor(self, x):
        if not isinstance(x, KerasSymbol):
            return
        if x not in self._neighbors:
            self._neighbors.append(x)
            x.add_neighbor(self)

    def get_neighbor(self):
        return self._neighbors

    def get_bind_values(self):
        return self._bind_values

    @property
    def symbol(self):
        sym = self._train_sym if learning_phase() else self._pred_sym
        assert sym is not None, "[Debug Info] %s, %s" % (self._train_sym, self._pred_sym)
        return sym

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self.symbol.name

    @property
    def dtype(self):
        return self._get_type()

    @property
    def shape(self):
        return self._get_shape()

    def eval(self):
        return self.tensor

    def _get_shape(self):
        if hasattr(self, '_keras_shape'):
            return self._keras_shape
        else:
            _, out_shape, _ = self.symbol.infer_shape_partial()
            return out_shape[0]

    def _get_type(self):
        _, out_type, _ = self.symbol.infer_type()
        t = out_type[0]
        return _convert_dtype_string(t)

    @keras_symbol_child
    def __getitem__(self, in_slice):
        begin = []
        end = []
        for i in in_slice:
            if isinstance(i, int):
                begin.append(i)
                end.append(i + 1)
            elif isinstance(i, slice):
                assert i.step is None or i.step == 1
                begin.append(i.start)
                end.append(i.stop)
            else:
                raise AttributeError
        return KerasSymbol(mx.sym.slice(self.symbol, begin=tuple(begin), end=tuple(end)), neighbors=[self])

    @keras_symbol_child
    def __abs__(self):
        return KerasSymbol(mx.sym.abs(self.symbol), neighbors=[self])

    @keras_symbol_child
    def __add__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_add(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(
                self.symbol + other)

    @keras_symbol_child
    def __radd__(self, other):
        return self.__add__(other)

    @keras_symbol_child
    def __sub__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_sub(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol - other)

    @keras_symbol_child
    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    @keras_symbol_child
    def __neg__(self):
        return KerasSymbol(self.symbol * (-1.0), neighbors=[self])

    @keras_symbol_child
    def __div__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __truediv__(self, other):
        return self.__div__(other)

    @keras_symbol_child
    def __itruediv__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __mul__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_mul(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol * other)

    @keras_symbol_child
    def __rmul__(self, other):
        return self.__mul__(other)

    @keras_symbol_child
    def __gt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(self.symbol > other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __ge__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(self.symbol >= other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater_equal(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __lt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol < other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_lesser(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __le__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol <= other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_lesser_equal(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __gt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol > other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_symbol_child
    def __pow__(self, power, modulo=None):
        return KerasSymbol(self.symbol.__pow__(power), neighbors=[self])

    def __repr__(self):
        return self.symbol.name + ':[tensor=' + str(hasattr(self, 'tensor')) + \
            ' dtype=' + self.dtype + ']'

    def __str__(self):
        return "Symbol: %s" % self.symbol.name


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
        >>> kvar.eval()
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if dtype is None:
        dtype = floatx()
    if isinstance(value, Number):
        value = np.array([value])
    if isinstance(value, KerasSymbol):
        value = eval(value)

    # MXNet backend do not support scalars
    if isinstance(value, np.ndarray) and len(value.shape) == 0:
        raise ValueError("MXNet backend: Do not support scalars. Provided value for variable - ", value)

    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'variable')
    ndarray = mx.nd.array(value, dtype=dtype)

    ret = _keras_variable(name, ndarray.shape, ndarray.dtype)
    ret.bind(ndarray)

    if isinstance(value, np.ndarray):
        ret._keras_shape = tuple([d if d != 0 else None for d in value.shape])
    elif hasattr(value, 'shape'):
        ret._keras_shape = tuple([d if d != 0 else None for d in map(int, value.shape)])
    return ret


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
    if shape is None:
        mx_ndarray = mx.nd.array(value, dtype=dtype)
    else:
        np_ndarray = np.ndarray(shape, dtype=dtype)
        np_ndarray.fill(value)
        mx_ndarray = mx.nd.array(np_ndarray)
    return mx_ndarray


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
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    """
    if not isinstance(x, KerasSymbol):
        raise ValueError('Unexpectedly found an instance of type `' +
                         str(type(x)) + '`.''Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


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
        placeholder1:[tensor=False dtype=float32]
    ```
    """
    if sparse:
        raise NotImplementedError("MXNet backend do not yet support sparse tensor operations.")

    if dtype is None:
        dtype = floatx()

    dtype = _convert_string_dtype(dtype)
    if shape is None and ndim is None:
        raise ValueError('Specify either a shape or ndim value.')
    name = _prepare_name(name, 'placeholder')
    if shape:
        shape = tuple([0 if x is None else x for x in shape])
    else:
        shape = tuple([0 for _ in range(ndim)])
    sym = _keras_variable(name, shape=shape, dtype=dtype)
    sym._keras_shape = tuple([d if d != 0 else None for d in shape])
    sym._mxnet_placeholder = True
    return sym


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2,4,5))
        >>> K.is_placeholder(input_ph)
        True
    ```
    """
    return hasattr(x, '_mxnet_placeholder') and x._mxnet_placeholder


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        (2,3)
        >>> K.shape(inputs)
        (2,4,5)
    ```
    """
    if isinstance(x, KerasSymbol):
        return x.shape
    else:
        return None


def int_shape(x):
    """Returns the shape tensor or variable as a tuple of int or None entries.

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
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        if type(x.shape) is tuple:
            return x.shape
        else:
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
    """
    shape = x.shape
    if shape is not None:
        return len(shape)
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
        'float32'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32'
    ```
    """
    return x.dtype


def eval(x):
    """Evaluates the value of a variable.

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
    if isinstance(x, KerasSymbol):
        if hasattr(x, 'tensor') and x.tensor is not None:
            if x.name in x.get_bind_values() and _MODEL is not None:
                _MODEL._sync_weights()
            ret = x.eval().asnumpy()
        else:
            bind_values = dfs_get_bind_values(x)
            executor = x.symbol.simple_bind(mx.cpu(), grad_req='null')
            for v in executor.arg_dict:
                bind_values[v].copyto(executor.arg_dict[v])
            outputs = executor.forward(is_train=learning_phase())
            ret = outputs[0].asnumpy()
        if ret.shape == (1,):
            return ret[0]
        return ret
    elif isinstance(x, mx.nd.NDArray):
        return x.asnumpy()
    else:
        return x


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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    value = mx.nd.zeros(shape, dtype=dtype)
    name = _prepare_name(name, 'zeroinit')
    kvar = _keras_variable(name, value.shape, value.dtype)
    kvar.bind(value)
    return kvar


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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    value = mx.nd.ones(shape, dtype=dtype)
    name = _prepare_name(name, 'oneinit')
    kvar = _keras_variable(name=name, shape=shape, dtype=dtype)
    kvar.bind(value)
    return kvar


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
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.nd.array(np.eye(size, dtype=dtype))
    name = _prepare_name(name, 'eyeinit')
    kvar = _keras_variable(name=name, shape=size, dtype=dtype)
    kvar.bind(value)
    return kvar


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
    if dtype is None:
        dtype = x.dtype
    else:
        dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'zeroslikeinit')
    mx_shape = tuple([0 if x is None else x for x in x.shape])
    mx_value = mx.nd.zeros(mx_shape, dtype=dtype)
    k_var = _keras_variable(name=name, dtype=dtype, shape=mx_shape)
    k_var.bind(mx_value)
    return k_var


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
    if dtype is None:
        dtype = x.dtype
    else:
        dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'onelikeinit')
    mx_shape = tuple([0 if x is None else x for x in x.shape])
    mx_value = mx.nd.ones(shape=mx_shape, dtype=dtype)
    k_var = _keras_variable(name=name, dtype=dtype, shape=x.shape)
    k_var.bind(mx_value)
    return k_var


def identity(x):
    """Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.

    # Returns
        A tensor of the same shape, type and content.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> kvar_copy = K.identity(kvar)
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
        >>> K.eval(kvar_copy)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    """
    name = _prepare_name(None, 'identityinit')
    dtype = x.dtype
    x_value = eval(x)
    mx_shape = tuple([0 if x is None else x for x in x.shape])
    mx_value = mx.nd.array(x_value, dtype=dtype)
    k_var = _keras_variable(name=name, dtype=dtype, shape=mx_shape)
    k_var.bind(mx_value)
    return k_var


# TODO: depreciated
def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
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
        >>> rand_var = K.random_uniform_variable(shape=(3,3), low=1, high=3)
        >>> K.eval(rand_var)
        array([[ 2.09762716,  2.18568921,  2.43037868],
               [ 2.6885314 ,  2.20552683,  2.71589136],
               [ 2.0897665 ,  2.69450331,  1.84730959]], dtype=float32)
        >>> rand_var1 = K.random_uniform_variable(shape=(3,3), low=1, high=3, seed=128)
        >>> rand_var2 = K.random_uniform_variable(shape=(3,3), low=1, high=3, seed=128)
        >>> K.eval(rand_var1)
        array([[ 1.07625926,  1.60461187,  1.30567968],
               [ 2.43757105,  2.77134657,  2.16382241],
               [ 1.7636764 ,  2.51851654,  1.96760654]], dtype=float32)
        >>> K.eval(rand_var2)
        array([[ 1.07625926,  1.60461187,  1.30567968],
               [ 2.43757105,  2.77134657,  2.16382241],
               [ 1.7636764 ,  2.51851654,  1.96760654]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, "randomuniform")
    if seed:
        mx.random.seed(seed)
    value = mx.random.uniform(low=low, high=high, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    kvar = _keras_variable(name=name, shape=shape, dtype=dtype)
    kvar.bind(value)
    return kvar


# TODO: depreciated
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
        >>> rand_var = K.random_normal_variable(shape=(3,3), mean=0, scale=1)
        >>> K.eval(rand_var)
        array([[ 1.16307867,  2.21220636,  0.48380461],
               [ 0.7740038 ,  0.29956347,  1.04344034],
               [ 0.15302546,  1.18392551, -1.16881478]], dtype=float32)
        >>> rand_var1 = K.random_normal_variable(shape=(3,3), mean=0, scale=1, seed=128)
        >>> rand_var2 = K.random_normal_variable(shape=(3,3), mean=0, scale=1, seed=128)
        >>> K.eval(rand_var1)
        array([[-0.75213492,  0.47400656,  0.95352972],
               [ 0.20251541, -0.62203991,  1.36481571],
               [-0.08511394, -1.4962182 , -0.20014545]], dtype=float32)
        >>> K.eval(rand_var2)
        array([[-0.75213492,  0.47400656,  0.95352972],
               [ 0.20251541, -0.62203991,  1.36481571],
               [-0.08511394, -1.4962182 , -0.20014545]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, "randomnormal")
    if seed:
        mx.random.seed(seed)
    value = mx.random.normal(loc=mean, scale=scale, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    kvar = _keras_variable(name=name, shape=shape, dtype=dtype)
    kvar.bind(value)
    return kvar


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
    shape = x.shape
    return np.prod([shape[i] for i in range(len(shape))])


@keras_symbol_child
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
        placeholder1:[tensor=True dtype=float32]
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        cast0:[tensor=True dtype=float16]
        >>> input
        placeholder1:[tensor=True dtype=float32]
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        cast2:[tensor=True dtype=float16]
    ```
    """
    if isinstance(x, KerasSymbol):
        return KerasSymbol(
            mx.sym.Cast(data=x.symbol, dtype=dtype))
    elif hasattr(x, astype):
        return x.astype(dtype)
    else:
        raise TypeError("The input is invalid for cast operation.")


# UPDATES OPS
def update(x, new_x):
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported')


def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported')


def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported')


def moving_average_update(x, value, momentum):
    """Compute the moving average of a variable.

    # Arguments
        x: A `Variable`.
        value: A tensor with the same shape as `x`.
        momentum: The moving average momentum.

    # Returns
        An operation to update the variable.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported')


# LINEAR ALGEBRA
@keras_symbol_child
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
        dot0:[tensor=True dtype=float32]
        >>> xy.shape
        (2, 4)
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        dot0:[tensor=True dtype=float32]
        >>> xy.shape
        (32, 28, 4)
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
    if ndim(y) > 2:
        axis = list(range(ndim(y)))
        axis = [axis.pop(-2)] + axis
        y = mx.sym.transpose(y.symbol, axes=axis)
    else:
        y = y.symbol
    return KerasSymbol(mx.sym.dot(lhs=x.symbol, rhs=y))


## TODO: batch_dot is not correct, now the argument becomes axes instead of dim
@keras_symbol_child
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

    if ndim(x) == ndim(y) == 2:
        assert axes is None or axes == 1 or tuple(axes) == (1, 1)
        axes = (2, 1)
        x = expand_dims(x, dim=1)
        y = expand_dims(y, dim=2)
        extra = True
    else:
        assert ndim(x) == ndim(y) == 3, "Only support 2d or 3d tensors for now"
        extra = False

    if isinstance(axes, Number):
        axes = (axes, axes)
    if axes is None:
        Ta = Tb = False
    else:
        Ta = not bool(axes[0] - 1)
        Tb = bool(axes[1] - 1)

    ret = KerasSymbol(mx.sym.batch_dot(lhs=x.symbol, rhs=y.symbol,
                                       transpose_a=Ta, transpose_b=Tb))
    if extra:
        ret = squeeze(ret, 2)

    return ret


@keras_symbol_child
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
        placeholder1:[tensor=True dtype=float32]
        >>> inputs.shape
        (2,3)
        >>> input_transposed = K.transpose(inputs)
        >>> input_transposed
        transpose1:[tensor=True dtype=float32]
        >>> input_transposed.shape
        (3,2)
    ```
    """
    return KerasSymbol(
        mx.sym.transpose(data=x.symbol))


@keras_symbol_child
def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    indices = mx.sym.Cast(indices.symbol, dtype=reference.dtype)
    return KerasSymbol(mx.sym.take(reference.symbol, indices))


# ELEMENT-WISE OPERATIONS
@keras_symbol_child
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
    return KerasSymbol(mx.sym.max(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_symbol_child
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
    return KerasSymbol(mx.sym.min(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_symbol_child
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
    return KerasSymbol(mx.sym.sum(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_symbol_child
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
    return KerasSymbol(mx.sym.prod(data=x.symbol, axis=axis, keepdims=keepdims))


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    raise NotImplementedError()


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    raise NotImplementedError()


@keras_symbol_child
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
    if isinstance(x, KerasSymbol):
        x = x.symbol
    mean_input = mx.sym.mean(data=x, axis=axis, keepdims=True)
    centered_input = mx.sym.broadcast_minus(lhs=x, rhs=mean_input)
    v = mx.sym.mean(data=(centered_input ** 2), axis=axis, keepdims=keepdims)
    return KerasSymbol(v)


@keras_symbol_child
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
    v = var(x, axis=axis, keepdims=keepdims)
    ret = mx.sym.sqrt(data=v.symbol)
    return KerasSymbol(ret)


@keras_symbol_child
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
    if axis == []:
        return x
    axis = _normalize_axis(axis, ndim(x))
    if dtype(x) == 'uint8':
        x = cast(x, floatx())
    if axis is not None:
        ret = mx.sym.mean(data=x.symbol, axis=axis, keepdims=keepdims)
    else:
        ret = mx.sym.mean(data=x.symbol, keepdims=keepdims)
    return KerasSymbol(ret)


@keras_symbol_child
def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    non_zero = (x != 0)
    var_cast = mx.sym.Cast(data=non_zero, dtype=np.int32)
    var_sum = mx.sym.sum_axis(data=var_cast, axis=axis, keepdims=keepdims)
    return KerasSymbol(var_sum > 0)


@keras_symbol_child
def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    var_abs = mx.sym.abs(data=x)
    var_min = mx.sym.min_axis(data=var_abs, axis=axis, keepdims=keepdims)
    return KerasSymbol(var_min > 0)


@keras_symbol_child
def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmax(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


@keras_symbol_child
def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmin(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


@keras_symbol_child
def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.square(data=x.symbol))


@keras_symbol_child
def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.abs(data=x.symbol))


@keras_symbol_child
def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    ret = mx.sym.Activation(data=x.symbol, act_type='relu')
    ret = mx.sym.sqrt(data=ret)
    return KerasSymbol(ret)


@keras_symbol_child
def exp(x):
    """Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.exp(data=x.symbol))


@keras_symbol_child
def log(x):
    """Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.log(data=x.symbol))


@keras_symbol_child
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
    raise NotImplementedError()


@keras_symbol_child
def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.round(data=x.symbol))


@keras_symbol_child
def sign(x):
    """Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sign(data=x.symbol))


@keras_symbol_child
def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(a, KerasSymbol):
        a = a.symbol
    return KerasSymbol(mx.sym.pow(base=x, exp=a))


@keras_symbol_child
def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float or integer.
        max_value: Python float or integer.

    # Returns
        A tensor.
    """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    min_value = np.float32(min_value)
    max_value = np.nan_to_num(np.float32(max_value))
    return KerasSymbol(mx.sym.clip(data=x.symbol, a_min=min_value, a_max=max_value))


@keras_symbol_child
def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x == y, dtype='uint8'))
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_equal(lhs=x, rhs=y), dtype='uint8'))
    else:
        raise TypeError


@keras_symbol_child
def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x != y, dtype='uint8'))
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_not_equal(lhs=x, rhs=y), dtype='uint8'))
    else:
        raise TypeError


@keras_symbol_child
def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x > y, dtype='uint8'))
    print("[Debug]", x)
    print("[Debug]", y)
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_greater(lhs=x, rhs=y), dtype='uint8'))
    else:
        raise TypeError


@keras_symbol_child
def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x >= y, dtype='uint8'))
    print("[Debug]", x)
    print("[Debug]", y)
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_greater_equal(lhs=x, rhs=y), dtype='uint8'))
    else:
        raise TypeError


@keras_symbol_child
def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x < y, dtype='uint8'))
    return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_lesser(lhs=x, rhs=y), dtype='uint8'))


@keras_symbol_child
def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if scalar:
        return KerasSymbol(mx.sym.Cast(x <= y, dtype='uint8'))
    return KerasSymbol(mx.sym.Cast(mx.sym.broadcast_lesser_equal(lhs=x, rhs=y), dtype='uint8'))


@keras_symbol_child
def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.maximum(left=x, right=y))


@keras_symbol_child
def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.minimum(left=x, right=y))


@keras_symbol_child
def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sin(data=x.symbol))


@keras_symbol_child
def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.cos(data=x.symbol))


@keras_symbol_child
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
    original_x = x
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    mean = mx.sym.mean(data=x, axis=reduction_axes, keepdims=False)
    var = _variance(x, axis=reduction_axes, keepdims=False)

    list_axe = list(range(ndim(original_x)))
    if sorted(reduction_axes) == list(range(ndim(original_x)))[:-1]:
        normed = batch_normalization(x, mean, var, beta, gamma, epsilon)
    else:
        # need broadcasting
        target_shape = []
        for axis in range(ndim(original_x)):
            if axis in reduction_axes:
                target_shape.append(1)
            else:
                target_shape.append(original_x.shape[axis])
        target_shape = tuple(target_shape)

        broadcast_mean = mx.sym.Reshape(data=mean, shape=target_shape)
        broadcast_var = mx.sym.Reshape(data=var, shape=target_shape)
        broadcast_gamma = mx.sym.Reshape(data=gamma, shape=target_shape)
        broadcast_beta = mx.sym.Reshape(data=beta, shape=target_shape)
        normed = batch_normalization(x, broadcast_mean, broadcast_var,
                                     broadcast_beta, broadcast_gamma,
                                     epsilon)

    return normed, KerasSymbol(mean), KerasSymbol(var)


@keras_symbol_child
def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(mean, KerasSymbol):
        mean = mean.symbol
    if isinstance(var, KerasSymbol):
        var = var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    mean = mx.sym.stop_gradient(mean)
    var = mx.sym.stop_gradient(var)

    # gradient explode when learning gamma and beta at together.
    gamma = mx.sym.stop_gradient(gamma)

    std = mx.sym.sqrt(data=(var + epsilon))

    x = mx.sym.broadcast_minus(x, mean)
    x = mx.sym.broadcast_div(x, std)
    x = mx.sym.broadcast_mul(x, gamma)
    x = mx.sym.broadcast_plus(x, beta)
    return KerasSymbol(x)


@keras_symbol_child
def mxnet_batchnorm(x, gamma, beta, moving_mean, moving_var, axis=-1, epsilon=1e-3):
    """Apply MXNet batch norm"""
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(moving_mean, KerasSymbol):
        moving_mean = moving_mean.symbol
    if isinstance(moving_var, KerasSymbol):
        moving_var = moving_var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    return KerasSymbol(
        mx.sym.BatchNorm(x, gamma, beta, moving_mean,
                         moving_var, axis=axis, eps=epsilon))


# SHAPE OPERATIONS
@keras_symbol_child
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.

    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(tensors[0])
    tensors = [t.symbol for t in tensors]
    return KerasSymbol(mx.sym.Concat(*tensors, dim=axis))


@keras_symbol_child
def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    """
    shape = tuple([0 if x is None else x for x in shape])
    return KerasSymbol(mx.sym.Reshape(data=x.symbol, shape=shape))


@keras_symbol_child
def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.transpose(x.symbol, axes=pattern))


@keras_symbol_child
def resize_images(x, height_factor, width_factor, data_format):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    x = x.symbol
    if data_format == 'channels_last':
        x = mx.sym.repeat(x, repeats=height_factor, axis=1)
        x = mx.sym.repeat(x, repeats=width_factor, axis=2)
    elif data_format == 'channels_first':
        x = mx.sym.repeat(x, repeats=height_factor, axis=2)
        x = mx.sym.repeat(x, repeats=width_factor, axis=3)
    else:
        raise ValueError("MXNET Backend: Data format is neither channels_first or channels_last")

    return KerasSymbol(x)


@keras_symbol_child
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
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    x = x.symbol
    if data_format == 'channels_last':
        x = mx.sym.repeat(x, repeats=depth_factor, axis=1)
        x = mx.sym.repeat(x, repeats=height_factor, axis=2)
        x = mx.sym.repeat(x, repeats=width_factor, axis=3)
    elif data_format == 'channels_first':
        x = mx.sym.repeat(x, repeats=depth_factor, axis=2)
        x = mx.sym.repeat(x, repeats=height_factor, axis=3)
        x = mx.sym.repeat(x, repeats=width_factor, axis=4)
    else:
        raise ValueError("MXNET Backend: Data format is neither channels_first or channels_last")

    return KerasSymbol(x)


@keras_symbol_child
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
    return KerasSymbol(mx.sym.repeat(x.symbol, repeats=rep, axis=axis))


@keras_symbol_child
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
    x = mx.sym.expand_dims(x.symbol, axis=1)
    x = mx.sym.repeat(x, repeats=n, axis=1)
    return KerasSymbol(x)


@keras_symbol_child
def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

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
    dtype = np.dtype(dtype)
    return KerasSymbol(mx.sym.arange(start=start, stop=stop, step=step, dtype=dtype))


@keras_symbol_child
def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    return KerasSymbol(mx.sym.tile(x.symbol, reps=n))


@keras_symbol_child
def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    return KerasSymbol(mx.sym.Reshape(data=x.symbol, shape=(-1,)))


@keras_symbol_child
def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.

    In other words, it flattens each data samples of a batch.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Flatten(data=x.symbol))


@keras_symbol_child
def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    if axis < 0:
        axis %= len(x.shape) + 1
    if isinstance(x, KerasSymbol):
        x = x.symbol
        return KerasSymbol(mx.sym.expand_dims(x, axis=axis))


@keras_symbol_child
def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    shape = list(x.shape)
    assert shape.pop(axis) == 1, "Can only squeeze size 1 dimension"

    if isinstance(x, KerasSymbol):
        x = x.symbol
        return KerasSymbol(mx.sym.Reshape(data=x, shape=tuple(shape)))


@keras_symbol_child
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

    assert ndim(x) == 3

    # MXNet only supports padding for 4D and 5D tensor.
    # Reshaping to 4D, perform padding, reshape back to 3D.
    x_shape = x.shape
    x_4d = KerasSymbol(mx.sym.Reshape(x.symbol, shape=(x_shape[0], 1, x_shape[1], x_shape[2])))
    x_4d_padded = KerasSymbol(mx.sym.pad(data=x_4d, mode='constant', constant_value=0,
                                         pad_width=(0, 0, 0, 0, padding[0], padding[1], 0, 0, )))
    x_3d_padded = KerasSymbol(mx.sym.Reshape(x_4d_padded, shape=(x_shape[0], x_shape[1] + padding[0]
                                                                 + padding[1], x_shape[2])))
    return x_3d_padded


@keras_symbol_child
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2

    assert ndim(x) == 4

    if data_format is None:
        data_format = image_data_format()

    if data_format not in {'channels_first'}:
        raise ValueError("MXNet Backend: MXNet supports only 'channels_first' data format. "
                         "Unknown data_format - {0} provided for padding".format(str(data_format)))

    pattern = (0, 0, 0, 0, padding[0][0], padding[0][1], padding[1][0], padding[1][1])

    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=pattern))


@keras_symbol_child
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
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.

    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2

    assert ndim(x) == 5

    if data_format is None:
        data_format = image_data_format()

    if data_format not in {'channels_first'}:
        raise ValueError("MXNet Backend: MXNet supports only 'channels_first' data format. "
                         "Unknown data_format - {0} provided for padding".format(str(data_format)))

    pattern = (
        0, 0,
        0, 0,
        padding[0][0], padding[0][1],
        padding[1][0], padding[1][1],
        padding[2][0], padding[2][1]
    )
    return KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant',
                                  constant_value=0,
                                  pad_width=pattern))


def stack(x, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.
    """
    raise NotImplementedError()


@keras_symbol_child
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
    return KerasSymbol(mx.symbol.one_hot(indices.symbol, depth=num_classes))


@keras_symbol_child
def reverse(x, axes):
    """Reverse a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.symbol.reverse(data=x.symbol, axis=axes))


# VALUE MANIPULATION
def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    return eval(x)


def batch_get_value(ops):
    """Returns the value of more than one tensor variable.

    # Arguments
        ops: list of ops to run.

    # Returns
        A list of Numpy arrays.
    """
    return [get_value(op) for op in ops]


def set_value(x, value):
    """Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    """
    if isinstance(value, Number):
        value = [value]
    x.bind(mx.nd.array(value))


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    for p, w in tuples:
        set_value(p, w)


def get_variable_shape(x):
    """Returns the shape of a variable.

    # Arguments
        x: A variable.

    # Returns
        A tuple of integers.
    """
    return x.shape


def print_tensor(x, message=''):
    """Prints `message` and the tensor value when evaluated.

    # Arguments
        x: Tensor to print.
        message: Message to print jointly with the tensor.

    # Returns
        The same tensor `x`, unchanged.
    """
    print(message, eval(x))


@keras_symbol_child
def group(variables):
    var = [x if isinstance(x, mx.sym.Symbol) else x.symbol for x in variables]
    sym = mx.sym.Group(var)
    return KerasSymbol(sym)


@keras_symbol_child
def make_loss(variables):
    sym = mx.sym.MakeLoss(variables.symbol)
    return KerasSymbol(sym)


# GRAPH MANIPULATION
class Function(object):
    def __init__(self, inputs, output, updates=[], **kwargs):
        self.output = output
        self.updates = updates
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            self.inputs = inputs[:-1]
        else:
            self.inputs = inputs
            self.is_train = learning_phase()

    def __call__(self, inputs):
        ret_outputs = []
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            inputs = inputs[:-1]
        for x in self.output:
            bind_values = dfs_get_bind_values(x)
            data = {k.name: v for k, v in zip(self.inputs, inputs)}
            data = dict(data, **bind_values)
            args = x.symbol.list_arguments()
            data_shapes = {k.name: v.shape for k, v in zip(self.inputs, inputs) if k.name in args}
            executor = x.symbol.simple_bind(mx.cpu(), grad_req='null', **data_shapes)
            for v in executor.arg_dict:
                if v in data:
                    executor.arg_dict[v][:] = data[v]
            outputs = executor.forward(is_train=self.is_train)
            ret_outputs.append(outputs[0].asnumpy())
        return ret_outputs


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    """
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    """Returns the gradients of `variables` w.r.t. `loss`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    """
    raise NotImplementedError()


@keras_symbol_child
def stop_gradient(variables):
    """Returns `variables` but with zero gradient w.r.t. every other variable.

    # Arguments
        variables: tensor or list of tensors to consider constant with respect
            to any other variable.

    # Returns
        A single tensor or a list of tensors (depending on the passed argument)
            that has constant gradient with respect to any other variable.
    """
    if isinstance(variables, KerasSymbol):
        return KerasSymbol(mx.sym.BlockGrad(variables.symbol))
    elif isinstance(variables, list):
        out = []
        for variable in variables:
            out.append(KerasSymbol(mx.sym.BlockGrad(variable.symbol)))
        return out
    else:
        raise ValueError("MXNet backend: Stop gradient requires tensor or "
                         "list of tensors, but, given {0}".format(type(variables)))


# CONTROL FLOW
def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        step_function: RNN step function.
            Parameters:
                inputs: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                outputs: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
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
    raise NotImplementedError()


@keras_symbol_child
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

    # Raises
        ValueError: If rank of `condition` is greater than rank of expressions.
    """
    if callable(then_expression):
        then_expression = then_expression()
    if callable(else_expression):
        else_expression = else_expression()
    assert isinstance(condition, KerasSymbol) and isinstance(then_expression, KerasSymbol) \
        and isinstance(else_expression, KerasSymbol)
    return KerasSymbol(
        mx.sym.where(condition.symbol, then_expression.symbol, else_expression.symbol))


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

    if training is 1:
        if callable(x):
            return x()
        else:
            return x
    elif training is 0:
        if callable(alt):
            return alt()
        else:
            return alt

    x = switch(training, x, alt)
    return x


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
    raise in_train_phase(alt, x, training=training)



# NN OPERATIONS
@keras_symbol_child
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
    ret = mx.sym.LeakyReLU(data=x.symbol, act_type='leaky', slope=alpha)
    if max_value and max_value > 0:
        ret = mx.sym.minimum(ret, max_value)
    return KerasSymbol(ret)


@keras_symbol_child
def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.LeakyReLU(data=x.symbol, act_type='elu', slope=alpha))


@keras_symbol_child
def softmax(x):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.SoftmaxActivation(data=x.symbol))


@keras_symbol_child
def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='softrelu'))


@keras_symbol_child
def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(x.symbol / (1 + mx.sym.abs(x.symbol)))


#TODO MXNet's softmax cross entropy throws error. Need to overcome this.
@keras_symbol_child
def categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    assert not from_logits
    axis = ndim(output) - 1
    mx_output = output.symbol
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1-epsilon())
    mx_output = - mx.sym.sum(target.symbol * mx.sym.log(mx_output), axis=axis, keepdims=True)
    return KerasSymbol(mx_output)


def sparse_categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    raise NotImplementedError()


@keras_symbol_child
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
    mx_output = output.symbol
    if from_logits:
        mx_output = mx.sym.Activation(mx_output, act_type='sigmoid')
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1-epsilon())
    mx_output = - (target.symbol * mx.sym.log(mx_output) + (1-target.symbol) * mx.sym.log(1-mx_output))
    return KerasSymbol(mx_output)


@keras_symbol_child
def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='sigmoid'))


@keras_symbol_child
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
    return KerasSymbol(mx.sym.clip(data=(0.2 * x.symbol + 0.5), a_min=0., a_max=1.))


@keras_symbol_child
def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.tanh(data=x.symbol))


@keras_symbol_child
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
    if not 0 <= level <= 1:
        raise ValueError("MXNet Backend: Invalid level provided for dropout '{0}'. "
                         "Expected between 0 and 1.".format(level))
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    name = _prepare_name(None, 'dropout')
    return KerasSymbol(mx.sym.Dropout(data=x.symbol, p=level, name=name))


@keras_symbol_child
def l2_normalize(x, axis=None):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    """
    norm = mx.sym.sqrt(data=mx.sym.sum(data=mx.sym.square(data=x.symbol), axis=axis, keepdims=True))
    return KerasSymbol(mx.sym.broadcast_div(x.symbol, norm))


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
    raise NotImplementedError()


# CONVOLUTIONS
@keras_symbol_child
def _preprocess_convnd_input(x, data_format):
    if data_format == 'channels_last' and ndim(x) > 3:
        idx = list(range(ndim(x)))
        idx.insert(1, idx.pop(-1))  # make it channel first format
        x = KerasSymbol(mx.sym.transpose(data=x.symbol, axes=idx))
    return x


@keras_symbol_child
def _postprocess_convnd_output(x, data_format):
    if data_format == 'channels_last' and ndim(x) > 3:
        idx = list(range(ndim(x)))
        idx.append(idx.pop(1))
        x = KerasSymbol(mx.sym.transpose(data=x.symbol, axes=idx))
    else:
        return KerasSymbol(x)


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
    """
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1),
                     padding='valid', data_format=None):
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

    # Returns
        A tensor, result of transposed 2D convolution.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


@keras_symbol_child
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
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("`data_format` should be either `channels_first` or `channels_last`.")
    if pool_mode not in {"max", "avg"}:
        raise ValueError("`pool_mode` should be either `max` or `avg`.")
    if padding not in {"same", "valid"}:
        raise ValueError("`padding` should be either `same` or `valid`.")
    x = _preprocess_convnd_input(x, data_format)
    mx_out = mx.sym.Pooling(data=x.symbol,
                            kernel=pool_size,
                            pool_type=pool_mode,
                            pooling_convention=padding,
                            stride=strides)
    return _postprocess_convnd_output(mx_out, data_format)


@keras_symbol_child
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
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("`data_format` should be either `channels_first` or `channels_last`.")
    if pool_mode not in {"max", "avg"}:
        raise ValueError("`pool_mode` should be either `max` or `avg`.")
    if padding not in {"same", "valid"}:
        raise ValueError("`padding` should be either `same` or `valid`.")
    x = _preprocess_convnd_input(x, data_format)
    mx_out = mx.sym.Pooling(data=x.symbol,
                            kernel=pool_size,
                            pool_type=pool_mode,
                            pooling_convention=padding,
                            stride=strides)
    return _postprocess_convnd_output(mx_out, data_format)


@keras_symbol_child
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
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    bias_shape = int_shape(bias)
    if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
        raise ValueError('Unexpected bias dimensions %d, expect to be 1 or %d dimensions'
                         % (len(bias_shape), ndim(x)))
    if ndim(x) == 5:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += bias
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1, ) + bias_shape)
    else:
        x += bias
    return x


# RANDOMNESS
@keras_symbol_child
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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.normal(shape=shape, loc=mean, scale=stddev, dtype=dtype)
    return KerasSymbol(sym)


@keras_symbol_child
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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.uniform(shape=shape, low=minval, high=maxval, dtype=dtype)
    return KerasSymbol(sym)


@keras_symbol_child
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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.uniform(shape=shape, low=0., high=1., dtype=dtype)
    sym = mx.sym.where(sym <= p,
                      mx.sym.ones(shape=shape, dtype=dtype),
                      mx.sym.zeros(shape=shape, dtype=dtype))
    return KerasSymbol(sym)


@keras_symbol_child
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
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if x is None else x for x in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.normal(shape=shape, loc=mean, scale=stddev, dtype=dtype)
    sym = mx.sym.clip(data=sym, a_min=mean - 2 * stddev, a_max=mean + 2 * stddev)
    return KerasSymbol(sym)


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError()


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
    raise NotImplementedError()


# Other Common Utilities
def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```
        >>> keras.backend.get_uid('dense')
        >>> 1
        >>> keras.backend.get_uid('dense')
        >>> 2
    ```
    """
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


def dfs_get_bind_values(node_start):
    stack_list = []
    visited = set()
    stack_list.append(node_start)
    while len(stack_list) > 0:
        cur_node = stack_list.pop()
        if cur_node in visited:
            continue
        visited.add(cur_node)
        next_nodes = cur_node.get_neighbor()
        for i in next_nodes:
            if i in visited:
                continue
            else:
                stack_list.append(i)
    bind_values = {}
    for key in visited:
        bind_values.update(key.get_bind_values())
    return bind_values


# Internal utility functions
def _keras_variable(name, shape, dtype, **kwargs):
    if dtype is None:
        dtype = floatx()
    v = mx.sym.Variable(name, shape=shape, dtype=dtype, **kwargs)
    ret = KerasSymbol(v, is_var=True)
    return ret


def _convert_string_dtype(dtype):
    """Get the type from a string.

    # Arguments
        dtype: A string representation of a type.

    # Returns
        The type requested.

    # Raises
        ValueError: if `dtype` is not supported.
    """
    mapping = {'float16': np.float16,
               'float32': np.float32,
               'float64': np.float64,
               'int8': np.int8,
               'int32': np.int32,
               'int64': np.int64,
               'uint8': np.int8,
               'uint16': np.uint16}

    if dtype not in mapping:
        raise ValueError('MXNet Backend: Unsupported dtype:', dtype)
    return mapping[dtype]


def _convert_dtype_string(dtype):
    """Get the String from type.

    # Arguments
        dtype: Type.

    # Returns
       A stromg representation of a type.

    # Raises
        ValueError: if `dtype` is not supported.
    """
    mapping = {np.float16: 'float16',
               np.float32: 'float32',
               np.float64: 'float64',
               np.int16: 'int16',
               np.int32: 'int32',
               np.int64: 'int64',
               np.int8: 'uint8',
               np.uint16: 'uint16'}

    if dtype not in mapping:
        raise ValueError('MXNet Backend: Unsupported dtype:', dtype)
    return mapping[dtype]


#@TODO check if this util function is correct
def _normalize_axis(axis, ndim):
    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list):
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    elif axis is None:
        return ()
    else:
        if axis < 0:
            axis = axis % ndim
    return axis


def _variance(x, axis=None, keepdims=False):
    mean_input = mx.sym.mean(data=x, axis=axis, keepdims=True)
    centered_input = mx.sym.broadcast_minus(lhs=x, rhs=mean_input)
    v = mx.sym.mean(data=(centered_input ** 2), axis=axis, keepdims=keepdims)
    return v
