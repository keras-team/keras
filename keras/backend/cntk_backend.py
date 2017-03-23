import cntk as C
import numpy as np
from cntk.cntk_py import NDArrayView

from .common import _FLOATX, _EPSILON, image_dim_ordering, reset_uids

from cntk.utils import sanitize_shape, sanitize_dtype_cntk, sanitize_dynamic_axes

from tempfile import TemporaryFile

#debug mode
_DEBUG = True
#flag for wether manual init or not
_MANUAL_VAR_INIT = False
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_LEARNING_PHASE = C.ops.Parameter(shape=(1,), dtype=np.float32)  # 0 = test, 1 = train

'''hijack
this is a temp approach to make the train function working, will have a better design later after confirm the solution works
'''
grad_placeholder_dict={}

def learning_phase():
    # False = test, True = train
    global _LEARNING_PHASE
    value = _LEARNING_PHASE.value
    return value[0]


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    v = np.asarray([value]).astype(np.float32)
    _LEARNING_PHASE.value = v

def in_train_phase(x, alt):
    global _LEARNING_PHASE
    if learning_phase() is 1.0:
        return x
    elif learning_phase() is 0.0:
        return alt

    if callable(x) and isinstance(x, C.cntk_py.Function) == False:
        x = x()
    if callable(alt) and isinstance(x, C.cntk_py.Function) == False:
        alt = alt()
    _LEARNING_PHASE.value = np.asarray([1])
    return x


def in_test_phase(x, alt):
    global _LEARNING_PHASE
    if learning_phase() is 1:
        return alt
    elif learning_phase() is 0:
        return x
    # else: assume learning phase is a placeholder tensor.
    _LEARNING_PHASE.value = np.asarray([0])
    return x

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return np.float16
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int32':
        return np.float32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'uint8':
        return np.int8
    elif dtype == 'uint16':
        return np.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)

def _convert_dtype_string(dtype):
    if dtype == np.float32:
        return 'float32'
    elif dtype == np.float64:
        return 'float64'
    elif dtype == np.int32:
        return np.int32
    else:
        raise ValueError('Unsupported dtype:', dtype)

def clear_session():
    raise NotImplementedError

def variable(value, dtype=_FLOATX, name=None):
    '''
    How to handle dtype
    '''
    cname = name
    if cname == None:
        cname = ''
    shape = value.shape if (hasattr(value, 'shape')) else (1)
    #cntk will init type based on the value type
    v = C.parameter(shape=shape, init=value, name=cname)
    v._keras_shape = v.shape
    v._uses_learning_phase = False
    return v

def eval(x):
    if (isinstance(x, C.cntk_py.Function)):
        return x.eval()
    else:
        #CNTK doesn't support to evaluate on a parameter/variable, so use this trick temporarly
        return C.alias(x, name='eval').eval()

def manual_variable_initialization(value):
    '''Returns a boolean:
    whether variables should be initialized
    as they are instantiated (default), or if
    the user should handle the initialization
    (e.g. via tf.initialize_all_variables()).
    '''
    raise NotImplementedError

def placeholder(shape=None, ndim=None, dtype=_FLOATX, sparse=False, name=None, dynamic_axis_num=1):
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    '''none dim is not support in cntk now'''
    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    keras_shape = shape
    '''hijack
        if input shape is (None, None), how to handle it in CNTK?
    '''
    if nones > dynamic_axis_num:
        raise  NotImplementedError

    if (dynamic_axis_num - nones) > len(shape):
        raise  NotImplementedError

    if name is None:
        name = ''

    shift_dim = dynamic_axis_num
    shape = shape[shift_dim:]

    if (dynamic_axis_num == 1):
        x = C.input_variable(shape=shape, dtype=_convert_string_dtype(dtype), is_sparse=sparse, name=name, dynamic_axes=[C.Axis.default_batch_axis()])
    elif (dynamic_axis_num == 2):
        x = C.input_variable(shape=shape, dtype=_convert_string_dtype(dtype), is_sparse=sparse, name=name)
    else:
        raise NotImplementedError

    x._keras_shape = keras_shape
    x._uses_learning_phase = False
    return x

def shape(x):
    raise int_shape(x);

def is_sparse(tensor):
    '''hijack'''
    return tensor.is_sparse

def to_dense(tensor):
    raise NotImplementedError;

def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape

    if isinstance(x, C.cntk_py.Function):
        x = x.output

    shape = x.shape
    if hasattr(x, 'dynamic_axes'):
        dynamic_shape = [None for a in x.dynamic_axes]
        shape = tuple(dynamic_shape) + shape
    return shape


def ndim(x):
    shape = int_shape(x)
    return len(shape)

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
    return random_uniform_variable(shape, 0.0, 1.0, dtype, seed=seed)

def random_uniform_variable(shape, low, high, dtype=_FLOATX,
                            name=None, seed=None):
    '''
    How to handle dtype?
    '''
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    x = NDArrayView.random_uniform_float(shape, low, high, seed).to_ndarray()
    return variable(x, name=name)

def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape).astype(np.float32),
                    dtype=dtype, name=name)

def random_normal(shape, mean=0.0, std=1.0, dtype=None, seed=None):
    '''hijack'''
    return variable(np.random.standard_normal(size=shape).astype(np.float32),dtype=dtype)

def zeros_like(x, dtype=None, name=None):
    return x * 0

def dtype(x):
    '''Returns the dtype of a tensor, as a string.
    '''
    return _convert_dtype_string(x.dtype)

def zeros(shape, dtype=_FLOATX, name=None):
    ctype = _convert_string_dtype(dtype)
    return variable(value=np.zeros(shape, ctype), dtype=dtype, name=name)

def ones(shape, dtype=_FLOATX, name=None):
    ctype = _convert_string_dtype(dtype)
    return variable(value=np.ones(shape, ctype), dtype=dtype, name=name)

def eye(size, dtype=_FLOATX, name=None):
    return variable(np.eye(size), dtype, name)

def ones_like(x, name=None):
    return (x+1) / (x + 1)

def count_params(x):
    shape = x.shape
    return np.prod([shape[i] for i in range(len(shape))])

def cast(x, dtype):
    '''To do'''
    return x

'''PASS'''
def dot(x, y):
    '''
    Todo: Theano behavior, maybe times_transpose
    '''
    if (isinstance(x, np.ndarray)):
        if(len(x) != 1):
            raise NotImplementedError;
        else:
            x = x[0]
    if (isinstance(y, np.ndarray)):
        if (len(y) != 1):
            raise NotImplementedError;
        else:
            y = y[0]

    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        y_shape = int_shape(y)
        j = len(y_shape) - 2
        while (j > 0):
            y = C.transpose(y, j, j - 1)
            j -= 1
        return C.times(x, y, len(y_shape) - 1)
    else:
        return C.times(x, y)

def batch_dot(x, y, axes=None):
    x_shape = int_shape(x)
    y_shape = int_shape(y)

    if (len(x_shape) == 2):
        expand_dims(x, 1)

    if (len(y_shape) == 2):
        expand_dims(y, 2)

    x_shape = int_shape(x)
    y_shape = int_shape(y)

    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [len(x_shape) - 1, len(y_shape) - 2]

    normalized_axis = []
    normalized_axis.append(_normalize_axis(axes[0], x_shape)[0])
    normalized_axis.append(_normalize_axis(axes[1], y_shape)[0])
    #transpose
    i = normalized_axis[0]
    while(i < len(x.shape) - 1):
        x = C.transpose(x, i, i+1)
        i+=1
    i = normalized_axis[1]
    while (i > 0):
        y = C.transpose(y, i, i-1)
        i -= 1
    return C.times(x, y, output_rank=(len(y.shape)-1))

'''PASS'''
def transpose(x):
    return C.transpose(x, 0, 1)

'''missing'''
def gather(reference, indices):
    """Retrieves the elements of indices `indices`
        in the tensor `reference`.

        # Arguments
            reference: A tensor.
            indices: An integer tensor of indices.

        # Returns
            A tensor of same type as `reference`.
        """
    #shape = reference.shape
    #one_hot_matrix = C.ops.one_hot_op(indices, shape[0])
    #return C.times(one_hot_matrix, reference)
    return C.ops.gather(indices, reference)

def max(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, int_shape(x))
    if isinstance(axis, list):
        for a in axis:
            x = C.reduce_max(x, a)
        output = x
    else:
        output = C.reduce_max(x, axis)

    if keepdims == False and isinstance(axis, list):
        return _reshape_dummy_dim(output, axis)
    else:
        return output

def min(x, axis=None, keepdims=False):
    #how to handle keepdims?
    axis = _normalize_axis(axis, int_shape(x))
    if isinstance(axis, list):
        for a in axis:
            x = C.reduce_min(x, a)
        output = x
    else:
        output = C.reduce_min(x, axis)

    if keepdims == False and isinstance(axis, list):
        return _reshape_dummy_dim(output, axis)
    else:
        return output
'''PASS'''
def sum(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, int_shape(x))
    if isinstance(axis, list):
        for a in axis:
            x = C.reduce_sum(x, a)
        output = x
    else:
        output = C.reduce_sum(x, axis)

    if keepdims == False and isinstance(axis, list):
        return _reshape_dummy_dim(output, axis)
    else:
        return output

def prod(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, int_shape(x))
    if isinstance(axis, list):
        for a in axis:
            x = C.reduce_prod(x, a)
        output = x
    else:
        output = C.reduce_prod(x, axis)

    if keepdims == False and isinstance(axis, list):
        return _reshape_dummy_dim(output, axis)
    else:
        return output

def var(x, axis=None, keepdims=False):
    m = mean(x, axis, keepdims=True)
    devs_squared = C.square(x - m)
    return mean(devs_squared,axis=axis,keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return C.sqrt(var(x, axis=axis, keepdims=keepdims))

def expand_dims(x, dim=-1):
    shape = list(int_shape(x))
    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    index = dim if dim >= 0 else len(shape) + 1
    shape.insert(index, 1)
    new_shape = tuple(shape[nones:])
    return C.reshape(x, new_shape)

def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    shape = list(int_shape(x))
    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    shape.pop(axis)
    new_shape = tuple(shape[nones:])
    return C.reshape(x, new_shape)

def tile(x, n):
    shape = int_shape(x)
    if (len(n) != len(shape)):
        raise NotImplementedError
    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    i = nones
    while (i < len(n)):
        if (shape[i] != None):
            rep = n[i]
            tmp = [x for i in range(rep)]
            x = C.splice(*tmp, axis = i-nones)
        i+=1

    return x

def _normalize_axis(axis, shape):
    ndim = len(shape)

    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    if type(axis) is tuple:
        axis = list(axis)
    elif type(axis) is int:
        axis = [axis]

    if type(axis) is list:
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = (a % ndim)
            if (axis[i] is not None):
                axis[i] -= nones
    else:
        if axis is None:
            axis = C.Axis.all_axes()

    return axis

def _reshape_dummy_dim(x, axis):
    shape = list(x.shape)
    for index in sorted(axis, reverse=True):
        del shape[index]

    shape = tuple(shape)
    return C.reshape(x, shape)

def mean(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, int_shape(x))
    if isinstance(axis, list):
        for a in axis:
            x = C.reduce_mean(x, a)
        output = x
    else:
        output = C.reduce_mean(x, axis)

    if keepdims == False and isinstance(axis, list):
        return _reshape_dummy_dim(output, axis)
    else:
        return output

def any(x, axis=None, keepdims=False):
    raise NotImplementedError

def all(x, axis=None, keepdims=False):
    raise NotImplementedError

def classification_error(output, target, axis = -1):
    #return C.equal(argmax(output, axis=-1), argmax(target, axis=-1))
    return C.ops.reduce_mean(C.equal(argmax(output, axis=-1),argmax(target, axis=-1)), axis=C.Axis.all_axes())
    #return C.ops.classification_error(output, target, axis)

def argmax(x, axis=-1):
    axis = [axis]
    axis = _normalize_axis(axis, int_shape(x))
    output = C.ops.argmax(x, axis=axis[0])
    return _reshape_dummy_dim(output, axis)

def argmin(x, axis=-1):
    axis = [axis]
    axis = _normalize_axis(axis, int_shape(x))
    output = C.ops.argmin(x, axis=axis[0])
    return _reshape_dummy_dim(output, axis)

def square(x):
    return C.square(x)
'''PASS'''
def abs(x):
    return C.abs(x)
'''PASS'''
def sqrt(x):
    return C.sqrt(x)
'''PASS'''
def exp(x):
    return C.exp(x)
'''PASS'''
def log(x):
    return C.log(x)
'''PASS'''
def round(x):
    return C.round(x)
'''PASS'''
def sigmoid(x):
    return C.sigmoid(x)

def sign(x):
    return x / C.abs(x)

def pow(x, a):
    '''may have issue with negative number'''
    raise NotImplementedError
'''PASS'''
def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return C.clip(x, min_value, max_value)

'''PASS'''
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
    if from_logits:
        output = C.sigmoid(output)
    output = C.clip(output, _EPSILON, 1.0 - _EPSILON)
    output = -1 * target * C.log(output) - (1.0 - target) * C.log(1.0 - output)
    return output

def get_variable_shape(x):
    return x.shape

def batch_get_value(xs):
    result = []
    for x in xs:
        if (isinstance(x, C.ops.variables.Parameter)):
            result.append(x.value)
        else:
            raise NotImplementedError
    return result;

def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.
        It returns `None`.

        # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    for pair in tuples:
        p = pair[0]
        v = pair[1]
        #v = _create_NDArrayView_from_NumPy(v.astype(np.float32))
        p.value=v.astype(np.float32)

def update(x, new_x):
    return {"function" : new_x, "variable" : x}

def update_add(x, increment):
    result = x + increment
    return {"function": result, "variable": x}

#use this structure to remember which parameter's gradiant is mapped to this input
class GradientPlaceHolder(C.ops.variables.Variable):
    def __init__(self, shape=None, dtype=np.float32, is_sparse=False, name='', variable=None, dynamic_axes=C.Axis.default_input_variable_dynamic_axes()):
        self.parameter = variable
        shape = sanitize_shape(shape)

        if dtype is None:
            dtype = np.float32
        dtype = sanitize_dtype_cntk(dtype)
        dynamic_axes = sanitize_dynamic_axes(dynamic_axes)

        super(self.__class__, self).__init__(shape=shape, dtype=dtype, is_sparse=is_sparse, name=name, dynamic_axes=dynamic_axes)

    def getParameter(self):
        return self.parameter

def gradients(loss, variables):
    '''Returns the gradients of `variables` (list of tensor variables)
    with regard to `loss`.
    '''
    '''this is a temp solution with global variable'''
    global grad_placeholder_dict
    if isinstance(variables, list) == False:
        variables = [variables]
    grads = []
    for v in variables:
        #p = placeholder(shape=v.shape, name=v.name + '_placehoder')
        p = C.ops.input_variable(shape=v.shape, name=v.name + '_placehoder')
        grads.append(p)
        grad_placeholder_dict[v] = p
    return grads
'''PASS'''
def equal(x, y):
    return C.equal(x, y)
'''PASS'''
def not_equal(x, y):
    return C.not_equal(x, y)
'''PASS'''
def greater(x, y):
    return C.greater(x, y)
'''PASS'''
def greater_equal(x, y):
    return C.greater_equal(x, y)
'''PASS'''
def lesser(x, y):
    return C.less(x, y)
'''PASS'''
def lesser_equal(x, y):
    return C.less_equal(x, y)

def maximum(x, y):
    return C.element_select(C.greater(x, y), x, y)

def minimum(x, y):
    return C.element_select(C.less(x, y), x, y)

'''PASS'''
def sin(x):
    return C.sin(x)
'''PASS'''
def cos(x):
    return C.cos(x)

def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    raise NotImplementedError

def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    #we have this op but need double check on the usage
    raise NotImplementedError
'''PASS'''
def concatenate(tensors, axis=-1):
    if (len(tensors)==0):
        return None

    axis = [axis]
    axis = _normalize_axis(axis, int_shape(tensors[0]))
    return C.splice(*tensors, axis=axis[0])

def reshape(x, shape):
    if isinstance(x, C.ops.variables.Parameter):
        return C.reshape(x, shape)
    else:
        num_dynamic_axis = _get_dynamic_axis_num(x)

        new_shape = list(shape)
        new_shape = new_shape[num_dynamic_axis:]
        new_shape = tuple(new_shape)
        return C.reshape(x, new_shape)

def permute_dimensions(x, pattern):
    #could be done by transpose several times
    dims = len(int_shape(x))
    num_dynamic_axis =  _get_dynamic_axis_num(x)
    current_layout = [i for i in range(dims)]
    axis_position = [i for i in range(dims)]
    i = 0
    while (i < num_dynamic_axis):
        if pattern[i] != current_layout[i]:
            raise NotImplementedError;
        i += 1

    while (i < dims):
        if (pattern[i] != current_layout[i]):
            current_pos = axis_position[pattern[i]]
            x = C.transpose(x, i - dims, current_pos - dims)
            axis_position[pattern[i]] = i
            axis_position[current_layout[i]] = current_pos
            current_layout[i], current_layout[current_pos] = current_layout[current_pos], current_layout[i]
        i += 1
    return x

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
    raise NotImplementedError

def repeat_elements(x, rep, axis):
    axis = _normalize_axis(axis, int_shape(x))
    axis = axis[0]
    slices = []
    shape = x.shape
    i = 0
    while (i < shape[axis]):
        tmp = C.ops.slice(x, axis, i, i + 1)
        for _ in range(rep):
            slices.append(tmp)
        i += 1
    return C.splice(*slices, axis = axis)


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Returns
        A tensor.
    """
    assert ndim(x) == 2
    index = 1 - _get_dynamic_axis_num(x)
    if index < 0 or index > 1:
        raise NotImplementedError

    new_shape = list(x.shape)
    new_shape.insert(index, 1)
    new_shape = tuple(new_shape)
    x = C.reshape(x, new_shape)
    temp = [x for i in range(n)]
    x = C.splice(*temp, axis=index)
    return x

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
    result = np.arange(start=start, stop=stop, step=step, dtype=dtype)
    result = variable(result, name='arange')
    return result


def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return C.ops.tanh(x)

def _static_rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    shape = int_shape(inputs)
    dims = len(shape)

    if (dims < 3):
        raise ValueError('Input should be at least 3D.')

    #if the second axis is static axis, CNTK will do unroll by default
    if (shape[1] == None):
        raise ValueError('Static rnn in cntk could only be executed with second axis as static axis')

    if constants is None:
        constants = []

    if mask is not None:
        mask_shape = int_shape(mask)
        if len(mask_shape) == dims - 1:
            mask = expand_dims(mask)

    nones = 0
    i = 0
    while (i < len(shape)):
        if shape[i] == None:
            nones += 1
        else:
            break
        i += 1

    states = tuple(initial_states)

    outputs = []

    time_axis = 1 - nones if nones > 0 else 1

    if go_backwards:
        i = shape[1] - 1
        while (i >= 0):
            current = C.ops.slice(inputs, time_axis, i, i + 1)
            # remove dummy dimension
            current = squeeze(current, time_axis)

            output, new_states = step_function(current, tuple(states) + tuple(constants))

            if mask is not None:
                mask_slice = C.ops.slice(mask, time_axis, i, i + 1)
                mask_slice = squeeze(mask_slice, time_axis)
                if len(outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = outputs[-1]
                output = C.ops.element_select(mask_slice, output, prev_output)

                return_states = []
                for s, n_s in zip(states, new_states):
                    return_states.append(C.ops.element_select(mask_slice, n_s, s))
                new_states = return_states
            # need reshape?
            outputs.append(output)
            states = new_states
            i -= 1
    else:
        i = 0
        while(i < shape[1]):
            current = C.ops.slice(inputs, time_axis, i, i+1)
            # remove dummy dimension
            current = squeeze(current, 1)

            output, new_states = step_function(current, tuple(states) + tuple(constants))

            if mask is not None:
                mask_slice = C.ops.slice(mask, time_axis, i, i+1)
                mask_slice = squeeze(mask_slice, 1)
                if len(outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = outputs[-1]
                output = C.ops.element_select(mask_slice, output, prev_output)

                return_states = []
                for s, n_s in zip(states, new_states):
                    return_states.append(C.ops.element_select(mask_slice, n_s, s))
                new_states = return_states
            #need reshape?
            outputs.append(output)
            states = new_states
            i+=1

    i = 1
    # add the time_step axis back
    final_output = expand_dims(outputs[0], 1)
    last_output = outputs[0]
    while(i < len(outputs)):
        #add the time_step axis back
        output_slice = expand_dims(outputs[i], 1)
        final_output = C.splice(final_output, output_slice, axis=time_axis)
        last_output = outputs[i]
        i+=1

    return last_output, final_output, states

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    shape = int_shape(inputs)
    dims = len(shape)

    if (dims < 3):
        raise ValueError('Input should be at least 3D.')

    if (has_seq_axis(inputs) == False):
        return _static_rnn(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length)

    #recurrent on seq axis, not fully implemented
    if (mask != None):
        raise NotImplementedError;
    if (unroll):
        raise  NotImplementedError;
    if (go_backwards):
        raise  NotImplementedError;

    if constants is None:
        constants = []

    #default for test, no unroll, no go_backwards, no mask
    states = tuple(initial_states)

    #todo: not finished
    def _recurrence(x, states):
        past_values = []
        for s in states:
            past_values.append(C.past_value(s))
        new_output, new_states = step_function(x, tuple(past_values) + tuple(constants))
        n_s = []
        for o, p in zip(new_states, states):
            n_s.append(o.replace_placeholders({p:o.output}))
        new_ouptut = n_s[0]
        return new_ouptut, n_s

    final_output, final_states = _recurrence(inputs, states)
    last_output = C.sequence.last(final_output)
    last_states = final_states

    return last_output, final_output, last_states

def has_seq_axis(x):
    return hasattr(x, 'dynamic_axes') and len(x.dynamic_axes) > 1
    shape = int_shape(x)
    return len(shape) > 1 and shape[1] == None


def l2_normalize(x, axis):
    axis=[axis]
    axis = _normalize_axis(axis, int_shape(x))
    norm = C.sqrt(C.reduce_sum(C.square(x), axis=axis[0]))
    return x / norm

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
    #todo
    return sigmoid(x)

def conv1d(x, kernel, stride=1, border_mode='valid', image_shape=None, filter_shape=None):
    """1D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: stride integer.
        border_mode: string, `"same"` or `"valid"`.

    # Returns
        A tensor, result of 1D convolution.
    """
    # pre-process dtype
    padding = _preprocess_border_mode(border_mode)
    stride=[stride]
    x = C.convolution(kernel, x, strides=tuple(stride), auto_padding=[padding, False])
    return x

def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering='default',
           image_shape=None, filter_shape=None, filter_dilation=(1, 1)):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    if filter_dilation == (1, 1):
        strides = (1,) + strides
        x = C.convolution(kernel, x, strides, auto_padding=[padding, padding, False])
    else:
        assert filter_dilation[0] == filter_dilation[1]
        assert strides == (1, 1), 'Invalid strides for dilated convolution'
        x = C.convolution(kernel, x, strides=filter_dilation[0], auto_padding=[padding, padding, False])
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
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)
    strides = strides + (1,)

    x = C.convolution(kernel, x, strides, auto_padding=[padding, padding, padding, False])
    return _postprocess_conv3d_output(x, dim_ordering)

def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='default',
           pool_mode='max'):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    strides = strides
    pool_size = pool_size
    x = _preprocess_conv2d_input(x, dim_ordering)
    if pool_mode == 'max':
        x = C.pooling(x, C.MAX_POOLING, pool_size, strides, auto_padding=[padding])
    elif pool_mode == 'avg':
        x = C.pooling(x, C.AVG_POOLING, pool_size, strides, auto_padding=[padding])
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))
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
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)

    x = _preprocess_conv3d_input(x, dim_ordering)

    x = _preprocess_conv2d_input(x, dim_ordering)
    if pool_mode == 'max':
        x = C.pooling(x, C.MAX_POOLING, pool_size, strides, auto_padding=[padding])
    elif pool_mode == 'avg':
        x = C.pooling(x, C.AVG_POOLING, pool_size, strides, auto_padding=[padding])
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    return _postprocess_conv3d_output(x, dim_ordering)


def relu(x, alpha=0., max_value=None):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    if alpha != 0.:
        negative_part = C.relu(-x)
    x = C.relu(x)
    if max_value is not None:
        x = C.clip(x, 0.0, max_value)
    if alpha != 0.:
        x -= alpha * negative_part
    return x

def get_outputs(x):
    if isinstance(x, C.cntk_py.Function):
        return x.output
    else:
        return x

def dropout(x, level, noise_shape=None, seed=None):
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    return C.dropout(x, level)

def batch_flatten(x):
    '''Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    '''
    dim = np.prod(x.shape)
    x = C.reshape(x, (dim))
    x._keras_shape = (None, dim)
    return x

def softmax(x):
    '''Softmax of a tensor.
    '''
    return C.softmax(x)

def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    raise NotImplementedError

def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return x / (1+C.abs(x))

def cross_entropy_with_softmax(x, y):
    return C.cross_entropy_with_softmax(x, y)

def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        return C.cross_entropy_with_softmax(output, target)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= C.reduce_sum(output, axis=-1)  # Not sure about this
        # avoid numerical instability with _EPSILON clipping
        output = C.clip(output, _EPSILON, 1.0 - _EPSILON)
        return -C.reduce_sum(target * C.log(output))

def sparse_categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy between an output tensor
    and a target tensor, where the target is an integer tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    target = C.ops.one_hot_op(target, output.shape[-1])
    target = C.reshape(target, output.shape)
    return categorical_crossentropy(output, target, from_logits)

class Function(object):
    '''hijack
    this is a temp solution, just want to confirm it works or not.
    '''
    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.placeholders= inputs
        self.grad_func = C.combine([outputs[0]])
        self.outputs = [f.output for f in outputs]
        self.combined_func = C.combine(outputs)

        self.update_placehoders = []
        self.update_params = {}
        for update in updates:
            if isinstance(update, tuple):
                if len(update) != 2:
                    raise  NotImplementedError
                f = update[1]
                new_p = update[0]
            else:
                f = update["function"]
                new_p = update["variable"]

            if hasattr(new_p, "is_placeholder") and new_p.is_placeholder == False:
                self.update_params[new_p] = f
                for input in f.arguments:
                    self.update_placehoders.append(input)
        list = [f for f in self.update_params.values()]
        if (len(list) > 0):
            self.update_combined_func = C.combine(list)
        else:
            self.update_combined_func = None


    def _apply_update(self, grads_dict):

        if self.update_combined_func is not None:
            input_dict = {}
            for input in self.update_placehoders:
                if input in grads_dict:
                    input_dict[input] = grads_dict[input]

            new_param_result = self.update_combined_func.eval(input_dict, as_numpy=_DEBUG)
            if (len(new_param_result) == 1):
                value = new_param_result[0]
                if _DEBUG:
                    value = value.reshape(value.shape[2:])
                    self.update_params.keys()[0].value = value.astype(np.float32)
                else:
                    shape = list(value.data().shape().dimensions())
                    shape = shape[2:]
                    shape = tuple(shape)
                    value = value.data().as_shape(shape)
                    self.update_params.keys()[0].value = value

            for new_p in self.update_params:
                value = new_param_result[self.update_params[new_p].output]
                if _DEBUG:
                    value = value.reshape(value.shape[2:])
                    new_p.value = value.astype(np.float32)
                else:
                    shape = list(value.data().shape().dimensions())
                    shape = shape[2:]
                    shape = tuple(shape)
                    value = value.data().as_shape(shape)
                    new_p.value = value

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        feed_dict = {}
        for tensor, value in zip(self.placeholders, inputs):
            feed_dict[tensor] = value
        updated = []
        '''the temp solution assume the first function is the loss function
        which is used to calculate the gradient. the reset function is for metrics
        '''
        if self.update_combined_func is not None:
            grad_func = self.grad_func
            parameter_list = []
            for parameter in grad_func.parameters:
                if (parameter.needs_gradient):
                    parameter_list.append(parameter)

            grads = grad_func.grad(feed_dict, parameter_list, as_numpy=_DEBUG)

            grads_dict = {}
            for param in grads:
                if param in grad_placeholder_dict:
                    if _DEBUG == False:
                        g = grads[param]
                        shape = list(g.data().shape().dimensions())
                        shape.insert(0,1)
                        shape.insert(0,1)
                        shape = tuple(shape)
                        g = g.data().as_shape(shape)
                        g = C.cntk_py.Value(g)
                    else:
                        g = [grads[param]]
                    grads_dict[grad_placeholder_dict[param]] = g

            self._apply_update(grads_dict)

        r = self.combined_func.eval(feed_dict)
        if (len(self.outputs) == 1):
            updated.append(r[0])
        else:
            updated.append(r[self.outputs[0]])
            i = 1
            while (i < len(self.outputs)):
                val = r[self.outputs[i]]
                updated.append(val)
                i += 1
        return updated

def function(inputs, outputs, updates=[], **kwargs):
    return Function(inputs, outputs, updates=updates, **kwargs)

def temporal_padding(x, padding=1):
    """Pads the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    # Returns
        A padded 3D tensor.
    """
    raise NotImplementedError

def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    """Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.

    # Returns
        A padded 3D tensor.
    """
    raise NotImplementedError

def spatial_2d_padding(x, padding=(1, 1), dim_ordering='default'):
    """Pads the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.

    # Returns
        A padded 4D tensor.
    """
    raise NotImplementedError

def asymmetric_spatial_2d_padding(x, top_pad=1, bottom_pad=1,
                                  left_pad=1, right_pad=1,
                                  dim_ordering='default'):
    """Pad the rows and columns of a 4D tensor
    with "top_pad", "bottom_pad", "left_pad", "right_pad" (resp.) zeros
    rows on top, bottom; cols on left, right.

    # Returns
        A padded 4D tensor.
    """
    raise NotImplementedError

def spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='default'):
    """Pads 5D tensor with zeros for the depth, height, width dimension with
    "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right

    For 'tf' dim_ordering, the 2nd, 3rd and 4th dimension will be padded.
    For 'th' dim_ordering, the 3rd, 4th and 5th dimension will be padded.

    # Returns
        A padded 5D tensor.
    """
    raise NotImplementedError

def one_hot(indices, nb_classes):
    """Input: nD integer tensor of shape `(batch_size, dim1, dim2, ... dim(n-1))`
    Output: (n + 1)D one hot representation of the input
    with shape `(batch_size, dim1, dim2, ... dim(n-1), nb_classes)`

    # Returns
        The one-hot tensor.
    """
    raise NotImplementedError

def reverse(x, axes):
    """Reverse a tensor along the the specified axes

    # Returns
        A tensor.
    """
    raise NotImplementedError

def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    if (isinstance(x, C.ops.variables.Parameter)):
        return x.value
    else:
        return eval(x)

def batch_get_value(xs):
    """Returns the value of more than one tensor variable.

    # Arguments
        x: list of variables.

    # Returns
        A list of Numpy arrays.
    """
    result = []
    for x in xs:
        if (isinstance(x, C.ops.variables.Parameter)):
            result.append(x.value)
        else:
            result.append(eval(x))
    return result

def set_value(x, value):
    """Sets the value of a variable,
    from a Numpy array. It returns `None`.
    """
    if (isinstance(x, C.ops.variables.Parameter)):
        x.value = value
    else:
        raise NotImplementedError

def print_tensor(x, message=''):
    """Print the message and the tensor when evaluated and return the same
    tensor.
    """
    raise NotImplementedError

def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.
    It returns `None`.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    for t in tuples:
        x = t[0]
        value = t[1]
        if isinstance(value, np.ndarray) == False:
            value = np.asarray(value)
        if (isinstance(x, C.ops.variables.Parameter)):
            x.value = value
        else:
            raise NotImplementedError

def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    raise NotImplementedError

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
    return C.element_select(condition, then_expression, else_expression)

def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    # Returns
        A tensor.
    """
    raise NotImplementedError

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
    raise NotImplementedError

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
    """
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    kernel = C.transpose(kernel, 0, 1)
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides
    #cntk output_shape does not include batch axis
    output_shape = output_shape[1:]

    x = C.convolution(kernel, x, strides, auto_padding=[padding, padding, False], transpose=True, output_shape=output_shape)
    return _postprocess_conv2d_output(x, dim_ordering)

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
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError


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
    raise NotImplementedError


def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = C.transpose(x, 0, 2)
        x = C.transpose(x, 1, 2)
    return x

def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = C.transpose(kernel, 0, 3)
        kernel = C.transpose(kernel, 1, 2)
        kernel = C.transpose(kernel, 2, 3)
    return kernel

def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = True
    elif border_mode == 'valid':
        padding = False
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    return padding

def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'tf':
        x = C.transpose(x, 0, 2)
        x = C.transpose(x, 0, 1)

    return x

def _preprocess_conv3d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        x = C.transpose(x, 2, 3)
        x = C.transpose(x, 1, 2)
        x = C.transpose(x, 0, 1)
    return x


def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        kernel = C.transpose(kernel, 0, 4)
        kernel = C.transpose(kernel, 1, 3)
        kernel = C.transpose(kernel, 3, 4)
    return kernel


def _postprocess_conv3d_output(x, dim_ordering):
    if dim_ordering == 'tf':
        x = C.transpose(x, 2, 3)
        x = C.transpose(x, 1, 2)
        x = C.transpose(x, 0, 1)
    return x

def create_placeholder(ndim):
    return [C.placeholder_variable(C.InferredDimension) for i in range(ndim)]

def slice(x, start, end):
    return C.ops.slice(x,-1,start,end)

def conv2d_bias_add(bias, num_filter, dim_ordering):
    if dim_ordering == 'th':
        return reshape(bias, (num_filter, 1, 1))
    elif dim_ordering == 'tf':
        return reshape(bias, (1, 1, num_filter))
    else:
        raise Exception('Invalid dim_ordering: ', dim_ordering)

def conv1d_bias_add(bias, num_filter):
        return reshape(bias, (1, num_filter))

def _get_dynamic_axis_num(x):
    if hasattr(x, 'dynamic_axes'):
        return len(x.dynamic_axes)
    else:
        return 0
