import numpy as np
import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import ov_to_keras_type


def _align_operand_types(x1, x2, op_name):
    x1_type = x1.element_type
    x2_type = x2.element_type
    if x1_type.is_dynamic() or x2_type.is_dynamic():
        raise ValueError(
            f"'{op_name}' operation is not supported for dynamic operand type "
            "with openvino backend"
        )
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    result_type = dtypes.result_type(x1_type, x2_type)
    result_type = OPENVINO_DTYPES[result_type]
    if x1_type != result_type:
        x1 = ov_opset.convert(x1, result_type)
    if x2_type != result_type:
        x2 = ov_opset.convert(x2, result_type)
    return x1, x2


def add(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "add()")
    return OpenVINOKerasTensor(ov_opset.add(x1, x2).output(0))


def einsum(subscripts, *operands, **kwargs):
    inputs = []
    for operand in operands:
        operand = get_ov_output(operand)
        inputs.append(operand)
    return OpenVINOKerasTensor(ov_opset.einsum(inputs, subscripts).output(0))


def subtract(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "subtract()")
    return OpenVINOKerasTensor(ov_opset.subtract(x1, x2).output(0))


def matmul(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "matmul()")
    return OpenVINOKerasTensor(ov_opset.matmul(x1, x2, False, False).output(0))


def multiply(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "multiply()")
    return OpenVINOKerasTensor(ov_opset.multiply(x1, x2).output(0))


def mean(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    axis_const = ov_opset.constant(axis, dtype=Type.i32)
    mean_ops = ov_opset.reduce_mean(x, axis_const, keepdims)
    return OpenVINOKerasTensor(mean_ops.output(0))


def max(x, axis=None, keepdims=False, initial=None):
    raise NotImplementedError("`max` is not supported with openvino backend")


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    const_one = ov_opset.constant(1, dtype=ov_type).output(0)
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    output_shape = ov_opset.constant(shape, dtype=Type.i32).output(0)
    ones = ov_opset.broadcast(const_one, output_shape)
    return OpenVINOKerasTensor(ones.output(0))


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    const_zero = ov_opset.constant(0, dtype=ov_type).output(0)
    if isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, int):
        shape = [shape]
    output_shape = ov_opset.constant(shape, dtype=Type.i32).output(0)
    zeros = ov_opset.broadcast(const_zero, output_shape)
    return OpenVINOKerasTensor(zeros.output(0))


def absolute(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def abs(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.absolute(x).output(0))


def all(x, axis=None, keepdims=False):
    raise NotImplementedError("`all` is not supported with openvino backend")


def any(x, axis=None, keepdims=False):
    raise NotImplementedError("`any` is not supported with openvino backend")


def amax(x, axis=None, keepdims=False):
    raise NotImplementedError("`amax` is not supported with openvino backend")


def amin(x, axis=None, keepdims=False):
    raise NotImplementedError("`amin` is not supported with openvino backend")


def append(x1, x2, axis=None):
    raise NotImplementedError("`append` is not supported with openvino backend")


def arange(start, stop=None, step=None, dtype=None):
    raise NotImplementedError("`arange` is not supported with openvino backend")


def arccos(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.acos(x).output(0))


def arccosh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.acosh(x).output(0))


def arcsin(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.asin(x).output(0))


def arcsinh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.asinh(x).output(0))


def arctan(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.atan(x).output(0))


def arctan2(x1, x2):
    raise NotImplementedError(
        "`arctan2` is not supported with openvino backend"
    )


def arctanh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.atanh(x).output(0))


def argmax(x, axis=None, keepdims=False):
    raise NotImplementedError("`argmax` is not supported with openvino backend")


def argmin(x, axis=None, keepdims=False):
    raise NotImplementedError("`argmin` is not supported with openvino backend")


def argsort(x, axis=-1):
    raise NotImplementedError(
        "`argsort` is not supported with openvino backend"
    )


def array(x, dtype=None):
    if dtype is not None:
        return np.array(x, dtype=dtype)
    return np.array(x)


def average(x, axis=None, weights=None):
    raise NotImplementedError(
        "`average` is not supported with openvino backend"
    )


def bincount(x, weights=None, minlength=0, sparse=False):
    raise NotImplementedError(
        "`bincount` is not supported with openvino backend"
    )


def broadcast_to(x, shape):
    assert isinstance(
        shape, (tuple, list)
    ), "`broadcast_to` is supported only for tuple and list `shape`"
    target_shape = ov_opset.constant(list(shape), Type.i32).output(0)
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.broadcast(x, target_shape).output(0))


def ceil(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.ceil(x).output(0))


def clip(x, x_min, x_max):
    x = get_ov_output(x)
    x_min = get_ov_output(x_min, x.get_element_type())
    x_max = get_ov_output(x_max, x.get_element_type())
    clip_by_min = ov_opset.maximum(x, x_min).output(0)
    clip_by_max = ov_opset.minimum(clip_by_min, x_max).output(0)
    return OpenVINOKerasTensor(clip_by_max)


def concatenate(xs, axis=0):
    raise NotImplementedError(
        "`concatenate` is not supported with openvino backend"
    )


def conjugate(x):
    raise NotImplementedError(
        "`conjugate` is not supported with openvino backend"
    )


def conj(x):
    raise NotImplementedError("`conj` is not supported with openvino backend")


def copy(x):
    return x


def cos(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.cos(x).output(0))


def cosh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.cosh(x).output(0))


def count_nonzero(x, axis=None):
    raise NotImplementedError(
        "`count_nonzero` is not supported with openvino backend"
    )


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    raise NotImplementedError("`cross` is not supported with openvino backend")


def cumprod(x, axis=None, dtype=None):
    raise NotImplementedError(
        "`cumprod` is not supported with openvino backend"
    )


def cumsum(x, axis=None, dtype=None):
    x = get_ov_output(x)
    if dtype is not None:
        ov_type = OPENVINO_DTYPES[dtype]
        x = ov_opset.convert(x, ov_type).output(0)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.cumsum(x, axis).output(0))


def diag(x, k=0):
    raise NotImplementedError("`diag` is not supported with openvino backend")


def diagonal(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError(
        "`diagonal` is not supported with openvino backend"
    )


def diff(a, n=1, axis=-1):
    raise NotImplementedError("`diff` is not supported with openvino backend")


def digitize(x, bins):
    raise NotImplementedError(
        "`digitize` is not supported with openvino backend"
    )


def dot(x, y):
    raise NotImplementedError("`dot` is not supported with openvino backend")


def empty(shape, dtype=None):
    raise NotImplementedError("`empty` is not supported with openvino backend")


def equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "equal()")
    return OpenVINOKerasTensor(ov_opset.equal(x1, x2).output(0))


def exp(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.exp(x).output(0))


def expand_dims(x, axis):
    if isinstance(x, OpenVINOKerasTensor):
        x = x.output
    else:
        assert False
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.unsqueeze(x, axis).output(0))


def expm1(x):
    raise NotImplementedError("`expm1` is not supported with openvino backend")


def flip(x, axis=None):
    raise NotImplementedError("`flip` is not supported with openvino backend")


def floor(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.floor(x).output(0))


def full(shape, fill_value, dtype=None):
    raise NotImplementedError("`full` is not supported with openvino backend")


def full_like(x, fill_value, dtype=None):
    raise NotImplementedError(
        "`full_like` is not supported with openvino backend"
    )


def greater(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "greater()")
    return OpenVINOKerasTensor(ov_opset.greater(x1, x2).output(0))


def greater_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "greater_equal()")
    return OpenVINOKerasTensor(ov_opset.greater_equal(x1, x2).output(0))


def hstack(xs):
    raise NotImplementedError("`hstack` is not supported with openvino backend")


def identity(n, dtype=None):
    raise NotImplementedError(
        "`identity` is not supported with openvino backend"
    )


def imag(x):
    raise NotImplementedError("`imag` is not supported with openvino backend")


def isclose(x1, x2):
    raise NotImplementedError(
        "`isclose` is not supported with openvino backend"
    )


def isfinite(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_finite(x).output(0))


def isinf(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_inf(x).output(0))


def isnan(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.is_nan(x).output(0))


def less(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "less()")
    return OpenVINOKerasTensor(ov_opset.less(x1, x2).output(0))


def less_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "less_equal()")
    return OpenVINOKerasTensor(ov_opset.less_equal(x1, x2).output(0))


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    raise NotImplementedError(
        "`linspace` is not supported with openvino backend"
    )


def log(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.log(x).output(0))


def log10(x):
    raise NotImplementedError("`log10` is not supported with openvino backend")


def log1p(x):
    raise NotImplementedError("`log1p` is not supported with openvino backend")


def log2(x):
    raise NotImplementedError("`log2` is not supported with openvino backend")


def logaddexp(x1, x2):
    raise NotImplementedError(
        "`logaddexp` is not supported with openvino backend"
    )


def logical_and(x1, x2):
    raise NotImplementedError(
        "`logical_and` is not supported with openvino backend"
    )


def logical_not(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.logical_not(x).output(0))


def logical_or(x1, x2):
    raise NotImplementedError(
        "`logical_or` is not supported with openvino backend"
    )


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    raise NotImplementedError(
        "`logspace` is not supported with openvino backend"
    )


def maximum(x1, x2):
    raise NotImplementedError(
        "`maximum` is not supported with openvino backend"
    )


def median(x, axis=None, keepdims=False):
    raise NotImplementedError("`median` is not supported with openvino backend")


def meshgrid(*x, indexing="xy"):
    raise NotImplementedError(
        "`meshgrid` is not supported with openvino backend"
    )


def min(x, axis=None, keepdims=False, initial=None):
    raise NotImplementedError("`min` is not supported with openvino backend")


def minimum(x1, x2):
    raise NotImplementedError(
        "`minimum` is not supported with openvino backend"
    )


def mod(x1, x2):
    raise NotImplementedError("`mod` is not supported with openvino backend")


def moveaxis(x, source, destination):
    raise NotImplementedError(
        "`moveaxis` is not supported with openvino backend"
    )


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    raise NotImplementedError(
        "`nan_to_num` is not supported with openvino backend"
    )


def ndim(x):
    raise NotImplementedError("`ndim` is not supported with openvino backend")


def nonzero(x):
    raise NotImplementedError(
        "`nonzero` is not supported with openvino backend"
    )


def not_equal(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "not_equal()")
    return OpenVINOKerasTensor(ov_opset.not_equal(x1, x2).output(0))


def zeros_like(x, dtype=None):
    raise NotImplementedError(
        "`zeros_like` is not supported with openvino backend"
    )


def ones_like(x, dtype=None):
    raise NotImplementedError(
        "`ones_like` is not supported with openvino backend"
    )


def outer(x1, x2):
    raise NotImplementedError("`outer` is not supported with openvino backend")


def pad(x, pad_width, mode="constant", constant_values=None):
    x = get_ov_output(x)
    pad_value = None
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        assert isinstance(constant_values, int), (
            "`pad` operation supports only scalar pad value "
            "in constant mode by openvino backend"
        )
        pad_value = constant_values

    # split pad_width into two tensors pads_begin and pads_end
    pads_begin = []
    pads_end = []
    for pads_pair in pad_width:
        pads_begin.append(pads_pair[0])
        pads_end.append(pads_pair[1])
    pads_begin = ov_opset.constant(pads_begin, Type.i32).output(0)
    pads_end = ov_opset.constant(pads_end, Type.i32).output(0)
    return OpenVINOKerasTensor(
        ov_opset.pad(x, pads_begin, pads_end, mode, pad_value).output(0)
    )


def prod(x, axis=None, keepdims=False, dtype=None):
    raise NotImplementedError("`prod` is not supported with openvino backend")


def quantile(x, q, axis=None, method="linear", keepdims=False):
    raise NotImplementedError(
        "`quantile` is not supported with openvino backend"
    )


def ravel(x):
    raise NotImplementedError("`ravel` is not supported with openvino backend")


def real(x):
    raise NotImplementedError("`real` is not supported with openvino backend")


def reciprocal(x):
    raise NotImplementedError(
        "`reciprocal` is not supported with openvino backend"
    )


def repeat(x, repeats, axis=None):
    raise NotImplementedError("`repeat` is not supported with openvino backend")


def reshape(x, newshape):
    x = get_ov_output(x)
    if isinstance(newshape, tuple):
        newshape = list(newshape)
    newshape = ov_opset.constant(newshape, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reshape(x, newshape, False).output(0))


def roll(x, shift, axis=None):
    raise NotImplementedError("`roll` is not supported with openvino backend")


def sign(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sign(x).output(0))


def sin(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sin(x).output(0))


def sinh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sinh(x).output(0))


def size(x):
    raise NotImplementedError("`size` is not supported with openvino backend")


def sort(x, axis=-1):
    raise NotImplementedError("`sort` is not supported with openvino backend")


def split(x, indices_or_sections, axis=0):
    raise NotImplementedError("`split` is not supported with openvino backend")


def stack(x, axis=0):
    raise NotImplementedError("`stack` is not supported with openvino backend")


def std(x, axis=None, keepdims=False):
    raise NotImplementedError("`std` is not supported with openvino backend")


def swapaxes(x, axis1, axis2):
    raise NotImplementedError(
        "`swapaxes` is not supported with openvino backend"
    )


def take(x, indices, axis=None):
    x = get_ov_output(x)
    indices = get_ov_output(indices)
    if axis is None:
        target_shape = ov_opset.constant([-1], dtype=Type.i32).output(0)
        x = ov_opset.reshape(x, target_shape).output(0)
        axis = ov_opset.constant(0, dtype=Type.i32).output(0)
    else:
        axis = ov_opset.constant(axis, dtype=Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.gather(x, indices, axis).output(0))


def take_along_axis(x, indices, axis=None):
    raise NotImplementedError(
        "`take_along_axis` is not supported with openvino backend"
    )


def tan(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.tan(x).output(0))


def tanh(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.tanh(x).output(0))


def tensordot(x1, x2, axes=2):
    raise NotImplementedError(
        "`tensordot` is not supported with openvino backend"
    )


def round(x, decimals=0):
    raise NotImplementedError("`round` is not supported with openvino backend")


def tile(x, repeats):
    raise NotImplementedError("`tile` is not supported with openvino backend")


def trace(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError("`trace` is not supported with openvino backend")


def tri(N, M=None, k=0, dtype=None):
    raise NotImplementedError("`tri` is not supported with openvino backend")


def tril(x, k=0):
    raise NotImplementedError("`tril` is not supported with openvino backend")


def triu(x, k=0):
    raise NotImplementedError("`triu` is not supported with openvino backend")


def vdot(x1, x2):
    raise NotImplementedError("`vdot` is not supported with openvino backend")


def vstack(xs):
    raise NotImplementedError("`vstack` is not supported with openvino backend")


def vectorize(pyfunc, *, excluded=None, signature=None):
    raise NotImplementedError(
        "`vectorize` is not supported with openvino backend"
    )


def where(condition, x1, x2):
    raise NotImplementedError("`where` is not supported with openvino backend")


def divide(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "divide()")
    return OpenVINOKerasTensor(ov_opset.divide(x1, x2).output(0))


def divide_no_nan(x1, x2):
    raise NotImplementedError(
        "`divide_no_nan` is not supported with openvino backend"
    )


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    element_type = None
    if isinstance(x1, OpenVINOKerasTensor):
        element_type = x1.output.get_element_type()
    if isinstance(x2, OpenVINOKerasTensor):
        element_type = x2.output.get_element_type()
    assert element_type is not None
    x1 = get_ov_output(x1, element_type)
    x2 = get_ov_output(x2, element_type)
    x1, x2 = _align_operand_types(x1, x2, "power()")
    return OpenVINOKerasTensor(ov_opset.power(x1, x2).output(0))


def negative(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.negative(x).output(0))


def square(x):
    raise NotImplementedError("`square` is not supported with openvino backend")


def sqrt(x):
    x = get_ov_output(x)
    return OpenVINOKerasTensor(ov_opset.sqrt(x).output(0))


def squeeze(x, axis=None):
    raise NotImplementedError(
        "`squeeze` is not supported with openvino backend"
    )


def transpose(x, axes=None):
    x = get_ov_output(x)
    axes = ov_opset.constant(axes, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.transpose(x, axes).output(0))


def var(x, axis=None, keepdims=False):
    raise NotImplementedError("`var` is not supported with openvino backend")


def sum(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    return OpenVINOKerasTensor(ov_opset.reduce_sum(x, axis, keepdims).output(0))


def eye(N, M=None, k=0, dtype=None):
    raise NotImplementedError("`eye` is not supported with openvino backend")


def floor_divide(x1, x2):
    raise NotImplementedError(
        "`floor_divide` is not supported with openvino backend"
    )


def logical_xor(x1, x2):
    x1 = get_ov_output(x1)
    x2 = get_ov_output(x2)
    return OpenVINOKerasTensor(ov_opset.logical_xor(x1, x2).output(0))


def correlate(x1, x2, mode="valid"):
    raise NotImplementedError(
        "`correlate` is not supported with openvino backend"
    )


def select(condlist, choicelist, default=0):
    raise NotImplementedError("`select` is not supported with openvino backend")


def slogdet(x):
    raise NotImplementedError(
        "`slogdet` is not supported with openvino backend"
    )


def argpartition(x, kth, axis=-1):
    raise NotImplementedError(
        "`argpartition` is not supported with openvino backend"
    )
