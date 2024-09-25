from openvino.runtime import opset14

from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import ov_to_keras_type


def _align_operand_types(x1, x2, op_name):
    x1_type = x1.element_type
    x2_type = x2.element_type
    if x1_type.is_dynamic() or x2_type.is_dynamic():
        raise ValueError(
            f"'{op_name}' operation is not supported for dynamic operand type with openvino backend"
        )
    x1_type = ov_to_keras_type(x1_type)
    x2_type = ov_to_keras_type(x2_type)
    result_type = dtypes.result_type(x1_type, x2_type)
    result_type = OPENVINO_DTYPES[result_type]
    if x1_type != result_type:
        x1 = opset14.convert(x1, result_type)
    if x2_type != result_type:
        x2 = opset14.convert(x2, result_type)
    return x1, x2


def add(x1, x2):
    x1, x2 = _align_operand_types(x1, x2, "add()")
    return opset14.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    raise NotImplementedError(
        "`einsum` is not supported with openvino backend"
    )


def subtract(x1, x2):
    x1, x2 = _align_operand_types(x1, x2, "subtract()")
    return opset14.subtract(x1, x2)


def matmul(x1, x2):
    raise NotImplementedError(
        "`matmul` is not supported with openvino backend"
    )


def multiply(x1, x2):
    x1, x2 = _align_operand_types(x1, x2, "multiply()")
    return opset14.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`mean` is not supported with openvino backend"
    )


def max(x, axis=None, keepdims=False, initial=None):
    raise NotImplementedError(
        "`max` is not supported with openvino backend"
    )


def ones(shape, dtype=None):
    raise NotImplementedError(
        "`ones` is not supported with openvino backend"
    )


def zeros(shape, dtype=None):
    raise NotImplementedError(
        "`zeros` is not supported with openvino backend"
    )


def absolute(x):
    return opset14.absolute(x)


def abs(x):
    return opset14.absolute(x)


def all(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`all` is not supported with openvino backend"
    )


def any(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`any` is not supported with openvino backend"
    )


def amax(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`amax` is not supported with openvino backend"
    )


def amin(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`amin` is not supported with openvino backend"
    )


def append(x1, x2, axis=None):
    raise NotImplementedError(
        "`append` is not supported with openvino backend"
    )


def arange(start, stop=None, step=None, dtype=None):
    raise NotImplementedError(
        "`arange` is not supported with openvino backend"
    )


def arccos(x):
    raise NotImplementedError(
        "`arccos` is not supported with openvino backend"
    )


def arccosh(x):
    raise NotImplementedError(
        "`arccosh` is not supported with openvino backend"
    )


def arcsin(x):
    raise NotImplementedError(
        "`arcsin` is not supported with openvino backend"
    )


def arcsinh(x):
    raise NotImplementedError(
        "`arcsinh` is not supported with openvino backend"
    )


def arctan(x):
    raise NotImplementedError(
        "`arctan` is not supported with openvino backend"
    )


def arctan2(x1, x2):
    raise NotImplementedError(
        "`arctan2` is not supported with openvino backend"
    )


def arctanh(x):
    raise NotImplementedError(
        "`arctanh` is not supported with openvino backend"
    )


def argmax(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`argmax` is not supported with openvino backend"
    )


def argmin(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`argmin` is not supported with openvino backend"
    )


def argsort(x, axis=-1):
    raise NotImplementedError(
        "`argsort` is not supported with openvino backend"
    )


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def average(x, axis=None, weights=None):
    raise NotImplementedError(
        "`average` is not supported with openvino backend"
    )


def bincount(x, weights=None, minlength=0, sparse=False):
    raise NotImplementedError(
        "`bincount` is not supported with openvino backend"
    )


def broadcast_to(x, shape):
    raise NotImplementedError(
        "`broadcast_to` is not supported with openvino backend"
    )


def ceil(x):
    raise NotImplementedError(
        "`ceil` is not supported with openvino backend"
    )


def clip(x, x_min, x_max):
    raise NotImplementedError(
        "`clip` is not supported with openvino backend"
    )


def concatenate(xs, axis=0):
    raise NotImplementedError(
        "`concatenate` is not supported with openvino backend"
    )


def conjugate(x):
    raise NotImplementedError(
        "`conjugate` is not supported with openvino backend"
    )


def conj(x):
    raise NotImplementedError(
        "`conj` is not supported with openvino backend"
    )


def copy(x):
    raise NotImplementedError(
        "`copy` is not supported with openvino backend"
    )


def cos(x):
    raise NotImplementedError(
        "`cos` is not supported with openvino backend"
    )


def cosh(x):
    raise NotImplementedError(
        "`cosh` is not supported with openvino backend"
    )


def count_nonzero(x, axis=None):
    raise NotImplementedError(
        "`count_nonzero` is not supported with openvino backend"
    )


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    raise NotImplementedError(
        "`cross` is not supported with openvino backend"
    )


def cumprod(x, axis=None, dtype=None):
    raise NotImplementedError(
        "`cumprod` is not supported with openvino backend"
    )


def cumsum(x, axis=None, dtype=None):
    raise NotImplementedError(
        "`cumsum` is not supported with openvino backend"
    )


def diag(x, k=0):
    raise NotImplementedError(
        "`diag` is not supported with openvino backend"
    )


def diagonal(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError(
        "`diagonal` is not supported with openvino backend"
    )


def diff(a, n=1, axis=-1):
    raise NotImplementedError(
        "`diff` is not supported with openvino backend"
    )


def digitize(x, bins):
    raise NotImplementedError(
        "`digitize` is not supported with openvino backend"
    )


def dot(x, y):
    raise NotImplementedError(
        "`dot` is not supported with openvino backend"
    )


def empty(shape, dtype=None):
    raise NotImplementedError(
        "`empty` is not supported with openvino backend"
    )


def equal(x1, x2):
    raise NotImplementedError(
        "`equal` is not supported with openvino backend"
    )


def exp(x):
    raise NotImplementedError(
        "`exp` is not supported with openvino backend"
    )


def expand_dims(x, axis):
    raise NotImplementedError(
        "`expand_dims` is not supported with openvino backend"
    )


def expm1(x):
    raise NotImplementedError(
        "`expm1` is not supported with openvino backend"
    )


def flip(x, axis=None):
    raise NotImplementedError(
        "`flip` is not supported with openvino backend"
    )


def floor(x):
    raise NotImplementedError(
        "`floor` is not supported with openvino backend"
    )


def full(shape, fill_value, dtype=None):
    raise NotImplementedError(
        "`full` is not supported with openvino backend"
    )


def full_like(x, fill_value, dtype=None):
    raise NotImplementedError(
        "`full_like` is not supported with openvino backend"
    )


def greater(x1, x2):
    raise NotImplementedError(
        "`greater` is not supported with openvino backend"
    )


def greater_equal(x1, x2):
    raise NotImplementedError(
        "`greater_equal` is not supported with openvino backend"
    )


def hstack(xs):
    raise NotImplementedError(
        "`hstack` is not supported with openvino backend"
    )


def identity(n, dtype=None):
    raise NotImplementedError(
        "`identity` is not supported with openvino backend"
    )


def imag(x):
    raise NotImplementedError(
        "`imag` is not supported with openvino backend"
    )


def isclose(x1, x2):
    raise NotImplementedError(
        "`isclose` is not supported with openvino backend"
    )


def isfinite(x):
    raise NotImplementedError(
        "`isfinite` is not supported with openvino backend"
    )


def isinf(x):
    raise NotImplementedError(
        "`isinf` is not supported with openvino backend"
    )


def isnan(x):
    raise NotImplementedError(
        "`isnan` is not supported with openvino backend"
    )


def less(x1, x2):
    raise NotImplementedError(
        "`less` is not supported with openvino backend"
    )


def less_equal(x1, x2):
    raise NotImplementedError(
        "`less_equal` is not supported with openvino backend"
    )


def linspace(
        start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    raise NotImplementedError(
        "`linspace` is not supported with openvino backend"
    )


def log(x):
    raise NotImplementedError(
        "`log` is not supported with openvino backend"
    )


def log10(x):
    raise NotImplementedError(
        "`log10` is not supported with openvino backend"
    )


def log1p(x):
    raise NotImplementedError(
        "`log1p` is not supported with openvino backend"
    )


def log2(x):
    raise NotImplementedError(
        "`log2` is not supported with openvino backend"
    )


def logaddexp(x1, x2):
    raise NotImplementedError(
        "`logaddexp` is not supported with openvino backend"
    )


def logical_and(x1, x2):
    raise NotImplementedError(
        "`logical_and` is not supported with openvino backend"
    )


def logical_not(x):
    raise NotImplementedError(
        "`logical_not` is not supported with openvino backend"
    )


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
    raise NotImplementedError(
        "`median` is not supported with openvino backend"
    )


def meshgrid(*x, indexing="xy"):
    raise NotImplementedError(
        "`meshgrid` is not supported with openvino backend"
    )


def min(x, axis=None, keepdims=False, initial=None):
    raise NotImplementedError(
        "`min` is not supported with openvino backend"
    )


def minimum(x1, x2):
    raise NotImplementedError(
        "`minimum` is not supported with openvino backend"
    )


def mod(x1, x2):
    raise NotImplementedError(
        "`mod` is not supported with openvino backend"
    )


def moveaxis(x, source, destination):
    raise NotImplementedError(
        "`moveaxis` is not supported with openvino backend"
    )


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    raise NotImplementedError(
        "`nan_to_num` is not supported with openvino backend"
    )


def ndim(x):
    raise NotImplementedError(
        "`ndim` is not supported with openvino backend"
    )


def nonzero(x):
    raise NotImplementedError(
        "`nonzero` is not supported with openvino backend"
    )


def not_equal(x1, x2):
    raise NotImplementedError(
        "`not_equal` is not supported with openvino backend"
    )


def zeros_like(x, dtype=None):
    raise NotImplementedError(
        "`zeros_like` is not supported with openvino backend"
    )


def ones_like(x, dtype=None):
    raise NotImplementedError(
        "`ones_like` is not supported with openvino backend"
    )


def outer(x1, x2):
    raise NotImplementedError(
        "`outer` is not supported with openvino backend"
    )


def pad(x, pad_width, mode="constant", constant_values=None):
    raise NotImplementedError(
        "`pad` is not supported with openvino backend"
    )


def prod(x, axis=None, keepdims=False, dtype=None):
    raise NotImplementedError(
        "`prod` is not supported with openvino backend"
    )


def quantile(x, q, axis=None, method="linear", keepdims=False):
    raise NotImplementedError(
        "`quantile` is not supported with openvino backend"
    )


def ravel(x):
    raise NotImplementedError(
        "`ravel` is not supported with openvino backend"
    )


def real(x):
    raise NotImplementedError(
        "`real` is not supported with openvino backend"
    )


def reciprocal(x):
    raise NotImplementedError(
        "`reciprocal` is not supported with openvino backend"
    )


def repeat(x, repeats, axis=None):
    raise NotImplementedError(
        "`repeat` is not supported with openvino backend"
    )


def reshape(x, newshape):
    raise NotImplementedError(
        "`reshape` is not supported with openvino backend"
    )


def roll(x, shift, axis=None):
    raise NotImplementedError(
        "`roll` is not supported with openvino backend"
    )


def sign(x):
    raise NotImplementedError(
        "`sign` is not supported with openvino backend"
    )


def sin(x):
    raise NotImplementedError(
        "`sin` is not supported with openvino backend"
    )


def sinh(x):
    raise NotImplementedError(
        "`sinh` is not supported with openvino backend"
    )


def size(x):
    raise NotImplementedError(
        "`size` is not supported with openvino backend"
    )


def sort(x, axis=-1):
    raise NotImplementedError(
        "`sort` is not supported with openvino backend"
    )


def split(x, indices_or_sections, axis=0):
    raise NotImplementedError(
        "`split` is not supported with openvino backend"
    )


def stack(x, axis=0):
    raise NotImplementedError(
        "`stack` is not supported with openvino backend"
    )


def std(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`std` is not supported with openvino backend"
    )


def swapaxes(x, axis1, axis2):
    raise NotImplementedError(
        "`swapaxes` is not supported with openvino backend"
    )


def take(x, indices, axis=None):
    raise NotImplementedError(
        "`take` is not supported with openvino backend"
    )


def take_along_axis(x, indices, axis=None):
    raise NotImplementedError(
        "`take_along_axis` is not supported with openvino backend"
    )


def tan(x):
    raise NotImplementedError(
        "`tan` is not supported with openvino backend"
    )


def tanh(x):
    raise NotImplementedError(
        "`tanh` is not supported with openvino backend"
    )


def tensordot(x1, x2, axes=2):
    raise NotImplementedError(
        "`tensordot` is not supported with openvino backend"
    )


def round(x, decimals=0):
    raise NotImplementedError(
        "`round` is not supported with openvino backend"
    )


def tile(x, repeats):
    raise NotImplementedError(
        "`tile` is not supported with openvino backend"
    )


def trace(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError(
        "`trace` is not supported with openvino backend"
    )


def tri(N, M=None, k=0, dtype=None):
    raise NotImplementedError(
        "`tri` is not supported with openvino backend"
    )


def tril(x, k=0):
    raise NotImplementedError(
        "`tril` is not supported with openvino backend"
    )


def triu(x, k=0):
    raise NotImplementedError(
        "`triu` is not supported with openvino backend"
    )


def vdot(x1, x2):
    raise NotImplementedError(
        "`vdot` is not supported with openvino backend"
    )


def vstack(xs):
    raise NotImplementedError(
        "`vstack` is not supported with openvino backend"
    )


def vectorize(pyfunc, *, excluded=None, signature=None):
    raise NotImplementedError(
        "`vectorize` is not supported with openvino backend"
    )


def where(condition, x1, x2):
    raise NotImplementedError(
        "`where` is not supported with openvino backend"
    )


def divide(x1, x2):
    x1, x2 = _align_operand_types(x1, x2)
    return opset14.divide(x1, x2)


def divide_no_nan(x1, x2):
    raise NotImplementedError(
        "`divide_no_nan` is not supported with openvino backend"
    )


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    x1, x2 = _align_operand_types(x1, x2)
    return opset14.power(x1, x2)


def negative(x):
    return opset14.negative(x)


def square(x):
    raise NotImplementedError(
        "`square` is not supported with openvino backend"
    )


def sqrt(x):
    raise NotImplementedError(
        "`sqrt` is not supported with openvino backend"
    )


def squeeze(x, axis=None):
    raise NotImplementedError(
        "`squeeze` is not supported with openvino backend"
    )


def transpose(x, axes=None):
    raise NotImplementedError(
        "`transpose` is not supported with openvino backend"
    )


def var(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`var` is not supported with openvino backend"
    )


def sum(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`sum` is not supported with openvino backend"
    )


def eye(N, M=None, k=0, dtype=None):
    raise NotImplementedError(
        "`eye` is not supported with openvino backend"
    )


def floor_divide(x1, x2):
    raise NotImplementedError(
        "`floor_divide` is not supported with openvino backend"
    )


def logical_xor(x1, x2):
    return opset14.logical_xor(x1, x2)


def correlate(x1, x2, mode="valid"):
    raise NotImplementedError(
        "`correlate` is not supported with openvino backend"
    )


def select(condlist, choicelist, default=0):
    raise NotImplementedError(
        "`select` is not supported with openvino backend"
    )


def slogdet(x):
    raise NotImplementedError(
        "`slogdet` is not supported with openvino backend"
    )


def argpartition(x, kth, axis=-1):
    raise NotImplementedError(
        "`argpartition` is not supported with openvino backend"
    )
