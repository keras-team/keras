import torch

from keras_core.backend.torch.core import cast
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import to_torch_dtype

TORCH_INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def add(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(operand) for operand in operands]
    return torch.einsum(subscripts, *operands)


def subtract(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.subtract(x1, x2)


def matmul(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.matmul(x1, x2)


def multiply(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # Conversion to float necessary for `torch.mean`
    x = cast(x, "float32") if x.dtype in TORCH_INT_TYPES else x
    return torch.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if axis is None:
        result = torch.max(x)
    else:
        if isinstance(axis, list):
            axis = axis[-1]
        result = torch.max(x, dim=axis, keepdim=keepdims)

    if isinstance(getattr(result, "values", None), torch.Tensor):
        result = result.values

    if initial is not None:
        return torch.maximum(result, initial)
    return result


def ones(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    return torch.ones(*shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    return torch.zeros(*shape, dtype=dtype)


def absolute(x):
    return abs(x)


def abs(x):
    x = convert_to_tensor(x)
    return torch.abs(x)


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is not None:
        if isinstance(axis, list):
            axis = axis[-1]
        return torch.all(x, dim=axis, keepdim=keepdims)
    else:
        return torch.all(x)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is not None:
        if isinstance(axis, list):
            axis = axis[-1]
        return torch.any(x, dim=axis, keepdim=keepdims)
    else:
        return torch.any(x)


def amax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is not None:
        return torch.amax(x, dim=axis, keepdim=keepdims)
    else:
        return torch.amax(x)


def amin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is not None:
        return torch.amin(x, dim=axis, keepdim=keepdims)
    else:
        return torch.amin(x)


def append(
    x1,
    x2,
    axis=None,
):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    if axis is None:
        return torch.cat((x1.flatten(), x2.flatten()))
    return torch.cat((x1, x2), dim=axis)


def arange(start, stop=None, step=None, dtype=None):
    dtype = to_torch_dtype(dtype)
    if stop is None:
        return torch.arange(start, step=step, dtype=dtype)
    step = step or 1
    return torch.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    x = convert_to_tensor(x)
    return torch.arccos(x)


def arcsin(x):
    x = convert_to_tensor(x)
    return torch.arcsin(x)


def arctan(x):
    x = convert_to_tensor(x)
    return torch.arctan(x)


def arctan2(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.arctan2(x1, x2)


def argmax(x, axis=None):
    x = convert_to_tensor(x)
    return torch.argmax(x, dim=axis)


def argmin(x, axis=None):
    x = convert_to_tensor(x)
    return torch.argmin(x, dim=axis)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        axis = -1
        x = x.reshape(-1)
    return torch.argsort(x, dim=axis)


def array(x, dtype=None):
    dtype = to_torch_dtype(dtype)
    if not isinstance(x, torch.Tensor):
        return x
    return x.numpy()


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    # Conversion to float necessary for `torch.mean`
    x = cast(x, "float32") if x.dtype in TORCH_INT_TYPES else x
    if weights is not None:
        weights = convert_to_tensor(weights)
        return torch.sum(torch.mul(x, weights), dim=axis) / torch.sum(
            weights, dim=-1
        )
    return torch.mean(x, axis)


def bincount(x, weights=None, minlength=0):
    x = convert_to_tensor(x, dtype=int)
    weights = convert_to_tensor(weights)
    return torch.bincount(x, weights, minlength)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return torch.broadcast_to(x, shape)


def ceil(x):
    x = convert_to_tensor(x)
    return torch.ceil(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    x_min, x_max = convert_to_tensor(x_min), convert_to_tensor(x_max)
    return torch.clip(x, min=x_min, max=x_max)


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.cat(xs, dim=axis)


def conjugate(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.conj(x).resolve_conj()


def conj(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.conj(x).resolve_conj()


def copy(x):
    x = convert_to_tensor(x)
    return torch.clone(x)


def cos(x):
    x = convert_to_tensor(x)
    return torch.cos(x)


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    return torch.count_nonzero(x, dim=axis).T


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # TODO: There is API divergence between np.cross and torch.cross,
    # preventing `axisa`, `axisb`, and `axisc` parameters from
    # being used.
    # https://github.com/pytorch/pytorch/issues/50273
    if axisa != -1 or axisb != -1 or axisc != -1:
        raise NotImplementedError(
            "Due to API divergence between `torch.cross()` and "
            "`np.cross`, the following arguments are not supported: "
            f"axisa={axisa}, axisb={axisb}, axisc={axisc}"
        )
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.cross(x1, x2, dim=axis)


def cumprod(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return torch.cumprod(x, dim=axis)


def cumsum(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return torch.cumsum(x, dim=axis)


def diag(x, k=0):
    x = convert_to_tensor(x)
    return torch.diag(x, diagonal=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return torch.diagonal(
        x,
        offset=offset,
        dim1=axis1,
        dim2=axis2,
    )


def dot(x, y):
    x, y = convert_to_tensor(x), convert_to_tensor(y)
    if x.ndim == 0 or y.ndim == 0:
        return torch.multiply(x, y)
    return torch.matmul(x, y)


def empty(shape, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    return torch.empty(size=shape, dtype=dtype)


def equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    return torch.exp(x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    return torch.unsqueeze(x, dim=axis)


def expm1(x):
    x = convert_to_tensor(x)
    return torch.expm1(x)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return torch.flip(x, dims=axis)


def floor(x):
    x = convert_to_tensor(x)
    return torch.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = to_torch_dtype(dtype)
    if hasattr(fill_value, "__len__"):
        fill_value = convert_to_tensor(fill_value)
        reps = shape[-1] // fill_value.shape[-1]  # channels-last reduction
        reps_by_dim = (*shape[:-1], reps)
        return torch.tile(fill_value, reps_by_dim)
    return torch.full(size=shape, fill_value=fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    dtype = to_torch_dtype(dtype)
    if hasattr(fill_value, "__len__"):
        fill_value = convert_to_tensor(fill_value)
        reps_by_dim = tuple(
            [x.shape[i] // fill_value.shape[i] for i in range(x.ndim)]
        )
        return torch.tile(fill_value, reps_by_dim)
    x = convert_to_tensor(x)
    return torch.full_like(input=x, fill_value=fill_value, dtype=dtype)


def greater(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.greater(x1, x2)


def greater_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.greater_equal(x1, x2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.hstack(xs)


def identity(n, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    return torch.eye(n, dtype=dtype)


def imag(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.imag(x)


def isclose(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    if torch.is_floating_point(x1) and not torch.is_floating_point(x2):
        x2 = cast(x2, x1.dtype)
    if torch.is_floating_point(x2) and not torch.is_floating_point(x1):
        x1 = cast(x1, x2.dtype)
    return torch.isclose(x1, x2)


def isfinite(x):
    x = convert_to_tensor(x)
    return torch.isfinite(x)


def isinf(x):
    x = convert_to_tensor(x)
    return torch.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    return torch.isnan(x)


def less(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.less(x1, x2)


def less_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if axis != 0:
        raise ValueError(
            "torch.linspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    dtype = to_torch_dtype(dtype)
    if endpoint is False:
        stop = stop - ((stop - start) / num)
    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start, stop = convert_to_tensor(start), convert_to_tensor(stop)
        stop = cast(stop, dtype) if endpoint is False and dtype else stop
        steps = torch.arange(num, dtype=dtype) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # increments from `start` to `stop` in each dimension
        linspace = start[None] + steps * (stop - start)[None]
    else:
        linspace = torch.linspace(
            start=start,
            end=stop,
            steps=num,
            dtype=dtype,
        )
    if retstep is True:
        return (linspace, num)
    return linspace


def log(x):
    x = convert_to_tensor(x)
    return torch.log(x)


def log10(x):
    x = convert_to_tensor(x)
    return torch.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    return torch.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    return torch.log2(x)


def logaddexp(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    x1 = cast(x1, "float32") if x1.dtype in TORCH_INT_TYPES else x1
    x2 = cast(x2, "float32") if x2.dtype in TORCH_INT_TYPES else x2
    return torch.logaddexp(x1, x2)


def logical_and(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return torch.logical_not(x)


def logical_or(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
            "torch.logspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    dtype = to_torch_dtype(dtype)
    if endpoint is False:
        stop = stop - ((stop - start) / num)
    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start, stop = convert_to_tensor(start), convert_to_tensor(stop)
        stop = cast(stop, dtype) if endpoint is False and dtype else stop
        steps = torch.arange(num, dtype=dtype) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # increments from `start` to `stop` in each dimension
        linspace = start[None] + steps * (stop - start)[None]
        logspace = base**linspace
    else:
        logspace = torch.logspace(
            start=start,
            end=stop,
            steps=num,
            base=base,
            dtype=dtype,
        )
    return logspace


def maximum(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(sc_tensor) for sc_tensor in x]
    result = torch.meshgrid(x, indexing=indexing)
    return [arr.numpy() for arr in result]


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if axis is None:
        result = torch.min(x)
    else:
        if isinstance(axis, list):
            axis = axis[-1]
        result = torch.min(x, dim=axis, keepdim=keepdims)

    if isinstance(getattr(result, "values", None), torch.Tensor):
        result = result.values

    if initial is not None:
        return torch.minimum(result, initial)
    return result


def minimum(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.minimum(x1, x2)


def mod(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.remainder(x1, x2)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    return torch.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    x = convert_to_tensor(x)
    return torch.nan_to_num(x)


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    x = convert_to_tensor(x)
    return torch.nonzero(x).T


def not_equal(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.not_equal(x1, x2)


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_torch_dtype(dtype)
    return torch.ones_like(x, dtype=dtype)


def outer(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.outer(x1.flatten(), x2.flatten())


def pad(x, pad_width, mode="constant"):
    x = convert_to_tensor(x)
    pad_sum = ()
    for pad in pad_width:
        pad_sum += pad
    return torch.nn.functional.pad(x, pad=pad_sum, mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_torch_dtype(dtype)
    if axis is None:
        return torch.prod(x, dtype=dtype)
    elif isinstance(axis, list):
        axis = axis[-1]
    return torch.prod(x, dim=axis, keepdim=keepdims, dtype=dtype)


def ravel(x):
    x = convert_to_tensor(x)
    return torch.ravel(x)


def real(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)  # needed for complex type conversion
    return torch.real(x)


def reciprocal(x):
    x = convert_to_tensor(x)
    return torch.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    repeats = convert_to_tensor(repeats, dtype=int)
    return torch.repeat_interleave(x, repeats, dim=axis)


def reshape(x, new_shape):
    x = convert_to_tensor(x)
    return torch.reshape(x, new_shape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    return torch.roll(x, shift, dims=axis)


def sign(x):
    x = convert_to_tensor(x)
    return torch.sign(x)


def sin(x):
    x = convert_to_tensor(x)
    return torch.sin(x)


def size(x):
    x_shape = convert_to_tensor(tuple(x.shape))
    return torch.prod(x_shape)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    return torch.sort(x, dim=axis).values


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    return torch.split(
        tensor=x,
        split_size_or_sections=indices_or_sections,
        dim=axis,
    )


def stack(x, axis=0):
    x = [convert_to_tensor(elem) for elem in x]
    return torch.stack(x, dim=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # Conversion to float necessary for `torch.std`
    x = cast(x, "float32") if x.dtype in TORCH_INT_TYPES else x
    # Remove Bessel correction to align with numpy
    return torch.std(x, dim=axis, keepdim=keepdims, unbiased=False)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return torch.swapaxes(x, axis0=axis1, axis1=axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    if axis is not None:
        return torch.index_select(x, dim=axis, index=indices)
    return torch.take(x, index=indices)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    return torch.take_along_dim(x, indices, dim=axis)


def tan(x):
    x = convert_to_tensor(x)
    return torch.tan(x)


def tensordot(x1, x2, axes=2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    # Conversion to long necessary for `torch.tensordot`
    x1 = cast(x1, "int64") if x1.dtype in TORCH_INT_TYPES else x1
    x2 = cast(x2, "int64") if x2.dtype in TORCH_INT_TYPES else x2
    return torch.tensordot(x1, x2, dims=axes)


def round(x, decimals=0):
    x = convert_to_tensor(x)
    return torch.round(x, decimals=decimals)


def tile(x, repeats):
    x = convert_to_tensor(x)
    return torch.tile(x, dims=repeats)


def trace(x, offset=None, axis1=None, axis2=None):
    x = convert_to_tensor(x)
    return torch.sum(torch.diagonal(x, offset, axis1, axis2), dim=-1)


def tri(N, M=None, k=0, dtype="float32"):
    dtype = to_torch_dtype(dtype)
    pass


def tril(x, k=0):
    x = convert_to_tensor(x)
    return torch.tril(x, diagonal=k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    return torch.triu(x, diagonal=k)


def vdot(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.vdot(x1, x2)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return torch.vstack(xs)


def where(condition, x1, x2):
    condition = convert_to_tensor(condition, dtype=bool)
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.where(condition, x1, x2)


def divide(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.divide(x1, x2)


def true_divide(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.true_divide(x1, x2)


def power(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return torch.pow(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return torch.negative(x)


def square(x):
    x = convert_to_tensor(x)
    return torch.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    return torch.sqrt(x)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    if axis is not None:
        return torch.squeeze(x, dim=axis)
    return torch.squeeze(x)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if axes is not None:
        return torch.permute(x, dims=axes)
    return x.T


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x, dtype="float32")
    # Conversion to float necessary for `torch.var`
    x = cast(x, "float32") if x.dtype in TORCH_INT_TYPES else x
    # Bessel correction removed for numpy compatibility
    return torch.var(x, dim=axis, keepdim=keepdims, correction=0)


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is not None:
        return torch.sum(x, axis=axis, keepdim=keepdims)
    return torch.sum(x)


def eye(N, M=None, k=None, dtype="float32"):
    # TODO: implement support for `k` diagonal arg,
    # does not exist in torch.eye()
    if k is not None:
        raise NotImplementedError(
            "Due to API divergence bewtween `torch.eye` "
            "and `np.eye`, the argument k is not supported: "
            f"Received: k={k}"
        )
    dtype = to_torch_dtype(dtype)
    if M is not None:
        return torch.eye(n=N, m=M, dtype=dtype)
    return torch.eye(n=N, dtype=dtype)
