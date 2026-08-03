import paddle

from keras.src.backend.common import dtypes
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import to_paddle_dtype


def cholesky(a, upper=False):
    return paddle.linalg.cholesky(convert_to_tensor(a), upper=upper)


def cholesky_inverse(x, upper=False):
    x = convert_to_tensor(x)
    return paddle.linalg.cholesky_inverse(x, upper=upper)


def det(a):
    return paddle.linalg.det(convert_to_tensor(a))


def eig(a):
    return paddle.linalg.eig(convert_to_tensor(a))


def eigh(a):
    w, v = paddle.linalg.eigh(convert_to_tensor(a))
    return w, v


def inv(a):
    return paddle.linalg.inv(convert_to_tensor(a))


def lu_factor(x):
    x = convert_to_tensor(x)
    LU, pivots = paddle.linalg.lu(x)
    # paddle returns pivots with 1-based indexing
    return LU, pivots - 1


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if tol is not None:
        return paddle.linalg.matrix_rank(x, tol=tol)
    return paddle.linalg.matrix_rank(x)


def _vector_norm(x, ord, axis, keepdims):
    if ord is None:
        ord = 2
    abs_x = paddle.abs(x)
    if ord == float("inf"):
        return paddle.max(abs_x, axis=axis, keepdim=keepdims)
    if ord == float("-inf"):
        return paddle.min(abs_x, axis=axis, keepdim=keepdims)
    if ord == 0:
        return paddle.sum(
            paddle.cast(x != 0, x.dtype), axis=axis, keepdim=keepdims
        )
    if ord == 1:
        return paddle.sum(abs_x, axis=axis, keepdim=keepdims)
    if ord == 2:
        return paddle.sqrt(
            paddle.sum(paddle.square(abs_x), axis=axis, keepdim=keepdims)
        )
    return paddle.pow(
        paddle.sum(paddle.pow(abs_x, ord), axis=axis, keepdim=keepdims),
        1.0 / ord,
    )


def _singular_values(x, row_axis, col_axis):
    # `paddle.linalg.svdvals` expects the matrix in the two trailing axes.
    perm = [i for i in range(x.ndim) if i not in (row_axis, col_axis)] + [
        row_axis,
        col_axis,
    ]
    return paddle.linalg.svdvals(paddle.transpose(x, perm))


def _matrix_norm(x, ord, axis, keepdims):
    row_axis, col_axis = (a % x.ndim for a in axis)
    if row_axis == col_axis:
        raise ValueError(
            f"Duplicate axes given for matrix norm. Received: axis={axis}"
        )
    if ord in ("nuc", 2, -2):
        s = _singular_values(x, row_axis, col_axis)
        if ord == "nuc":
            result = paddle.sum(s, axis=-1)
        elif ord == 2:
            result = paddle.max(s, axis=-1)
        else:
            result = paddle.min(s, axis=-1)
        if keepdims:
            # `_singular_values` moved the reduced axes to the end, so the
            # reduced dimensions have to be inserted back in place.
            shape = list(x.shape)
            shape[row_axis] = 1
            shape[col_axis] = 1
            result = paddle.reshape(result, shape)
        return result

    abs_x = paddle.abs(x)
    if ord in (1, -1):
        reduce_first, reduce_second = row_axis, col_axis
    elif ord in (float("inf"), float("-inf")):
        reduce_first, reduce_second = col_axis, row_axis
    elif ord in (None, "fro"):
        return paddle.sqrt(
            paddle.sum(
                paddle.square(abs_x),
                axis=[row_axis, col_axis],
                keepdim=keepdims,
            )
        )
    else:
        raise ValueError(
            f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
        )
    # Sum over one axis, then take the max/min over the other. The second
    # axis shifts by one when the first one is dropped.
    result = paddle.sum(abs_x, axis=reduce_first, keepdim=True)
    if ord in (1, float("inf")):
        result = paddle.max(result, axis=reduce_second, keepdim=True)
    else:
        result = paddle.min(result, axis=reduce_second, keepdim=True)
    if not keepdims:
        result = paddle.squeeze(result, axis=[row_axis, col_axis])
    return result


def norm(x, ord=None, axis=None, keepdims=False):
    # `paddle.linalg.norm` does not follow the numpy semantics that Keras
    # specifies (it flattens for `axis=None` and silently accepts invalid
    # `ord` values), so the numpy behavior is reproduced here.
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
        x = x.cast(to_paddle_dtype(dtype))
    ndim = x.ndim

    if isinstance(axis, (list, tuple)) and len(axis) == 1:
        axis = axis[0]
    if axis is None:
        if ord is None:
            result = _vector_norm(
                paddle.reshape(x, [-1]), 2, axis=0, keepdims=False
            )
            if keepdims:
                result = paddle.reshape(result, [1] * ndim)
            return result
        if ndim == 1:
            axis = 0
        elif ndim == 2:
            axis = (0, 1)
        else:
            raise ValueError(
                "Improper number of dimensions to norm. "
                f"Received: x.ndim={ndim}, ord={ord}, axis={axis}"
            )

    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError(
                f"Invalid axis {a} for input with {ndim} dimension(s). "
                f"Received: axis={axis}"
            )

    if len(axes) == 1:
        if isinstance(ord, str):
            raise ValueError(
                f"Invalid `ord` argument for vector norm. Received: ord={ord}"
            )
        return _vector_norm(x, ord, axis=axes[0], keepdims=keepdims)
    if len(axes) == 2:
        return _matrix_norm(x, ord, axes, keepdims)
    raise ValueError(
        "Invalid `axis` argument: must be `None`, an integer or a tuple of "
        f"two integers. Received: axis={axis}"
    )


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    if rcond is not None:
        return paddle.linalg.pinv(x, rcond=rcond)
    return paddle.linalg.pinv(x)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return paddle.linalg.qr(convert_to_tensor(x), mode=mode)


def solve(a, b):
    return paddle.linalg.solve(convert_to_tensor(a), convert_to_tensor(b))


def solve_triangular(a, b, lower=False):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    if b.ndim == a.ndim - 1:
        b = paddle.unsqueeze(b, axis=-1)
        return paddle.linalg.triangular_solve(a, b, upper=not lower).squeeze(
            axis=-1
        )
    return paddle.linalg.triangular_solve(a, b, upper=not lower)


def svd(x, full_matrices=True, compute_uv=True):
    x = convert_to_tensor(x)
    if not compute_uv:
        return paddle.linalg.svdvals(x)
    return paddle.linalg.svd(x, full_matrices=full_matrices)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # `paddle.linalg.lstsq` requires `b` to have a trailing "columns" axis.
    b_rank_1 = b.ndim == a.ndim - 1
    if b_rank_1:
        b = paddle.unsqueeze(b, axis=-1)
    result = paddle.linalg.lstsq(a, b, rcond=rcond)[0]
    if b_rank_1:
        result = paddle.squeeze(result, axis=-1)
    return result


def jvp(fun, primals, tangents, has_aux=False):
    primals = [convert_to_tensor(p) for p in primals]
    tangents = [convert_to_tensor(t) for t in tangents]
    if has_aux:
        aux_container = []

        def wrapped_fun(*args):
            outputs, aux = fun(*args)
            aux_container.append(aux)
            return outputs

        primals_out, tangents_out = paddle.incubate.autograd.jvp(
            wrapped_fun, primals, tangents
        )
        return primals_out, tangents_out, aux_container[-1]
    primals_out, tangents_out = paddle.incubate.autograd.jvp(
        fun, primals, tangents
    )
    return primals_out, tangents_out
