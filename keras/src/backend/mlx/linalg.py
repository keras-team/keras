import mlx.core as mx
import numpy as np

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.mlx.core import convert_to_tensor


def _det_2x2(a):
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]


def _det_3x3(a):
    return (
        a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2]
        + a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0]
        + a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1]
        - a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0]
        - a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1]
        - a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2]
    )


def det(a):
    a_shape = a.shape
    if len(a_shape) >= 2 and a_shape[-1] == 2 and a_shape[-2] == 2:
        return _det_2x2(a)
    elif len(a_shape) >= 2 and a_shape[-1] == 3 and a_shape[-2] == 3:
        return _det_3x3(a)
    # elif len(a_shape) >= 2 and a_shape[-1] == a_shape[-2]:
    # TODO: Swap to mlx.linalg.det when supported
    a = np.array(a)
    output = np.linalg.det(a)
    return mx.array(output)


def eig(a):
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        return mx.linalg.eig(a)


def eigh(a):
    with mx.stream(mx.cpu):
        return mx.linalg.eigh(a)


def lu_factor(a):
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        return mx.linalg.lu_factor(a)


def solve(a, b):
    with mx.stream(mx.cpu):
        # [linalg::solve] This op is not yet supported on the GPU.
        # Explicitly pass a CPU stream to run it.
        return mx.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    upper = not lower
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        if b.ndim == a.ndim - 1:
            b = mx.expand_dims(b, axis=-1)
            return mx.squeeze(
                mx.linalg.solve_triangular(a, b, upper=upper), axis=-1
            )
        return mx.linalg.solve_triangular(a, b, upper=upper)


def qr(x, mode="reduced"):
    if mode != "reduced":
        raise ValueError(
            "`mode` argument value not supported. "
            "Only 'reduced' is supported by the mlx backend. "
            f"Received: mode={mode}"
        )
    with mx.stream(mx.cpu):
        return mx.linalg.qr(x)


def svd(x, full_matrices=True, compute_uv=True):
    with mx.stream(mx.cpu):
        u, s, vt = mx.linalg.svd(x)
        if not compute_uv:
            return s
        if not full_matrices:
            n = min(x.shape[-2:])
            return u[..., :n], s, vt[:n, ...]
        # mlx returns full matrices by default
        return u, s, vt


def cholesky(a):
    with mx.stream(mx.cpu):
        return mx.linalg.cholesky(a)


def norm(x, ord=None, axis=None, keepdims=False):
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    x = convert_to_tensor(x, dtype=dtype)
    # TODO: swap to mlx.linalg.norm when it support singular value norms
    x = np.array(x)
    output = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return mx.array(output)


def inv(a):
    with mx.stream(mx.cpu):
        return mx.linalg.inv(a)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "Incompatible shapes: a and b must have the same number of rows."
        )
    b_orig_ndim = b.ndim
    if b.ndim == 1:
        b = mx.expand_dims(b, axis=-1)
    elif b.ndim > 2:
        raise ValueError("b must be 1D or 2D.")

    if b.ndim != 2:
        raise ValueError("b must be 1D or 2D.")

    m, n = a.shape
    dtype = a.dtype

    eps = np.finfo(np.array(a).dtype).eps
    if a.shape == ():
        s = mx.zeros((0,), dtype=dtype)
        x = mx.zeros((n, *b.shape[1:]), dtype=dtype)
    else:
        if rcond is None:
            rcond = eps * max(m, n)
        else:
            rcond = mx.where(rcond < 0, eps, rcond)
    u, s, vt = svd(a, full_matrices=False)

    mask = s >= mx.array(rcond, dtype=s.dtype) * s[0]
    safe_s = mx.array(mx.where(mask, s, 1), dtype=dtype)
    s_inv = mx.where(mask, 1 / safe_s, 0)
    s_inv = mx.expand_dims(s_inv, axis=1)
    u_t_b = mx.matmul(mx.transpose(mx.conj(u)), b)
    x = mx.matmul(mx.transpose(mx.conj(vt)), s_inv * u_t_b)

    if b_orig_ndim == 1:
        x = mx.squeeze(x, axis=-1)

    return x
