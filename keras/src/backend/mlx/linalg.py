import mlx.core as mx
import numpy as np

from keras.src import tree
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
            return u[..., :n], s, vt[..., :n, :]
        # mlx returns full matrices by default
        return u, s, vt


def cholesky(a, upper=False):
    with mx.stream(mx.cpu):
        return mx.linalg.cholesky(a, upper=upper)


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


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    target = x.dtype
    # mlx pinv is svd-based and CPU-only, so fall back to numpy to match the
    # reference rcond handling exactly.
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    result = np.linalg.pinv(np.asarray(x), rcond=rcond)
    return mx.array(result).astype(target)


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    rank = np.linalg.matrix_rank(np.asarray(x), tol=tol)
    return mx.array(rank.astype("int32"))


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    identity = mx.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    inv_chol_t = mx.swapaxes(inv_chol, -1, -2)
    if upper:
        return mx.matmul(inv_chol, inv_chol_t)
    return mx.matmul(inv_chol_t, inv_chol)


def jvp(fun, primals, tangents, has_aux=False):
    primals = list(primals)
    tangents = list(tangents)

    if has_aux:
        # mx.jvp has no has_aux, so strip the aux output before differentiating
        # and recover it from a concrete evaluation below.
        aux_holder = [None]

        def target(*args):
            out, aux = fun(*args)
            aux_holder[0] = aux
            return out
    else:
        target = fun

    primals_out, tangents_out = mx.jvp(target, primals, tangents)

    # mx.jvp returns flat lists, so pack them back into the function's output
    # structure (a bare tensor when `fun` returns a single output).
    structure = target(*primals)
    primals_out = tree.pack_sequence_as(structure, primals_out)
    tangents_out = tree.pack_sequence_as(structure, tangents_out)

    if has_aux:
        return primals_out, tangents_out, aux_holder[0]
    return primals_out, tangents_out
