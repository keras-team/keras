import numpy as np
import scipy.linalg as sl

import mlx.core as mx

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.mlx.core import _cast
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_numpy
from keras.src.backend.mlx.core import convert_to_tensor

# Many MLX linalg ops (inv, solve, cholesky, svd, eigh, eig, lu) are not yet
# supported on the GPU stream. We run them on an explicit CPU stream. MLX uses
# unified memory, so this is a zero-copy dispatch choice, not a host transfer.
_CPU = mx.Device(mx.DeviceType.cpu, 0)


def _cpu(fn, *args, **kwargs):
    with mx.stream(_CPU):
        out = fn(*args, **kwargs)
    return out


def _to_np(x):
    if hasattr(x, "dtype") and "mlx" in str(type(x)):
        return np.asarray(convert_to_numpy(x))
    return np.asarray(x)


def _wrap(arr, dtype=None):
    out = convert_to_tensor(arr)
    if dtype is not None:
        out = _cast(out, dtype)
    return out


def _wrap_tuple(result, dtype=None):
    if isinstance(result, tuple):
        return tuple(_wrap(r, dtype) for r in result)
    return _wrap(result, dtype)


def cholesky(a, upper=False):
    a = convert_to_tensor(a)
    # `mx.linalg.cholesky` runs on CPU and does NOT validate positive-
    # definiteness: it silently returns garbage for non-PSD inputs. NumPy raises
    # `LinAlgError` (a `ValueError` subclass) in that case, so verify the
    # factorization reconstructs `a` and raise to match.
    with mx.stream(_CPU):
        L = mx.linalg.cholesky(a)
        recon = mx.matmul(L, mx.swapaxes(L, -1, -2))
        scale = mx.max(mx.abs(a)) + 1.0
        if not bool(mx.all(mx.abs(recon - a) < 1e-3 * scale)):
            raise ValueError(
                "Cholesky decomposition failed: the input is not positive "
                "definite."
            )
    if upper:
        L = mx.swapaxes(L, -1, -2)
    return L


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    identity = mx.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    if upper:
        a_inv = mx.matmul(inv_chol, mx.swapaxes(inv_chol, -1, -2))
    else:
        a_inv = mx.matmul(mx.swapaxes(inv_chol, -1, -2), inv_chol)
    return a_inv


def det(a):
    a = convert_to_tensor(a)
    return _wrap(np.linalg.det(_to_np(a)))


def eig(a):
    a = convert_to_tensor(a)
    w, v = np.linalg.eig(_to_np(a))
    return _wrap(w), _wrap(v)


def eigh(a):
    a = convert_to_tensor(a)
    w, v = np.linalg.eigh(_to_np(a))
    return _wrap(w), _wrap(v)


def inv(a):
    a = convert_to_tensor(a)
    return _cpu(mx.linalg.inv, a)


def lu_factor(a):
    # scipy gufunc form; no batched MLX equivalent with the same return shape.
    a = convert_to_tensor(a)
    if a.ndim == 2:
        return _wrap_tuple(sl.lu_factor(_to_np(a)))
    m, n = a.shape[-2:]
    signature = "(m,n) -> (m,n), "
    signature += "(m)" if m <= n else "(n)"
    _lu_factor_gufunc = np.vectorize(
        sl.lu_factor,
        signature=signature,
    )
    return _wrap_tuple(_lu_factor_gufunc(_to_np(a)))


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
        x = _cast(x, dtype)
    # mx.linalg.norm supports ord/axis/keepdims for the common cases. Fall back
    # to numpy for any ord/axis combination it rejects. MLX raises IndexError
    # for out-of-bounds axes; numpy raises AxisError (a ValueError subclass), so
    # routing those through numpy yields the ValueError Keras/NumPy expect.
    try:
        out = mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        # mx collapses `axis=None` to a single kept dim `(1,)` with keepdims,
        # but numpy keeps every dim: `(1,)*ndim`. Reshape to match numpy.
        if axis is None and keepdims:
            out = mx.reshape(out, (1,) * x.ndim)
        return out
    except (ValueError, TypeError, IndexError):
        out = np.linalg.norm(_to_np(x), ord=ord, axis=axis, keepdims=keepdims)
        return _wrap(out, dtype)


def qr(x, mode="reduced"):
    x = convert_to_tensor(x)
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    # MLX `qr` has no `mode` argument (always reduced). Use numpy for
    # `complete` and for batched inputs to guarantee the exact layout.
    q, r = np.linalg.qr(_to_np(x), mode=mode)
    return _wrap(q), _wrap(r)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return _cpu(mx.linalg.solve, a, b)


def solve_triangular(a, b, lower=False):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    upper = not lower
    squeeze_b = False
    if b.ndim == a.ndim - 1:
        b = mx.expand_dims(b, axis=-1)
        squeeze_b = True
    out = _cpu(mx.linalg.solve_triangular, a, b, upper=upper)
    if squeeze_b:
        out = mx.squeeze(out, axis=-1)
    return out


def svd(x, full_matrices=True, compute_uv=True):
    x = convert_to_tensor(x)
    # MLX `svd` has no `full_matrices` argument and is GPU-unsupported; the
    # numpy reference is sign-dependent, so use numpy for an exact match.
    return _wrap_tuple(
        np.linalg.svd(_to_np(x), full_matrices=full_matrices, compute_uv=compute_uv)
    )


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return _wrap(np.linalg.lstsq(_to_np(a), _to_np(b), rcond=rcond)[0])


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    return _wrap(np.linalg.matrix_rank(_to_np(x), tol=tol), "int32")


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    return _wrap(np.linalg.pinv(_to_np(x), rcond=rcond))


def jvp(fun, primals, tangents, has_aux=False):
    primals_out, tangents_out = mx.jvp(fun, primals, tangents)
    # `mx.jvp` returns the function outputs as flat lists matching its return
    # tuple, and differentiates every output. Keras expects the function's own
    # output structure (a bare value for a single output) and, for `has_aux`,
    # the auxiliary data returned separately.
    if has_aux:
        # By Keras contract `fun` returns `(out, aux)`; mx flattened both, so
        # the main output is first and the auxiliary primal is second.
        return primals_out[0], tangents_out[0], primals_out[1]
    if isinstance(primals_out, (list, tuple)) and len(primals_out) == 1:
        return primals_out[0], tangents_out[0]
    return tuple(primals_out), tuple(tangents_out)
