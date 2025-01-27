import jax.numpy as jnp
import mlx.core as mx

from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.mlx.core import convert_to_tensor


def det(a):
    # TODO: Swap to mlx.linalg.det when supported
    a = jnp.array(a)
    output = jnp.linalg.det(a)
    return mx.array(output)


def eig(a):
    raise NotImplementedError("eig not yet implemented in mlx.")


def eigh(a):
    raise NotImplementedError("eigh not yet implemented in mlx.")


def lu_factor(a):
    raise NotImplementedError("lu_factor not yet implemented in mlx.")


def solve(a, b):
    raise NotImplementedError("solve_triangular not yet implemented in mlx.")


def solve_triangular(a, b, lower=False):
    raise NotImplementedError("solve_triangular not yet implemented in mlx.")


def qr(x, mode="reduced"):
    # TODO: Swap to mlx.linalg.qr when it supports non-square matrices
    x = jnp.array(x)
    output = jnp.linalg.qr(x, mode=mode)
    return mx.array(output[0]), mx.array(output[1])


def svd(x, full_matrices=True, compute_uv=True):
    with mx.stream(mx.cpu):
        return mx.linalg.svd(x)


def cholesky(a):
    with mx.stream(mx.cpu):
        return mx.linalg.cholesky(a)


def norm(x, ord=None, axis=None, keepdims=False):
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    x = convert_to_tensor(x, dtype=dtype)
    # TODO: swap to mlx.linalg.norm when it support singular value norms
    x = jnp.array(x)
    output = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return mx.array(output)


def inv(a):
    with mx.stream(mx.cpu):
        return mx.linalg.inv(a)
