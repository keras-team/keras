import jax
import jax.numpy as jnp
import jax.scipy as jsp

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.jax.core import cast
from keras.src.backend.jax.core import convert_to_tensor


def cholesky(a):
    out = jnp.linalg.cholesky(a)
    try:
        # In eager mode, raise for nan to
        # achieve behavior consistency with numpy
        if jnp.any(jnp.isnan(out)):
            raise ValueError(
                "Cholesky decomposition failed. "
                "The input might not be a valid "
                "positive definite matrix."
            )
    except jax.errors.TracerBoolConversionError:
        # Cannot raise for nan in tracing mode
        pass
    return out


def det(a):
    return jnp.linalg.det(a)


def eig(x):
    return jnp.linalg.eig(x)


def eigh(x):
    return jnp.linalg.eigh(x)


def inv(a):
    return jnp.linalg.inv(a)


def lu_factor(x):
    lu_factor_fn = jsp.linalg.lu_factor
    if x.ndim > 2:
        for i in range(x.ndim - 2):
            lu_factor_fn = jax.vmap(lu_factor_fn)

    return lu_factor_fn(x)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return jnp.linalg.qr(x, mode=mode)


def solve(a, b):
    return jnp.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    return jsp.linalg.solve_triangular(a, b, lower=lower)


def svd(x, full_matrices=True, compute_uv=True):
    return jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return jnp.linalg.lstsq(a, b, rcond=rcond)[0]
