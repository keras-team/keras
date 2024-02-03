import tensorflow as tf
from tensorflow.experimental import numpy as tfnp

from keras.backend import config
from keras.backend import standardize_dtype
from keras.backend.common import dtypes
from keras.backend.tensorflow.core import cast
from keras.backend.tensorflow.core import convert_to_tensor


def cholesky(a):
    out = tf.linalg.cholesky(a)
    # tf.linalg.cholesky simply returns NaNs for non-positive definite matrices
    return tf.debugging.check_numerics(out, "Cholesky")


def det(a):
    return tf.linalg.det(a)


def eig(a):
    return tf.linalg.eig(a)


def inv(a):
    return tf.linalg.inv(a)


def lu_factor(a):
    lu, p = tf.linalg.lu(a)
    return lu, tf.math.invert_permutation(p)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x_shape = x.shape
    ndim = x_shape.rank

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    axis = axis[0] if len(axis) == 1 else axis
    num_axes = 1 if isinstance(axis, int) else len(axis)

    if num_axes == 1 and ord is None:
        ord = "euclidean"
    elif num_axes == 2 and ord is None:
        ord = "fro"

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)

    # Fast path to utilze `tf.linalg.norm`
    if (num_axes == 1 and ord in ("euclidean", 1, 2, float("inf"))) or (
        num_axes == 2 and ord in ("euclidean", "fro", 1, 2, float("inf"))
    ):
        return tf.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    # Ref: jax.numpy.linalg.norm
    if num_axes == 1 and ord not in ("fro", "nuc"):
        if ord == float("-inf"):
            return tf.math.reduce_min(
                tf.math.abs(x), axis=axis, keepdims=keepdims
            )
        elif ord == 0:
            return tf.math.reduce_sum(
                tf.cast(tf.not_equal(x, 0), dtype=x.dtype),
                axis=axis,
                keepdims=keepdims,
            )
        else:
            ord = convert_to_tensor(ord, dtype=x.dtype)
            out = tf.math.reduce_sum(
                tf.pow(tf.math.abs(x), ord), axis=axis, keepdims=keepdims
            )
            return tf.pow(out, 1.0 / ord)
    elif num_axes == 2 and ord in ("nuc", float("-inf"), -2, -1):
        row_axis, col_axis = axis[0], axis[1]
        row_axis = row_axis + ndim if row_axis < 0 else row_axis
        col_axis = col_axis + ndim if col_axis < 0 else col_axis
        if ord == float("-inf"):
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=col_axis, keepdims=keepdims),
                axis=row_axis,
                keepdims=keepdims,
            )
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis,
                keepdims=keepdims,
            )
        else:
            x = tfnp.moveaxis(x, axis, (-2, -1))
            if ord == -2:
                x = tf.math.reduce_min(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            else:
                x = tf.math.reduce_sum(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            if keepdims:
                x = tf.expand_dims(x, axis[0])
                x = tf.expand_dims(x, axis[1])
        return x

    if num_axes == 1:
        raise ValueError(
            f"Invalid `ord` argument for vector norm. Received: ord={ord}"
        )
    elif num_axes == 2:
        raise ValueError(
            f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
        )
    else:
        raise ValueError(f"Invalid axis values. Received: axis={axis}")


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)


def solve(a, b):
    # tensorflow.linalg.solve only supports same rank inputs
    if tf.rank(b) == tf.rank(a) - 1:
        b = tf.expand_dims(b, axis=-1)
        return tf.squeeze(tf.linalg.solve(a, b), axis=-1)
    return tf.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if b.shape.ndims == a.shape.ndims - 1:
        b = tf.expand_dims(b, axis=-1)
        return tf.squeeze(
            tf.linalg.triangular_solve(a, b, lower=lower), axis=-1
        )
    return tf.linalg.triangular_solve(a, b, lower=lower)


def svd(x, full_matrices=True, compute_uv=True):
    s, u, v = tf.linalg.svd(
        x, full_matrices=full_matrices, compute_uv=compute_uv
    )
    return u, s, tf.linalg.adjoint(v)
