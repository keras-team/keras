import builtins
import functools
import math
import warnings

import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

from keras.backend import config
from keras.backend import standardize_dtype
from keras.backend.common import dtypes
from keras.backend.tensorflow.core import convert_to_tensor


def add(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if isinstance(x1, tf.SparseTensor) or isinstance(x2, tf.SparseTensor):
        return tf.sparse.add(x1, x2)
    return tfnp.add(x1, x2)


def bincount(x, weights=None, minlength=None):
    if minlength is not None:
        x = tf.cast(x, tf.int32)
    if isinstance(x, tf.SparseTensor):
        result = tf.sparse.bincount(
            x,
            weights=weights,
            minlength=minlength,
            axis=-1,
        )
        if x.shape.rank == 1:
            output_shape = (minlength,)
        else:
            batch_size = tf.shape(result)[0]
            output_shape = (batch_size, minlength)
        return tf.SparseTensor(
            indices=result.indices,
            values=result.values,
            dense_shape=output_shape,
        )
    return tf.math.bincount(x, weights=weights, minlength=minlength, axis=-1)


def einsum(subscripts, *operands, **kwargs):
    return tfnp.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    if isinstance(x1, tf.SparseTensor) or isinstance(x2, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.add(x1, tf.sparse.map_values(tf.negative, x2))
        else:
            return tf.sparse.add(x1, tf.negative(x2))
    return tfnp.subtract(x1, x2)


def matmul(x1, x2):
    def with_combined_batch_dimensions(a, b, fn_3d):
        batch_shape = (
            b.shape[:-2] if isinstance(b, tf.SparseTensor) else a.shape[:-2]
        )
        batch_size = math.prod(batch_shape)
        a_3d = reshape(a, [batch_size] + a.shape[-2:])
        b_3d = reshape(b, [batch_size] + b.shape[-2:])
        result = fn_3d(a_3d, b_3d)
        return reshape(result, batch_shape + result.shape[1:])

    def sparse_sparse_matmul(a, b):
        dtype = a.values.dtype
        # Convert SparseTensors to CSR SparseMatrix.
        a_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a.indices, a.values, a.dense_shape
        )
        b_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b.indices, b.values, b.dense_shape
        )
        # Compute the CSR SparseMatrix matrix multiplication.
        result_csr = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a_csr, b_csr, dtype
        )
        # Convert the CSR SparseMatrix to a SparseTensor.
        res = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            result_csr, dtype
        )
        return tf.SparseTensor(res.indices, res.values, res.dense_shape)

    def embedding_lookup_sparse_dense_matmul(a, b):
        # We need at least one id per rows for embedding_lookup_sparse,
        # otherwise there will be missing rows in the output.
        a, _ = tf.sparse.fill_empty_rows(a, 0)
        # We need to split x1 into separate ids and weights tensors. The ids
        # should be the column indices of x1 and the values of the weights
        # can continue to be the actual x1. The column arrangement of ids
        # and weights does not matter as we sum over columns. See details in
        # the documentation for sparse_ops.sparse_tensor_dense_matmul.
        ids = tf.SparseTensor(
            indices=a.indices,
            values=a.indices[:, 1],
            dense_shape=a.dense_shape,
        )
        return tf.nn.embedding_lookup_sparse(b, ids, a, combiner="sum")

    # Either a or b is sparse
    def sparse_dense_matmul_3d(a, b):
        return tf.map_fn(
            lambda x: tf.sparse.sparse_dense_matmul(x[0], x[1]),
            elems=(a, b),
            fn_output_signature=a.dtype,
        )

    x1_sparse = isinstance(x1, tf.SparseTensor)
    x2_sparse = isinstance(x2, tf.SparseTensor)
    if x1_sparse and x2_sparse:
        if x1.shape.rank <= 3:
            return sparse_sparse_matmul(x1, x2)
        else:
            return with_combined_batch_dimensions(x1, x2, sparse_sparse_matmul)
    elif x1_sparse or x2_sparse:
        # Sparse * dense or dense * sparse
        sparse_rank = x1.shape.rank if x1_sparse else x2.shape.rank

        # Special case: embedding_lookup_sparse for sparse * dense and rank 2
        if x1_sparse and sparse_rank == 2:
            return embedding_lookup_sparse_dense_matmul(x1, x2)
        elif sparse_rank == 2:
            return tf.sparse.sparse_dense_matmul(x1, x2)
        elif sparse_rank == 3:
            return sparse_dense_matmul_3d(x1, x2)
        else:
            return with_combined_batch_dimensions(
                x1, x2, sparse_dense_matmul_3d
            )

    return tfnp.matmul(x1, x2)


def multiply(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            ones_like_int8 = functools.partial(tf.ones_like, dtype=tf.int8)
            zeros_like_int8 = functools.partial(tf.zeros_like, dtype=tf.int8)

            # compute the intersection of indices in the form of a sparse tensor
            # containing ones as values
            ones1 = tf.sparse.map_values(ones_like_int8, x1)
            ones2 = tf.sparse.map_values(ones_like_int8, x2)
            # tf.sets.intersection ignores the last dimension when comparing,
            # so we need to add a dummy extra dimension and then remove it
            intersection = tf.sparse.reshape(
                tf.sets.intersection(
                    tf.sparse.expand_dims(ones1, axis=-1),
                    tf.sparse.expand_dims(ones2, axis=-1),
                ),
                x1.dense_shape,
            )

            # compute the masks to remove indices in x1 and x2 that are not part
            # of the intersection, then trim x1 and x2
            zeros1 = tf.sparse.map_values(zeros_like_int8, x1)
            zeros2 = tf.sparse.map_values(zeros_like_int8, x2)
            mask1 = tf.sparse.add(zeros1, intersection)
            mask2 = tf.sparse.add(zeros2, intersection)
            x1_trimmed = tf.sparse.retain(x1, tf.cast(mask1.values, tf.bool))
            x2_trimmed = tf.sparse.retain(x2, tf.cast(mask2.values, tf.bool))

            # now it is an element-wise multiplication on the values
            return tf.sparse.map_values(tf.multiply, x1_trimmed, x2_trimmed)
        else:
            return x1 * x2
    elif isinstance(x2, tf.SparseTensor):
        return x2 * x1
    return tfnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    return tfnp.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    if initial is not None:
        return tf.math.maximum(
            tfnp.max(x, axis=axis, keepdims=keepdims), initial
        )

    # TensorFlow returns -inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the max of an empty tensor.",
        )

    return tfnp.max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.zeros(shape, dtype=dtype)


def absolute(x):
    return tfnp.absolute(x)


def abs(x):
    return absolute(x)


def all(x, axis=None, keepdims=False):
    return tfnp.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    return tfnp.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    return tfnp.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    return tfnp.amin(x, axis=axis, keepdims=keepdims)


def append(
    x1,
    x2,
    axis=None,
):
    return tfnp.append(x1, x2, axis=axis)


def arange(start, stop=None, step=1, dtype=None):
    # tfnp.arange has trouble with dynamic Tensors in compiled function.
    # tf.range does not.
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(step, "dtype", type(step)),
        ]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    return tf.range(start, stop, delta=step, dtype=dtype)


def arccos(x):
    return tfnp.arccos(x)


def arccosh(x):
    return tfnp.arccosh(x)


def arcsin(x):
    return tfnp.arcsin(x)


def arcsinh(x):
    return tfnp.arcsinh(x)


def arctan(x):
    return tfnp.arctan(x)


def arctan2(x1, x2):
    return tfnp.arctan2(x1, x2)


def arctanh(x):
    return tfnp.arctanh(x)


def argmax(x, axis=None):
    return tfnp.argmax(x, axis=axis)


def argmin(x, axis=None):
    return tfnp.argmin(x, axis=axis)


def argsort(x, axis=-1):
    return tfnp.argsort(x, axis=axis)


def array(x, dtype=None):
    return tfnp.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    for a in axis:
        # `tfnp.average` does not handle multiple axes.
        x = tfnp.average(x, weights=weights, axis=a)
    return x


def broadcast_to(x, shape):
    return tfnp.broadcast_to(x, shape)


def ceil(x):
    return tfnp.ceil(x)


def clip(x, x_min, x_max):
    return tfnp.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    sparse_count = builtins.sum(isinstance(x, tf.SparseTensor) for x in xs)
    if sparse_count:
        if sparse_count == len(xs):
            return tf.sparse.concat(axis=axis, sp_inputs=xs)
        else:
            xs = [
                tf.sparse.to_dense(x) if isinstance(x, tf.SparseTensor) else x
                for x in xs
            ]
    return tfnp.concatenate(xs, axis=axis)


def conjugate(x):
    return tfnp.conjugate(x)


def conj(x):
    return conjugate(x)


def copy(x):
    return tfnp.copy(x)


def cos(x):
    return tfnp.cos(x)


def cosh(x):
    return tfnp.cosh(x)


def count_nonzero(x, axis=None):
    return tfnp.count_nonzero(x, axis=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return tfnp.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None):
    return tfnp.cumprod(x, axis=axis)


def cumsum(x, axis=None):
    return tfnp.cumsum(x, axis=axis)


def diag(x, k=0):
    return tfnp.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return tfnp.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def digitize(x, bins):
    bins = list(bins)
    if isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(
            lambda y: tf.raw_ops.Bucketize(input=y, boundaries=bins), x
        )
    elif isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(
            indices=tf.identity(x.indices),
            values=tf.raw_ops.Bucketize(input=x.values, boundaries=bins),
            dense_shape=tf.identity(x.dense_shape),
        )
    x = convert_to_tensor(x)
    return tf.raw_ops.Bucketize(input=x, boundaries=bins)


def dot(x, y):
    return tfnp.dot(x, y)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.empty(shape, dtype=dtype)


def equal(x1, x2):
    return tfnp.equal(x1, x2)


def exp(x):
    return tfnp.exp(x)


def expand_dims(x, axis):
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.expand_dims(x, axis)
    return tfnp.expand_dims(x, axis)


def expm1(x):
    return tfnp.expm1(x)


def flip(x, axis=None):
    return tfnp.flip(x, axis=axis)


def floor(x):
    return tfnp.floor(x)


def full(shape, fill_value, dtype=None):
    return tfnp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return tfnp.full_like(x, fill_value, dtype=dtype)


def greater(x1, x2):
    return tfnp.greater(x1, x2)


def greater_equal(x1, x2):
    return tfnp.greater_equal(x1, x2)


def hstack(xs):
    return tfnp.hstack(xs)


def identity(n, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.identity(n, dtype=dtype)


def imag(x):
    return tfnp.imag(x)


def isclose(x1, x2):
    return tfnp.isclose(x1, x2)


def isfinite(x):
    return tfnp.isfinite(x)


def isinf(x):
    return tfnp.isinf(x)


def isnan(x):
    return tfnp.isnan(x)


def less(x1, x2):
    return tfnp.less(x1, x2)


def less_equal(x1, x2):
    return tfnp.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    return tfnp.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


def log(x):
    return tfnp.log(x)


def log10(x):
    return tfnp.log10(x)


def log1p(x):
    return tfnp.log1p(x)


def log2(x):
    return tfnp.log2(x)


def logaddexp(x1, x2):
    return tfnp.logaddexp(x1, x2)


def logical_and(x1, x2):
    return tfnp.logical_and(x1, x2)


def logical_not(x):
    return tfnp.logical_not(x)


def logical_or(x1, x2):
    return tfnp.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    return tfnp.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


def maximum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.maximum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tfnp.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    return tfnp.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    if initial is not None:
        return tf.math.minimum(
            tfnp.min(x, axis=axis, keepdims=keepdims), initial
        )

    # TensorFlow returns inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the min of an empty tensor.",
        )

    return tfnp.min(x, axis=axis, keepdims=keepdims)


def minimum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.minimum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tfnp.minimum(x1, x2)


def mod(x1, x2):
    return tfnp.mod(x1, x2)


def moveaxis(x, source, destination):
    return tfnp.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    # Replace NaN with 0
    x = tf.where(tf.math.is_nan(x), 0, x)

    # Replace positive infinitiy with dtype.max
    x = tf.where(tf.math.is_inf(x) & (x > 0), x.dtype.max, x)

    # Replace negative infinity with dtype.min
    x = tf.where(tf.math.is_inf(x) & (x < 0), x.dtype.min, x)

    return x


def ndim(x):
    return tfnp.ndim(x)


def nonzero(x):
    return tfnp.nonzero(x)


def not_equal(x1, x2):
    return tfnp.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return tfnp.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return tf.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    return tfnp.outer(x1, x2)


def pad(x, pad_width, mode="constant"):
    return tfnp.pad(x, pad_width, mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    return tfnp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def ravel(x):
    return tfnp.ravel(x)


def real(x):
    return tfnp.real(x)


def reciprocal(x):
    return tfnp.reciprocal(x)


def repeat(x, repeats, axis=None):
    # tfnp.repeat has trouble with dynamic Tensors in compiled function.
    # tf.repeat does not.
    return tf.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.reshape(x, new_shape)
    return tfnp.reshape(x, new_shape)


def roll(x, shift, axis=None):
    return tfnp.roll(x, shift, axis=axis)


def sign(x):
    return tfnp.sign(x)


def sin(x):
    return tfnp.sin(x)


def sinh(x):
    return tfnp.sinh(x)


def size(x):
    return tfnp.size(x)


def sort(x, axis=-1):
    return tfnp.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    return tfnp.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    return tfnp.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    return tfnp.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    return tfnp.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    if isinstance(indices, tf.SparseTensor):
        if x.dtype not in (tf.float16, tf.float32, tf.float64, tf.bfloat16):
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`x.dtype={x.dtype}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            return tfnp.take(x, tf.sparse.to_dense(indices), axis=axis)
        if axis is None:
            x = tf.reshape(x, (-1,))
        elif axis != 0:
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`axis={axis}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            return tfnp.take(x, tf.sparse.to_dense(indices), axis=axis)
        return tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=x,
            sparse_ids=tf.sparse.expand_dims(indices, axis=-1),
            default_id=0,
        )
    return tfnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    return tfnp.take_along_axis(x, indices, axis=axis)


def tan(x):
    return tfnp.tan(x)


def tanh(x):
    return tfnp.tanh(x)


def tensordot(x1, x2, axes=2):
    return tfnp.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    return tfnp.round(x, decimals=decimals)


def tile(x, repeats):
    # The TFNP implementation is buggy, we roll our own.
    x = convert_to_tensor(x)
    repeats = tf.reshape(convert_to_tensor(repeats, dtype="int32"), [-1])
    repeats_size = tf.size(repeats)
    repeats = tf.pad(
        repeats,
        [[tf.maximum(x.shape.rank - repeats_size, 0), 0]],
        constant_values=1,
    )
    x_shape = tf.pad(
        tf.shape(x),
        [[tf.maximum(repeats_size - x.shape.rank, 0), 0]],
        constant_values=1,
    )
    x = tf.reshape(x, x_shape)
    return tf.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    return tfnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    return tfnp.tril(x, k=k)


def triu(x, k=0):
    return tfnp.triu(x, k=k)


def vdot(x1, x2):
    return tfnp.vdot(x1, x2)


def vstack(xs):
    return tfnp.vstack(xs)


def where(condition, x1, x2):
    return tfnp.where(condition, x1, x2)


def divide(x1, x2):
    return tfnp.divide(x1, x2)


def true_divide(x1, x2):
    return tfnp.true_divide(x1, x2)


def power(x1, x2):
    return tfnp.power(x1, x2)


def negative(x):
    return tfnp.negative(x)


def square(x):
    return tfnp.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    # upcast to float64 for int64 which matches JAX's behavior
    dtype = (
        "float64"
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    # TODO: Use tfnp.sqrt. Currently, tfnp.sqrt will aggressively upcast to
    # float64 if the input is bfloat16. This behavior mismatches with JAX.
    return tf.sqrt(x)


def squeeze(x, axis=None):
    if isinstance(x, tf.SparseTensor):
        new_shape = list(x.shape)
        gather_indices = list(range(len(new_shape)))
        if axis is None:
            for i in range(len(new_shape) - 1, -1, -1):
                if new_shape[i] == 1:
                    del new_shape[i]
                    del gather_indices[i]
        else:
            if new_shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {axis}, because the "
                    "dimension is not 1."
                )
            del new_shape[axis]
            del gather_indices[axis]
        new_indices = tf.gather(x.indices, gather_indices, axis=1)
        return tf.SparseTensor(new_indices, x.values, tuple(new_shape))
    return tfnp.squeeze(x, axis=axis)


def transpose(x, axes=None):
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.transpose(x, perm=axes)
    return tfnp.transpose(x, axes=axes)


def var(x, axis=None, keepdims=False):
    return tfnp.var(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    return tfnp.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.eye(N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    return tfnp.floor_divide(x1, x2)


def logical_xor(x1, x2):
    return tfnp.logical_xor(x1, x2)
