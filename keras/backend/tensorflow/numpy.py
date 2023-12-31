import builtins
import collections
import math
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

from keras.backend import config
from keras.backend import standardize_dtype
from keras.backend.common import dtypes
from keras.backend.tensorflow import sparse
from keras.backend.tensorflow.core import convert_to_tensor


@sparse.elementwise_binary_union(tf.sparse.add)
def add(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.add(x1, x2)


def bincount(x, weights=None, minlength=0):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype]
    if standardize_dtype(x.dtype) not in ["int32", "int64"]:
        x = tf.cast(x, tf.int32)
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
        if standardize_dtype(weights.dtype) not in [
            "int32",
            "int64",
            "float32",
            "float64",
        ]:
            if "int" in standardize_dtype(weights.dtype):
                weights = tf.cast(weights, tf.int32)
            else:
                weights = tf.cast(weights, tf.float32)
    else:
        dtype = "int32"
    if isinstance(x, tf.SparseTensor):
        output = tf.sparse.bincount(
            x,
            weights=weights,
            minlength=minlength,
            axis=-1,
        )
        output = tf.cast(output, dtype)
        if x.shape.rank == 1:
            output_shape = (minlength,)
        else:
            batch_size = tf.shape(output)[0]
            output_shape = (batch_size, minlength)
        return tf.SparseTensor(
            indices=output.indices,
            values=output.values,
            dense_shape=output_shape,
        )
    return tf.cast(
        tf.math.bincount(x, weights=weights, minlength=minlength, axis=-1),
        dtype,
    )


def einsum(subscripts, *operands, **kwargs):
    operands = tf.nest.map_structure(convert_to_tensor, operands)

    dtypes_to_resolve = []
    for x in operands:
        dtypes_to_resolve.append(x.dtype)
    result_dtype = dtypes.result_type(*dtypes_to_resolve)
    compute_dtype = result_dtype
    # TODO: tf.einsum doesn't support integer dtype with gpu
    if "int" in compute_dtype:
        compute_dtype = config.floatx()

    operands = tf.nest.map_structure(
        lambda x: tf.cast(x, compute_dtype), operands
    )
    return tf.cast(tf.einsum(subscripts, *operands, **kwargs), result_dtype)


@sparse.elementwise_binary_union(sparse.sparse_subtract)
def subtract(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_shape = x1.shape
    x2_shape = x2.shape
    # TODO: GPU and XLA only support float types
    compute_dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)

    def with_combined_batch_dimensions(a, b, output_shape, fn_3d):
        a_sparse = isinstance(a, tf.SparseTensor)
        b_sparse = isinstance(b, tf.SparseTensor)
        batch_shape = b.shape[:-2] if b_sparse else a.shape[:-2]
        batch_size = math.prod(batch_shape)
        a3d_shape = [batch_size] + a.shape[-2:]
        a_3d = (
            tf.sparse.reshape(a, a3d_shape)
            if a_sparse
            else tf.reshape(a, a3d_shape)
        )
        b3d_shape = [batch_size] + b.shape[-2:]
        b_3d = (
            tf.sparse.reshape(b, b3d_shape)
            if b_sparse
            else tf.reshape(b, b3d_shape)
        )
        result_3d = fn_3d(a_3d, b_3d)
        return (
            tf.sparse.reshape(result_3d, output_shape)
            if isinstance(result_3d, tf.SparseTensor)
            else tf.reshape(result_3d, output_shape)
        )

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
    if x1_sparse or x2_sparse:
        from keras.ops.operation_utils import compute_matmul_output_shape

        output_shape = compute_matmul_output_shape(x1_shape, x2_shape)
        if x1_sparse and x2_sparse:
            if x1_shape.rank <= 3:
                output = sparse_sparse_matmul(x1, x2)
            else:
                output = with_combined_batch_dimensions(
                    x1, x2, output_shape, sparse_sparse_matmul
                )
        else:
            # Sparse * dense or dense * sparse
            sparse_rank = x1_shape.rank if x1_sparse else x2_shape.rank

            # Special case: embedding_lookup_sparse for sparse * dense, rank 2
            if x1_sparse and sparse_rank == 2:
                output = embedding_lookup_sparse_dense_matmul(x1, x2)
            elif sparse_rank == 2:
                output = tf.sparse.sparse_dense_matmul(x1, x2)
            elif sparse_rank == 3:
                output = sparse_dense_matmul_3d(x1, x2)
            else:
                output = with_combined_batch_dimensions(
                    x1, x2, output_shape, sparse_dense_matmul_3d
                )
        output = tf.cast(output, result_dtype)
        output.set_shape(output_shape)
        return output
    else:
        if x1_shape.rank == 2 and x2_shape.rank == 2:
            output = tf.matmul(x1, x2)
        elif x2_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=1)
        elif x1_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=[[0], [-2]])
        else:
            output = tf.matmul(x1, x2)
        return tf.cast(output, result_dtype)


@sparse.elementwise_binary_intersection
def multiply(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    if isinstance(x, tf.IndexedSlices):
        if axis is None:
            # Reduce against all axes, result is a single value and dense.
            # The denominator has to account for `dense_shape`.
            sum = tf.reduce_sum(x.values, keepdims=keepdims)
            return sum / tf.cast(tf.reduce_prod(x.dense_shape), dtype=sum.dtype)

        if isinstance(axis, int):
            axis = [axis]
        elif not axis:
            # Empty axis tuple, this is a no-op
            return x

        dense_shape = tf.convert_to_tensor(x.dense_shape)
        rank = tf.shape(dense_shape)[0]
        # Normalize axis: convert negative values and sort
        axis = [rank + a if a < 0 else a for a in axis]
        axis.sort()

        if axis == [0]:
            # Reduce against `axis=0` only, result is dense.
            # The denominator has to account for `dense_shape[0]`.
            sum = tf.reduce_sum(x.values, axis=0, keepdims=keepdims)
            return sum / tf.cast(dense_shape[0], dtype=sum.dtype)
        elif axis[0] == 0:
            # Reduce against axis 0 and other axes, result is dense.
            # We do `axis=0` separately first. The denominator has to account
            # for `dense_shape[0]`.
            # We use `keepdims=True` in `reduce_sum`` so that we can leave the
            # 0 in axis and do `reduce_mean` with `keepdims` to apply it for all
            # axes.
            sum = tf.reduce_sum(x.values, axis=0, keepdims=True)
            axis_0_mean = sum / tf.cast(dense_shape[0], dtype=sum.dtype)
            return tf.reduce_mean(axis_0_mean, axis=axis, keepdims=keepdims)
        elif keepdims:
            # With `keepdims=True`, result is an `IndexedSlices` with the same
            # indices since axis 0 is not touched. The only thing to do is to
            # correct `dense_shape` to account for dimensions that became 1.
            new_values = tf.reduce_mean(x.values, axis=axis, keepdims=True)
            new_dense_shape = tf.concat(
                [dense_shape[0:1], new_values.shape[1:]], axis=0
            )
            return tf.IndexedSlices(new_values, x.indices, new_dense_shape)
        elif rank == len(axis) + 1:
            # `keepdims=False` and reducing against all axes exept 0, result is
            # a 1D tensor, which cannot be `IndexedSlices`. We have to scatter
            # the computed means to construct the correct dense tensor.
            return tf.scatter_nd(
                tf.expand_dims(x.indices, axis=1),
                tf.reduce_mean(x.values, axis=axis),
                [dense_shape[0]],
            )
        else:
            # `keepdims=False`, not reducing against axis 0 and there is at
            # least one other axis we are not reducing against. We simply need
            # to fix `dense_shape` to remove dimensions that were reduced.
            gather_indices = [i for i in range(rank) if i not in axis]
            return tf.IndexedSlices(
                tf.reduce_mean(x.values, axis=axis),
                x.indices,
                tf.gather(x.dense_shape, gather_indices, axis=0),
            )
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    # `tf.reduce_mean` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype
    output = tf.reduce_mean(
        tf.cast(x, compute_dtype), axis=axis, keepdims=keepdims
    )
    return tf.cast(output, result_dtype)


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


@sparse.elementwise_unary
def absolute(x):
    # uintx and bool are always non-negative
    dtype = standardize_dtype(x.dtype)
    if "uint" in dtype or dtype == "bool":
        return x
    return tf.abs(x)


@sparse.elementwise_unary
def abs(x):
    return absolute(x)


def all(x, axis=None, keepdims=False):
    x = tf.cast(x, "bool")
    return tf.reduce_all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    x = tf.cast(x, "bool")
    return tf.reduce_any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    return max(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    return min(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if axis is None:
        return tf.concat([tf.reshape(x1, [-1]), tf.reshape(x2, [-1])], axis=0)
    else:
        return tf.concat([x1, x2], axis=axis)


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
    dtype = standardize_dtype(dtype)
    return tf.range(start, stop, delta=step, dtype=dtype)


@sparse.densifying_unary(0.5 * np.pi)
def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.acos(x)


@sparse.densifying_unary(np.nan)
def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.acosh(x)


@sparse.elementwise_unary
def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.asin(x)


@sparse.elementwise_unary
def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.asinh(x)


@sparse.elementwise_unary
def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.atan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.math.atan2(x1, x2)


@sparse.elementwise_unary
def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.atanh(x)


def argmax(x, axis=None):
    if axis is None:
        x = tf.reshape(x, [-1])
    return tf.cast(tf.argmax(x, axis=axis), dtype="int32")


def argmin(x, axis=None):
    if axis is None:
        x = tf.reshape(x, [-1])
    return tf.cast(tf.argmin(x, axis=axis), dtype="int32")


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "uint8")

    x_shape = x.shape
    if x_shape.rank == 0:
        return tf.cast([0], "int32")

    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.argsort(x, axis=axis)


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    result_dtype = dtypes.result_type(*dtypes_to_resolve)
    compute_dtype = result_dtype
    # TODO: since tfnp.average incorrectly promote bfloat16 to float64, we
    # need to cast to float32 first and then cast back to bfloat16
    if compute_dtype == "bfloat16":
        compute_dtype = "float32"
    x = tf.cast(x, compute_dtype)
    if weights is not None:
        weights = tf.cast(weights, compute_dtype)
    for a in axis:
        # `tfnp.average` does not handle multiple axes.
        x = tfnp.average(x, weights=weights, axis=a)
    return tf.cast(x, result_dtype)


def broadcast_to(x, shape):
    return tf.broadcast_to(x, shape)


@sparse.elementwise_unary
def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.ceil(x)


def clip(x, x_min, x_max):
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        x = tf.cast(x, "int32")
    return tf.clip_by_value(x, x_min, x_max)


def concatenate(xs, axis=0):
    sparse_count = builtins.sum(isinstance(x, tf.SparseTensor) for x in xs)
    if sparse_count:
        if sparse_count == len(xs):
            return tf.sparse.concat(axis=axis, sp_inputs=xs)
        else:
            xs = [
                convert_to_tensor(x, sparse=False)
                if isinstance(x, tf.SparseTensor)
                else x
                for x in xs
            ]
    xs = tf.nest.map_structure(convert_to_tensor, xs)
    dtype_set = set([x.dtype for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tf.nest.map_structure(lambda x: tf.cast(x, dtype), xs)
    return tf.concat(xs, axis=axis)


@sparse.elementwise_unary
def conjugate(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def conj(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def copy(x):
    return tfnp.copy(x)


@sparse.densifying_unary(1)
def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.cos(x)


@sparse.densifying_unary(1)
def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.cosh(x)


def count_nonzero(x, axis=None):
    return tf.math.count_nonzero(x, axis=axis, dtype="int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tfnp.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    # tf.math.cumprod doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.math.cumprod(x, axis=axis)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x, dtype=dtype)
    # tf.math.cumprod doesn't support bool
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    return tf.math.cumsum(x, axis=axis)


def diag(x, k=0):
    return tfnp.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return tfnp.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def diff(a, n=1, axis=-1):
    return tfnp.diff(a, n=n, axis=axis)


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = list(bins)

    # bins must be float type
    bins = tf.nest.map_structure(lambda x: float(x), bins)

    # TODO: tf.raw_ops.Bucketize doesn't support bool, bfloat16, float16, int8
    # int16, uint8, uint16, uint32
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype in ("bool", "int8", "int16", "uint8", "uint16"):
        x = tf.cast(x, "int32")
    elif ori_dtype == "uint32":
        x = tf.cast(x, "int64")
    elif ori_dtype in ("bfloat16", "float16"):
        x = tf.cast(x, "float32")

    if isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(
            lambda y: tf.raw_ops.Bucketize(input=y, boundaries=bins), x
        )
    elif isinstance(x, tf.SparseTensor):
        output = tf.SparseTensor(
            indices=tf.identity(x.indices),
            values=tf.raw_ops.Bucketize(input=x.values, boundaries=bins),
            dense_shape=tf.identity(x.dense_shape),
        )
        output.set_shape(x.shape)
        return output
    return tf.raw_ops.Bucketize(input=x, boundaries=bins)


def dot(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    result_dtype = dtypes.result_type(x.dtype, y.dtype)
    # GPU only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)
    x = tf.cast(x, compute_dtype)
    y = tf.cast(y, compute_dtype)

    x_shape = x.shape
    y_shape = y.shape
    if x_shape.rank == 0 or y_shape.rank == 0:
        output = x * y
    elif y_shape.rank == 1:
        output = tf.tensordot(x, y, axes=[[-1], [-1]])
    else:
        output = tf.tensordot(x, y, axes=[[-1], [-2]])
    return tf.cast(output, result_dtype)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.zeros(shape, dtype=dtype)


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.equal(x1, x2)


@sparse.densifying_unary(1)
def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.exp(x)


def expand_dims(x, axis):
    if isinstance(x, tf.SparseTensor):
        from keras.ops.operation_utils import compute_expand_dims_output_shape

        output = tf.sparse.expand_dims(x, axis)
        output.set_shape(compute_expand_dims_output_shape(x.shape, axis))
        return output
    return tf.expand_dims(x, axis)


@sparse.elementwise_unary
def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.expm1(x)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        return tf.reverse(x, tf.range(tf.rank(x)))
    return tf.reverse(x, [axis])


@sparse.elementwise_unary
def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    fill_value = convert_to_tensor(fill_value, dtype)
    return tf.broadcast_to(fill_value, shape)


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    fill_value = convert_to_tensor(fill_value, dtype)
    return tf.broadcast_to(fill_value, tf.shape(x))


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.greater_equal(x1, x2)


def hstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tf.nest.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    rank = tf.rank(xs[0])
    return tf.cond(
        tf.equal(rank, 1),
        lambda: tf.concat(xs, axis=0),
        lambda: tf.concat(xs, axis=1),
    )


def identity(n, dtype=None):
    return eye(N=n, M=n, dtype=dtype)


@sparse.elementwise_unary
def imag(x):
    return tf.math.imag(x)


def isclose(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if "float" in dtype:
        # atol defaults to 1e-08
        # rtol defaults to 1e-05
        return tf.abs(x1 - x2) <= (1e-08 + 1e-05 * tf.abs(x2))
    else:
        return tf.equal(x1, x2)


@sparse.densifying_unary(True)
def isfinite(x):
    # `tfnp.isfinite` requires `enable_numpy_behavior`, so we reimplement it.
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.ones(x.shape, tf.bool)
    return tf.math.is_finite(x)


def isinf(x):
    # `tfnp.isinf` requires `enable_numpy_behavior`, so we reimplement it.
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_inf(x)


def isnan(x):
    # `tfnp.isnan` requires `enable_numpy_behavior`, so we reimplement it.
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_nan(x)


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    return tfnp.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


@sparse.densifying_unary(-np.inf)
def log(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x)


@sparse.densifying_unary(-np.inf)
def log10(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x) / tf.math.log(tf.constant(10, x.dtype))


@sparse.elementwise_unary
def log1p(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log1p(x)


@sparse.densifying_unary(-np.inf)
def log2(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.log(x) / tf.math.log(tf.constant(2, x.dtype))


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    # Below is the same implementation as tfnp.logaddexp using all native
    # ops to prevent incorrect promotion of bfloat16.
    delta = x1 - x2
    return tf.where(
        tf.math.is_nan(delta),
        x1 + x2,
        tf.maximum(x1, x2) + tf.math.log1p(tf.math.exp(-tf.abs(delta))),
    )


def logical_and(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.logical_and(x1, x2)


def logical_not(x):
    x = tf.cast(x, "bool")
    return tf.logical_not(x)


def logical_or(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    start = tf.cast(start, dtype)
    stop = tf.cast(stop, dtype)
    return tfnp.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


@sparse.elementwise_binary_union(tf.sparse.maximum, densify_mixed=True)
def maximum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    return quantile(x, 0.5, axis=axis, keepdims=keepdims)


def meshgrid(*x, indexing="xy"):
    return tf.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
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


@sparse.elementwise_binary_union(tf.sparse.minimum, densify_mixed=True)
def minimum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.minimum(x1, x2)


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.math.mod(x1, x2)


def moveaxis(x, source, destination):
    return tfnp.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    x = convert_to_tensor(x)

    dtype = x.dtype
    dtype_as_dtype = tf.as_dtype(dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return x

    # Replace NaN with 0
    x = tf.where(tf.math.is_nan(x), tf.constant(0, dtype), x)

    # Replace positive infinitiy with dtype.max
    x = tf.where(tf.math.is_inf(x) & (x > 0), tf.constant(dtype.max, dtype), x)

    # Replace negative infinity with dtype.min
    x = tf.where(tf.math.is_inf(x) & (x < 0), tf.constant(dtype.min, dtype), x)

    return x


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    x = convert_to_tensor(x)
    result = tf.unstack(tf.where(tf.cast(x, "bool")), x.shape.rank, axis=1)
    return tf.nest.map_structure(
        lambda indices: tf.cast(indices, "int32"), result
    )


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return tf.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return tf.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    return tf.reshape(x1, [-1, 1]) * tf.reshape(x2, [-1])


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    kwargs = {}
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        kwargs["constant_values"] = constant_values
    pad_width = convert_to_tensor(pad_width, "int32")
    return tf.pad(x, pad_width, mode.upper(), **kwargs)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = dtypes.result_type(x.dtype)
        if dtype == "bool":
            dtype = "int32"
        elif dtype in ("int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
        x = tf.cast(x, dtype)
    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)


def _quantile(x, q, axis=None, method="linear", keepdims=False):
    # ref: tfp.stats.percentile
    # float64 is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = tf.cast(q, "float64")

    # Move `axis` dims of `x` to the rightmost, call it `y`.
    if axis is None:
        y = tf.reshape(x, [-1])
    else:
        x_ndims = len(x.shape)

        # _make_static_axis_non_negative_list
        axis = list(map(lambda x: x if x >= 0 else x + x_ndims, axis))

        # _move_dims_to_flat_end
        other_dims = sorted(set(range(x_ndims)).difference(axis))
        perm = other_dims + list(axis)
        x_permed = tf.transpose(a=x, perm=perm)
        if None not in x.shape:
            x_shape = list(x.shape)
            other_shape = [x_shape[i] for i in other_dims]
            end_shape = [math.prod([x_shape[i] for i in axis])]
            full_shape = other_shape + end_shape
        else:
            other_shape = tf.gather(tf.shape(x), tf.cast(other_dims, tf.int64))
            full_shape = tf.concat([other_shape, [-1]], axis=0)
        y = tf.reshape(x_permed, shape=full_shape)

    # Sort (in ascending order) everything which allows multiple calls to sort
    # only once (under the hood) and use CSE.
    sorted_y = tf.sort(y, axis=-1, direction="ASCENDING")

    d = tf.cast(tf.shape(y)[-1], "float64")

    def _get_indices(method):
        """Get values of y at the indices implied by method."""
        if method == "lower":
            indices = tf.math.floor((d - 1) * q)
        elif method == "higher":
            indices = tf.math.ceil((d - 1) * q)
        elif method == "nearest":
            indices = tf.round((d - 1) * q)
        # d - 1 will be distinct from d in int32, but not necessarily double.
        # So clip to avoid out of bounds errors.
        return tf.clip_by_value(
            tf.cast(indices, "int32"), 0, tf.shape(y)[-1] - 1
        )

    if method in ["nearest", "lower", "higher"]:
        gathered_y = tf.gather(sorted_y, _get_indices(method), axis=-1)
    elif method == "midpoint":
        gathered_y = 0.5 * (
            tf.gather(sorted_y, _get_indices("lower"), axis=-1)
            + tf.gather(sorted_y, _get_indices("higher"), axis=-1)
        )
    elif method == "linear":
        larger_y_idx = _get_indices("higher")
        exact_idx = (d - 1) * q
        # preserve_gradients
        smaller_y_idx = tf.maximum(larger_y_idx - 1, 0)
        larger_y_idx = tf.minimum(smaller_y_idx + 1, tf.shape(y)[-1] - 1)
        fraction = tf.cast(larger_y_idx, tf.float64) - exact_idx
        fraction = tf.cast(fraction, y.dtype)
        gathered_y = (
            tf.gather(sorted_y, larger_y_idx, axis=-1) * (1 - fraction)
            + tf.gather(sorted_y, smaller_y_idx, axis=-1) * fraction
        )

    # Propagate NaNs
    if x.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
        # Apparently tf.is_nan doesn't like other dtypes
        nan_batch_members = tf.reduce_any(tf.math.is_nan(x), axis=axis)
        right_rank_matched_shape = tf.pad(
            tf.shape(nan_batch_members),
            paddings=[[0, tf.rank(q)]],
            constant_values=1,
        )
        nan_batch_members = tf.reshape(
            nan_batch_members, shape=right_rank_matched_shape
        )
        gathered_y = tf.where(nan_batch_members, float("NaN"), gathered_y)

    # Expand dimensions if requested
    if keepdims:
        if axis is None:
            ones_vec = tf.ones(shape=[tf.rank(x) + tf.rank(q)], dtype="int32")
            gathered_y *= tf.ones(ones_vec, dtype=gathered_y.dtype)
        else:
            for i in sorted(axis):
                gathered_y = tf.expand_dims(gathered_y, axis=i)

    # rotate_transpose
    shift_value_static = tf.get_static_value(tf.rank(q))
    ndims = tf.TensorShape(gathered_y.shape).rank
    if ndims < 2:
        return gathered_y
    shift_value_static = int(
        math.copysign(1, shift_value_static)
        * (builtins.abs(shift_value_static) % ndims)
    )
    if shift_value_static == 0:
        return gathered_y
    perm = collections.deque(range(ndims))
    perm.rotate(shift_value_static)
    return tf.transpose(a=gathered_y, perm=perm)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    if isinstance(axis, int):
        axis = [axis]

    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    compute_dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, compute_dtype)
    return _quantile(x, q, axis=axis, method=method, keepdims=keepdims)


def ravel(x):
    x = convert_to_tensor(x)
    return tf.reshape(x, [-1])


@sparse.elementwise_unary
def real(x):
    x = convert_to_tensor(x)
    return tf.math.real(x)


@sparse.densifying_unary(np.inf)
def reciprocal(x):
    x = convert_to_tensor(x)
    return tf.math.reciprocal(x)


def repeat(x, repeats, axis=None):
    # tfnp.repeat has trouble with dynamic Tensors in compiled function.
    # tf.repeat does not.
    x = convert_to_tensor(x)
    # TODO: tf.repeat doesn't support uint16
    if standardize_dtype(x.dtype) == "uint16":
        x = tf.cast(x, "uint32")
        return tf.cast(tf.repeat(x, repeats, axis=axis), "uint16")
    return tf.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    x = convert_to_tensor(x)
    if isinstance(x, tf.SparseTensor):
        from keras.ops.operation_utils import compute_reshape_output_shape

        output_shape = compute_reshape_output_shape(
            x.shape, new_shape, "new_shape"
        )
        output = tf.sparse.reshape(x, new_shape)
        output.set_shape(output_shape)
        return output
    return tf.reshape(x, new_shape)


def roll(x, shift, axis=None):
    return tfnp.roll(x, shift, axis=axis)


@sparse.elementwise_unary
def sign(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # TODO: tf.sign doesn't support uint8, uint16, uint32
    if ori_dtype in ("uint8", "uint16", "uint32"):
        x = tf.cast(x, "int32")
        return tf.cast(tf.sign(x), ori_dtype)
    return tf.sign(x)


@sparse.elementwise_unary
def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.sin(x)


@sparse.elementwise_unary
def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    return tf.size(x)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # TODO: tf.sort doesn't support bool
    if ori_dtype == "bool":
        x = tf.cast(x, "int8")
        return tf.cast(tf.sort(x, axis=axis), ori_dtype)
    return tf.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    if not isinstance(indices_or_sections, int):
        # `tf.split` requires `num_or_size_splits`, so we need to convert
        # `indices_or_sections` to the appropriate format.
        # The following implementation offers better compatibility for the
        # tensor argument `indices_or_sections` than original `tfnp.split`.
        total_size = x.shape[axis]
        indices_or_sections = convert_to_tensor(indices_or_sections)
        start_size = indices_or_sections[0:1]
        end_size = total_size - indices_or_sections[-1:]
        num_or_size_splits = tf.concat(
            [start_size, tfnp.diff(indices_or_sections), end_size], axis=0
        )
    else:
        num_or_size_splits = indices_or_sections
    return tf.split(x, num_or_size_splits, axis=axis)


def stack(x, axis=0):
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tf.nest.map_structure(lambda a: convert_to_tensor(a, dtype), x)
    return tf.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


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
            return tfnp.take(
                x, convert_to_tensor(indices, sparse=False), axis=axis
            )
        if axis is None:
            x = tf.reshape(x, (-1,))
        elif axis != 0:
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`axis={axis}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            return tfnp.take(
                x, convert_to_tensor(indices, sparse=False), axis=axis
            )
        output = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=tf.convert_to_tensor(x),
            sparse_ids=tf.sparse.expand_dims(indices, axis=-1),
            default_id=0,
        )
        output.set_shape(indices.shape + output.shape[len(indices.shape) :])
        return output
    return tfnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    return tfnp.take_along_axis(x, indices, axis=axis)


@sparse.elementwise_unary
def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.tan(x)


@sparse.elementwise_unary
def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, dtype)
    return tf.math.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # TODO: tf.tensordot only supports float types
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    return tf.cast(tf.tensordot(x1, x2, axes=axes), dtype=result_dtype)


@sparse.elementwise_unary
def round(x, decimals=0):
    # `tfnp.round` requires `enable_numpy_behavior`, so we reimplement it.
    if decimals == 0:
        return tf.round(x)
    x_dtype = x.dtype
    if tf.as_dtype(x_dtype).is_integer:
        # int
        if decimals > 0:
            return x
        # temporarilaly convert to floats
        factor = tf.cast(math.pow(10, decimals), config.floatx())
        x = tf.cast(x, config.floatx())
    else:
        # float
        factor = tf.cast(math.pow(10, decimals), x.dtype)
    x = tf.multiply(x, factor)
    x = tf.round(x)
    x = tf.divide(x, factor)
    return tf.cast(x, x_dtype)


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
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype not in ("int64", "uint32", "uint64"):
        dtype = dtypes.result_type(dtype, "int32")
    return tfnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    x = convert_to_tensor(x)
    if k >= 0:
        return tf.linalg.band_part(x, -1, k)

    # deal with negative k using mask
    k = -k - 1
    mask = tf.ones_like(x, dtype="bool")
    mask = tf.logical_not(tf.linalg.band_part(mask, k, -1))
    return tf.where(mask, x, tf.constant(0, x.dtype))


def triu(x, k=0):
    x = convert_to_tensor(x)
    if k >= 0:
        return tf.linalg.band_part(x, k, -1)

    # deal with negative k using mask
    k = -k
    mask = tf.ones_like(x, dtype="bool")
    mask = tf.logical_not(tf.linalg.band_part(mask, k, -1))
    return tf.where(mask, tf.constant(0, x.dtype), x)


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = tf.cast(x1, compute_dtype)
    x2 = tf.cast(x2, compute_dtype)
    x1 = tf.reshape(x1, [-1])
    x2 = tf.reshape(x2, [-1])
    return tf.cast(dot(x1, x2), result_dtype)


def vstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tf.nest.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    return tf.concat(xs, axis=0)


def where(condition, x1, x2):
    condition = tf.cast(condition, "bool")
    if x1 is not None and x2 is not None:
        if not isinstance(x1, (int, float)):
            x1 = convert_to_tensor(x1)
        if not isinstance(x2, (int, float)):
            x2 = convert_to_tensor(x2)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1 = convert_to_tensor(x1, dtype)
        x2 = convert_to_tensor(x2, dtype)
        return tf.where(condition, x1, x2)
    if x1 is None and x2 is None:
        return nonzero(condition)
    raise ValueError(
        "`x1` and `x2` either both should be `None`"
        " or both should have non-None value."
    )


@sparse.elementwise_division
def divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.divide(x1, x2)


@sparse.elementwise_division
def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    # TODO: tf.pow doesn't support uint* types
    if "uint" in dtype:
        x1 = convert_to_tensor(x1, "int32")
        x2 = convert_to_tensor(x2, "int32")
        return tf.cast(tf.pow(x1, x2), dtype)
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.pow(x1, x2)


@sparse.elementwise_unary
def negative(x):
    return tf.negative(x)


@sparse.elementwise_unary
def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    return tf.square(x)


@sparse.elementwise_unary
def sqrt(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = tf.cast(x, dtype)
    return tf.math.sqrt(x)


def squeeze(x, axis=None):
    if isinstance(x, tf.SparseTensor):
        static_shape = x.shape.as_list()
        if axis is not None and static_shape[axis] != 1:
            raise ValueError(
                f"Cannot squeeze axis {axis}, because the "
                "dimension is not 1."
            )
        dynamic_shape = tf.shape(x)
        new_shape = []
        gather_indices = []
        for i, dim in enumerate(static_shape):
            if not (dim == 1 if axis is None else i == axis):
                new_shape.append(dim if dim is not None else dynamic_shape[i])
                gather_indices.append(i)
        new_indices = tf.gather(x.indices, gather_indices, axis=1)
        return tf.SparseTensor(new_indices, x.values, tuple(new_shape))
    return tf.squeeze(x, axis=axis)


def transpose(x, axes=None):
    if isinstance(x, tf.SparseTensor):
        from keras.ops.operation_utils import compute_transpose_output_shape

        output = tf.sparse.transpose(x, perm=axes)
        output.set_shape(compute_transpose_output_shape(x.shape, axes))
        return output
    return tf.transpose(x, perm=axes)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    x = tf.cast(x, compute_dtype)
    return tf.cast(
        tf.math.reduce_variance(x, axis=axis, keepdims=keepdims),
        result_dtype,
    )


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    # follow jax's rule
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = tf.cast(x, dtype)
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return tfnp.eye(N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return tf.math.floordiv(x1, x2)


def logical_xor(x1, x2):
    x1 = tf.cast(x1, "bool")
    x2 = tf.cast(x2, "bool")
    return tf.math.logical_xor(x1, x2)
