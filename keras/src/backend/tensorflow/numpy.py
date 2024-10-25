import builtins
import collections
import functools
import math
import string
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.ops.math_ops import is_nan

from keras.src import tree
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.common.backend_utils import vectorize_impl
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
from keras.src.backend.tensorflow.core import shape as shape_op


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

    # Special case of `tf.add`: `tf.nn.bias_add`
    # `BiasAdd` can be fused with `MatMul` and `Conv*` kernels
    # Expecting `x1` to be `inputs` and `x2` to be `bias` (no swapping)
    x2_squeeze_shape = [d for d in x2.shape if d is None or d > 1]
    if (
        # `x2` looks like bias (can be squeezed to vector)
        1 == len(x2_squeeze_shape)
        # `x1` looks like input tensor (rank >= 2)
        and len(x1.shape) > 1
        # `x2` non-squeezable dimension defined
        and x2_squeeze_shape[0] is not None
        # `x2` non-squeezable dimension match `x1` channel dimension
        and x2_squeeze_shape[0]
        in {x1.shape.as_list()[1], x1.shape.as_list()[-1]}
    ):
        if x1.shape[-1] == x2_squeeze_shape[0]:
            data_format = "NHWC"
        else:
            data_format = "NCHW"
        if len(x2.shape) > 1:
            x2 = tf.squeeze(x2)
        return tf.nn.bias_add(x1, x2, data_format=data_format)

    return tf.add(x1, x2)


def bincount(x, weights=None, minlength=0, sparse=False):
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
    if sparse or isinstance(x, tf.SparseTensor):
        output = tf.sparse.bincount(
            x,
            weights=weights,
            minlength=minlength,
            axis=-1,
        )
        actual_length = output.shape[-1]
        if actual_length is None:
            actual_length = tf.shape(output)[-1]
        output = cast(output, dtype)
        if x.shape.rank == 1:
            output_shape = (actual_length,)
        else:
            batch_size = output.shape[0]
            if batch_size is None:
                batch_size = tf.shape(output)[0]
            output_shape = (batch_size, actual_length)
        return tf.SparseTensor(
            indices=output.indices,
            values=output.values,
            dense_shape=output_shape,
        )
    return tf.cast(
        tf.math.bincount(x, weights=weights, minlength=minlength, axis=-1),
        dtype,
    )


@functools.lru_cache(512)
def _normalize_einsum_subscripts(subscripts):
    # string.ascii_letters
    mapping = {}
    normalized_subscripts = ""
    for c in subscripts:
        if c in string.ascii_letters:
            if c not in mapping:
                mapping[c] = string.ascii_letters[len(mapping)]
            normalized_subscripts += mapping[c]
        else:
            normalized_subscripts += c
    return normalized_subscripts


def einsum(subscripts, *operands, **kwargs):
    operands = tree.map_structure(convert_to_tensor, operands)
    subscripts = _normalize_einsum_subscripts(subscripts)

    def is_valid_for_custom_ops(subscripts, *operands):
        # Check that `subscripts` is supported and the shape of operands is not
        # `None`.
        if subscripts in [
            "a,b->ab",
            "ab,b->a",
            "ab,bc->ac",
            "ab,cb->ac",
            "abc,cd->abd",
            "abc,dc->abd",
            "abcd,abde->abce",
            "abcd,abed->abce",
            "abcd,acbe->adbe",
            "abcd,adbe->acbe",
            "abcd,aecd->acbe",
            "abcd,aecd->aceb",
        ]:
            # These subscripts don't require the shape information
            return True
        elif subscripts == "abc,cde->abde":
            _, b1, c1 = operands[0].shape
            c2, d2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abc,dce->abde":
            _, b1, c1 = operands[0].shape
            d2, c2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abc,dec->abde":
            _, b1, c1 = operands[0].shape
            d2, e2, c2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,cde->abe":
            _, b1, c1, d1 = operands[0].shape
            c2, d2, e2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,ced->abe":
            _, b1, c1, d1 = operands[0].shape
            c2, e2, d2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcd,ecd->abe":
            _, b1, c1, d1 = operands[0].shape
            e2, c2, d2 = operands[1].shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            if None in (b, c, d, e):
                return False
            return True
        elif subscripts == "abcde,aebf->adbcf":
            _, b1, c1, d1, e1 = operands[0].shape
            _, e2, b2, f2 = operands[1].shape
            b, c, d, e, f = b1 or b2, c1, d1, e1 or e2, f2
            if None in (b, c, d, e, f):
                return False
            return True
        elif subscripts == "abcde,afce->acdbf":
            _, b1, c1, d1, e1 = operands[0].shape
            _, f2, c2, e2 = operands[1].shape
            b, c, d, e, f = b1, c1 or c2, d1, e1 or e2, f2
            if None in (b, c, d, e, f):
                return False
            return True
        else:
            # No match in subscripts
            return False

    def use_custom_ops(subscripts, *operands, output_type):
        # Replace tf.einsum with custom ops to utilize hardware-accelerated
        # matmul
        x, y = operands[0], operands[1]
        if subscripts == "a,b->ab":
            x = tf.expand_dims(x, axis=-1)
            y = tf.expand_dims(y, axis=0)
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "ab,b->a":
            y = tf.expand_dims(y, axis=-1)
            result = tf.matmul(x, y, output_type=output_type)
            return tf.squeeze(result, axis=-1)
        elif subscripts == "ab,bc->ac":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "ab,cb->ac":
            y = tf.transpose(y, [1, 0])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,cd->abd":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,cde->abde":
            _, b1, c1 = x.shape
            c2, d2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abc,dc->abd":
            y = tf.transpose(y, [1, 0])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abc,dce->abde":
            _, b1, c1 = x.shape
            d2, c2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.transpose(y, [1, 0, 2])  # cde
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abc,dec->abde":
            _, b1, c1 = x.shape
            d2, e2, c2 = y.shape
            b, c, d, e = b1, c1 or c2, d2, e2
            y = tf.transpose(y, [2, 0, 1])  # cde
            y = tf.reshape(y, [c, -1])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.reshape(result, [-1, b, d, e])
        elif subscripts == "abcd,abde->abce":
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,abed->abce":
            y = tf.transpose(y, [0, 1, 3, 2])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,acbe->adbe":
            x = tf.transpose(x, [0, 1, 3, 2])
            y = tf.transpose(y, [0, 2, 1, 3])
            result = tf.matmul(x, y, output_type=output_type)
            return tf.transpose(result, [0, 2, 1, 3])
        elif subscripts == "abcd,adbe->acbe":
            y = tf.transpose(y, [0, 2, 1, 3])  # abde
            result = tf.matmul(x, y, output_type=output_type)  # abce
            return tf.transpose(result, [0, 2, 1, 3])
        elif subscripts == "abcd,aecd->acbe":
            x = tf.transpose(x, [0, 2, 1, 3])  # acbd
            y = tf.transpose(y, [0, 2, 3, 1])  # acde
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,aecd->aceb":
            x = tf.transpose(x, [0, 2, 1, 3])
            y = tf.transpose(y, [0, 2, 3, 1])
            result = tf.matmul(x, y, output_type=output_type)  # acbe
            return tf.transpose(result, [0, 1, 3, 2])
        elif subscripts == "abcd,cde->abe":
            _, b1, c1, d1 = x.shape
            c2, d2, e2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,ced->abe":
            _, b1, c1, d1 = x.shape
            c2, e2, d2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.transpose(y, [0, 2, 1])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcd,ecd->abe":
            _, b1, c1, d1 = x.shape
            e2, c2, d2 = y.shape
            b, c, d, e = b1, c1 or c2, d1 or d2, e2
            x = tf.reshape(x, [-1, b, c * d])
            y = tf.transpose(y, [1, 2, 0])
            y = tf.reshape(y, [-1, e])
            return tf.matmul(x, y, output_type=output_type)
        elif subscripts == "abcde,aebf->adbcf":
            _, b1, c1, d1, e1 = x.shape
            _, e2, b2, f2 = y.shape
            b, c, d, e, f = b1 or b2, c1, d1, e1 or e2, f2
            x = tf.reshape(x, [-1, b, c * d, e])  # ab(cd)e
            y = tf.transpose(y, [0, 2, 1, 3])  # abef
            result = tf.matmul(x, y, output_type=output_type)  # ab(cd)f
            result = tf.reshape(result, [-1, b, c, d, f])  # abcdf
            return tf.transpose(result, [0, 3, 1, 2, 4])
        elif subscripts == "abcde,afce->acdbf":
            _, b1, c1, d1, e1 = x.shape
            _, f2, c2, e2 = y.shape
            b, c, d, e, f = b1, c1 or c2, d1, e1 or e2, f2
            x = tf.transpose(x, [0, 2, 3, 1, 4])  # acdbe
            x = tf.reshape(x, [-1, c, d * b, e])  # ac(db)e
            y = tf.transpose(y, [0, 2, 3, 1])  # acef
            result = tf.matmul(x, y, output_type=output_type)  # ac(db)f
            return tf.reshape(result, [-1, c, d, b, f])
        else:
            raise NotImplementedError

    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    # When operands are of int8, we cast the result to int32 to align with
    # the behavior of jax.
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        compute_dtype = "int8"
        result_dtype = "int32"
        output_type = "int32"
    else:
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
        compute_dtype = result_dtype
        output_type = None

    # TODO: Remove the condition once `tf.einsum` supports int8xint8->int32
    if is_valid_for_custom_ops(subscripts, *operands) and not kwargs:
        # TODO: tf.matmul doesn't support integer dtype if not specifying
        # output_type="int32"
        if "int" in compute_dtype and output_type is None:
            compute_dtype = config.floatx()
        operands = tree.map_structure(
            lambda x: tf.cast(x, compute_dtype), operands
        )
        result = use_custom_ops(subscripts, *operands, output_type=output_type)
    else:
        # TODO: tf.einsum doesn't support integer dtype with gpu
        if "int" in compute_dtype:
            compute_dtype = config.floatx()
        operands = tree.map_structure(
            lambda x: tf.cast(x, compute_dtype), operands
        )
        result = tf.einsum(subscripts, *operands, **kwargs)
    return tf.cast(result, result_dtype)


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
    x1_sparse = isinstance(x1, tf.SparseTensor)
    x2_sparse = isinstance(x2, tf.SparseTensor)
    # When both x1 and x2 are of int8 and dense tensor, specifying `output_type`
    # as int32 to enable hardware-accelerated matmul
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if (
        x1_dtype == "int8"
        and x2_dtype == "int8"
        and not x1_sparse
        and not x2_sparse
        and x1_shape.rank != 1  # TODO: support tf.tensordot
        and x2_shape.rank != 1  # TODO: support tf.tensordot
    ):
        compute_dtype = "int8"
        result_dtype = "int32"
        output_type = result_dtype
    else:
        # TODO: Typically, GPU and XLA only support float types
        compute_dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
        result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
        output_type = None
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

    if x1_sparse or x2_sparse:
        from keras.src.ops.operation_utils import compute_matmul_output_shape

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
            output = tf.matmul(x1, x2, output_type=output_type)
        elif x2_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=1)
        elif x1_shape.rank == 1:
            output = tf.tensordot(x1, x2, axes=[[0], [-2]])
        else:
            output = tf.matmul(x1, x2, output_type=output_type)
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

        axis = to_tuple_or_list(axis)
        if not axis:
            # Empty axis tuple, this is a no-op
            return x

        dense_shape = tf.convert_to_tensor(x.dense_shape)
        rank = tf.shape(dense_shape)[0]
        # Normalize axis: convert negative values and sort
        axis = [canonicalize_axis(a, rank) for a in axis]
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
            # `keepdims=False` and reducing against all axes except 0, result is
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
    x = convert_to_tensor(x)

    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    if initial is not None:
        if standardize_dtype(x.dtype) == "bool":
            x = tf.reduce_any(x, axis=axis, keepdims=keepdims)
            x = tf.math.maximum(tf.cast(x, "int32"), tf.cast(initial, "int32"))
            return tf.cast(x, "bool")
        else:
            x = tf.reduce_max(x, axis=axis, keepdims=keepdims)
            return tf.math.maximum(x, initial)

    # TensorFlow returns -inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the max of an empty tensor.",
        )

    if standardize_dtype(x.dtype) == "bool":
        return tf.reduce_any(x, axis=axis, keepdims=keepdims)
    else:
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return tf.zeros(shape, dtype=dtype)


@sparse.elementwise_unary
def absolute(x):
    x = convert_to_tensor(x)
    # uintx and bool are always non-negative
    dtype = standardize_dtype(x.dtype)
    if "uint" in dtype or dtype == "bool":
        return x
    return tf.abs(x)


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
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(step, "dtype", type(step)),
        ]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = standardize_dtype(dtype)
    try:
        out = tf.range(start, stop, delta=step, dtype=dtype)
    except tf.errors.NotFoundError:
        # Some dtypes may not work in eager mode on CPU or GPU.
        out = tf.range(start, stop, delta=step, dtype="float32")
        out = tf.cast(out, dtype)
    return out


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


def _keepdims(x, y, axis):
    if axis is None:
        shape = [1 for _ in range(len(x.shape))]
    else:
        shape = list(shape_op(x))
        for axis in tree.flatten(axis):
            shape[axis] = 1
    y = tf.reshape(y, shape)
    return y


def argmax(x, axis=None, keepdims=False):
    _x = x
    if axis is None:
        x = tf.reshape(x, [-1])
    y = tf.argmax(x, axis=axis, output_type="int32")
    if keepdims:
        y = _keepdims(_x, y, axis)
    return y


def argmin(x, axis=None, keepdims=False):
    _x = x
    if axis is None:
        x = tf.reshape(x, [-1])
    y = tf.argmin(x, axis=axis, output_type="int32")
    if keepdims:
        y = _keepdims(_x, y, axis)
    return y


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

    if weights is None:  # Treat all weights as 1
        dtype = dtypes.result_type(x.dtype, float)
        x = tf.cast(x, dtype)
        avg = tf.reduce_mean(x, axis=axis)
    else:
        weights = convert_to_tensor(weights)
        dtype = dtypes.result_type(x.dtype, weights.dtype, float)
        x = tf.cast(x, dtype)
        weights = tf.cast(weights, dtype)

        def _rank_equal_case():
            weights_sum = tf.reduce_sum(weights, axis=axis)
            return tf.reduce_sum(x * weights, axis=axis) / weights_sum

        def _rank_not_equal_case():
            weights_sum = tf.reduce_sum(weights)
            axes = tf.convert_to_tensor([[axis], [0]])
            return tf.tensordot(x, weights, axes) / weights_sum

        if axis is None:
            avg = _rank_equal_case()
        else:
            if len(x.shape) == len(weights.shape):
                avg = _rank_equal_case()
            else:
                avg = _rank_not_equal_case()
    return avg


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return tf.bitwise.invert(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)
    return tf.bitwise.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


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
                (
                    convert_to_tensor(x, sparse=False)
                    if isinstance(x, tf.SparseTensor)
                    else x
                )
                for x in xs
            ]
    xs = tree.map_structure(convert_to_tensor, xs)
    dtype_set = set([x.dtype for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(lambda x: tf.cast(x, dtype), xs)
    return tf.concat(xs, axis=axis)


@sparse.elementwise_unary
def conjugate(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def conj(x):
    return tf.math.conj(x)


@sparse.elementwise_unary
def copy(x):
    x = convert_to_tensor(x)
    return tf.identity(x)


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

    if axis is not None:
        axisa = axis
        axisb = axis
        axisc = axis
    x1 = moveaxis(x1, axisa, -1)
    x2 = moveaxis(x2, axisb, -1)

    def maybe_pad_zeros(x, size_of_last_dim):
        def pad_zeros(x):
            return tf.pad(
                x,
                tf.concat(
                    [
                        tf.zeros([tf.rank(x) - 1, 2], "int32"),
                        tf.constant([[0, 1]], "int32"),
                    ],
                    axis=0,
                ),
            )

        if isinstance(size_of_last_dim, int):
            if size_of_last_dim == 2:
                return pad_zeros(x)
            return x

        return tf.cond(
            tf.equal(size_of_last_dim, 2), lambda: pad_zeros(x), lambda: x
        )

    x1_dim = shape_op(x1)[-1]
    x2_dim = shape_op(x2)[-1]

    x1 = maybe_pad_zeros(x1, x1_dim)
    x2 = maybe_pad_zeros(x2, x2_dim)

    # Broadcast each other
    shape = shape_op(x1)

    shape = tf.broadcast_dynamic_shape(shape, shape_op(x2))
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)

    c = tf.linalg.cross(x1, x2)

    if isinstance(x1_dim, int) and isinstance(x2_dim, int):
        if (x1_dim == 2) & (x2_dim == 2):
            return c[..., 2]
        return moveaxis(c, -1, axisc)

    return tf.cond(
        (x1_dim == 2) & (x2_dim == 2),
        lambda: c[..., 2],
        lambda: moveaxis(c, -1, axisc),
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
    x = convert_to_tensor(x)
    if len(x.shape) == 1:
        return tf.cond(
            tf.equal(tf.size(x), 0),
            lambda: tf.zeros([builtins.abs(k), builtins.abs(k)], dtype=x.dtype),
            lambda: tf.linalg.diag(x, k=k),
        )
    elif len(x.shape) == 2:
        return diagonal(x, offset=k)
    else:
        raise ValueError(f"`x` must be 1d or 2d. Received: x.shape={x.shape}")


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    x_rank = x.ndim
    if (
        offset == 0
        and (axis1 == x_rank - 2 or axis1 == -2)
        and (axis2 == x_rank - 1 or axis2 == -1)
    ):
        return tf.linalg.diag_part(x)

    x = moveaxis(x, (axis1, axis2), (-2, -1))
    x_shape = shape_op(x)

    def _zeros():
        return tf.zeros(tf.concat([x_shape[:-1], [0]], 0), dtype=x.dtype)

    if isinstance(x_shape[-1], int) and isinstance(x_shape[-2], int):
        if offset <= -1 * x_shape[-2] or offset >= x_shape[-1]:
            x = _zeros()
    else:
        x = tf.cond(
            tf.logical_or(
                tf.less_equal(offset, -1 * x_shape[-2]),
                tf.greater_equal(offset, x_shape[-1]),
            ),
            lambda: _zeros(),
            lambda: x,
        )
    return tf.linalg.diag_part(x, k=offset)


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    if n == 0:
        return a
    elif n < 0:
        raise ValueError(f"Order `n` must be non-negative. Received n={n}")
    elif a.ndim == 0:
        raise ValueError(
            "`diff` requires input that is at least one dimensional. "
            f"Received: a={a}"
        )
    axis = canonicalize_axis(axis, a.ndim)
    slice1 = [slice(None)] * a.ndim
    slice2 = [slice(None)] * a.ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1_tuple = tuple(slice1)
    slice2_tuple = tuple(slice2)
    for _ in range(n):
        if standardize_dtype(a.dtype) == "bool":
            a = tf.not_equal(a[slice1_tuple], a[slice2_tuple])
        else:
            a = tf.subtract(a[slice1_tuple], a[slice2_tuple])
    return a


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = list(bins)

    # bins must be float type
    bins = tree.map_structure(lambda x: float(x), bins)

    # TODO: tf.raw_ops.Bucketize doesn't support bool, bfloat16, float16, int8
    # int16, uint8, uint16, uint32
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype in ("bool", "int8", "int16", "uint8", "uint16"):
        x = cast(x, "int32")
    elif ori_dtype == "uint32":
        x = cast(x, "int64")
    elif ori_dtype in ("bfloat16", "float16"):
        x = cast(x, "float32")

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
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    out_ndim = len(x.shape) + len(axis)
    axis = sorted([canonicalize_axis(a, out_ndim) for a in axis])
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import (
            compute_expand_dims_output_shape,
        )

        output_shape = compute_expand_dims_output_shape(x.shape, axis)
        for a in axis:
            x = tf.sparse.expand_dims(x, a)
        x.set_shape(output_shape)
        return x
    for a in axis:
        x = tf.expand_dims(x, a)
    return x


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
        xs = tree.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    if len(xs[0].shape) == 1:
        return tf.concat(xs, axis=0)
    return tf.concat(xs, axis=1)


def identity(n, dtype=None):
    return eye(N=n, M=n, dtype=dtype)


@sparse.elementwise_unary
def imag(x):
    return tf.math.imag(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)
    if "float" in dtype:
        result = tf.abs(x1 - x2) <= (atol + rtol * tf.abs(x2))
        if equal_nan:
            result = result | (is_nan(x1) & is_nan(x2))
        return result
    else:
        return tf.equal(x1, x2)


@sparse.densifying_unary(True)
def isfinite(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.ones(x.shape, tf.bool)
    return tf.math.is_finite(x)


def isinf(x):
    x = convert_to_tensor(x)
    dtype_as_dtype = tf.as_dtype(x.dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return tf.zeros(x.shape, tf.bool)
    return tf.math.is_inf(x)


def isnan(x):
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
    if num < 0:
        raise ValueError(
            f"`num` must be a non-negative integer. Received: num={num}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = standardize_dtype(dtype)
    start = convert_to_tensor(start, dtype=dtype)
    stop = convert_to_tensor(stop, dtype=dtype)
    step = convert_to_tensor(np.nan)
    if endpoint:
        result = tf.linspace(start, stop, num, axis=axis)
        if num > 1:
            step = (stop - start) / (tf.cast(num, dtype) - 1)
    else:
        # tf.linspace doesn't support endpoint=False, so we manually handle it
        if num > 0:
            step = (stop - start) / tf.cast(num, dtype)
        if num > 1:
            new_stop = tf.cast(stop, step.dtype) - step
            start = tf.cast(start, new_stop.dtype)
            result = tf.linspace(start, new_stop, num, axis=axis)
        else:
            result = tf.linspace(start, stop, num, axis=axis)
    if dtype is not None:
        if "int" in dtype:
            result = tf.floor(result)
        result = tf.cast(result, dtype)
    if retstep:
        return (result, step)
    else:
        return result


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
    result = linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
        axis=axis,
    )
    return tf.pow(tf.cast(base, result.dtype), result)


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
        if standardize_dtype(x.dtype) == "bool":
            x = tf.reduce_all(x, axis=axis, keepdims=keepdims)
            x = tf.math.minimum(tf.cast(x, "int32"), tf.cast(initial, "int32"))
            return tf.cast(x, "bool")
        else:
            x = tf.reduce_min(x, axis=axis, keepdims=keepdims)
        return tf.math.minimum(x, initial)

    # TensorFlow returns inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    if tf.executing_eagerly():
        size_x = size(x)
        tf.assert_greater(
            size_x,
            tf.constant(0, dtype=size_x.dtype),
            message="Cannot compute the min of an empty tensor.",
        )

    if standardize_dtype(x.dtype) == "bool":
        return tf.reduce_all(x, axis=axis, keepdims=keepdims)
    else:
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)


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
    x = convert_to_tensor(x)

    _source = to_tuple_or_list(source)
    _destination = to_tuple_or_list(destination)
    _source = tuple(canonicalize_axis(i, x.ndim) for i in _source)
    _destination = tuple(canonicalize_axis(i, x.ndim) for i in _destination)
    if len(_source) != len(_destination):
        raise ValueError(
            "Inconsistent number of `source` and `destination`. "
            f"Received: source={source}, destination={destination}"
        )
    # Directly return x if no movement is required
    if _source == _destination:
        return x
    perm = [i for i in range(x.ndim) if i not in _source]
    for dest, src in sorted(zip(_destination, _source)):
        perm.insert(dest, src)
    return tf.transpose(x, perm)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)

    dtype = x.dtype
    dtype_as_dtype = tf.as_dtype(dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return x

    # Replace NaN with `nan`
    x = tf.where(tf.math.is_nan(x), tf.constant(nan, dtype), x)

    # Replace positive infinity with `posinf` or `dtype.max`
    if posinf is None:
        posinf = dtype.max
    x = tf.where(tf.math.is_inf(x) & (x > 0), tf.constant(posinf, dtype), x)

    # Replace negative infinity with `neginf` or `dtype.min`
    if neginf is None:
        neginf = dtype.min
    x = tf.where(tf.math.is_inf(x) & (x < 0), tf.constant(neginf, dtype), x)

    return x


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    x = convert_to_tensor(x)
    result = tf.unstack(tf.where(tf.cast(x, "bool")), x.shape.rank, axis=1)
    return tree.map_structure(lambda indices: tf.cast(indices, "int32"), result)


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
        axis = [canonicalize_axis(a, x_ndims) for a in axis]

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
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    axis = to_tuple_or_list(axis)
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
    x = convert_to_tensor(x)
    # TODO: tf.repeat doesn't support uint16
    if standardize_dtype(x.dtype) == "uint16":
        x = tf.cast(x, "uint32")
        return tf.cast(tf.repeat(x, repeats, axis=axis), "uint16")
    return tf.repeat(x, repeats, axis=axis)


def reshape(x, newshape):
    x = convert_to_tensor(x)
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import compute_reshape_output_shape

        output_shape = compute_reshape_output_shape(
            x.shape, newshape, "newshape"
        )
        output = tf.sparse.reshape(x, newshape)
        output.set_shape(output_shape)
        return output
    return tf.reshape(x, newshape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    if axis is not None:
        return tf.roll(x, shift=shift, axis=axis)

    # If axis is None, the roll happens as a 1-d tensor.
    original_shape = tf.shape(x)
    x = tf.roll(tf.reshape(x, [-1]), shift, 0)
    return tf.reshape(x, original_shape)


def searchsorted(sorted_sequence, values, side="left"):
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    out_type = (
        "int32" if len(sorted_sequence) <= np.iinfo(np.int32).max else "int64"
    )
    return tf.searchsorted(
        sorted_sequence, values, side=side, out_type=out_type
    )


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
        total_size = x.shape[axis]
        indices_or_sections = convert_to_tensor(indices_or_sections)
        start_size = indices_or_sections[0:1]
        end_size = total_size - indices_or_sections[-1:]
        num_or_size_splits = tf.concat(
            [start_size, diff(indices_or_sections), end_size], axis=0
        )
    else:
        num_or_size_splits = indices_or_sections
    return tf.split(x, num_or_size_splits, axis=axis)


def stack(x, axis=0):
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tree.map_structure(lambda a: convert_to_tensor(a, dtype), x)
    return tf.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = tf.cast(x, config.floatx())
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)

    if (
        x.shape.rank is not None
        and isinstance(axis1, int)
        and isinstance(axis2, int)
    ):
        # This branch makes sure `perm` is statically known, to avoid a
        # not-compile-time-constant XLA error.
        axis1 = canonicalize_axis(axis1, x.ndim)
        axis2 = canonicalize_axis(axis2, x.ndim)

        # Directly return x if no movement is required
        if axis1 == axis2:
            return x

        perm = list(range(x.ndim))
        perm[axis1] = axis2
        perm[axis2] = axis1
    else:
        x_rank = tf.rank(x)
        axis1 = tf.where(axis1 < 0, tf.add(axis1, x_rank), axis1)
        axis2 = tf.where(axis2 < 0, tf.add(axis2, x_rank), axis2)
        perm = tf.range(x_rank)
        perm = tf.tensor_scatter_nd_update(
            perm, [[axis1], [axis2]], [axis2, axis1]
        )
    return tf.transpose(x, perm)


def take(x, indices, axis=None):
    if isinstance(indices, tf.SparseTensor):
        if x.dtype not in (tf.float16, tf.float32, tf.float64, tf.bfloat16):
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`x.dtype={x.dtype}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            return take(x, convert_to_tensor(indices, sparse=False), axis=axis)
        if axis is None:
            x = tf.reshape(x, (-1,))
        elif axis != 0:
            warnings.warn(
                "`take` with the TensorFlow backend does not support "
                f"`axis={axis}` when `indices` is a sparse tensor; "
                "densifying `indices`."
            )
            return take(x, convert_to_tensor(indices, sparse=False), axis=axis)
        output = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=tf.convert_to_tensor(x),
            sparse_ids=tf.sparse.expand_dims(indices, axis=-1),
            default_id=0,
        )
        output.set_shape(indices.shape + output.shape[len(indices.shape) :])
        return output

    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    # Correct the indices using "fill" mode which is the same as in jax
    indices = tf.where(
        indices < 0,
        indices + tf.cast(tf.shape(x)[axis], indices.dtype),
        indices,
    )
    return tf.gather(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    from keras.src.ops.operation_utils import (
        compute_take_along_axis_output_shape,
    )

    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, "int64")
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                "`indices` must be 1D if axis=None. "
                f"Received: indices.shape={indices.shape}"
            )
        return take_along_axis(tf.reshape(x, [-1]), indices, 0)

    # Compute the static output shape as later on, all shapes manipulations
    # use dynamic shapes.
    static_output_shape = compute_take_along_axis_output_shape(
        x.shape, indices.shape, axis
    )
    rank = x.ndim
    static_axis = axis
    axis = axis + rank if axis < 0 else axis

    # Broadcast shapes to match, ensure that the axis of interest is not
    # broadcast.
    x_shape_original = tf.shape(x, out_type=indices.dtype)
    indices_shape_original = tf.shape(indices, out_type=indices.dtype)
    x_shape = tf.tensor_scatter_nd_update(x_shape_original, [[axis]], [1])
    indices_shape = tf.tensor_scatter_nd_update(
        indices_shape_original, [[axis]], [1]
    )
    broadcasted_shape = tf.broadcast_dynamic_shape(x_shape, indices_shape)
    x_shape = tf.tensor_scatter_nd_update(
        broadcasted_shape, [[axis]], [x_shape_original[axis]]
    )
    indices_shape = tf.tensor_scatter_nd_update(
        broadcasted_shape, [[axis]], [indices_shape_original[axis]]
    )
    x = tf.broadcast_to(x, x_shape)
    indices = tf.broadcast_to(indices, indices_shape)

    # Correct the indices using "fill" mode which is the same as in jax
    indices = tf.where(indices < 0, indices + x_shape[static_axis], indices)

    x = swapaxes(x, static_axis, -1)
    indices = swapaxes(indices, static_axis, -1)

    x_shape = tf.shape(x)
    x = tf.reshape(x, [-1, x_shape[-1]])
    indices_shape = tf.shape(indices)
    indices = tf.reshape(indices, [-1, indices_shape[-1]])

    result = tf.gather(x, indices, batch_dims=1)
    result = tf.reshape(result, indices_shape)
    result = swapaxes(result, static_axis, -1)
    result.set_shape(static_output_shape)
    return result


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
    if decimals == 0:
        return tf.round(x)
    x_dtype = x.dtype
    if tf.as_dtype(x_dtype).is_integer:
        # int
        if decimals > 0:
            return x
        # temporarily convert to floats
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
    x_shape = tf.shape(x)
    x = moveaxis(x, (axis1, axis2), (-2, -1))
    # Mask out the diagonal and reduce.
    x = tf.where(
        eye(x_shape[axis1], x_shape[axis2], k=offset, dtype="bool"),
        x,
        tf.zeros_like(x),
    )
    # The output dtype is set to "int32" if the input dtype is "bool"
    if standardize_dtype(x.dtype) == "bool":
        x = tf.cast(x, "int32")
    return tf.cast(tf.reduce_sum(x, axis=(-2, -1)), dtype)


def tri(N, M=None, k=0, dtype=None):
    M = M if M is not None else N
    dtype = standardize_dtype(dtype or config.floatx())
    if k < 0:
        lower = -k - 1
        if lower > N:
            r = tf.zeros([N, M], dtype=dtype)
        else:
            o = tf.ones([N, M], dtype="bool")
            r = tf.cast(
                tf.logical_not(tf.linalg.band_part(o, lower, -1)), dtype=dtype
            )
    else:
        o = tf.ones([N, M], dtype=dtype)
        if k > M:
            r = o
        else:
            r = tf.linalg.band_part(o, -1, k)
    return r


def tril(x, k=0):
    x = convert_to_tensor(x)

    def _negative_k_branch():
        shape = tf.shape(x)
        rows, cols = shape[-2], shape[-1]
        i, j = tf.meshgrid(tf.range(rows), tf.range(cols), indexing="ij")
        mask = i >= j - k
        return tf.where(tf.broadcast_to(mask, shape), x, tf.zeros_like(x))

    return tf.cond(
        k >= 0, lambda: tf.linalg.band_part(x, -1, k), _negative_k_branch
    )


def triu(x, k=0):
    x = convert_to_tensor(x)

    def _positive_k_branch():
        shape = tf.shape(x)
        rows, cols = shape[-2], shape[-1]
        i, j = tf.meshgrid(tf.range(rows), tf.range(cols), indexing="ij")
        mask = i <= j - k
        return tf.where(tf.broadcast_to(mask, shape), x, tf.zeros_like(x))

    return tf.cond(
        k <= 0, lambda: tf.linalg.band_part(x, -k, -1), _positive_k_branch
    )


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool" or "int" in dtype:
        return x
    return tf.where(x < 0, tf.math.ceil(x), tf.math.floor(x))


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
        xs = tree.map_structure(lambda x: convert_to_tensor(x, dtype), xs)
    return tf.concat(xs, axis=0)


def _vmap_fn(fn, in_axes=0):
    if in_axes != 0:
        raise ValueError(
            "Not supported with `vectorize()` with the TensorFlow backend."
        )

    @functools.wraps(fn)
    def wrapped(x):
        return tf.vectorized_map(fn, x)

    return wrapped


def vectorize(pyfunc, *, excluded=None, signature=None):
    return vectorize_impl(
        pyfunc, _vmap_fn, excluded=excluded, signature=signature
    )


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


def divide_no_nan(x1, x2):
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
    return tf.math.divide_no_nan(x1, x2)


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
    x = convert_to_tensor(x)
    axis = to_tuple_or_list(axis)
    static_shape = x.shape.as_list()
    if axis is not None:
        for a in axis:
            if static_shape[a] != 1:
                raise ValueError(
                    f"Cannot squeeze axis={a}, because the "
                    "dimension is not 1."
                )
        axis = sorted([canonicalize_axis(a, len(static_shape)) for a in axis])
    if isinstance(x, tf.SparseTensor):
        dynamic_shape = tf.shape(x)
        new_shape = []
        gather_indices = []
        for i, dim in enumerate(static_shape):
            if not (dim == 1 if axis is None else i in axis):
                new_shape.append(dim if dim is not None else dynamic_shape[i])
                gather_indices.append(i)
        new_indices = tf.gather(x.indices, gather_indices, axis=1)
        return tf.SparseTensor(new_indices, x.values, tuple(new_shape))
    return tf.squeeze(x, axis=axis)


def transpose(x, axes=None):
    if isinstance(x, tf.SparseTensor):
        from keras.src.ops.operation_utils import compute_transpose_output_shape

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
    x = cast(x, dtype)
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.reduce_sum(
            x, axis=axis, keepdims=keepdims, output_is_sparse=True
        )
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    M = N if M is None else M
    if isinstance(k, int) and k == 0:
        return tf.eye(N, M, dtype=dtype)
    # Create a smaller square eye and pad appropriately.
    return tf.pad(
        tf.eye(tf.minimum(M - k, N + k), dtype=dtype),
        paddings=(
            (tf.maximum(-k, 0), tf.maximum(N - M + k, 0)),
            (tf.maximum(k, 0), tf.maximum(M - N - k, 0)),
        ),
    )


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


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == tf.int64:
        dtype = tf.float64
    elif dtype not in [tf.bfloat16, tf.float16, tf.float64]:
        dtype = tf.float32

    x1 = tf.cast(x1, dtype)
    x2 = tf.cast(x2, dtype)

    x1_len, x2_len = int(x1.shape[0]), int(x2.shape[0])

    if mode == "full":
        full_len = x1_len + x2_len - 1

        x1_pad = (full_len - x1_len) / 2
        x2_pad = (full_len - x2_len) / 2

        x1 = tf.pad(
            x1, paddings=[[tf.math.floor(x1_pad), tf.math.ceil(x1_pad)]]
        )
        x2 = tf.pad(
            x2, paddings=[[tf.math.floor(x2_pad), tf.math.ceil(x2_pad)]]
        )

        x1 = tf.reshape(x1, (1, full_len, 1))
        x2 = tf.reshape(x2, (full_len, 1, 1))

        return tf.squeeze(tf.nn.conv1d(x1, x2, stride=1, padding="SAME"))

    x1 = tf.reshape(x1, (1, x1_len, 1))
    x2 = tf.reshape(x2, (x2_len, 1, 1))

    return tf.squeeze(tf.nn.conv1d(x1, x2, stride=1, padding=mode.upper()))


def select(condlist, choicelist, default=0):
    return tf.experimental.numpy.select(condlist, choicelist, default=default)


def slogdet(x):
    x = convert_to_tensor(x)
    return tuple(tf.linalg.slogdet(x))


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x, tf.int32)

    x = swapaxes(x, axis, -1)
    bottom_ind = tf.math.top_k(-x, kth + 1).indices

    n = tf.shape(x)[-1]

    mask = tf.reduce_sum(tf.one_hot(bottom_ind, n, dtype=tf.int32), axis=0)

    indices = tf.where(mask)
    updates = tf.squeeze(tf.zeros(tf.shape(indices)[0], dtype=tf.int32))

    final_mask = tf.tensor_scatter_nd_update(x, indices, updates)

    top_ind = tf.math.top_k(final_mask, tf.shape(x)[-1] - kth - 1).indices

    out = tf.concat([bottom_ind, top_ind], axis=x.ndim - 1)
    return swapaxes(out, -1, axis)


def histogram(x, bins, range):
    """Computes a histogram of the data tensor `x`.

    Note: the `tf.histogram_fixed_width()` and
    `tf.histogram_fixed_width_bins()` functions
    yield slight numerical differences for some edge cases.
    """

    x = tf.convert_to_tensor(x, dtype=x.dtype)

    # Handle the range argument
    if range is None:
        min_val = tf.reduce_min(x)
        max_val = tf.reduce_max(x)
    else:
        min_val, max_val = range

    x = tf.boolean_mask(x, (x >= min_val) & (x <= max_val))
    bin_edges = tf.linspace(min_val, max_val, bins + 1)
    bin_edges_list = bin_edges.numpy().tolist()
    bin_indices = tf.raw_ops.Bucketize(input=x, boundaries=bin_edges_list[1:-1])

    bin_counts = tf.math.bincount(
        bin_indices, minlength=bins, maxlength=bins, dtype=x.dtype
    )
    return bin_counts, bin_edges
