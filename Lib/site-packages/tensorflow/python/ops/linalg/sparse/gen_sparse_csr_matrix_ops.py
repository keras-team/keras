"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated
_CSRSparseMatrixComponentsOutput = collections.namedtuple(
    "CSRSparseMatrixComponents",
    ["row_ptrs", "col_inds", "values"])


TV_CSRSparseMatrixComponents_type = TypeVar("TV_CSRSparseMatrixComponents_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def csr_sparse_matrix_components(csr_sparse_matrix: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], type: TV_CSRSparseMatrixComponents_type, name=None):
  r"""Reads out the CSR components at batch `index`.

  This op is meant only for debugging / testing, and its interface is not expected
  to be stable.

  Args:
    csr_sparse_matrix: A `Tensor` of type `variant`.
      A batched CSRSparseMatrix.
    index: A `Tensor` of type `int32`.
      The index in `csr_sparse_matrix`'s batch.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_ptrs, col_inds, values).

    row_ptrs: A `Tensor` of type `int32`.
    col_inds: A `Tensor` of type `int32`.
    values: A `Tensor` of type `type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CSRSparseMatrixComponents", name, csr_sparse_matrix, index,
        "type", type)
      _result = _CSRSparseMatrixComponentsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return csr_sparse_matrix_components_eager_fallback(
          csr_sparse_matrix, index, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CSRSparseMatrixComponents", csr_sparse_matrix=csr_sparse_matrix,
                                     index=index, type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CSRSparseMatrixComponents", _inputs_flat, _attrs, _result)
  _result = _CSRSparseMatrixComponentsOutput._make(_result)
  return _result

CSRSparseMatrixComponents = tf_export("raw_ops.CSRSparseMatrixComponents")(_ops.to_raw_op(csr_sparse_matrix_components))


def csr_sparse_matrix_components_eager_fallback(csr_sparse_matrix: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], type: TV_CSRSparseMatrixComponents_type, name, ctx):
  type = _execute.make_type(type, "type")
  csr_sparse_matrix = _ops.convert_to_tensor(csr_sparse_matrix, _dtypes.variant)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  _inputs_flat = [csr_sparse_matrix, index]
  _attrs = ("type", type)
  _result = _execute.execute(b"CSRSparseMatrixComponents", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CSRSparseMatrixComponents", _inputs_flat, _attrs, _result)
  _result = _CSRSparseMatrixComponentsOutput._make(_result)
  return _result


TV_CSRSparseMatrixToDense_type = TypeVar("TV_CSRSparseMatrixToDense_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def csr_sparse_matrix_to_dense(sparse_input: Annotated[Any, _atypes.Variant], type: TV_CSRSparseMatrixToDense_type, name=None) -> Annotated[Any, TV_CSRSparseMatrixToDense_type]:
  r"""Convert a (possibly batched) CSRSparseMatrix to dense.

  Args:
    sparse_input: A `Tensor` of type `variant`. A batched CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CSRSparseMatrixToDense", name, sparse_input, "type", type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return csr_sparse_matrix_to_dense_eager_fallback(
          sparse_input, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CSRSparseMatrixToDense", sparse_input=sparse_input, type=type,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CSRSparseMatrixToDense", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CSRSparseMatrixToDense = tf_export("raw_ops.CSRSparseMatrixToDense")(_ops.to_raw_op(csr_sparse_matrix_to_dense))


def csr_sparse_matrix_to_dense_eager_fallback(sparse_input: Annotated[Any, _atypes.Variant], type: TV_CSRSparseMatrixToDense_type, name, ctx) -> Annotated[Any, TV_CSRSparseMatrixToDense_type]:
  type = _execute.make_type(type, "type")
  sparse_input = _ops.convert_to_tensor(sparse_input, _dtypes.variant)
  _inputs_flat = [sparse_input]
  _attrs = ("type", type)
  _result = _execute.execute(b"CSRSparseMatrixToDense", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CSRSparseMatrixToDense", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_CSRSparseMatrixToSparseTensorOutput = collections.namedtuple(
    "CSRSparseMatrixToSparseTensor",
    ["indices", "values", "dense_shape"])


TV_CSRSparseMatrixToSparseTensor_type = TypeVar("TV_CSRSparseMatrixToSparseTensor_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def csr_sparse_matrix_to_sparse_tensor(sparse_matrix: Annotated[Any, _atypes.Variant], type: TV_CSRSparseMatrixToSparseTensor_type, name=None):
  r"""Converts a (possibly batched) CSRSparesMatrix to a SparseTensor.

  Args:
    sparse_matrix: A `Tensor` of type `variant`.
      A (possibly batched) CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, dense_shape).

    indices: A `Tensor` of type `int64`.
    values: A `Tensor` of type `type`.
    dense_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CSRSparseMatrixToSparseTensor", name, sparse_matrix, "type",
        type)
      _result = _CSRSparseMatrixToSparseTensorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return csr_sparse_matrix_to_sparse_tensor_eager_fallback(
          sparse_matrix, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CSRSparseMatrixToSparseTensor", sparse_matrix=sparse_matrix,
                                         type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CSRSparseMatrixToSparseTensor", _inputs_flat, _attrs, _result)
  _result = _CSRSparseMatrixToSparseTensorOutput._make(_result)
  return _result

CSRSparseMatrixToSparseTensor = tf_export("raw_ops.CSRSparseMatrixToSparseTensor")(_ops.to_raw_op(csr_sparse_matrix_to_sparse_tensor))


def csr_sparse_matrix_to_sparse_tensor_eager_fallback(sparse_matrix: Annotated[Any, _atypes.Variant], type: TV_CSRSparseMatrixToSparseTensor_type, name, ctx):
  type = _execute.make_type(type, "type")
  sparse_matrix = _ops.convert_to_tensor(sparse_matrix, _dtypes.variant)
  _inputs_flat = [sparse_matrix]
  _attrs = ("type", type)
  _result = _execute.execute(b"CSRSparseMatrixToSparseTensor", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CSRSparseMatrixToSparseTensor", _inputs_flat, _attrs, _result)
  _result = _CSRSparseMatrixToSparseTensorOutput._make(_result)
  return _result


TV_DenseToCSRSparseMatrix_T = TypeVar("TV_DenseToCSRSparseMatrix_T", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def dense_to_csr_sparse_matrix(dense_input: Annotated[Any, TV_DenseToCSRSparseMatrix_T], indices: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Converts a dense tensor to a (possibly batched) CSRSparseMatrix.

  Args:
    dense_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `complex64`, `complex128`.
      A Dense tensor.
    indices: A `Tensor` of type `int64`. Indices of nonzero elements.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseToCSRSparseMatrix", name, dense_input, indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_to_csr_sparse_matrix_eager_fallback(
          dense_input, indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseToCSRSparseMatrix", dense_input=dense_input, indices=indices,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseToCSRSparseMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DenseToCSRSparseMatrix = tf_export("raw_ops.DenseToCSRSparseMatrix")(_ops.to_raw_op(dense_to_csr_sparse_matrix))


def dense_to_csr_sparse_matrix_eager_fallback(dense_input: Annotated[Any, TV_DenseToCSRSparseMatrix_T], indices: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_T, (dense_input,) = _execute.args_to_matching_eager([dense_input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  _inputs_flat = [dense_input, indices]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DenseToCSRSparseMatrix", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseToCSRSparseMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixAdd_T = TypeVar("TV_SparseMatrixAdd_T", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_matrix_add(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, _atypes.Variant], alpha: Annotated[Any, TV_SparseMatrixAdd_T], beta: Annotated[Any, TV_SparseMatrixAdd_T], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Sparse addition of two CSR matrices, C = alpha * A + beta * B.

  The gradients of SparseMatrixAdd outputs with respect to alpha and beta are not
  currently defined (TensorFlow will return zeros for these entries).

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor` of type `variant`. A CSRSparseMatrix.
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `complex64`, `complex128`.
      A constant scalar.
    beta: A `Tensor`. Must have the same type as `alpha`. A constant scalar.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixAdd", name, a, b, alpha, beta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_add_eager_fallback(
          a, b, alpha, beta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixAdd", a=a, b=b, alpha=alpha, beta=beta, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixAdd = tf_export("raw_ops.SparseMatrixAdd")(_ops.to_raw_op(sparse_matrix_add))


def sparse_matrix_add_eager_fallback(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, _atypes.Variant], alpha: Annotated[Any, TV_SparseMatrixAdd_T], beta: Annotated[Any, TV_SparseMatrixAdd_T], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, beta], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (alpha, beta) = _inputs_T
  a = _ops.convert_to_tensor(a, _dtypes.variant)
  b = _ops.convert_to_tensor(b, _dtypes.variant)
  _inputs_flat = [a, b, alpha, beta]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseMatrixAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixMatMul_T = TypeVar("TV_SparseMatrixMatMul_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_matrix_mat_mul(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, TV_SparseMatrixMatMul_T], transpose_a:bool=False, transpose_b:bool=False, adjoint_a:bool=False, adjoint_b:bool=False, transpose_output:bool=False, conjugate_output:bool=False, name=None) -> Annotated[Any, TV_SparseMatrixMatMul_T]:
  r"""Matrix-multiplies a sparse matrix with a dense matrix.

  Returns a dense matrix.
  For inputs A and B, where A is CSR and B is dense; this op returns a dense C;

  If transpose_output is false, returns:
  ```
    C = A . B
  ```

  If transpose_output is `true`, returns:
  ```
    C = transpose(A . B) = transpose(B) . transpose(A)
  ```
  where the transposition is performed along the two innermost (matrix)
  dimensions.

  If conjugate_output is `true`, returns:
  ```
    C = conjugate(A . B) = conjugate(A) . conjugate(B)
  ```

  If both conjugate_output and transpose_output are `true`, returns:
  ```
    C = conjugate(transpose(A . B)) = conjugate(transpose(B)) .
                                      conjugate(transpose(A))
  ```

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor`. A dense tensor.
    transpose_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be transposed.
    transpose_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be transposed.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be conjugate-transposed.
    adjoint_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be conjugate-transposed.
    transpose_output: An optional `bool`. Defaults to `False`.
      Transposes the product of `a` and `b`.
    conjugate_output: An optional `bool`. Defaults to `False`.
      Conjugates the product of `a` and `b`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `b`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixMatMul", name, a, b, "transpose_a", transpose_a,
        "transpose_b", transpose_b, "adjoint_a", adjoint_a, "adjoint_b",
        adjoint_b, "transpose_output", transpose_output, "conjugate_output",
        conjugate_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_mat_mul_eager_fallback(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b,
          adjoint_a=adjoint_a, adjoint_b=adjoint_b,
          transpose_output=transpose_output,
          conjugate_output=conjugate_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  if transpose_output is None:
    transpose_output = False
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  if conjugate_output is None:
    conjugate_output = False
  conjugate_output = _execute.make_bool(conjugate_output, "conjugate_output")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixMatMul", a=a, b=b, transpose_a=transpose_a,
                              transpose_b=transpose_b, adjoint_a=adjoint_a,
                              adjoint_b=adjoint_b,
                              transpose_output=transpose_output,
                              conjugate_output=conjugate_output, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "adjoint_a",
              _op._get_attr_bool("adjoint_a"), "adjoint_b",
              _op._get_attr_bool("adjoint_b"), "transpose_output",
              _op._get_attr_bool("transpose_output"), "conjugate_output",
              _op._get_attr_bool("conjugate_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixMatMul = tf_export("raw_ops.SparseMatrixMatMul")(_ops.to_raw_op(sparse_matrix_mat_mul))


def sparse_matrix_mat_mul_eager_fallback(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, TV_SparseMatrixMatMul_T], transpose_a: bool, transpose_b: bool, adjoint_a: bool, adjoint_b: bool, transpose_output: bool, conjugate_output: bool, name, ctx) -> Annotated[Any, TV_SparseMatrixMatMul_T]:
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  if transpose_output is None:
    transpose_output = False
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  if conjugate_output is None:
    conjugate_output = False
  conjugate_output = _execute.make_bool(conjugate_output, "conjugate_output")
  _attr_T, (b,) = _execute.args_to_matching_eager([b], ctx, [])
  a = _ops.convert_to_tensor(a, _dtypes.variant)
  _inputs_flat = [a, b]
  _attrs = ("T", _attr_T, "transpose_a", transpose_a, "transpose_b",
  transpose_b, "adjoint_a", adjoint_a, "adjoint_b", adjoint_b,
  "transpose_output", transpose_output, "conjugate_output", conjugate_output)
  _result = _execute.execute(b"SparseMatrixMatMul", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixMul_T = TypeVar("TV_SparseMatrixMul_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_matrix_mul(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, TV_SparseMatrixMul_T], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Element-wise multiplication of a sparse matrix with a dense tensor.

  Returns a sparse matrix.

  The dense tensor `b` may be either a scalar; otherwise `a` must be a rank-3
  `SparseMatrix`; in this case `b` must be shaped `[batch_size, 1, 1]` and the
  multiply operation broadcasts.

  **NOTE** even if `b` is zero, the sparsity structure of the output does not
  change.

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor`. A dense tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixMul", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_mul_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixMul", a=a, b=b, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixMul = tf_export("raw_ops.SparseMatrixMul")(_ops.to_raw_op(sparse_matrix_mul))


def sparse_matrix_mul_eager_fallback(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, TV_SparseMatrixMul_T], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_T, (b,) = _execute.args_to_matching_eager([b], ctx, [])
  a = _ops.convert_to_tensor(a, _dtypes.variant)
  _inputs_flat = [a, b]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseMatrixMul", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sparse_matrix_nnz(sparse_matrix: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the number of nonzeroes of `sparse_matrix`.

  Args:
    sparse_matrix: A `Tensor` of type `variant`. A CSRSparseMatrix.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixNNZ", name, sparse_matrix)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_nnz_eager_fallback(
          sparse_matrix, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixNNZ", sparse_matrix=sparse_matrix, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixNNZ", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixNNZ = tf_export("raw_ops.SparseMatrixNNZ")(_ops.to_raw_op(sparse_matrix_nnz))


def sparse_matrix_nnz_eager_fallback(sparse_matrix: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int32]:
  sparse_matrix = _ops.convert_to_tensor(sparse_matrix, _dtypes.variant)
  _inputs_flat = [sparse_matrix]
  _attrs = None
  _result = _execute.execute(b"SparseMatrixNNZ", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixNNZ", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sparse_matrix_ordering_amd(input: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Computes the Approximate Minimum Degree (AMD) ordering of `input`.

  Computes the Approximate Minimum Degree (AMD) ordering for a sparse matrix.

  The returned permutation may be used to permute the rows and columns of the
  given sparse matrix. This typically results in permuted sparse matrix's sparse
  Cholesky (or other decompositions) in having fewer zero fill-in compared to
  decomposition of the original matrix.

  The input sparse matrix may have rank 2 or rank 3. The output Tensor,
  representing would then have rank 1 or 2 respectively, with the same batch
  shape as the input.

  Each component of the input sparse matrix must represent a square symmetric
  matrix; only the lower triangular part of the matrix is read. The values of the
  sparse matrix does not affect the returned permutation, only the sparsity
  pattern of the sparse matrix is used. Hence, a single AMD ordering may be
  reused for the Cholesky decompositions of sparse matrices with the same sparsity
  pattern but with possibly different values.

  Each batch component of the output permutation represents a permutation of `N`
  elements, where the input sparse matrix components each have `N` rows. That is,
  the component contains each of the integers `{0, .. N-1}` exactly once. The
  `i`th element represents the row index that the `i`th row maps to.

  Usage example:

  ```python
      from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

      a_indices = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 3]])
      a_values = np.array([1.0, 2.0, 1.0, 3.0, 4.0], np.float32)
      a_dense_shape = [4, 4]

      with tf.Session() as sess:
        # Define (COO format) SparseTensor over Numpy array.
        a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)

        # Convert SparseTensors to CSR SparseMatrix.
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a_st.indices, a_st.values, a_st.dense_shape)

        # Obtain the AMD Ordering for the CSR SparseMatrix.
        ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)

        ordering_amd_value = sess.run(ordering_amd)
  ```

  `ordering_amd_value` stores the AMD ordering: `[1 2 3 0]`.

  input: A `CSRSparseMatrix`.

  Args:
    input: A `Tensor` of type `variant`. A `CSRSparseMatrix`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixOrderingAMD", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_ordering_amd_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixOrderingAMD", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixOrderingAMD", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixOrderingAMD = tf_export("raw_ops.SparseMatrixOrderingAMD")(_ops.to_raw_op(sparse_matrix_ordering_amd))


def sparse_matrix_ordering_amd_eager_fallback(input: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int32]:
  input = _ops.convert_to_tensor(input, _dtypes.variant)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"SparseMatrixOrderingAMD", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixOrderingAMD", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixSoftmax_type = TypeVar("TV_SparseMatrixSoftmax_type", _atypes.Float32, _atypes.Float64)

def sparse_matrix_softmax(logits: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSoftmax_type, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Calculates the softmax of a CSRSparseMatrix.

  Calculate the softmax of the innermost dimensions of a SparseMatrix.

  Missing values are treated as `-inf` (i.e., logits of zero probability); and
  the output has the same sparsity structure as the input (though missing values
  in the output may now be treated as having probability zero).

  Args:
    logits: A `Tensor` of type `variant`. A CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixSoftmax", name, logits, "type", type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_softmax_eager_fallback(
          logits, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixSoftmax", logits=logits, type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixSoftmax = tf_export("raw_ops.SparseMatrixSoftmax")(_ops.to_raw_op(sparse_matrix_softmax))


def sparse_matrix_softmax_eager_fallback(logits: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSoftmax_type, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  logits = _ops.convert_to_tensor(logits, _dtypes.variant)
  _inputs_flat = [logits]
  _attrs = ("type", type)
  _result = _execute.execute(b"SparseMatrixSoftmax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixSoftmaxGrad_type = TypeVar("TV_SparseMatrixSoftmaxGrad_type", _atypes.Float32, _atypes.Float64)

def sparse_matrix_softmax_grad(softmax: Annotated[Any, _atypes.Variant], grad_softmax: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSoftmaxGrad_type, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Calculates the gradient of the SparseMatrixSoftmax op.

  Args:
    softmax: A `Tensor` of type `variant`. A CSRSparseMatrix.
    grad_softmax: A `Tensor` of type `variant`. The gradient of `softmax`.
    type: A `tf.DType` from: `tf.float32, tf.float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixSoftmaxGrad", name, softmax, grad_softmax, "type",
        type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_softmax_grad_eager_fallback(
          softmax, grad_softmax, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixSoftmaxGrad", softmax=softmax, grad_softmax=grad_softmax,
                                   type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixSoftmaxGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixSoftmaxGrad = tf_export("raw_ops.SparseMatrixSoftmaxGrad")(_ops.to_raw_op(sparse_matrix_softmax_grad))


def sparse_matrix_softmax_grad_eager_fallback(softmax: Annotated[Any, _atypes.Variant], grad_softmax: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSoftmaxGrad_type, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  softmax = _ops.convert_to_tensor(softmax, _dtypes.variant)
  grad_softmax = _ops.convert_to_tensor(grad_softmax, _dtypes.variant)
  _inputs_flat = [softmax, grad_softmax]
  _attrs = ("type", type)
  _result = _execute.execute(b"SparseMatrixSoftmaxGrad", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixSoftmaxGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixSparseCholesky_type = TypeVar("TV_SparseMatrixSparseCholesky_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_matrix_sparse_cholesky(input: Annotated[Any, _atypes.Variant], permutation: Annotated[Any, _atypes.Int32], type: TV_SparseMatrixSparseCholesky_type, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Computes the sparse Cholesky decomposition of `input`.

  Computes the Sparse Cholesky decomposition of a sparse matrix, with the given
  fill-in reducing permutation.

  The input sparse matrix and the fill-in reducing permutation `permutation` must
  have compatible shapes. If the sparse matrix has rank 3; with the batch
  dimension `B`, then the `permutation` must be of rank 2; with the same batch
  dimension `B`. There is no support for broadcasting.

  Furthermore, each component vector of `permutation` must be of length `N`,
  containing each of the integers {0, 1, ..., N - 1} exactly once, where `N` is
  the number of rows of each component of the sparse matrix.

  Each component of the input sparse matrix must represent a symmetric positive
  definite (SPD) matrix; although only the lower triangular part of the matrix is
  read. If any individual component is not SPD, then an InvalidArgument error is
  thrown.

  The returned sparse matrix has the same dense shape as the input sparse matrix.
  For each component `A` of the input sparse matrix, the corresponding output
  sparse matrix represents `L`, the lower triangular Cholesky factor satisfying
  the following identity:

  ```
    A = L * Lt
  ```

  where Lt denotes the transpose of L (or its conjugate transpose, if `type` is
  `complex64` or `complex128`).

  The `type` parameter denotes the type of the matrix elements. The supported
  types are: `float32`, `float64`, `complex64` and `complex128`.

  Usage example:

  ```python
      from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

      a_indices = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 3]])
      a_values = np.array([1.0, 2.0, 1.0, 3.0, 4.0], np.float32)
      a_dense_shape = [4, 4]

      with tf.Session() as sess:
        # Define (COO format) SparseTensor over Numpy array.
        a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)

        # Convert SparseTensors to CSR SparseMatrix.
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a_st.indices, a_st.values, a_st.dense_shape)

        # Obtain the Sparse Cholesky factor using AMD Ordering for reducing zero
        # fill-in (number of structural non-zeros in the sparse Cholesky factor).
        ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)
        cholesky_sparse_matrices = (
            sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
                sparse_matrix, ordering_amd, type=tf.float32))

        # Convert the CSRSparseMatrix Cholesky factor to a dense Tensor
        dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            cholesky_sparse_matrices, tf.float32)

        # Evaluate the dense Tensor value.
        dense_cholesky_value = sess.run(dense_cholesky)
  ```

  `dense_cholesky_value` stores the dense Cholesky factor:

  ```
      [[  1.  0.    0.    0.]
       [  0.  1.41  0.    0.]
       [  0.  0.70  1.58  0.]
       [  0.  0.    0.    2.]]
  ```


  input: A `CSRSparseMatrix`.
  permutation: A `Tensor`.
  type: The type of `input`.

  Args:
    input: A `Tensor` of type `variant`. A `CSRSparseMatrix`.
    permutation: A `Tensor` of type `int32`.
      A fill-in reducing permutation matrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixSparseCholesky", name, input, permutation, "type",
        type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_sparse_cholesky_eager_fallback(
          input, permutation, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixSparseCholesky", input=input, permutation=permutation,
                                      type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixSparseCholesky", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixSparseCholesky = tf_export("raw_ops.SparseMatrixSparseCholesky")(_ops.to_raw_op(sparse_matrix_sparse_cholesky))


def sparse_matrix_sparse_cholesky_eager_fallback(input: Annotated[Any, _atypes.Variant], permutation: Annotated[Any, _atypes.Int32], type: TV_SparseMatrixSparseCholesky_type, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  input = _ops.convert_to_tensor(input, _dtypes.variant)
  permutation = _ops.convert_to_tensor(permutation, _dtypes.int32)
  _inputs_flat = [input, permutation]
  _attrs = ("type", type)
  _result = _execute.execute(b"SparseMatrixSparseCholesky", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixSparseCholesky", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixSparseMatMul_type = TypeVar("TV_SparseMatrixSparseMatMul_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_matrix_sparse_mat_mul(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSparseMatMul_type, transpose_a:bool=False, transpose_b:bool=False, adjoint_a:bool=False, adjoint_b:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Sparse-matrix-multiplies two CSR matrices `a` and `b`.

  Performs a matrix multiplication of a sparse matrix `a` with a sparse matrix
  `b`; returns a sparse matrix `a * b`, unless either `a` or `b` is transposed or
  adjointed.

  Each matrix may be transposed or adjointed (conjugated and transposed)
  according to the Boolean parameters `transpose_a`, `adjoint_a`, `transpose_b`
  and `adjoint_b`. At most one of `transpose_a` or `adjoint_a` may be True.
  Similarly, at most one of `transpose_b` or `adjoint_b` may be True.

  The inputs must have compatible shapes. That is, the inner dimension of `a`
  must be equal to the outer dimension of `b`. This requirement is adjusted
  according to whether either `a` or `b` is transposed or adjointed.

  The `type` parameter denotes the type of the matrix elements. Both `a` and `b`
  must have the same type. The supported types are: `float32`, `float64`,
  `complex64` and `complex128`.

  Both `a` and `b` must have the same rank. Broadcasting is not supported. If they
  have rank 3, each batch of 2D CSRSparseMatrices within `a` and `b` must have the
  same dense shape.

  The sparse matrix product may have numeric (non-structural) zeros.
  TODO(anudhyan): Consider adding a boolean attribute to control whether to prune
  zeros.

  Usage example:

  ```python
      from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

      a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
      a_values = np.array([1.0, 5.0, -1.0, -2.0], np.float32)
      a_dense_shape = [4, 5]

      b_indices = np.array([[0, 0], [3, 0], [3, 1]])
      b_values = np.array([2.0, 7.0, 8.0], np.float32)
      b_dense_shape = [5, 3]

      with tf.Session() as sess:
        # Define (COO format) Sparse Tensors over Numpy arrays
        a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)
        b_st = tf.sparse.SparseTensor(b_indices, b_values, b_dense_shape)

        # Convert SparseTensors to CSR SparseMatrix
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a_st.indices, a_st.values, a_st.dense_shape)
        b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b_st.indices, b_st.values, b_st.dense_shape)

        # Compute the CSR SparseMatrix matrix multiplication
        c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a=a_sm, b=b_sm, type=tf.float32)

        # Convert the CSR SparseMatrix product to a dense Tensor
        c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            c_sm, tf.float32)
        # Evaluate the dense Tensor value
        c_sm_dense_value = sess.run(c_sm_dense)
  ```

  `c_sm_dense_value` stores the dense matrix product:

  ```
      [[  2.   0.   0.]
       [  0.   0.   0.]
       [ 35.  40.   0.]
       [ -4.   0.   0.]]
  ```

  a: A `CSRSparseMatrix`.
  b: A `CSRSparseMatrix` with the same type and rank as `a`.
  type: The type of both `a` and `b`.
  transpose_a: If True, `a` transposed before multiplication.
  transpose_b: If True, `b` transposed before multiplication.
  adjoint_a: If True, `a` adjointed before multiplication.
  adjoint_b: If True, `b` adjointed before multiplication.

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor` of type `variant`. A CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    transpose_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be transposed.
    transpose_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be transposed.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be conjugate-transposed.
    adjoint_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be conjugate-transposed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixSparseMatMul", name, a, b, "type", type,
        "transpose_a", transpose_a, "transpose_b", transpose_b, "adjoint_a",
        adjoint_a, "adjoint_b", adjoint_b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_sparse_mat_mul_eager_fallback(
          a, b, type=type, transpose_a=transpose_a, transpose_b=transpose_b,
          adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixSparseMatMul", a=a, b=b, type=type,
                                    transpose_a=transpose_a,
                                    transpose_b=transpose_b,
                                    adjoint_a=adjoint_a, adjoint_b=adjoint_b,
                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "adjoint_a",
              _op._get_attr_bool("adjoint_a"), "adjoint_b",
              _op._get_attr_bool("adjoint_b"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixSparseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixSparseMatMul = tf_export("raw_ops.SparseMatrixSparseMatMul")(_ops.to_raw_op(sparse_matrix_sparse_mat_mul))


def sparse_matrix_sparse_mat_mul_eager_fallback(a: Annotated[Any, _atypes.Variant], b: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixSparseMatMul_type, transpose_a: bool, transpose_b: bool, adjoint_a: bool, adjoint_b: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  a = _ops.convert_to_tensor(a, _dtypes.variant)
  b = _ops.convert_to_tensor(b, _dtypes.variant)
  _inputs_flat = [a, b]
  _attrs = ("type", type, "transpose_a", transpose_a, "transpose_b",
  transpose_b, "adjoint_a", adjoint_a, "adjoint_b", adjoint_b)
  _result = _execute.execute(b"SparseMatrixSparseMatMul", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixSparseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixTranspose_type = TypeVar("TV_SparseMatrixTranspose_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_matrix_transpose(input: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixTranspose_type, conjugate:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Transposes the inner (matrix) dimensions of a CSRSparseMatrix.

  Transposes the inner (matrix) dimensions of a SparseMatrix and optionally
  conjugates its values.

  Args:
    input: A `Tensor` of type `variant`. A CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    conjugate: An optional `bool`. Defaults to `False`.
      Indicates whether `input` should be conjugated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixTranspose", name, input, "conjugate", conjugate,
        "type", type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_transpose_eager_fallback(
          input, conjugate=conjugate, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  if conjugate is None:
    conjugate = False
  conjugate = _execute.make_bool(conjugate, "conjugate")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixTranspose", input=input, type=type, conjugate=conjugate,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("conjugate", _op._get_attr_bool("conjugate"), "type",
              _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixTranspose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixTranspose = tf_export("raw_ops.SparseMatrixTranspose")(_ops.to_raw_op(sparse_matrix_transpose))


def sparse_matrix_transpose_eager_fallback(input: Annotated[Any, _atypes.Variant], type: TV_SparseMatrixTranspose_type, conjugate: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  if conjugate is None:
    conjugate = False
  conjugate = _execute.make_bool(conjugate, "conjugate")
  input = _ops.convert_to_tensor(input, _dtypes.variant)
  _inputs_flat = [input]
  _attrs = ("conjugate", conjugate, "type", type)
  _result = _execute.execute(b"SparseMatrixTranspose", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixTranspose", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatrixZeros_type = TypeVar("TV_SparseMatrixZeros_type", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_matrix_zeros(dense_shape: Annotated[Any, _atypes.Int64], type: TV_SparseMatrixZeros_type, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates an all-zeros CSRSparseMatrix with shape `dense_shape`.

  Args:
    dense_shape: A `Tensor` of type `int64`. The desired matrix shape.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatrixZeros", name, dense_shape, "type", type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_matrix_zeros_eager_fallback(
          dense_shape, type=type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  type = _execute.make_type(type, "type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatrixZeros", dense_shape=dense_shape, type=type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("type", _op._get_attr_type("type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatrixZeros", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatrixZeros = tf_export("raw_ops.SparseMatrixZeros")(_ops.to_raw_op(sparse_matrix_zeros))


def sparse_matrix_zeros_eager_fallback(dense_shape: Annotated[Any, _atypes.Int64], type: TV_SparseMatrixZeros_type, name, ctx) -> Annotated[Any, _atypes.Variant]:
  type = _execute.make_type(type, "type")
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [dense_shape]
  _attrs = ("type", type)
  _result = _execute.execute(b"SparseMatrixZeros", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatrixZeros", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseTensorToCSRSparseMatrix_T = TypeVar("TV_SparseTensorToCSRSparseMatrix_T", _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64)

def sparse_tensor_to_csr_sparse_matrix(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseTensorToCSRSparseMatrix_T], dense_shape: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Converts a SparseTensor to a (possibly batched) CSRSparseMatrix.

  Args:
    indices: A `Tensor` of type `int64`. SparseTensor indices.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `complex64`, `complex128`.
      SparseTensor values.
    dense_shape: A `Tensor` of type `int64`. SparseTensor dense shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseTensorToCSRSparseMatrix", name, indices, values,
        dense_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_tensor_to_csr_sparse_matrix_eager_fallback(
          indices, values, dense_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTensorToCSRSparseMatrix", indices=indices, values=values,
                                         dense_shape=dense_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseTensorToCSRSparseMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseTensorToCSRSparseMatrix = tf_export("raw_ops.SparseTensorToCSRSparseMatrix")(_ops.to_raw_op(sparse_tensor_to_csr_sparse_matrix))


def sparse_tensor_to_csr_sparse_matrix_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseTensorToCSRSparseMatrix_T], dense_shape: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [indices, values, dense_shape]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseTensorToCSRSparseMatrix", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseTensorToCSRSparseMatrix", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

