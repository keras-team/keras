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
_DenseToDenseSetOperationOutput = collections.namedtuple(
    "DenseToDenseSetOperation",
    ["result_indices", "result_values", "result_shape"])


TV_DenseToDenseSetOperation_T = TypeVar("TV_DenseToDenseSetOperation_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt8)

def dense_to_dense_set_operation(set1: Annotated[Any, TV_DenseToDenseSetOperation_T], set2: Annotated[Any, TV_DenseToDenseSetOperation_T], set_operation: str, validate_indices:bool=True, name=None):
  r"""Applies set operation along last dimension of 2 `Tensor` inputs.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set2: A `Tensor`. Must have the same type as `set1`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).

    result_indices: A `Tensor` of type `int64`.
    result_values: A `Tensor`. Has the same type as `set1`.
    result_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseToDenseSetOperation", name, set1, set2, "set_operation",
        set_operation, "validate_indices", validate_indices)
      _result = _DenseToDenseSetOperationOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_to_dense_set_operation_eager_fallback(
          set1, set2, set_operation=set_operation,
          validate_indices=validate_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseToDenseSetOperation", set1=set1, set2=set2,
                                    set_operation=set_operation,
                                    validate_indices=validate_indices,
                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("set_operation", _op.get_attr("set_operation"),
              "validate_indices", _op._get_attr_bool("validate_indices"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseToDenseSetOperation", _inputs_flat, _attrs, _result)
  _result = _DenseToDenseSetOperationOutput._make(_result)
  return _result

DenseToDenseSetOperation = tf_export("raw_ops.DenseToDenseSetOperation")(_ops.to_raw_op(dense_to_dense_set_operation))


def dense_to_dense_set_operation_eager_fallback(set1: Annotated[Any, TV_DenseToDenseSetOperation_T], set2: Annotated[Any, TV_DenseToDenseSetOperation_T], set_operation: str, validate_indices: bool, name, ctx):
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([set1, set2], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string, ])
  (set1, set2) = _inputs_T
  _inputs_flat = [set1, set2]
  _attrs = ("set_operation", set_operation, "validate_indices",
  validate_indices, "T", _attr_T)
  _result = _execute.execute(b"DenseToDenseSetOperation", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseToDenseSetOperation", _inputs_flat, _attrs, _result)
  _result = _DenseToDenseSetOperationOutput._make(_result)
  return _result

_DenseToSparseSetOperationOutput = collections.namedtuple(
    "DenseToSparseSetOperation",
    ["result_indices", "result_values", "result_shape"])


TV_DenseToSparseSetOperation_T = TypeVar("TV_DenseToSparseSetOperation_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt8)

def dense_to_sparse_set_operation(set1: Annotated[Any, TV_DenseToSparseSetOperation_T], set2_indices: Annotated[Any, _atypes.Int64], set2_values: Annotated[Any, TV_DenseToSparseSetOperation_T], set2_shape: Annotated[Any, _atypes.Int64], set_operation: str, validate_indices:bool=True, name=None):
  r"""Applies set operation along last dimension of `Tensor` and `SparseTensor`.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
  and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
  as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set2`
  indices.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set2_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set2_values: A `Tensor`. Must have the same type as `set1`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set2_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
      be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
      max set size across `n-1` dimensions.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).

    result_indices: A `Tensor` of type `int64`.
    result_values: A `Tensor`. Has the same type as `set1`.
    result_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseToSparseSetOperation", name, set1, set2_indices,
        set2_values, set2_shape, "set_operation", set_operation,
        "validate_indices", validate_indices)
      _result = _DenseToSparseSetOperationOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_to_sparse_set_operation_eager_fallback(
          set1, set2_indices, set2_values, set2_shape,
          set_operation=set_operation, validate_indices=validate_indices,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseToSparseSetOperation", set1=set1, set2_indices=set2_indices,
                                     set2_values=set2_values,
                                     set2_shape=set2_shape,
                                     set_operation=set_operation,
                                     validate_indices=validate_indices,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("set_operation", _op.get_attr("set_operation"),
              "validate_indices", _op._get_attr_bool("validate_indices"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseToSparseSetOperation", _inputs_flat, _attrs, _result)
  _result = _DenseToSparseSetOperationOutput._make(_result)
  return _result

DenseToSparseSetOperation = tf_export("raw_ops.DenseToSparseSetOperation")(_ops.to_raw_op(dense_to_sparse_set_operation))


def dense_to_sparse_set_operation_eager_fallback(set1: Annotated[Any, TV_DenseToSparseSetOperation_T], set2_indices: Annotated[Any, _atypes.Int64], set2_values: Annotated[Any, TV_DenseToSparseSetOperation_T], set2_shape: Annotated[Any, _atypes.Int64], set_operation: str, validate_indices: bool, name, ctx):
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([set1, set2_values], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string, ])
  (set1, set2_values) = _inputs_T
  set2_indices = _ops.convert_to_tensor(set2_indices, _dtypes.int64)
  set2_shape = _ops.convert_to_tensor(set2_shape, _dtypes.int64)
  _inputs_flat = [set1, set2_indices, set2_values, set2_shape]
  _attrs = ("set_operation", set_operation, "validate_indices",
  validate_indices, "T", _attr_T)
  _result = _execute.execute(b"DenseToSparseSetOperation", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseToSparseSetOperation", _inputs_flat, _attrs, _result)
  _result = _DenseToSparseSetOperationOutput._make(_result)
  return _result


TV_SetSize_T = TypeVar("TV_SetSize_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt8)

def set_size(set_indices: Annotated[Any, _atypes.Int64], set_values: Annotated[Any, TV_SetSize_T], set_shape: Annotated[Any, _atypes.Int64], validate_indices:bool=True, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Number of unique elements along last dimension of input `set`.

  Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
  and `set_shape`. The last dimension contains values in a set, duplicates are
  allowed but ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set`
  indices. Setting is to `False` while passing invalid arguments results in
  undefined behavior.

  Args:
    set_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`.
    set_values: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      1D `Tensor`, values of a `SparseTensor`.
    set_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SetSize", name, set_indices, set_values, set_shape,
        "validate_indices", validate_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return set_size_eager_fallback(
          set_indices, set_values, set_shape,
          validate_indices=validate_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SetSize", set_indices=set_indices, set_values=set_values,
                   set_shape=set_shape, validate_indices=validate_indices,
                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("validate_indices", _op._get_attr_bool("validate_indices"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SetSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SetSize = tf_export("raw_ops.SetSize")(_ops.to_raw_op(set_size))


def set_size_eager_fallback(set_indices: Annotated[Any, _atypes.Int64], set_values: Annotated[Any, TV_SetSize_T], set_shape: Annotated[Any, _atypes.Int64], validate_indices: bool, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_T, (set_values,) = _execute.args_to_matching_eager([set_values], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string, ])
  set_indices = _ops.convert_to_tensor(set_indices, _dtypes.int64)
  set_shape = _ops.convert_to_tensor(set_shape, _dtypes.int64)
  _inputs_flat = [set_indices, set_values, set_shape]
  _attrs = ("validate_indices", validate_indices, "T", _attr_T)
  _result = _execute.execute(b"SetSize", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SetSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseToSparseSetOperationOutput = collections.namedtuple(
    "SparseToSparseSetOperation",
    ["result_indices", "result_values", "result_shape"])


TV_SparseToSparseSetOperation_T = TypeVar("TV_SparseToSparseSetOperation_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt8)

def sparse_to_sparse_set_operation(set1_indices: Annotated[Any, _atypes.Int64], set1_values: Annotated[Any, TV_SparseToSparseSetOperation_T], set1_shape: Annotated[Any, _atypes.Int64], set2_indices: Annotated[Any, _atypes.Int64], set2_values: Annotated[Any, TV_SparseToSparseSetOperation_T], set2_shape: Annotated[Any, _atypes.Int64], set_operation: str, validate_indices:bool=True, name=None):
  r"""Applies set operation along last dimension of 2 `SparseTensor` inputs.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
  order and range of `set1` and `set2` indices.

  Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
  and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
  as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
  and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
  as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set1`
  and `set2` indices.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set1_values: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set1_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
      be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set2_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set2_values: A `Tensor`. Must have the same type as `set1_values`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set2_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
      be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).

    result_indices: A `Tensor` of type `int64`.
    result_values: A `Tensor`. Has the same type as `set1_values`.
    result_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseToSparseSetOperation", name, set1_indices, set1_values,
        set1_shape, set2_indices, set2_values, set2_shape, "set_operation",
        set_operation, "validate_indices", validate_indices)
      _result = _SparseToSparseSetOperationOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_to_sparse_set_operation_eager_fallback(
          set1_indices, set1_values, set1_shape, set2_indices, set2_values,
          set2_shape, set_operation=set_operation,
          validate_indices=validate_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseToSparseSetOperation", set1_indices=set1_indices,
                                      set1_values=set1_values,
                                      set1_shape=set1_shape,
                                      set2_indices=set2_indices,
                                      set2_values=set2_values,
                                      set2_shape=set2_shape,
                                      set_operation=set_operation,
                                      validate_indices=validate_indices,
                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("set_operation", _op.get_attr("set_operation"),
              "validate_indices", _op._get_attr_bool("validate_indices"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseToSparseSetOperation", _inputs_flat, _attrs, _result)
  _result = _SparseToSparseSetOperationOutput._make(_result)
  return _result

SparseToSparseSetOperation = tf_export("raw_ops.SparseToSparseSetOperation")(_ops.to_raw_op(sparse_to_sparse_set_operation))


def sparse_to_sparse_set_operation_eager_fallback(set1_indices: Annotated[Any, _atypes.Int64], set1_values: Annotated[Any, TV_SparseToSparseSetOperation_T], set1_shape: Annotated[Any, _atypes.Int64], set2_indices: Annotated[Any, _atypes.Int64], set2_values: Annotated[Any, TV_SparseToSparseSetOperation_T], set2_shape: Annotated[Any, _atypes.Int64], set_operation: str, validate_indices: bool, name, ctx):
  set_operation = _execute.make_str(set_operation, "set_operation")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([set1_values, set2_values], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string, ])
  (set1_values, set2_values) = _inputs_T
  set1_indices = _ops.convert_to_tensor(set1_indices, _dtypes.int64)
  set1_shape = _ops.convert_to_tensor(set1_shape, _dtypes.int64)
  set2_indices = _ops.convert_to_tensor(set2_indices, _dtypes.int64)
  set2_shape = _ops.convert_to_tensor(set2_shape, _dtypes.int64)
  _inputs_flat = [set1_indices, set1_values, set1_shape, set2_indices, set2_values, set2_shape]
  _attrs = ("set_operation", set_operation, "validate_indices",
  validate_indices, "T", _attr_T)
  _result = _execute.execute(b"SparseToSparseSetOperation", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseToSparseSetOperation", _inputs_flat, _attrs, _result)
  _result = _SparseToSparseSetOperationOutput._make(_result)
  return _result

