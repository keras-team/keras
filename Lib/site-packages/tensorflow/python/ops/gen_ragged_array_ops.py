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
_RaggedCrossOutput = collections.namedtuple(
    "RaggedCross",
    ["output_values", "output_row_splits"])


TV_RaggedCross_out_values_type = TypeVar("TV_RaggedCross_out_values_type", _atypes.Int64, _atypes.String)
TV_RaggedCross_out_row_splits_type = TypeVar("TV_RaggedCross_out_row_splits_type", _atypes.Int32, _atypes.Int64)

def ragged_cross(ragged_values, ragged_row_splits, sparse_indices: Annotated[List[Any], _atypes.Int64], sparse_values, sparse_shape: Annotated[List[Any], _atypes.Int64], dense_inputs, input_order: str, hashed_output: bool, num_buckets: int, hash_key: int, out_values_type: TV_RaggedCross_out_values_type, out_row_splits_type: TV_RaggedCross_out_row_splits_type, name=None):
  r"""Generates a feature cross from a list of tensors, and returns it as a
RaggedTensor.  See `tf.ragged.cross` for more details.

  Args:
    ragged_values: A list of `Tensor` objects with types from: `int64`, `string`.
      The values tensor for each RaggedTensor input.
    ragged_row_splits: A list of `Tensor` objects with types from: `int32`, `int64`.
      The row_splits tensor for each RaggedTensor input.
    sparse_indices: A list of `Tensor` objects with type `int64`.
      The indices tensor for each SparseTensor input.
    sparse_values: A list of `Tensor` objects with types from: `int64`, `string`.
      The values tensor for each SparseTensor input.
    sparse_shape: A list with the same length as `sparse_indices` of `Tensor` objects with type `int64`.
      The dense_shape tensor for each SparseTensor input.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      The tf.Tensor inputs.
    input_order: A `string`.
      String specifying the tensor type for each input.  The `i`th character in
      this string specifies the type of the `i`th input, and is one of: 'R' (ragged),
      'D' (dense), or 'S' (sparse).  This attr is used to ensure that the crossed
      values are combined in the order of the inputs from the call to tf.ragged.cross.
    hashed_output: A `bool`.
    num_buckets: An `int` that is `>= 0`.
    hash_key: An `int`.
    out_values_type: A `tf.DType` from: `tf.int64, tf.string`.
    out_row_splits_type: A `tf.DType` from: `tf.int32, tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_values, output_row_splits).

    output_values: A `Tensor` of type `out_values_type`.
    output_row_splits: A `Tensor` of type `out_row_splits_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedCross", name, ragged_values, ragged_row_splits,
        sparse_indices, sparse_values, sparse_shape, dense_inputs,
        "input_order", input_order, "hashed_output", hashed_output,
        "num_buckets", num_buckets, "hash_key", hash_key, "out_values_type",
        out_values_type, "out_row_splits_type", out_row_splits_type)
      _result = _RaggedCrossOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_cross_eager_fallback(
          ragged_values, ragged_row_splits, sparse_indices, sparse_values,
          sparse_shape, dense_inputs, input_order=input_order,
          hashed_output=hashed_output, num_buckets=num_buckets,
          hash_key=hash_key, out_values_type=out_values_type,
          out_row_splits_type=out_row_splits_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sparse_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_indices' argument to "
        "'ragged_cross' Op, not %r." % sparse_indices)
  _attr_Nsparse = len(sparse_indices)
  if not isinstance(sparse_shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_shape' argument to "
        "'ragged_cross' Op, not %r." % sparse_shape)
  if len(sparse_shape) != _attr_Nsparse:
    raise ValueError(
        "List argument 'sparse_shape' to 'ragged_cross' Op with length %d "
        "must match length %d of argument 'sparse_indices'." %
        (len(sparse_shape), _attr_Nsparse))
  input_order = _execute.make_str(input_order, "input_order")
  hashed_output = _execute.make_bool(hashed_output, "hashed_output")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  hash_key = _execute.make_int(hash_key, "hash_key")
  out_values_type = _execute.make_type(out_values_type, "out_values_type")
  out_row_splits_type = _execute.make_type(out_row_splits_type, "out_row_splits_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedCross", ragged_values=ragged_values,
                       ragged_row_splits=ragged_row_splits,
                       sparse_indices=sparse_indices,
                       sparse_values=sparse_values, sparse_shape=sparse_shape,
                       dense_inputs=dense_inputs, input_order=input_order,
                       hashed_output=hashed_output, num_buckets=num_buckets,
                       hash_key=hash_key, out_values_type=out_values_type,
                       out_row_splits_type=out_row_splits_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Nsparse", _op._get_attr_int("Nsparse"), "input_order",
              _op.get_attr("input_order"), "hashed_output",
              _op._get_attr_bool("hashed_output"), "num_buckets",
              _op._get_attr_int("num_buckets"), "hash_key",
              _op._get_attr_int("hash_key"), "ragged_values_types",
              _op.get_attr("ragged_values_types"), "ragged_splits_types",
              _op.get_attr("ragged_splits_types"), "sparse_values_types",
              _op.get_attr("sparse_values_types"), "dense_types",
              _op.get_attr("dense_types"), "out_values_type",
              _op._get_attr_type("out_values_type"), "out_row_splits_type",
              _op._get_attr_type("out_row_splits_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedCross", _inputs_flat, _attrs, _result)
  _result = _RaggedCrossOutput._make(_result)
  return _result

RaggedCross = tf_export("raw_ops.RaggedCross")(_ops.to_raw_op(ragged_cross))


def ragged_cross_eager_fallback(ragged_values, ragged_row_splits, sparse_indices: Annotated[List[Any], _atypes.Int64], sparse_values, sparse_shape: Annotated[List[Any], _atypes.Int64], dense_inputs, input_order: str, hashed_output: bool, num_buckets: int, hash_key: int, out_values_type: TV_RaggedCross_out_values_type, out_row_splits_type: TV_RaggedCross_out_row_splits_type, name, ctx):
  if not isinstance(sparse_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_indices' argument to "
        "'ragged_cross' Op, not %r." % sparse_indices)
  _attr_Nsparse = len(sparse_indices)
  if not isinstance(sparse_shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_shape' argument to "
        "'ragged_cross' Op, not %r." % sparse_shape)
  if len(sparse_shape) != _attr_Nsparse:
    raise ValueError(
        "List argument 'sparse_shape' to 'ragged_cross' Op with length %d "
        "must match length %d of argument 'sparse_indices'." %
        (len(sparse_shape), _attr_Nsparse))
  input_order = _execute.make_str(input_order, "input_order")
  hashed_output = _execute.make_bool(hashed_output, "hashed_output")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  hash_key = _execute.make_int(hash_key, "hash_key")
  out_values_type = _execute.make_type(out_values_type, "out_values_type")
  out_row_splits_type = _execute.make_type(out_row_splits_type, "out_row_splits_type")
  _attr_ragged_values_types, ragged_values = _execute.convert_to_mixed_eager_tensors(ragged_values, ctx)
  _attr_ragged_splits_types, ragged_row_splits = _execute.convert_to_mixed_eager_tensors(ragged_row_splits, ctx)
  _attr_sparse_values_types, sparse_values = _execute.convert_to_mixed_eager_tensors(sparse_values, ctx)
  _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
  sparse_indices = _ops.convert_n_to_tensor(sparse_indices, _dtypes.int64)
  sparse_shape = _ops.convert_n_to_tensor(sparse_shape, _dtypes.int64)
  _inputs_flat = list(ragged_values) + list(ragged_row_splits) + list(sparse_indices) + list(sparse_values) + list(sparse_shape) + list(dense_inputs)
  _attrs = ("Nsparse", _attr_Nsparse, "input_order", input_order,
  "hashed_output", hashed_output, "num_buckets", num_buckets, "hash_key",
  hash_key, "ragged_values_types", _attr_ragged_values_types,
  "ragged_splits_types", _attr_ragged_splits_types, "sparse_values_types",
  _attr_sparse_values_types, "dense_types", _attr_dense_types,
  "out_values_type", out_values_type, "out_row_splits_type",
  out_row_splits_type)
  _result = _execute.execute(b"RaggedCross", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedCross", _inputs_flat, _attrs, _result)
  _result = _RaggedCrossOutput._make(_result)
  return _result

_RaggedFillEmptyRowsOutput = collections.namedtuple(
    "RaggedFillEmptyRows",
    ["output_value_rowids", "output_values", "empty_row_indicator", "reverse_index_map"])


TV_RaggedFillEmptyRows_T = TypeVar("TV_RaggedFillEmptyRows_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ragged_fill_empty_rows')
def ragged_fill_empty_rows(value_rowids: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedFillEmptyRows_T], nrows: Annotated[Any, _atypes.Int64], default_value: Annotated[Any, TV_RaggedFillEmptyRows_T], name=None):
  r"""TODO: add doc.

  Args:
    value_rowids: A `Tensor` of type `int64`.
    values: A `Tensor`.
    nrows: A `Tensor` of type `int64`.
    default_value: A `Tensor`. Must have the same type as `values`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_value_rowids, output_values, empty_row_indicator, reverse_index_map).

    output_value_rowids: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `values`.
    empty_row_indicator: A `Tensor` of type `bool`.
    reverse_index_map: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedFillEmptyRows", name, value_rowids, values, nrows,
        default_value)
      _result = _RaggedFillEmptyRowsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ragged_fill_empty_rows(
          (value_rowids, values, nrows, default_value, name,), None)
      if _result is not NotImplemented:
        return _result
      return ragged_fill_empty_rows_eager_fallback(
          value_rowids, values, nrows, default_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ragged_fill_empty_rows, (), dict(value_rowids=value_rowids,
                                             values=values, nrows=nrows,
                                             default_value=default_value,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ragged_fill_empty_rows(
        (value_rowids, values, nrows, default_value, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedFillEmptyRows", value_rowids=value_rowids, values=values,
                               nrows=nrows, default_value=default_value,
                               name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ragged_fill_empty_rows, (), dict(value_rowids=value_rowids,
                                           values=values, nrows=nrows,
                                           default_value=default_value,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedFillEmptyRows", _inputs_flat, _attrs, _result)
  _result = _RaggedFillEmptyRowsOutput._make(_result)
  return _result

RaggedFillEmptyRows = tf_export("raw_ops.RaggedFillEmptyRows")(_ops.to_raw_op(ragged_fill_empty_rows))
_dispatcher_for_ragged_fill_empty_rows = ragged_fill_empty_rows._tf_type_based_dispatcher.Dispatch


def ragged_fill_empty_rows_eager_fallback(value_rowids: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedFillEmptyRows_T], nrows: Annotated[Any, _atypes.Int64], default_value: Annotated[Any, TV_RaggedFillEmptyRows_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([values, default_value], ctx, [])
  (values, default_value) = _inputs_T
  value_rowids = _ops.convert_to_tensor(value_rowids, _dtypes.int64)
  nrows = _ops.convert_to_tensor(nrows, _dtypes.int64)
  _inputs_flat = [value_rowids, values, nrows, default_value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RaggedFillEmptyRows", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedFillEmptyRows", _inputs_flat, _attrs, _result)
  _result = _RaggedFillEmptyRowsOutput._make(_result)
  return _result

_RaggedFillEmptyRowsGradOutput = collections.namedtuple(
    "RaggedFillEmptyRowsGrad",
    ["d_values", "d_default_value"])


TV_RaggedFillEmptyRowsGrad_T = TypeVar("TV_RaggedFillEmptyRowsGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ragged_fill_empty_rows_grad')
def ragged_fill_empty_rows_grad(reverse_index_map: Annotated[Any, _atypes.Int64], grad_values: Annotated[Any, TV_RaggedFillEmptyRowsGrad_T], name=None):
  r"""TODO: add doc.

  Args:
    reverse_index_map: A `Tensor` of type `int64`.
    grad_values: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_values, d_default_value).

    d_values: A `Tensor`. Has the same type as `grad_values`.
    d_default_value: A `Tensor`. Has the same type as `grad_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedFillEmptyRowsGrad", name, reverse_index_map, grad_values)
      _result = _RaggedFillEmptyRowsGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ragged_fill_empty_rows_grad(
          (reverse_index_map, grad_values, name,), None)
      if _result is not NotImplemented:
        return _result
      return ragged_fill_empty_rows_grad_eager_fallback(
          reverse_index_map, grad_values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ragged_fill_empty_rows_grad, (), dict(reverse_index_map=reverse_index_map,
                                                  grad_values=grad_values,
                                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ragged_fill_empty_rows_grad(
        (reverse_index_map, grad_values, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedFillEmptyRowsGrad", reverse_index_map=reverse_index_map,
                                   grad_values=grad_values, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ragged_fill_empty_rows_grad, (), dict(reverse_index_map=reverse_index_map,
                                                grad_values=grad_values,
                                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedFillEmptyRowsGrad", _inputs_flat, _attrs, _result)
  _result = _RaggedFillEmptyRowsGradOutput._make(_result)
  return _result

RaggedFillEmptyRowsGrad = tf_export("raw_ops.RaggedFillEmptyRowsGrad")(_ops.to_raw_op(ragged_fill_empty_rows_grad))
_dispatcher_for_ragged_fill_empty_rows_grad = ragged_fill_empty_rows_grad._tf_type_based_dispatcher.Dispatch


def ragged_fill_empty_rows_grad_eager_fallback(reverse_index_map: Annotated[Any, _atypes.Int64], grad_values: Annotated[Any, TV_RaggedFillEmptyRowsGrad_T], name, ctx):
  _attr_T, (grad_values,) = _execute.args_to_matching_eager([grad_values], ctx, [])
  reverse_index_map = _ops.convert_to_tensor(reverse_index_map, _dtypes.int64)
  _inputs_flat = [reverse_index_map, grad_values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RaggedFillEmptyRowsGrad", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedFillEmptyRowsGrad", _inputs_flat, _attrs, _result)
  _result = _RaggedFillEmptyRowsGradOutput._make(_result)
  return _result

_RaggedGatherOutput = collections.namedtuple(
    "RaggedGather",
    ["output_nested_splits", "output_dense_values"])


TV_RaggedGather_Tvalues = TypeVar("TV_RaggedGather_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedGather_Tindices = TypeVar("TV_RaggedGather_Tindices", _atypes.Int32, _atypes.Int64)
TV_RaggedGather_Tsplits = TypeVar("TV_RaggedGather_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_gather(params_nested_splits: Annotated[List[Any], TV_RaggedGather_Tsplits], params_dense_values: Annotated[Any, TV_RaggedGather_Tvalues], indices: Annotated[Any, TV_RaggedGather_Tindices], OUTPUT_RAGGED_RANK: int, name=None):
  r"""Gather ragged slices from `params` axis `0` according to `indices`.

  Outputs a `RaggedTensor` output composed from `output_dense_values` and
  `output_nested_splits`, such that:

  ```python
  output.shape = indices.shape + params.shape[1:]
  output.ragged_rank = indices.shape.ndims + params.ragged_rank
  output[i...j, d0...dn] = params[indices[i...j], d0...dn]
  ```

  where

  * `params =
     ragged.from_nested_row_splits(params_dense_values, params_nested_splits)`
     provides the values that should be gathered.
  * `indices` ia a dense tensor with dtype `int32` or `int64`, indicating which
     values should be gathered.
  * `output =
     ragged.from_nested_row_splits(output_dense_values, output_nested_splits)`
     is the output tensor.

  (Note: This c++ op is used to implement the higher-level python
  `tf.ragged.gather` op, which also supports ragged indices.)

  Args:
    params_nested_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      The `nested_row_splits` tensors that define the row-partitioning for the
      `params` RaggedTensor input.
    params_dense_values: A `Tensor`.
      The `flat_values` for the `params` RaggedTensor. There was a terminology change
      at the python level from dense_values to flat_values, so dense_values is the
      deprecated name.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Indices in the outermost dimension of `params` of the values that should be
      gathered.
    OUTPUT_RAGGED_RANK: An `int` that is `>= 0`.
      The ragged rank of the output RaggedTensor. `output_nested_splits` will contain
      this number of `row_splits` tensors. This value should equal
      `indices.shape.ndims + params.ragged_rank - 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_nested_splits, output_dense_values).

    output_nested_splits: A list of `OUTPUT_RAGGED_RANK` `Tensor` objects with the same type as `params_nested_splits`.
    output_dense_values: A `Tensor`. Has the same type as `params_dense_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedGather", name, params_nested_splits, params_dense_values,
        indices, "OUTPUT_RAGGED_RANK", OUTPUT_RAGGED_RANK)
      _result = _RaggedGatherOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_gather_eager_fallback(
          params_nested_splits, params_dense_values, indices,
          OUTPUT_RAGGED_RANK=OUTPUT_RAGGED_RANK, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(params_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'params_nested_splits' argument to "
        "'ragged_gather' Op, not %r." % params_nested_splits)
  _attr_PARAMS_RAGGED_RANK = len(params_nested_splits)
  OUTPUT_RAGGED_RANK = _execute.make_int(OUTPUT_RAGGED_RANK, "OUTPUT_RAGGED_RANK")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedGather", params_nested_splits=params_nested_splits,
                        params_dense_values=params_dense_values,
                        indices=indices,
                        OUTPUT_RAGGED_RANK=OUTPUT_RAGGED_RANK, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tvalues", _op._get_attr_type("Tvalues"), "Tindices",
              _op._get_attr_type("Tindices"), "Tsplits",
              _op._get_attr_type("Tsplits"), "PARAMS_RAGGED_RANK",
              _op._get_attr_int("PARAMS_RAGGED_RANK"), "OUTPUT_RAGGED_RANK",
              _op._get_attr_int("OUTPUT_RAGGED_RANK"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedGather", _inputs_flat, _attrs, _result)
  _result = [_result[:OUTPUT_RAGGED_RANK]] + _result[OUTPUT_RAGGED_RANK:]
  _result = _RaggedGatherOutput._make(_result)
  return _result

RaggedGather = tf_export("raw_ops.RaggedGather")(_ops.to_raw_op(ragged_gather))


def ragged_gather_eager_fallback(params_nested_splits: Annotated[List[Any], TV_RaggedGather_Tsplits], params_dense_values: Annotated[Any, TV_RaggedGather_Tvalues], indices: Annotated[Any, TV_RaggedGather_Tindices], OUTPUT_RAGGED_RANK: int, name, ctx):
  if not isinstance(params_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'params_nested_splits' argument to "
        "'ragged_gather' Op, not %r." % params_nested_splits)
  _attr_PARAMS_RAGGED_RANK = len(params_nested_splits)
  OUTPUT_RAGGED_RANK = _execute.make_int(OUTPUT_RAGGED_RANK, "OUTPUT_RAGGED_RANK")
  _attr_Tvalues, (params_dense_values,) = _execute.args_to_matching_eager([params_dense_values], ctx, [])
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tsplits, params_nested_splits = _execute.args_to_matching_eager(list(params_nested_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = list(params_nested_splits) + [params_dense_values, indices]
  _attrs = ("Tvalues", _attr_Tvalues, "Tindices", _attr_Tindices, "Tsplits",
  _attr_Tsplits, "PARAMS_RAGGED_RANK", _attr_PARAMS_RAGGED_RANK,
  "OUTPUT_RAGGED_RANK", OUTPUT_RAGGED_RANK)
  _result = _execute.execute(b"RaggedGather", OUTPUT_RAGGED_RANK + 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedGather", _inputs_flat, _attrs, _result)
  _result = [_result[:OUTPUT_RAGGED_RANK]] + _result[OUTPUT_RAGGED_RANK:]
  _result = _RaggedGatherOutput._make(_result)
  return _result

