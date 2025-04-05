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
_DenseCountSparseOutputOutput = collections.namedtuple(
    "DenseCountSparseOutput",
    ["output_indices", "output_values", "output_dense_shape"])


TV_DenseCountSparseOutput_T = TypeVar("TV_DenseCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_DenseCountSparseOutput_output_type = TypeVar("TV_DenseCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def dense_count_sparse_output(values: Annotated[Any, TV_DenseCountSparseOutput_T], weights: Annotated[Any, TV_DenseCountSparseOutput_output_type], binary_output: bool, minlength:int=-1, maxlength:int=-1, name=None):
  r"""Performs sparse-output bin counting for a tf.tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing data to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values. May
      also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseCountSparseOutput", name, values, weights, "minlength",
        minlength, "maxlength", maxlength, "binary_output", binary_output)
      _result = _DenseCountSparseOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_count_sparse_output_eager_fallback(
          values, weights, minlength=minlength, maxlength=maxlength,
          binary_output=binary_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseCountSparseOutput", values=values, weights=weights,
                                  binary_output=binary_output,
                                  minlength=minlength, maxlength=maxlength,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "minlength",
              _op._get_attr_int("minlength"), "maxlength",
              _op._get_attr_int("maxlength"), "binary_output",
              _op._get_attr_bool("binary_output"), "output_type",
              _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _DenseCountSparseOutputOutput._make(_result)
  return _result

DenseCountSparseOutput = tf_export("raw_ops.DenseCountSparseOutput")(_ops.to_raw_op(dense_count_sparse_output))


def dense_count_sparse_output_eager_fallback(values: Annotated[Any, TV_DenseCountSparseOutput_T], weights: Annotated[Any, TV_DenseCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx):
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_output_type, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [values, weights]
  _attrs = ("T", _attr_T, "minlength", minlength, "maxlength", maxlength,
  "binary_output", binary_output, "output_type", _attr_output_type)
  _result = _execute.execute(b"DenseCountSparseOutput", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _DenseCountSparseOutputOutput._make(_result)
  return _result

_RaggedCountSparseOutputOutput = collections.namedtuple(
    "RaggedCountSparseOutput",
    ["output_indices", "output_values", "output_dense_shape"])


TV_RaggedCountSparseOutput_T = TypeVar("TV_RaggedCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_RaggedCountSparseOutput_output_type = TypeVar("TV_RaggedCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def ragged_count_sparse_output(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedCountSparseOutput_T], weights: Annotated[Any, TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength:int=-1, maxlength:int=-1, name=None):
  r"""Performs sparse-output bin counting for a ragged tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    splits: A `Tensor` of type `int64`.
      Tensor containing the row splits of the ragged tensor to count.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing values of the sparse tensor to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values.
      May also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedCountSparseOutput", name, splits, values, weights,
        "minlength", minlength, "maxlength", maxlength, "binary_output",
        binary_output)
      _result = _RaggedCountSparseOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_count_sparse_output_eager_fallback(
          splits, values, weights, minlength=minlength, maxlength=maxlength,
          binary_output=binary_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedCountSparseOutput", splits=splits, values=values,
                                   weights=weights,
                                   binary_output=binary_output,
                                   minlength=minlength, maxlength=maxlength,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "minlength",
              _op._get_attr_int("minlength"), "maxlength",
              _op._get_attr_int("maxlength"), "binary_output",
              _op._get_attr_bool("binary_output"), "output_type",
              _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _RaggedCountSparseOutputOutput._make(_result)
  return _result

RaggedCountSparseOutput = tf_export("raw_ops.RaggedCountSparseOutput")(_ops.to_raw_op(ragged_count_sparse_output))


def ragged_count_sparse_output_eager_fallback(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedCountSparseOutput_T], weights: Annotated[Any, TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx):
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_output_type, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  splits = _ops.convert_to_tensor(splits, _dtypes.int64)
  _inputs_flat = [splits, values, weights]
  _attrs = ("T", _attr_T, "minlength", minlength, "maxlength", maxlength,
  "binary_output", binary_output, "output_type", _attr_output_type)
  _result = _execute.execute(b"RaggedCountSparseOutput", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _RaggedCountSparseOutputOutput._make(_result)
  return _result

_SparseCountSparseOutputOutput = collections.namedtuple(
    "SparseCountSparseOutput",
    ["output_indices", "output_values", "output_dense_shape"])


TV_SparseCountSparseOutput_T = TypeVar("TV_SparseCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_SparseCountSparseOutput_output_type = TypeVar("TV_SparseCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def sparse_count_sparse_output(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseCountSparseOutput_T], dense_shape: Annotated[Any, _atypes.Int64], weights: Annotated[Any, TV_SparseCountSparseOutput_output_type], binary_output: bool, minlength:int=-1, maxlength:int=-1, name=None):
  r"""Performs sparse-output bin counting for a sparse tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    indices: A `Tensor` of type `int64`.
      Tensor containing the indices of the sparse tensor to count.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing values of the sparse tensor to count.
    dense_shape: A `Tensor` of type `int64`.
      Tensor containing the dense shape of the sparse tensor to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values.
      May also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseCountSparseOutput", name, indices, values, dense_shape,
        weights, "minlength", minlength, "maxlength", maxlength,
        "binary_output", binary_output)
      _result = _SparseCountSparseOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_count_sparse_output_eager_fallback(
          indices, values, dense_shape, weights, minlength=minlength,
          maxlength=maxlength, binary_output=binary_output, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseCountSparseOutput", indices=indices, values=values,
                                   dense_shape=dense_shape, weights=weights,
                                   binary_output=binary_output,
                                   minlength=minlength, maxlength=maxlength,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "minlength",
              _op._get_attr_int("minlength"), "maxlength",
              _op._get_attr_int("maxlength"), "binary_output",
              _op._get_attr_bool("binary_output"), "output_type",
              _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _SparseCountSparseOutputOutput._make(_result)
  return _result

SparseCountSparseOutput = tf_export("raw_ops.SparseCountSparseOutput")(_ops.to_raw_op(sparse_count_sparse_output))


def sparse_count_sparse_output_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseCountSparseOutput_T], dense_shape: Annotated[Any, _atypes.Int64], weights: Annotated[Any, TV_SparseCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx):
  binary_output = _execute.make_bool(binary_output, "binary_output")
  if minlength is None:
    minlength = -1
  minlength = _execute.make_int(minlength, "minlength")
  if maxlength is None:
    maxlength = -1
  maxlength = _execute.make_int(maxlength, "maxlength")
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_output_type, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [indices, values, dense_shape, weights]
  _attrs = ("T", _attr_T, "minlength", minlength, "maxlength", maxlength,
  "binary_output", binary_output, "output_type", _attr_output_type)
  _result = _execute.execute(b"SparseCountSparseOutput", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseCountSparseOutput", _inputs_flat, _attrs, _result)
  _result = _SparseCountSparseOutputOutput._make(_result)
  return _result

