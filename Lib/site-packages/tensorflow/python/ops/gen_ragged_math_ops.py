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
_RaggedRangeOutput = collections.namedtuple(
    "RaggedRange",
    ["rt_nested_splits", "rt_dense_values"])


TV_RaggedRange_T = TypeVar("TV_RaggedRange_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)
TV_RaggedRange_Tsplits = TypeVar("TV_RaggedRange_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_range(starts: Annotated[Any, TV_RaggedRange_T], limits: Annotated[Any, TV_RaggedRange_T], deltas: Annotated[Any, TV_RaggedRange_T], Tsplits:TV_RaggedRange_Tsplits=_dtypes.int64, name=None):
  r"""Returns a `RaggedTensor` containing the specified sequences of numbers.

  
  Returns a `RaggedTensor` `result` composed from `rt_dense_values` and
  `rt_nested_splits`, such that
  `result[i] = range(starts[i], limits[i], deltas[i])`.

  ```python
  (rt_nested_splits, rt_dense_values) = ragged_range(
        starts=[2, 5, 8], limits=[3, 5, 12], deltas=1)
  result = tf.ragged.from_row_splits(rt_dense_values, rt_nested_splits)
  print(result)
  <tf.RaggedTensor [[2], [], [8, 9, 10, 11]] >
  ```

  The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
  The vector inputs must all have the same size.  Scalar inputs are broadcast
  to match the size of the vector inputs.

  Args:
    starts: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `float64`, `int32`, `int64`.
      The starts of each range.
    limits: A `Tensor`. Must have the same type as `starts`.
      The limits of each range.
    deltas: A `Tensor`. Must have the same type as `starts`.
      The deltas of each range.
    Tsplits: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (rt_nested_splits, rt_dense_values).

    rt_nested_splits: A `Tensor` of type `Tsplits`.
    rt_dense_values: A `Tensor`. Has the same type as `starts`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedRange", name, starts, limits, deltas, "Tsplits", Tsplits)
      _result = _RaggedRangeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_range_eager_fallback(
          starts, limits, deltas, Tsplits=Tsplits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedRange", starts=starts, limits=limits, deltas=deltas,
                       Tsplits=Tsplits, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedRange", _inputs_flat, _attrs, _result)
  _result = _RaggedRangeOutput._make(_result)
  return _result

RaggedRange = tf_export("raw_ops.RaggedRange")(_ops.to_raw_op(ragged_range))


def ragged_range_eager_fallback(starts: Annotated[Any, TV_RaggedRange_T], limits: Annotated[Any, TV_RaggedRange_T], deltas: Annotated[Any, TV_RaggedRange_T], Tsplits: TV_RaggedRange_Tsplits, name, ctx):
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([starts, limits, deltas], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  (starts, limits, deltas) = _inputs_T
  _inputs_flat = [starts, limits, deltas]
  _attrs = ("T", _attr_T, "Tsplits", Tsplits)
  _result = _execute.execute(b"RaggedRange", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedRange", _inputs_flat, _attrs, _result)
  _result = _RaggedRangeOutput._make(_result)
  return _result

