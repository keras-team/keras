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

TV_Roll_T = TypeVar("TV_Roll_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Roll_Tshift = TypeVar("TV_Roll_Tshift", _atypes.Int32, _atypes.Int64)
TV_Roll_Taxis = TypeVar("TV_Roll_Taxis", _atypes.Int32, _atypes.Int64)

def roll(input: Annotated[Any, TV_Roll_T], shift: Annotated[Any, TV_Roll_Tshift], axis: Annotated[Any, TV_Roll_Taxis], name=None) -> Annotated[Any, TV_Roll_T]:
  r"""Rolls the elements of a tensor along an axis.

  The elements are shifted positively (towards larger indices) by the offset of
  `shift` along the dimension of `axis`. Negative `shift` values will shift
  elements in the opposite direction. Elements that roll passed the last position
  will wrap around to the first and vice versa. Multiple shifts along multiple
  axes may be specified.

  For example:

  ```
  # 't' is [0, 1, 2, 3, 4]
  roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]

  # shifting along multiple dimensions
  # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
  roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]

  # shifting along the same axis multiple times
  # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
  roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
  ```

  Args:
    input: A `Tensor`.
    shift: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
      elements are shifted positively (towards larger indices) along the dimension
      specified by `axis[i]`. Negative shifts will roll the elements in the opposite
      direction.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
      `shift[i]` should occur. If the same axis is referenced more than once, the
      total shift for that axis will be the sum of all the shifts that belong to that
      axis.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Roll", name, input, shift, axis)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return roll_eager_fallback(
          input, shift, axis, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Roll", input=input, shift=shift, axis=axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tshift",
              _op._get_attr_type("Tshift"), "Taxis",
              _op._get_attr_type("Taxis"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Roll", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Roll = tf_export("raw_ops.Roll")(_ops.to_raw_op(roll))


def roll_eager_fallback(input: Annotated[Any, TV_Roll_T], shift: Annotated[Any, TV_Roll_Tshift], axis: Annotated[Any, TV_Roll_Taxis], name, ctx) -> Annotated[Any, TV_Roll_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tshift, (shift,) = _execute.args_to_matching_eager([shift], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Taxis, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input, shift, axis]
  _attrs = ("T", _attr_T, "Tshift", _attr_Tshift, "Taxis", _attr_Taxis)
  _result = _execute.execute(b"Roll", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Roll", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

