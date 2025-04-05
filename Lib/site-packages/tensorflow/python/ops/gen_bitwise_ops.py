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

TV_BitwiseAnd_T = TypeVar("TV_BitwiseAnd_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.bitwise_and')
def bitwise_and(x: Annotated[Any, TV_BitwiseAnd_T], y: Annotated[Any, TV_BitwiseAnd_T], name=None) -> Annotated[Any, TV_BitwiseAnd_T]:
  r"""Elementwise computes the bitwise AND of `x` and `y`.

  The result will have those bits set, that are set in both `x` and `y`. The
  computation is performed on the underlying representations of `x` and `y`.

  For example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
                tf.uint8, tf.uint16, tf.uint32, tf.uint64]

  for dtype in dtype_list:
    lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
    exp = tf.constant([0, 0, 3, 10], dtype=tf.float32)

    res = bitwise_ops.bitwise_and(lhs, rhs)
    tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BitwiseAnd", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_bitwise_and(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return bitwise_and_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            bitwise_and, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_bitwise_and(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BitwiseAnd", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          bitwise_and, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BitwiseAnd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BitwiseAnd = tf_export("raw_ops.BitwiseAnd")(_ops.to_raw_op(bitwise_and))
_dispatcher_for_bitwise_and = bitwise_and._tf_type_based_dispatcher.Dispatch


def bitwise_and_eager_fallback(x: Annotated[Any, TV_BitwiseAnd_T], y: Annotated[Any, TV_BitwiseAnd_T], name, ctx) -> Annotated[Any, TV_BitwiseAnd_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BitwiseAnd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BitwiseAnd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BitwiseOr_T = TypeVar("TV_BitwiseOr_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.bitwise_or')
def bitwise_or(x: Annotated[Any, TV_BitwiseOr_T], y: Annotated[Any, TV_BitwiseOr_T], name=None) -> Annotated[Any, TV_BitwiseOr_T]:
  r"""Elementwise computes the bitwise OR of `x` and `y`.

  The result will have those bits set, that are set in `x`, `y` or both. The
  computation is performed on the underlying representations of `x` and `y`.

  For example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
                tf.uint8, tf.uint16, tf.uint32, tf.uint64]

  for dtype in dtype_list:
    lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
    exp = tf.constant([5, 5, 7, 15], dtype=tf.float32)

    res = bitwise_ops.bitwise_or(lhs, rhs)
    tf.assert_equal(tf.cast(res,  tf.float32), exp)  # TRUE
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BitwiseOr", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_bitwise_or(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return bitwise_or_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            bitwise_or, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_bitwise_or(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BitwiseOr", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          bitwise_or, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BitwiseOr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BitwiseOr = tf_export("raw_ops.BitwiseOr")(_ops.to_raw_op(bitwise_or))
_dispatcher_for_bitwise_or = bitwise_or._tf_type_based_dispatcher.Dispatch


def bitwise_or_eager_fallback(x: Annotated[Any, TV_BitwiseOr_T], y: Annotated[Any, TV_BitwiseOr_T], name, ctx) -> Annotated[Any, TV_BitwiseOr_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BitwiseOr", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BitwiseOr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BitwiseXor_T = TypeVar("TV_BitwiseXor_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.bitwise_xor')
def bitwise_xor(x: Annotated[Any, TV_BitwiseXor_T], y: Annotated[Any, TV_BitwiseXor_T], name=None) -> Annotated[Any, TV_BitwiseXor_T]:
  r"""Elementwise computes the bitwise XOR of `x` and `y`.

  The result will have those bits set, that are different in `x` and `y`. The
  computation is performed on the underlying representations of `x` and `y`.

  For example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
                tf.uint8, tf.uint16, tf.uint32, tf.uint64]

  for dtype in dtype_list:
    lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
    exp = tf.constant([5, 5, 4, 5],  dtype=tf.float32)

    res = bitwise_ops.bitwise_xor(lhs, rhs)
    tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BitwiseXor", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_bitwise_xor(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return bitwise_xor_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            bitwise_xor, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_bitwise_xor(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BitwiseXor", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          bitwise_xor, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BitwiseXor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BitwiseXor = tf_export("raw_ops.BitwiseXor")(_ops.to_raw_op(bitwise_xor))
_dispatcher_for_bitwise_xor = bitwise_xor._tf_type_based_dispatcher.Dispatch


def bitwise_xor_eager_fallback(x: Annotated[Any, TV_BitwiseXor_T], y: Annotated[Any, TV_BitwiseXor_T], name, ctx) -> Annotated[Any, TV_BitwiseXor_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BitwiseXor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BitwiseXor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Invert_T = TypeVar("TV_Invert_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.invert')
def invert(x: Annotated[Any, TV_Invert_T], name=None) -> Annotated[Any, TV_Invert_T]:
  r"""Invert (flip) each bit of supported types; for example, type `uint8` value 01010101 becomes 10101010.

  Flip each bit of supported types.  For example, type `int8` (decimal 2) binary 00000010 becomes (decimal -3) binary 11111101.
  This operation is performed on each element of the tensor argument `x`.

  Example:
  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops

  # flip 2 (00000010) to -3 (11111101)
  tf.assert_equal(-3, bitwise_ops.invert(2))

  dtype_list = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
                dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]

  inputs = [0, 5, 3, 14]
  for dtype in dtype_list:
    # Because of issues with negative numbers, let's test this indirectly.
    # 1. invert(a) and a = 0
    # 2. invert(a) or a = invert(0)
    input_tensor = tf.constant([0, 5, 3, 14], dtype=dtype)
    not_a_and_a, not_a_or_a, not_0 = [bitwise_ops.bitwise_and(
                                        input_tensor, bitwise_ops.invert(input_tensor)),
                                      bitwise_ops.bitwise_or(
                                        input_tensor, bitwise_ops.invert(input_tensor)),
                                      bitwise_ops.invert(
                                        tf.constant(0, dtype=dtype))]

    expected = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    tf.assert_equal(tf.cast(not_a_and_a, tf.float32), expected)

    expected = tf.cast([not_0] * 4, tf.float32)
    tf.assert_equal(tf.cast(not_a_or_a, tf.float32), expected)

    # For unsigned dtypes let's also check the result directly.
    if dtype.is_unsigned:
      inverted = bitwise_ops.invert(input_tensor)
      expected = tf.constant([dtype.max - x for x in inputs], dtype=tf.float32)
      tf.assert_equal(tf.cast(inverted, tf.float32), tf.cast(expected, tf.float32))
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Invert", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_invert(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return invert_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            invert, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_invert(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Invert", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          invert, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Invert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Invert = tf_export("raw_ops.Invert")(_ops.to_raw_op(invert))
_dispatcher_for_invert = invert._tf_type_based_dispatcher.Dispatch


def invert_eager_fallback(x: Annotated[Any, TV_Invert_T], name, ctx) -> Annotated[Any, TV_Invert_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Invert", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Invert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LeftShift_T = TypeVar("TV_LeftShift_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.left_shift')
def left_shift(x: Annotated[Any, TV_LeftShift_T], y: Annotated[Any, TV_LeftShift_T], name=None) -> Annotated[Any, TV_LeftShift_T]:
  r"""Elementwise computes the bitwise left-shift of `x` and `y`.

  If `y` is negative, or greater than or equal to the width of `x` in bits the
  result is implementation defined.

  Example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  import numpy as np
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]

  for dtype in dtype_list:
    lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

    left_shift_result = bitwise_ops.left_shift(lhs, rhs)

    print(left_shift_result)

  # This will print:
  # tf.Tensor([ -32   -5 -128    0], shape=(4,), dtype=int8)
  # tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int16)
  # tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int32)
  # tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int64)

  lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
  rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
  bitwise_ops.left_shift(lhs, rhs)
  # <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LeftShift", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_left_shift(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return left_shift_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            left_shift, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_left_shift(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LeftShift", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          left_shift, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LeftShift", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LeftShift = tf_export("raw_ops.LeftShift")(_ops.to_raw_op(left_shift))
_dispatcher_for_left_shift = left_shift._tf_type_based_dispatcher.Dispatch


def left_shift_eager_fallback(x: Annotated[Any, TV_LeftShift_T], y: Annotated[Any, TV_LeftShift_T], name, ctx) -> Annotated[Any, TV_LeftShift_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"LeftShift", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LeftShift", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PopulationCount_T = TypeVar("TV_PopulationCount_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def population_count(x: Annotated[Any, TV_PopulationCount_T], name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).

  For each entry in `x`, calculates the number of `1` (on) bits in the binary
  representation of that entry.

  **NOTE**: It is more efficient to first `tf.bitcast` your tensors into
  `int32` or `int64` and perform the bitcount on the result, than to feed in
  8- or 16-bit inputs and then aggregate the resulting counts.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PopulationCount", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return population_count_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PopulationCount", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PopulationCount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PopulationCount = tf_export("raw_ops.PopulationCount")(_ops.to_raw_op(population_count))


def population_count_eager_fallback(x: Annotated[Any, TV_PopulationCount_T], name, ctx) -> Annotated[Any, _atypes.UInt8]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"PopulationCount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PopulationCount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RightShift_T = TypeVar("TV_RightShift_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.right_shift')
def right_shift(x: Annotated[Any, TV_RightShift_T], y: Annotated[Any, TV_RightShift_T], name=None) -> Annotated[Any, TV_RightShift_T]:
  r"""Elementwise computes the bitwise right-shift of `x` and `y`.

  Performs a logical shift for unsigned integer types, and an arithmetic shift
  for signed integer types.

  If `y` is negative, or greater than or equal to than the width of `x` in bits
  the result is implementation defined.

  Example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  import numpy as np
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]

  for dtype in dtype_list:
    lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

    right_shift_result = bitwise_ops.right_shift(lhs, rhs)

    print(right_shift_result)

  # This will print:
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int8)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int16)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int32)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int64)

  lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
  rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
  bitwise_ops.right_shift(lhs, rhs)
  # <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RightShift", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_right_shift(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return right_shift_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            right_shift, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_right_shift(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RightShift", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          right_shift, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RightShift", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RightShift = tf_export("raw_ops.RightShift")(_ops.to_raw_op(right_shift))
_dispatcher_for_right_shift = right_shift._tf_type_based_dispatcher.Dispatch


def right_shift_eager_fallback(x: Annotated[Any, TV_RightShift_T], y: Annotated[Any, TV_RightShift_T], name, ctx) -> Annotated[Any, TV_RightShift_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RightShift", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RightShift", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

