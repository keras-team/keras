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

TV_NcclAllReduce_T = TypeVar("TV_NcclAllReduce_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def nccl_all_reduce(input: Annotated[Any, TV_NcclAllReduce_T], reduction: str, num_devices: int, shared_name: str, name=None) -> Annotated[Any, TV_NcclAllReduce_T]:
  r"""Outputs a tensor containing the reduction across all input tensors.

  Outputs a tensor containing the reduction across all input tensors passed to ops
  within the same `shared_name.

  The graph should be constructed so if one op runs with shared_name value `c`,
  then `num_devices` ops will run with shared_name value `c`.  Failure to do so
  will cause the graph execution to fail to complete.

  input: the input to the reduction
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.
  num_devices: The number of devices participating in this reduction.
  shared_name: Identifier that shared between ops of the same reduction.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    num_devices: An `int`.
    shared_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NcclAllReduce", name, input, "reduction", reduction,
        "num_devices", num_devices, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return nccl_all_reduce_eager_fallback(
          input, reduction=reduction, num_devices=num_devices,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  reduction = _execute.make_str(reduction, "reduction")
  num_devices = _execute.make_int(num_devices, "num_devices")
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NcclAllReduce", input=input, reduction=reduction,
                         num_devices=num_devices, shared_name=shared_name,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reduction", _op.get_attr("reduction"), "T",
              _op._get_attr_type("T"), "num_devices",
              _op._get_attr_int("num_devices"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NcclAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NcclAllReduce = tf_export("raw_ops.NcclAllReduce")(_ops.to_raw_op(nccl_all_reduce))


def nccl_all_reduce_eager_fallback(input: Annotated[Any, TV_NcclAllReduce_T], reduction: str, num_devices: int, shared_name: str, name, ctx) -> Annotated[Any, TV_NcclAllReduce_T]:
  reduction = _execute.make_str(reduction, "reduction")
  num_devices = _execute.make_int(num_devices, "num_devices")
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input]
  _attrs = ("reduction", reduction, "T", _attr_T, "num_devices", num_devices,
  "shared_name", shared_name)
  _result = _execute.execute(b"NcclAllReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NcclAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NcclBroadcast_T = TypeVar("TV_NcclBroadcast_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def nccl_broadcast(input: Annotated[Any, TV_NcclBroadcast_T], shape, name=None) -> Annotated[Any, TV_NcclBroadcast_T]:
  r"""Sends `input` to all devices that are connected to the output.

  Sends `input` to all devices that are connected to the output.

  The graph should be constructed so that all ops connected to the output have a
  valid device assignment, and the op itself is assigned one of these devices.

  input: The input to the broadcast.
  output: The same as input.
  shape: The shape of the input tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    shape: A `tf.TensorShape` or list of `ints`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NcclBroadcast", name, input, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return nccl_broadcast_eager_fallback(
          input, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NcclBroadcast", input=input, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NcclBroadcast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NcclBroadcast = tf_export("raw_ops.NcclBroadcast")(_ops.to_raw_op(nccl_broadcast))


def nccl_broadcast_eager_fallback(input: Annotated[Any, TV_NcclBroadcast_T], shape, name, ctx) -> Annotated[Any, TV_NcclBroadcast_T]:
  shape = _execute.make_shape(shape, "shape")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "shape", shape)
  _result = _execute.execute(b"NcclBroadcast", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NcclBroadcast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NcclReduce_T = TypeVar("TV_NcclReduce_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def nccl_reduce(input: Annotated[List[Any], TV_NcclReduce_T], reduction: str, name=None) -> Annotated[Any, TV_NcclReduce_T]:
  r"""Reduces `input` from `num_devices` using `reduction` to a single device.

  Reduces `input` from `num_devices` using `reduction` to a single device.

  The graph should be constructed so that all inputs have a valid device
  assignment, and the op itself is assigned one of these devices.

  input: The input to the reduction.
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type in: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NcclReduce", name, input, "reduction", reduction)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return nccl_reduce_eager_fallback(
          input, reduction=reduction, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'nccl_reduce' Op, not %r." % input)
  _attr_num_devices = len(input)
  reduction = _execute.make_str(reduction, "reduction")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NcclReduce", input=input, reduction=reduction, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reduction", _op.get_attr("reduction"), "T",
              _op._get_attr_type("T"), "num_devices",
              _op._get_attr_int("num_devices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NcclReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NcclReduce = tf_export("raw_ops.NcclReduce")(_ops.to_raw_op(nccl_reduce))


def nccl_reduce_eager_fallback(input: Annotated[List[Any], TV_NcclReduce_T], reduction: str, name, ctx) -> Annotated[Any, TV_NcclReduce_T]:
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'nccl_reduce' Op, not %r." % input)
  _attr_num_devices = len(input)
  reduction = _execute.make_str(reduction, "reduction")
  _attr_T, input = _execute.args_to_matching_eager(list(input), ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = list(input)
  _attrs = ("reduction", reduction, "T", _attr_T, "num_devices",
  _attr_num_devices)
  _result = _execute.execute(b"NcclReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NcclReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

