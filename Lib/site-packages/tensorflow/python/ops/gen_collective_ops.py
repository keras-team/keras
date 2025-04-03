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

TV_CollectiveAllToAllV2_T = TypeVar("TV_CollectiveAllToAllV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_all_to_all_v2(input: Annotated[Any, TV_CollectiveAllToAllV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], communication_hint:str="auto", timeout_seconds:float=0, is_stateless:bool=False, name=None) -> Annotated[Any, TV_CollectiveAllToAllV2_T]:
  r"""Mutually exchanges multiple tensors of identical type and shape.

  `is_stateless` means each op does not need control dependencies to other
  collective ops. In this case, keys that are unique at runtime
  (e.g. `instance_key`) should be used to distinguish collective groups.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    ordering_token: A list of `Tensor` objects with type `resource`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    is_stateless: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveAllToAllV2", name, input, group_size, group_key,
        instance_key, ordering_token, "communication_hint",
        communication_hint, "timeout_seconds", timeout_seconds,
        "is_stateless", is_stateless)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_all_to_all_v2_eager_fallback(
          input, group_size, group_key, instance_key, ordering_token,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, is_stateless=is_stateless,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_all_to_all_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveAllToAllV2", input=input, group_size=group_size,
                                group_key=group_key,
                                instance_key=instance_key,
                                ordering_token=ordering_token,
                                communication_hint=communication_hint,
                                timeout_seconds=timeout_seconds,
                                is_stateless=is_stateless, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"), "is_stateless",
              _op._get_attr_bool("is_stateless"), "Nordering_token",
              _op._get_attr_int("Nordering_token"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveAllToAllV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveAllToAllV2 = tf_export("raw_ops.CollectiveAllToAllV2")(_ops.to_raw_op(collective_all_to_all_v2))


def collective_all_to_all_v2_eager_fallback(input: Annotated[Any, TV_CollectiveAllToAllV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], communication_hint: str, timeout_seconds: float, is_stateless: bool, name, ctx) -> Annotated[Any, TV_CollectiveAllToAllV2_T]:
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_all_to_all_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  ordering_token = _ops.convert_n_to_tensor(ordering_token, _dtypes.resource)
  _inputs_flat = [input, group_size, group_key, instance_key] + list(ordering_token)
  _attrs = ("T", _attr_T, "communication_hint", communication_hint,
  "timeout_seconds", timeout_seconds, "is_stateless", is_stateless,
  "Nordering_token", _attr_Nordering_token)
  _result = _execute.execute(b"CollectiveAllToAllV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveAllToAllV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveAllToAllV3_T = TypeVar("TV_CollectiveAllToAllV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_all_to_all_v3(input: Annotated[Any, TV_CollectiveAllToAllV3_T], communicator: Annotated[Any, _atypes.Resource], group_assignment: Annotated[Any, _atypes.Int32], timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveAllToAllV3_T]:
  r"""Mutually exchanges multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    communicator: A `Tensor` of type `resource`.
    group_assignment: A `Tensor` of type `int32`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveAllToAllV3", name, input, communicator,
        group_assignment, "timeout_seconds", timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_all_to_all_v3_eager_fallback(
          input, communicator, group_assignment,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveAllToAllV3", input=input, communicator=communicator,
                                group_assignment=group_assignment,
                                timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveAllToAllV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveAllToAllV3 = tf_export("raw_ops.CollectiveAllToAllV3")(_ops.to_raw_op(collective_all_to_all_v3))


def collective_all_to_all_v3_eager_fallback(input: Annotated[Any, TV_CollectiveAllToAllV3_T], communicator: Annotated[Any, _atypes.Resource], group_assignment: Annotated[Any, _atypes.Int32], timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveAllToAllV3_T]:
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  communicator = _ops.convert_to_tensor(communicator, _dtypes.resource)
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, communicator, group_assignment]
  _attrs = ("T", _attr_T, "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveAllToAllV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveAllToAllV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_CollectiveAssignGroupV2Output = collections.namedtuple(
    "CollectiveAssignGroupV2",
    ["group_size", "group_key"])


def collective_assign_group_v2(group_assignment: Annotated[Any, _atypes.Int32], device_index: Annotated[Any, _atypes.Int32], base_key: Annotated[Any, _atypes.Int32], name=None):
  r"""Assign group keys based on group assignment.

  Args:
    group_assignment: A `Tensor` of type `int32`.
    device_index: A `Tensor` of type `int32`.
    base_key: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (group_size, group_key).

    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveAssignGroupV2", name, group_assignment, device_index,
        base_key)
      _result = _CollectiveAssignGroupV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_assign_group_v2_eager_fallback(
          group_assignment, device_index, base_key, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveAssignGroupV2", group_assignment=group_assignment,
                                   device_index=device_index,
                                   base_key=base_key, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveAssignGroupV2", _inputs_flat, _attrs, _result)
  _result = _CollectiveAssignGroupV2Output._make(_result)
  return _result

CollectiveAssignGroupV2 = tf_export("raw_ops.CollectiveAssignGroupV2")(_ops.to_raw_op(collective_assign_group_v2))


def collective_assign_group_v2_eager_fallback(group_assignment: Annotated[Any, _atypes.Int32], device_index: Annotated[Any, _atypes.Int32], base_key: Annotated[Any, _atypes.Int32], name, ctx):
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  device_index = _ops.convert_to_tensor(device_index, _dtypes.int32)
  base_key = _ops.convert_to_tensor(base_key, _dtypes.int32)
  _inputs_flat = [group_assignment, device_index, base_key]
  _attrs = None
  _result = _execute.execute(b"CollectiveAssignGroupV2", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveAssignGroupV2", _inputs_flat, _attrs, _result)
  _result = _CollectiveAssignGroupV2Output._make(_result)
  return _result


TV_CollectiveBcastRecv_T = TypeVar("TV_CollectiveBcastRecv_T", _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_bcast_recv(T: TV_CollectiveBcastRecv_T, group_size: int, group_key: int, instance_key: int, shape, communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveBcastRecv_T]:
  r"""Receives a tensor value broadcast from another device.

  Args:
    T: A `tf.DType` from: `tf.bool, tf.float32, tf.half, tf.float64, tf.int32, tf.int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    shape: A `tf.TensorShape` or list of `ints`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveBcastRecv", name, "T", T, "group_size", group_size,
        "group_key", group_key, "instance_key", instance_key, "shape", shape,
        "communication_hint", communication_hint, "timeout_seconds",
        timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_bcast_recv_eager_fallback(
          T=T, group_size=group_size, group_key=group_key,
          instance_key=instance_key, shape=shape,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveBcastRecv", T=T, group_size=group_size,
                               group_key=group_key, instance_key=instance_key,
                               shape=shape,
                               communication_hint=communication_hint,
                               timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "group_size",
              _op._get_attr_int("group_size"), "group_key",
              _op._get_attr_int("group_key"), "instance_key",
              _op._get_attr_int("instance_key"), "shape",
              _op.get_attr("shape"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveBcastRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveBcastRecv = tf_export("raw_ops.CollectiveBcastRecv")(_ops.to_raw_op(collective_bcast_recv))


def collective_bcast_recv_eager_fallback(T: TV_CollectiveBcastRecv_T, group_size: int, group_key: int, instance_key: int, shape, communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveBcastRecv_T]:
  T = _execute.make_type(T, "T")
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _inputs_flat = []
  _attrs = ("T", T, "group_size", group_size, "group_key", group_key,
  "instance_key", instance_key, "shape", shape, "communication_hint",
  communication_hint, "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveBcastRecv", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveBcastRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveBcastRecvV2_T = TypeVar("TV_CollectiveBcastRecvV2_T", _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_CollectiveBcastRecvV2_Tshape = TypeVar("TV_CollectiveBcastRecvV2_Tshape", _atypes.Int32, _atypes.Int64)

def collective_bcast_recv_v2(group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], shape: Annotated[Any, TV_CollectiveBcastRecvV2_Tshape], T: TV_CollectiveBcastRecvV2_T, communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveBcastRecvV2_T]:
  r"""Receives a tensor value broadcast from another device.

  Args:
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    T: A `tf.DType` from: `tf.bool, tf.float32, tf.half, tf.float64, tf.int32, tf.int64`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveBcastRecvV2", name, group_size, group_key,
        instance_key, shape, "T", T, "communication_hint", communication_hint,
        "timeout_seconds", timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_bcast_recv_v2_eager_fallback(
          group_size, group_key, instance_key, shape, T=T,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveBcastRecvV2", group_size=group_size, group_key=group_key,
                                 instance_key=instance_key, shape=shape, T=T,
                                 communication_hint=communication_hint,
                                 timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tshape",
              _op._get_attr_type("Tshape"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveBcastRecvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveBcastRecvV2 = tf_export("raw_ops.CollectiveBcastRecvV2")(_ops.to_raw_op(collective_bcast_recv_v2))


def collective_bcast_recv_v2_eager_fallback(group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], shape: Annotated[Any, TV_CollectiveBcastRecvV2_Tshape], T: TV_CollectiveBcastRecvV2_T, communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveBcastRecvV2_T]:
  T = _execute.make_type(T, "T")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  _inputs_flat = [group_size, group_key, instance_key, shape]
  _attrs = ("T", T, "Tshape", _attr_Tshape, "communication_hint",
  communication_hint, "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveBcastRecvV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveBcastRecvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveBcastSend_T = TypeVar("TV_CollectiveBcastSend_T", _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_bcast_send(input: Annotated[Any, TV_CollectiveBcastSend_T], group_size: int, group_key: int, instance_key: int, shape, communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveBcastSend_T]:
  r"""Broadcasts a tensor value to one or more other devices.

  Args:
    input: A `Tensor`. Must be one of the following types: `bool`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    shape: A `tf.TensorShape` or list of `ints`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveBcastSend", name, input, "group_size", group_size,
        "group_key", group_key, "instance_key", instance_key, "shape", shape,
        "communication_hint", communication_hint, "timeout_seconds",
        timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_bcast_send_eager_fallback(
          input, group_size=group_size, group_key=group_key,
          instance_key=instance_key, shape=shape,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveBcastSend", input=input, group_size=group_size,
                               group_key=group_key, instance_key=instance_key,
                               shape=shape,
                               communication_hint=communication_hint,
                               timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "group_size",
              _op._get_attr_int("group_size"), "group_key",
              _op._get_attr_int("group_key"), "instance_key",
              _op._get_attr_int("instance_key"), "shape",
              _op.get_attr("shape"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveBcastSend", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveBcastSend = tf_export("raw_ops.CollectiveBcastSend")(_ops.to_raw_op(collective_bcast_send))


def collective_bcast_send_eager_fallback(input: Annotated[Any, TV_CollectiveBcastSend_T], group_size: int, group_key: int, instance_key: int, shape, communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveBcastSend_T]:
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bool, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "group_size", group_size, "group_key", group_key,
  "instance_key", instance_key, "shape", shape, "communication_hint",
  communication_hint, "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveBcastSend", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveBcastSend", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveBcastSendV2_T = TypeVar("TV_CollectiveBcastSendV2_T", _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_bcast_send_v2(input: Annotated[Any, TV_CollectiveBcastSendV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveBcastSendV2_T]:
  r"""Broadcasts a tensor value to one or more other devices.

  Args:
    input: A `Tensor`. Must be one of the following types: `bool`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveBcastSendV2", name, input, group_size, group_key,
        instance_key, "communication_hint", communication_hint,
        "timeout_seconds", timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_bcast_send_v2_eager_fallback(
          input, group_size, group_key, instance_key,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveBcastSendV2", input=input, group_size=group_size,
                                 group_key=group_key,
                                 instance_key=instance_key,
                                 communication_hint=communication_hint,
                                 timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveBcastSendV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveBcastSendV2 = tf_export("raw_ops.CollectiveBcastSendV2")(_ops.to_raw_op(collective_bcast_send_v2))


def collective_bcast_send_v2_eager_fallback(input: Annotated[Any, TV_CollectiveBcastSendV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveBcastSendV2_T]:
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bool, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  _inputs_flat = [input, group_size, group_key, instance_key]
  _attrs = ("T", _attr_T, "communication_hint", communication_hint,
  "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveBcastSendV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveBcastSendV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveGather_T = TypeVar("TV_CollectiveGather_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_gather(input: Annotated[Any, TV_CollectiveGather_T], group_size: int, group_key: int, instance_key: int, shape, communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveGather_T]:
  r"""Mutually accumulates multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    shape: A `tf.TensorShape` or list of `ints`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveGather", name, input, "group_size", group_size,
        "group_key", group_key, "instance_key", instance_key, "shape", shape,
        "communication_hint", communication_hint, "timeout_seconds",
        timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_gather_eager_fallback(
          input, group_size=group_size, group_key=group_key,
          instance_key=instance_key, shape=shape,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveGather", input=input, group_size=group_size,
                            group_key=group_key, instance_key=instance_key,
                            shape=shape,
                            communication_hint=communication_hint,
                            timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "group_size",
              _op._get_attr_int("group_size"), "group_key",
              _op._get_attr_int("group_key"), "instance_key",
              _op._get_attr_int("instance_key"), "shape",
              _op.get_attr("shape"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveGather = tf_export("raw_ops.CollectiveGather")(_ops.to_raw_op(collective_gather))


def collective_gather_eager_fallback(input: Annotated[Any, TV_CollectiveGather_T], group_size: int, group_key: int, instance_key: int, shape, communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveGather_T]:
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  shape = _execute.make_shape(shape, "shape")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "group_size", group_size, "group_key", group_key,
  "instance_key", instance_key, "shape", shape, "communication_hint",
  communication_hint, "timeout_seconds", timeout_seconds)
  _result = _execute.execute(b"CollectiveGather", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveGatherV2_T = TypeVar("TV_CollectiveGatherV2_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_gather_v2(input: Annotated[Any, TV_CollectiveGatherV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], communication_hint:str="auto", timeout_seconds:float=0, is_stateless:bool=False, name=None) -> Annotated[Any, TV_CollectiveGatherV2_T]:
  r"""Mutually accumulates multiple tensors of identical type and shape.

  `is_stateless` means each op does not need control dependencies to other
  collective ops. In this case, keys that are unique at runtime
  (e.g. `instance_key`) should be used to distinguish collective groups.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    ordering_token: A list of `Tensor` objects with type `resource`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    is_stateless: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveGatherV2", name, input, group_size, group_key,
        instance_key, ordering_token, "communication_hint",
        communication_hint, "timeout_seconds", timeout_seconds,
        "is_stateless", is_stateless)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_gather_v2_eager_fallback(
          input, group_size, group_key, instance_key, ordering_token,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, is_stateless=is_stateless,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_gather_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveGatherV2", input=input, group_size=group_size,
                              group_key=group_key, instance_key=instance_key,
                              ordering_token=ordering_token,
                              communication_hint=communication_hint,
                              timeout_seconds=timeout_seconds,
                              is_stateless=is_stateless, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"), "is_stateless",
              _op._get_attr_bool("is_stateless"), "Nordering_token",
              _op._get_attr_int("Nordering_token"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveGatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveGatherV2 = tf_export("raw_ops.CollectiveGatherV2")(_ops.to_raw_op(collective_gather_v2))


def collective_gather_v2_eager_fallback(input: Annotated[Any, TV_CollectiveGatherV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], communication_hint: str, timeout_seconds: float, is_stateless: bool, name, ctx) -> Annotated[Any, TV_CollectiveGatherV2_T]:
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_gather_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  ordering_token = _ops.convert_n_to_tensor(ordering_token, _dtypes.resource)
  _inputs_flat = [input, group_size, group_key, instance_key] + list(ordering_token)
  _attrs = ("T", _attr_T, "communication_hint", communication_hint,
  "timeout_seconds", timeout_seconds, "is_stateless", is_stateless,
  "Nordering_token", _attr_Nordering_token)
  _result = _execute.execute(b"CollectiveGatherV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveGatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def collective_initialize_communicator(group_key: Annotated[Any, _atypes.Int32], rank: Annotated[Any, _atypes.Int32], group_size: Annotated[Any, _atypes.Int32], communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Initializes a group for collective operations.

  Args:
    group_key: A `Tensor` of type `int32`.
    rank: A `Tensor` of type `int32`.
    group_size: A `Tensor` of type `int32`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveInitializeCommunicator", name, group_key, rank,
        group_size, "communication_hint", communication_hint,
        "timeout_seconds", timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_initialize_communicator_eager_fallback(
          group_key, rank, group_size, communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveInitializeCommunicator", group_key=group_key, rank=rank,
                                            group_size=group_size,
                                            communication_hint=communication_hint,
                                            timeout_seconds=timeout_seconds,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("communication_hint", _op.get_attr("communication_hint"),
              "timeout_seconds", _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveInitializeCommunicator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveInitializeCommunicator = tf_export("raw_ops.CollectiveInitializeCommunicator")(_ops.to_raw_op(collective_initialize_communicator))


def collective_initialize_communicator_eager_fallback(group_key: Annotated[Any, _atypes.Int32], rank: Annotated[Any, _atypes.Int32], group_size: Annotated[Any, _atypes.Int32], communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  rank = _ops.convert_to_tensor(rank, _dtypes.int32)
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  _inputs_flat = [group_key, rank, group_size]
  _attrs = ("communication_hint", communication_hint, "timeout_seconds",
  timeout_seconds)
  _result = _execute.execute(b"CollectiveInitializeCommunicator", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveInitializeCommunicator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveReduce_T = TypeVar("TV_CollectiveReduce_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_reduce(input: Annotated[Any, TV_CollectiveReduce_T], group_size: int, group_key: int, instance_key: int, merge_op: str, final_op: str, subdiv_offsets, wait_for=[], communication_hint:str="auto", timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveReduce_T]:
  r"""Mutually reduces multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    subdiv_offsets: A list of `ints`.
    wait_for: An optional list of `ints`. Defaults to `[]`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveReduce", name, input, "group_size", group_size,
        "group_key", group_key, "instance_key", instance_key, "merge_op",
        merge_op, "final_op", final_op, "subdiv_offsets", subdiv_offsets,
        "wait_for", wait_for, "communication_hint", communication_hint,
        "timeout_seconds", timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_reduce_eager_fallback(
          input, group_size=group_size, group_key=group_key,
          instance_key=instance_key, merge_op=merge_op, final_op=final_op,
          subdiv_offsets=subdiv_offsets, wait_for=wait_for,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if not isinstance(subdiv_offsets, (list, tuple)):
    raise TypeError(
        "Expected list for 'subdiv_offsets' argument to "
        "'collective_reduce' Op, not %r." % subdiv_offsets)
  subdiv_offsets = [_execute.make_int(_i, "subdiv_offsets") for _i in subdiv_offsets]
  if wait_for is None:
    wait_for = []
  if not isinstance(wait_for, (list, tuple)):
    raise TypeError(
        "Expected list for 'wait_for' argument to "
        "'collective_reduce' Op, not %r." % wait_for)
  wait_for = [_execute.make_int(_i, "wait_for") for _i in wait_for]
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveReduce", input=input, group_size=group_size,
                            group_key=group_key, instance_key=instance_key,
                            merge_op=merge_op, final_op=final_op,
                            subdiv_offsets=subdiv_offsets, wait_for=wait_for,
                            communication_hint=communication_hint,
                            timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "group_size",
              _op._get_attr_int("group_size"), "group_key",
              _op._get_attr_int("group_key"), "instance_key",
              _op._get_attr_int("instance_key"), "merge_op",
              _op.get_attr("merge_op"), "final_op", _op.get_attr("final_op"),
              "subdiv_offsets", _op.get_attr("subdiv_offsets"), "wait_for",
              _op.get_attr("wait_for"), "communication_hint",
              _op.get_attr("communication_hint"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveReduce = tf_export("raw_ops.CollectiveReduce")(_ops.to_raw_op(collective_reduce))


def collective_reduce_eager_fallback(input: Annotated[Any, TV_CollectiveReduce_T], group_size: int, group_key: int, instance_key: int, merge_op: str, final_op: str, subdiv_offsets, wait_for, communication_hint: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveReduce_T]:
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if not isinstance(subdiv_offsets, (list, tuple)):
    raise TypeError(
        "Expected list for 'subdiv_offsets' argument to "
        "'collective_reduce' Op, not %r." % subdiv_offsets)
  subdiv_offsets = [_execute.make_int(_i, "subdiv_offsets") for _i in subdiv_offsets]
  if wait_for is None:
    wait_for = []
  if not isinstance(wait_for, (list, tuple)):
    raise TypeError(
        "Expected list for 'wait_for' argument to "
        "'collective_reduce' Op, not %r." % wait_for)
  wait_for = [_execute.make_int(_i, "wait_for") for _i in wait_for]
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "group_size", group_size, "group_key", group_key,
  "instance_key", instance_key, "merge_op", merge_op, "final_op", final_op,
  "subdiv_offsets", subdiv_offsets, "wait_for", wait_for,
  "communication_hint", communication_hint, "timeout_seconds",
  timeout_seconds)
  _result = _execute.execute(b"CollectiveReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveReduceScatterV2_T = TypeVar("TV_CollectiveReduceScatterV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_reduce_scatter_v2(input: Annotated[Any, TV_CollectiveReduceScatterV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], merge_op: str, final_op: str, communication_hint:str="auto", timeout_seconds:float=0, is_stateless:bool=False, max_subdivs_per_device:int=-1, name=None) -> Annotated[Any, TV_CollectiveReduceScatterV2_T]:
  r"""Mutually reduces multiple tensors of identical type and shape and scatters the result.

  `is_stateless` means each op does not need control dependencies to other
  collective ops. In this case, keys that are unique at runtime
  (e.g. `instance_key`) should be used to distinguish collective groups.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    ordering_token: A list of `Tensor` objects with type `resource`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    is_stateless: An optional `bool`. Defaults to `False`.
    max_subdivs_per_device: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveReduceScatterV2", name, input, group_size, group_key,
        instance_key, ordering_token, "merge_op", merge_op, "final_op",
        final_op, "communication_hint", communication_hint, "timeout_seconds",
        timeout_seconds, "is_stateless", is_stateless,
        "max_subdivs_per_device", max_subdivs_per_device)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_reduce_scatter_v2_eager_fallback(
          input, group_size, group_key, instance_key, ordering_token,
          merge_op=merge_op, final_op=final_op,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, is_stateless=is_stateless,
          max_subdivs_per_device=max_subdivs_per_device, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_reduce_scatter_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  if max_subdivs_per_device is None:
    max_subdivs_per_device = -1
  max_subdivs_per_device = _execute.make_int(max_subdivs_per_device, "max_subdivs_per_device")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveReduceScatterV2", input=input, group_size=group_size,
                                     group_key=group_key,
                                     instance_key=instance_key,
                                     ordering_token=ordering_token,
                                     merge_op=merge_op, final_op=final_op,
                                     communication_hint=communication_hint,
                                     timeout_seconds=timeout_seconds,
                                     is_stateless=is_stateless,
                                     max_subdivs_per_device=max_subdivs_per_device,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "merge_op",
              _op.get_attr("merge_op"), "final_op", _op.get_attr("final_op"),
              "communication_hint", _op.get_attr("communication_hint"),
              "timeout_seconds", _op.get_attr("timeout_seconds"),
              "is_stateless", _op._get_attr_bool("is_stateless"),
              "Nordering_token", _op._get_attr_int("Nordering_token"),
              "max_subdivs_per_device",
              _op._get_attr_int("max_subdivs_per_device"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveReduceScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveReduceScatterV2 = tf_export("raw_ops.CollectiveReduceScatterV2")(_ops.to_raw_op(collective_reduce_scatter_v2))


def collective_reduce_scatter_v2_eager_fallback(input: Annotated[Any, TV_CollectiveReduceScatterV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], merge_op: str, final_op: str, communication_hint: str, timeout_seconds: float, is_stateless: bool, max_subdivs_per_device: int, name, ctx) -> Annotated[Any, TV_CollectiveReduceScatterV2_T]:
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_reduce_scatter_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  if max_subdivs_per_device is None:
    max_subdivs_per_device = -1
  max_subdivs_per_device = _execute.make_int(max_subdivs_per_device, "max_subdivs_per_device")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  ordering_token = _ops.convert_n_to_tensor(ordering_token, _dtypes.resource)
  _inputs_flat = [input, group_size, group_key, instance_key] + list(ordering_token)
  _attrs = ("T", _attr_T, "merge_op", merge_op, "final_op", final_op,
  "communication_hint", communication_hint, "timeout_seconds",
  timeout_seconds, "is_stateless", is_stateless, "Nordering_token",
  _attr_Nordering_token, "max_subdivs_per_device", max_subdivs_per_device)
  _result = _execute.execute(b"CollectiveReduceScatterV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveReduceScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveReduceV2_T = TypeVar("TV_CollectiveReduceV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_reduce_v2(input: Annotated[Any, TV_CollectiveReduceV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], merge_op: str, final_op: str, communication_hint:str="auto", timeout_seconds:float=0, is_stateless:bool=False, max_subdivs_per_device:int=-1, name=None) -> Annotated[Any, TV_CollectiveReduceV2_T]:
  r"""Mutually reduces multiple tensors of identical type and shape.

  `is_stateless` means each op does not need control dependencies to other
  collective ops. In this case, keys that are unique at runtime
  (e.g. `instance_key`) should be used to distinguish collective groups.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    ordering_token: A list of `Tensor` objects with type `resource`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    is_stateless: An optional `bool`. Defaults to `False`.
    max_subdivs_per_device: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveReduceV2", name, input, group_size, group_key,
        instance_key, ordering_token, "merge_op", merge_op, "final_op",
        final_op, "communication_hint", communication_hint, "timeout_seconds",
        timeout_seconds, "is_stateless", is_stateless,
        "max_subdivs_per_device", max_subdivs_per_device)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_reduce_v2_eager_fallback(
          input, group_size, group_key, instance_key, ordering_token,
          merge_op=merge_op, final_op=final_op,
          communication_hint=communication_hint,
          timeout_seconds=timeout_seconds, is_stateless=is_stateless,
          max_subdivs_per_device=max_subdivs_per_device, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_reduce_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  if max_subdivs_per_device is None:
    max_subdivs_per_device = -1
  max_subdivs_per_device = _execute.make_int(max_subdivs_per_device, "max_subdivs_per_device")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveReduceV2", input=input, group_size=group_size,
                              group_key=group_key, instance_key=instance_key,
                              ordering_token=ordering_token,
                              merge_op=merge_op, final_op=final_op,
                              communication_hint=communication_hint,
                              timeout_seconds=timeout_seconds,
                              is_stateless=is_stateless,
                              max_subdivs_per_device=max_subdivs_per_device,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "merge_op",
              _op.get_attr("merge_op"), "final_op", _op.get_attr("final_op"),
              "communication_hint", _op.get_attr("communication_hint"),
              "timeout_seconds", _op.get_attr("timeout_seconds"),
              "is_stateless", _op._get_attr_bool("is_stateless"),
              "Nordering_token", _op._get_attr_int("Nordering_token"),
              "max_subdivs_per_device",
              _op._get_attr_int("max_subdivs_per_device"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveReduceV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveReduceV2 = tf_export("raw_ops.CollectiveReduceV2")(_ops.to_raw_op(collective_reduce_v2))


def collective_reduce_v2_eager_fallback(input: Annotated[Any, TV_CollectiveReduceV2_T], group_size: Annotated[Any, _atypes.Int32], group_key: Annotated[Any, _atypes.Int32], instance_key: Annotated[Any, _atypes.Int32], ordering_token: Annotated[List[Any], _atypes.Resource], merge_op: str, final_op: str, communication_hint: str, timeout_seconds: float, is_stateless: bool, max_subdivs_per_device: int, name, ctx) -> Annotated[Any, TV_CollectiveReduceV2_T]:
  if not isinstance(ordering_token, (list, tuple)):
    raise TypeError(
        "Expected list for 'ordering_token' argument to "
        "'collective_reduce_v2' Op, not %r." % ordering_token)
  _attr_Nordering_token = len(ordering_token)
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  if is_stateless is None:
    is_stateless = False
  is_stateless = _execute.make_bool(is_stateless, "is_stateless")
  if max_subdivs_per_device is None:
    max_subdivs_per_device = -1
  max_subdivs_per_device = _execute.make_int(max_subdivs_per_device, "max_subdivs_per_device")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
  group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
  instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
  ordering_token = _ops.convert_n_to_tensor(ordering_token, _dtypes.resource)
  _inputs_flat = [input, group_size, group_key, instance_key] + list(ordering_token)
  _attrs = ("T", _attr_T, "merge_op", merge_op, "final_op", final_op,
  "communication_hint", communication_hint, "timeout_seconds",
  timeout_seconds, "is_stateless", is_stateless, "Nordering_token",
  _attr_Nordering_token, "max_subdivs_per_device", max_subdivs_per_device)
  _result = _execute.execute(b"CollectiveReduceV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveReduceV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CollectiveReduceV3_T = TypeVar("TV_CollectiveReduceV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def collective_reduce_v3(input: Annotated[Any, TV_CollectiveReduceV3_T], communicator: Annotated[Any, _atypes.Resource], group_assignment: Annotated[Any, _atypes.Int32], reduction: str, timeout_seconds:float=0, name=None) -> Annotated[Any, TV_CollectiveReduceV3_T]:
  r"""Mutually reduces multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    communicator: A `Tensor` of type `resource`.
    group_assignment: A `Tensor` of type `int32`.
    reduction: A `string` from: `"Min", "Max", "Mul", "Add"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectiveReduceV3", name, input, communicator,
        group_assignment, "reduction", reduction, "timeout_seconds",
        timeout_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_reduce_v3_eager_fallback(
          input, communicator, group_assignment, reduction=reduction,
          timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  reduction = _execute.make_str(reduction, "reduction")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectiveReduceV3", input=input, communicator=communicator,
                              group_assignment=group_assignment,
                              reduction=reduction,
                              timeout_seconds=timeout_seconds, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "reduction",
              _op.get_attr("reduction"), "timeout_seconds",
              _op.get_attr("timeout_seconds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveReduceV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectiveReduceV3 = tf_export("raw_ops.CollectiveReduceV3")(_ops.to_raw_op(collective_reduce_v3))


def collective_reduce_v3_eager_fallback(input: Annotated[Any, TV_CollectiveReduceV3_T], communicator: Annotated[Any, _atypes.Resource], group_assignment: Annotated[Any, _atypes.Int32], reduction: str, timeout_seconds: float, name, ctx) -> Annotated[Any, TV_CollectiveReduceV3_T]:
  reduction = _execute.make_str(reduction, "reduction")
  if timeout_seconds is None:
    timeout_seconds = 0
  timeout_seconds = _execute.make_float(timeout_seconds, "timeout_seconds")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  communicator = _ops.convert_to_tensor(communicator, _dtypes.resource)
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, communicator, group_assignment]
  _attrs = ("T", _attr_T, "reduction", reduction, "timeout_seconds",
  timeout_seconds)
  _result = _execute.execute(b"CollectiveReduceV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectiveReduceV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

