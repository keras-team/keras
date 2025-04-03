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

TV_AccumulatorApplyGradient_dtype = TypeVar("TV_AccumulatorApplyGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def accumulator_apply_gradient(handle: Annotated[Any, _atypes.String], local_step: Annotated[Any, _atypes.Int64], gradient: Annotated[Any, TV_AccumulatorApplyGradient_dtype], name=None):
  r"""Applies a gradient to a given accumulator.

  Does not add if local_step is lesser than the accumulator's global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the gradient was computed.
    gradient: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A tensor of the gradient to be accumulated.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("accumulator_apply_gradient op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AccumulatorApplyGradient", handle=handle, local_step=local_step,
                                    gradient=gradient, name=name)
  return _op
AccumulatorApplyGradient = tf_export("raw_ops.AccumulatorApplyGradient")(_ops.to_raw_op(accumulator_apply_gradient))


def accumulator_apply_gradient_eager_fallback(handle: Annotated[Any, _atypes.String], local_step: Annotated[Any, _atypes.Int64], gradient: Annotated[Any, TV_AccumulatorApplyGradient_dtype], name, ctx):
  raise RuntimeError("accumulator_apply_gradient op does not support eager execution. Arg 'handle' is a ref.")

def accumulator_num_accumulated(handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the number of gradients aggregated in the given accumulators.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("accumulator_num_accumulated op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AccumulatorNumAccumulated", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AccumulatorNumAccumulated", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AccumulatorNumAccumulated = tf_export("raw_ops.AccumulatorNumAccumulated")(_ops.to_raw_op(accumulator_num_accumulated))


def accumulator_num_accumulated_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("accumulator_num_accumulated op does not support eager execution. Arg 'handle' is a ref.")

def accumulator_set_global_step(handle: Annotated[Any, _atypes.String], new_global_step: Annotated[Any, _atypes.Int64], name=None):
  r"""Updates the accumulator with a new value for global_step.

  Logs warning if the accumulator's value is already higher than
  new_global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    new_global_step: A `Tensor` of type `int64`.
      The new global_step value to set.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("accumulator_set_global_step op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AccumulatorSetGlobalStep", handle=handle,
                                    new_global_step=new_global_step,
                                    name=name)
  return _op
AccumulatorSetGlobalStep = tf_export("raw_ops.AccumulatorSetGlobalStep")(_ops.to_raw_op(accumulator_set_global_step))


def accumulator_set_global_step_eager_fallback(handle: Annotated[Any, _atypes.String], new_global_step: Annotated[Any, _atypes.Int64], name, ctx):
  raise RuntimeError("accumulator_set_global_step op does not support eager execution. Arg 'handle' is a ref.")

TV_AccumulatorTakeGradient_dtype = TypeVar("TV_AccumulatorTakeGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def accumulator_take_gradient(handle: Annotated[Any, _atypes.String], num_required: Annotated[Any, _atypes.Int32], dtype: TV_AccumulatorTakeGradient_dtype, name=None) -> Annotated[Any, TV_AccumulatorTakeGradient_dtype]:
  r"""Extracts the average gradient in the given ConditionalAccumulator.

  The op blocks until sufficient (i.e., more than num_required)
  gradients have been accumulated.  If the accumulator has already
  aggregated more than num_required gradients, it returns the average of
  the accumulated gradients.  Also automatically increments the recorded
  global_step in the accumulator by 1, and resets the aggregate to 0.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("accumulator_take_gradient op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AccumulatorTakeGradient", handle=handle, num_required=num_required,
                                   dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AccumulatorTakeGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AccumulatorTakeGradient = tf_export("raw_ops.AccumulatorTakeGradient")(_ops.to_raw_op(accumulator_take_gradient))


def accumulator_take_gradient_eager_fallback(handle: Annotated[Any, _atypes.String], num_required: Annotated[Any, _atypes.Int32], dtype: TV_AccumulatorTakeGradient_dtype, name, ctx) -> Annotated[Any, TV_AccumulatorTakeGradient_dtype]:
  raise RuntimeError("accumulator_take_gradient op does not support eager execution. Arg 'handle' is a ref.")

def barrier(component_types, shapes=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Defines a barrier that persists across different graph executions.

  A barrier represents a key-value map, where each key is a string, and
  each value is a tuple of tensors.

  At runtime, the barrier contains 'complete' and 'incomplete'
  elements. A complete element has defined tensors for all components of
  its value tuple, and may be accessed using BarrierTakeMany. An
  incomplete element has some undefined components in its value tuple,
  and may be updated using BarrierInsertMany.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. Each shape must be 1 in the
      first dimension. The length of this attr must be the same as the length of
      component_types.
    capacity: An optional `int`. Defaults to `-1`.
      The capacity of the barrier.  The default capacity is MAX_INT32,
      which is the largest capacity of the underlying queue.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this barrier is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this barrier will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'barrier' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'barrier' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Barrier", component_types=component_types, shapes=shapes,
                   capacity=capacity, container=container,
                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Barrier", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Barrier = tf_export("raw_ops.Barrier")(_ops.to_raw_op(barrier))


def barrier_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("barrier op does not support eager execution. Arg 'handle' is a ref.")

def barrier_close(handle: Annotated[Any, _atypes.String], cancel_pending_enqueues:bool=False, name=None):
  r"""Closes the given barrier.

  This operation signals that no more new elements will be inserted in the
  given barrier. Subsequent InsertMany that try to introduce a new key will fail.
  Subsequent InsertMany operations that just add missing components to already
  existing elements will continue to succeed. Subsequent TakeMany operations will
  continue to succeed if sufficient completed elements remain in the barrier.
  Subsequent TakeMany operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the barrier's queue will be canceled. InsertMany will fail, even
      if no new key is introduced.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier_close op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if cancel_pending_enqueues is None:
    cancel_pending_enqueues = False
  cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, "cancel_pending_enqueues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BarrierClose", handle=handle,
                        cancel_pending_enqueues=cancel_pending_enqueues,
                        name=name)
  return _op
BarrierClose = tf_export("raw_ops.BarrierClose")(_ops.to_raw_op(barrier_close))


def barrier_close_eager_fallback(handle: Annotated[Any, _atypes.String], cancel_pending_enqueues: bool, name, ctx):
  raise RuntimeError("barrier_close op does not support eager execution. Arg 'handle' is a ref.")

def barrier_incomplete_size(handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Computes the number of incomplete elements in the given barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier_incomplete_size op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BarrierIncompleteSize", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BarrierIncompleteSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BarrierIncompleteSize = tf_export("raw_ops.BarrierIncompleteSize")(_ops.to_raw_op(barrier_incomplete_size))


def barrier_incomplete_size_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("barrier_incomplete_size op does not support eager execution. Arg 'handle' is a ref.")

TV_BarrierInsertMany_T = TypeVar("TV_BarrierInsertMany_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def barrier_insert_many(handle: Annotated[Any, _atypes.String], keys: Annotated[Any, _atypes.String], values: Annotated[Any, TV_BarrierInsertMany_T], component_index: int, name=None):
  r"""For each key, assigns the respective value to the specified component.

  If a key is not found in the barrier, this operation will create a new
  incomplete element. If a key is found in the barrier, and the element
  already has a value at component_index, this operation will fail with
  INVALID_ARGUMENT, and leave the barrier in an undefined state.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    keys: A `Tensor` of type `string`.
      A one-dimensional tensor of keys, with length n.
    values: A `Tensor`.
      An any-dimensional tensor of values, which are associated with the
      respective keys. The 0th dimension must have length n.
    component_index: An `int`.
      The component of the barrier elements that is being assigned.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier_insert_many op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  component_index = _execute.make_int(component_index, "component_index")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BarrierInsertMany", handle=handle, keys=keys, values=values,
                             component_index=component_index, name=name)
  return _op
BarrierInsertMany = tf_export("raw_ops.BarrierInsertMany")(_ops.to_raw_op(barrier_insert_many))


def barrier_insert_many_eager_fallback(handle: Annotated[Any, _atypes.String], keys: Annotated[Any, _atypes.String], values: Annotated[Any, TV_BarrierInsertMany_T], component_index: int, name, ctx):
  raise RuntimeError("barrier_insert_many op does not support eager execution. Arg 'handle' is a ref.")

def barrier_ready_size(handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Computes the number of complete elements in the given barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier_ready_size op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BarrierReadySize", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BarrierReadySize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BarrierReadySize = tf_export("raw_ops.BarrierReadySize")(_ops.to_raw_op(barrier_ready_size))


def barrier_ready_size_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("barrier_ready_size op does not support eager execution. Arg 'handle' is a ref.")
_BarrierTakeManyOutput = collections.namedtuple(
    "BarrierTakeMany",
    ["indices", "keys", "values"])


def barrier_take_many(handle: Annotated[Any, _atypes.String], num_elements: Annotated[Any, _atypes.Int32], component_types, allow_small_batch:bool=False, wait_for_incomplete:bool=False, timeout_ms:int=-1, name=None):
  r"""Takes the given number of completed elements from a barrier.

  This operation concatenates completed-element component tensors along
  the 0th dimension to make a single component tensor.

  Elements come out of the barrier when they are complete, and in the order
  in which they were placed into the barrier.  The indices output provides
  information about the batch in which each element was originally inserted
  into the barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    num_elements: A `Tensor` of type `int32`.
      A single-element tensor containing the number of elements to
      take.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    allow_small_batch: An optional `bool`. Defaults to `False`.
      Allow to return less than num_elements items if barrier is
      already closed.
    wait_for_incomplete: An optional `bool`. Defaults to `False`.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, keys, values).

    indices: A `Tensor` of type `int64`.
    keys: A `Tensor` of type `string`.
    values: A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("barrier_take_many op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'barrier_take_many' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if allow_small_batch is None:
    allow_small_batch = False
  allow_small_batch = _execute.make_bool(allow_small_batch, "allow_small_batch")
  if wait_for_incomplete is None:
    wait_for_incomplete = False
  wait_for_incomplete = _execute.make_bool(wait_for_incomplete, "wait_for_incomplete")
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BarrierTakeMany", handle=handle, num_elements=num_elements,
                           component_types=component_types,
                           allow_small_batch=allow_small_batch,
                           wait_for_incomplete=wait_for_incomplete,
                           timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "allow_small_batch", _op._get_attr_bool("allow_small_batch"),
              "wait_for_incomplete",
              _op._get_attr_bool("wait_for_incomplete"), "timeout_ms",
              _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BarrierTakeMany", _inputs_flat, _attrs, _result)
  _result = _result[:2] + [_result[2:]]
  _result = _BarrierTakeManyOutput._make(_result)
  return _result

BarrierTakeMany = tf_export("raw_ops.BarrierTakeMany")(_ops.to_raw_op(barrier_take_many))


def barrier_take_many_eager_fallback(handle: Annotated[Any, _atypes.String], num_elements: Annotated[Any, _atypes.Int32], component_types, allow_small_batch: bool, wait_for_incomplete: bool, timeout_ms: int, name, ctx):
  raise RuntimeError("barrier_take_many op does not support eager execution. Arg 'handle' is a ref.")

TV_ConditionalAccumulator_dtype = TypeVar("TV_ConditionalAccumulator_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def conditional_accumulator(dtype: TV_ConditionalAccumulator_dtype, shape, container:str="", shared_name:str="", reduction_type:str="MEAN", name=None) -> Annotated[Any, _atypes.String]:
  r"""A conditional accumulator for aggregating gradients.

  The accumulator accepts gradients marked with local_step greater or
  equal to the most recent global_step known to the accumulator. The
  average can be extracted from the accumulator, provided sufficient
  gradients have been accumulated. Extracting the average automatically
  resets the aggregate to 0, and increments the global_step recorded by
  the accumulator.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`.
      The shape of the values, can be [], in which case shape is unknown.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the
      given name across multiple sessions.
    reduction_type: An optional `string` from: `"MEAN", "SUM"`. Defaults to `"MEAN"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("conditional_accumulator op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if reduction_type is None:
    reduction_type = "MEAN"
  reduction_type = _execute.make_str(reduction_type, "reduction_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConditionalAccumulator", dtype=dtype, shape=shape,
                                  container=container,
                                  shared_name=shared_name,
                                  reduction_type=reduction_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"), "reduction_type",
              _op.get_attr("reduction_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConditionalAccumulator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConditionalAccumulator = tf_export("raw_ops.ConditionalAccumulator")(_ops.to_raw_op(conditional_accumulator))


def conditional_accumulator_eager_fallback(dtype: TV_ConditionalAccumulator_dtype, shape, container: str, shared_name: str, reduction_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("conditional_accumulator op does not support eager execution. Arg 'handle' is a ref.")

def delete_session_tensor(handle: Annotated[Any, _atypes.String], name=None):
  r"""Delete the tensor specified by its handle in the session.

  Args:
    handle: A `Tensor` of type `string`.
      The handle for a tensor stored in the session state.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteSessionTensor", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_session_tensor_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteSessionTensor", handle=handle, name=name)
  return _op
DeleteSessionTensor = tf_export("raw_ops.DeleteSessionTensor")(_ops.to_raw_op(delete_session_tensor))


def delete_session_tensor_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"DeleteSessionTensor", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_DynamicPartition_T = TypeVar("TV_DynamicPartition_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('dynamic_partition')
def dynamic_partition(data: Annotated[Any, TV_DynamicPartition_T], partitions: Annotated[Any, _atypes.Int32], num_partitions: int, name=None):
  r"""Partitions `data` into `num_partitions` tensors using indices from `partitions`.

  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
  are placed in `outputs[i]` in lexicographic order of `js`, and the first
  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
  In detail,

  ```python
      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
  ```

  `data.shape` must start with `partitions.shape`.

  For example:

  ```python
      # Scalar partitions.
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions.
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]
  ```

  See `dynamic_stitch` for an example on how to merge partitions back.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
  </div>


  Raises:
    * `InvalidArgumentError` in following cases:
      - If partitions is not in range `[0, num_partiions)`
      - If `partitions.shape` does not match prefix of `data.shape` argument.

  Args:
    data: A `Tensor`.
    partitions: A `Tensor` of type `int32`.
      Any shape.  Indices in the range `[0, num_partitions)`.
    num_partitions: An `int` that is `>= 1`.
      The number of partitions to output.
    name: A name for the operation (optional).

  Returns:
    A list of `num_partitions` `Tensor` objects with the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DynamicPartition", name, data, partitions, "num_partitions",
        num_partitions)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_dynamic_partition(
          (data, partitions, num_partitions, name,), None)
      if _result is not NotImplemented:
        return _result
      return dynamic_partition_eager_fallback(
          data, partitions, num_partitions=num_partitions, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            dynamic_partition, (), dict(data=data, partitions=partitions,
                                        num_partitions=num_partitions,
                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_dynamic_partition(
        (data, partitions, num_partitions, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  num_partitions = _execute.make_int(num_partitions, "num_partitions")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DynamicPartition", data=data, partitions=partitions,
                            num_partitions=num_partitions, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          dynamic_partition, (), dict(data=data, partitions=partitions,
                                      num_partitions=num_partitions,
                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_partitions", _op._get_attr_int("num_partitions"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DynamicPartition", _inputs_flat, _attrs, _result)
  return _result

DynamicPartition = tf_export("raw_ops.DynamicPartition")(_ops.to_raw_op(dynamic_partition))
_dispatcher_for_dynamic_partition = dynamic_partition._tf_type_based_dispatcher.Dispatch


def dynamic_partition_eager_fallback(data: Annotated[Any, TV_DynamicPartition_T], partitions: Annotated[Any, _atypes.Int32], num_partitions: int, name, ctx):
  num_partitions = _execute.make_int(num_partitions, "num_partitions")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [])
  partitions = _ops.convert_to_tensor(partitions, _dtypes.int32)
  _inputs_flat = [data, partitions]
  _attrs = ("num_partitions", num_partitions, "T", _attr_T)
  _result = _execute.execute(b"DynamicPartition", num_partitions,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DynamicPartition", _inputs_flat, _attrs, _result)
  return _result


TV_DynamicStitch_T = TypeVar("TV_DynamicStitch_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('dynamic_stitch')
def dynamic_stitch(indices: Annotated[List[Any], _atypes.Int32], data: Annotated[List[Any], TV_DynamicStitch_T], name=None) -> Annotated[Any, TV_DynamicStitch_T]:
  r"""Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

  ```python
      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
  ```

  For example, if each `indices[m]` is scalar or vector, we have

  ```python
      # Scalar indices:
      merged[indices[m], ...] = data[m][...]

      # Vector indices:
      merged[indices[m][i], ...] = data[m][i, ...]
  ```

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices) + 1] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result. If you do not need this guarantee, ParallelDynamicStitch might
  perform better on some devices.

  For example:

  ```python
      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]
  ```

  This method can be used to merge partitions created by `dynamic_partition`
  as illustrated on the following example:

  ```python
      # Apply function (increments x_i) on elements for which a certain condition
      # apply (x_i != -1 in this example).
      x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
      condition_mask=tf.not_equal(x,tf.constant(-1.))
      partitioned_data = tf.dynamic_partition(
          x, tf.cast(condition_mask, tf.int32) , 2)
      partitioned_data[1] = partitioned_data[1] + 1.0
      condition_indices = tf.dynamic_partition(
          tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
      x = tf.dynamic_stitch(condition_indices, partitioned_data)
      # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
      # unchanged.
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 1 `Tensor` objects with type `int32`.
    data: A list with the same length as `indices` of `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DynamicStitch", name, indices, data)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_dynamic_stitch(
          (indices, data, name,), None)
      if _result is not NotImplemented:
        return _result
      return dynamic_stitch_eager_fallback(
          indices, data, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            dynamic_stitch, (), dict(indices=indices, data=data, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_dynamic_stitch(
        (indices, data, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'dynamic_stitch' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(data, (list, tuple)):
    raise TypeError(
        "Expected list for 'data' argument to "
        "'dynamic_stitch' Op, not %r." % data)
  if len(data) != _attr_N:
    raise ValueError(
        "List argument 'data' to 'dynamic_stitch' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(data), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DynamicStitch", indices=indices, data=data, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          dynamic_stitch, (), dict(indices=indices, data=data, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DynamicStitch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DynamicStitch = tf_export("raw_ops.DynamicStitch")(_ops.to_raw_op(dynamic_stitch))
_dispatcher_for_dynamic_stitch = dynamic_stitch._tf_type_based_dispatcher.Dispatch


def dynamic_stitch_eager_fallback(indices: Annotated[List[Any], _atypes.Int32], data: Annotated[List[Any], TV_DynamicStitch_T], name, ctx) -> Annotated[Any, TV_DynamicStitch_T]:
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'dynamic_stitch' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(data, (list, tuple)):
    raise TypeError(
        "Expected list for 'data' argument to "
        "'dynamic_stitch' Op, not %r." % data)
  if len(data) != _attr_N:
    raise ValueError(
        "List argument 'data' to 'dynamic_stitch' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(data), _attr_N))
  _attr_T, data = _execute.args_to_matching_eager(list(data), ctx, [])
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int32)
  _inputs_flat = list(indices) + list(data)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"DynamicStitch", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DynamicStitch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fifo_queue(component_types, shapes=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A queue that produces elements in first-in first-out order.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("fifo_queue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'fifo_queue' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'fifo_queue' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FIFOQueue", component_types=component_types, shapes=shapes,
                     capacity=capacity, container=container,
                     shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FIFOQueue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FIFOQueue = tf_export("raw_ops.FIFOQueue")(_ops.to_raw_op(fifo_queue))


def fifo_queue_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("fifo_queue op does not support eager execution. Arg 'handle' is a ref.")

def fifo_queue_v2(component_types, shapes=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A queue that produces elements in first-in first-out order.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FIFOQueueV2", name, "component_types", component_types,
        "shapes", shapes, "capacity", capacity, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fifo_queue_v2_eager_fallback(
          component_types=component_types, shapes=shapes, capacity=capacity,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'fifo_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'fifo_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FIFOQueueV2", component_types=component_types, shapes=shapes,
                       capacity=capacity, container=container,
                       shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FIFOQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FIFOQueueV2 = tf_export("raw_ops.FIFOQueueV2")(_ops.to_raw_op(fifo_queue_v2))


def fifo_queue_v2_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'fifo_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'fifo_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("component_types", component_types, "shapes", shapes, "capacity",
  capacity, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"FIFOQueueV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FIFOQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fake_queue(resource: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Deprecated. Do not use.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("fake_queue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeQueue", resource=resource, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeQueue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeQueue = tf_export("raw_ops.FakeQueue")(_ops.to_raw_op(fake_queue))


def fake_queue_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("fake_queue op does not support eager execution. Arg 'handle' is a ref.")

TV_GetSessionHandle_T = TypeVar("TV_GetSessionHandle_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def get_session_handle(value: Annotated[Any, TV_GetSessionHandle_T], name=None) -> Annotated[Any, _atypes.String]:
  r"""Store the input tensor in the state of the current session.

  Args:
    value: A `Tensor`. The tensor to be stored.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetSessionHandle", name, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_session_handle_eager_fallback(
          value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetSessionHandle", value=value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetSessionHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetSessionHandle = tf_export("raw_ops.GetSessionHandle")(_ops.to_raw_op(get_session_handle))


def get_session_handle_eager_fallback(value: Annotated[Any, TV_GetSessionHandle_T], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _inputs_flat = [value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GetSessionHandle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetSessionHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GetSessionHandleV2_T = TypeVar("TV_GetSessionHandleV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def get_session_handle_v2(value: Annotated[Any, TV_GetSessionHandleV2_T], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Store the input tensor in the state of the current session.

  Args:
    value: A `Tensor`. The tensor to be stored.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetSessionHandleV2", name, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_session_handle_v2_eager_fallback(
          value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetSessionHandleV2", value=value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetSessionHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetSessionHandleV2 = tf_export("raw_ops.GetSessionHandleV2")(_ops.to_raw_op(get_session_handle_v2))


def get_session_handle_v2_eager_fallback(value: Annotated[Any, TV_GetSessionHandleV2_T], name, ctx) -> Annotated[Any, _atypes.Resource]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  _inputs_flat = [value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GetSessionHandleV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetSessionHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GetSessionTensor_dtype = TypeVar("TV_GetSessionTensor_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def get_session_tensor(handle: Annotated[Any, _atypes.String], dtype: TV_GetSessionTensor_dtype, name=None) -> Annotated[Any, TV_GetSessionTensor_dtype]:
  r"""Get the value of the tensor specified by its handle.

  Args:
    handle: A `Tensor` of type `string`.
      The handle for a tensor stored in the session state.
    dtype: A `tf.DType`. The type of the output value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetSessionTensor", name, handle, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_session_tensor_eager_fallback(
          handle, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetSessionTensor", handle=handle, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetSessionTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetSessionTensor = tf_export("raw_ops.GetSessionTensor")(_ops.to_raw_op(get_session_tensor))


def get_session_tensor_eager_fallback(handle: Annotated[Any, _atypes.String], dtype: TV_GetSessionTensor_dtype, name, ctx) -> Annotated[Any, TV_GetSessionTensor_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  _inputs_flat = [handle]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"GetSessionTensor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetSessionTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def map_clear(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes all elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapClear", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_clear_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapClear", dtypes=dtypes, capacity=capacity,
                    memory_limit=memory_limit, container=container,
                    shared_name=shared_name, name=name)
  return _op
MapClear = tf_export("raw_ops.MapClear")(_ops.to_raw_op(map_clear))


def map_clear_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapClear", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def map_incomplete_size(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Op returns the number of incomplete elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapIncompleteSize", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_incomplete_size_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_incomplete_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapIncompleteSize", dtypes=dtypes, capacity=capacity,
                             memory_limit=memory_limit, container=container,
                             shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapIncompleteSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MapIncompleteSize = tf_export("raw_ops.MapIncompleteSize")(_ops.to_raw_op(map_incomplete_size))


def map_incomplete_size_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_incomplete_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapIncompleteSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapIncompleteSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def map_peek(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op peeks at the values at the specified key.  If the

  underlying container does not contain this key
  this op will block until it does.

  Args:
    key: A `Tensor` of type `int64`.
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapPeek", name, key, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_peek_eager_fallback(
          key, indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapPeek", key=key, indices=indices, dtypes=dtypes, capacity=capacity,
                   memory_limit=memory_limit, container=container,
                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapPeek", _inputs_flat, _attrs, _result)
  return _result

MapPeek = tf_export("raw_ops.MapPeek")(_ops.to_raw_op(map_peek))


def map_peek_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapPeek", len(dtypes), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapPeek", _inputs_flat, _attrs, _result)
  return _result


def map_size(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Op returns the number of elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapSize", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_size_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapSize", dtypes=dtypes, capacity=capacity,
                   memory_limit=memory_limit, container=container,
                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MapSize = tf_export("raw_ops.MapSize")(_ops.to_raw_op(map_size))


def map_size_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapSize", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def map_stage(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], values, dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Stage (key, values) in the underlying container which behaves like a hashtable.

  Args:
    key: A `Tensor` of type `int64`. int64
    indices: A `Tensor` of type `int32`.
    values: A list of `Tensor` objects. a list of tensors
      dtypes A list of data types that inserted values should adhere to.
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
      Maximum number of elements in the Staging Area. If > 0, inserts
      on the container will block when the capacity is reached.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container. Otherwise,
      a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      It is necessary to match this name to the matching Unstage Op.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapStage", name, key, indices, values, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_stage_eager_fallback(
          key, indices, values, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_stage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapStage", key=key, indices=indices, values=values, dtypes=dtypes,
                    capacity=capacity, memory_limit=memory_limit,
                    container=container, shared_name=shared_name, name=name)
  return _op
MapStage = tf_export("raw_ops.MapStage")(_ops.to_raw_op(map_stage))


def map_stage_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], values, dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_stage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_fake_dtypes, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices] + list(values)
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "fake_dtypes", _attr_fake_dtypes, "container", container,
  "shared_name", shared_name)
  _result = _execute.execute(b"MapStage", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def map_unstage(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes and returns the values associated with the key

  from the underlying container.   If the underlying container
  does not contain this key, the op will block until it does.

  Args:
    key: A `Tensor` of type `int64`.
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapUnstage", name, key, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_unstage_eager_fallback(
          key, indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapUnstage", key=key, indices=indices, dtypes=dtypes,
                      capacity=capacity, memory_limit=memory_limit,
                      container=container, shared_name=shared_name, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapUnstage", _inputs_flat, _attrs, _result)
  return _result

MapUnstage = tf_export("raw_ops.MapUnstage")(_ops.to_raw_op(map_unstage))


def map_unstage_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapUnstage", len(dtypes), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapUnstage", _inputs_flat, _attrs, _result)
  return _result

_MapUnstageNoKeyOutput = collections.namedtuple(
    "MapUnstageNoKey",
    ["key", "values"])


def map_unstage_no_key(indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes and returns a random (key, value)

  from the underlying container.   If the underlying container
  does not contain elements, the op will block until it does.

  Args:
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, values).

    key: A `Tensor` of type `int64`.
    values: A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapUnstageNoKey", name, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      _result = _MapUnstageNoKeyOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_unstage_no_key_eager_fallback(
          indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_unstage_no_key' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapUnstageNoKey", indices=indices, dtypes=dtypes, capacity=capacity,
                           memory_limit=memory_limit, container=container,
                           shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapUnstageNoKey", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _MapUnstageNoKeyOutput._make(_result)
  return _result

MapUnstageNoKey = tf_export("raw_ops.MapUnstageNoKey")(_ops.to_raw_op(map_unstage_no_key))


def map_unstage_no_key_eager_fallback(indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'map_unstage_no_key' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"MapUnstageNoKey", len(dtypes) + 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapUnstageNoKey", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _MapUnstageNoKeyOutput._make(_result)
  return _result


def ordered_map_clear(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes all elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapClear", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_clear_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapClear", dtypes=dtypes, capacity=capacity,
                           memory_limit=memory_limit, container=container,
                           shared_name=shared_name, name=name)
  return _op
OrderedMapClear = tf_export("raw_ops.OrderedMapClear")(_ops.to_raw_op(ordered_map_clear))


def ordered_map_clear_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapClear", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def ordered_map_incomplete_size(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Op returns the number of incomplete elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapIncompleteSize", name, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_incomplete_size_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_incomplete_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapIncompleteSize", dtypes=dtypes, capacity=capacity,
                                    memory_limit=memory_limit,
                                    container=container,
                                    shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OrderedMapIncompleteSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OrderedMapIncompleteSize = tf_export("raw_ops.OrderedMapIncompleteSize")(_ops.to_raw_op(ordered_map_incomplete_size))


def ordered_map_incomplete_size_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_incomplete_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapIncompleteSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OrderedMapIncompleteSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def ordered_map_peek(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op peeks at the values at the specified key.  If the

  underlying container does not contain this key
  this op will block until it does.   This Op is optimized for
  performance.

  Args:
    key: A `Tensor` of type `int64`.
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapPeek", name, key, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_peek_eager_fallback(
          key, indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapPeek", key=key, indices=indices, dtypes=dtypes,
                          capacity=capacity, memory_limit=memory_limit,
                          container=container, shared_name=shared_name,
                          name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OrderedMapPeek", _inputs_flat, _attrs, _result)
  return _result

OrderedMapPeek = tf_export("raw_ops.OrderedMapPeek")(_ops.to_raw_op(ordered_map_peek))


def ordered_map_peek_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapPeek", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OrderedMapPeek", _inputs_flat, _attrs, _result)
  return _result


def ordered_map_size(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Op returns the number of elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapSize", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_size_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapSize", dtypes=dtypes, capacity=capacity,
                          memory_limit=memory_limit, container=container,
                          shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OrderedMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OrderedMapSize = tf_export("raw_ops.OrderedMapSize")(_ops.to_raw_op(ordered_map_size))


def ordered_map_size_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OrderedMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def ordered_map_stage(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], values, dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Stage (key, values) in the underlying container which behaves like a ordered

  associative container.   Elements are ordered by key.

  Args:
    key: A `Tensor` of type `int64`. int64
    indices: A `Tensor` of type `int32`.
    values: A list of `Tensor` objects. a list of tensors
      dtypes A list of data types that inserted values should adhere to.
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
      Maximum number of elements in the Staging Area. If > 0, inserts
      on the container will block when the capacity is reached.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container. Otherwise,
      a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      It is necessary to match this name to the matching Unstage Op.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapStage", name, key, indices, values, "capacity",
        capacity, "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_stage_eager_fallback(
          key, indices, values, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_stage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapStage", key=key, indices=indices, values=values,
                           dtypes=dtypes, capacity=capacity,
                           memory_limit=memory_limit, container=container,
                           shared_name=shared_name, name=name)
  return _op
OrderedMapStage = tf_export("raw_ops.OrderedMapStage")(_ops.to_raw_op(ordered_map_stage))


def ordered_map_stage_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], values, dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_stage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_fake_dtypes, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices] + list(values)
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "fake_dtypes", _attr_fake_dtypes, "container", container,
  "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapStage", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def ordered_map_unstage(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes and returns the values associated with the key

  from the underlying container.   If the underlying container
  does not contain this key, the op will block until it does.

  Args:
    key: A `Tensor` of type `int64`.
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapUnstage", name, key, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_unstage_eager_fallback(
          key, indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapUnstage", key=key, indices=indices, dtypes=dtypes,
                             capacity=capacity, memory_limit=memory_limit,
                             container=container, shared_name=shared_name,
                             name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OrderedMapUnstage", _inputs_flat, _attrs, _result)
  return _result

OrderedMapUnstage = tf_export("raw_ops.OrderedMapUnstage")(_ops.to_raw_op(ordered_map_unstage))


def ordered_map_unstage_eager_fallback(key: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  key = _ops.convert_to_tensor(key, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [key, indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapUnstage", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OrderedMapUnstage", _inputs_flat, _attrs, _result)
  return _result

_OrderedMapUnstageNoKeyOutput = collections.namedtuple(
    "OrderedMapUnstageNoKey",
    ["key", "values"])


def ordered_map_unstage_no_key(indices: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes and returns the (key, value) element with the smallest

  key from the underlying container.   If the underlying container
  does not contain elements, the op will block until it does.

  Args:
    indices: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, values).

    key: A `Tensor` of type `int64`.
    values: A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OrderedMapUnstageNoKey", name, indices, "capacity", capacity,
        "memory_limit", memory_limit, "dtypes", dtypes, "container",
        container, "shared_name", shared_name)
      _result = _OrderedMapUnstageNoKeyOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ordered_map_unstage_no_key_eager_fallback(
          indices, capacity=capacity, memory_limit=memory_limit,
          dtypes=dtypes, container=container, shared_name=shared_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_unstage_no_key' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OrderedMapUnstageNoKey", indices=indices, dtypes=dtypes,
                                  capacity=capacity,
                                  memory_limit=memory_limit,
                                  container=container,
                                  shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OrderedMapUnstageNoKey", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _OrderedMapUnstageNoKeyOutput._make(_result)
  return _result

OrderedMapUnstageNoKey = tf_export("raw_ops.OrderedMapUnstageNoKey")(_ops.to_raw_op(ordered_map_unstage_no_key))


def ordered_map_unstage_no_key_eager_fallback(indices: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'ordered_map_unstage_no_key' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [indices]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"OrderedMapUnstageNoKey", len(dtypes) + 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OrderedMapUnstageNoKey", _inputs_flat, _attrs, _result)
  _result = _result[:1] + [_result[1:]]
  _result = _OrderedMapUnstageNoKeyOutput._make(_result)
  return _result


def padding_fifo_queue(component_types, shapes=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A queue that produces elements in first-in first-out order.

  Variable-size shapes are allowed by setting the corresponding shape dimensions
  to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
  size of any given element in the minibatch.  See below for details.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types.
      Shapes of fixed rank but variable size are allowed by setting
      any shape dimension to -1.  In this case, the inputs' shape may vary along
      the given dimension, and DequeueMany will pad the given dimension with
      zeros up to the maximum shape of all elements in the given batch.
      If the length of this attr is 0, different queue elements may have
      different ranks and shapes, but only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("padding_fifo_queue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'padding_fifo_queue' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'padding_fifo_queue' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PaddingFIFOQueue", component_types=component_types, shapes=shapes,
                            capacity=capacity, container=container,
                            shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PaddingFIFOQueue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PaddingFIFOQueue = tf_export("raw_ops.PaddingFIFOQueue")(_ops.to_raw_op(padding_fifo_queue))


def padding_fifo_queue_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("padding_fifo_queue op does not support eager execution. Arg 'handle' is a ref.")

def padding_fifo_queue_v2(component_types, shapes=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A queue that produces elements in first-in first-out order.

  Variable-size shapes are allowed by setting the corresponding shape dimensions
  to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
  size of any given element in the minibatch.  See below for details.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types.
      Shapes of fixed rank but variable size are allowed by setting
      any shape dimension to -1.  In this case, the inputs' shape may vary along
      the given dimension, and DequeueMany will pad the given dimension with
      zeros up to the maximum shape of all elements in the given batch.
      If the length of this attr is 0, different queue elements may have
      different ranks and shapes, but only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PaddingFIFOQueueV2", name, "component_types", component_types,
        "shapes", shapes, "capacity", capacity, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return padding_fifo_queue_v2_eager_fallback(
          component_types=component_types, shapes=shapes, capacity=capacity,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'padding_fifo_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'padding_fifo_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PaddingFIFOQueueV2", component_types=component_types, shapes=shapes,
                              capacity=capacity, container=container,
                              shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PaddingFIFOQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PaddingFIFOQueueV2 = tf_export("raw_ops.PaddingFIFOQueueV2")(_ops.to_raw_op(padding_fifo_queue_v2))


def padding_fifo_queue_v2_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'padding_fifo_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'padding_fifo_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("component_types", component_types, "shapes", shapes, "capacity",
  capacity, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"PaddingFIFOQueueV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PaddingFIFOQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ParallelDynamicStitch_T = TypeVar("TV_ParallelDynamicStitch_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def parallel_dynamic_stitch(indices: Annotated[List[Any], _atypes.Int32], data: Annotated[List[Any], TV_ParallelDynamicStitch_T], name=None) -> Annotated[Any, TV_ParallelDynamicStitch_T]:
  r"""Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

  ```python
      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
  ```

  For example, if each `indices[m]` is scalar or vector, we have

  ```python
      # Scalar indices:
      merged[indices[m], ...] = data[m][...]

      # Vector indices:
      merged[indices[m][i], ...] = data[m][i, ...]
  ```

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values may be merged in parallel, so if an index appears in both `indices[m][i]`
  and `indices[n][j]`, the result may be invalid. This differs from the normal
  DynamicStitch operator that defines the behavior in that case.

  For example:

  ```python
      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]
  ```

  This method can be used to merge partitions created by `dynamic_partition`
  as illustrated on the following example:

  ```python
      # Apply function (increments x_i) on elements for which a certain condition
      # apply (x_i != -1 in this example).
      x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
      condition_mask=tf.not_equal(x,tf.constant(-1.))
      partitioned_data = tf.dynamic_partition(
          x, tf.cast(condition_mask, tf.int32) , 2)
      partitioned_data[1] = partitioned_data[1] + 1.0
      condition_indices = tf.dynamic_partition(
          tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
      x = tf.dynamic_stitch(condition_indices, partitioned_data)
      # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
      # unchanged.
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 1 `Tensor` objects with type `int32`.
    data: A list with the same length as `indices` of `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelDynamicStitch", name, indices, data)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_dynamic_stitch_eager_fallback(
          indices, data, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'parallel_dynamic_stitch' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(data, (list, tuple)):
    raise TypeError(
        "Expected list for 'data' argument to "
        "'parallel_dynamic_stitch' Op, not %r." % data)
  if len(data) != _attr_N:
    raise ValueError(
        "List argument 'data' to 'parallel_dynamic_stitch' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(data), _attr_N))
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelDynamicStitch", indices=indices, data=data, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelDynamicStitch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelDynamicStitch = tf_export("raw_ops.ParallelDynamicStitch")(_ops.to_raw_op(parallel_dynamic_stitch))


def parallel_dynamic_stitch_eager_fallback(indices: Annotated[List[Any], _atypes.Int32], data: Annotated[List[Any], TV_ParallelDynamicStitch_T], name, ctx) -> Annotated[Any, TV_ParallelDynamicStitch_T]:
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'parallel_dynamic_stitch' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(data, (list, tuple)):
    raise TypeError(
        "Expected list for 'data' argument to "
        "'parallel_dynamic_stitch' Op, not %r." % data)
  if len(data) != _attr_N:
    raise ValueError(
        "List argument 'data' to 'parallel_dynamic_stitch' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(data), _attr_N))
  _attr_T, data = _execute.args_to_matching_eager(list(data), ctx, [])
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int32)
  _inputs_flat = list(indices) + list(data)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"ParallelDynamicStitch", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelDynamicStitch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def priority_queue(shapes, component_types=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A queue that produces elements sorted by the first component value.

  Note that the PriorityQueue requires the first component of any element
  to be a scalar int64, in addition to the other elements declared by
  component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
  and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
  entry in their input (resp. output) lists.

  Args:
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    component_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type of each component in a value.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("priority_queue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'priority_queue' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if component_types is None:
    component_types = []
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'priority_queue' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PriorityQueue", shapes=shapes, component_types=component_types,
                         capacity=capacity, container=container,
                         shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PriorityQueue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PriorityQueue = tf_export("raw_ops.PriorityQueue")(_ops.to_raw_op(priority_queue))


def priority_queue_eager_fallback(shapes, component_types, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("priority_queue op does not support eager execution. Arg 'handle' is a ref.")

def priority_queue_v2(shapes, component_types=[], capacity:int=-1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A queue that produces elements sorted by the first component value.

  Note that the PriorityQueue requires the first component of any element
  to be a scalar int64, in addition to the other elements declared by
  component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
  and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
  entry in their input (resp. output) lists.

  Args:
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    component_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type of each component in a value.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PriorityQueueV2", name, "component_types", component_types,
        "shapes", shapes, "capacity", capacity, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return priority_queue_v2_eager_fallback(
          component_types=component_types, shapes=shapes, capacity=capacity,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'priority_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if component_types is None:
    component_types = []
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'priority_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PriorityQueueV2", shapes=shapes, component_types=component_types,
                           capacity=capacity, container=container,
                           shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PriorityQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PriorityQueueV2 = tf_export("raw_ops.PriorityQueueV2")(_ops.to_raw_op(priority_queue_v2))


def priority_queue_v2_eager_fallback(shapes, component_types, capacity: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'priority_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if component_types is None:
    component_types = []
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'priority_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("component_types", component_types, "shapes", shapes, "capacity",
  capacity, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"PriorityQueueV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PriorityQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def queue_close(handle: Annotated[Any, _atypes.String], cancel_pending_enqueues:bool=False, name=None):
  r"""Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be canceled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_close op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if cancel_pending_enqueues is None:
    cancel_pending_enqueues = False
  cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, "cancel_pending_enqueues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueClose", handle=handle,
                      cancel_pending_enqueues=cancel_pending_enqueues,
                      name=name)
  return _op
QueueClose = tf_export("raw_ops.QueueClose")(_ops.to_raw_op(queue_close))


def queue_close_eager_fallback(handle: Annotated[Any, _atypes.String], cancel_pending_enqueues: bool, name, ctx):
  raise RuntimeError("queue_close op does not support eager execution. Arg 'handle' is a ref.")

def queue_close_v2(handle: Annotated[Any, _atypes.Resource], cancel_pending_enqueues:bool=False, name=None):
  r"""Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be canceled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueCloseV2", name, handle, "cancel_pending_enqueues",
        cancel_pending_enqueues)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_close_v2_eager_fallback(
          handle, cancel_pending_enqueues=cancel_pending_enqueues, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if cancel_pending_enqueues is None:
    cancel_pending_enqueues = False
  cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, "cancel_pending_enqueues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueCloseV2", handle=handle,
                        cancel_pending_enqueues=cancel_pending_enqueues,
                        name=name)
  return _op
QueueCloseV2 = tf_export("raw_ops.QueueCloseV2")(_ops.to_raw_op(queue_close_v2))


def queue_close_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], cancel_pending_enqueues: bool, name, ctx):
  if cancel_pending_enqueues is None:
    cancel_pending_enqueues = False
  cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, "cancel_pending_enqueues")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = ("cancel_pending_enqueues", cancel_pending_enqueues)
  _result = _execute.execute(b"QueueCloseV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def queue_dequeue(handle: Annotated[Any, _atypes.String], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_dequeue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeue", handle=handle, component_types=component_types,
                        timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeue", _inputs_flat, _attrs, _result)
  return _result

QueueDequeue = tf_export("raw_ops.QueueDequeue")(_ops.to_raw_op(queue_dequeue))


def queue_dequeue_eager_fallback(handle: Annotated[Any, _atypes.String], component_types, timeout_ms: int, name, ctx):
  raise RuntimeError("queue_dequeue op does not support eager execution. Arg 'handle' is a ref.")

def queue_dequeue_many(handle: Annotated[Any, _atypes.String], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues `n` tuples of one or more tensors from the given queue.

  If the queue is closed and there are fewer than `n` elements, then an
  OutOfRange error is returned.

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size `n` in the 0th dimension.

  This operation has `k` outputs, where `k` is the number of components in
  the tuples stored in the given queue, and output `i` is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until `n` elements
  have been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_dequeue_many op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_many' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeueMany", handle=handle, n=n,
                            component_types=component_types,
                            timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeueMany", _inputs_flat, _attrs, _result)
  return _result

QueueDequeueMany = tf_export("raw_ops.QueueDequeueMany")(_ops.to_raw_op(queue_dequeue_many))


def queue_dequeue_many_eager_fallback(handle: Annotated[Any, _atypes.String], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms: int, name, ctx):
  raise RuntimeError("queue_dequeue_many op does not support eager execution. Arg 'handle' is a ref.")

def queue_dequeue_many_v2(handle: Annotated[Any, _atypes.Resource], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues `n` tuples of one or more tensors from the given queue.

  If the queue is closed and there are fewer than `n` elements, then an
  OutOfRange error is returned.

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size `n` in the 0th dimension.

  This operation has `k` outputs, where `k` is the number of components in
  the tuples stored in the given queue, and output `i` is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until `n` elements
  have been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueDequeueManyV2", name, handle, n, "component_types",
        component_types, "timeout_ms", timeout_ms)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_dequeue_many_v2_eager_fallback(
          handle, n, component_types=component_types, timeout_ms=timeout_ms,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_many_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeueManyV2", handle=handle, n=n,
                              component_types=component_types,
                              timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeueManyV2", _inputs_flat, _attrs, _result)
  return _result

QueueDequeueManyV2 = tf_export("raw_ops.QueueDequeueManyV2")(_ops.to_raw_op(queue_dequeue_many_v2))


def queue_dequeue_many_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms: int, name, ctx):
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_many_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  n = _ops.convert_to_tensor(n, _dtypes.int32)
  _inputs_flat = [handle, n]
  _attrs = ("component_types", component_types, "timeout_ms", timeout_ms)
  _result = _execute.execute(b"QueueDequeueManyV2", len(component_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QueueDequeueManyV2", _inputs_flat, _attrs, _result)
  return _result


def queue_dequeue_up_to(handle: Annotated[Any, _atypes.String], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues `n` tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than `n`
  elements remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, less than `n` elements are returned immediately.  If
  the queue is closed and there are 0 elements left in the queue, then
  an OutOfRange error is returned just like in QueueDequeueMany.
  Otherwise the behavior is identical to QueueDequeueMany:

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size `n` in the 0th dimension.

  This operation has k outputs, where `k` is the number of components in
  the tuples stored in the given queue, and output `i` is the ith
  component of the dequeued tuple.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_dequeue_up_to op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_up_to' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeueUpTo", handle=handle, n=n,
                            component_types=component_types,
                            timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeueUpTo", _inputs_flat, _attrs, _result)
  return _result

QueueDequeueUpTo = tf_export("raw_ops.QueueDequeueUpTo")(_ops.to_raw_op(queue_dequeue_up_to))


def queue_dequeue_up_to_eager_fallback(handle: Annotated[Any, _atypes.String], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms: int, name, ctx):
  raise RuntimeError("queue_dequeue_up_to op does not support eager execution. Arg 'handle' is a ref.")

def queue_dequeue_up_to_v2(handle: Annotated[Any, _atypes.Resource], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues `n` tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than `n`
  elements remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, less than `n` elements are returned immediately.  If
  the queue is closed and there are 0 elements left in the queue, then
  an OutOfRange error is returned just like in QueueDequeueMany.
  Otherwise the behavior is identical to QueueDequeueMany:

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has `k` outputs, where `k` is the number of components in
  the tuples stored in the given queue, and output `i` is the ith
  component of the dequeued tuple.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueDequeueUpToV2", name, handle, n, "component_types",
        component_types, "timeout_ms", timeout_ms)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_dequeue_up_to_v2_eager_fallback(
          handle, n, component_types=component_types, timeout_ms=timeout_ms,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_up_to_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeueUpToV2", handle=handle, n=n,
                              component_types=component_types,
                              timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeueUpToV2", _inputs_flat, _attrs, _result)
  return _result

QueueDequeueUpToV2 = tf_export("raw_ops.QueueDequeueUpToV2")(_ops.to_raw_op(queue_dequeue_up_to_v2))


def queue_dequeue_up_to_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], n: Annotated[Any, _atypes.Int32], component_types, timeout_ms: int, name, ctx):
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_up_to_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  n = _ops.convert_to_tensor(n, _dtypes.int32)
  _inputs_flat = [handle, n]
  _attrs = ("component_types", component_types, "timeout_ms", timeout_ms)
  _result = _execute.execute(b"QueueDequeueUpToV2", len(component_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QueueDequeueUpToV2", _inputs_flat, _attrs, _result)
  return _result


def queue_dequeue_v2(handle: Annotated[Any, _atypes.Resource], component_types, timeout_ms:int=-1, name=None):
  r"""Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueDequeueV2", name, handle, "component_types",
        component_types, "timeout_ms", timeout_ms)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_dequeue_v2_eager_fallback(
          handle, component_types=component_types, timeout_ms=timeout_ms,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueDequeueV2", handle=handle, component_types=component_types,
                          timeout_ms=timeout_ms, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"),
              "timeout_ms", _op._get_attr_int("timeout_ms"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueDequeueV2", _inputs_flat, _attrs, _result)
  return _result

QueueDequeueV2 = tf_export("raw_ops.QueueDequeueV2")(_ops.to_raw_op(queue_dequeue_v2))


def queue_dequeue_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], component_types, timeout_ms: int, name, ctx):
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'queue_dequeue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = ("component_types", component_types, "timeout_ms", timeout_ms)
  _result = _execute.execute(b"QueueDequeueV2", len(component_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QueueDequeueV2", _inputs_flat, _attrs, _result)
  return _result


def queue_enqueue(handle: Annotated[Any, _atypes.String], components, timeout_ms:int=-1, name=None):
  r"""Enqueues a tuple of one or more tensors in the given queue.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  element has been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is full, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_enqueue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueEnqueue", handle=handle, components=components,
                        timeout_ms=timeout_ms, name=name)
  return _op
QueueEnqueue = tf_export("raw_ops.QueueEnqueue")(_ops.to_raw_op(queue_enqueue))


def queue_enqueue_eager_fallback(handle: Annotated[Any, _atypes.String], components, timeout_ms: int, name, ctx):
  raise RuntimeError("queue_enqueue op does not support eager execution. Arg 'handle' is a ref.")

def queue_enqueue_many(handle: Annotated[Any, _atypes.String], components, timeout_ms:int=-1, name=None):
  r"""Enqueues zero or more tuples of one or more tensors in the given queue.

  This operation slices each component tensor along the 0th dimension to
  make multiple queue elements. All of the tuple components must have the
  same size in the 0th dimension.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  elements have been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should
      be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is too full, this operation will block for up
      to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_enqueue_many op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueEnqueueMany", handle=handle, components=components,
                            timeout_ms=timeout_ms, name=name)
  return _op
QueueEnqueueMany = tf_export("raw_ops.QueueEnqueueMany")(_ops.to_raw_op(queue_enqueue_many))


def queue_enqueue_many_eager_fallback(handle: Annotated[Any, _atypes.String], components, timeout_ms: int, name, ctx):
  raise RuntimeError("queue_enqueue_many op does not support eager execution. Arg 'handle' is a ref.")

def queue_enqueue_many_v2(handle: Annotated[Any, _atypes.Resource], components, timeout_ms:int=-1, name=None):
  r"""Enqueues zero or more tuples of one or more tensors in the given queue.

  This operation slices each component tensor along the 0th dimension to
  make multiple queue elements. All of the tuple components must have the
  same size in the 0th dimension.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  elements have been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should
      be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is too full, this operation will block for up
      to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueEnqueueManyV2", name, handle, components, "timeout_ms",
        timeout_ms)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_enqueue_many_v2_eager_fallback(
          handle, components, timeout_ms=timeout_ms, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueEnqueueManyV2", handle=handle, components=components,
                              timeout_ms=timeout_ms, name=name)
  return _op
QueueEnqueueManyV2 = tf_export("raw_ops.QueueEnqueueManyV2")(_ops.to_raw_op(queue_enqueue_many_v2))


def queue_enqueue_many_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], components, timeout_ms: int, name, ctx):
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _attr_Tcomponents, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle] + list(components)
  _attrs = ("Tcomponents", _attr_Tcomponents, "timeout_ms", timeout_ms)
  _result = _execute.execute(b"QueueEnqueueManyV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def queue_enqueue_v2(handle: Annotated[Any, _atypes.Resource], components, timeout_ms:int=-1, name=None):
  r"""Enqueues a tuple of one or more tensors in the given queue.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  element has been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is full, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueEnqueueV2", name, handle, components, "timeout_ms",
        timeout_ms)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_enqueue_v2_eager_fallback(
          handle, components, timeout_ms=timeout_ms, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueEnqueueV2", handle=handle, components=components,
                          timeout_ms=timeout_ms, name=name)
  return _op
QueueEnqueueV2 = tf_export("raw_ops.QueueEnqueueV2")(_ops.to_raw_op(queue_enqueue_v2))


def queue_enqueue_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], components, timeout_ms: int, name, ctx):
  if timeout_ms is None:
    timeout_ms = -1
  timeout_ms = _execute.make_int(timeout_ms, "timeout_ms")
  _attr_Tcomponents, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle] + list(components)
  _attrs = ("Tcomponents", _attr_Tcomponents, "timeout_ms", timeout_ms)
  _result = _execute.execute(b"QueueEnqueueV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def queue_is_closed(handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns true if queue is closed.

  This operation returns true if the queue is closed and false if the queue
  is open.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_is_closed op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueIsClosed", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueIsClosed", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QueueIsClosed = tf_export("raw_ops.QueueIsClosed")(_ops.to_raw_op(queue_is_closed))


def queue_is_closed_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Bool]:
  raise RuntimeError("queue_is_closed op does not support eager execution. Arg 'handle' is a ref.")

def queue_is_closed_v2(handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns true if queue is closed.

  This operation returns true if the queue is closed and false if the queue
  is open.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueIsClosedV2", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_is_closed_v2_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueIsClosedV2", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueIsClosedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QueueIsClosedV2 = tf_export("raw_ops.QueueIsClosedV2")(_ops.to_raw_op(queue_is_closed_v2))


def queue_is_closed_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Bool]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"QueueIsClosedV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QueueIsClosedV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def queue_size(handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("queue_size op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueSize", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QueueSize = tf_export("raw_ops.QueueSize")(_ops.to_raw_op(queue_size))


def queue_size_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("queue_size op does not support eager execution. Arg 'handle' is a ref.")

def queue_size_v2(handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QueueSizeV2", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return queue_size_v2_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QueueSizeV2", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QueueSizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QueueSizeV2 = tf_export("raw_ops.QueueSizeV2")(_ops.to_raw_op(queue_size_v2))


def queue_size_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int32]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"QueueSizeV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QueueSizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def random_shuffle_queue(component_types, shapes=[], capacity:int=-1, min_after_dequeue:int=0, seed:int=0, seed2:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""A queue that randomizes the order of elements.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    min_after_dequeue: An optional `int`. Defaults to `0`.
      Dequeue will block unless there would be this
      many elements after the dequeue or the queue is closed. This
      ensures a minimum level of mixing of elements.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("random_shuffle_queue op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'random_shuffle_queue' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'random_shuffle_queue' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if min_after_dequeue is None:
    min_after_dequeue = 0
  min_after_dequeue = _execute.make_int(min_after_dequeue, "min_after_dequeue")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomShuffleQueue", component_types=component_types, shapes=shapes,
                              capacity=capacity,
                              min_after_dequeue=min_after_dequeue, seed=seed,
                              seed2=seed2, container=container,
                              shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "min_after_dequeue",
              _op._get_attr_int("min_after_dequeue"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomShuffleQueue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomShuffleQueue = tf_export("raw_ops.RandomShuffleQueue")(_ops.to_raw_op(random_shuffle_queue))


def random_shuffle_queue_eager_fallback(component_types, shapes, capacity: int, min_after_dequeue: int, seed: int, seed2: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("random_shuffle_queue op does not support eager execution. Arg 'handle' is a ref.")

def random_shuffle_queue_v2(component_types, shapes=[], capacity:int=-1, min_after_dequeue:int=0, seed:int=0, seed2:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A queue that randomizes the order of elements.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    min_after_dequeue: An optional `int`. Defaults to `0`.
      Dequeue will block unless there would be this
      many elements after the dequeue or the queue is closed. This
      ensures a minimum level of mixing of elements.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomShuffleQueueV2", name, "component_types",
        component_types, "shapes", shapes, "capacity", capacity,
        "min_after_dequeue", min_after_dequeue, "seed", seed, "seed2", seed2,
        "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_shuffle_queue_v2_eager_fallback(
          component_types=component_types, shapes=shapes, capacity=capacity,
          min_after_dequeue=min_after_dequeue, seed=seed, seed2=seed2,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'random_shuffle_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'random_shuffle_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if min_after_dequeue is None:
    min_after_dequeue = 0
  min_after_dequeue = _execute.make_int(min_after_dequeue, "min_after_dequeue")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomShuffleQueueV2", component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                min_after_dequeue=min_after_dequeue,
                                seed=seed, seed2=seed2, container=container,
                                shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("component_types", _op.get_attr("component_types"), "shapes",
              _op.get_attr("shapes"), "capacity",
              _op._get_attr_int("capacity"), "min_after_dequeue",
              _op._get_attr_int("min_after_dequeue"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomShuffleQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomShuffleQueueV2 = tf_export("raw_ops.RandomShuffleQueueV2")(_ops.to_raw_op(random_shuffle_queue_v2))


def random_shuffle_queue_v2_eager_fallback(component_types, shapes, capacity: int, min_after_dequeue: int, seed: int, seed2: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(component_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'component_types' argument to "
        "'random_shuffle_queue_v2' Op, not %r." % component_types)
  component_types = [_execute.make_type(_t, "component_types") for _t in component_types]
  if shapes is None:
    shapes = []
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'random_shuffle_queue_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if capacity is None:
    capacity = -1
  capacity = _execute.make_int(capacity, "capacity")
  if min_after_dequeue is None:
    min_after_dequeue = 0
  min_after_dequeue = _execute.make_int(min_after_dequeue, "min_after_dequeue")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("component_types", component_types, "shapes", shapes, "capacity",
  capacity, "min_after_dequeue", min_after_dequeue, "seed", seed, "seed2",
  seed2, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"RandomShuffleQueueV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomShuffleQueueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def record_input(file_pattern: str, file_random_seed:int=301, file_shuffle_shift_ratio:float=0, file_buffer_size:int=10000, file_parallelism:int=16, batch_size:int=32, compression_type:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Emits randomized records.

  Args:
    file_pattern: A `string`. Glob pattern for the data files.
    file_random_seed: An optional `int`. Defaults to `301`.
      Random seeds used to produce randomized records.
    file_shuffle_shift_ratio: An optional `float`. Defaults to `0`.
      Shifts the list of files after the list is randomly
      shuffled.
    file_buffer_size: An optional `int`. Defaults to `10000`.
      The randomization shuffling buffer.
    file_parallelism: An optional `int`. Defaults to `16`.
      How many sstables are opened and concurrently iterated over.
    batch_size: An optional `int`. Defaults to `32`. The batch size.
    compression_type: An optional `string`. Defaults to `""`.
      The type of compression for the file. Currently ZLIB and
      GZIP are supported. Defaults to none.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RecordInput", name, "file_pattern", file_pattern,
        "file_random_seed", file_random_seed, "file_shuffle_shift_ratio",
        file_shuffle_shift_ratio, "file_buffer_size", file_buffer_size,
        "file_parallelism", file_parallelism, "batch_size", batch_size,
        "compression_type", compression_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return record_input_eager_fallback(
          file_pattern=file_pattern, file_random_seed=file_random_seed,
          file_shuffle_shift_ratio=file_shuffle_shift_ratio,
          file_buffer_size=file_buffer_size,
          file_parallelism=file_parallelism, batch_size=batch_size,
          compression_type=compression_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  file_pattern = _execute.make_str(file_pattern, "file_pattern")
  if file_random_seed is None:
    file_random_seed = 301
  file_random_seed = _execute.make_int(file_random_seed, "file_random_seed")
  if file_shuffle_shift_ratio is None:
    file_shuffle_shift_ratio = 0
  file_shuffle_shift_ratio = _execute.make_float(file_shuffle_shift_ratio, "file_shuffle_shift_ratio")
  if file_buffer_size is None:
    file_buffer_size = 10000
  file_buffer_size = _execute.make_int(file_buffer_size, "file_buffer_size")
  if file_parallelism is None:
    file_parallelism = 16
  file_parallelism = _execute.make_int(file_parallelism, "file_parallelism")
  if batch_size is None:
    batch_size = 32
  batch_size = _execute.make_int(batch_size, "batch_size")
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RecordInput", file_pattern=file_pattern,
                       file_random_seed=file_random_seed,
                       file_shuffle_shift_ratio=file_shuffle_shift_ratio,
                       file_buffer_size=file_buffer_size,
                       file_parallelism=file_parallelism,
                       batch_size=batch_size,
                       compression_type=compression_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("file_pattern", _op.get_attr("file_pattern"),
              "file_random_seed", _op._get_attr_int("file_random_seed"),
              "file_shuffle_shift_ratio",
              _op.get_attr("file_shuffle_shift_ratio"), "file_buffer_size",
              _op._get_attr_int("file_buffer_size"), "file_parallelism",
              _op._get_attr_int("file_parallelism"), "batch_size",
              _op._get_attr_int("batch_size"), "compression_type",
              _op.get_attr("compression_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RecordInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RecordInput = tf_export("raw_ops.RecordInput")(_ops.to_raw_op(record_input))


def record_input_eager_fallback(file_pattern: str, file_random_seed: int, file_shuffle_shift_ratio: float, file_buffer_size: int, file_parallelism: int, batch_size: int, compression_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  file_pattern = _execute.make_str(file_pattern, "file_pattern")
  if file_random_seed is None:
    file_random_seed = 301
  file_random_seed = _execute.make_int(file_random_seed, "file_random_seed")
  if file_shuffle_shift_ratio is None:
    file_shuffle_shift_ratio = 0
  file_shuffle_shift_ratio = _execute.make_float(file_shuffle_shift_ratio, "file_shuffle_shift_ratio")
  if file_buffer_size is None:
    file_buffer_size = 10000
  file_buffer_size = _execute.make_int(file_buffer_size, "file_buffer_size")
  if file_parallelism is None:
    file_parallelism = 16
  file_parallelism = _execute.make_int(file_parallelism, "file_parallelism")
  if batch_size is None:
    batch_size = 32
  batch_size = _execute.make_int(batch_size, "batch_size")
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  _inputs_flat = []
  _attrs = ("file_pattern", file_pattern, "file_random_seed",
  file_random_seed, "file_shuffle_shift_ratio", file_shuffle_shift_ratio,
  "file_buffer_size", file_buffer_size, "file_parallelism", file_parallelism,
  "batch_size", batch_size, "compression_type", compression_type)
  _result = _execute.execute(b"RecordInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RecordInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResourceAccumulatorApplyGradient_dtype = TypeVar("TV_ResourceAccumulatorApplyGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_accumulator_apply_gradient(handle: Annotated[Any, _atypes.Resource], local_step: Annotated[Any, _atypes.Int64], gradient: Annotated[Any, TV_ResourceAccumulatorApplyGradient_dtype], name=None):
  r"""Applies a gradient to a given accumulator.

  Does not add if local_step is lesser than the accumulator's global_step.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the gradient was computed.
    gradient: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A tensor of the gradient to be accumulated.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceAccumulatorApplyGradient", name, handle, local_step,
        gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_accumulator_apply_gradient_eager_fallback(
          handle, local_step, gradient, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceAccumulatorApplyGradient", handle=handle,
                                            local_step=local_step,
                                            gradient=gradient, name=name)
  return _op
ResourceAccumulatorApplyGradient = tf_export("raw_ops.ResourceAccumulatorApplyGradient")(_ops.to_raw_op(resource_accumulator_apply_gradient))


def resource_accumulator_apply_gradient_eager_fallback(handle: Annotated[Any, _atypes.Resource], local_step: Annotated[Any, _atypes.Int64], gradient: Annotated[Any, TV_ResourceAccumulatorApplyGradient_dtype], name, ctx):
  _attr_dtype, (gradient,) = _execute.args_to_matching_eager([gradient], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  local_step = _ops.convert_to_tensor(local_step, _dtypes.int64)
  _inputs_flat = [handle, local_step, gradient]
  _attrs = ("dtype", _attr_dtype)
  _result = _execute.execute(b"ResourceAccumulatorApplyGradient", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def resource_accumulator_num_accumulated(handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the number of gradients aggregated in the given accumulators.

  Args:
    handle: A `Tensor` of type `resource`. The handle to an accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceAccumulatorNumAccumulated", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_accumulator_num_accumulated_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceAccumulatorNumAccumulated", handle=handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResourceAccumulatorNumAccumulated", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResourceAccumulatorNumAccumulated = tf_export("raw_ops.ResourceAccumulatorNumAccumulated")(_ops.to_raw_op(resource_accumulator_num_accumulated))


def resource_accumulator_num_accumulated_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int32]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"ResourceAccumulatorNumAccumulated", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResourceAccumulatorNumAccumulated", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def resource_accumulator_set_global_step(handle: Annotated[Any, _atypes.Resource], new_global_step: Annotated[Any, _atypes.Int64], name=None):
  r"""Updates the accumulator with a new value for global_step.

  Logs warning if the accumulator's value is already higher than
  new_global_step.

  Args:
    handle: A `Tensor` of type `resource`. The handle to an accumulator.
    new_global_step: A `Tensor` of type `int64`.
      The new global_step value to set.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceAccumulatorSetGlobalStep", name, handle,
        new_global_step)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_accumulator_set_global_step_eager_fallback(
          handle, new_global_step, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceAccumulatorSetGlobalStep", handle=handle,
                                            new_global_step=new_global_step,
                                            name=name)
  return _op
ResourceAccumulatorSetGlobalStep = tf_export("raw_ops.ResourceAccumulatorSetGlobalStep")(_ops.to_raw_op(resource_accumulator_set_global_step))


def resource_accumulator_set_global_step_eager_fallback(handle: Annotated[Any, _atypes.Resource], new_global_step: Annotated[Any, _atypes.Int64], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  new_global_step = _ops.convert_to_tensor(new_global_step, _dtypes.int64)
  _inputs_flat = [handle, new_global_step]
  _attrs = None
  _result = _execute.execute(b"ResourceAccumulatorSetGlobalStep", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceAccumulatorTakeGradient_dtype = TypeVar("TV_ResourceAccumulatorTakeGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_accumulator_take_gradient(handle: Annotated[Any, _atypes.Resource], num_required: Annotated[Any, _atypes.Int32], dtype: TV_ResourceAccumulatorTakeGradient_dtype, name=None) -> Annotated[Any, TV_ResourceAccumulatorTakeGradient_dtype]:
  r"""Extracts the average gradient in the given ConditionalAccumulator.

  The op blocks until sufficient (i.e., more than num_required)
  gradients have been accumulated.  If the accumulator has already
  aggregated more than num_required gradients, it returns the average of
  the accumulated gradients.  Also automatically increments the recorded
  global_step in the accumulator by 1, and resets the aggregate to 0.

  Args:
    handle: A `Tensor` of type `resource`. The handle to an accumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceAccumulatorTakeGradient", name, handle, num_required,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_accumulator_take_gradient_eager_fallback(
          handle, num_required, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceAccumulatorTakeGradient", handle=handle,
                                           num_required=num_required,
                                           dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResourceAccumulatorTakeGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResourceAccumulatorTakeGradient = tf_export("raw_ops.ResourceAccumulatorTakeGradient")(_ops.to_raw_op(resource_accumulator_take_gradient))


def resource_accumulator_take_gradient_eager_fallback(handle: Annotated[Any, _atypes.Resource], num_required: Annotated[Any, _atypes.Int32], dtype: TV_ResourceAccumulatorTakeGradient_dtype, name, ctx) -> Annotated[Any, TV_ResourceAccumulatorTakeGradient_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  num_required = _ops.convert_to_tensor(num_required, _dtypes.int32)
  _inputs_flat = [handle, num_required]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"ResourceAccumulatorTakeGradient", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResourceAccumulatorTakeGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResourceConditionalAccumulator_dtype = TypeVar("TV_ResourceConditionalAccumulator_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_conditional_accumulator(dtype: TV_ResourceConditionalAccumulator_dtype, shape, container:str="", shared_name:str="", reduction_type:str="MEAN", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A conditional accumulator for aggregating gradients.

  The accumulator accepts gradients marked with local_step greater or
  equal to the most recent global_step known to the accumulator. The
  average can be extracted from the accumulator, provided sufficient
  gradients have been accumulated. Extracting the average automatically
  resets the aggregate to 0, and increments the global_step recorded by
  the accumulator.
  This is a resource version of ConditionalAccumulator that will work in TF2.0
  with tf.cond version 2.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`.
      The shape of the values, can be [], in which case shape is unknown.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the
      given name across multiple sessions.
    reduction_type: An optional `string` from: `"MEAN", "SUM"`. Defaults to `"MEAN"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceConditionalAccumulator", name, "dtype", dtype, "shape",
        shape, "container", container, "shared_name", shared_name,
        "reduction_type", reduction_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_conditional_accumulator_eager_fallback(
          dtype=dtype, shape=shape, container=container,
          shared_name=shared_name, reduction_type=reduction_type, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if reduction_type is None:
    reduction_type = "MEAN"
  reduction_type = _execute.make_str(reduction_type, "reduction_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceConditionalAccumulator", dtype=dtype, shape=shape,
                                          container=container,
                                          shared_name=shared_name,
                                          reduction_type=reduction_type,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"), "reduction_type",
              _op.get_attr("reduction_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResourceConditionalAccumulator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResourceConditionalAccumulator = tf_export("raw_ops.ResourceConditionalAccumulator")(_ops.to_raw_op(resource_conditional_accumulator))


def resource_conditional_accumulator_eager_fallback(dtype: TV_ResourceConditionalAccumulator_dtype, shape, container: str, shared_name: str, reduction_type: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if reduction_type is None:
    reduction_type = "MEAN"
  reduction_type = _execute.make_str(reduction_type, "reduction_type")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape, "container", container,
  "shared_name", shared_name, "reduction_type", reduction_type)
  _result = _execute.execute(b"ResourceConditionalAccumulator", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResourceConditionalAccumulator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseAccumulatorApplyGradient_dtype = TypeVar("TV_SparseAccumulatorApplyGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_accumulator_apply_gradient(handle: Annotated[Any, _atypes.String], local_step: Annotated[Any, _atypes.Int64], gradient_indices: Annotated[Any, _atypes.Int64], gradient_values: Annotated[Any, TV_SparseAccumulatorApplyGradient_dtype], gradient_shape: Annotated[Any, _atypes.Int64], has_known_shape: bool, name=None):
  r"""Applies a sparse gradient to a given accumulator.

  Does not add if local_step is smaller than the accumulator's
  global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the sparse gradient was computed.
    gradient_indices: A `Tensor` of type `int64`.
      Indices of the sparse gradient to be accumulated. Must be a
      vector.
    gradient_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Values are the non-zero slices of the gradient, and must have
      the same first dimension as indices, i.e., the nnz represented by indices and
      values must be consistent.
    gradient_shape: A `Tensor` of type `int64`.
      Shape of the sparse gradient to be accumulated.
    has_known_shape: A `bool`.
      Boolean indicating whether gradient_shape is unknown, in which
      case the input is ignored during validation.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_accumulator_apply_gradient op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  has_known_shape = _execute.make_bool(has_known_shape, "has_known_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseAccumulatorApplyGradient", handle=handle,
                                          local_step=local_step,
                                          gradient_indices=gradient_indices,
                                          gradient_values=gradient_values,
                                          gradient_shape=gradient_shape,
                                          has_known_shape=has_known_shape,
                                          name=name)
  return _op
SparseAccumulatorApplyGradient = tf_export("raw_ops.SparseAccumulatorApplyGradient")(_ops.to_raw_op(sparse_accumulator_apply_gradient))


def sparse_accumulator_apply_gradient_eager_fallback(handle: Annotated[Any, _atypes.String], local_step: Annotated[Any, _atypes.Int64], gradient_indices: Annotated[Any, _atypes.Int64], gradient_values: Annotated[Any, TV_SparseAccumulatorApplyGradient_dtype], gradient_shape: Annotated[Any, _atypes.Int64], has_known_shape: bool, name, ctx):
  raise RuntimeError("sparse_accumulator_apply_gradient op does not support eager execution. Arg 'handle' is a ref.")
_SparseAccumulatorTakeGradientOutput = collections.namedtuple(
    "SparseAccumulatorTakeGradient",
    ["indices", "values", "shape"])


TV_SparseAccumulatorTakeGradient_dtype = TypeVar("TV_SparseAccumulatorTakeGradient_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_accumulator_take_gradient(handle: Annotated[Any, _atypes.String], num_required: Annotated[Any, _atypes.Int32], dtype: TV_SparseAccumulatorTakeGradient_dtype, name=None):
  r"""Extracts the average sparse gradient in a SparseConditionalAccumulator.

  The op will blocks until sufficient (i.e., more than num_required)
  gradients have been accumulated. If the accumulator has already
  aggregated more than num_required gradients, it will return its
  average of the accumulated gradients.  Also automatically increments
  the recorded global_step in the accumulator by 1, and resets the
  aggregate to 0.

  Args:
    handle: A `Tensor` of type mutable `string`.
      The handle to a SparseConditionalAccumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).

    indices: A `Tensor` of type `int64`.
    values: A `Tensor` of type `dtype`.
    shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_accumulator_take_gradient op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseAccumulatorTakeGradient", handle=handle,
                                         num_required=num_required,
                                         dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseAccumulatorTakeGradient", _inputs_flat, _attrs, _result)
  _result = _SparseAccumulatorTakeGradientOutput._make(_result)
  return _result

SparseAccumulatorTakeGradient = tf_export("raw_ops.SparseAccumulatorTakeGradient")(_ops.to_raw_op(sparse_accumulator_take_gradient))


def sparse_accumulator_take_gradient_eager_fallback(handle: Annotated[Any, _atypes.String], num_required: Annotated[Any, _atypes.Int32], dtype: TV_SparseAccumulatorTakeGradient_dtype, name, ctx):
  raise RuntimeError("sparse_accumulator_take_gradient op does not support eager execution. Arg 'handle' is a ref.")

TV_SparseConditionalAccumulator_dtype = TypeVar("TV_SparseConditionalAccumulator_dtype", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_conditional_accumulator(dtype: TV_SparseConditionalAccumulator_dtype, shape, container:str="", shared_name:str="", reduction_type:str="MEAN", name=None) -> Annotated[Any, _atypes.String]:
  r"""A conditional accumulator for aggregating sparse gradients.

  The accumulator accepts gradients marked with local_step greater or
  equal to the most recent global_step known to the accumulator. The
  average can be extracted from the accumulator, provided sufficient
  gradients have been accumulated. Extracting the average automatically
  resets the aggregate to 0, and increments the global_step recorded by
  the accumulator.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the given name
      across multiple sessions.
    reduction_type: An optional `string` from: `"MEAN", "SUM"`. Defaults to `"MEAN"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_conditional_accumulator op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if reduction_type is None:
    reduction_type = "MEAN"
  reduction_type = _execute.make_str(reduction_type, "reduction_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseConditionalAccumulator", dtype=dtype, shape=shape,
                                        container=container,
                                        shared_name=shared_name,
                                        reduction_type=reduction_type,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"), "reduction_type",
              _op.get_attr("reduction_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseConditionalAccumulator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseConditionalAccumulator = tf_export("raw_ops.SparseConditionalAccumulator")(_ops.to_raw_op(sparse_conditional_accumulator))


def sparse_conditional_accumulator_eager_fallback(dtype: TV_SparseConditionalAccumulator_dtype, shape, container: str, shared_name: str, reduction_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("sparse_conditional_accumulator op does not support eager execution. Arg 'handle' is a ref.")

TV_Stack_elem_type = TypeVar("TV_Stack_elem_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def _stack(elem_type: TV_Stack_elem_type, stack_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Deprecated, use StackV2.

  Args:
    elem_type: A `tf.DType`.
    stack_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("stack op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  elem_type = _execute.make_type(elem_type, "elem_type")
  if stack_name is None:
    stack_name = ""
  stack_name = _execute.make_str(stack_name, "stack_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Stack", elem_type=elem_type, stack_name=stack_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("elem_type", _op._get_attr_type("elem_type"), "stack_name",
              _op.get_attr("stack_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Stack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Stack = tf_export("raw_ops.Stack")(_ops.to_raw_op(_stack))


def _stack_eager_fallback(elem_type: TV_Stack_elem_type, stack_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("stack op does not support eager execution. Arg 'handle' is a ref.")

def stack_close(handle: Annotated[Any, _atypes.String], name=None):
  r"""Deprecated, use StackCloseV2.

  Args:
    handle: A `Tensor` of type mutable `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("stack_close op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackClose", handle=handle, name=name)
  return _op
StackClose = tf_export("raw_ops.StackClose")(_ops.to_raw_op(stack_close))


def stack_close_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx):
  raise RuntimeError("stack_close op does not support eager execution. Arg 'handle' is a ref.")

def stack_close_v2(handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Delete the stack from its resource container.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a stack.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StackCloseV2", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stack_close_v2_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackCloseV2", handle=handle, name=name)
  return _op
StackCloseV2 = tf_export("raw_ops.StackCloseV2")(_ops.to_raw_op(stack_close_v2))


def stack_close_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"StackCloseV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_StackPop_elem_type = TypeVar("TV_StackPop_elem_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stack_pop(handle: Annotated[Any, _atypes.String], elem_type: TV_StackPop_elem_type, name=None) -> Annotated[Any, TV_StackPop_elem_type]:
  r"""Deprecated, use StackPopV2.

  Args:
    handle: A `Tensor` of type mutable `string`.
    elem_type: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `elem_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("stack_pop op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  elem_type = _execute.make_type(elem_type, "elem_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackPop", handle=handle, elem_type=elem_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("elem_type", _op._get_attr_type("elem_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StackPop", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StackPop = tf_export("raw_ops.StackPop")(_ops.to_raw_op(stack_pop))


def stack_pop_eager_fallback(handle: Annotated[Any, _atypes.String], elem_type: TV_StackPop_elem_type, name, ctx) -> Annotated[Any, TV_StackPop_elem_type]:
  raise RuntimeError("stack_pop op does not support eager execution. Arg 'handle' is a ref.")

TV_StackPopV2_elem_type = TypeVar("TV_StackPopV2_elem_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stack_pop_v2(handle: Annotated[Any, _atypes.Resource], elem_type: TV_StackPopV2_elem_type, name=None) -> Annotated[Any, TV_StackPopV2_elem_type]:
  r"""Pop the element at the top of the stack.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a stack.
    elem_type: A `tf.DType`. The type of the elem that is popped.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `elem_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StackPopV2", name, handle, "elem_type", elem_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stack_pop_v2_eager_fallback(
          handle, elem_type=elem_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  elem_type = _execute.make_type(elem_type, "elem_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackPopV2", handle=handle, elem_type=elem_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("elem_type", _op._get_attr_type("elem_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StackPopV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StackPopV2 = tf_export("raw_ops.StackPopV2")(_ops.to_raw_op(stack_pop_v2))


def stack_pop_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], elem_type: TV_StackPopV2_elem_type, name, ctx) -> Annotated[Any, TV_StackPopV2_elem_type]:
  elem_type = _execute.make_type(elem_type, "elem_type")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = ("elem_type", elem_type)
  _result = _execute.execute(b"StackPopV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StackPopV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StackPush_T = TypeVar("TV_StackPush_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stack_push(handle: Annotated[Any, _atypes.String], elem: Annotated[Any, TV_StackPush_T], swap_memory:bool=False, name=None) -> Annotated[Any, TV_StackPush_T]:
  r"""Deprecated, use StackPushV2.

  Args:
    handle: A `Tensor` of type mutable `string`.
    elem: A `Tensor`.
    swap_memory: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `elem`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("stack_push op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  if swap_memory is None:
    swap_memory = False
  swap_memory = _execute.make_bool(swap_memory, "swap_memory")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackPush", handle=handle, elem=elem, swap_memory=swap_memory,
                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "swap_memory",
              _op._get_attr_bool("swap_memory"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StackPush", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StackPush = tf_export("raw_ops.StackPush")(_ops.to_raw_op(stack_push))


def stack_push_eager_fallback(handle: Annotated[Any, _atypes.String], elem: Annotated[Any, TV_StackPush_T], swap_memory: bool, name, ctx) -> Annotated[Any, TV_StackPush_T]:
  raise RuntimeError("stack_push op does not support eager execution. Arg 'handle' is a ref.")

TV_StackPushV2_T = TypeVar("TV_StackPushV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stack_push_v2(handle: Annotated[Any, _atypes.Resource], elem: Annotated[Any, TV_StackPushV2_T], swap_memory:bool=False, name=None) -> Annotated[Any, TV_StackPushV2_T]:
  r"""Push an element onto the stack.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a stack.
    elem: A `Tensor`. The tensor to be pushed onto the stack.
    swap_memory: An optional `bool`. Defaults to `False`.
      Swap `elem` to CPU. Default to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `elem`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StackPushV2", name, handle, elem, "swap_memory", swap_memory)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stack_push_v2_eager_fallback(
          handle, elem, swap_memory=swap_memory, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if swap_memory is None:
    swap_memory = False
  swap_memory = _execute.make_bool(swap_memory, "swap_memory")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackPushV2", handle=handle, elem=elem, swap_memory=swap_memory,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "swap_memory",
              _op._get_attr_bool("swap_memory"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StackPushV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StackPushV2 = tf_export("raw_ops.StackPushV2")(_ops.to_raw_op(stack_push_v2))


def stack_push_v2_eager_fallback(handle: Annotated[Any, _atypes.Resource], elem: Annotated[Any, TV_StackPushV2_T], swap_memory: bool, name, ctx) -> Annotated[Any, TV_StackPushV2_T]:
  if swap_memory is None:
    swap_memory = False
  swap_memory = _execute.make_bool(swap_memory, "swap_memory")
  _attr_T, (elem,) = _execute.args_to_matching_eager([elem], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle, elem]
  _attrs = ("T", _attr_T, "swap_memory", swap_memory)
  _result = _execute.execute(b"StackPushV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StackPushV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StackV2_elem_type = TypeVar("TV_StackV2_elem_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stack_v2(max_size: Annotated[Any, _atypes.Int32], elem_type: TV_StackV2_elem_type, stack_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A stack that produces elements in first-in last-out order.

  Args:
    max_size: A `Tensor` of type `int32`.
      The maximum size of the stack if non-negative. If negative, the stack
      size is unlimited.
    elem_type: A `tf.DType`. The type of the elements on the stack.
    stack_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary stack resource. Default
      value is the name of the 'Stack' op (which is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StackV2", name, max_size, "elem_type", elem_type, "stack_name",
        stack_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stack_v2_eager_fallback(
          max_size, elem_type=elem_type, stack_name=stack_name, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  elem_type = _execute.make_type(elem_type, "elem_type")
  if stack_name is None:
    stack_name = ""
  stack_name = _execute.make_str(stack_name, "stack_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StackV2", max_size=max_size, elem_type=elem_type,
                   stack_name=stack_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("elem_type", _op._get_attr_type("elem_type"), "stack_name",
              _op.get_attr("stack_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StackV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StackV2 = tf_export("raw_ops.StackV2")(_ops.to_raw_op(stack_v2))


def stack_v2_eager_fallback(max_size: Annotated[Any, _atypes.Int32], elem_type: TV_StackV2_elem_type, stack_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  elem_type = _execute.make_type(elem_type, "elem_type")
  if stack_name is None:
    stack_name = ""
  stack_name = _execute.make_str(stack_name, "stack_name")
  max_size = _ops.convert_to_tensor(max_size, _dtypes.int32)
  _inputs_flat = [max_size]
  _attrs = ("elem_type", elem_type, "stack_name", stack_name)
  _result = _execute.execute(b"StackV2", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StackV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def stage(values, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Stage values similar to a lightweight Enqueue.

  The basic functionality of this Op is similar to a queue with many
  fewer capabilities and options.  This Op is optimized for performance.

  Args:
    values: A list of `Tensor` objects. a list of tensors
      dtypes A list of data types that inserted values should adhere to.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
      Maximum number of elements in the Staging Area. If > 0, inserts
      on the container will block when the capacity is reached.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
      The maximum number of bytes allowed for Tensors in the Staging Area.
      If > 0, inserts will block until sufficient space is available.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container. Otherwise,
      a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      It is necessary to match this name to the matching Unstage Op.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Stage", name, values, "capacity", capacity, "memory_limit",
        memory_limit, "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stage_eager_fallback(
          values, capacity=capacity, memory_limit=memory_limit,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Stage", values=values, capacity=capacity, memory_limit=memory_limit,
                 container=container, shared_name=shared_name, name=name)
  return _op
Stage = tf_export("raw_ops.Stage")(_ops.to_raw_op(stage))


def stage_eager_fallback(values, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_dtypes, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  _inputs_flat = list(values)
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  _attr_dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"Stage", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


def stage_clear(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op removes all elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StageClear", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stage_clear_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StageClear", dtypes=dtypes, capacity=capacity,
                      memory_limit=memory_limit, container=container,
                      shared_name=shared_name, name=name)
  return _op
StageClear = tf_export("raw_ops.StageClear")(_ops.to_raw_op(stage_clear))


def stage_clear_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_clear' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StageClear", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def stage_peek(index: Annotated[Any, _atypes.Int32], dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op peeks at the values at the specified index.  If the

  underlying container does not contain sufficient elements
  this op will block until it does.   This Op is optimized for
  performance.

  Args:
    index: A `Tensor` of type `int32`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StagePeek", name, index, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stage_peek_eager_fallback(
          index, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StagePeek", index=index, dtypes=dtypes, capacity=capacity,
                     memory_limit=memory_limit, container=container,
                     shared_name=shared_name, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StagePeek", _inputs_flat, _attrs, _result)
  return _result

StagePeek = tf_export("raw_ops.StagePeek")(_ops.to_raw_op(stage_peek))


def stage_peek_eager_fallback(index: Annotated[Any, _atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_peek' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  _inputs_flat = [index]
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StagePeek", len(dtypes), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StagePeek", _inputs_flat, _attrs, _result)
  return _result


def stage_size(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Op returns the number of elements in the underlying container.

  Args:
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StageSize", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stage_size_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StageSize", dtypes=dtypes, capacity=capacity,
                     memory_limit=memory_limit, container=container,
                     shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StageSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StageSize = tf_export("raw_ops.StageSize")(_ops.to_raw_op(stage_size))


def stage_size_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'stage_size' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StageSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StageSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArray_dtype = TypeVar("TV_TensorArray_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArray_dtype, dynamic_size:bool=False, clear_after_read:bool=True, tensor_array_name:str="", element_shape=None, name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    size: A `Tensor` of type `int32`.
    dtype: A `tf.DType`.
    dynamic_size: An optional `bool`. Defaults to `False`.
    clear_after_read: An optional `bool`. Defaults to `True`.
    tensor_array_name: An optional `string`. Defaults to `""`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if dynamic_size is None:
    dynamic_size = False
  dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
  if clear_after_read is None:
    clear_after_read = True
  clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
  if tensor_array_name is None:
    tensor_array_name = ""
  tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArray", size=size, dtype=dtype, dynamic_size=dynamic_size,
                       clear_after_read=clear_after_read,
                       tensor_array_name=tensor_array_name,
                       element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "dynamic_size",
              _op._get_attr_bool("dynamic_size"), "clear_after_read",
              _op._get_attr_bool("clear_after_read"), "tensor_array_name",
              _op.get_attr("tensor_array_name"), "element_shape",
              _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArray", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArray = tf_export("raw_ops.TensorArray")(_ops.to_raw_op(tensor_array))


def tensor_array_eager_fallback(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArray_dtype, dynamic_size: bool, clear_after_read: bool, tensor_array_name: str, element_shape, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("tensor_array op does not support eager execution. Arg 'handle' is a ref.")

def tensor_array_close(handle: Annotated[Any, _atypes.String], name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_close op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayClose", handle=handle, name=name)
  return _op
TensorArrayClose = tf_export("raw_ops.TensorArrayClose")(_ops.to_raw_op(tensor_array_close))


def tensor_array_close_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx):
  raise RuntimeError("tensor_array_close op does not support eager execution. Arg 'handle' is a ref.")

def tensor_array_close_v2(handle: Annotated[Any, _atypes.String], name=None):
  r"""Deprecated. Use TensorArrayCloseV3

  Args:
    handle: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayCloseV2", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_close_v2_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayCloseV2", handle=handle, name=name)
  return _op
TensorArrayCloseV2 = tf_export("raw_ops.TensorArrayCloseV2")(_ops.to_raw_op(tensor_array_close_v2))


def tensor_array_close_v2_eager_fallback(handle: Annotated[Any, _atypes.String], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"TensorArrayCloseV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def tensor_array_close_v3(handle: Annotated[Any, _atypes.Resource], name=None):
  r"""Delete the TensorArray from its resource container.

  This enables the user to close and release the resource in the middle
  of a step/run.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayCloseV3", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_close_v3_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayCloseV3", handle=handle, name=name)
  return _op
TensorArrayCloseV3 = tf_export("raw_ops.TensorArrayCloseV3")(_ops.to_raw_op(tensor_array_close_v3))


def tensor_array_close_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"TensorArrayCloseV3", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

_TensorArrayConcatOutput = collections.namedtuple(
    "TensorArrayConcat",
    ["value", "lengths"])


TV_TensorArrayConcat_dtype = TypeVar("TV_TensorArrayConcat_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_concat(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcat_dtype, element_shape_except0=None, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).

    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_concat op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape_except0 is None:
    element_shape_except0 = None
  element_shape_except0 = _execute.make_shape(element_shape_except0, "element_shape_except0")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayConcat", handle=handle, flow_in=flow_in, dtype=dtype,
                             element_shape_except0=element_shape_except0,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape_except0",
              _op.get_attr("element_shape_except0"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayConcat", _inputs_flat, _attrs, _result)
  _result = _TensorArrayConcatOutput._make(_result)
  return _result

TensorArrayConcat = tf_export("raw_ops.TensorArrayConcat")(_ops.to_raw_op(tensor_array_concat))


def tensor_array_concat_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcat_dtype, element_shape_except0, name, ctx):
  raise RuntimeError("tensor_array_concat op does not support eager execution. Arg 'handle' is a ref.")
_TensorArrayConcatV2Output = collections.namedtuple(
    "TensorArrayConcatV2",
    ["value", "lengths"])


TV_TensorArrayConcatV2_dtype = TypeVar("TV_TensorArrayConcatV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_concat_v2(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcatV2_dtype, element_shape_except0=None, name=None):
  r"""Deprecated. Use TensorArrayConcatV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).

    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayConcatV2", name, handle, flow_in, "dtype", dtype,
        "element_shape_except0", element_shape_except0)
      _result = _TensorArrayConcatV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_concat_v2_eager_fallback(
          handle, flow_in, dtype=dtype,
          element_shape_except0=element_shape_except0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape_except0 is None:
    element_shape_except0 = None
  element_shape_except0 = _execute.make_shape(element_shape_except0, "element_shape_except0")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayConcatV2", handle=handle, flow_in=flow_in, dtype=dtype,
                               element_shape_except0=element_shape_except0,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape_except0",
              _op.get_attr("element_shape_except0"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayConcatV2", _inputs_flat, _attrs, _result)
  _result = _TensorArrayConcatV2Output._make(_result)
  return _result

TensorArrayConcatV2 = tf_export("raw_ops.TensorArrayConcatV2")(_ops.to_raw_op(tensor_array_concat_v2))


def tensor_array_concat_v2_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcatV2_dtype, element_shape_except0, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape_except0 is None:
    element_shape_except0 = None
  element_shape_except0 = _execute.make_shape(element_shape_except0, "element_shape_except0")
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = ("dtype", dtype, "element_shape_except0", element_shape_except0)
  _result = _execute.execute(b"TensorArrayConcatV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayConcatV2", _inputs_flat, _attrs, _result)
  _result = _TensorArrayConcatV2Output._make(_result)
  return _result

_TensorArrayConcatV3Output = collections.namedtuple(
    "TensorArrayConcatV3",
    ["value", "lengths"])


TV_TensorArrayConcatV3_dtype = TypeVar("TV_TensorArrayConcatV3_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_concat_v3(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcatV3_dtype, element_shape_except0=None, name=None):
  r"""Concat the elements from the TensorArray into value `value`.

  Takes `T` elements of shapes

    ```
    (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
    ```

  and concatenates them into a Tensor of shape:

    ```
    (n0 + n1 + ... + n(T-1) x d0 x d1 x ...)
    ```

  All elements must have the same shape (excepting the first dimension).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known,
      excluding the first dimension. Used to validate the shapes of
      TensorArray elements. If this shape is not fully specified, concatenating
      zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).

    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayConcatV3", name, handle, flow_in, "dtype", dtype,
        "element_shape_except0", element_shape_except0)
      _result = _TensorArrayConcatV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_concat_v3_eager_fallback(
          handle, flow_in, dtype=dtype,
          element_shape_except0=element_shape_except0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape_except0 is None:
    element_shape_except0 = None
  element_shape_except0 = _execute.make_shape(element_shape_except0, "element_shape_except0")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayConcatV3", handle=handle, flow_in=flow_in, dtype=dtype,
                               element_shape_except0=element_shape_except0,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape_except0",
              _op.get_attr("element_shape_except0"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayConcatV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayConcatV3Output._make(_result)
  return _result

TensorArrayConcatV3 = tf_export("raw_ops.TensorArrayConcatV3")(_ops.to_raw_op(tensor_array_concat_v3))


def tensor_array_concat_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayConcatV3_dtype, element_shape_except0, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape_except0 is None:
    element_shape_except0 = None
  element_shape_except0 = _execute.make_shape(element_shape_except0, "element_shape_except0")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = ("dtype", dtype, "element_shape_except0", element_shape_except0)
  _result = _execute.execute(b"TensorArrayConcatV3", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayConcatV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayConcatV3Output._make(_result)
  return _result


TV_TensorArrayGather_dtype = TypeVar("TV_TensorArrayGather_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_gather(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGather_dtype, element_shape=None, name=None) -> Annotated[Any, TV_TensorArrayGather_dtype]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    indices: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_gather op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGather", handle=handle, indices=indices, flow_in=flow_in,
                             dtype=dtype, element_shape=element_shape,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayGather = tf_export("raw_ops.TensorArrayGather")(_ops.to_raw_op(tensor_array_gather))


def tensor_array_gather_eager_fallback(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGather_dtype, element_shape, name, ctx) -> Annotated[Any, TV_TensorArrayGather_dtype]:
  raise RuntimeError("tensor_array_gather op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayGatherV2_dtype = TypeVar("TV_TensorArrayGatherV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_gather_v2(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGatherV2_dtype, element_shape=None, name=None) -> Annotated[Any, TV_TensorArrayGatherV2_dtype]:
  r"""Deprecated. Use TensorArrayGatherV3

  Args:
    handle: A `Tensor` of type `string`.
    indices: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayGatherV2", name, handle, indices, flow_in, "dtype",
        dtype, "element_shape", element_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_gather_v2_eager_fallback(
          handle, indices, flow_in, dtype=dtype, element_shape=element_shape,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGatherV2", handle=handle, indices=indices,
                               flow_in=flow_in, dtype=dtype,
                               element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayGatherV2 = tf_export("raw_ops.TensorArrayGatherV2")(_ops.to_raw_op(tensor_array_gather_v2))


def tensor_array_gather_v2_eager_fallback(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGatherV2_dtype, element_shape, name, ctx) -> Annotated[Any, TV_TensorArrayGatherV2_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, indices, flow_in]
  _attrs = ("dtype", dtype, "element_shape", element_shape)
  _result = _execute.execute(b"TensorArrayGatherV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayGatherV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayGatherV3_dtype = TypeVar("TV_TensorArrayGatherV3_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_gather_v3(handle: Annotated[Any, _atypes.Resource], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGatherV3_dtype, element_shape=None, name=None) -> Annotated[Any, TV_TensorArrayGatherV3_dtype]:
  r"""Gather specific elements from the TensorArray into output `value`.

  All elements selected by `indices` must have the same shape.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    indices: A `Tensor` of type `int32`.
      The locations in the TensorArray from which to read tensor elements.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayGatherV3", name, handle, indices, flow_in, "dtype",
        dtype, "element_shape", element_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_gather_v3_eager_fallback(
          handle, indices, flow_in, dtype=dtype, element_shape=element_shape,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGatherV3", handle=handle, indices=indices,
                               flow_in=flow_in, dtype=dtype,
                               element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGatherV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayGatherV3 = tf_export("raw_ops.TensorArrayGatherV3")(_ops.to_raw_op(tensor_array_gather_v3))


def tensor_array_gather_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], indices: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayGatherV3_dtype, element_shape, name, ctx) -> Annotated[Any, TV_TensorArrayGatherV3_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, indices, flow_in]
  _attrs = ("dtype", dtype, "element_shape", element_shape)
  _result = _execute.execute(b"TensorArrayGatherV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayGatherV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_array_grad(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], source: str, name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    source: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_grad op does not support eager execution. Arg 'grad_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  source = _execute.make_str(source, "source")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGrad", handle=handle, flow_in=flow_in, source=source,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("source", _op.get_attr("source"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayGrad = tf_export("raw_ops.TensorArrayGrad")(_ops.to_raw_op(tensor_array_grad))


def tensor_array_grad_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], source: str, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("tensor_array_grad op does not support eager execution. Arg 'grad_handle' is a ref.")

def tensor_array_grad_v2(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], source: str, name=None) -> Annotated[Any, _atypes.String]:
  r"""Deprecated. Use TensorArrayGradV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    source: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayGradV2", name, handle, flow_in, "source", source)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_grad_v2_eager_fallback(
          handle, flow_in, source=source, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  source = _execute.make_str(source, "source")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGradV2", handle=handle, flow_in=flow_in, source=source,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("source", _op.get_attr("source"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayGradV2 = tf_export("raw_ops.TensorArrayGradV2")(_ops.to_raw_op(tensor_array_grad_v2))


def tensor_array_grad_v2_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], source: str, name, ctx) -> Annotated[Any, _atypes.String]:
  source = _execute.make_str(source, "source")
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = ("source", source)
  _result = _execute.execute(b"TensorArrayGradV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TensorArrayGradV3Output = collections.namedtuple(
    "TensorArrayGradV3",
    ["grad_handle", "flow_out"])


def tensor_array_grad_v3(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], source: str, name=None):
  r"""Creates a TensorArray for storing the gradients of values in the given handle.

  If the given TensorArray gradient already exists, returns a reference to it.

  Locks the size of the original TensorArray by disabling its dynamic size flag.

  **A note about the input flow_in:**

  The handle flow_in forces the execution of the gradient lookup to occur
  only after certain other operations have occurred.  For example, when
  the forward TensorArray is dynamically sized, writes to this TensorArray
  may resize the object.  The gradient TensorArray is statically sized based
  on the size of the forward TensorArray when this operation executes.
  Furthermore, the size of the forward TensorArray is frozen by this call.
  As a result, the flow is used to ensure that the call to generate the gradient
  TensorArray only happens after all writes are executed.

  In the case of dynamically sized TensorArrays, gradient computation should
  only be performed on read operations that have themselves been chained via
  flow to occur only after all writes have executed. That way the final size
  of the forward TensorArray is known when this operation is called.

  **A note about the source attribute:**

  TensorArray gradient calls use an accumulator TensorArray object.  If
  multiple gradients are calculated and run in the same session, the multiple
  gradient nodes may accidentally flow through the same accumulator TensorArray.
  This double counts and generally breaks the TensorArray gradient flow.

  The solution is to identify which gradient call this particular
  TensorArray gradient is being called in.  This is performed by identifying
  a unique string (e.g. "gradients", "gradients_1", ...) from the input
  gradient Tensor's name.  This string is used as a suffix when creating
  the TensorArray gradient object here (the attribute `source`).

  The attribute `source` is added as a suffix to the forward TensorArray's
  name when performing the creation / lookup, so that each separate gradient
  calculation gets its own TensorArray accumulator.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to the forward TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    source: A `string`.
      The gradient source string, used to decide which gradient TensorArray
      to return.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (grad_handle, flow_out).

    grad_handle: A `Tensor` of type `resource`.
    flow_out: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayGradV3", name, handle, flow_in, "source", source)
      _result = _TensorArrayGradV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_grad_v3_eager_fallback(
          handle, flow_in, source=source, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  source = _execute.make_str(source, "source")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGradV3", handle=handle, flow_in=flow_in, source=source,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("source", _op.get_attr("source"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGradV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayGradV3Output._make(_result)
  return _result

TensorArrayGradV3 = tf_export("raw_ops.TensorArrayGradV3")(_ops.to_raw_op(tensor_array_grad_v3))


def tensor_array_grad_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], source: str, name, ctx):
  source = _execute.make_str(source, "source")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = ("source", source)
  _result = _execute.execute(b"TensorArrayGradV3", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayGradV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayGradV3Output._make(_result)
  return _result

_TensorArrayGradWithShapeOutput = collections.namedtuple(
    "TensorArrayGradWithShape",
    ["grad_handle", "flow_out"])


def tensor_array_grad_with_shape(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], shape_to_prepend: Annotated[Any, _atypes.Int32], source: str, name=None):
  r"""Creates a TensorArray for storing multiple gradients of values in the given handle.

  Similar to TensorArrayGradV3. However it creates an accumulator with an
  expanded shape compared to the input TensorArray whose gradient is being
  computed. This enables multiple gradients for the same TensorArray to be
  calculated using the same accumulator.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to the forward TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    shape_to_prepend: A `Tensor` of type `int32`.
      An int32 vector representing a shape. Elements in the gradient accumulator will
      have shape which is this shape_to_prepend value concatenated with shape of the
      elements in the TensorArray corresponding to the input handle.
    source: A `string`.
      The gradient source string, used to decide which gradient TensorArray
      to return.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (grad_handle, flow_out).

    grad_handle: A `Tensor` of type `resource`.
    flow_out: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayGradWithShape", name, handle, flow_in,
        shape_to_prepend, "source", source)
      _result = _TensorArrayGradWithShapeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_grad_with_shape_eager_fallback(
          handle, flow_in, shape_to_prepend, source=source, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  source = _execute.make_str(source, "source")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayGradWithShape", handle=handle, flow_in=flow_in,
                                    shape_to_prepend=shape_to_prepend,
                                    source=source, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("source", _op.get_attr("source"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayGradWithShape", _inputs_flat, _attrs, _result)
  _result = _TensorArrayGradWithShapeOutput._make(_result)
  return _result

TensorArrayGradWithShape = tf_export("raw_ops.TensorArrayGradWithShape")(_ops.to_raw_op(tensor_array_grad_with_shape))


def tensor_array_grad_with_shape_eager_fallback(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], shape_to_prepend: Annotated[Any, _atypes.Int32], source: str, name, ctx):
  source = _execute.make_str(source, "source")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  shape_to_prepend = _ops.convert_to_tensor(shape_to_prepend, _dtypes.int32)
  _inputs_flat = [handle, flow_in, shape_to_prepend]
  _attrs = ("source", source)
  _result = _execute.execute(b"TensorArrayGradWithShape", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayGradWithShape", _inputs_flat, _attrs, _result)
  _result = _TensorArrayGradWithShapeOutput._make(_result)
  return _result


TV_TensorArrayPack_dtype = TypeVar("TV_TensorArrayPack_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_pack(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayPack_dtype, element_shape=None, name=None) -> Annotated[Any, TV_TensorArrayPack_dtype]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_pack op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayPack", handle=handle, flow_in=flow_in, dtype=dtype,
                           element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayPack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayPack = tf_export("raw_ops.TensorArrayPack")(_ops.to_raw_op(tensor_array_pack))


def tensor_array_pack_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayPack_dtype, element_shape, name, ctx) -> Annotated[Any, TV_TensorArrayPack_dtype]:
  raise RuntimeError("tensor_array_pack op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayRead_dtype = TypeVar("TV_TensorArrayRead_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_read(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayRead_dtype, name=None) -> Annotated[Any, TV_TensorArrayRead_dtype]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_read op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayRead", handle=handle, index=index, flow_in=flow_in,
                           dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayRead", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayRead = tf_export("raw_ops.TensorArrayRead")(_ops.to_raw_op(tensor_array_read))


def tensor_array_read_eager_fallback(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayRead_dtype, name, ctx) -> Annotated[Any, TV_TensorArrayRead_dtype]:
  raise RuntimeError("tensor_array_read op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayReadV2_dtype = TypeVar("TV_TensorArrayReadV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_read_v2(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayReadV2_dtype, name=None) -> Annotated[Any, TV_TensorArrayReadV2_dtype]:
  r"""Deprecated. Use TensorArrayReadV3

  Args:
    handle: A `Tensor` of type `string`.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayReadV2", name, handle, index, flow_in, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_read_v2_eager_fallback(
          handle, index, flow_in, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayReadV2", handle=handle, index=index, flow_in=flow_in,
                             dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayReadV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayReadV2 = tf_export("raw_ops.TensorArrayReadV2")(_ops.to_raw_op(tensor_array_read_v2))


def tensor_array_read_v2_eager_fallback(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayReadV2_dtype, name, ctx) -> Annotated[Any, TV_TensorArrayReadV2_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, index, flow_in]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"TensorArrayReadV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayReadV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayReadV3_dtype = TypeVar("TV_TensorArrayReadV3_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_read_v3(handle: Annotated[Any, _atypes.Resource], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayReadV3_dtype, name=None) -> Annotated[Any, TV_TensorArrayReadV3_dtype]:
  r"""Read an element from the TensorArray into output `value`.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayReadV3", name, handle, index, flow_in, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_read_v3_eager_fallback(
          handle, index, flow_in, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayReadV3", handle=handle, index=index, flow_in=flow_in,
                             dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayReadV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayReadV3 = tf_export("raw_ops.TensorArrayReadV3")(_ops.to_raw_op(tensor_array_read_v3))


def tensor_array_read_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], index: Annotated[Any, _atypes.Int32], flow_in: Annotated[Any, _atypes.Float32], dtype: TV_TensorArrayReadV3_dtype, name, ctx) -> Annotated[Any, TV_TensorArrayReadV3_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, index, flow_in]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"TensorArrayReadV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayReadV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayScatter_T = TypeVar("TV_TensorArrayScatter_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_scatter(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatter_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    indices: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_scatter op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayScatter", handle=handle, indices=indices, value=value,
                              flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayScatter = tf_export("raw_ops.TensorArrayScatter")(_ops.to_raw_op(tensor_array_scatter))


def tensor_array_scatter_eager_fallback(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatter_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  raise RuntimeError("tensor_array_scatter op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayScatterV2_T = TypeVar("TV_TensorArrayScatterV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_scatter_v2(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatterV2_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Deprecated. Use TensorArrayScatterV3

  Args:
    handle: A `Tensor` of type `string`.
    indices: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayScatterV2", name, handle, indices, value, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_scatter_v2_eager_fallback(
          handle, indices, value, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayScatterV2", handle=handle, indices=indices, value=value,
                                flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayScatterV2 = tf_export("raw_ops.TensorArrayScatterV2")(_ops.to_raw_op(tensor_array_scatter_v2))


def tensor_array_scatter_v2_eager_fallback(handle: Annotated[Any, _atypes.String], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatterV2_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, indices, value, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArrayScatterV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayScatterV3_T = TypeVar("TV_TensorArrayScatterV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_scatter_v3(handle: Annotated[Any, _atypes.Resource], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatterV3_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Scatter the data from the input value into specific TensorArray elements.

  `indices` must be a vector, its length must match the first dim of `value`.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    indices: A `Tensor` of type `int32`.
      The locations at which to write the tensor elements.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayScatterV3", name, handle, indices, value, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_scatter_v3_eager_fallback(
          handle, indices, value, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayScatterV3", handle=handle, indices=indices, value=value,
                                flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayScatterV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayScatterV3 = tf_export("raw_ops.TensorArrayScatterV3")(_ops.to_raw_op(tensor_array_scatter_v3))


def tensor_array_scatter_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], indices: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayScatterV3_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, indices, value, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArrayScatterV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayScatterV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_array_size(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_size op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySize", handle=handle, flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySize = tf_export("raw_ops.TensorArraySize")(_ops.to_raw_op(tensor_array_size))


def tensor_array_size_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("tensor_array_size op does not support eager execution. Arg 'handle' is a ref.")

def tensor_array_size_v2(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Deprecated. Use TensorArraySizeV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArraySizeV2", name, handle, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_size_v2_eager_fallback(
          handle, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySizeV2", handle=handle, flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySizeV2 = tf_export("raw_ops.TensorArraySizeV2")(_ops.to_raw_op(tensor_array_size_v2))


def tensor_array_size_v2_eager_fallback(handle: Annotated[Any, _atypes.String], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = None
  _result = _execute.execute(b"TensorArraySizeV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArraySizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_array_size_v3(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Get the current size of the TensorArray.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArraySizeV3", name, handle, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_size_v3_eager_fallback(
          handle, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySizeV3", handle=handle, flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySizeV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySizeV3 = tf_export("raw_ops.TensorArraySizeV3")(_ops.to_raw_op(tensor_array_size_v3))


def tensor_array_size_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, flow_in]
  _attrs = None
  _result = _execute.execute(b"TensorArraySizeV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArraySizeV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArraySplit_T = TypeVar("TV_TensorArraySplit_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_split(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArraySplit_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    value: A `Tensor`.
    lengths: A `Tensor` of type `int64`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_split op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySplit", handle=handle, value=value, lengths=lengths,
                            flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySplit", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySplit = tf_export("raw_ops.TensorArraySplit")(_ops.to_raw_op(tensor_array_split))


def tensor_array_split_eager_fallback(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArraySplit_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  raise RuntimeError("tensor_array_split op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArraySplitV2_T = TypeVar("TV_TensorArraySplitV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_split_v2(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArraySplitV2_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Deprecated. Use TensorArraySplitV3

  Args:
    handle: A `Tensor` of type `string`.
    value: A `Tensor`.
    lengths: A `Tensor` of type `int64`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArraySplitV2", name, handle, value, lengths, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_split_v2_eager_fallback(
          handle, value, lengths, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySplitV2", handle=handle, value=value, lengths=lengths,
                              flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySplitV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySplitV2 = tf_export("raw_ops.TensorArraySplitV2")(_ops.to_raw_op(tensor_array_split_v2))


def tensor_array_split_v2_eager_fallback(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArraySplitV2_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  lengths = _ops.convert_to_tensor(lengths, _dtypes.int64)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, value, lengths, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArraySplitV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArraySplitV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArraySplitV3_T = TypeVar("TV_TensorArraySplitV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_split_v3(handle: Annotated[Any, _atypes.Resource], value: Annotated[Any, TV_TensorArraySplitV3_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Split the data from the input value into TensorArray elements.

  Assuming that `lengths` takes on values

    ```
    (n0, n1, ..., n(T-1))
    ```

  and that `value` has shape

    ```
    (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
    ```

  this splits values into a TensorArray with T tensors.

  TensorArray index t will be the subtensor of values with starting position

    ```
    (n0 + n1 + ... + n(t-1), 0, 0, ...)
    ```

  and having size

    ```
    nt x d0 x d1 x ...
    ```

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    lengths: A `Tensor` of type `int64`.
      The vector of lengths, how to split the rows of value into the
      TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArraySplitV3", name, handle, value, lengths, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_split_v3_eager_fallback(
          handle, value, lengths, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArraySplitV3", handle=handle, value=value, lengths=lengths,
                              flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArraySplitV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArraySplitV3 = tf_export("raw_ops.TensorArraySplitV3")(_ops.to_raw_op(tensor_array_split_v3))


def tensor_array_split_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], value: Annotated[Any, TV_TensorArraySplitV3_T], lengths: Annotated[Any, _atypes.Int64], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  lengths = _ops.convert_to_tensor(lengths, _dtypes.int64)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, value, lengths, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArraySplitV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArraySplitV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayUnpack_T = TypeVar("TV_TensorArrayUnpack_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_unpack(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArrayUnpack_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_unpack op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayUnpack", handle=handle, value=value, flow_in=flow_in,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayUnpack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayUnpack = tf_export("raw_ops.TensorArrayUnpack")(_ops.to_raw_op(tensor_array_unpack))


def tensor_array_unpack_eager_fallback(handle: Annotated[Any, _atypes.String], value: Annotated[Any, TV_TensorArrayUnpack_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  raise RuntimeError("tensor_array_unpack op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayV2_dtype = TypeVar("TV_TensorArrayV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_v2(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArrayV2_dtype, element_shape=None, dynamic_size:bool=False, clear_after_read:bool=True, tensor_array_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Deprecated. Use TensorArrayV3

  Args:
    size: A `Tensor` of type `int32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    dynamic_size: An optional `bool`. Defaults to `False`.
    clear_after_read: An optional `bool`. Defaults to `True`.
    tensor_array_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayV2", name, size, "dtype", dtype, "element_shape",
        element_shape, "dynamic_size", dynamic_size, "clear_after_read",
        clear_after_read, "tensor_array_name", tensor_array_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_v2_eager_fallback(
          size, dtype=dtype, element_shape=element_shape,
          dynamic_size=dynamic_size, clear_after_read=clear_after_read,
          tensor_array_name=tensor_array_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  if dynamic_size is None:
    dynamic_size = False
  dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
  if clear_after_read is None:
    clear_after_read = True
  clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
  if tensor_array_name is None:
    tensor_array_name = ""
  tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayV2", size=size, dtype=dtype, element_shape=element_shape,
                         dynamic_size=dynamic_size,
                         clear_after_read=clear_after_read,
                         tensor_array_name=tensor_array_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"), "dynamic_size",
              _op._get_attr_bool("dynamic_size"), "clear_after_read",
              _op._get_attr_bool("clear_after_read"), "tensor_array_name",
              _op.get_attr("tensor_array_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayV2 = tf_export("raw_ops.TensorArrayV2")(_ops.to_raw_op(tensor_array_v2))


def tensor_array_v2_eager_fallback(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArrayV2_dtype, element_shape, dynamic_size: bool, clear_after_read: bool, tensor_array_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  if dynamic_size is None:
    dynamic_size = False
  dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
  if clear_after_read is None:
    clear_after_read = True
  clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
  if tensor_array_name is None:
    tensor_array_name = ""
  tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [size]
  _attrs = ("dtype", dtype, "element_shape", element_shape, "dynamic_size",
  dynamic_size, "clear_after_read", clear_after_read, "tensor_array_name",
  tensor_array_name)
  _result = _execute.execute(b"TensorArrayV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TensorArrayV3Output = collections.namedtuple(
    "TensorArrayV3",
    ["handle", "flow"])


TV_TensorArrayV3_dtype = TypeVar("TV_TensorArrayV3_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_v3(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArrayV3_dtype, element_shape=None, dynamic_size:bool=False, clear_after_read:bool=True, identical_element_shapes:bool=False, tensor_array_name:str="", name=None):
  r"""An array of Tensors of given size.

  Write data via Write and read via Read or Pack.

  Args:
    size: A `Tensor` of type `int32`. The size of the array.
    dtype: A `tf.DType`. The type of the elements on the tensor_array.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    dynamic_size: An optional `bool`. Defaults to `False`.
      A boolean that determines whether writes to the TensorArray
      are allowed to grow the size.  By default, this is not allowed.
    clear_after_read: An optional `bool`. Defaults to `True`.
      If true (default), Tensors in the TensorArray are cleared
      after being read.  This disables multiple read semantics but allows early
      release of memory.
    identical_element_shapes: An optional `bool`. Defaults to `False`.
      If true (default is false), then all
      elements in the TensorArray will be expected to have identical shapes.
      This allows certain behaviors, like dynamically checking for
      consistent shapes on write, and being able to fill in properly
      shaped zero tensors on stack -- even if the element_shape attribute
      is not fully defined.
    tensor_array_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary tensor_array
      resource. Default value is the name of the 'TensorArray' op (which
      is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, flow).

    handle: A `Tensor` of type `resource`.
    flow: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayV3", name, size, "dtype", dtype, "element_shape",
        element_shape, "dynamic_size", dynamic_size, "clear_after_read",
        clear_after_read, "identical_element_shapes",
        identical_element_shapes, "tensor_array_name", tensor_array_name)
      _result = _TensorArrayV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_v3_eager_fallback(
          size, dtype=dtype, element_shape=element_shape,
          dynamic_size=dynamic_size, clear_after_read=clear_after_read,
          identical_element_shapes=identical_element_shapes,
          tensor_array_name=tensor_array_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  if dynamic_size is None:
    dynamic_size = False
  dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
  if clear_after_read is None:
    clear_after_read = True
  clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
  if identical_element_shapes is None:
    identical_element_shapes = False
  identical_element_shapes = _execute.make_bool(identical_element_shapes, "identical_element_shapes")
  if tensor_array_name is None:
    tensor_array_name = ""
  tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayV3", size=size, dtype=dtype, element_shape=element_shape,
                         dynamic_size=dynamic_size,
                         clear_after_read=clear_after_read,
                         identical_element_shapes=identical_element_shapes,
                         tensor_array_name=tensor_array_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "element_shape",
              _op.get_attr("element_shape"), "dynamic_size",
              _op._get_attr_bool("dynamic_size"), "clear_after_read",
              _op._get_attr_bool("clear_after_read"),
              "identical_element_shapes",
              _op._get_attr_bool("identical_element_shapes"),
              "tensor_array_name", _op.get_attr("tensor_array_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayV3Output._make(_result)
  return _result

TensorArrayV3 = tf_export("raw_ops.TensorArrayV3")(_ops.to_raw_op(tensor_array_v3))


def tensor_array_v3_eager_fallback(size: Annotated[Any, _atypes.Int32], dtype: TV_TensorArrayV3_dtype, element_shape, dynamic_size: bool, clear_after_read: bool, identical_element_shapes: bool, tensor_array_name: str, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  if dynamic_size is None:
    dynamic_size = False
  dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
  if clear_after_read is None:
    clear_after_read = True
  clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
  if identical_element_shapes is None:
    identical_element_shapes = False
  identical_element_shapes = _execute.make_bool(identical_element_shapes, "identical_element_shapes")
  if tensor_array_name is None:
    tensor_array_name = ""
  tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [size]
  _attrs = ("dtype", dtype, "element_shape", element_shape, "dynamic_size",
  dynamic_size, "clear_after_read", clear_after_read,
  "identical_element_shapes", identical_element_shapes, "tensor_array_name",
  tensor_array_name)
  _result = _execute.execute(b"TensorArrayV3", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayV3", _inputs_flat, _attrs, _result)
  _result = _TensorArrayV3Output._make(_result)
  return _result


TV_TensorArrayWrite_T = TypeVar("TV_TensorArrayWrite_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_write(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWrite_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    index: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("tensor_array_write op does not support eager execution. Arg 'handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayWrite", handle=handle, index=index, value=value,
                            flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayWrite", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayWrite = tf_export("raw_ops.TensorArrayWrite")(_ops.to_raw_op(tensor_array_write))


def tensor_array_write_eager_fallback(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWrite_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  raise RuntimeError("tensor_array_write op does not support eager execution. Arg 'handle' is a ref.")

TV_TensorArrayWriteV2_T = TypeVar("TV_TensorArrayWriteV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_write_v2(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWriteV2_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Deprecated. Use TensorArrayGradV3

  Args:
    handle: A `Tensor` of type `string`.
    index: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayWriteV2", name, handle, index, value, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_write_v2_eager_fallback(
          handle, index, value, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayWriteV2", handle=handle, index=index, value=value,
                              flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayWriteV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayWriteV2 = tf_export("raw_ops.TensorArrayWriteV2")(_ops.to_raw_op(tensor_array_write_v2))


def tensor_array_write_v2_eager_fallback(handle: Annotated[Any, _atypes.String], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWriteV2_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.string)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, index, value, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArrayWriteV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayWriteV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorArrayWriteV3_T = TypeVar("TV_TensorArrayWriteV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_array_write_v3(handle: Annotated[Any, _atypes.Resource], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWriteV3_T], flow_in: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Push an element onto the tensor_array.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    index: A `Tensor` of type `int32`.
      The position to write to inside the TensorArray.
    value: A `Tensor`. The tensor to write to the TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorArrayWriteV3", name, handle, index, value, flow_in)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_array_write_v3_eager_fallback(
          handle, index, value, flow_in, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorArrayWriteV3", handle=handle, index=index, value=value,
                              flow_in=flow_in, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorArrayWriteV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorArrayWriteV3 = tf_export("raw_ops.TensorArrayWriteV3")(_ops.to_raw_op(tensor_array_write_v3))


def tensor_array_write_v3_eager_fallback(handle: Annotated[Any, _atypes.Resource], index: Annotated[Any, _atypes.Int32], value: Annotated[Any, TV_TensorArrayWriteV3_T], flow_in: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
  _inputs_flat = [handle, index, value, flow_in]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorArrayWriteV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorArrayWriteV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def unstage(dtypes, capacity:int=0, memory_limit:int=0, container:str="", shared_name:str="", name=None):
  r"""Op is similar to a lightweight Dequeue.

  The basic functionality is similar to dequeue with many fewer
  capabilities and options.  This Op is optimized for performance.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Unstage", name, "capacity", capacity, "memory_limit",
        memory_limit, "dtypes", dtypes, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unstage_eager_fallback(
          capacity=capacity, memory_limit=memory_limit, dtypes=dtypes,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Unstage", dtypes=dtypes, capacity=capacity,
                   memory_limit=memory_limit, container=container,
                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("capacity", _op._get_attr_int("capacity"), "memory_limit",
              _op._get_attr_int("memory_limit"), "dtypes",
              _op.get_attr("dtypes"), "container", _op.get_attr("container"),
              "shared_name", _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Unstage", _inputs_flat, _attrs, _result)
  return _result

Unstage = tf_export("raw_ops.Unstage")(_ops.to_raw_op(unstage))


def unstage_eager_fallback(dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'unstage' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if capacity is None:
    capacity = 0
  capacity = _execute.make_int(capacity, "capacity")
  if memory_limit is None:
    memory_limit = 0
  memory_limit = _execute.make_int(memory_limit, "memory_limit")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("capacity", capacity, "memory_limit", memory_limit, "dtypes",
  dtypes, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"Unstage", len(dtypes), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Unstage", _inputs_flat, _attrs, _result)
  return _result

