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

TV_TPUPartitionedInput_T = TypeVar("TV_TPUPartitionedInput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_partitioned_input(inputs: Annotated[List[Any], TV_TPUPartitionedInput_T], partition_dim:int=0, name=None) -> Annotated[Any, TV_TPUPartitionedInput_T]:
  r"""An op that groups a list of partitioned inputs together. This op

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      A list of partitioned inputs which must have the same shape.
    partition_dim: An optional `int`. Defaults to `0`.
      An integer describles which dimension is partitioned. -1 means
      those inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedInput", name, inputs, "partition_dim",
        partition_dim)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_input_eager_fallback(
          inputs, partition_dim=partition_dim, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedInput", inputs=inputs, partition_dim=partition_dim,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "partition_dim", _op._get_attr_int("partition_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUPartitionedInput = tf_export("raw_ops.TPUPartitionedInput")(_ops.to_raw_op(tpu_partitioned_input))


def tpu_partitioned_input_eager_fallback(inputs: Annotated[List[Any], TV_TPUPartitionedInput_T], partition_dim: int, name, ctx) -> Annotated[Any, TV_TPUPartitionedInput_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T, "partition_dim", partition_dim)
  _result = _execute.execute(b"TPUPartitionedInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TPUPartitionedInputV2_T = TypeVar("TV_TPUPartitionedInputV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_partitioned_input_v2(inputs: Annotated[List[Any], TV_TPUPartitionedInputV2_T], partition_dims, is_packed:bool=False, name=None) -> Annotated[Any, TV_TPUPartitionedInputV2_T]:
  r"""An op that groups a list of partitioned inputs together. Supports ND sharding.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      A list of partitioned inputs which must have the same shape.
    partition_dims: A list of `ints`.
      A list of integers describing how each dimension is partitioned. Emptiness
      indicates the inputs are replicated.
    is_packed: An optional `bool`. Defaults to `False`.
      Indicates whether the input is a packed resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedInputV2", name, inputs, "partition_dims",
        partition_dims, "is_packed", is_packed)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_input_v2_eager_fallback(
          inputs, partition_dims=partition_dims, is_packed=is_packed,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input_v2' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(partition_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'partition_dims' argument to "
        "'tpu_partitioned_input_v2' Op, not %r." % partition_dims)
  partition_dims = [_execute.make_int(_i, "partition_dims") for _i in partition_dims]
  if is_packed is None:
    is_packed = False
  is_packed = _execute.make_bool(is_packed, "is_packed")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedInputV2", inputs=inputs, partition_dims=partition_dims,
                                 is_packed=is_packed, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "partition_dims", _op.get_attr("partition_dims"), "is_packed",
              _op._get_attr_bool("is_packed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUPartitionedInputV2 = tf_export("raw_ops.TPUPartitionedInputV2")(_ops.to_raw_op(tpu_partitioned_input_v2))


def tpu_partitioned_input_v2_eager_fallback(inputs: Annotated[List[Any], TV_TPUPartitionedInputV2_T], partition_dims, is_packed: bool, name, ctx) -> Annotated[Any, TV_TPUPartitionedInputV2_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input_v2' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(partition_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'partition_dims' argument to "
        "'tpu_partitioned_input_v2' Op, not %r." % partition_dims)
  partition_dims = [_execute.make_int(_i, "partition_dims") for _i in partition_dims]
  if is_packed is None:
    is_packed = False
  is_packed = _execute.make_bool(is_packed, "is_packed")
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T, "partition_dims", partition_dims,
  "is_packed", is_packed)
  _result = _execute.execute(b"TPUPartitionedInputV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TPUPartitionedOutput_T = TypeVar("TV_TPUPartitionedOutput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_partitioned_output(inputs: Annotated[Any, TV_TPUPartitionedOutput_T], num_splits: int, partition_dim:int=0, name=None):
  r"""An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned

  outputs outside the XLA computation.

  Args:
    inputs: A `Tensor`.
      A tensor which represents the full shape of partitioned tensors.
    num_splits: An `int` that is `>= 1`.
    partition_dim: An optional `int`. Defaults to `0`.
      An integer describles which dimension is partitioned.
    name: A name for the operation (optional).

  Returns:
    A list of `num_splits` `Tensor` objects with the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedOutput", name, inputs, "num_splits", num_splits,
        "partition_dim", partition_dim)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_output_eager_fallback(
          inputs, num_splits=num_splits, partition_dim=partition_dim,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_splits = _execute.make_int(num_splits, "num_splits")
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedOutput", inputs=inputs, num_splits=num_splits,
                                partition_dim=partition_dim, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_splits",
              _op._get_attr_int("num_splits"), "partition_dim",
              _op._get_attr_int("partition_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedOutput", _inputs_flat, _attrs, _result)
  return _result

TPUPartitionedOutput = tf_export("raw_ops.TPUPartitionedOutput")(_ops.to_raw_op(tpu_partitioned_output))


def tpu_partitioned_output_eager_fallback(inputs: Annotated[Any, TV_TPUPartitionedOutput_T], num_splits: int, partition_dim: int, name, ctx):
  num_splits = _execute.make_int(num_splits, "num_splits")
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [])
  _inputs_flat = [inputs]
  _attrs = ("T", _attr_T, "num_splits", num_splits, "partition_dim",
  partition_dim)
  _result = _execute.execute(b"TPUPartitionedOutput", num_splits,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedOutput", _inputs_flat, _attrs, _result)
  return _result


TV_TPUPartitionedOutputV2_T = TypeVar("TV_TPUPartitionedOutputV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_partitioned_output_v2(inputs: Annotated[Any, TV_TPUPartitionedOutputV2_T], num_splits: int, partition_dims, name=None):
  r"""An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned

  outputs outside the XLA computation. Supports ND sharding.

  Args:
    inputs: A `Tensor`.
      A tensor which represents the full shape of partitioned tensors.
    num_splits: An `int` that is `>= 1`.
    partition_dims: A list of `ints`.
      A list of integers describing how each dimension is partitioned. Emptiness
      indicates the inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A list of `num_splits` `Tensor` objects with the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedOutputV2", name, inputs, "num_splits",
        num_splits, "partition_dims", partition_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_output_v2_eager_fallback(
          inputs, num_splits=num_splits, partition_dims=partition_dims,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_splits = _execute.make_int(num_splits, "num_splits")
  if not isinstance(partition_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'partition_dims' argument to "
        "'tpu_partitioned_output_v2' Op, not %r." % partition_dims)
  partition_dims = [_execute.make_int(_i, "partition_dims") for _i in partition_dims]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedOutputV2", inputs=inputs, num_splits=num_splits,
                                  partition_dims=partition_dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_splits",
              _op._get_attr_int("num_splits"), "partition_dims",
              _op.get_attr("partition_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedOutputV2", _inputs_flat, _attrs, _result)
  return _result

TPUPartitionedOutputV2 = tf_export("raw_ops.TPUPartitionedOutputV2")(_ops.to_raw_op(tpu_partitioned_output_v2))


def tpu_partitioned_output_v2_eager_fallback(inputs: Annotated[Any, TV_TPUPartitionedOutputV2_T], num_splits: int, partition_dims, name, ctx):
  num_splits = _execute.make_int(num_splits, "num_splits")
  if not isinstance(partition_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'partition_dims' argument to "
        "'tpu_partitioned_output_v2' Op, not %r." % partition_dims)
  partition_dims = [_execute.make_int(_i, "partition_dims") for _i in partition_dims]
  _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [])
  _inputs_flat = [inputs]
  _attrs = ("T", _attr_T, "num_splits", num_splits, "partition_dims",
  partition_dims)
  _result = _execute.execute(b"TPUPartitionedOutputV2", num_splits,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedOutputV2", _inputs_flat, _attrs, _result)
  return _result

