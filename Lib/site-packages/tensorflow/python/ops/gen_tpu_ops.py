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

TV_AllToAll_T = TypeVar("TV_AllToAll_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def all_to_all(input: Annotated[Any, TV_AllToAll_T], group_assignment: Annotated[Any, _atypes.Int32], concat_dimension: int, split_dimension: int, split_count: int, name=None) -> Annotated[Any, TV_AllToAll_T]:
  r"""An Op to exchange data across TPU replicas.

  On each replica, the input is split into `split_count` blocks along
  `split_dimension` and send to the other replicas given group_assignment. After
  receiving `split_count` - 1 blocks from other replicas, we concatenate the
  blocks along `concat_dimension` as the output.

  For example, suppose there are 2 TPU replicas:
  replica 0 receives input: `[[A, B]]`
  replica 1 receives input: `[[C, D]]`

  group_assignment=`[[0, 1]]`
  concat_dimension=0
  split_dimension=1
  split_count=2

  replica 0's output: `[[A], [C]]`
  replica 1's output: `[[B], [D]]`

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      The local input to the sum.
    group_assignment: A `Tensor` of type `int32`. An int32 tensor with shape
      [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
      replica ids in the ith subgroup.
    concat_dimension: An `int`. The dimension number to concatenate.
    split_dimension: An `int`. The dimension number to split.
    split_count: An `int`.
      The number of splits, this number must equal to the sub-group
      size(group_assignment.get_shape()[1])
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AllToAll", name, input, group_assignment, "concat_dimension",
        concat_dimension, "split_dimension", split_dimension, "split_count",
        split_count)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return all_to_all_eager_fallback(
          input, group_assignment, concat_dimension=concat_dimension,
          split_dimension=split_dimension, split_count=split_count, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  concat_dimension = _execute.make_int(concat_dimension, "concat_dimension")
  split_dimension = _execute.make_int(split_dimension, "split_dimension")
  split_count = _execute.make_int(split_count, "split_count")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AllToAll", input=input, group_assignment=group_assignment,
                    concat_dimension=concat_dimension,
                    split_dimension=split_dimension, split_count=split_count,
                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "concat_dimension",
              _op._get_attr_int("concat_dimension"), "split_dimension",
              _op._get_attr_int("split_dimension"), "split_count",
              _op._get_attr_int("split_count"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AllToAll", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AllToAll = tf_export("raw_ops.AllToAll")(_ops.to_raw_op(all_to_all))


def all_to_all_eager_fallback(input: Annotated[Any, TV_AllToAll_T], group_assignment: Annotated[Any, _atypes.Int32], concat_dimension: int, split_dimension: int, split_count: int, name, ctx) -> Annotated[Any, TV_AllToAll_T]:
  concat_dimension = _execute.make_int(concat_dimension, "concat_dimension")
  split_dimension = _execute.make_int(split_dimension, "split_dimension")
  split_count = _execute.make_int(split_count, "split_count")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, group_assignment]
  _attrs = ("T", _attr_T, "concat_dimension", concat_dimension,
  "split_dimension", split_dimension, "split_count", split_count)
  _result = _execute.execute(b"AllToAll", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AllToAll", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AssignVariableXlaConcatND_T = TypeVar("TV_AssignVariableXlaConcatND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def assign_variable_xla_concat_nd(resource: Annotated[Any, _atypes.Resource], inputs: Annotated[List[Any], TV_AssignVariableXlaConcatND_T], num_concats, paddings=[], name=None):
  r"""Concats input tensor across all dimensions.

  An op which merges slices the input tensor based on the given num_splits
  attribute, strips paddings optionally, and writes the merged tensor without
  paddings to the resource variable.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1],
   [4, 5]]
  [[2, 3],
   [6, 7]]
  [[8, 9],
   [12, 13]]
  [[10, 11],
   [14, 15]]
  ```
  `num_splits`:
  ```
  [2, 2]
  ```
  and `paddings`:
  ```
  [1, 1]
  ```
  the expected `outputs` is:
  ```
  [[0, 1, 2],
   [4, 5, 6],
   [8, 9, 10]]
  ```

  Args:
    resource: A `Tensor` of type `resource`.
      Resource variable for concatenated input tensors across all dimensions.
    inputs: A list of at least 1 `Tensor` objects with the same type.
      Input tensor slices in row-major order to merge across all dimensions. All
      inputs must have the same shape.
    num_concats: A list of `ints`. Number of ways to merge per dimension.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension to strip from the final merged
      tensor. These paddings must not exceed the dimension size of the merged result
      prior to stripping paddings.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AssignVariableXlaConcatND", name, resource, inputs,
        "num_concats", num_concats, "paddings", paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return assign_variable_xla_concat_nd_eager_fallback(
          resource, inputs, num_concats=num_concats, paddings=paddings,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(num_concats, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_concats' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % num_concats)
  num_concats = [_execute.make_int(_i, "num_concats") for _i in num_concats]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AssignVariableXlaConcatND", resource=resource, inputs=inputs,
                                     num_concats=num_concats,
                                     paddings=paddings, name=name)
  return _op
AssignVariableXlaConcatND = tf_export("raw_ops.AssignVariableXlaConcatND")(_ops.to_raw_op(assign_variable_xla_concat_nd))


def assign_variable_xla_concat_nd_eager_fallback(resource: Annotated[Any, _atypes.Resource], inputs: Annotated[List[Any], TV_AssignVariableXlaConcatND_T], num_concats, paddings, name, ctx):
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(num_concats, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_concats' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % num_concats)
  num_concats = [_execute.make_int(_i, "num_concats") for _i in num_concats]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'assign_variable_xla_concat_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource] + list(inputs)
  _attrs = ("T", _attr_T, "N", _attr_N, "num_concats", num_concats,
  "paddings", paddings)
  _result = _execute.execute(b"AssignVariableXlaConcatND", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_CollectivePermute_T = TypeVar("TV_CollectivePermute_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def collective_permute(input: Annotated[Any, TV_CollectivePermute_T], source_target_pairs: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_CollectivePermute_T]:
  r"""An Op to permute tensors across replicated TPU instances.

  Each instance supplies its own input.

  For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
  source_target_pairs=`[[0,1],[1,2],[2,3],[3,0]]` gets the outputs:
  `[D, A, B, C]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The local input to be permuted. Currently only supports float and
      bfloat16.
    source_target_pairs: A `Tensor` of type `int32`.
      A tensor with shape [num_pairs, 2].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollectivePermute", name, input, source_target_pairs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return collective_permute_eager_fallback(
          input, source_target_pairs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollectivePermute", input=input,
                             source_target_pairs=source_target_pairs,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectivePermute", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollectivePermute = tf_export("raw_ops.CollectivePermute")(_ops.to_raw_op(collective_permute))


def collective_permute_eager_fallback(input: Annotated[Any, TV_CollectivePermute_T], source_target_pairs: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_CollectivePermute_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  source_target_pairs = _ops.convert_to_tensor(source_target_pairs, _dtypes.int32)
  _inputs_flat = [input, source_target_pairs]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"CollectivePermute", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollectivePermute", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def configure_distributed_tpu(embedding_config:str="", tpu_embedding_config:str="", is_global_init:bool=False, enable_whole_mesh_compilations:bool=False, compilation_failure_closes_chips:bool=True, tpu_cancellation_closes_chips:int=0, name=None) -> Annotated[Any, _atypes.String]:
  r"""Sets up the centralized structures for a distributed TPU system.

  Args:
    embedding_config: An optional `string`. Defaults to `""`.
      Reserved. Do not use.
    tpu_embedding_config: An optional `string`. Defaults to `""`.
      Serialized tensorflow.tpu.TPUEmbeddingConfiguration that
      describes the embedding lookups of the program.
    is_global_init: An optional `bool`. Defaults to `False`.
      Reserved. Do not use.
    enable_whole_mesh_compilations: An optional `bool`. Defaults to `False`.
    compilation_failure_closes_chips: An optional `bool`. Defaults to `True`.
    tpu_cancellation_closes_chips: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureDistributedTPU", name, "embedding_config",
        embedding_config, "tpu_embedding_config", tpu_embedding_config,
        "is_global_init", is_global_init, "enable_whole_mesh_compilations",
        enable_whole_mesh_compilations, "compilation_failure_closes_chips",
        compilation_failure_closes_chips, "tpu_cancellation_closes_chips",
        tpu_cancellation_closes_chips)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return configure_distributed_tpu_eager_fallback(
          embedding_config=embedding_config,
          tpu_embedding_config=tpu_embedding_config,
          is_global_init=is_global_init,
          enable_whole_mesh_compilations=enable_whole_mesh_compilations,
          compilation_failure_closes_chips=compilation_failure_closes_chips,
          tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if embedding_config is None:
    embedding_config = ""
  embedding_config = _execute.make_str(embedding_config, "embedding_config")
  if tpu_embedding_config is None:
    tpu_embedding_config = ""
  tpu_embedding_config = _execute.make_str(tpu_embedding_config, "tpu_embedding_config")
  if is_global_init is None:
    is_global_init = False
  is_global_init = _execute.make_bool(is_global_init, "is_global_init")
  if enable_whole_mesh_compilations is None:
    enable_whole_mesh_compilations = False
  enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, "enable_whole_mesh_compilations")
  if compilation_failure_closes_chips is None:
    compilation_failure_closes_chips = True
  compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, "compilation_failure_closes_chips")
  if tpu_cancellation_closes_chips is None:
    tpu_cancellation_closes_chips = 0
  tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, "tpu_cancellation_closes_chips")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureDistributedTPU", embedding_config=embedding_config,
                                   tpu_embedding_config=tpu_embedding_config,
                                   is_global_init=is_global_init,
                                   enable_whole_mesh_compilations=enable_whole_mesh_compilations,
                                   compilation_failure_closes_chips=compilation_failure_closes_chips,
                                   tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("embedding_config", _op.get_attr("embedding_config"),
              "tpu_embedding_config", _op.get_attr("tpu_embedding_config"),
              "is_global_init", _op._get_attr_bool("is_global_init"),
              "enable_whole_mesh_compilations",
              _op._get_attr_bool("enable_whole_mesh_compilations"),
              "compilation_failure_closes_chips",
              _op._get_attr_bool("compilation_failure_closes_chips"),
              "tpu_cancellation_closes_chips",
              _op._get_attr_int("tpu_cancellation_closes_chips"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConfigureDistributedTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConfigureDistributedTPU = tf_export("raw_ops.ConfigureDistributedTPU")(_ops.to_raw_op(configure_distributed_tpu))


def configure_distributed_tpu_eager_fallback(embedding_config: str, tpu_embedding_config: str, is_global_init: bool, enable_whole_mesh_compilations: bool, compilation_failure_closes_chips: bool, tpu_cancellation_closes_chips: int, name, ctx) -> Annotated[Any, _atypes.String]:
  if embedding_config is None:
    embedding_config = ""
  embedding_config = _execute.make_str(embedding_config, "embedding_config")
  if tpu_embedding_config is None:
    tpu_embedding_config = ""
  tpu_embedding_config = _execute.make_str(tpu_embedding_config, "tpu_embedding_config")
  if is_global_init is None:
    is_global_init = False
  is_global_init = _execute.make_bool(is_global_init, "is_global_init")
  if enable_whole_mesh_compilations is None:
    enable_whole_mesh_compilations = False
  enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, "enable_whole_mesh_compilations")
  if compilation_failure_closes_chips is None:
    compilation_failure_closes_chips = True
  compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, "compilation_failure_closes_chips")
  if tpu_cancellation_closes_chips is None:
    tpu_cancellation_closes_chips = 0
  tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, "tpu_cancellation_closes_chips")
  _inputs_flat = []
  _attrs = ("embedding_config", embedding_config, "tpu_embedding_config",
  tpu_embedding_config, "is_global_init", is_global_init,
  "enable_whole_mesh_compilations", enable_whole_mesh_compilations,
  "compilation_failure_closes_chips", compilation_failure_closes_chips,
  "tpu_cancellation_closes_chips", tpu_cancellation_closes_chips)
  _result = _execute.execute(b"ConfigureDistributedTPU", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConfigureDistributedTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def configure_tpu_embedding(config: str, name=None):
  r"""Sets up TPUEmbedding in a distributed TPU system.

  Args:
    config: A `string`.
      Serialized tensorflow.tpu.TPUEmbeddingConfiguration that
      describes the embedding lookups of the program.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureTPUEmbedding", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return configure_tpu_embedding_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureTPUEmbedding", config=config, name=name)
  return _op
ConfigureTPUEmbedding = tf_export("raw_ops.ConfigureTPUEmbedding")(_ops.to_raw_op(configure_tpu_embedding))


def configure_tpu_embedding_eager_fallback(config: str, name, ctx):
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"ConfigureTPUEmbedding", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_CrossReplicaSum_T = TypeVar("TV_CrossReplicaSum_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.UInt32)

def cross_replica_sum(input: Annotated[Any, TV_CrossReplicaSum_T], group_assignment: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_CrossReplicaSum_T]:
  r"""An Op to sum inputs across replicated TPU instances.

  Each instance supplies its own input.

  For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
  Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
  and `B, D, F, H` as group 1. Thus we get the outputs:
  `[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `uint32`.
      The local input to the sum.
    group_assignment: A `Tensor` of type `int32`. An int32 tensor with shape
      [num_groups, num_replicas_per_group]. `group_assignment[i]` represents the
      replica ids in the ith subgroup.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CrossReplicaSum", name, input, group_assignment)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cross_replica_sum_eager_fallback(
          input, group_assignment, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CrossReplicaSum", input=input, group_assignment=group_assignment,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CrossReplicaSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CrossReplicaSum = tf_export("raw_ops.CrossReplicaSum")(_ops.to_raw_op(cross_replica_sum))


def cross_replica_sum_eager_fallback(input: Annotated[Any, TV_CrossReplicaSum_T], group_assignment: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_CrossReplicaSum_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint32, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, group_assignment]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"CrossReplicaSum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CrossReplicaSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1 = TypeVar("TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1", _atypes.Int32, _atypes.Int64)
TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2 = TypeVar("TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2", _atypes.Int32, _atypes.Int64)
TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3 = TypeVar("TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3", _atypes.Float32, _atypes.Float64)

def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: Annotated[Any, _atypes.Int32], combiners=[], name=None):
  r"""Eases the porting of code that uses tf.nn.embedding_lookup_sparse().

  embedding_indices[i] and aggregation_weights[i] correspond
  to the ith feature.

  The tensors at corresponding positions in the three input lists (sample_indices,
  embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
  with dim_size() equal to the total number of lookups into the table described by
  the corresponding feature.

  Args:
    sample_indices_or_row_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 2 Tensors specifying the training example to which the
      corresponding embedding_indices and aggregation_weights values belong.
      If the size of its first dimension is 0, we assume each embedding_indices
      belongs to a different sample. Both int32 and int64 are allowed and will
      be converted to int32 internally.

      Or a list of rank 1 Tensors specifying the row splits for splitting
      embedding_indices and aggregation_weights into rows. It corresponds to
      ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. When
      enqueuing N-D ragged tensor, only the last dimension is allowed to be ragged.
      the row splits is 1-D dense tensor. When empty, we assume a dense tensor is
      passed to the op Both int32 and int64 are allowed and will be converted to
      int32 internally.
    embedding_indices: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding
      tables. Both int32 and int64 are allowed and will be converted to
      int32 internally.
    aggregation_weights: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per training
      example aggregation weights. Both float32 and float64 are allowed and will
      be converted to float32 internally.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    device_ordinal: A `Tensor` of type `int32`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch", name,
        sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, device_ordinal, "combiners", combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(
          sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal,
          combiners=combiners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch", sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                          embedding_indices=embedding_indices,
                                                          aggregation_weights=aggregation_weights,
                                                          mode_override=mode_override,
                                                          device_ordinal=device_ordinal,
                                                          combiners=combiners,
                                                          name=name)
  return _op
DynamicEnqueueTPUEmbeddingArbitraryTensorBatch = tf_export("raw_ops.DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")(_ops.to_raw_op(dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch))


def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: Annotated[Any, _atypes.Int32], combiners, name, ctx):
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices_or_row_splits = _execute.args_to_matching_eager(list(sample_indices_or_row_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = list(sample_indices_or_row_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override, device_ordinal]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "combiners", combiners)
  _result = _execute.execute(b"DynamicEnqueueTPUEmbeddingArbitraryTensorBatch",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T1 = TypeVar("TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T1", _atypes.Int32, _atypes.Int64)
TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T2 = TypeVar("TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T2", _atypes.Int32, _atypes.Int64)
TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T3 = TypeVar("TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T3", _atypes.Float32, _atypes.Float64)

def dynamic_enqueue_tpu_embedding_ragged_tensor_batch(sample_splits: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: Annotated[Any, _atypes.Int32], table_ids, combiners=[], max_sequence_lengths=[], num_features=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: A `Tensor` of type `int32`.
    table_ids: A list of `ints`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    max_sequence_lengths: An optional list of `ints`. Defaults to `[]`.
    num_features: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DynamicEnqueueTPUEmbeddingRaggedTensorBatch", name,
        sample_splits, embedding_indices, aggregation_weights, mode_override,
        device_ordinal, "combiners", combiners, "table_ids", table_ids,
        "max_sequence_lengths", max_sequence_lengths, "num_features",
        num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dynamic_enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(
          sample_splits, embedding_indices, aggregation_weights,
          mode_override, device_ordinal, combiners=combiners,
          table_ids=table_ids, max_sequence_lengths=max_sequence_lengths,
          num_features=num_features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DynamicEnqueueTPUEmbeddingRaggedTensorBatch", sample_splits=sample_splits,
                                                       embedding_indices=embedding_indices,
                                                       aggregation_weights=aggregation_weights,
                                                       mode_override=mode_override,
                                                       device_ordinal=device_ordinal,
                                                       table_ids=table_ids,
                                                       combiners=combiners,
                                                       max_sequence_lengths=max_sequence_lengths,
                                                       num_features=num_features,
                                                       name=name)
  return _op
DynamicEnqueueTPUEmbeddingRaggedTensorBatch = tf_export("raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch")(_ops.to_raw_op(dynamic_enqueue_tpu_embedding_ragged_tensor_batch))


def dynamic_enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(sample_splits: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_DynamicEnqueueTPUEmbeddingRaggedTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: Annotated[Any, _atypes.Int32], table_ids, combiners, max_sequence_lengths, num_features, name, ctx):
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'dynamic_enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _attr_T1, sample_splits = _execute.args_to_matching_eager(list(sample_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = list(sample_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override, device_ordinal]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "combiners", combiners, "table_ids", table_ids, "max_sequence_lengths",
  max_sequence_lengths, "num_features", num_features)
  _result = _execute.execute(b"DynamicEnqueueTPUEmbeddingRaggedTensorBatch",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T1 = TypeVar("TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T1", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T2 = TypeVar("TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T2", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T3 = TypeVar("TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T3", _atypes.Float32, _atypes.Float64)

def enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal:int=-1, combiners=[], name=None):
  r"""Eases the porting of code that uses tf.nn.embedding_lookup_sparse().

  embedding_indices[i] and aggregation_weights[i] correspond
  to the ith feature.

  The tensors at corresponding positions in the three input lists (sample_indices,
  embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
  with dim_size() equal to the total number of lookups into the table described by
  the corresponding feature.

  Args:
    sample_indices_or_row_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 2 Tensors specifying the training example to which the
      corresponding embedding_indices and aggregation_weights values belong.
      If the size of its first dimension is 0, we assume each embedding_indices
      belongs to a different sample. Both int32 and int64 are allowed and will
      be converted to int32 internally.

      Or a list of rank 1 Tensors specifying the row splits for splitting
      embedding_indices and aggregation_weights into rows. It corresponds to
      ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. When
      enqueuing N-D ragged tensor, only the last dimension is allowed to be ragged.
      the row splits is 1-D dense tensor. When empty, we assume a dense tensor is
      passed to the op Both int32 and int64 are allowed and will be converted to
      int32 internally.
    embedding_indices: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding
      tables. Both int32 and int64 are allowed and will be converted to
      int32 internally.
    aggregation_weights: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per training
      example aggregation weights. Both float32 and float64 are allowed and will
      be converted to float32 internally.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingArbitraryTensorBatch", name,
        sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, "device_ordinal", device_ordinal, "combiners",
        combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(
          sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal=device_ordinal,
          combiners=combiners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingArbitraryTensorBatch", sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                   embedding_indices=embedding_indices,
                                                   aggregation_weights=aggregation_weights,
                                                   mode_override=mode_override,
                                                   device_ordinal=device_ordinal,
                                                   combiners=combiners,
                                                   name=name)
  return _op
EnqueueTPUEmbeddingArbitraryTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingArbitraryTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_arbitrary_tensor_batch))


def enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingArbitraryTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: int, combiners, name, ctx):
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices_or_row_splits = _execute.args_to_matching_eager(list(sample_indices_or_row_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices_or_row_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners)
  _result = _execute.execute(b"EnqueueTPUEmbeddingArbitraryTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def enqueue_tpu_embedding_integer_batch(batch: Annotated[List[Any], _atypes.Int32], mode_override: Annotated[Any, _atypes.String], device_ordinal:int=-1, name=None):
  r"""An op that enqueues a list of input batch tensors to TPUEmbedding.

  Args:
    batch: A list of at least 1 `Tensor` objects with type `int32`.
      A list of 1D tensors, one for each embedding table, containing the
      indices into the tables.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingIntegerBatch", name, batch, mode_override,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return enqueue_tpu_embedding_integer_batch_eager_fallback(
          batch, mode_override, device_ordinal=device_ordinal, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_integer_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingIntegerBatch", batch=batch,
                                           mode_override=mode_override,
                                           device_ordinal=device_ordinal,
                                           name=name)
  return _op
EnqueueTPUEmbeddingIntegerBatch = tf_export("raw_ops.EnqueueTPUEmbeddingIntegerBatch")(_ops.to_raw_op(enqueue_tpu_embedding_integer_batch))


def enqueue_tpu_embedding_integer_batch_eager_fallback(batch: Annotated[List[Any], _atypes.Int32], mode_override: Annotated[Any, _atypes.String], device_ordinal: int, name, ctx):
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_integer_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  batch = _ops.convert_n_to_tensor(batch, _dtypes.int32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(batch) + [mode_override]
  _attrs = ("N", _attr_N, "device_ordinal", device_ordinal)
  _result = _execute.execute(b"EnqueueTPUEmbeddingIntegerBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_EnqueueTPUEmbeddingRaggedTensorBatch_T1 = TypeVar("TV_EnqueueTPUEmbeddingRaggedTensorBatch_T1", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingRaggedTensorBatch_T2 = TypeVar("TV_EnqueueTPUEmbeddingRaggedTensorBatch_T2", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingRaggedTensorBatch_T3 = TypeVar("TV_EnqueueTPUEmbeddingRaggedTensorBatch_T3", _atypes.Float32, _atypes.Float64)

def enqueue_tpu_embedding_ragged_tensor_batch(sample_splits: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], table_ids, device_ordinal:int=-1, combiners=[], max_sequence_lengths=[], num_features=[], name=None):
  r"""Eases the porting of code that uses tf.nn.embedding_lookup().

  sample_splits[i], embedding_indices[i] and aggregation_weights[i] correspond
  to the ith feature. table_ids[i] indicates which embedding table to look up ith
  feature.

  The tensors at corresponding positions in two of the input lists,
  embedding_indices and aggregation_weights, must have the same shape, i.e. rank 1
  with dim_size() equal to the total number of lookups into the table described by
  the corresponding feature.

  Args:
    sample_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors specifying the break points for splitting
      embedding_indices and aggregation_weights into rows.
      It corresponds to ids.row_splits in embedding_lookup(), when ids is a
      RaggedTensor.
    embedding_indices: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding tables.
      It corresponds to ids.values in embedding_lookup(), when ids is a RaggedTensor.
    aggregation_weights: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per training example
      aggregation weights. It corresponds to the values field of a RaggedTensor
      with the same row_splits as ids in embedding_lookup(), when ids is a
      RaggedTensor.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    table_ids: A list of `ints`.
      A list of integers specifying the identifier of the embedding table
      (offset of TableDescriptor in the TPUEmbeddingConfiguration) to lookup the
      corresponding input. The ith input is looked up using table_ids[i]. The size
      of the table_ids list must be equal to that of sample_indices,
      embedding_indices and aggregation_weights.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    max_sequence_lengths: An optional list of `ints`. Defaults to `[]`.
    num_features: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingRaggedTensorBatch", name, sample_splits,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
        table_ids, "max_sequence_lengths", max_sequence_lengths,
        "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(
          sample_splits, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          table_ids=table_ids, max_sequence_lengths=max_sequence_lengths,
          num_features=num_features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingRaggedTensorBatch", sample_splits=sample_splits,
                                                embedding_indices=embedding_indices,
                                                aggregation_weights=aggregation_weights,
                                                mode_override=mode_override,
                                                table_ids=table_ids,
                                                device_ordinal=device_ordinal,
                                                combiners=combiners,
                                                max_sequence_lengths=max_sequence_lengths,
                                                num_features=num_features,
                                                name=name)
  return _op
EnqueueTPUEmbeddingRaggedTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingRaggedTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_ragged_tensor_batch))


def enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(sample_splits: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingRaggedTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], table_ids, device_ordinal: int, combiners, max_sequence_lengths, num_features, name, ctx):
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _attr_T1, sample_splits = _execute.args_to_matching_eager(list(sample_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
  table_ids, "max_sequence_lengths", max_sequence_lengths, "num_features",
  num_features)
  _result = _execute.execute(b"EnqueueTPUEmbeddingRaggedTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_EnqueueTPUEmbeddingSparseBatch_T1 = TypeVar("TV_EnqueueTPUEmbeddingSparseBatch_T1", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingSparseBatch_T2 = TypeVar("TV_EnqueueTPUEmbeddingSparseBatch_T2", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingSparseBatch_T3 = TypeVar("TV_EnqueueTPUEmbeddingSparseBatch_T3", _atypes.Float32, _atypes.Float64)

def enqueue_tpu_embedding_sparse_batch(sample_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal:int=-1, combiners=[], name=None):
  r"""An op that enqueues TPUEmbedding input indices from a SparseTensor.

  This Op eases the porting of code that uses embedding_lookup_sparse(),
  although some Python preprocessing of the SparseTensor arguments to
  embedding_lookup_sparse() is required to produce the arguments to this Op,
  since only a single EnqueueTPUEmbeddingSparseBatch Op is allowed per training
  step.

  The tensors at corresponding positions in the three input lists
  must have the same shape, i.e. rank 1 with dim_size() equal to the total
  number of lookups into the table described by the corresponding table_id.

  Args:
    sample_indices: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors specifying the training example and
      feature to which the corresponding embedding_indices and aggregation_weights
      values belong. sample_indices[i] must equal b * nf + f, where nf is the
      number of features from the corresponding table, f is in [0, nf), and
      b is in [0, batch size).
    embedding_indices: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding tables.
    aggregation_weights: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per sample -- i.e. per
      (training example, feature) -- aggregation weights.
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingSparseBatch", name, sample_indices,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return enqueue_tpu_embedding_sparse_batch_eager_fallback(
          sample_indices, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingSparseBatch", sample_indices=sample_indices,
                                          embedding_indices=embedding_indices,
                                          aggregation_weights=aggregation_weights,
                                          mode_override=mode_override,
                                          device_ordinal=device_ordinal,
                                          combiners=combiners, name=name)
  return _op
EnqueueTPUEmbeddingSparseBatch = tf_export("raw_ops.EnqueueTPUEmbeddingSparseBatch")(_ops.to_raw_op(enqueue_tpu_embedding_sparse_batch))


def enqueue_tpu_embedding_sparse_batch_eager_fallback(sample_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseBatch_T3], mode_override: Annotated[Any, _atypes.String], device_ordinal: int, combiners, name, ctx):
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices = _execute.args_to_matching_eager(list(sample_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners)
  _result = _execute.execute(b"EnqueueTPUEmbeddingSparseBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_EnqueueTPUEmbeddingSparseTensorBatch_T1 = TypeVar("TV_EnqueueTPUEmbeddingSparseTensorBatch_T1", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingSparseTensorBatch_T2 = TypeVar("TV_EnqueueTPUEmbeddingSparseTensorBatch_T2", _atypes.Int32, _atypes.Int64)
TV_EnqueueTPUEmbeddingSparseTensorBatch_T3 = TypeVar("TV_EnqueueTPUEmbeddingSparseTensorBatch_T3", _atypes.Float32, _atypes.Float64)

def enqueue_tpu_embedding_sparse_tensor_batch(sample_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], table_ids, device_ordinal:int=-1, combiners=[], max_sequence_lengths=[], num_features=[], name=None):
  r"""Eases the porting of code that uses tf.nn.embedding_lookup_sparse().

  sample_indices[i], embedding_indices[i] and aggregation_weights[i] correspond
  to the ith feature. table_ids[i] indicates which embedding table to look up ith
  feature.

  The tensors at corresponding positions in the three input lists (sample_indices,
  embedding_indices and aggregation_weights) must have the same shape, i.e. rank 1
  with dim_size() equal to the total number of lookups into the table described by
  the corresponding feature.

  Args:
    sample_indices: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors specifying the training example to
      which the corresponding embedding_indices and aggregation_weights values
      belong. It corresponds to sp_ids.indices[:,0] in  embedding_lookup_sparse().
    embedding_indices: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of rank 1 Tensors, indices into the embedding tables.
      It corresponds to sp_ids.values in embedding_lookup_sparse().
    aggregation_weights: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `float32`, `float64`.
      A list of rank 1 Tensors containing per training example
      aggregation weights. It corresponds to sp_weights.values in
      embedding_lookup_sparse().
    mode_override: A `Tensor` of type `string`.
      A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified', 'inference',
      'training', 'backward_pass_only'}. When set to 'unspecified', the mode set
      in TPUEmbeddingConfiguration is used, otherwise mode_override is used.
    table_ids: A list of `ints`.
      A list of integers specifying the identifier of the embedding table
      (offset of TableDescriptor in the TPUEmbeddingConfiguration) to lookup the
      corresponding input. The ith input is looked up using table_ids[i]. The size
      of the table_ids list must be equal to that of sample_indices,
      embedding_indices and aggregation_weights.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. Should be >= 0 and less than the number
      of TPU cores in the task on which the node is placed.
    combiners: An optional list of `strings`. Defaults to `[]`.
      A list of string scalars, one for each embedding table that specify
      how to normalize the embedding activations after weighted summation.
      Supported combiners are 'mean', 'sum', or 'sqrtn'. It is invalid to have
      the sum of the weights be 0 for 'mean' or the sum of the squared weights be
      0 for 'sqrtn'. If combiners isn't passed, the default is to use 'sum' for
      all tables.
    max_sequence_lengths: An optional list of `ints`. Defaults to `[]`.
    num_features: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingSparseTensorBatch", name, sample_indices,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
        table_ids, "max_sequence_lengths", max_sequence_lengths,
        "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return enqueue_tpu_embedding_sparse_tensor_batch_eager_fallback(
          sample_indices, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          table_ids=table_ids, max_sequence_lengths=max_sequence_lengths,
          num_features=num_features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingSparseTensorBatch", sample_indices=sample_indices,
                                                embedding_indices=embedding_indices,
                                                aggregation_weights=aggregation_weights,
                                                mode_override=mode_override,
                                                table_ids=table_ids,
                                                device_ordinal=device_ordinal,
                                                combiners=combiners,
                                                max_sequence_lengths=max_sequence_lengths,
                                                num_features=num_features,
                                                name=name)
  return _op
EnqueueTPUEmbeddingSparseTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingSparseTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_sparse_tensor_batch))


def enqueue_tpu_embedding_sparse_tensor_batch_eager_fallback(sample_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T1], embedding_indices: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T2], aggregation_weights: Annotated[List[Any], TV_EnqueueTPUEmbeddingSparseTensorBatch_T3], mode_override: Annotated[Any, _atypes.String], table_ids, device_ordinal: int, combiners, max_sequence_lengths, num_features, name, ctx):
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _attr_T1, sample_indices = _execute.args_to_matching_eager(list(sample_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
  table_ids, "max_sequence_lengths", max_sequence_lengths, "num_features",
  num_features)
  _result = _execute.execute(b"EnqueueTPUEmbeddingSparseTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_InfeedDequeue_dtype = TypeVar("TV_InfeedDequeue_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def infeed_dequeue(dtype: TV_InfeedDequeue_dtype, shape, name=None) -> Annotated[Any, TV_InfeedDequeue_dtype]:
  r"""A placeholder op for a value that will be fed into the computation.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InfeedDequeue", name, "dtype", dtype, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return infeed_dequeue_eager_fallback(
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InfeedDequeue", dtype=dtype, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InfeedDequeue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InfeedDequeue = tf_export("raw_ops.InfeedDequeue")(_ops.to_raw_op(infeed_dequeue))


def infeed_dequeue_eager_fallback(dtype: TV_InfeedDequeue_dtype, shape, name, ctx) -> Annotated[Any, TV_InfeedDequeue_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape)
  _result = _execute.execute(b"InfeedDequeue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InfeedDequeue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def infeed_dequeue_tuple(dtypes, shapes, name=None):
  r"""Fetches multiple values from infeed as an XLA tuple.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InfeedDequeueTuple", name, "dtypes", dtypes, "shapes", shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return infeed_dequeue_tuple_eager_fallback(
          dtypes=dtypes, shapes=shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InfeedDequeueTuple", dtypes=dtypes, shapes=shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InfeedDequeueTuple", _inputs_flat, _attrs, _result)
  return _result

InfeedDequeueTuple = tf_export("raw_ops.InfeedDequeueTuple")(_ops.to_raw_op(infeed_dequeue_tuple))


def infeed_dequeue_tuple_eager_fallback(dtypes, shapes, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  _inputs_flat = []
  _attrs = ("dtypes", dtypes, "shapes", shapes)
  _result = _execute.execute(b"InfeedDequeueTuple", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InfeedDequeueTuple", _inputs_flat, _attrs, _result)
  return _result


TV_InfeedEnqueue_dtype = TypeVar("TV_InfeedEnqueue_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def infeed_enqueue(input: Annotated[Any, TV_InfeedEnqueue_dtype], shape=[], layout=[], device_ordinal:int=-1, name=None):
  r"""An op which feeds a single Tensor value into the computation.

  Args:
    input: A `Tensor`.
      A tensor that will be provided using the infeed mechanism.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of the tensor.
    layout: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence.
      If a layout attribute is passed, but its values are all -1, the layout will
      be computed by the infeed operation.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InfeedEnqueue", name, input, "shape", shape, "layout", layout,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return infeed_enqueue_eager_fallback(
          input, shape=shape, layout=layout, device_ordinal=device_ordinal,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if shape is None:
    shape = []
  shape = _execute.make_shape(shape, "shape")
  if layout is None:
    layout = []
  if not isinstance(layout, (list, tuple)):
    raise TypeError(
        "Expected list for 'layout' argument to "
        "'infeed_enqueue' Op, not %r." % layout)
  layout = [_execute.make_int(_i, "layout") for _i in layout]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InfeedEnqueue", input=input, shape=shape, layout=layout,
                         device_ordinal=device_ordinal, name=name)
  return _op
InfeedEnqueue = tf_export("raw_ops.InfeedEnqueue")(_ops.to_raw_op(infeed_enqueue))


def infeed_enqueue_eager_fallback(input: Annotated[Any, TV_InfeedEnqueue_dtype], shape, layout, device_ordinal: int, name, ctx):
  if shape is None:
    shape = []
  shape = _execute.make_shape(shape, "shape")
  if layout is None:
    layout = []
  if not isinstance(layout, (list, tuple)):
    raise TypeError(
        "Expected list for 'layout' argument to "
        "'infeed_enqueue' Op, not %r." % layout)
  layout = [_execute.make_int(_i, "layout") for _i in layout]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _attr_dtype, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("dtype", _attr_dtype, "shape", shape, "layout", layout,
  "device_ordinal", device_ordinal)
  _result = _execute.execute(b"InfeedEnqueue", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def infeed_enqueue_prelinearized_buffer(input: Annotated[Any, _atypes.Variant], device_ordinal:int=-1, name=None):
  r"""An op which enqueues prelinearized buffer into TPU infeed.

  Args:
    input: A `Tensor` of type `variant`.
      A variant tensor representing linearized output.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op is running on a TPU device
      and = 0 when the Op is running on the CPU device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InfeedEnqueuePrelinearizedBuffer", name, input,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return infeed_enqueue_prelinearized_buffer_eager_fallback(
          input, device_ordinal=device_ordinal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InfeedEnqueuePrelinearizedBuffer", input=input,
                                            device_ordinal=device_ordinal,
                                            name=name)
  return _op
InfeedEnqueuePrelinearizedBuffer = tf_export("raw_ops.InfeedEnqueuePrelinearizedBuffer")(_ops.to_raw_op(infeed_enqueue_prelinearized_buffer))


def infeed_enqueue_prelinearized_buffer_eager_fallback(input: Annotated[Any, _atypes.Variant], device_ordinal: int, name, ctx):
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  input = _ops.convert_to_tensor(input, _dtypes.variant)
  _inputs_flat = [input]
  _attrs = ("device_ordinal", device_ordinal)
  _result = _execute.execute(b"InfeedEnqueuePrelinearizedBuffer", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def infeed_enqueue_tuple(inputs, shapes, layouts=[], device_ordinal:int=-1, name=None):
  r"""Feeds multiple Tensor values into the computation as an XLA tuple.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be provided using the infeed mechanism.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `inputs`.
    layouts: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence for
      all the tuple shapes, in the order the shapes appear in the "shapes" input.
      The layout elements for a sub-shape can be set to -1, in which case the
      corresponding layout will be computed by the infeed operation.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InfeedEnqueueTuple", name, inputs, "shapes", shapes, "layouts",
        layouts, "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return infeed_enqueue_tuple_eager_fallback(
          inputs, shapes=shapes, layouts=layouts,
          device_ordinal=device_ordinal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_enqueue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if layouts is None:
    layouts = []
  if not isinstance(layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'layouts' argument to "
        "'infeed_enqueue_tuple' Op, not %r." % layouts)
  layouts = [_execute.make_int(_i, "layouts") for _i in layouts]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InfeedEnqueueTuple", inputs=inputs, shapes=shapes, layouts=layouts,
                              device_ordinal=device_ordinal, name=name)
  return _op
InfeedEnqueueTuple = tf_export("raw_ops.InfeedEnqueueTuple")(_ops.to_raw_op(infeed_enqueue_tuple))


def infeed_enqueue_tuple_eager_fallback(inputs, shapes, layouts, device_ordinal: int, name, ctx):
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_enqueue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if layouts is None:
    layouts = []
  if not isinstance(layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'layouts' argument to "
        "'infeed_enqueue_tuple' Op, not %r." % layouts)
  layouts = [_execute.make_int(_i, "layouts") for _i in layouts]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("dtypes", _attr_dtypes, "shapes", shapes, "layouts", layouts,
  "device_ordinal", device_ordinal)
  _result = _execute.execute(b"InfeedEnqueueTuple", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def is_tpu_embedding_initialized(config:str="", name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Whether TPU Embedding is initialized in a distributed TPU system.

  Args:
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsTPUEmbeddingInitialized", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return is_tpu_embedding_initialized_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsTPUEmbeddingInitialized", config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsTPUEmbeddingInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsTPUEmbeddingInitialized = tf_export("raw_ops.IsTPUEmbeddingInitialized")(_ops.to_raw_op(is_tpu_embedding_initialized))


def is_tpu_embedding_initialized_eager_fallback(config: str, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"IsTPUEmbeddingInitialized", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsTPUEmbeddingInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def load_tpu_embedding_adam_parameters(parameters: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocities: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load ADAM embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the ADAM optimization algorithm.
    momenta: A `Tensor` of type `float32`.
      Value of momenta used in the ADAM optimization algorithm.
    velocities: A `Tensor` of type `float32`.
      Value of velocities used in the ADAM optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingADAMParameters", name, parameters, momenta,
        velocities, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_adam_parameters_eager_fallback(
          parameters, momenta, velocities, table_id=table_id,
          table_name=table_name, num_shards=num_shards, shard_id=shard_id,
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingADAMParameters", parameters=parameters,
                                          momenta=momenta,
                                          velocities=velocities,
                                          num_shards=num_shards,
                                          shard_id=shard_id,
                                          table_id=table_id,
                                          table_name=table_name,
                                          config=config, name=name)
  return _op
LoadTPUEmbeddingADAMParameters = tf_export("raw_ops.LoadTPUEmbeddingADAMParameters")(_ops.to_raw_op(load_tpu_embedding_adam_parameters))


def load_tpu_embedding_adam_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocities: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  velocities = _ops.convert_to_tensor(velocities, _dtypes.float32)
  _inputs_flat = [parameters, momenta, velocities]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingADAMParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_adadelta_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], updates: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load Adadelta embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the Adadelta optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the Adadelta optimization algorithm.
    updates: A `Tensor` of type `float32`.
      Value of updates used in the Adadelta optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingAdadeltaParameters", name, parameters,
        accumulators, updates, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_adadelta_parameters_eager_fallback(
          parameters, accumulators, updates, table_id=table_id,
          table_name=table_name, num_shards=num_shards, shard_id=shard_id,
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingAdadeltaParameters", parameters=parameters,
                                              accumulators=accumulators,
                                              updates=updates,
                                              num_shards=num_shards,
                                              shard_id=shard_id,
                                              table_id=table_id,
                                              table_name=table_name,
                                              config=config, name=name)
  return _op
LoadTPUEmbeddingAdadeltaParameters = tf_export("raw_ops.LoadTPUEmbeddingAdadeltaParameters")(_ops.to_raw_op(load_tpu_embedding_adadelta_parameters))


def load_tpu_embedding_adadelta_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], updates: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  updates = _ops.convert_to_tensor(updates, _dtypes.float32)
  _inputs_flat = [parameters, accumulators, updates]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingAdadeltaParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_adagrad_momentum_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load Adagrad Momentum embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the Adagrad Momentum optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the Adagrad Momentum optimization algorithm.
    momenta: A `Tensor` of type `float32`.
      Value of momenta used in the Adagrad Momentum optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingAdagradMomentumParameters", name, parameters,
        accumulators, momenta, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_adagrad_momentum_parameters_eager_fallback(
          parameters, accumulators, momenta, table_id=table_id,
          table_name=table_name, num_shards=num_shards, shard_id=shard_id,
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingAdagradMomentumParameters", parameters=parameters,
                                                     accumulators=accumulators,
                                                     momenta=momenta,
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     table_id=table_id,
                                                     table_name=table_name,
                                                     config=config, name=name)
  return _op
LoadTPUEmbeddingAdagradMomentumParameters = tf_export("raw_ops.LoadTPUEmbeddingAdagradMomentumParameters")(_ops.to_raw_op(load_tpu_embedding_adagrad_momentum_parameters))


def load_tpu_embedding_adagrad_momentum_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  _inputs_flat = [parameters, accumulators, momenta]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingAdagradMomentumParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_adagrad_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load Adagrad embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the Adagrad optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the Adagrad optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingAdagradParameters", name, parameters,
        accumulators, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_adagrad_parameters_eager_fallback(
          parameters, accumulators, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingAdagradParameters", parameters=parameters,
                                             accumulators=accumulators,
                                             num_shards=num_shards,
                                             shard_id=shard_id,
                                             table_id=table_id,
                                             table_name=table_name,
                                             config=config, name=name)
  return _op
LoadTPUEmbeddingAdagradParameters = tf_export("raw_ops.LoadTPUEmbeddingAdagradParameters")(_ops.to_raw_op(load_tpu_embedding_adagrad_parameters))


def load_tpu_embedding_adagrad_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  _inputs_flat = [parameters, accumulators]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingAdagradParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_centered_rms_prop_parameters(parameters: Annotated[Any, _atypes.Float32], ms: Annotated[Any, _atypes.Float32], mom: Annotated[Any, _atypes.Float32], mg: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load centered RMSProp embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the centered RMSProp optimization algorithm.
    ms: A `Tensor` of type `float32`.
      Value of ms used in the centered RMSProp optimization algorithm.
    mom: A `Tensor` of type `float32`.
      Value of mom used in the centered RMSProp optimization algorithm.
    mg: A `Tensor` of type `float32`.
      Value of mg used in the centered RMSProp optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingCenteredRMSPropParameters", name, parameters,
        ms, mom, mg, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_centered_rms_prop_parameters_eager_fallback(
          parameters, ms, mom, mg, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingCenteredRMSPropParameters", parameters=parameters,
                                                     ms=ms, mom=mom, mg=mg,
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     table_id=table_id,
                                                     table_name=table_name,
                                                     config=config, name=name)
  return _op
LoadTPUEmbeddingCenteredRMSPropParameters = tf_export("raw_ops.LoadTPUEmbeddingCenteredRMSPropParameters")(_ops.to_raw_op(load_tpu_embedding_centered_rms_prop_parameters))


def load_tpu_embedding_centered_rms_prop_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], ms: Annotated[Any, _atypes.Float32], mom: Annotated[Any, _atypes.Float32], mg: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  ms = _ops.convert_to_tensor(ms, _dtypes.float32)
  mom = _ops.convert_to_tensor(mom, _dtypes.float32)
  mg = _ops.convert_to_tensor(mg, _dtypes.float32)
  _inputs_flat = [parameters, ms, mom, mg]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingCenteredRMSPropParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_ftrl_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], linears: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load FTRL embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the FTRL optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the FTRL optimization algorithm.
    linears: A `Tensor` of type `float32`.
      Value of linears used in the FTRL optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingFTRLParameters", name, parameters,
        accumulators, linears, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_ftrl_parameters_eager_fallback(
          parameters, accumulators, linears, table_id=table_id,
          table_name=table_name, num_shards=num_shards, shard_id=shard_id,
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingFTRLParameters", parameters=parameters,
                                          accumulators=accumulators,
                                          linears=linears,
                                          num_shards=num_shards,
                                          shard_id=shard_id,
                                          table_id=table_id,
                                          table_name=table_name,
                                          config=config, name=name)
  return _op
LoadTPUEmbeddingFTRLParameters = tf_export("raw_ops.LoadTPUEmbeddingFTRLParameters")(_ops.to_raw_op(load_tpu_embedding_ftrl_parameters))


def load_tpu_embedding_ftrl_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], linears: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  linears = _ops.convert_to_tensor(linears, _dtypes.float32)
  _inputs_flat = [parameters, accumulators, linears]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingFTRLParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_frequency_estimator_parameters(parameters: Annotated[Any, _atypes.Float32], last_hit_step: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load frequency estimator embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the frequency estimator optimization algorithm.
    last_hit_step: A `Tensor` of type `float32`.
      Value of last_hit_step used in the frequency estimator optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingFrequencyEstimatorParameters", name,
        parameters, last_hit_step, "table_id", table_id, "table_name",
        table_name, "num_shards", num_shards, "shard_id", shard_id, "config",
        config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_frequency_estimator_parameters_eager_fallback(
          parameters, last_hit_step, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingFrequencyEstimatorParameters", parameters=parameters,
                                                        last_hit_step=last_hit_step,
                                                        num_shards=num_shards,
                                                        shard_id=shard_id,
                                                        table_id=table_id,
                                                        table_name=table_name,
                                                        config=config,
                                                        name=name)
  return _op
LoadTPUEmbeddingFrequencyEstimatorParameters = tf_export("raw_ops.LoadTPUEmbeddingFrequencyEstimatorParameters")(_ops.to_raw_op(load_tpu_embedding_frequency_estimator_parameters))


def load_tpu_embedding_frequency_estimator_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], last_hit_step: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  last_hit_step = _ops.convert_to_tensor(last_hit_step, _dtypes.float32)
  _inputs_flat = [parameters, last_hit_step]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingFrequencyEstimatorParameters",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_mdl_adagrad_light_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], weights: Annotated[Any, _atypes.Float32], benefits: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load MDL Adagrad Light embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the MDL Adagrad Light optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the MDL Adagrad Light optimization algorithm.
    weights: A `Tensor` of type `float32`.
      Value of weights used in the MDL Adagrad Light optimization algorithm.
    benefits: A `Tensor` of type `float32`.
      Value of benefits used in the MDL Adagrad Light optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingMDLAdagradLightParameters", name, parameters,
        accumulators, weights, benefits, "table_id", table_id, "table_name",
        table_name, "num_shards", num_shards, "shard_id", shard_id, "config",
        config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_mdl_adagrad_light_parameters_eager_fallback(
          parameters, accumulators, weights, benefits, table_id=table_id,
          table_name=table_name, num_shards=num_shards, shard_id=shard_id,
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingMDLAdagradLightParameters", parameters=parameters,
                                                     accumulators=accumulators,
                                                     weights=weights,
                                                     benefits=benefits,
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     table_id=table_id,
                                                     table_name=table_name,
                                                     config=config, name=name)
  return _op
LoadTPUEmbeddingMDLAdagradLightParameters = tf_export("raw_ops.LoadTPUEmbeddingMDLAdagradLightParameters")(_ops.to_raw_op(load_tpu_embedding_mdl_adagrad_light_parameters))


def load_tpu_embedding_mdl_adagrad_light_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], weights: Annotated[Any, _atypes.Float32], benefits: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  weights = _ops.convert_to_tensor(weights, _dtypes.float32)
  benefits = _ops.convert_to_tensor(benefits, _dtypes.float32)
  _inputs_flat = [parameters, accumulators, weights, benefits]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingMDLAdagradLightParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_momentum_parameters(parameters: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load Momentum embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the Momentum optimization algorithm.
    momenta: A `Tensor` of type `float32`.
      Value of momenta used in the Momentum optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingMomentumParameters", name, parameters, momenta,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_momentum_parameters_eager_fallback(
          parameters, momenta, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingMomentumParameters", parameters=parameters,
                                              momenta=momenta,
                                              num_shards=num_shards,
                                              shard_id=shard_id,
                                              table_id=table_id,
                                              table_name=table_name,
                                              config=config, name=name)
  return _op
LoadTPUEmbeddingMomentumParameters = tf_export("raw_ops.LoadTPUEmbeddingMomentumParameters")(_ops.to_raw_op(load_tpu_embedding_momentum_parameters))


def load_tpu_embedding_momentum_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  _inputs_flat = [parameters, momenta]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingMomentumParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_proximal_adagrad_parameters(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load proximal Adagrad embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the proximal Adagrad optimization algorithm.
    accumulators: A `Tensor` of type `float32`.
      Value of accumulators used in the proximal Adagrad optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingProximalAdagradParameters", name, parameters,
        accumulators, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_proximal_adagrad_parameters_eager_fallback(
          parameters, accumulators, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingProximalAdagradParameters", parameters=parameters,
                                                     accumulators=accumulators,
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     table_id=table_id,
                                                     table_name=table_name,
                                                     config=config, name=name)
  return _op
LoadTPUEmbeddingProximalAdagradParameters = tf_export("raw_ops.LoadTPUEmbeddingProximalAdagradParameters")(_ops.to_raw_op(load_tpu_embedding_proximal_adagrad_parameters))


def load_tpu_embedding_proximal_adagrad_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], accumulators: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  accumulators = _ops.convert_to_tensor(accumulators, _dtypes.float32)
  _inputs_flat = [parameters, accumulators]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingProximalAdagradParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_proximal_yogi_parameters(parameters: Annotated[Any, _atypes.Float32], v: Annotated[Any, _atypes.Float32], m: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""TODO: add doc.

  Args:
    parameters: A `Tensor` of type `float32`.
    v: A `Tensor` of type `float32`.
    m: A `Tensor` of type `float32`.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingProximalYogiParameters", name, parameters, v,
        m, "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_proximal_yogi_parameters_eager_fallback(
          parameters, v, m, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingProximalYogiParameters", parameters=parameters, v=v,
                                                  m=m, num_shards=num_shards,
                                                  shard_id=shard_id,
                                                  table_id=table_id,
                                                  table_name=table_name,
                                                  config=config, name=name)
  return _op
LoadTPUEmbeddingProximalYogiParameters = tf_export("raw_ops.LoadTPUEmbeddingProximalYogiParameters")(_ops.to_raw_op(load_tpu_embedding_proximal_yogi_parameters))


def load_tpu_embedding_proximal_yogi_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], v: Annotated[Any, _atypes.Float32], m: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  v = _ops.convert_to_tensor(v, _dtypes.float32)
  m = _ops.convert_to_tensor(m, _dtypes.float32)
  _inputs_flat = [parameters, v, m]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingProximalYogiParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_rms_prop_parameters(parameters: Annotated[Any, _atypes.Float32], ms: Annotated[Any, _atypes.Float32], mom: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load RMSProp embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the RMSProp optimization algorithm.
    ms: A `Tensor` of type `float32`.
      Value of ms used in the RMSProp optimization algorithm.
    mom: A `Tensor` of type `float32`.
      Value of mom used in the RMSProp optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingRMSPropParameters", name, parameters, ms, mom,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_rms_prop_parameters_eager_fallback(
          parameters, ms, mom, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingRMSPropParameters", parameters=parameters, ms=ms,
                                             mom=mom, num_shards=num_shards,
                                             shard_id=shard_id,
                                             table_id=table_id,
                                             table_name=table_name,
                                             config=config, name=name)
  return _op
LoadTPUEmbeddingRMSPropParameters = tf_export("raw_ops.LoadTPUEmbeddingRMSPropParameters")(_ops.to_raw_op(load_tpu_embedding_rms_prop_parameters))


def load_tpu_embedding_rms_prop_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], ms: Annotated[Any, _atypes.Float32], mom: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  ms = _ops.convert_to_tensor(ms, _dtypes.float32)
  mom = _ops.convert_to_tensor(mom, _dtypes.float32)
  _inputs_flat = [parameters, ms, mom]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingRMSPropParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def load_tpu_embedding_stochastic_gradient_descent_parameters(parameters: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Load SGD embedding parameters.

  An op that loads optimization parameters into HBM for embedding. Must be
  preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
  embedding table configuration. For example, this op is used to install
  parameters that are loaded from a checkpoint before a training loop is
  executed.

  Args:
    parameters: A `Tensor` of type `float32`.
      Value of parameters used in the stochastic gradient descent optimization algorithm.
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadTPUEmbeddingStochasticGradientDescentParameters", name,
        parameters, "table_id", table_id, "table_name", table_name,
        "num_shards", num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_tpu_embedding_stochastic_gradient_descent_parameters_eager_fallback(
          parameters, table_id=table_id, table_name=table_name,
          num_shards=num_shards, shard_id=shard_id, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadTPUEmbeddingStochasticGradientDescentParameters", parameters=parameters,
                                                               num_shards=num_shards,
                                                               shard_id=shard_id,
                                                               table_id=table_id,
                                                               table_name=table_name,
                                                               config=config,
                                                               name=name)
  return _op
LoadTPUEmbeddingStochasticGradientDescentParameters = tf_export("raw_ops.LoadTPUEmbeddingStochasticGradientDescentParameters")(_ops.to_raw_op(load_tpu_embedding_stochastic_gradient_descent_parameters))


def load_tpu_embedding_stochastic_gradient_descent_parameters_eager_fallback(parameters: Annotated[Any, _atypes.Float32], num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  parameters = _ops.convert_to_tensor(parameters, _dtypes.float32)
  _inputs_flat = [parameters]
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"LoadTPUEmbeddingStochasticGradientDescentParameters",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_OutfeedDequeue_dtype = TypeVar("TV_OutfeedDequeue_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def outfeed_dequeue(dtype: TV_OutfeedDequeue_dtype, shape, device_ordinal:int=-1, name=None) -> Annotated[Any, TV_OutfeedDequeue_dtype]:
  r"""Retrieves a single tensor from the computation outfeed.

  This operation will block indefinitely until data is available.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedDequeue", name, "dtype", dtype, "shape", shape,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_dequeue_eager_fallback(
          dtype=dtype, shape=shape, device_ordinal=device_ordinal, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedDequeue", dtype=dtype, shape=shape,
                          device_ordinal=device_ordinal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "device_ordinal",
              _op._get_attr_int("device_ordinal"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutfeedDequeue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OutfeedDequeue = tf_export("raw_ops.OutfeedDequeue")(_ops.to_raw_op(outfeed_dequeue))


def outfeed_dequeue_eager_fallback(dtype: TV_OutfeedDequeue_dtype, shape, device_ordinal: int, name, ctx) -> Annotated[Any, TV_OutfeedDequeue_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape, "device_ordinal", device_ordinal)
  _result = _execute.execute(b"OutfeedDequeue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutfeedDequeue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def outfeed_dequeue_tuple(dtypes, shapes, device_ordinal:int=-1, name=None):
  r"""Retrieve multiple values from the computation outfeed.

  This operation will block indefinitely until data is available. Output `i`
  corresponds to XLA tuple element `i`.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedDequeueTuple", name, "dtypes", dtypes, "shapes", shapes,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_dequeue_tuple_eager_fallback(
          dtypes=dtypes, shapes=shapes, device_ordinal=device_ordinal,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedDequeueTuple", dtypes=dtypes, shapes=shapes,
                               device_ordinal=device_ordinal, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"), "device_ordinal",
              _op._get_attr_int("device_ordinal"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutfeedDequeueTuple", _inputs_flat, _attrs, _result)
  return _result

OutfeedDequeueTuple = tf_export("raw_ops.OutfeedDequeueTuple")(_ops.to_raw_op(outfeed_dequeue_tuple))


def outfeed_dequeue_tuple_eager_fallback(dtypes, shapes, device_ordinal: int, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _inputs_flat = []
  _attrs = ("dtypes", dtypes, "shapes", shapes, "device_ordinal",
  device_ordinal)
  _result = _execute.execute(b"OutfeedDequeueTuple", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutfeedDequeueTuple", _inputs_flat, _attrs, _result)
  return _result


def outfeed_dequeue_tuple_v2(device_ordinal: Annotated[Any, _atypes.Int32], dtypes, shapes, name=None):
  r"""Retrieve multiple values from the computation outfeed. Device ordinal is a
tensor allowing dynamic outfeed.

  This operation will block indefinitely until data is available. Output `i`
  corresponds to XLA tuple element `i`.

  Args:
    device_ordinal: A `Tensor` of type `int32`.
      An int scalar tensor, representing the TPU device to use. This should be -1 when
      the Op is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedDequeueTupleV2", name, device_ordinal, "dtypes", dtypes,
        "shapes", shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_dequeue_tuple_v2_eager_fallback(
          device_ordinal, dtypes=dtypes, shapes=shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'outfeed_dequeue_tuple_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'outfeed_dequeue_tuple_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedDequeueTupleV2", device_ordinal=device_ordinal, dtypes=dtypes,
                                 shapes=shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutfeedDequeueTupleV2", _inputs_flat, _attrs, _result)
  return _result

OutfeedDequeueTupleV2 = tf_export("raw_ops.OutfeedDequeueTupleV2")(_ops.to_raw_op(outfeed_dequeue_tuple_v2))


def outfeed_dequeue_tuple_v2_eager_fallback(device_ordinal: Annotated[Any, _atypes.Int32], dtypes, shapes, name, ctx):
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'outfeed_dequeue_tuple_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'outfeed_dequeue_tuple_v2' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = [device_ordinal]
  _attrs = ("dtypes", dtypes, "shapes", shapes)
  _result = _execute.execute(b"OutfeedDequeueTupleV2", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutfeedDequeueTupleV2", _inputs_flat, _attrs, _result)
  return _result


TV_OutfeedDequeueV2_dtype = TypeVar("TV_OutfeedDequeueV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def outfeed_dequeue_v2(device_ordinal: Annotated[Any, _atypes.Int32], dtype: TV_OutfeedDequeueV2_dtype, shape, name=None) -> Annotated[Any, TV_OutfeedDequeueV2_dtype]:
  r"""Retrieves a single tensor from the computation outfeed. Device ordinal is a
tensor allowing dynamic outfeed.

  This operation will block indefinitely until data is available.

  Args:
    device_ordinal: A `Tensor` of type `int32`.
      An int scalar tensor, representing the TPU device to use. This should be -1 when
      the Op is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedDequeueV2", name, device_ordinal, "dtype", dtype,
        "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_dequeue_v2_eager_fallback(
          device_ordinal, dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedDequeueV2", device_ordinal=device_ordinal, dtype=dtype,
                            shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutfeedDequeueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OutfeedDequeueV2 = tf_export("raw_ops.OutfeedDequeueV2")(_ops.to_raw_op(outfeed_dequeue_v2))


def outfeed_dequeue_v2_eager_fallback(device_ordinal: Annotated[Any, _atypes.Int32], dtype: TV_OutfeedDequeueV2_dtype, shape, name, ctx) -> Annotated[Any, TV_OutfeedDequeueV2_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = [device_ordinal]
  _attrs = ("dtype", dtype, "shape", shape)
  _result = _execute.execute(b"OutfeedDequeueV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutfeedDequeueV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_OutfeedEnqueue_dtype = TypeVar("TV_OutfeedEnqueue_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def outfeed_enqueue(input: Annotated[Any, TV_OutfeedEnqueue_dtype], name=None):
  r"""Enqueue a Tensor on the computation outfeed.

  Args:
    input: A `Tensor`. A tensor that will be inserted into the outfeed queue.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedEnqueue", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_enqueue_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedEnqueue", input=input, name=name)
  return _op
OutfeedEnqueue = tf_export("raw_ops.OutfeedEnqueue")(_ops.to_raw_op(outfeed_enqueue))


def outfeed_enqueue_eager_fallback(input: Annotated[Any, TV_OutfeedEnqueue_dtype], name, ctx):
  _attr_dtype, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("dtype", _attr_dtype)
  _result = _execute.execute(b"OutfeedEnqueue", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def outfeed_enqueue_tuple(inputs, name=None):
  r"""Enqueue multiple Tensor values on the computation outfeed.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be inserted into the outfeed queue as an
      XLA tuple.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutfeedEnqueueTuple", name, inputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return outfeed_enqueue_tuple_eager_fallback(
          inputs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutfeedEnqueueTuple", inputs=inputs, name=name)
  return _op
OutfeedEnqueueTuple = tf_export("raw_ops.OutfeedEnqueueTuple")(_ops.to_raw_op(outfeed_enqueue_tuple))


def outfeed_enqueue_tuple_eager_fallback(inputs, name, ctx):
  _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("dtypes", _attr_dtypes)
  _result = _execute.execute(b"OutfeedEnqueueTuple", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_Prelinearize_dtype = TypeVar("TV_Prelinearize_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def prelinearize(input: Annotated[Any, TV_Prelinearize_dtype], shape=[], layout=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""An op which linearizes one Tensor value to an opaque variant tensor.

  Args:
    input: A `Tensor`. A tensor that will be linearized.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of the tensor.
    layout: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence. If a layout
      attribute is passed but its values are all -1 the layout will be computed by
      the infeed operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Prelinearize", name, input, "shape", shape, "layout", layout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return prelinearize_eager_fallback(
          input, shape=shape, layout=layout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if shape is None:
    shape = []
  shape = _execute.make_shape(shape, "shape")
  if layout is None:
    layout = []
  if not isinstance(layout, (list, tuple)):
    raise TypeError(
        "Expected list for 'layout' argument to "
        "'prelinearize' Op, not %r." % layout)
  layout = [_execute.make_int(_i, "layout") for _i in layout]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Prelinearize", input=input, shape=shape, layout=layout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"), "layout", _op.get_attr("layout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Prelinearize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Prelinearize = tf_export("raw_ops.Prelinearize")(_ops.to_raw_op(prelinearize))


def prelinearize_eager_fallback(input: Annotated[Any, TV_Prelinearize_dtype], shape, layout, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if shape is None:
    shape = []
  shape = _execute.make_shape(shape, "shape")
  if layout is None:
    layout = []
  if not isinstance(layout, (list, tuple)):
    raise TypeError(
        "Expected list for 'layout' argument to "
        "'prelinearize' Op, not %r." % layout)
  layout = [_execute.make_int(_i, "layout") for _i in layout]
  _attr_dtype, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("dtype", _attr_dtype, "shape", shape, "layout", layout)
  _result = _execute.execute(b"Prelinearize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Prelinearize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def prelinearize_tuple(inputs, shapes, layouts=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""An op which linearizes multiple Tensor values to an opaque variant tensor.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be provided using the infeed mechanism.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `inputs`.
    layouts: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence for all the
      tuple shapes in the order the shapes appear in the "shapes" input. The layout
      elements for a sub-shape can be set to -1 in which case the corresponding layout
      will be computed by the infeed operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PrelinearizeTuple", name, inputs, "shapes", shapes, "layouts",
        layouts)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return prelinearize_tuple_eager_fallback(
          inputs, shapes=shapes, layouts=layouts, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'prelinearize_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if layouts is None:
    layouts = []
  if not isinstance(layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'layouts' argument to "
        "'prelinearize_tuple' Op, not %r." % layouts)
  layouts = [_execute.make_int(_i, "layouts") for _i in layouts]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PrelinearizeTuple", inputs=inputs, shapes=shapes, layouts=layouts,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"), "layouts", _op.get_attr("layouts"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PrelinearizeTuple", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PrelinearizeTuple = tf_export("raw_ops.PrelinearizeTuple")(_ops.to_raw_op(prelinearize_tuple))


def prelinearize_tuple_eager_fallback(inputs, shapes, layouts, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'prelinearize_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if layouts is None:
    layouts = []
  if not isinstance(layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'layouts' argument to "
        "'prelinearize_tuple' Op, not %r." % layouts)
  layouts = [_execute.make_int(_i, "layouts") for _i in layouts]
  _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("dtypes", _attr_dtypes, "shapes", shapes, "layouts", layouts)
  _result = _execute.execute(b"PrelinearizeTuple", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PrelinearizeTuple", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ReadVariableXlaSplitND_T = TypeVar("TV_ReadVariableXlaSplitND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def read_variable_xla_split_nd(resource: Annotated[Any, _atypes.Resource], T: TV_ReadVariableXlaSplitND_T, N: int, num_splits, paddings=[], name=None):
  r"""Splits resource variable input tensor across all dimensions.

  An op which splits the resource variable input tensor based on the given
  num_splits attribute, pads slices optionally, and returned the slices. Slices
  are returned in row-major order.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1, 2],
   [3, 4, 5],
   [6, 7, 8]]
  ```
  `num_splits`:
  ```
  [2, 2]
  ```
  and `paddings`:
  ```
  [1, 1]
  ```
  the expected `outputs` is:
  ```
  [[0, 1],
   [3, 4]]
  [[2, 0],
   [5, 0]]
  [[6, 7],
   [0, 0]]
  [[8, 0],
   [0, 0]]
  ```

  Args:
    resource: A `Tensor` of type `resource`.
      Resource variable of input tensor to split across all dimensions.
    T: A `tf.DType`.
    N: An `int` that is `>= 1`.
    num_splits: A list of `ints`.
      Number of ways to split per dimension. Shape dimensions must be evenly
      divisible.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension of input tensor to apply before
      splitting. This can be used to make a dimension evenly divisible.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReadVariableXlaSplitND", name, resource, "T", T, "N", N,
        "num_splits", num_splits, "paddings", paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return read_variable_xla_split_nd_eager_fallback(
          resource, T=T, N=N, num_splits=num_splits, paddings=paddings,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  if not isinstance(num_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_splits' argument to "
        "'read_variable_xla_split_nd' Op, not %r." % num_splits)
  num_splits = [_execute.make_int(_i, "num_splits") for _i in num_splits]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'read_variable_xla_split_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReadVariableXlaSplitND", resource=resource, T=T, N=N,
                                  num_splits=num_splits, paddings=paddings,
                                  name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"),
              "num_splits", _op.get_attr("num_splits"), "paddings",
              _op.get_attr("paddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReadVariableXlaSplitND", _inputs_flat, _attrs, _result)
  return _result

ReadVariableXlaSplitND = tf_export("raw_ops.ReadVariableXlaSplitND")(_ops.to_raw_op(read_variable_xla_split_nd))


def read_variable_xla_split_nd_eager_fallback(resource: Annotated[Any, _atypes.Resource], T: TV_ReadVariableXlaSplitND_T, N: int, num_splits, paddings, name, ctx):
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  if not isinstance(num_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_splits' argument to "
        "'read_variable_xla_split_nd' Op, not %r." % num_splits)
  num_splits = [_execute.make_int(_i, "num_splits") for _i in num_splits]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'read_variable_xla_split_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = ("T", T, "N", N, "num_splits", num_splits, "paddings", paddings)
  _result = _execute.execute(b"ReadVariableXlaSplitND", N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReadVariableXlaSplitND", _inputs_flat, _attrs, _result)
  return _result


def recv_tpu_embedding_activations(num_outputs: int, config: str, name=None):
  r"""An op that receives embedding activations on the TPU.

  The TPU system performs the embedding lookups and aggregations specified by
  the arguments to TPUEmbeddingEnqueue(Integer/Sparse/SparseTensor)Batch. The
  results of these aggregations are visible to the Tensorflow Graph as the
  outputs of a RecvTPUEmbeddingActivations op. This op returns a list containing
  one Tensor of activations per table specified in the model. There can be at
  most one RecvTPUEmbeddingActivations op in the TPU graph.

  Args:
    num_outputs: An `int` that is `>= 1`.
      The number of output activation tensors, equal to the number of
      embedding tables in the model.
    config: A `string`. Serialized TPUEmbeddingConfiguration proto.
    name: A name for the operation (optional).

  Returns:
    A list of `num_outputs` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RecvTPUEmbeddingActivations", name, "num_outputs", num_outputs,
        "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return recv_tpu_embedding_activations_eager_fallback(
          num_outputs=num_outputs, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_outputs = _execute.make_int(num_outputs, "num_outputs")
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RecvTPUEmbeddingActivations", num_outputs=num_outputs, config=config,
                                       name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("num_outputs", _op._get_attr_int("num_outputs"), "config",
              _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result

RecvTPUEmbeddingActivations = tf_export("raw_ops.RecvTPUEmbeddingActivations")(_ops.to_raw_op(recv_tpu_embedding_activations))


def recv_tpu_embedding_activations_eager_fallback(num_outputs: int, config: str, name, ctx):
  num_outputs = _execute.make_int(num_outputs, "num_outputs")
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("num_outputs", num_outputs, "config", config)
  _result = _execute.execute(b"RecvTPUEmbeddingActivations", num_outputs,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result

_RetrieveTPUEmbeddingADAMParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingADAMParameters",
    ["parameters", "momenta", "velocities"])


def retrieve_tpu_embedding_adam_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve ADAM embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, momenta, velocities).

    parameters: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    velocities: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingADAMParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingADAMParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_adam_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingADAMParameters", num_shards=num_shards,
                                              shard_id=shard_id,
                                              table_id=table_id,
                                              table_name=table_name,
                                              config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingADAMParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingADAMParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingADAMParameters = tf_export("raw_ops.RetrieveTPUEmbeddingADAMParameters")(_ops.to_raw_op(retrieve_tpu_embedding_adam_parameters))


def retrieve_tpu_embedding_adam_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingADAMParameters", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingADAMParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingADAMParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingAdadeltaParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingAdadeltaParameters",
    ["parameters", "accumulators", "updates"])


def retrieve_tpu_embedding_adadelta_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve Adadelta embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators, updates).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
    updates: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingAdadeltaParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingAdadeltaParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_adadelta_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingAdadeltaParameters", num_shards=num_shards,
                                                  shard_id=shard_id,
                                                  table_id=table_id,
                                                  table_name=table_name,
                                                  config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdadeltaParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdadeltaParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingAdadeltaParameters = tf_export("raw_ops.RetrieveTPUEmbeddingAdadeltaParameters")(_ops.to_raw_op(retrieve_tpu_embedding_adadelta_parameters))


def retrieve_tpu_embedding_adadelta_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingAdadeltaParameters", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdadeltaParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdadeltaParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingAdagradMomentumParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingAdagradMomentumParameters",
    ["parameters", "accumulators", "momenta"])


def retrieve_tpu_embedding_adagrad_momentum_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve Adagrad Momentum embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators, momenta).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingAdagradMomentumParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingAdagradMomentumParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_adagrad_momentum_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingAdagradMomentumParameters", num_shards=num_shards,
                                                         shard_id=shard_id,
                                                         table_id=table_id,
                                                         table_name=table_name,
                                                         config=config,
                                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdagradMomentumParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdagradMomentumParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingAdagradMomentumParameters = tf_export("raw_ops.RetrieveTPUEmbeddingAdagradMomentumParameters")(_ops.to_raw_op(retrieve_tpu_embedding_adagrad_momentum_parameters))


def retrieve_tpu_embedding_adagrad_momentum_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingAdagradMomentumParameters",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdagradMomentumParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdagradMomentumParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingAdagradParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingAdagradParameters",
    ["parameters", "accumulators"])


def retrieve_tpu_embedding_adagrad_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve Adagrad embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingAdagradParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingAdagradParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_adagrad_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingAdagradParameters", num_shards=num_shards,
                                                 shard_id=shard_id,
                                                 table_id=table_id,
                                                 table_name=table_name,
                                                 config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdagradParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdagradParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingAdagradParameters = tf_export("raw_ops.RetrieveTPUEmbeddingAdagradParameters")(_ops.to_raw_op(retrieve_tpu_embedding_adagrad_parameters))


def retrieve_tpu_embedding_adagrad_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingAdagradParameters", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingAdagradParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingAdagradParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingCenteredRMSPropParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingCenteredRMSPropParameters",
    ["parameters", "ms", "mom", "mg"])


def retrieve_tpu_embedding_centered_rms_prop_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve centered RMSProp embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, ms, mom, mg).

    parameters: A `Tensor` of type `float32`.
    ms: A `Tensor` of type `float32`.
    mom: A `Tensor` of type `float32`.
    mg: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingCenteredRMSPropParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingCenteredRMSPropParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_centered_rms_prop_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingCenteredRMSPropParameters", num_shards=num_shards,
                                                         shard_id=shard_id,
                                                         table_id=table_id,
                                                         table_name=table_name,
                                                         config=config,
                                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingCenteredRMSPropParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingCenteredRMSPropParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingCenteredRMSPropParameters = tf_export("raw_ops.RetrieveTPUEmbeddingCenteredRMSPropParameters")(_ops.to_raw_op(retrieve_tpu_embedding_centered_rms_prop_parameters))


def retrieve_tpu_embedding_centered_rms_prop_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingCenteredRMSPropParameters",
                             4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingCenteredRMSPropParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingCenteredRMSPropParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingFTRLParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingFTRLParameters",
    ["parameters", "accumulators", "linears"])


def retrieve_tpu_embedding_ftrl_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve FTRL embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators, linears).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
    linears: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingFTRLParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingFTRLParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_ftrl_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingFTRLParameters", num_shards=num_shards,
                                              shard_id=shard_id,
                                              table_id=table_id,
                                              table_name=table_name,
                                              config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingFTRLParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingFTRLParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingFTRLParameters = tf_export("raw_ops.RetrieveTPUEmbeddingFTRLParameters")(_ops.to_raw_op(retrieve_tpu_embedding_ftrl_parameters))


def retrieve_tpu_embedding_ftrl_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingFTRLParameters", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingFTRLParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingFTRLParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingFrequencyEstimatorParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingFrequencyEstimatorParameters",
    ["parameters", "last_hit_step"])


def retrieve_tpu_embedding_frequency_estimator_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve frequency estimator embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, last_hit_step).

    parameters: A `Tensor` of type `float32`.
    last_hit_step: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingFrequencyEstimatorParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingFrequencyEstimatorParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_frequency_estimator_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingFrequencyEstimatorParameters", num_shards=num_shards,
                                                            shard_id=shard_id,
                                                            table_id=table_id,
                                                            table_name=table_name,
                                                            config=config,
                                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingFrequencyEstimatorParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingFrequencyEstimatorParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingFrequencyEstimatorParameters = tf_export("raw_ops.RetrieveTPUEmbeddingFrequencyEstimatorParameters")(_ops.to_raw_op(retrieve_tpu_embedding_frequency_estimator_parameters))


def retrieve_tpu_embedding_frequency_estimator_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingFrequencyEstimatorParameters",
                             2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingFrequencyEstimatorParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingFrequencyEstimatorParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingMDLAdagradLightParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingMDLAdagradLightParameters",
    ["parameters", "accumulators", "weights", "benefits"])


def retrieve_tpu_embedding_mdl_adagrad_light_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve MDL Adagrad Light embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators, weights, benefits).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
    weights: A `Tensor` of type `float32`.
    benefits: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingMDLAdagradLightParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingMDLAdagradLightParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_mdl_adagrad_light_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingMDLAdagradLightParameters", num_shards=num_shards,
                                                         shard_id=shard_id,
                                                         table_id=table_id,
                                                         table_name=table_name,
                                                         config=config,
                                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingMDLAdagradLightParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingMDLAdagradLightParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingMDLAdagradLightParameters = tf_export("raw_ops.RetrieveTPUEmbeddingMDLAdagradLightParameters")(_ops.to_raw_op(retrieve_tpu_embedding_mdl_adagrad_light_parameters))


def retrieve_tpu_embedding_mdl_adagrad_light_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingMDLAdagradLightParameters",
                             4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingMDLAdagradLightParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingMDLAdagradLightParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingMomentumParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingMomentumParameters",
    ["parameters", "momenta"])


def retrieve_tpu_embedding_momentum_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve Momentum embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, momenta).

    parameters: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingMomentumParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingMomentumParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_momentum_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingMomentumParameters", num_shards=num_shards,
                                                  shard_id=shard_id,
                                                  table_id=table_id,
                                                  table_name=table_name,
                                                  config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingMomentumParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingMomentumParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingMomentumParameters = tf_export("raw_ops.RetrieveTPUEmbeddingMomentumParameters")(_ops.to_raw_op(retrieve_tpu_embedding_momentum_parameters))


def retrieve_tpu_embedding_momentum_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingMomentumParameters", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingMomentumParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingMomentumParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingProximalAdagradParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingProximalAdagradParameters",
    ["parameters", "accumulators"])


def retrieve_tpu_embedding_proximal_adagrad_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve proximal Adagrad embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, accumulators).

    parameters: A `Tensor` of type `float32`.
    accumulators: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingProximalAdagradParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingProximalAdagradParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_proximal_adagrad_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingProximalAdagradParameters", num_shards=num_shards,
                                                         shard_id=shard_id,
                                                         table_id=table_id,
                                                         table_name=table_name,
                                                         config=config,
                                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingProximalAdagradParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingProximalAdagradParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingProximalAdagradParameters = tf_export("raw_ops.RetrieveTPUEmbeddingProximalAdagradParameters")(_ops.to_raw_op(retrieve_tpu_embedding_proximal_adagrad_parameters))


def retrieve_tpu_embedding_proximal_adagrad_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingProximalAdagradParameters",
                             2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingProximalAdagradParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingProximalAdagradParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingProximalYogiParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingProximalYogiParameters",
    ["parameters", "v", "m"])


def retrieve_tpu_embedding_proximal_yogi_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""TODO: add doc.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, v, m).

    parameters: A `Tensor` of type `float32`.
    v: A `Tensor` of type `float32`.
    m: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingProximalYogiParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingProximalYogiParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_proximal_yogi_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingProximalYogiParameters", num_shards=num_shards,
                                                      shard_id=shard_id,
                                                      table_id=table_id,
                                                      table_name=table_name,
                                                      config=config,
                                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingProximalYogiParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingProximalYogiParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingProximalYogiParameters = tf_export("raw_ops.RetrieveTPUEmbeddingProximalYogiParameters")(_ops.to_raw_op(retrieve_tpu_embedding_proximal_yogi_parameters))


def retrieve_tpu_embedding_proximal_yogi_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingProximalYogiParameters", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingProximalYogiParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingProximalYogiParametersOutput._make(_result)
  return _result

_RetrieveTPUEmbeddingRMSPropParametersOutput = collections.namedtuple(
    "RetrieveTPUEmbeddingRMSPropParameters",
    ["parameters", "ms", "mom"])


def retrieve_tpu_embedding_rms_prop_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None):
  r"""Retrieve RMSProp embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, ms, mom).

    parameters: A `Tensor` of type `float32`.
    ms: A `Tensor` of type `float32`.
    mom: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingRMSPropParameters", name, "table_id",
        table_id, "table_name", table_name, "num_shards", num_shards,
        "shard_id", shard_id, "config", config)
      _result = _RetrieveTPUEmbeddingRMSPropParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_rms_prop_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingRMSPropParameters", num_shards=num_shards,
                                                 shard_id=shard_id,
                                                 table_id=table_id,
                                                 table_name=table_name,
                                                 config=config, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingRMSPropParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingRMSPropParametersOutput._make(_result)
  return _result

RetrieveTPUEmbeddingRMSPropParameters = tf_export("raw_ops.RetrieveTPUEmbeddingRMSPropParameters")(_ops.to_raw_op(retrieve_tpu_embedding_rms_prop_parameters))


def retrieve_tpu_embedding_rms_prop_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx):
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingRMSPropParameters", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingRMSPropParameters", _inputs_flat, _attrs, _result)
  _result = _RetrieveTPUEmbeddingRMSPropParametersOutput._make(_result)
  return _result


def retrieve_tpu_embedding_stochastic_gradient_descent_parameters(num_shards: int, shard_id: int, table_id:int=-1, table_name:str="", config:str="", name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Retrieve SGD embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveTPUEmbeddingStochasticGradientDescentParameters", name,
        "table_id", table_id, "table_name", table_name, "num_shards",
        num_shards, "shard_id", shard_id, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return retrieve_tpu_embedding_stochastic_gradient_descent_parameters_eager_fallback(
          table_id=table_id, table_name=table_name, num_shards=num_shards,
          shard_id=shard_id, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveTPUEmbeddingStochasticGradientDescentParameters", num_shards=num_shards,
                                                                   shard_id=shard_id,
                                                                   table_id=table_id,
                                                                   table_name=table_name,
                                                                   config=config,
                                                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "table_name",
              _op.get_attr("table_name"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"), "config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveTPUEmbeddingStochasticGradientDescentParameters", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RetrieveTPUEmbeddingStochasticGradientDescentParameters = tf_export("raw_ops.RetrieveTPUEmbeddingStochasticGradientDescentParameters")(_ops.to_raw_op(retrieve_tpu_embedding_stochastic_gradient_descent_parameters))


def retrieve_tpu_embedding_stochastic_gradient_descent_parameters_eager_fallback(num_shards: int, shard_id: int, table_id: int, table_name: str, config: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  if table_id is None:
    table_id = -1
  table_id = _execute.make_int(table_id, "table_id")
  if table_name is None:
    table_name = ""
  table_name = _execute.make_str(table_name, "table_name")
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("table_id", table_id, "table_name", table_name, "num_shards",
  num_shards, "shard_id", shard_id, "config", config)
  _result = _execute.execute(b"RetrieveTPUEmbeddingStochasticGradientDescentParameters",
                             1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveTPUEmbeddingStochasticGradientDescentParameters", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def send_tpu_embedding_gradients(inputs: Annotated[List[Any], _atypes.Float32], learning_rates: Annotated[List[Any], _atypes.Float32], config: str, name=None):
  r"""Performs gradient updates of embedding tables.

  Args:
    inputs: A list of at least 1 `Tensor` objects with type `float32`.
      A TensorList of gradients with which to update embedding tables.
      This argument has the same length and shapes as the return value of
      RecvTPUEmbeddingActivations, but contains gradients of the model's loss
      with respect to the embedding activations. The embedding tables are updated
      from these gradients via the optimizer specified in the TPU embedding
      configuration given to tpu.initialize_system.
    learning_rates: A list of `Tensor` objects with type `float32`.
      A TensorList of float32 scalars, one for each dynamic learning
      rate tag: see the comments in
      //third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
      Multiple tables can share the same dynamic learning rate tag as specified
      in the configuration. If the learning rates for all tables are constant,
      this list should be empty.
    config: A `string`. Serialized TPUEmbeddingConfiguration proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SendTPUEmbeddingGradients", name, inputs, learning_rates,
        "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return send_tpu_embedding_gradients_eager_fallback(
          inputs, learning_rates, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NN = len(learning_rates)
  config = _execute.make_str(config, "config")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SendTPUEmbeddingGradients", inputs=inputs,
                                     learning_rates=learning_rates,
                                     config=config, name=name)
  return _op
SendTPUEmbeddingGradients = tf_export("raw_ops.SendTPUEmbeddingGradients")(_ops.to_raw_op(send_tpu_embedding_gradients))


def send_tpu_embedding_gradients_eager_fallback(inputs: Annotated[List[Any], _atypes.Float32], learning_rates: Annotated[List[Any], _atypes.Float32], config: str, name, ctx):
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NN = len(learning_rates)
  config = _execute.make_str(config, "config")
  inputs = _ops.convert_n_to_tensor(inputs, _dtypes.float32)
  learning_rates = _ops.convert_n_to_tensor(learning_rates, _dtypes.float32)
  _inputs_flat = list(inputs) + list(learning_rates)
  _attrs = ("N", _attr_N, "NN", _attr_NN, "config", config)
  _result = _execute.execute(b"SendTPUEmbeddingGradients", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def shutdown_distributed_tpu(name=None):
  r"""Shuts down a running distributed TPU system.

  The op returns an error if no system is running.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShutdownDistributedTPU", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shutdown_distributed_tpu_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShutdownDistributedTPU", name=name)
  return _op
ShutdownDistributedTPU = tf_export("raw_ops.ShutdownDistributedTPU")(_ops.to_raw_op(shutdown_distributed_tpu))


def shutdown_distributed_tpu_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"ShutdownDistributedTPU", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def tpu_compilation_result(name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the result of a TPU compilation.

  This operation returns the result of a TPU compilation as a serialized
  CompilationResultProto, which holds a status and an error message if an error
  occurred during compilation.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUCompilationResult", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_compilation_result_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUCompilationResult", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUCompilationResult", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUCompilationResult = tf_export("raw_ops.TPUCompilationResult")(_ops.to_raw_op(tpu_compilation_result))


def tpu_compilation_result_eager_fallback(name, ctx) -> Annotated[Any, _atypes.String]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"TPUCompilationResult", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUCompilationResult", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tpu_embedding_activations(embedding_variable: Annotated[Any, _atypes.Float32], sliced_activations: Annotated[Any, _atypes.Float32], table_id: int, lookup_id: int, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""An op enabling differentiation of TPU Embeddings.

  This op simply returns its first input, which is assumed to have been sliced
  from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of
  this op, and its first argument being a trainable Variable, enables automatic
  differentiation of graphs containing embeddings via the TPU Embedding Python
  libraries.

  Args:
    embedding_variable: A `Tensor` of type `float32`.
      A trainable variable, enabling optimizers to find this op.
    sliced_activations: A `Tensor` of type `float32`.
      The embedding activations Tensor to return.
    table_id: An `int` that is `>= 0`.
      The id of the table in the embedding layer configuration from which
      these activations were computed.
    lookup_id: An `int` that is `>= 0`.
      Identifier of the set of embedding indices which produced these
      activations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUEmbeddingActivations", name, embedding_variable,
        sliced_activations, "table_id", table_id, "lookup_id", lookup_id)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_embedding_activations_eager_fallback(
          embedding_variable, sliced_activations, table_id=table_id,
          lookup_id=lookup_id, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  table_id = _execute.make_int(table_id, "table_id")
  lookup_id = _execute.make_int(lookup_id, "lookup_id")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUEmbeddingActivations", embedding_variable=embedding_variable,
                                   sliced_activations=sliced_activations,
                                   table_id=table_id, lookup_id=lookup_id,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "lookup_id",
              _op._get_attr_int("lookup_id"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUEmbeddingActivations = tf_export("raw_ops.TPUEmbeddingActivations")(_ops.to_raw_op(tpu_embedding_activations))


def tpu_embedding_activations_eager_fallback(embedding_variable: Annotated[Any, _atypes.Float32], sliced_activations: Annotated[Any, _atypes.Float32], table_id: int, lookup_id: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  table_id = _execute.make_int(table_id, "table_id")
  lookup_id = _execute.make_int(lookup_id, "lookup_id")
  embedding_variable = _ops.convert_to_tensor(embedding_variable, _dtypes.float32)
  sliced_activations = _ops.convert_to_tensor(sliced_activations, _dtypes.float32)
  _inputs_flat = [embedding_variable, sliced_activations]
  _attrs = ("table_id", table_id, "lookup_id", lookup_id)
  _result = _execute.execute(b"TPUEmbeddingActivations", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tpu_ordinal_selector(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""A TPU core selector Op.

  This Op produces a set of TPU cores (for warm-up) or a single TPU core
  (for regular inference) to execute the TPU program on. The output is
  consumed by TPUPartitionedCall.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUOrdinalSelector", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_ordinal_selector_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUOrdinalSelector", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUOrdinalSelector", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUOrdinalSelector = tf_export("raw_ops.TPUOrdinalSelector")(_ops.to_raw_op(tpu_ordinal_selector))


def tpu_ordinal_selector_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"TPUOrdinalSelector", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUOrdinalSelector", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tpu_partitioned_call(args, device_ordinal: Annotated[Any, _atypes.Int32], Tout, f, autotuner_thresh:int=0, name=None):
  r"""Calls a function placed on a specified TPU device.

  Args:
    args: A list of `Tensor` objects. The arguments to the function.
    device_ordinal: A `Tensor` of type `int32`.
      The TPU device ordinal to run the function on.
    Tout: A list of `tf.DTypes`. The types of the outputs of the function.
    f: A function decorated with @Defun. The function to call.
    autotuner_thresh: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedCall", name, args, device_ordinal, "Tout", Tout,
        "f", f, "autotuner_thresh", autotuner_thresh)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_call_eager_fallback(
          args, device_ordinal, Tout=Tout, f=f,
          autotuner_thresh=autotuner_thresh, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'tpu_partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if autotuner_thresh is None:
    autotuner_thresh = 0
  autotuner_thresh = _execute.make_int(autotuner_thresh, "autotuner_thresh")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedCall", args=args, device_ordinal=device_ordinal,
                              Tout=Tout, f=f,
                              autotuner_thresh=autotuner_thresh, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"), "autotuner_thresh",
              _op._get_attr_int("autotuner_thresh"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedCall", _inputs_flat, _attrs, _result)
  return _result

TPUPartitionedCall = tf_export("raw_ops.TPUPartitionedCall")(_ops.to_raw_op(tpu_partitioned_call))


def tpu_partitioned_call_eager_fallback(args, device_ordinal: Annotated[Any, _atypes.Int32], Tout, f, autotuner_thresh: int, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'tpu_partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if autotuner_thresh is None:
    autotuner_thresh = 0
  autotuner_thresh = _execute.make_int(autotuner_thresh, "autotuner_thresh")
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = list(args) + [device_ordinal]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f, "autotuner_thresh",
  autotuner_thresh)
  _result = _execute.execute(b"TPUPartitionedCall", len(Tout),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedCall", _inputs_flat, _attrs, _result)
  return _result


def tpu_replicate_metadata(num_replicas: int, num_cores_per_replica:int=1, topology:str="", use_tpu:bool=True, device_assignment=[], computation_shape=[], host_compute_core=[], padding_map=[], step_marker_location:str="STEP_MARK_AT_ENTRY", allow_soft_placement:bool=False, use_spmd_for_xla_partitioning:bool=False, tpu_compile_options_proto:str="", name=None):
  r"""Metadata indicating how the TPU computation should be replicated.

  This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.

  Args:
    num_replicas: An `int` that is `>= 0`.
      Number of replicas of the computation
    num_cores_per_replica: An optional `int`. Defaults to `1`.
      Number of cores per replica. Used for model parallelism.
    topology: An optional `string`. Defaults to `""`.
      TopologyProto indicating the topology of the TPU pod slice.
    use_tpu: An optional `bool`. Defaults to `True`.
      Whether to place the computation on the TPU.
    device_assignment: An optional list of `ints`. Defaults to `[]`.
      The assignment of devices for the computation.
    computation_shape: An optional list of `ints`. Defaults to `[]`.
      DEPRECATED. Use num_cores_per_replica instead.
    host_compute_core: An optional list of `strings`. Defaults to `[]`.
    padding_map: An optional list of `strings`. Defaults to `[]`.
    step_marker_location: An optional `string`. Defaults to `"STEP_MARK_AT_ENTRY"`.
    allow_soft_placement: An optional `bool`. Defaults to `False`.
    use_spmd_for_xla_partitioning: An optional `bool`. Defaults to `False`.
    tpu_compile_options_proto: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUReplicateMetadata", name, "num_replicas", num_replicas,
        "num_cores_per_replica", num_cores_per_replica, "topology", topology,
        "use_tpu", use_tpu, "device_assignment", device_assignment,
        "computation_shape", computation_shape, "host_compute_core",
        host_compute_core, "padding_map", padding_map, "step_marker_location",
        step_marker_location, "allow_soft_placement", allow_soft_placement,
        "use_spmd_for_xla_partitioning", use_spmd_for_xla_partitioning,
        "tpu_compile_options_proto", tpu_compile_options_proto)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_replicate_metadata_eager_fallback(
          num_replicas=num_replicas,
          num_cores_per_replica=num_cores_per_replica, topology=topology,
          use_tpu=use_tpu, device_assignment=device_assignment,
          computation_shape=computation_shape,
          host_compute_core=host_compute_core, padding_map=padding_map,
          step_marker_location=step_marker_location,
          allow_soft_placement=allow_soft_placement,
          use_spmd_for_xla_partitioning=use_spmd_for_xla_partitioning,
          tpu_compile_options_proto=tpu_compile_options_proto, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  if num_cores_per_replica is None:
    num_cores_per_replica = 1
  num_cores_per_replica = _execute.make_int(num_cores_per_replica, "num_cores_per_replica")
  if topology is None:
    topology = ""
  topology = _execute.make_str(topology, "topology")
  if use_tpu is None:
    use_tpu = True
  use_tpu = _execute.make_bool(use_tpu, "use_tpu")
  if device_assignment is None:
    device_assignment = []
  if not isinstance(device_assignment, (list, tuple)):
    raise TypeError(
        "Expected list for 'device_assignment' argument to "
        "'tpu_replicate_metadata' Op, not %r." % device_assignment)
  device_assignment = [_execute.make_int(_i, "device_assignment") for _i in device_assignment]
  if computation_shape is None:
    computation_shape = []
  if not isinstance(computation_shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'computation_shape' argument to "
        "'tpu_replicate_metadata' Op, not %r." % computation_shape)
  computation_shape = [_execute.make_int(_i, "computation_shape") for _i in computation_shape]
  if host_compute_core is None:
    host_compute_core = []
  if not isinstance(host_compute_core, (list, tuple)):
    raise TypeError(
        "Expected list for 'host_compute_core' argument to "
        "'tpu_replicate_metadata' Op, not %r." % host_compute_core)
  host_compute_core = [_execute.make_str(_s, "host_compute_core") for _s in host_compute_core]
  if padding_map is None:
    padding_map = []
  if not isinstance(padding_map, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_map' argument to "
        "'tpu_replicate_metadata' Op, not %r." % padding_map)
  padding_map = [_execute.make_str(_s, "padding_map") for _s in padding_map]
  if step_marker_location is None:
    step_marker_location = "STEP_MARK_AT_ENTRY"
  step_marker_location = _execute.make_str(step_marker_location, "step_marker_location")
  if allow_soft_placement is None:
    allow_soft_placement = False
  allow_soft_placement = _execute.make_bool(allow_soft_placement, "allow_soft_placement")
  if use_spmd_for_xla_partitioning is None:
    use_spmd_for_xla_partitioning = False
  use_spmd_for_xla_partitioning = _execute.make_bool(use_spmd_for_xla_partitioning, "use_spmd_for_xla_partitioning")
  if tpu_compile_options_proto is None:
    tpu_compile_options_proto = ""
  tpu_compile_options_proto = _execute.make_str(tpu_compile_options_proto, "tpu_compile_options_proto")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUReplicateMetadata", num_replicas=num_replicas,
                                num_cores_per_replica=num_cores_per_replica,
                                topology=topology, use_tpu=use_tpu,
                                device_assignment=device_assignment,
                                computation_shape=computation_shape,
                                host_compute_core=host_compute_core,
                                padding_map=padding_map,
                                step_marker_location=step_marker_location,
                                allow_soft_placement=allow_soft_placement,
                                use_spmd_for_xla_partitioning=use_spmd_for_xla_partitioning,
                                tpu_compile_options_proto=tpu_compile_options_proto,
                                name=name)
  return _op
TPUReplicateMetadata = tf_export("raw_ops.TPUReplicateMetadata")(_ops.to_raw_op(tpu_replicate_metadata))


def tpu_replicate_metadata_eager_fallback(num_replicas: int, num_cores_per_replica: int, topology: str, use_tpu: bool, device_assignment, computation_shape, host_compute_core, padding_map, step_marker_location: str, allow_soft_placement: bool, use_spmd_for_xla_partitioning: bool, tpu_compile_options_proto: str, name, ctx):
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  if num_cores_per_replica is None:
    num_cores_per_replica = 1
  num_cores_per_replica = _execute.make_int(num_cores_per_replica, "num_cores_per_replica")
  if topology is None:
    topology = ""
  topology = _execute.make_str(topology, "topology")
  if use_tpu is None:
    use_tpu = True
  use_tpu = _execute.make_bool(use_tpu, "use_tpu")
  if device_assignment is None:
    device_assignment = []
  if not isinstance(device_assignment, (list, tuple)):
    raise TypeError(
        "Expected list for 'device_assignment' argument to "
        "'tpu_replicate_metadata' Op, not %r." % device_assignment)
  device_assignment = [_execute.make_int(_i, "device_assignment") for _i in device_assignment]
  if computation_shape is None:
    computation_shape = []
  if not isinstance(computation_shape, (list, tuple)):
    raise TypeError(
        "Expected list for 'computation_shape' argument to "
        "'tpu_replicate_metadata' Op, not %r." % computation_shape)
  computation_shape = [_execute.make_int(_i, "computation_shape") for _i in computation_shape]
  if host_compute_core is None:
    host_compute_core = []
  if not isinstance(host_compute_core, (list, tuple)):
    raise TypeError(
        "Expected list for 'host_compute_core' argument to "
        "'tpu_replicate_metadata' Op, not %r." % host_compute_core)
  host_compute_core = [_execute.make_str(_s, "host_compute_core") for _s in host_compute_core]
  if padding_map is None:
    padding_map = []
  if not isinstance(padding_map, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_map' argument to "
        "'tpu_replicate_metadata' Op, not %r." % padding_map)
  padding_map = [_execute.make_str(_s, "padding_map") for _s in padding_map]
  if step_marker_location is None:
    step_marker_location = "STEP_MARK_AT_ENTRY"
  step_marker_location = _execute.make_str(step_marker_location, "step_marker_location")
  if allow_soft_placement is None:
    allow_soft_placement = False
  allow_soft_placement = _execute.make_bool(allow_soft_placement, "allow_soft_placement")
  if use_spmd_for_xla_partitioning is None:
    use_spmd_for_xla_partitioning = False
  use_spmd_for_xla_partitioning = _execute.make_bool(use_spmd_for_xla_partitioning, "use_spmd_for_xla_partitioning")
  if tpu_compile_options_proto is None:
    tpu_compile_options_proto = ""
  tpu_compile_options_proto = _execute.make_str(tpu_compile_options_proto, "tpu_compile_options_proto")
  _inputs_flat = []
  _attrs = ("num_replicas", num_replicas, "num_cores_per_replica",
  num_cores_per_replica, "topology", topology, "use_tpu", use_tpu,
  "device_assignment", device_assignment, "computation_shape",
  computation_shape, "host_compute_core", host_compute_core, "padding_map",
  padding_map, "step_marker_location", step_marker_location,
  "allow_soft_placement", allow_soft_placement,
  "use_spmd_for_xla_partitioning", use_spmd_for_xla_partitioning,
  "tpu_compile_options_proto", tpu_compile_options_proto)
  _result = _execute.execute(b"TPUReplicateMetadata", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_TPUReplicatedInput_T = TypeVar("TV_TPUReplicatedInput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_replicated_input(inputs: Annotated[List[Any], TV_TPUReplicatedInput_T], is_mirrored_variable:bool=False, index:int=-1, is_packed:bool=False, name=None) -> Annotated[Any, TV_TPUReplicatedInput_T]:
  r"""Connects N inputs to an N-way replicated TPU computation.

  This operation holds a replicated input to a `tpu.replicate()` computation subgraph.
  Each replicated input has the same shape and type alongside the output.

  For example:
  ```
  %a = "tf.opA"()
  %b = "tf.opB"()
  %replicated_input = "tf.TPUReplicatedInput"(%a, %b)
  %computation = "tf.Computation"(%replicated_input)
  ```
  The above computation has a replicated input of two replicas.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
    is_mirrored_variable: An optional `bool`. Defaults to `False`.
    index: An optional `int`. Defaults to `-1`.
    is_packed: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUReplicatedInput", name, inputs, "is_mirrored_variable",
        is_mirrored_variable, "index", index, "is_packed", is_packed)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_replicated_input_eager_fallback(
          inputs, is_mirrored_variable=is_mirrored_variable, index=index,
          is_packed=is_packed, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_replicated_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if is_mirrored_variable is None:
    is_mirrored_variable = False
  is_mirrored_variable = _execute.make_bool(is_mirrored_variable, "is_mirrored_variable")
  if index is None:
    index = -1
  index = _execute.make_int(index, "index")
  if is_packed is None:
    is_packed = False
  is_packed = _execute.make_bool(is_packed, "is_packed")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUReplicatedInput", inputs=inputs,
                              is_mirrored_variable=is_mirrored_variable,
                              index=index, is_packed=is_packed, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "is_mirrored_variable",
              _op._get_attr_bool("is_mirrored_variable"), "index",
              _op._get_attr_int("index"), "is_packed",
              _op._get_attr_bool("is_packed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUReplicatedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUReplicatedInput = tf_export("raw_ops.TPUReplicatedInput")(_ops.to_raw_op(tpu_replicated_input))


def tpu_replicated_input_eager_fallback(inputs: Annotated[List[Any], TV_TPUReplicatedInput_T], is_mirrored_variable: bool, index: int, is_packed: bool, name, ctx) -> Annotated[Any, TV_TPUReplicatedInput_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_replicated_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if is_mirrored_variable is None:
    is_mirrored_variable = False
  is_mirrored_variable = _execute.make_bool(is_mirrored_variable, "is_mirrored_variable")
  if index is None:
    index = -1
  index = _execute.make_int(index, "index")
  if is_packed is None:
    is_packed = False
  is_packed = _execute.make_bool(is_packed, "is_packed")
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T, "is_mirrored_variable",
  is_mirrored_variable, "index", index, "is_packed", is_packed)
  _result = _execute.execute(b"TPUReplicatedInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUReplicatedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TPUReplicatedOutput_T = TypeVar("TV_TPUReplicatedOutput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tpu_replicated_output(input: Annotated[Any, TV_TPUReplicatedOutput_T], num_replicas: int, name=None):
  r"""Connects N outputs from an N-way replicated TPU computation.

  This operation holds a replicated output from a `tpu.replicate()` computation subgraph.
  Each replicated output has the same shape and type alongside the input.

  For example:
  ```
  %computation = "tf.Computation"()
  %replicated_output:2 = "tf.TPUReplicatedOutput"(%computation)
  ```
  The above computation has a replicated output of two replicas.

  Args:
    input: A `Tensor`.
    num_replicas: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_replicas` `Tensor` objects with the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUReplicatedOutput", name, input, "num_replicas",
        num_replicas)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_replicated_output_eager_fallback(
          input, num_replicas=num_replicas, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUReplicatedOutput", input=input, num_replicas=num_replicas,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_replicas", _op._get_attr_int("num_replicas"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUReplicatedOutput", _inputs_flat, _attrs, _result)
  return _result

TPUReplicatedOutput = tf_export("raw_ops.TPUReplicatedOutput")(_ops.to_raw_op(tpu_replicated_output))


def tpu_replicated_output_eager_fallback(input: Annotated[Any, TV_TPUReplicatedOutput_T], num_replicas: int, name, ctx):
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("num_replicas", num_replicas, "T", _attr_T)
  _result = _execute.execute(b"TPUReplicatedOutput", num_replicas,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUReplicatedOutput", _inputs_flat, _attrs, _result)
  return _result


def worker_heartbeat(request: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Worker heartbeat op.

  Heartbeats may be sent periodically to indicate the coordinator is still active,
  to retrieve the current worker status and to expedite shutdown when necessary.

  Args:
    request: A `Tensor` of type `string`.
      A string tensor containing a serialized WorkerHeartbeatRequest
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WorkerHeartbeat", name, request)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return worker_heartbeat_eager_fallback(
          request, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WorkerHeartbeat", request=request, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WorkerHeartbeat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WorkerHeartbeat = tf_export("raw_ops.WorkerHeartbeat")(_ops.to_raw_op(worker_heartbeat))


def worker_heartbeat_eager_fallback(request: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  request = _ops.convert_to_tensor(request, _dtypes.string)
  _inputs_flat = [request]
  _attrs = None
  _result = _execute.execute(b"WorkerHeartbeat", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WorkerHeartbeat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaConcatND_T = TypeVar("TV_XlaConcatND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def xla_concat_nd(inputs: Annotated[List[Any], TV_XlaConcatND_T], num_concats, paddings=[], name=None) -> Annotated[Any, TV_XlaConcatND_T]:
  r"""Concats input tensor across all dimensions.

  An op which merges slices the input tensor based on the given num_splits
  attribute, strips paddings optionally, and returns the merged tensor without
  paddings.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1],
   [4, 5]]
  [[2, 3],
   [6, 7]]
  [[8, 9],
   [12, 13]]
  [[10, 11],
   [14, 15]]
  ```
  `num_splits`:
  ```
  [2, 2]
  ```
  and `paddings`:
  ```
  [1, 1]
  ```
  the expected `outputs` is:
  ```
  [[0, 1, 2],
   [4, 5, 6],
   [8, 9, 10]]
  ```

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      Input tensor slices in row-major order to merge across all dimensions. All
      inputs must have the same shape.
    num_concats: A list of `ints`. Number of ways to merge per dimension.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension to strip from the final merged
      tensor. These paddings must not exceed the dimension size of the merged result
      prior to stripping paddings.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaConcatND", name, inputs, "num_concats", num_concats,
        "paddings", paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_concat_nd_eager_fallback(
          inputs, num_concats=num_concats, paddings=paddings, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'xla_concat_nd' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(num_concats, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_concats' argument to "
        "'xla_concat_nd' Op, not %r." % num_concats)
  num_concats = [_execute.make_int(_i, "num_concats") for _i in num_concats]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'xla_concat_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaConcatND", inputs=inputs, num_concats=num_concats,
                       paddings=paddings, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"),
              "num_concats", _op.get_attr("num_concats"), "paddings",
              _op.get_attr("paddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaConcatND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaConcatND = tf_export("raw_ops.XlaConcatND")(_ops.to_raw_op(xla_concat_nd))


def xla_concat_nd_eager_fallback(inputs: Annotated[List[Any], TV_XlaConcatND_T], num_concats, paddings, name, ctx) -> Annotated[Any, TV_XlaConcatND_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'xla_concat_nd' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(num_concats, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_concats' argument to "
        "'xla_concat_nd' Op, not %r." % num_concats)
  num_concats = [_execute.make_int(_i, "num_concats") for _i in num_concats]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'xla_concat_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  _inputs_flat = list(inputs)
  _attrs = ("T", _attr_T, "N", _attr_N, "num_concats", num_concats,
  "paddings", paddings)
  _result = _execute.execute(b"XlaConcatND", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaConcatND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSplitND_T = TypeVar("TV_XlaSplitND_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def xla_split_nd(input: Annotated[Any, TV_XlaSplitND_T], N: int, num_splits, paddings=[], name=None):
  r"""Splits input tensor across all dimensions.

  An op which slices the input tensor based on the given num_splits attribute,
  pads slices optionally, and returned the slices. Slices are returned in
  row-major order.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1, 2],
   [3, 4, 5],
   [6, 7, 8]]
  ```
  `num_splits`:
  ```
  [2, 2]
  ```
  and `paddings`:
  ```
  [1, 1]
  ```
  the expected `outputs` is:
  ```
  [[0, 1],
   [3, 4]]
  [[2, 0],
   [5, 0]]
  [[6, 7],
   [0, 0]]
  [[8, 0],
   [0, 0]]
  ```

  Args:
    input: A `Tensor`. Input tensor to split across all dimensions.
    N: An `int` that is `>= 1`.
    num_splits: A list of `ints`.
      Number of ways to split per dimension. Shape dimensions must be evenly
      divisible.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension of input tensor to apply before
      splitting. This can be used to make a dimension evenly divisible.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSplitND", name, input, "N", N, "num_splits", num_splits,
        "paddings", paddings)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_split_nd_eager_fallback(
          input, N=N, num_splits=num_splits, paddings=paddings, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  N = _execute.make_int(N, "N")
  if not isinstance(num_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_splits' argument to "
        "'xla_split_nd' Op, not %r." % num_splits)
  num_splits = [_execute.make_int(_i, "num_splits") for _i in num_splits]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'xla_split_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSplitND", input=input, N=N, num_splits=num_splits,
                      paddings=paddings, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"),
              "num_splits", _op.get_attr("num_splits"), "paddings",
              _op.get_attr("paddings"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSplitND", _inputs_flat, _attrs, _result)
  return _result

XlaSplitND = tf_export("raw_ops.XlaSplitND")(_ops.to_raw_op(xla_split_nd))


def xla_split_nd_eager_fallback(input: Annotated[Any, TV_XlaSplitND_T], N: int, num_splits, paddings, name, ctx):
  N = _execute.make_int(N, "N")
  if not isinstance(num_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_splits' argument to "
        "'xla_split_nd' Op, not %r." % num_splits)
  num_splits = [_execute.make_int(_i, "num_splits") for _i in num_splits]
  if paddings is None:
    paddings = []
  if not isinstance(paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'paddings' argument to "
        "'xla_split_nd' Op, not %r." % paddings)
  paddings = [_execute.make_int(_i, "paddings") for _i in paddings]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "N", N, "num_splits", num_splits, "paddings",
  paddings)
  _result = _execute.execute(b"XlaSplitND", N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSplitND", _inputs_flat, _attrs, _result)
  return _result

