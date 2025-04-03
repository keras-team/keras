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

def assert_cardinality_dataset(input_dataset: Annotated[Any, _atypes.Variant], cardinality: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    cardinality: A `Tensor` of type `int64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AssertCardinalityDataset", name, input_dataset, cardinality,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return assert_cardinality_dataset_eager_fallback(
          input_dataset, cardinality, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_cardinality_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_cardinality_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AssertCardinalityDataset", input_dataset=input_dataset,
                                    cardinality=cardinality,
                                    output_types=output_types,
                                    output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AssertCardinalityDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AssertCardinalityDataset = tf_export("raw_ops.AssertCardinalityDataset")(_ops.to_raw_op(assert_cardinality_dataset))


def assert_cardinality_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], cardinality: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_cardinality_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_cardinality_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cardinality = _ops.convert_to_tensor(cardinality, _dtypes.int64)
  _inputs_flat = [input_dataset, cardinality]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AssertCardinalityDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AssertCardinalityDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def assert_next_dataset(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A transformation that asserts which transformations happen next.

  This transformation checks whether the camel-case names (i.e. "FlatMap", not
  "flat_map") of the transformations following this transformation match the list
  of names in the `transformations` argument. If there is a mismatch, the
  transformation raises an exception.

  The check occurs when iterating over the contents of the dataset, which
  means that the check happens *after* any static optimizations are applied
  to the dataset graph.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
      `AssertNextDataset` passes through the outputs of its input dataset.
    transformations: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying the transformations that are
      expected to happen next.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AssertNextDataset", name, input_dataset, transformations,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return assert_next_dataset_eager_fallback(
          input_dataset, transformations, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_next_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_next_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AssertNextDataset", input_dataset=input_dataset,
                             transformations=transformations,
                             output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AssertNextDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AssertNextDataset = tf_export("raw_ops.AssertNextDataset")(_ops.to_raw_op(assert_next_dataset))


def assert_next_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_next_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_next_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  transformations = _ops.convert_to_tensor(transformations, _dtypes.string)
  _inputs_flat = [input_dataset, transformations]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AssertNextDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AssertNextDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def assert_prev_dataset(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A transformation that asserts which transformations happened previously.

  This transformation checks the names and, optionally, the attribute name-value
  pairs in the `transformations` argument against those of the transformations
  that preceded this transformation.  If there is a mismatch, the transformation
  raises an exception.

  The check occurs when iterating over the contents of the dataset, which
  means that the check happens *after* any static optimizations are applied
  to the dataset graph.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
      `AssertPrevDataset` passes through the outputs of its input dataset.
    transformations: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying the transformations, with optional
      attribute name-value pairs, that are expected to have happened previously.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AssertPrevDataset", name, input_dataset, transformations,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return assert_prev_dataset_eager_fallback(
          input_dataset, transformations, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_prev_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_prev_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AssertPrevDataset", input_dataset=input_dataset,
                             transformations=transformations,
                             output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AssertPrevDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AssertPrevDataset = tf_export("raw_ops.AssertPrevDataset")(_ops.to_raw_op(assert_prev_dataset))


def assert_prev_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'assert_prev_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'assert_prev_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  transformations = _ops.convert_to_tensor(transformations, _dtypes.string)
  _inputs_flat = [input_dataset, transformations]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AssertPrevDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AssertPrevDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def auto_shard_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_workers: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, auto_shard_policy:int=0, num_replicas:int=0, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that shards the input dataset.

  Creates a dataset that shards the input dataset by num_workers, returning a
  sharded dataset for the index-th worker. This attempts to automatically shard
  a dataset by examining the Dataset graph and inserting a shard op before the
  inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).

  This dataset will throw a NotFound error if we cannot shard the dataset
  automatically.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    num_workers: A `Tensor` of type `int64`.
      A scalar representing the number of workers to distribute this dataset across.
    index: A `Tensor` of type `int64`.
      A scalar representing the index of the current worker out of num_workers.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    auto_shard_policy: An optional `int`. Defaults to `0`.
    num_replicas: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AutoShardDataset", name, input_dataset, num_workers, index,
        "auto_shard_policy", auto_shard_policy, "output_types", output_types,
        "output_shapes", output_shapes, "num_replicas", num_replicas)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return auto_shard_dataset_eager_fallback(
          input_dataset, num_workers, index,
          auto_shard_policy=auto_shard_policy, output_types=output_types,
          output_shapes=output_shapes, num_replicas=num_replicas, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'auto_shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'auto_shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if auto_shard_policy is None:
    auto_shard_policy = 0
  auto_shard_policy = _execute.make_int(auto_shard_policy, "auto_shard_policy")
  if num_replicas is None:
    num_replicas = 0
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AutoShardDataset", input_dataset=input_dataset,
                            num_workers=num_workers, index=index,
                            output_types=output_types,
                            output_shapes=output_shapes,
                            auto_shard_policy=auto_shard_policy,
                            num_replicas=num_replicas, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("auto_shard_policy", _op._get_attr_int("auto_shard_policy"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "num_replicas",
              _op._get_attr_int("num_replicas"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AutoShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AutoShardDataset = tf_export("raw_ops.AutoShardDataset")(_ops.to_raw_op(auto_shard_dataset))


def auto_shard_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_workers: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, auto_shard_policy: int, num_replicas: int, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'auto_shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'auto_shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if auto_shard_policy is None:
    auto_shard_policy = 0
  auto_shard_policy = _execute.make_int(auto_shard_policy, "auto_shard_policy")
  if num_replicas is None:
    num_replicas = 0
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_workers = _ops.convert_to_tensor(num_workers, _dtypes.int64)
  index = _ops.convert_to_tensor(index, _dtypes.int64)
  _inputs_flat = [input_dataset, num_workers, index]
  _attrs = ("auto_shard_policy", auto_shard_policy, "output_types",
  output_types, "output_shapes", output_shapes, "num_replicas", num_replicas)
  _result = _execute.execute(b"AutoShardDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AutoShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def bytes_produced_stats_dataset(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Records the bytes size of each element of `input_dataset` in a StatsAggregator.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    tag: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BytesProducedStatsDataset", name, input_dataset, tag,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bytes_produced_stats_dataset_eager_fallback(
          input_dataset, tag, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'bytes_produced_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'bytes_produced_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BytesProducedStatsDataset", input_dataset=input_dataset, tag=tag,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BytesProducedStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BytesProducedStatsDataset = tf_export("raw_ops.BytesProducedStatsDataset")(_ops.to_raw_op(bytes_produced_stats_dataset))


def bytes_produced_stats_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'bytes_produced_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'bytes_produced_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [input_dataset, tag]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"BytesProducedStatsDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BytesProducedStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def csv_dataset(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    compression_type: A `Tensor` of type `string`.
    buffer_size: A `Tensor` of type `int64`.
    header: A `Tensor` of type `bool`.
    field_delim: A `Tensor` of type `string`.
    use_quote_delim: A `Tensor` of type `bool`.
    na_value: A `Tensor` of type `string`.
    select_cols: A `Tensor` of type `int64`.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `float64`, `int32`, `int64`, `string`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CSVDataset", name, filenames, compression_type, buffer_size,
        header, field_delim, use_quote_delim, na_value, select_cols,
        record_defaults, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return csv_dataset_eager_fallback(
          filenames, compression_type, buffer_size, header, field_delim,
          use_quote_delim, na_value, select_cols, record_defaults,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'csv_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CSVDataset", filenames=filenames, compression_type=compression_type,
                      buffer_size=buffer_size, header=header,
                      field_delim=field_delim,
                      use_quote_delim=use_quote_delim, na_value=na_value,
                      select_cols=select_cols,
                      record_defaults=record_defaults,
                      output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CSVDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CSVDataset = tf_export("raw_ops.CSVDataset")(_ops.to_raw_op(csv_dataset))


def csv_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'csv_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_output_types, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  header = _ops.convert_to_tensor(header, _dtypes.bool)
  field_delim = _ops.convert_to_tensor(field_delim, _dtypes.string)
  use_quote_delim = _ops.convert_to_tensor(use_quote_delim, _dtypes.bool)
  na_value = _ops.convert_to_tensor(na_value, _dtypes.string)
  select_cols = _ops.convert_to_tensor(select_cols, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols] + list(record_defaults)
  _attrs = ("output_types", _attr_output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"CSVDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CSVDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def csv_dataset_v2(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, exclude_cols: Annotated[Any, _atypes.Int64], output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    compression_type: A `Tensor` of type `string`.
    buffer_size: A `Tensor` of type `int64`.
    header: A `Tensor` of type `bool`.
    field_delim: A `Tensor` of type `string`.
    use_quote_delim: A `Tensor` of type `bool`.
    na_value: A `Tensor` of type `string`.
    select_cols: A `Tensor` of type `int64`.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `float64`, `int32`, `int64`, `string`.
    exclude_cols: A `Tensor` of type `int64`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CSVDatasetV2", name, filenames, compression_type, buffer_size,
        header, field_delim, use_quote_delim, na_value, select_cols,
        record_defaults, exclude_cols, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return csv_dataset_v2_eager_fallback(
          filenames, compression_type, buffer_size, header, field_delim,
          use_quote_delim, na_value, select_cols, record_defaults,
          exclude_cols, output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'csv_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CSVDatasetV2", filenames=filenames,
                        compression_type=compression_type,
                        buffer_size=buffer_size, header=header,
                        field_delim=field_delim,
                        use_quote_delim=use_quote_delim, na_value=na_value,
                        select_cols=select_cols,
                        record_defaults=record_defaults,
                        exclude_cols=exclude_cols,
                        output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CSVDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CSVDatasetV2 = tf_export("raw_ops.CSVDatasetV2")(_ops.to_raw_op(csv_dataset_v2))


def csv_dataset_v2_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, exclude_cols: Annotated[Any, _atypes.Int64], output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'csv_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_output_types, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  header = _ops.convert_to_tensor(header, _dtypes.bool)
  field_delim = _ops.convert_to_tensor(field_delim, _dtypes.string)
  use_quote_delim = _ops.convert_to_tensor(use_quote_delim, _dtypes.bool)
  na_value = _ops.convert_to_tensor(na_value, _dtypes.string)
  select_cols = _ops.convert_to_tensor(select_cols, _dtypes.int64)
  exclude_cols = _ops.convert_to_tensor(exclude_cols, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols] + list(record_defaults) + [exclude_cols]
  _attrs = ("output_types", _attr_output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"CSVDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CSVDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CheckPinned_T = TypeVar("TV_CheckPinned_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('check_pinned')
def check_pinned(tensor: Annotated[Any, TV_CheckPinned_T], name=None) -> Annotated[Any, TV_CheckPinned_T]:
  r"""Checks whether a tensor is located in host memory pinned for GPU.

  When run:
  - Reports an `InvalidArgument` error if `tensor` is not in pinned memory.
  - Reports a `FailedPrecondition` error if not built with CUDA.

  Args:
    tensor: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CheckPinned", name, tensor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_check_pinned(
          (tensor, name,), None)
      if _result is not NotImplemented:
        return _result
      return check_pinned_eager_fallback(
          tensor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            check_pinned, (), dict(tensor=tensor, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_check_pinned(
        (tensor, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CheckPinned", tensor=tensor, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          check_pinned, (), dict(tensor=tensor, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CheckPinned", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CheckPinned = tf_export("raw_ops.CheckPinned")(_ops.to_raw_op(check_pinned))
_dispatcher_for_check_pinned = check_pinned._tf_type_based_dispatcher.Dispatch


def check_pinned_eager_fallback(tensor: Annotated[Any, TV_CheckPinned_T], name, ctx) -> Annotated[Any, TV_CheckPinned_T]:
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"CheckPinned", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CheckPinned", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def choose_fastest_branch_dataset(input_dataset: Annotated[Any, _atypes.Variant], ratio_numerator: Annotated[Any, _atypes.Int64], ratio_denominator: Annotated[Any, _atypes.Int64], other_arguments, num_elements_per_branch: int, branches, other_arguments_lengths, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    ratio_numerator: A `Tensor` of type `int64`.
    ratio_denominator: A `Tensor` of type `int64`.
    other_arguments: A list of `Tensor` objects.
    num_elements_per_branch: An `int` that is `>= 1`.
    branches: A list of functions decorated with @Defun that has length `>= 1`.
    other_arguments_lengths: A list of `ints` that has length `>= 1`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ChooseFastestBranchDataset", name, input_dataset,
        ratio_numerator, ratio_denominator, other_arguments,
        "num_elements_per_branch", num_elements_per_branch, "branches",
        branches, "other_arguments_lengths", other_arguments_lengths,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return choose_fastest_branch_dataset_eager_fallback(
          input_dataset, ratio_numerator, ratio_denominator, other_arguments,
          num_elements_per_branch=num_elements_per_branch, branches=branches,
          other_arguments_lengths=other_arguments_lengths,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_elements_per_branch = _execute.make_int(num_elements_per_branch, "num_elements_per_branch")
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % branches)
  if not isinstance(other_arguments_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'other_arguments_lengths' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % other_arguments_lengths)
  other_arguments_lengths = [_execute.make_int(_i, "other_arguments_lengths") for _i in other_arguments_lengths]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ChooseFastestBranchDataset", input_dataset=input_dataset,
                                      ratio_numerator=ratio_numerator,
                                      ratio_denominator=ratio_denominator,
                                      other_arguments=other_arguments,
                                      num_elements_per_branch=num_elements_per_branch,
                                      branches=branches,
                                      other_arguments_lengths=other_arguments_lengths,
                                      output_types=output_types,
                                      output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Targuments", _op.get_attr("Targuments"),
              "num_elements_per_branch",
              _op._get_attr_int("num_elements_per_branch"), "branches",
              _op.get_attr("branches"), "other_arguments_lengths",
              _op.get_attr("other_arguments_lengths"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ChooseFastestBranchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ChooseFastestBranchDataset = tf_export("raw_ops.ChooseFastestBranchDataset")(_ops.to_raw_op(choose_fastest_branch_dataset))


def choose_fastest_branch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], ratio_numerator: Annotated[Any, _atypes.Int64], ratio_denominator: Annotated[Any, _atypes.Int64], other_arguments, num_elements_per_branch: int, branches, other_arguments_lengths, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  num_elements_per_branch = _execute.make_int(num_elements_per_branch, "num_elements_per_branch")
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % branches)
  if not isinstance(other_arguments_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'other_arguments_lengths' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % other_arguments_lengths)
  other_arguments_lengths = [_execute.make_int(_i, "other_arguments_lengths") for _i in other_arguments_lengths]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'choose_fastest_branch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  ratio_numerator = _ops.convert_to_tensor(ratio_numerator, _dtypes.int64)
  ratio_denominator = _ops.convert_to_tensor(ratio_denominator, _dtypes.int64)
  _inputs_flat = [input_dataset, ratio_numerator, ratio_denominator] + list(other_arguments)
  _attrs = ("Targuments", _attr_Targuments, "num_elements_per_branch",
  num_elements_per_branch, "branches", branches, "other_arguments_lengths",
  other_arguments_lengths, "output_types", output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"ChooseFastestBranchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ChooseFastestBranchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def choose_fastest_dataset(input_datasets: Annotated[List[Any], _atypes.Variant], num_experiments: int, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_datasets: A list of at least 2 `Tensor` objects with type `variant`.
    num_experiments: An `int`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ChooseFastestDataset", name, input_datasets, "num_experiments",
        num_experiments, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return choose_fastest_dataset_eager_fallback(
          input_datasets, num_experiments=num_experiments,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'choose_fastest_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  num_experiments = _execute.make_int(num_experiments, "num_experiments")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'choose_fastest_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'choose_fastest_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ChooseFastestDataset", input_datasets=input_datasets,
                                num_experiments=num_experiments,
                                output_types=output_types,
                                output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "num_experiments",
              _op._get_attr_int("num_experiments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ChooseFastestDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ChooseFastestDataset = tf_export("raw_ops.ChooseFastestDataset")(_ops.to_raw_op(choose_fastest_dataset))


def choose_fastest_dataset_eager_fallback(input_datasets: Annotated[List[Any], _atypes.Variant], num_experiments: int, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'choose_fastest_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  num_experiments = _execute.make_int(num_experiments, "num_experiments")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'choose_fastest_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'choose_fastest_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_datasets = _ops.convert_n_to_tensor(input_datasets, _dtypes.variant)
  _inputs_flat = list(input_datasets)
  _attrs = ("N", _attr_N, "num_experiments", num_experiments, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ChooseFastestDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ChooseFastestDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def compress_element(components, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Compresses a dataset element.

  Args:
    components: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CompressElement", name, components)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return compress_element_eager_fallback(
          components, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CompressElement", components=components, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_types", _op.get_attr("input_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CompressElement", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CompressElement = tf_export("raw_ops.CompressElement")(_ops.to_raw_op(compress_element))


def compress_element_eager_fallback(components, name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_input_types, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  _inputs_flat = list(components)
  _attrs = ("input_types", _attr_input_types)
  _result = _execute.execute(b"CompressElement", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CompressElement", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def compute_batch_size(input_dataset: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Computes the static batch size of a dataset sans partial batches.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ComputeBatchSize", name, input_dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return compute_batch_size_eager_fallback(
          input_dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ComputeBatchSize", input_dataset=input_dataset, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ComputeBatchSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ComputeBatchSize = tf_export("raw_ops.ComputeBatchSize")(_ops.to_raw_op(compute_batch_size))


def compute_batch_size_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int64]:
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = None
  _result = _execute.execute(b"ComputeBatchSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ComputeBatchSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def data_service_dataset(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms:int=-1, data_transfer_protocol:str="", target_workers:str="AUTO", cross_trainer_cache_options:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that reads data from the tf.data service.

  Args:
    dataset_id: A `Tensor` of type `int64`.
    processing_mode: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    job_name: A `Tensor` of type `string`.
    max_outstanding_requests: A `Tensor` of type `int64`.
    iteration_counter: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    task_refresh_interval_hint_ms: An optional `int`. Defaults to `-1`.
    data_transfer_protocol: An optional `string`. Defaults to `""`.
    target_workers: An optional `string`. Defaults to `"AUTO"`.
    cross_trainer_cache_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataServiceDataset", name, dataset_id, processing_mode,
        address, protocol, job_name, max_outstanding_requests,
        iteration_counter, "task_refresh_interval_hint_ms",
        task_refresh_interval_hint_ms, "output_types", output_types,
        "output_shapes", output_shapes, "data_transfer_protocol",
        data_transfer_protocol, "target_workers", target_workers,
        "cross_trainer_cache_options", cross_trainer_cache_options)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_service_dataset_eager_fallback(
          dataset_id, processing_mode, address, protocol, job_name,
          max_outstanding_requests, iteration_counter,
          task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
          output_types=output_types, output_shapes=output_shapes,
          data_transfer_protocol=data_transfer_protocol,
          target_workers=target_workers,
          cross_trainer_cache_options=cross_trainer_cache_options, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataServiceDataset", dataset_id=dataset_id,
                              processing_mode=processing_mode,
                              address=address, protocol=protocol,
                              job_name=job_name,
                              max_outstanding_requests=max_outstanding_requests,
                              iteration_counter=iteration_counter,
                              output_types=output_types,
                              output_shapes=output_shapes,
                              task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
                              data_transfer_protocol=data_transfer_protocol,
                              target_workers=target_workers,
                              cross_trainer_cache_options=cross_trainer_cache_options,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("task_refresh_interval_hint_ms",
              _op._get_attr_int("task_refresh_interval_hint_ms"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "data_transfer_protocol",
              _op.get_attr("data_transfer_protocol"), "target_workers",
              _op.get_attr("target_workers"), "cross_trainer_cache_options",
              _op.get_attr("cross_trainer_cache_options"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataServiceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataServiceDataset = tf_export("raw_ops.DataServiceDataset")(_ops.to_raw_op(data_service_dataset))


def data_service_dataset_eager_fallback(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms: int, data_transfer_protocol: str, target_workers: str, cross_trainer_cache_options: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  dataset_id = _ops.convert_to_tensor(dataset_id, _dtypes.int64)
  processing_mode = _ops.convert_to_tensor(processing_mode, _dtypes.string)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  job_name = _ops.convert_to_tensor(job_name, _dtypes.string)
  max_outstanding_requests = _ops.convert_to_tensor(max_outstanding_requests, _dtypes.int64)
  iteration_counter = _ops.convert_to_tensor(iteration_counter, _dtypes.resource)
  _inputs_flat = [dataset_id, processing_mode, address, protocol, job_name, max_outstanding_requests, iteration_counter]
  _attrs = ("task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
  "output_types", output_types, "output_shapes", output_shapes,
  "data_transfer_protocol", data_transfer_protocol, "target_workers",
  target_workers, "cross_trainer_cache_options", cross_trainer_cache_options)
  _result = _execute.execute(b"DataServiceDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataServiceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def data_service_dataset_v2(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms:int=-1, data_transfer_protocol:str="", target_workers:str="AUTO", cross_trainer_cache_options:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that reads data from the tf.data service.

  Args:
    dataset_id: A `Tensor` of type `int64`.
    processing_mode: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    job_name: A `Tensor` of type `string`.
    consumer_index: A `Tensor` of type `int64`.
    num_consumers: A `Tensor` of type `int64`.
    max_outstanding_requests: A `Tensor` of type `int64`.
    iteration_counter: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    task_refresh_interval_hint_ms: An optional `int`. Defaults to `-1`.
    data_transfer_protocol: An optional `string`. Defaults to `""`.
    target_workers: An optional `string`. Defaults to `"AUTO"`.
    cross_trainer_cache_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataServiceDatasetV2", name, dataset_id, processing_mode,
        address, protocol, job_name, consumer_index, num_consumers,
        max_outstanding_requests, iteration_counter,
        "task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
        "output_types", output_types, "output_shapes", output_shapes,
        "data_transfer_protocol", data_transfer_protocol, "target_workers",
        target_workers, "cross_trainer_cache_options",
        cross_trainer_cache_options)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_service_dataset_v2_eager_fallback(
          dataset_id, processing_mode, address, protocol, job_name,
          consumer_index, num_consumers, max_outstanding_requests,
          iteration_counter,
          task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
          output_types=output_types, output_shapes=output_shapes,
          data_transfer_protocol=data_transfer_protocol,
          target_workers=target_workers,
          cross_trainer_cache_options=cross_trainer_cache_options, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataServiceDatasetV2", dataset_id=dataset_id,
                                processing_mode=processing_mode,
                                address=address, protocol=protocol,
                                job_name=job_name,
                                consumer_index=consumer_index,
                                num_consumers=num_consumers,
                                max_outstanding_requests=max_outstanding_requests,
                                iteration_counter=iteration_counter,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
                                data_transfer_protocol=data_transfer_protocol,
                                target_workers=target_workers,
                                cross_trainer_cache_options=cross_trainer_cache_options,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("task_refresh_interval_hint_ms",
              _op._get_attr_int("task_refresh_interval_hint_ms"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "data_transfer_protocol",
              _op.get_attr("data_transfer_protocol"), "target_workers",
              _op.get_attr("target_workers"), "cross_trainer_cache_options",
              _op.get_attr("cross_trainer_cache_options"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataServiceDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataServiceDatasetV2 = tf_export("raw_ops.DataServiceDatasetV2")(_ops.to_raw_op(data_service_dataset_v2))


def data_service_dataset_v2_eager_fallback(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms: int, data_transfer_protocol: str, target_workers: str, cross_trainer_cache_options: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  dataset_id = _ops.convert_to_tensor(dataset_id, _dtypes.int64)
  processing_mode = _ops.convert_to_tensor(processing_mode, _dtypes.string)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  job_name = _ops.convert_to_tensor(job_name, _dtypes.string)
  consumer_index = _ops.convert_to_tensor(consumer_index, _dtypes.int64)
  num_consumers = _ops.convert_to_tensor(num_consumers, _dtypes.int64)
  max_outstanding_requests = _ops.convert_to_tensor(max_outstanding_requests, _dtypes.int64)
  iteration_counter = _ops.convert_to_tensor(iteration_counter, _dtypes.resource)
  _inputs_flat = [dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter]
  _attrs = ("task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
  "output_types", output_types, "output_shapes", output_shapes,
  "data_transfer_protocol", data_transfer_protocol, "target_workers",
  target_workers, "cross_trainer_cache_options", cross_trainer_cache_options)
  _result = _execute.execute(b"DataServiceDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataServiceDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def data_service_dataset_v3(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, uncompress_fn, task_refresh_interval_hint_ms:int=-1, data_transfer_protocol:str="", target_workers:str="AUTO", uncompress:bool=False, cross_trainer_cache_options:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that reads data from the tf.data service.

  Args:
    dataset_id: A `Tensor` of type `int64`.
    processing_mode: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    job_name: A `Tensor` of type `string`.
    consumer_index: A `Tensor` of type `int64`.
    num_consumers: A `Tensor` of type `int64`.
    max_outstanding_requests: A `Tensor` of type `int64`.
    iteration_counter: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    uncompress_fn: A function decorated with @Defun.
    task_refresh_interval_hint_ms: An optional `int`. Defaults to `-1`.
    data_transfer_protocol: An optional `string`. Defaults to `""`.
    target_workers: An optional `string`. Defaults to `"AUTO"`.
    uncompress: An optional `bool`. Defaults to `False`.
    cross_trainer_cache_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataServiceDatasetV3", name, dataset_id, processing_mode,
        address, protocol, job_name, consumer_index, num_consumers,
        max_outstanding_requests, iteration_counter,
        "task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
        "output_types", output_types, "output_shapes", output_shapes,
        "data_transfer_protocol", data_transfer_protocol, "target_workers",
        target_workers, "uncompress", uncompress, "uncompress_fn",
        uncompress_fn, "cross_trainer_cache_options",
        cross_trainer_cache_options)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_service_dataset_v3_eager_fallback(
          dataset_id, processing_mode, address, protocol, job_name,
          consumer_index, num_consumers, max_outstanding_requests,
          iteration_counter,
          task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
          output_types=output_types, output_shapes=output_shapes,
          data_transfer_protocol=data_transfer_protocol,
          target_workers=target_workers, uncompress=uncompress,
          uncompress_fn=uncompress_fn,
          cross_trainer_cache_options=cross_trainer_cache_options, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if uncompress is None:
    uncompress = False
  uncompress = _execute.make_bool(uncompress, "uncompress")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataServiceDatasetV3", dataset_id=dataset_id,
                                processing_mode=processing_mode,
                                address=address, protocol=protocol,
                                job_name=job_name,
                                consumer_index=consumer_index,
                                num_consumers=num_consumers,
                                max_outstanding_requests=max_outstanding_requests,
                                iteration_counter=iteration_counter,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                uncompress_fn=uncompress_fn,
                                task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
                                data_transfer_protocol=data_transfer_protocol,
                                target_workers=target_workers,
                                uncompress=uncompress,
                                cross_trainer_cache_options=cross_trainer_cache_options,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("task_refresh_interval_hint_ms",
              _op._get_attr_int("task_refresh_interval_hint_ms"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "data_transfer_protocol",
              _op.get_attr("data_transfer_protocol"), "target_workers",
              _op.get_attr("target_workers"), "uncompress",
              _op._get_attr_bool("uncompress"), "uncompress_fn",
              _op.get_attr("uncompress_fn"), "cross_trainer_cache_options",
              _op.get_attr("cross_trainer_cache_options"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataServiceDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataServiceDatasetV3 = tf_export("raw_ops.DataServiceDatasetV3")(_ops.to_raw_op(data_service_dataset_v3))


def data_service_dataset_v3_eager_fallback(dataset_id: Annotated[Any, _atypes.Int64], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, uncompress_fn, task_refresh_interval_hint_ms: int, data_transfer_protocol: str, target_workers: str, uncompress: bool, cross_trainer_cache_options: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if uncompress is None:
    uncompress = False
  uncompress = _execute.make_bool(uncompress, "uncompress")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  dataset_id = _ops.convert_to_tensor(dataset_id, _dtypes.int64)
  processing_mode = _ops.convert_to_tensor(processing_mode, _dtypes.string)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  job_name = _ops.convert_to_tensor(job_name, _dtypes.string)
  consumer_index = _ops.convert_to_tensor(consumer_index, _dtypes.int64)
  num_consumers = _ops.convert_to_tensor(num_consumers, _dtypes.int64)
  max_outstanding_requests = _ops.convert_to_tensor(max_outstanding_requests, _dtypes.int64)
  iteration_counter = _ops.convert_to_tensor(iteration_counter, _dtypes.resource)
  _inputs_flat = [dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter]
  _attrs = ("task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
  "output_types", output_types, "output_shapes", output_shapes,
  "data_transfer_protocol", data_transfer_protocol, "target_workers",
  target_workers, "uncompress", uncompress, "uncompress_fn", uncompress_fn,
  "cross_trainer_cache_options", cross_trainer_cache_options)
  _result = _execute.execute(b"DataServiceDatasetV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataServiceDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def data_service_dataset_v4(dataset_id: Annotated[Any, _atypes.String], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, uncompress_fn, task_refresh_interval_hint_ms:int=-1, data_transfer_protocol:str="", target_workers:str="AUTO", uncompress:bool=False, cross_trainer_cache_options:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that reads data from the tf.data service.

  Args:
    dataset_id: A `Tensor` of type `string`.
    processing_mode: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    job_name: A `Tensor` of type `string`.
    consumer_index: A `Tensor` of type `int64`.
    num_consumers: A `Tensor` of type `int64`.
    max_outstanding_requests: A `Tensor` of type `int64`.
    iteration_counter: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    uncompress_fn: A function decorated with @Defun.
    task_refresh_interval_hint_ms: An optional `int`. Defaults to `-1`.
    data_transfer_protocol: An optional `string`. Defaults to `""`.
    target_workers: An optional `string`. Defaults to `"AUTO"`.
    uncompress: An optional `bool`. Defaults to `False`.
    cross_trainer_cache_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataServiceDatasetV4", name, dataset_id, processing_mode,
        address, protocol, job_name, consumer_index, num_consumers,
        max_outstanding_requests, iteration_counter,
        "task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
        "output_types", output_types, "output_shapes", output_shapes,
        "data_transfer_protocol", data_transfer_protocol, "target_workers",
        target_workers, "uncompress", uncompress, "uncompress_fn",
        uncompress_fn, "cross_trainer_cache_options",
        cross_trainer_cache_options)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_service_dataset_v4_eager_fallback(
          dataset_id, processing_mode, address, protocol, job_name,
          consumer_index, num_consumers, max_outstanding_requests,
          iteration_counter,
          task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
          output_types=output_types, output_shapes=output_shapes,
          data_transfer_protocol=data_transfer_protocol,
          target_workers=target_workers, uncompress=uncompress,
          uncompress_fn=uncompress_fn,
          cross_trainer_cache_options=cross_trainer_cache_options, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v4' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v4' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if uncompress is None:
    uncompress = False
  uncompress = _execute.make_bool(uncompress, "uncompress")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataServiceDatasetV4", dataset_id=dataset_id,
                                processing_mode=processing_mode,
                                address=address, protocol=protocol,
                                job_name=job_name,
                                consumer_index=consumer_index,
                                num_consumers=num_consumers,
                                max_outstanding_requests=max_outstanding_requests,
                                iteration_counter=iteration_counter,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                uncompress_fn=uncompress_fn,
                                task_refresh_interval_hint_ms=task_refresh_interval_hint_ms,
                                data_transfer_protocol=data_transfer_protocol,
                                target_workers=target_workers,
                                uncompress=uncompress,
                                cross_trainer_cache_options=cross_trainer_cache_options,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("task_refresh_interval_hint_ms",
              _op._get_attr_int("task_refresh_interval_hint_ms"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "data_transfer_protocol",
              _op.get_attr("data_transfer_protocol"), "target_workers",
              _op.get_attr("target_workers"), "uncompress",
              _op._get_attr_bool("uncompress"), "uncompress_fn",
              _op.get_attr("uncompress_fn"), "cross_trainer_cache_options",
              _op.get_attr("cross_trainer_cache_options"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataServiceDatasetV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataServiceDatasetV4 = tf_export("raw_ops.DataServiceDatasetV4")(_ops.to_raw_op(data_service_dataset_v4))


def data_service_dataset_v4_eager_fallback(dataset_id: Annotated[Any, _atypes.String], processing_mode: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], job_name: Annotated[Any, _atypes.String], consumer_index: Annotated[Any, _atypes.Int64], num_consumers: Annotated[Any, _atypes.Int64], max_outstanding_requests: Annotated[Any, _atypes.Int64], iteration_counter: Annotated[Any, _atypes.Resource], output_types, output_shapes, uncompress_fn, task_refresh_interval_hint_ms: int, data_transfer_protocol: str, target_workers: str, uncompress: bool, cross_trainer_cache_options: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'data_service_dataset_v4' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'data_service_dataset_v4' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if task_refresh_interval_hint_ms is None:
    task_refresh_interval_hint_ms = -1
  task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, "task_refresh_interval_hint_ms")
  if data_transfer_protocol is None:
    data_transfer_protocol = ""
  data_transfer_protocol = _execute.make_str(data_transfer_protocol, "data_transfer_protocol")
  if target_workers is None:
    target_workers = "AUTO"
  target_workers = _execute.make_str(target_workers, "target_workers")
  if uncompress is None:
    uncompress = False
  uncompress = _execute.make_bool(uncompress, "uncompress")
  if cross_trainer_cache_options is None:
    cross_trainer_cache_options = ""
  cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, "cross_trainer_cache_options")
  dataset_id = _ops.convert_to_tensor(dataset_id, _dtypes.string)
  processing_mode = _ops.convert_to_tensor(processing_mode, _dtypes.string)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  job_name = _ops.convert_to_tensor(job_name, _dtypes.string)
  consumer_index = _ops.convert_to_tensor(consumer_index, _dtypes.int64)
  num_consumers = _ops.convert_to_tensor(num_consumers, _dtypes.int64)
  max_outstanding_requests = _ops.convert_to_tensor(max_outstanding_requests, _dtypes.int64)
  iteration_counter = _ops.convert_to_tensor(iteration_counter, _dtypes.resource)
  _inputs_flat = [dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter]
  _attrs = ("task_refresh_interval_hint_ms", task_refresh_interval_hint_ms,
  "output_types", output_types, "output_shapes", output_shapes,
  "data_transfer_protocol", data_transfer_protocol, "target_workers",
  target_workers, "uncompress", uncompress, "uncompress_fn", uncompress_fn,
  "cross_trainer_cache_options", cross_trainer_cache_options)
  _result = _execute.execute(b"DataServiceDatasetV4", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataServiceDatasetV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_from_graph(graph_def: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset from the given `graph_def`.

  Creates a dataset from the provided `graph_def`.

  Args:
    graph_def: A `Tensor` of type `string`.
      The graph representation of the dataset (as serialized GraphDef).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetFromGraph", name, graph_def)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_from_graph_eager_fallback(
          graph_def, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetFromGraph", graph_def=graph_def, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetFromGraph", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DatasetFromGraph = tf_export("raw_ops.DatasetFromGraph")(_ops.to_raw_op(dataset_from_graph))


def dataset_from_graph_eager_fallback(graph_def: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Variant]:
  graph_def = _ops.convert_to_tensor(graph_def, _dtypes.string)
  _inputs_flat = [graph_def]
  _attrs = None
  _result = _execute.execute(b"DatasetFromGraph", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetFromGraph", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_to_tf_record(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], name=None):
  r"""Writes the given dataset to the given file using the TFRecord format.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to write.
    filename: A `Tensor` of type `string`.
      A scalar string tensor representing the filename to use.
    compression_type: A `Tensor` of type `string`.
      A scalar string tensor containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetToTFRecord", name, input_dataset, filename,
        compression_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_to_tf_record_eager_fallback(
          input_dataset, filename, compression_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetToTFRecord", input_dataset=input_dataset, filename=filename,
                             compression_type=compression_type, name=name)
  return _op
DatasetToTFRecord = tf_export("raw_ops.DatasetToTFRecord")(_ops.to_raw_op(dataset_to_tf_record))


def dataset_to_tf_record_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], name, ctx):
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  _inputs_flat = [input_dataset, filename, compression_type]
  _attrs = None
  _result = _execute.execute(b"DatasetToTFRecord", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def dense_to_sparse_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], row_shape: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches input elements into a SparseTensor.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A handle to an input dataset. Must have a single component.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    row_shape: A `Tensor` of type `int64`.
      A vector representing the dense shape of each row in the produced
      SparseTensor. The shape may be partially specified, using `-1` to indicate
      that a particular dimension should use the maximum size of all batch elements.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseToSparseBatchDataset", name, input_dataset, batch_size,
        row_shape, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_to_sparse_batch_dataset_eager_fallback(
          input_dataset, batch_size, row_shape, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseToSparseBatchDataset", input_dataset=input_dataset,
                                     batch_size=batch_size,
                                     row_shape=row_shape,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseToSparseBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DenseToSparseBatchDataset = tf_export("raw_ops.DenseToSparseBatchDataset")(_ops.to_raw_op(dense_to_sparse_batch_dataset))


def dense_to_sparse_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], row_shape: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  row_shape = _ops.convert_to_tensor(row_shape, _dtypes.int64)
  _inputs_flat = [input_dataset, batch_size, row_shape]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"DenseToSparseBatchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseToSparseBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def directed_interleave_dataset(selector_input_dataset: Annotated[Any, _atypes.Variant], data_input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, stop_on_empty_dataset:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

  Args:
    selector_input_dataset: A `Tensor` of type `variant`.
      A dataset of scalar `DT_INT64` elements that determines which of the
      `N` data inputs should produce the next output element.
    data_input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
      `N` datasets with the same type that will be interleaved according to
      the values of `selector_input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    stop_on_empty_dataset: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DirectedInterleaveDataset", name, selector_input_dataset,
        data_input_datasets, "output_types", output_types, "output_shapes",
        output_shapes, "stop_on_empty_dataset", stop_on_empty_dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return directed_interleave_dataset_eager_fallback(
          selector_input_dataset, data_input_datasets,
          output_types=output_types, output_shapes=output_shapes,
          stop_on_empty_dataset=stop_on_empty_dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(data_input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'data_input_datasets' argument to "
        "'directed_interleave_dataset' Op, not %r." % data_input_datasets)
  _attr_N = len(data_input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'directed_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'directed_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if stop_on_empty_dataset is None:
    stop_on_empty_dataset = False
  stop_on_empty_dataset = _execute.make_bool(stop_on_empty_dataset, "stop_on_empty_dataset")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DirectedInterleaveDataset", selector_input_dataset=selector_input_dataset,
                                     data_input_datasets=data_input_datasets,
                                     output_types=output_types,
                                     output_shapes=output_shapes,
                                     stop_on_empty_dataset=stop_on_empty_dataset,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"),
              "stop_on_empty_dataset",
              _op._get_attr_bool("stop_on_empty_dataset"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DirectedInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DirectedInterleaveDataset = tf_export("raw_ops.DirectedInterleaveDataset")(_ops.to_raw_op(directed_interleave_dataset))


def directed_interleave_dataset_eager_fallback(selector_input_dataset: Annotated[Any, _atypes.Variant], data_input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, stop_on_empty_dataset: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(data_input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'data_input_datasets' argument to "
        "'directed_interleave_dataset' Op, not %r." % data_input_datasets)
  _attr_N = len(data_input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'directed_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'directed_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if stop_on_empty_dataset is None:
    stop_on_empty_dataset = False
  stop_on_empty_dataset = _execute.make_bool(stop_on_empty_dataset, "stop_on_empty_dataset")
  selector_input_dataset = _ops.convert_to_tensor(selector_input_dataset, _dtypes.variant)
  data_input_datasets = _ops.convert_n_to_tensor(data_input_datasets, _dtypes.variant)
  _inputs_flat = [selector_input_dataset] + list(data_input_datasets)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes, "N",
  _attr_N, "stop_on_empty_dataset", stop_on_empty_dataset)
  _result = _execute.execute(b"DirectedInterleaveDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DirectedInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def distributed_save(dataset: Annotated[Any, _atypes.Variant], directory: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], metadata:str="", name=None):
  r"""TODO: add doc.

  Args:
    dataset: A `Tensor` of type `variant`.
    directory: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DistributedSave", name, dataset, directory, address,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return distributed_save_eager_fallback(
          dataset, directory, address, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DistributedSave", dataset=dataset, directory=directory,
                           address=address, metadata=metadata, name=name)
  return _op
DistributedSave = tf_export("raw_ops.DistributedSave")(_ops.to_raw_op(distributed_save))


def distributed_save_eager_fallback(dataset: Annotated[Any, _atypes.Variant], directory: Annotated[Any, _atypes.String], address: Annotated[Any, _atypes.String], metadata: str, name, ctx):
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  directory = _ops.convert_to_tensor(directory, _dtypes.string)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  _inputs_flat = [dataset, directory, address]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"DistributedSave", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def dummy_iteration_counter(name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DummyIterationCounter", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dummy_iteration_counter_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DummyIterationCounter", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DummyIterationCounter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DummyIterationCounter = tf_export("raw_ops.DummyIterationCounter")(_ops.to_raw_op(dummy_iteration_counter))


def dummy_iteration_counter_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Resource]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"DummyIterationCounter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DummyIterationCounter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_assert_next_dataset(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    transformations: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalAssertNextDataset", name, input_dataset,
        transformations, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_assert_next_dataset_eager_fallback(
          input_dataset, transformations, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_assert_next_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_assert_next_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalAssertNextDataset", input_dataset=input_dataset,
                                         transformations=transformations,
                                         output_types=output_types,
                                         output_shapes=output_shapes,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalAssertNextDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalAssertNextDataset = tf_export("raw_ops.ExperimentalAssertNextDataset")(_ops.to_raw_op(experimental_assert_next_dataset))


def experimental_assert_next_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], transformations: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_assert_next_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_assert_next_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  transformations = _ops.convert_to_tensor(transformations, _dtypes.string)
  _inputs_flat = [input_dataset, transformations]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalAssertNextDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalAssertNextDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_auto_shard_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_workers: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, auto_shard_policy:int=0, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that shards the input dataset.

  Creates a dataset that shards the input dataset by num_workers, returning a
  sharded dataset for the index-th worker. This attempts to automatically shard
  a dataset by examining the Dataset graph and inserting a shard op before the
  inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).

  This dataset will throw a NotFound error if we cannot shard the dataset
  automatically.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    num_workers: A `Tensor` of type `int64`.
      A scalar representing the number of workers to distribute this dataset across.
    index: A `Tensor` of type `int64`.
      A scalar representing the index of the current worker out of num_workers.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    auto_shard_policy: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalAutoShardDataset", name, input_dataset,
        num_workers, index, "auto_shard_policy", auto_shard_policy,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_auto_shard_dataset_eager_fallback(
          input_dataset, num_workers, index,
          auto_shard_policy=auto_shard_policy, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_auto_shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_auto_shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if auto_shard_policy is None:
    auto_shard_policy = 0
  auto_shard_policy = _execute.make_int(auto_shard_policy, "auto_shard_policy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalAutoShardDataset", input_dataset=input_dataset,
                                        num_workers=num_workers, index=index,
                                        output_types=output_types,
                                        output_shapes=output_shapes,
                                        auto_shard_policy=auto_shard_policy,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("auto_shard_policy", _op._get_attr_int("auto_shard_policy"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalAutoShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalAutoShardDataset = tf_export("raw_ops.ExperimentalAutoShardDataset")(_ops.to_raw_op(experimental_auto_shard_dataset))


def experimental_auto_shard_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_workers: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, auto_shard_policy: int, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_auto_shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_auto_shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if auto_shard_policy is None:
    auto_shard_policy = 0
  auto_shard_policy = _execute.make_int(auto_shard_policy, "auto_shard_policy")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_workers = _ops.convert_to_tensor(num_workers, _dtypes.int64)
  index = _ops.convert_to_tensor(index, _dtypes.int64)
  _inputs_flat = [input_dataset, num_workers, index]
  _attrs = ("auto_shard_policy", auto_shard_policy, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalAutoShardDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalAutoShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_bytes_produced_stats_dataset(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Records the bytes size of each element of `input_dataset` in a StatsAggregator.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    tag: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalBytesProducedStatsDataset", name, input_dataset,
        tag, "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_bytes_produced_stats_dataset_eager_fallback(
          input_dataset, tag, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_bytes_produced_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_bytes_produced_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalBytesProducedStatsDataset", input_dataset=input_dataset,
                                                 tag=tag,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalBytesProducedStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalBytesProducedStatsDataset = tf_export("raw_ops.ExperimentalBytesProducedStatsDataset")(_ops.to_raw_op(experimental_bytes_produced_stats_dataset))


def experimental_bytes_produced_stats_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_bytes_produced_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_bytes_produced_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [input_dataset, tag]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalBytesProducedStatsDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalBytesProducedStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_csv_dataset(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    compression_type: A `Tensor` of type `string`.
    buffer_size: A `Tensor` of type `int64`.
    header: A `Tensor` of type `bool`.
    field_delim: A `Tensor` of type `string`.
    use_quote_delim: A `Tensor` of type `bool`.
    na_value: A `Tensor` of type `string`.
    select_cols: A `Tensor` of type `int64`.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `float64`, `int32`, `int64`, `string`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalCSVDataset", name, filenames, compression_type,
        buffer_size, header, field_delim, use_quote_delim, na_value,
        select_cols, record_defaults, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_csv_dataset_eager_fallback(
          filenames, compression_type, buffer_size, header, field_delim,
          use_quote_delim, na_value, select_cols, record_defaults,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_csv_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalCSVDataset", filenames=filenames,
                                  compression_type=compression_type,
                                  buffer_size=buffer_size, header=header,
                                  field_delim=field_delim,
                                  use_quote_delim=use_quote_delim,
                                  na_value=na_value, select_cols=select_cols,
                                  record_defaults=record_defaults,
                                  output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalCSVDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalCSVDataset = tf_export("raw_ops.ExperimentalCSVDataset")(_ops.to_raw_op(experimental_csv_dataset))


def experimental_csv_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], header: Annotated[Any, _atypes.Bool], field_delim: Annotated[Any, _atypes.String], use_quote_delim: Annotated[Any, _atypes.Bool], na_value: Annotated[Any, _atypes.String], select_cols: Annotated[Any, _atypes.Int64], record_defaults, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_csv_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_output_types, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  header = _ops.convert_to_tensor(header, _dtypes.bool)
  field_delim = _ops.convert_to_tensor(field_delim, _dtypes.string)
  use_quote_delim = _ops.convert_to_tensor(use_quote_delim, _dtypes.bool)
  na_value = _ops.convert_to_tensor(na_value, _dtypes.string)
  select_cols = _ops.convert_to_tensor(select_cols, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols] + list(record_defaults)
  _attrs = ("output_types", _attr_output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"ExperimentalCSVDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalCSVDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_choose_fastest_dataset(input_datasets: Annotated[List[Any], _atypes.Variant], num_experiments: int, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_datasets: A list of at least 2 `Tensor` objects with type `variant`.
    num_experiments: An `int`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalChooseFastestDataset", name, input_datasets,
        "num_experiments", num_experiments, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_choose_fastest_dataset_eager_fallback(
          input_datasets, num_experiments=num_experiments,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  num_experiments = _execute.make_int(num_experiments, "num_experiments")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalChooseFastestDataset", input_datasets=input_datasets,
                                            num_experiments=num_experiments,
                                            output_types=output_types,
                                            output_shapes=output_shapes,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "num_experiments",
              _op._get_attr_int("num_experiments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalChooseFastestDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalChooseFastestDataset = tf_export("raw_ops.ExperimentalChooseFastestDataset")(_ops.to_raw_op(experimental_choose_fastest_dataset))


def experimental_choose_fastest_dataset_eager_fallback(input_datasets: Annotated[List[Any], _atypes.Variant], num_experiments: int, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  num_experiments = _execute.make_int(num_experiments, "num_experiments")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_choose_fastest_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_datasets = _ops.convert_n_to_tensor(input_datasets, _dtypes.variant)
  _inputs_flat = list(input_datasets)
  _attrs = ("N", _attr_N, "num_experiments", num_experiments, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalChooseFastestDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalChooseFastestDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_dataset_cardinality(input_dataset: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the cardinality of `input_dataset`.

  Returns the cardinality of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return cardinality for.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalDatasetCardinality", name, input_dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_dataset_cardinality_eager_fallback(
          input_dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalDatasetCardinality", input_dataset=input_dataset,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalDatasetCardinality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalDatasetCardinality = tf_export("raw_ops.ExperimentalDatasetCardinality")(_ops.to_raw_op(experimental_dataset_cardinality))


def experimental_dataset_cardinality_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int64]:
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = None
  _result = _execute.execute(b"ExperimentalDatasetCardinality", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalDatasetCardinality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_dataset_to_tf_record(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], name=None):
  r"""Writes the given dataset to the given file using the TFRecord format.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to write.
    filename: A `Tensor` of type `string`.
      A scalar string tensor representing the filename to use.
    compression_type: A `Tensor` of type `string`.
      A scalar string tensor containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalDatasetToTFRecord", name, input_dataset, filename,
        compression_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_dataset_to_tf_record_eager_fallback(
          input_dataset, filename, compression_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalDatasetToTFRecord", input_dataset=input_dataset,
                                         filename=filename,
                                         compression_type=compression_type,
                                         name=name)
  return _op
ExperimentalDatasetToTFRecord = tf_export("raw_ops.ExperimentalDatasetToTFRecord")(_ops.to_raw_op(experimental_dataset_to_tf_record))


def experimental_dataset_to_tf_record_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], name, ctx):
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  _inputs_flat = [input_dataset, filename, compression_type]
  _attrs = None
  _result = _execute.execute(b"ExperimentalDatasetToTFRecord", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def experimental_dense_to_sparse_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], row_shape: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches input elements into a SparseTensor.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A handle to an input dataset. Must have a single component.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    row_shape: A `Tensor` of type `int64`.
      A vector representing the dense shape of each row in the produced
      SparseTensor. The shape may be partially specified, using `-1` to indicate
      that a particular dimension should use the maximum size of all batch elements.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalDenseToSparseBatchDataset", name, input_dataset,
        batch_size, row_shape, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_dense_to_sparse_batch_dataset_eager_fallback(
          input_dataset, batch_size, row_shape, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_dense_to_sparse_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_dense_to_sparse_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalDenseToSparseBatchDataset", input_dataset=input_dataset,
                                                 batch_size=batch_size,
                                                 row_shape=row_shape,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalDenseToSparseBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalDenseToSparseBatchDataset = tf_export("raw_ops.ExperimentalDenseToSparseBatchDataset")(_ops.to_raw_op(experimental_dense_to_sparse_batch_dataset))


def experimental_dense_to_sparse_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], row_shape: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_dense_to_sparse_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_dense_to_sparse_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  row_shape = _ops.convert_to_tensor(row_shape, _dtypes.int64)
  _inputs_flat = [input_dataset, batch_size, row_shape]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalDenseToSparseBatchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalDenseToSparseBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_directed_interleave_dataset(selector_input_dataset: Annotated[Any, _atypes.Variant], data_input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

  Args:
    selector_input_dataset: A `Tensor` of type `variant`.
      A dataset of scalar `DT_INT64` elements that determines which of the
      `N` data inputs should produce the next output element.
    data_input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
      `N` datasets with the same type that will be interleaved according to
      the values of `selector_input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalDirectedInterleaveDataset", name,
        selector_input_dataset, data_input_datasets, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_directed_interleave_dataset_eager_fallback(
          selector_input_dataset, data_input_datasets,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(data_input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'data_input_datasets' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % data_input_datasets)
  _attr_N = len(data_input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalDirectedInterleaveDataset", selector_input_dataset=selector_input_dataset,
                                                 data_input_datasets=data_input_datasets,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalDirectedInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalDirectedInterleaveDataset = tf_export("raw_ops.ExperimentalDirectedInterleaveDataset")(_ops.to_raw_op(experimental_directed_interleave_dataset))


def experimental_directed_interleave_dataset_eager_fallback(selector_input_dataset: Annotated[Any, _atypes.Variant], data_input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(data_input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'data_input_datasets' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % data_input_datasets)
  _attr_N = len(data_input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  selector_input_dataset = _ops.convert_to_tensor(selector_input_dataset, _dtypes.variant)
  data_input_datasets = _ops.convert_n_to_tensor(data_input_datasets, _dtypes.variant)
  _inputs_flat = [selector_input_dataset] + list(data_input_datasets)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes, "N",
  _attr_N)
  _result = _execute.execute(b"ExperimentalDirectedInterleaveDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalDirectedInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_group_by_reducer_dataset(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, init_func_other_arguments, reduce_func_other_arguments, finalize_func_other_arguments, key_func, init_func, reduce_func, finalize_func, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that computes a group-by on `input_dataset`.

  Creates a dataset that computes a group-by on `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    key_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `key_func`.
    init_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `init_func`.
    reduce_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `reduce_func`.
    finalize_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `finalize_func`.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    init_func: A function decorated with @Defun.
      A function mapping a key of type DT_INT64, concatenated with
      `init_func_other_arguments` to the initial reducer state.
    reduce_func: A function decorated with @Defun.
      A function mapping the current reducer state and an element of `input_dataset`,
      concatenated with `reduce_func_other_arguments` to a new reducer state.
    finalize_func: A function decorated with @Defun.
      A function mapping the final reducer state to an output element.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalGroupByReducerDataset", name, input_dataset,
        key_func_other_arguments, init_func_other_arguments,
        reduce_func_other_arguments, finalize_func_other_arguments,
        "key_func", key_func, "init_func", init_func, "reduce_func",
        reduce_func, "finalize_func", finalize_func, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_group_by_reducer_dataset_eager_fallback(
          input_dataset, key_func_other_arguments, init_func_other_arguments,
          reduce_func_other_arguments, finalize_func_other_arguments,
          key_func=key_func, init_func=init_func, reduce_func=reduce_func,
          finalize_func=finalize_func, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalGroupByReducerDataset", input_dataset=input_dataset,
                                             key_func_other_arguments=key_func_other_arguments,
                                             init_func_other_arguments=init_func_other_arguments,
                                             reduce_func_other_arguments=reduce_func_other_arguments,
                                             finalize_func_other_arguments=finalize_func_other_arguments,
                                             key_func=key_func,
                                             init_func=init_func,
                                             reduce_func=reduce_func,
                                             finalize_func=finalize_func,
                                             output_types=output_types,
                                             output_shapes=output_shapes,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_func", _op.get_attr("key_func"), "init_func",
              _op.get_attr("init_func"), "reduce_func",
              _op.get_attr("reduce_func"), "finalize_func",
              _op.get_attr("finalize_func"), "Tkey_func_other_arguments",
              _op.get_attr("Tkey_func_other_arguments"),
              "Tinit_func_other_arguments",
              _op.get_attr("Tinit_func_other_arguments"),
              "Treduce_func_other_arguments",
              _op.get_attr("Treduce_func_other_arguments"),
              "Tfinalize_func_other_arguments",
              _op.get_attr("Tfinalize_func_other_arguments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalGroupByReducerDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalGroupByReducerDataset = tf_export("raw_ops.ExperimentalGroupByReducerDataset")(_ops.to_raw_op(experimental_group_by_reducer_dataset))


def experimental_group_by_reducer_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, init_func_other_arguments, reduce_func_other_arguments, finalize_func_other_arguments, key_func, init_func, reduce_func, finalize_func, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, ctx)
  _attr_Tinit_func_other_arguments, init_func_other_arguments = _execute.convert_to_mixed_eager_tensors(init_func_other_arguments, ctx)
  _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, ctx)
  _attr_Tfinalize_func_other_arguments, finalize_func_other_arguments = _execute.convert_to_mixed_eager_tensors(finalize_func_other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(init_func_other_arguments) + list(reduce_func_other_arguments) + list(finalize_func_other_arguments)
  _attrs = ("key_func", key_func, "init_func", init_func, "reduce_func",
  reduce_func, "finalize_func", finalize_func, "Tkey_func_other_arguments",
  _attr_Tkey_func_other_arguments, "Tinit_func_other_arguments",
  _attr_Tinit_func_other_arguments, "Treduce_func_other_arguments",
  _attr_Treduce_func_other_arguments, "Tfinalize_func_other_arguments",
  _attr_Tfinalize_func_other_arguments, "output_types", output_types,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalGroupByReducerDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalGroupByReducerDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_group_by_window_dataset(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that computes a windowed group-by on `input_dataset`.

  // TODO(mrry): Support non-int64 keys.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    key_func_other_arguments: A list of `Tensor` objects.
    reduce_func_other_arguments: A list of `Tensor` objects.
    window_size_func_other_arguments: A list of `Tensor` objects.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    reduce_func: A function decorated with @Defun.
    window_size_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalGroupByWindowDataset", name, input_dataset,
        key_func_other_arguments, reduce_func_other_arguments,
        window_size_func_other_arguments, "key_func", key_func, "reduce_func",
        reduce_func, "window_size_func", window_size_func, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_group_by_window_dataset_eager_fallback(
          input_dataset, key_func_other_arguments,
          reduce_func_other_arguments, window_size_func_other_arguments,
          key_func=key_func, reduce_func=reduce_func,
          window_size_func=window_size_func, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalGroupByWindowDataset", input_dataset=input_dataset,
                                            key_func_other_arguments=key_func_other_arguments,
                                            reduce_func_other_arguments=reduce_func_other_arguments,
                                            window_size_func_other_arguments=window_size_func_other_arguments,
                                            key_func=key_func,
                                            reduce_func=reduce_func,
                                            window_size_func=window_size_func,
                                            output_types=output_types,
                                            output_shapes=output_shapes,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_func", _op.get_attr("key_func"), "reduce_func",
              _op.get_attr("reduce_func"), "window_size_func",
              _op.get_attr("window_size_func"), "Tkey_func_other_arguments",
              _op.get_attr("Tkey_func_other_arguments"),
              "Treduce_func_other_arguments",
              _op.get_attr("Treduce_func_other_arguments"),
              "Twindow_size_func_other_arguments",
              _op.get_attr("Twindow_size_func_other_arguments"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalGroupByWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalGroupByWindowDataset = tf_export("raw_ops.ExperimentalGroupByWindowDataset")(_ops.to_raw_op(experimental_group_by_window_dataset))


def experimental_group_by_window_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, ctx)
  _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, ctx)
  _attr_Twindow_size_func_other_arguments, window_size_func_other_arguments = _execute.convert_to_mixed_eager_tensors(window_size_func_other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(reduce_func_other_arguments) + list(window_size_func_other_arguments)
  _attrs = ("key_func", key_func, "reduce_func", reduce_func,
  "window_size_func", window_size_func, "Tkey_func_other_arguments",
  _attr_Tkey_func_other_arguments, "Treduce_func_other_arguments",
  _attr_Treduce_func_other_arguments, "Twindow_size_func_other_arguments",
  _attr_Twindow_size_func_other_arguments, "output_types", output_types,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalGroupByWindowDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalGroupByWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_ignore_errors_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, log_warning:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that contains the elements of `input_dataset` ignoring errors.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    log_warning: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalIgnoreErrorsDataset", name, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes,
        "log_warning", log_warning)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_ignore_errors_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, log_warning=log_warning, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_ignore_errors_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_ignore_errors_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if log_warning is None:
    log_warning = False
  log_warning = _execute.make_bool(log_warning, "log_warning")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalIgnoreErrorsDataset", input_dataset=input_dataset,
                                           output_types=output_types,
                                           output_shapes=output_shapes,
                                           log_warning=log_warning, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "log_warning",
              _op._get_attr_bool("log_warning"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalIgnoreErrorsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalIgnoreErrorsDataset = tf_export("raw_ops.ExperimentalIgnoreErrorsDataset")(_ops.to_raw_op(experimental_ignore_errors_dataset))


def experimental_ignore_errors_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, log_warning: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_ignore_errors_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_ignore_errors_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if log_warning is None:
    log_warning = False
  log_warning = _execute.make_bool(log_warning, "log_warning")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "log_warning", log_warning)
  _result = _execute.execute(b"ExperimentalIgnoreErrorsDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalIgnoreErrorsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_iterator_get_device(resource: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the name of the device on which `resource` has been placed.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalIteratorGetDevice", name, resource)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_iterator_get_device_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalIteratorGetDevice", resource=resource, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalIteratorGetDevice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalIteratorGetDevice = tf_export("raw_ops.ExperimentalIteratorGetDevice")(_ops.to_raw_op(experimental_iterator_get_device))


def experimental_iterator_get_device_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"ExperimentalIteratorGetDevice", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalIteratorGetDevice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_lmdb_dataset(filenames: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalLMDBDataset", name, filenames, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_lmdb_dataset_eager_fallback(
          filenames, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_lmdb_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_lmdb_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalLMDBDataset", filenames=filenames,
                                   output_types=output_types,
                                   output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalLMDBDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalLMDBDataset = tf_export("raw_ops.ExperimentalLMDBDataset")(_ops.to_raw_op(experimental_lmdb_dataset))


def experimental_lmdb_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_lmdb_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_lmdb_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  _inputs_flat = [filenames]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalLMDBDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalLMDBDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_latency_stats_dataset(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Records the latency of producing `input_dataset` elements in a StatsAggregator.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    tag: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalLatencyStatsDataset", name, input_dataset, tag,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_latency_stats_dataset_eager_fallback(
          input_dataset, tag, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_latency_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_latency_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalLatencyStatsDataset", input_dataset=input_dataset,
                                           tag=tag, output_types=output_types,
                                           output_shapes=output_shapes,
                                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalLatencyStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalLatencyStatsDataset = tf_export("raw_ops.ExperimentalLatencyStatsDataset")(_ops.to_raw_op(experimental_latency_stats_dataset))


def experimental_latency_stats_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_latency_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_latency_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [input_dataset, tag]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalLatencyStatsDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalLatencyStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_map_and_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], f, output_types, output_shapes, preserve_cardinality:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that fuses mapping with batching.

  Creates a dataset that applies `f` to the outputs of `input_dataset` and then
  batches `batch_size` of them.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `batch_size * num_parallel_batches` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when building a closure
      for `f`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch. It determines the number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    num_parallel_calls: A `Tensor` of type `int64`.
      A scalar representing the maximum number of parallel invocations of the `map_fn`
      function. Applying the `map_fn` on consecutive input elements in parallel has
      the potential to improve input pipeline throughput.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    f: A function decorated with @Defun.
      A function to apply to the outputs of `input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalMapAndBatchDataset", name, input_dataset,
        other_arguments, batch_size, num_parallel_calls, drop_remainder, "f",
        f, "output_types", output_types, "output_shapes", output_shapes,
        "preserve_cardinality", preserve_cardinality)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_map_and_batch_dataset_eager_fallback(
          input_dataset, other_arguments, batch_size, num_parallel_calls,
          drop_remainder, f=f, output_types=output_types,
          output_shapes=output_shapes,
          preserve_cardinality=preserve_cardinality, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_map_and_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_map_and_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalMapAndBatchDataset", input_dataset=input_dataset,
                                          other_arguments=other_arguments,
                                          batch_size=batch_size,
                                          num_parallel_calls=num_parallel_calls,
                                          drop_remainder=drop_remainder, f=f,
                                          output_types=output_types,
                                          output_shapes=output_shapes,
                                          preserve_cardinality=preserve_cardinality,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalMapAndBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalMapAndBatchDataset = tf_export("raw_ops.ExperimentalMapAndBatchDataset")(_ops.to_raw_op(experimental_map_and_batch_dataset))


def experimental_map_and_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], f, output_types, output_shapes, preserve_cardinality: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_map_and_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_map_and_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset] + list(other_arguments) + [batch_size, num_parallel_calls, drop_remainder]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "preserve_cardinality",
  preserve_cardinality)
  _result = _execute.execute(b"ExperimentalMapAndBatchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalMapAndBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_map_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, use_inter_op_parallelism:bool=True, preserve_cardinality:bool=False, force_synchronous:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    force_synchronous: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalMapDataset", name, input_dataset, other_arguments,
        "f", f, "output_types", output_types, "output_shapes", output_shapes,
        "use_inter_op_parallelism", use_inter_op_parallelism,
        "preserve_cardinality", preserve_cardinality, "force_synchronous",
        force_synchronous)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_map_dataset_eager_fallback(
          input_dataset, other_arguments, f=f, output_types=output_types,
          output_shapes=output_shapes,
          use_inter_op_parallelism=use_inter_op_parallelism,
          preserve_cardinality=preserve_cardinality,
          force_synchronous=force_synchronous, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if force_synchronous is None:
    force_synchronous = False
  force_synchronous = _execute.make_bool(force_synchronous, "force_synchronous")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalMapDataset", input_dataset=input_dataset,
                                  other_arguments=other_arguments, f=f,
                                  output_types=output_types,
                                  output_shapes=output_shapes,
                                  use_inter_op_parallelism=use_inter_op_parallelism,
                                  preserve_cardinality=preserve_cardinality,
                                  force_synchronous=force_synchronous,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_inter_op_parallelism",
              _op._get_attr_bool("use_inter_op_parallelism"),
              "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"), "force_synchronous",
              _op._get_attr_bool("force_synchronous"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalMapDataset = tf_export("raw_ops.ExperimentalMapDataset")(_ops.to_raw_op(experimental_map_dataset))


def experimental_map_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, use_inter_op_parallelism: bool, preserve_cardinality: bool, force_synchronous: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if force_synchronous is None:
    force_synchronous = False
  force_synchronous = _execute.make_bool(force_synchronous, "force_synchronous")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "use_inter_op_parallelism",
  use_inter_op_parallelism, "preserve_cardinality", preserve_cardinality,
  "force_synchronous", force_synchronous)
  _result = _execute.execute(b"ExperimentalMapDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_matching_files_dataset(patterns: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    patterns: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalMatchingFilesDataset", name, patterns)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_matching_files_dataset_eager_fallback(
          patterns, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalMatchingFilesDataset", patterns=patterns, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalMatchingFilesDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalMatchingFilesDataset = tf_export("raw_ops.ExperimentalMatchingFilesDataset")(_ops.to_raw_op(experimental_matching_files_dataset))


def experimental_matching_files_dataset_eager_fallback(patterns: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Variant]:
  patterns = _ops.convert_to_tensor(patterns, _dtypes.string)
  _inputs_flat = [patterns]
  _attrs = None
  _result = _execute.execute(b"ExperimentalMatchingFilesDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalMatchingFilesDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_max_intra_op_parallelism_dataset(input_dataset: Annotated[Any, _atypes.Variant], max_intra_op_parallelism: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that overrides the maximum intra-op parallelism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    max_intra_op_parallelism: A `Tensor` of type `int64`.
      Identifies the maximum intra-op parallelism to use.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalMaxIntraOpParallelismDataset", name, input_dataset,
        max_intra_op_parallelism, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_max_intra_op_parallelism_dataset_eager_fallback(
          input_dataset, max_intra_op_parallelism, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_max_intra_op_parallelism_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_max_intra_op_parallelism_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalMaxIntraOpParallelismDataset", input_dataset=input_dataset,
                                                    max_intra_op_parallelism=max_intra_op_parallelism,
                                                    output_types=output_types,
                                                    output_shapes=output_shapes,
                                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalMaxIntraOpParallelismDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalMaxIntraOpParallelismDataset = tf_export("raw_ops.ExperimentalMaxIntraOpParallelismDataset")(_ops.to_raw_op(experimental_max_intra_op_parallelism_dataset))


def experimental_max_intra_op_parallelism_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], max_intra_op_parallelism: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_max_intra_op_parallelism_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_max_intra_op_parallelism_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  max_intra_op_parallelism = _ops.convert_to_tensor(max_intra_op_parallelism, _dtypes.int64)
  _inputs_flat = [input_dataset, max_intra_op_parallelism]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalMaxIntraOpParallelismDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalMaxIntraOpParallelismDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_non_serializable_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalNonSerializableDataset", name, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_non_serializable_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_non_serializable_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_non_serializable_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalNonSerializableDataset", input_dataset=input_dataset,
                                              output_types=output_types,
                                              output_shapes=output_shapes,
                                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalNonSerializableDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalNonSerializableDataset = tf_export("raw_ops.ExperimentalNonSerializableDataset")(_ops.to_raw_op(experimental_non_serializable_dataset))


def experimental_non_serializable_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_non_serializable_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_non_serializable_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalNonSerializableDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalNonSerializableDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_parallel_interleave_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], sloppy: Annotated[Any, _atypes.Bool], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, with the exception
  that if retrieving the next value from a dataset would cause the requester to
  block, it will skip that input dataset. This dataset is especially useful
  when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
  allows the training step to proceed so long as some data is available.

  !! WARNING !! This dataset is not deterministic!

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    cycle_length: A `Tensor` of type `int64`.
    block_length: A `Tensor` of type `int64`.
    sloppy: A `Tensor` of type `bool`.
    buffer_output_elements: A `Tensor` of type `int64`.
    prefetch_input_elements: A `Tensor` of type `int64`.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalParallelInterleaveDataset", name, input_dataset,
        other_arguments, cycle_length, block_length, sloppy,
        buffer_output_elements, prefetch_input_elements, "f", f,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_parallel_interleave_dataset_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length, sloppy,
          buffer_output_elements, prefetch_input_elements, f=f,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_parallel_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_parallel_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalParallelInterleaveDataset", input_dataset=input_dataset,
                                                 other_arguments=other_arguments,
                                                 cycle_length=cycle_length,
                                                 block_length=block_length,
                                                 sloppy=sloppy,
                                                 buffer_output_elements=buffer_output_elements,
                                                 prefetch_input_elements=prefetch_input_elements,
                                                 f=f,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalParallelInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalParallelInterleaveDataset = tf_export("raw_ops.ExperimentalParallelInterleaveDataset")(_ops.to_raw_op(experimental_parallel_interleave_dataset))


def experimental_parallel_interleave_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], sloppy: Annotated[Any, _atypes.Bool], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_parallel_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_parallel_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  sloppy = _ops.convert_to_tensor(sloppy, _dtypes.bool)
  buffer_output_elements = _ops.convert_to_tensor(buffer_output_elements, _dtypes.int64)
  prefetch_input_elements = _ops.convert_to_tensor(prefetch_input_elements, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalParallelInterleaveDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalParallelInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_parse_example_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, sloppy:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_parallel_calls: A `Tensor` of type `int64`.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    sparse_keys: A list of `strings`.
      A list of string keys in the examples features.
      The results for these keys will be returned as `SparseTensor` objects.
    dense_keys: A list of `strings`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples features associated with dense values.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    sloppy: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalParseExampleDataset", name, input_dataset,
        num_parallel_calls, dense_defaults, "sparse_keys", sparse_keys,
        "dense_keys", dense_keys, "sparse_types", sparse_types,
        "dense_shapes", dense_shapes, "output_types", output_types,
        "output_shapes", output_shapes, "sloppy", sloppy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_parse_example_dataset_eager_fallback(
          input_dataset, num_parallel_calls, dense_defaults,
          sparse_keys=sparse_keys, dense_keys=dense_keys,
          sparse_types=sparse_types, dense_shapes=dense_shapes,
          output_types=output_types, output_shapes=output_shapes,
          sloppy=sloppy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalParseExampleDataset", input_dataset=input_dataset,
                                           num_parallel_calls=num_parallel_calls,
                                           dense_defaults=dense_defaults,
                                           sparse_keys=sparse_keys,
                                           dense_keys=dense_keys,
                                           sparse_types=sparse_types,
                                           dense_shapes=dense_shapes,
                                           output_types=output_types,
                                           output_shapes=output_shapes,
                                           sloppy=sloppy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sparse_keys", _op.get_attr("sparse_keys"), "dense_keys",
              _op.get_attr("dense_keys"), "sparse_types",
              _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
              "dense_shapes", _op.get_attr("dense_shapes"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "sloppy",
              _op._get_attr_bool("sloppy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalParseExampleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalParseExampleDataset = tf_export("raw_ops.ExperimentalParseExampleDataset")(_ops.to_raw_op(experimental_parse_example_dataset))


def experimental_parse_example_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, sloppy: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_parse_example_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset, num_parallel_calls] + list(dense_defaults)
  _attrs = ("sparse_keys", sparse_keys, "dense_keys", dense_keys,
  "sparse_types", sparse_types, "Tdense", _attr_Tdense, "dense_shapes",
  dense_shapes, "output_types", output_types, "output_shapes", output_shapes,
  "sloppy", sloppy)
  _result = _execute.execute(b"ExperimentalParseExampleDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalParseExampleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_private_thread_pool_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_threads: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_threads: A `Tensor` of type `int64`.
      Identifies the number of threads to use for the private threadpool.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalPrivateThreadPoolDataset", name, input_dataset,
        num_threads, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_private_thread_pool_dataset_eager_fallback(
          input_dataset, num_threads, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_private_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_private_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalPrivateThreadPoolDataset", input_dataset=input_dataset,
                                                num_threads=num_threads,
                                                output_types=output_types,
                                                output_shapes=output_shapes,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalPrivateThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalPrivateThreadPoolDataset = tf_export("raw_ops.ExperimentalPrivateThreadPoolDataset")(_ops.to_raw_op(experimental_private_thread_pool_dataset))


def experimental_private_thread_pool_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_threads: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_private_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_private_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_threads = _ops.convert_to_tensor(num_threads, _dtypes.int64)
  _inputs_flat = [input_dataset, num_threads]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalPrivateThreadPoolDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalPrivateThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_random_dataset(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a Dataset that returns pseudorandom numbers.

  Args:
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either seed or
      seed2 is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalRandomDataset", name, seed, seed2, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_random_dataset_eager_fallback(
          seed, seed2, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_random_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_random_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalRandomDataset", seed=seed, seed2=seed2,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalRandomDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalRandomDataset = tf_export("raw_ops.ExperimentalRandomDataset")(_ops.to_raw_op(experimental_random_dataset))


def experimental_random_dataset_eager_fallback(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_random_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_random_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  _inputs_flat = [seed, seed2]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalRandomDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalRandomDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_rebatch_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_replicas: Annotated[Any, _atypes.Int64], output_types, output_shapes, use_fallback:bool=True, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that changes the batch size.

  Creates a dataset that changes the batch size of the dataset to current batch
  size // num_replicas.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    num_replicas: A `Tensor` of type `int64`.
      A scalar representing the number of replicas to distribute this batch across. As
      a result of this transformation the current batch size would end up being
      divided  by this parameter.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_fallback: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalRebatchDataset", name, input_dataset, num_replicas,
        "output_types", output_types, "output_shapes", output_shapes,
        "use_fallback", use_fallback)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_rebatch_dataset_eager_fallback(
          input_dataset, num_replicas, output_types=output_types,
          output_shapes=output_shapes, use_fallback=use_fallback, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_rebatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_rebatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_fallback is None:
    use_fallback = True
  use_fallback = _execute.make_bool(use_fallback, "use_fallback")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalRebatchDataset", input_dataset=input_dataset,
                                      num_replicas=num_replicas,
                                      output_types=output_types,
                                      output_shapes=output_shapes,
                                      use_fallback=use_fallback, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_fallback",
              _op._get_attr_bool("use_fallback"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalRebatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalRebatchDataset = tf_export("raw_ops.ExperimentalRebatchDataset")(_ops.to_raw_op(experimental_rebatch_dataset))


def experimental_rebatch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_replicas: Annotated[Any, _atypes.Int64], output_types, output_shapes, use_fallback: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_rebatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_rebatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_fallback is None:
    use_fallback = True
  use_fallback = _execute.make_bool(use_fallback, "use_fallback")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_replicas = _ops.convert_to_tensor(num_replicas, _dtypes.int64)
  _inputs_flat = [input_dataset, num_replicas]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "use_fallback", use_fallback)
  _result = _execute.execute(b"ExperimentalRebatchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalRebatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_scan_dataset(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, preserve_cardinality:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset successively reduces `f` over the elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    initial_state: A list of `Tensor` objects.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalScanDataset", name, input_dataset, initial_state,
        other_arguments, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "preserve_cardinality",
        preserve_cardinality)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_scan_dataset_eager_fallback(
          input_dataset, initial_state, other_arguments, f=f,
          output_types=output_types, output_shapes=output_shapes,
          preserve_cardinality=preserve_cardinality, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_scan_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_scan_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalScanDataset", input_dataset=input_dataset,
                                   initial_state=initial_state,
                                   other_arguments=other_arguments, f=f,
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   preserve_cardinality=preserve_cardinality,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Tstate", _op.get_attr("Tstate"),
              "Targuments", _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalScanDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalScanDataset = tf_export("raw_ops.ExperimentalScanDataset")(_ops.to_raw_op(experimental_scan_dataset))


def experimental_scan_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, preserve_cardinality: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_scan_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_scan_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  _attr_Tstate, initial_state = _execute.convert_to_mixed_eager_tensors(initial_state, ctx)
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(initial_state) + list(other_arguments)
  _attrs = ("f", f, "Tstate", _attr_Tstate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes,
  "preserve_cardinality", preserve_cardinality)
  _result = _execute.execute(b"ExperimentalScanDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalScanDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_set_stats_aggregator_dataset(input_dataset: Annotated[Any, _atypes.Variant], stats_aggregator: Annotated[Any, _atypes.Resource], tag: Annotated[Any, _atypes.String], counter_prefix: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    stats_aggregator: A `Tensor` of type `resource`.
    tag: A `Tensor` of type `string`.
    counter_prefix: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalSetStatsAggregatorDataset", name, input_dataset,
        stats_aggregator, tag, counter_prefix, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_set_stats_aggregator_dataset_eager_fallback(
          input_dataset, stats_aggregator, tag, counter_prefix,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_set_stats_aggregator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_set_stats_aggregator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalSetStatsAggregatorDataset", input_dataset=input_dataset,
                                                 stats_aggregator=stats_aggregator,
                                                 tag=tag,
                                                 counter_prefix=counter_prefix,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalSetStatsAggregatorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalSetStatsAggregatorDataset = tf_export("raw_ops.ExperimentalSetStatsAggregatorDataset")(_ops.to_raw_op(experimental_set_stats_aggregator_dataset))


def experimental_set_stats_aggregator_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], stats_aggregator: Annotated[Any, _atypes.Resource], tag: Annotated[Any, _atypes.String], counter_prefix: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_set_stats_aggregator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_set_stats_aggregator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  stats_aggregator = _ops.convert_to_tensor(stats_aggregator, _dtypes.resource)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  counter_prefix = _ops.convert_to_tensor(counter_prefix, _dtypes.string)
  _inputs_flat = [input_dataset, stats_aggregator, tag, counter_prefix]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalSetStatsAggregatorDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalSetStatsAggregatorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_sleep_dataset(input_dataset: Annotated[Any, _atypes.Variant], sleep_microseconds: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    sleep_microseconds: A `Tensor` of type `int64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalSleepDataset", name, input_dataset,
        sleep_microseconds, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_sleep_dataset_eager_fallback(
          input_dataset, sleep_microseconds, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sleep_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sleep_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalSleepDataset", input_dataset=input_dataset,
                                    sleep_microseconds=sleep_microseconds,
                                    output_types=output_types,
                                    output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalSleepDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalSleepDataset = tf_export("raw_ops.ExperimentalSleepDataset")(_ops.to_raw_op(experimental_sleep_dataset))


def experimental_sleep_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], sleep_microseconds: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sleep_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sleep_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  sleep_microseconds = _ops.convert_to_tensor(sleep_microseconds, _dtypes.int64)
  _inputs_flat = [input_dataset, sleep_microseconds]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalSleepDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalSleepDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_sliding_window_dataset(input_dataset: Annotated[Any, _atypes.Variant], window_size: Annotated[Any, _atypes.Int64], window_shift: Annotated[Any, _atypes.Int64], window_stride: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that passes a sliding window over `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    window_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements in the
      sliding window.
    window_shift: A `Tensor` of type `int64`.
      A scalar representing the steps moving the sliding window
      forward in one iteration. It must be positive.
    window_stride: A `Tensor` of type `int64`.
      A scalar representing the stride of the input elements of the sliding window.
      It must be positive.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalSlidingWindowDataset", name, input_dataset,
        window_size, window_shift, window_stride, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_sliding_window_dataset_eager_fallback(
          input_dataset, window_size, window_shift, window_stride,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sliding_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sliding_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalSlidingWindowDataset", input_dataset=input_dataset,
                                            window_size=window_size,
                                            window_shift=window_shift,
                                            window_stride=window_stride,
                                            output_types=output_types,
                                            output_shapes=output_shapes,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalSlidingWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalSlidingWindowDataset = tf_export("raw_ops.ExperimentalSlidingWindowDataset")(_ops.to_raw_op(experimental_sliding_window_dataset))


def experimental_sliding_window_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], window_size: Annotated[Any, _atypes.Int64], window_shift: Annotated[Any, _atypes.Int64], window_stride: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sliding_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sliding_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  window_size = _ops.convert_to_tensor(window_size, _dtypes.int64)
  window_shift = _ops.convert_to_tensor(window_shift, _dtypes.int64)
  window_stride = _ops.convert_to_tensor(window_stride, _dtypes.int64)
  _inputs_flat = [input_dataset, window_size, window_shift, window_stride]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalSlidingWindowDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalSlidingWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_sql_dataset(driver_name: Annotated[Any, _atypes.String], data_source_name: Annotated[Any, _atypes.String], query: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that executes a SQL query and emits rows of the result set.

  Args:
    driver_name: A `Tensor` of type `string`.
      The database type. Currently, the only supported type is 'sqlite'.
    data_source_name: A `Tensor` of type `string`.
      A connection string to connect to the database.
    query: A `Tensor` of type `string`. A SQL query to execute.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalSqlDataset", name, driver_name, data_source_name,
        query, "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_sql_dataset_eager_fallback(
          driver_name, data_source_name, query, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sql_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sql_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalSqlDataset", driver_name=driver_name,
                                  data_source_name=data_source_name,
                                  query=query, output_types=output_types,
                                  output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalSqlDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalSqlDataset = tf_export("raw_ops.ExperimentalSqlDataset")(_ops.to_raw_op(experimental_sql_dataset))


def experimental_sql_dataset_eager_fallback(driver_name: Annotated[Any, _atypes.String], data_source_name: Annotated[Any, _atypes.String], query: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_sql_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_sql_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  driver_name = _ops.convert_to_tensor(driver_name, _dtypes.string)
  data_source_name = _ops.convert_to_tensor(data_source_name, _dtypes.string)
  query = _ops.convert_to_tensor(query, _dtypes.string)
  _inputs_flat = [driver_name, data_source_name, query]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalSqlDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalSqlDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_stats_aggregator_handle(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a statistics manager resource.

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalStatsAggregatorHandle", name, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_stats_aggregator_handle_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalStatsAggregatorHandle", container=container,
                                             shared_name=shared_name,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalStatsAggregatorHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalStatsAggregatorHandle = tf_export("raw_ops.ExperimentalStatsAggregatorHandle")(_ops.to_raw_op(experimental_stats_aggregator_handle))


def experimental_stats_aggregator_handle_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"ExperimentalStatsAggregatorHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalStatsAggregatorHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_stats_aggregator_summary(iterator: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Produces a summary of any statistics recorded by the given statistics manager.

  Args:
    iterator: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalStatsAggregatorSummary", name, iterator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_stats_aggregator_summary_eager_fallback(
          iterator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalStatsAggregatorSummary", iterator=iterator, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalStatsAggregatorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalStatsAggregatorSummary = tf_export("raw_ops.ExperimentalStatsAggregatorSummary")(_ops.to_raw_op(experimental_stats_aggregator_summary))


def experimental_stats_aggregator_summary_eager_fallback(iterator: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = None
  _result = _execute.execute(b"ExperimentalStatsAggregatorSummary", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalStatsAggregatorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_take_while_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that stops iteration when predicate` is false.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalTakeWhileDataset", name, input_dataset,
        other_arguments, "predicate", predicate, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_take_while_dataset_eager_fallback(
          input_dataset, other_arguments, predicate=predicate,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_take_while_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_take_while_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalTakeWhileDataset", input_dataset=input_dataset,
                                        other_arguments=other_arguments,
                                        predicate=predicate,
                                        output_types=output_types,
                                        output_shapes=output_shapes,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("predicate", _op.get_attr("predicate"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalTakeWhileDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalTakeWhileDataset = tf_export("raw_ops.ExperimentalTakeWhileDataset")(_ops.to_raw_op(experimental_take_while_dataset))


def experimental_take_while_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_take_while_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_take_while_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("predicate", predicate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalTakeWhileDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalTakeWhileDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_thread_pool_dataset(input_dataset: Annotated[Any, _atypes.Variant], thread_pool: Annotated[Any, _atypes.Resource], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    thread_pool: A `Tensor` of type `resource`.
      A resource produced by the ThreadPoolHandle op.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalThreadPoolDataset", name, input_dataset,
        thread_pool, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_thread_pool_dataset_eager_fallback(
          input_dataset, thread_pool, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalThreadPoolDataset", input_dataset=input_dataset,
                                         thread_pool=thread_pool,
                                         output_types=output_types,
                                         output_shapes=output_shapes,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalThreadPoolDataset = tf_export("raw_ops.ExperimentalThreadPoolDataset")(_ops.to_raw_op(experimental_thread_pool_dataset))


def experimental_thread_pool_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], thread_pool: Annotated[Any, _atypes.Resource], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  thread_pool = _ops.convert_to_tensor(thread_pool, _dtypes.resource)
  _inputs_flat = [input_dataset, thread_pool]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalThreadPoolDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_thread_pool_handle(num_threads: int, display_name: str, max_intra_op_parallelism:int=1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    num_threads: An `int`. The number of threads in the thread pool.
    display_name: A `string`.
      A human-readable name for the threads that may be visible in some
      visualizations.
      threadpool.
    max_intra_op_parallelism: An optional `int`. Defaults to `1`.
      The maximum degree of parallelism to use within operations that execute on this
      threadpool.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalThreadPoolHandle", name, "num_threads",
        num_threads, "max_intra_op_parallelism", max_intra_op_parallelism,
        "display_name", display_name, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_thread_pool_handle_eager_fallback(
          num_threads=num_threads,
          max_intra_op_parallelism=max_intra_op_parallelism,
          display_name=display_name, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_threads = _execute.make_int(num_threads, "num_threads")
  display_name = _execute.make_str(display_name, "display_name")
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalThreadPoolHandle", num_threads=num_threads,
                                        display_name=display_name,
                                        max_intra_op_parallelism=max_intra_op_parallelism,
                                        container=container,
                                        shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_threads", _op._get_attr_int("num_threads"),
              "max_intra_op_parallelism",
              _op._get_attr_int("max_intra_op_parallelism"), "display_name",
              _op.get_attr("display_name"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalThreadPoolHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalThreadPoolHandle = tf_export("raw_ops.ExperimentalThreadPoolHandle")(_ops.to_raw_op(experimental_thread_pool_handle))


def experimental_thread_pool_handle_eager_fallback(num_threads: int, display_name: str, max_intra_op_parallelism: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  num_threads = _execute.make_int(num_threads, "num_threads")
  display_name = _execute.make_str(display_name, "display_name")
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("num_threads", num_threads, "max_intra_op_parallelism",
  max_intra_op_parallelism, "display_name", display_name, "container",
  container, "shared_name", shared_name)
  _result = _execute.execute(b"ExperimentalThreadPoolHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalThreadPoolHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_unbatch_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A dataset that splits the elements of its input into multiple elements.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalUnbatchDataset", name, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_unbatch_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_unbatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_unbatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalUnbatchDataset", input_dataset=input_dataset,
                                      output_types=output_types,
                                      output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalUnbatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalUnbatchDataset = tf_export("raw_ops.ExperimentalUnbatchDataset")(_ops.to_raw_op(experimental_unbatch_dataset))


def experimental_unbatch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_unbatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_unbatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalUnbatchDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalUnbatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def experimental_unique_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that contains the unique elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExperimentalUniqueDataset", name, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return experimental_unique_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_unique_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_unique_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExperimentalUniqueDataset", input_dataset=input_dataset,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExperimentalUniqueDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExperimentalUniqueDataset = tf_export("raw_ops.ExperimentalUniqueDataset")(_ops.to_raw_op(experimental_unique_dataset))


def experimental_unique_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_unique_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_unique_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ExperimentalUniqueDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExperimentalUniqueDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def get_element_at_index(dataset: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None):
  r"""Gets the element at the specified index in a dataset.

  Args:
    dataset: A `Tensor` of type `variant`.
    index: A `Tensor` of type `int64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetElementAtIndex", name, dataset, index, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_element_at_index_eager_fallback(
          dataset, index, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'get_element_at_index' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'get_element_at_index' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetElementAtIndex", dataset=dataset, index=index,
                             output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetElementAtIndex", _inputs_flat, _attrs, _result)
  return _result

GetElementAtIndex = tf_export("raw_ops.GetElementAtIndex")(_ops.to_raw_op(get_element_at_index))


def get_element_at_index_eager_fallback(dataset: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'get_element_at_index' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'get_element_at_index' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  index = _ops.convert_to_tensor(index, _dtypes.int64)
  _inputs_flat = [dataset, index]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"GetElementAtIndex", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetElementAtIndex", _inputs_flat, _attrs, _result)
  return _result


def global_shuffle_dataset(input_dataset: Annotated[Any, _atypes.Variant], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    seed_generator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GlobalShuffleDataset", name, input_dataset, seed, seed2,
        seed_generator, "reshuffle_each_iteration", reshuffle_each_iteration,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return global_shuffle_dataset_eager_fallback(
          input_dataset, seed, seed2, seed_generator,
          reshuffle_each_iteration=reshuffle_each_iteration,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'global_shuffle_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'global_shuffle_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GlobalShuffleDataset", input_dataset=input_dataset, seed=seed,
                                seed2=seed2, seed_generator=seed_generator,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                reshuffle_each_iteration=reshuffle_each_iteration,
                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reshuffle_each_iteration",
              _op._get_attr_bool("reshuffle_each_iteration"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GlobalShuffleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GlobalShuffleDataset = tf_export("raw_ops.GlobalShuffleDataset")(_ops.to_raw_op(global_shuffle_dataset))


def global_shuffle_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'global_shuffle_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'global_shuffle_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
  _inputs_flat = [input_dataset, seed, seed2, seed_generator]
  _attrs = ("reshuffle_each_iteration", reshuffle_each_iteration,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"GlobalShuffleDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GlobalShuffleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def group_by_reducer_dataset(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, init_func_other_arguments, reduce_func_other_arguments, finalize_func_other_arguments, key_func, init_func, reduce_func, finalize_func, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that computes a group-by on `input_dataset`.

  Creates a dataset that computes a group-by on `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    key_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `key_func`.
    init_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `init_func`.
    reduce_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `reduce_func`.
    finalize_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `finalize_func`.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    init_func: A function decorated with @Defun.
      A function mapping a key of type DT_INT64, concatenated with
      `init_func_other_arguments` to the initial reducer state.
    reduce_func: A function decorated with @Defun.
      A function mapping the current reducer state and an element of `input_dataset`,
      concatenated with `reduce_func_other_arguments` to a new reducer state.
    finalize_func: A function decorated with @Defun.
      A function mapping the final reducer state to an output element.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GroupByReducerDataset", name, input_dataset,
        key_func_other_arguments, init_func_other_arguments,
        reduce_func_other_arguments, finalize_func_other_arguments,
        "key_func", key_func, "init_func", init_func, "reduce_func",
        reduce_func, "finalize_func", finalize_func, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return group_by_reducer_dataset_eager_fallback(
          input_dataset, key_func_other_arguments, init_func_other_arguments,
          reduce_func_other_arguments, finalize_func_other_arguments,
          key_func=key_func, init_func=init_func, reduce_func=reduce_func,
          finalize_func=finalize_func, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'group_by_reducer_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'group_by_reducer_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GroupByReducerDataset", input_dataset=input_dataset,
                                 key_func_other_arguments=key_func_other_arguments,
                                 init_func_other_arguments=init_func_other_arguments,
                                 reduce_func_other_arguments=reduce_func_other_arguments,
                                 finalize_func_other_arguments=finalize_func_other_arguments,
                                 key_func=key_func, init_func=init_func,
                                 reduce_func=reduce_func,
                                 finalize_func=finalize_func,
                                 output_types=output_types,
                                 output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_func", _op.get_attr("key_func"), "init_func",
              _op.get_attr("init_func"), "reduce_func",
              _op.get_attr("reduce_func"), "finalize_func",
              _op.get_attr("finalize_func"), "Tkey_func_other_arguments",
              _op.get_attr("Tkey_func_other_arguments"),
              "Tinit_func_other_arguments",
              _op.get_attr("Tinit_func_other_arguments"),
              "Treduce_func_other_arguments",
              _op.get_attr("Treduce_func_other_arguments"),
              "Tfinalize_func_other_arguments",
              _op.get_attr("Tfinalize_func_other_arguments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GroupByReducerDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GroupByReducerDataset = tf_export("raw_ops.GroupByReducerDataset")(_ops.to_raw_op(group_by_reducer_dataset))


def group_by_reducer_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, init_func_other_arguments, reduce_func_other_arguments, finalize_func_other_arguments, key_func, init_func, reduce_func, finalize_func, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'group_by_reducer_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'group_by_reducer_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, ctx)
  _attr_Tinit_func_other_arguments, init_func_other_arguments = _execute.convert_to_mixed_eager_tensors(init_func_other_arguments, ctx)
  _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, ctx)
  _attr_Tfinalize_func_other_arguments, finalize_func_other_arguments = _execute.convert_to_mixed_eager_tensors(finalize_func_other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(init_func_other_arguments) + list(reduce_func_other_arguments) + list(finalize_func_other_arguments)
  _attrs = ("key_func", key_func, "init_func", init_func, "reduce_func",
  reduce_func, "finalize_func", finalize_func, "Tkey_func_other_arguments",
  _attr_Tkey_func_other_arguments, "Tinit_func_other_arguments",
  _attr_Tinit_func_other_arguments, "Treduce_func_other_arguments",
  _attr_Treduce_func_other_arguments, "Tfinalize_func_other_arguments",
  _attr_Tfinalize_func_other_arguments, "output_types", output_types,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"GroupByReducerDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GroupByReducerDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def group_by_window_dataset(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that computes a windowed group-by on `input_dataset`.

  // TODO(mrry): Support non-int64 keys.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    key_func_other_arguments: A list of `Tensor` objects.
    reduce_func_other_arguments: A list of `Tensor` objects.
    window_size_func_other_arguments: A list of `Tensor` objects.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    reduce_func: A function decorated with @Defun.
    window_size_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GroupByWindowDataset", name, input_dataset,
        key_func_other_arguments, reduce_func_other_arguments,
        window_size_func_other_arguments, "key_func", key_func, "reduce_func",
        reduce_func, "window_size_func", window_size_func, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return group_by_window_dataset_eager_fallback(
          input_dataset, key_func_other_arguments,
          reduce_func_other_arguments, window_size_func_other_arguments,
          key_func=key_func, reduce_func=reduce_func,
          window_size_func=window_size_func, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GroupByWindowDataset", input_dataset=input_dataset,
                                key_func_other_arguments=key_func_other_arguments,
                                reduce_func_other_arguments=reduce_func_other_arguments,
                                window_size_func_other_arguments=window_size_func_other_arguments,
                                key_func=key_func, reduce_func=reduce_func,
                                window_size_func=window_size_func,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_func", _op.get_attr("key_func"), "reduce_func",
              _op.get_attr("reduce_func"), "window_size_func",
              _op.get_attr("window_size_func"), "Tkey_func_other_arguments",
              _op.get_attr("Tkey_func_other_arguments"),
              "Treduce_func_other_arguments",
              _op.get_attr("Treduce_func_other_arguments"),
              "Twindow_size_func_other_arguments",
              _op.get_attr("Twindow_size_func_other_arguments"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GroupByWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GroupByWindowDataset = tf_export("raw_ops.GroupByWindowDataset")(_ops.to_raw_op(group_by_window_dataset))


def group_by_window_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, ctx)
  _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, ctx)
  _attr_Twindow_size_func_other_arguments, window_size_func_other_arguments = _execute.convert_to_mixed_eager_tensors(window_size_func_other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(reduce_func_other_arguments) + list(window_size_func_other_arguments)
  _attrs = ("key_func", key_func, "reduce_func", reduce_func,
  "window_size_func", window_size_func, "Tkey_func_other_arguments",
  _attr_Tkey_func_other_arguments, "Treduce_func_other_arguments",
  _attr_Treduce_func_other_arguments, "Twindow_size_func_other_arguments",
  _attr_Twindow_size_func_other_arguments, "output_types", output_types,
  "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"GroupByWindowDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GroupByWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def ignore_errors_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, log_warning:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that contains the elements of `input_dataset` ignoring errors.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    log_warning: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IgnoreErrorsDataset", name, input_dataset, "output_types",
        output_types, "output_shapes", output_shapes, "log_warning",
        log_warning)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ignore_errors_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, log_warning=log_warning, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if log_warning is None:
    log_warning = False
  log_warning = _execute.make_bool(log_warning, "log_warning")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IgnoreErrorsDataset", input_dataset=input_dataset,
                               output_types=output_types,
                               output_shapes=output_shapes,
                               log_warning=log_warning, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "log_warning",
              _op._get_attr_bool("log_warning"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IgnoreErrorsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IgnoreErrorsDataset = tf_export("raw_ops.IgnoreErrorsDataset")(_ops.to_raw_op(ignore_errors_dataset))


def ignore_errors_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, log_warning: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if log_warning is None:
    log_warning = False
  log_warning = _execute.make_bool(log_warning, "log_warning")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "log_warning", log_warning)
  _result = _execute.execute(b"IgnoreErrorsDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IgnoreErrorsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def index_flat_map_dataset(input_dataset: Annotated[Any, _atypes.Variant], map_func_other_args, index_map_func_other_args, output_cardinality: Annotated[Any, _atypes.Int64], map_func, index_map_func, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    map_func_other_args: A list of `Tensor` objects.
    index_map_func_other_args: A list of `Tensor` objects.
    output_cardinality: A `Tensor` of type `int64`.
    map_func: A function decorated with @Defun.
    index_map_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IndexFlatMapDataset", name, input_dataset, map_func_other_args,
        index_map_func_other_args, output_cardinality, "map_func", map_func,
        "index_map_func", index_map_func, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return index_flat_map_dataset_eager_fallback(
          input_dataset, map_func_other_args, index_map_func_other_args,
          output_cardinality, map_func=map_func,
          index_map_func=index_map_func, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'index_flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'index_flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IndexFlatMapDataset", input_dataset=input_dataset,
                               map_func_other_args=map_func_other_args,
                               index_map_func_other_args=index_map_func_other_args,
                               output_cardinality=output_cardinality,
                               map_func=map_func,
                               index_map_func=index_map_func,
                               output_types=output_types,
                               output_shapes=output_shapes, metadata=metadata,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("map_func", _op.get_attr("map_func"), "index_map_func",
              _op.get_attr("index_map_func"), "Tmap_func_args",
              _op.get_attr("Tmap_func_args"), "Tindex_map_func_args",
              _op.get_attr("Tindex_map_func_args"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IndexFlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IndexFlatMapDataset = tf_export("raw_ops.IndexFlatMapDataset")(_ops.to_raw_op(index_flat_map_dataset))


def index_flat_map_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], map_func_other_args, index_map_func_other_args, output_cardinality: Annotated[Any, _atypes.Int64], map_func, index_map_func, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'index_flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'index_flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tmap_func_args, map_func_other_args = _execute.convert_to_mixed_eager_tensors(map_func_other_args, ctx)
  _attr_Tindex_map_func_args, index_map_func_other_args = _execute.convert_to_mixed_eager_tensors(index_map_func_other_args, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  output_cardinality = _ops.convert_to_tensor(output_cardinality, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(map_func_other_args) + list(index_map_func_other_args) + [output_cardinality]
  _attrs = ("map_func", map_func, "index_map_func", index_map_func,
  "Tmap_func_args", _attr_Tmap_func_args, "Tindex_map_func_args",
  _attr_Tindex_map_func_args, "output_types", output_types, "output_shapes",
  output_shapes, "metadata", metadata)
  _result = _execute.execute(b"IndexFlatMapDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IndexFlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def initialize_table_from_dataset(table_handle: Annotated[Any, _atypes.Resource], dataset: Annotated[Any, _atypes.Variant], name=None):
  r"""TODO: add doc.

  Args:
    table_handle: A `Tensor` of type `resource`.
    dataset: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InitializeTableFromDataset", name, table_handle, dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return initialize_table_from_dataset_eager_fallback(
          table_handle, dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTableFromDataset", table_handle=table_handle,
                                      dataset=dataset, name=name)
  return _op
InitializeTableFromDataset = tf_export("raw_ops.InitializeTableFromDataset")(_ops.to_raw_op(initialize_table_from_dataset))


def initialize_table_from_dataset_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], dataset: Annotated[Any, _atypes.Variant], name, ctx):
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  _inputs_flat = [table_handle, dataset]
  _attrs = None
  _result = _execute.execute(b"InitializeTableFromDataset", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def iterator_get_device(resource: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the name of the device on which `resource` has been placed.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorGetDevice", name, resource)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_get_device_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorGetDevice", resource=resource, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorGetDevice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorGetDevice = tf_export("raw_ops.IteratorGetDevice")(_ops.to_raw_op(iterator_get_device))


def iterator_get_device_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"IteratorGetDevice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorGetDevice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_get_model_proto(iterator: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the serialized model proto of an iterator resource.

  Returns the serialized model proto of an iterator resource.

  Args:
    iterator: A `Tensor` of type `resource`.
      An resource from an dataset iterator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorGetModelProto", name, iterator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_get_model_proto_eager_fallback(
          iterator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorGetModelProto", iterator=iterator, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorGetModelProto", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorGetModelProto = tf_export("raw_ops.IteratorGetModelProto")(_ops.to_raw_op(iterator_get_model_proto))


def iterator_get_model_proto_eager_fallback(iterator: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = None
  _result = _execute.execute(b"IteratorGetModelProto", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorGetModelProto", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def lmdb_dataset(filenames: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the key-value pairs in one or more LMDB files.

  The Lightning Memory-Mapped Database Manager, or LMDB, is an embedded binary
  key-value database. This dataset can read the contents of LMDB database files,
  the names of which generally have the `.mdb` suffix.

  Each output element consists of a key-value pair represented as a pair of
  scalar string `Tensor`s, where the first `Tensor` contains the key and the
  second `Tensor` contains the value.

  LMDB uses different file formats on big- and little-endian machines.
  `LMDBDataset` can only read files in the format of the host machine.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the binary file(s) to be
      read.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LMDBDataset", name, filenames, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lmdb_dataset_eager_fallback(
          filenames, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'lmdb_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'lmdb_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LMDBDataset", filenames=filenames, output_types=output_types,
                       output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LMDBDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LMDBDataset = tf_export("raw_ops.LMDBDataset")(_ops.to_raw_op(lmdb_dataset))


def lmdb_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'lmdb_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'lmdb_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  _inputs_flat = [filenames]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"LMDBDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LMDBDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def latency_stats_dataset(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Records the latency of producing `input_dataset` elements in a StatsAggregator.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    tag: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LatencyStatsDataset", name, input_dataset, tag, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return latency_stats_dataset_eager_fallback(
          input_dataset, tag, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'latency_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'latency_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LatencyStatsDataset", input_dataset=input_dataset, tag=tag,
                               output_types=output_types,
                               output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LatencyStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LatencyStatsDataset = tf_export("raw_ops.LatencyStatsDataset")(_ops.to_raw_op(latency_stats_dataset))


def latency_stats_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], tag: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'latency_stats_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'latency_stats_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [input_dataset, tag]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"LatencyStatsDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LatencyStatsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def legacy_parallel_interleave_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic:str="default", metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, with the exception
  that if retrieving the next value from a dataset would cause the requester to
  block, it will skip that input dataset. This dataset is especially useful
  when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
  allows the training step to proceed so long as some data is available.

  !! WARNING !! This dataset is not deterministic!

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    cycle_length: A `Tensor` of type `int64`.
    block_length: A `Tensor` of type `int64`.
    buffer_output_elements: A `Tensor` of type `int64`.
    prefetch_input_elements: A `Tensor` of type `int64`.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LegacyParallelInterleaveDatasetV2", name, input_dataset,
        other_arguments, cycle_length, block_length, buffer_output_elements,
        prefetch_input_elements, "f", f, "deterministic", deterministic,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return legacy_parallel_interleave_dataset_v2_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length,
          buffer_output_elements, prefetch_input_elements, f=f,
          deterministic=deterministic, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'legacy_parallel_interleave_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'legacy_parallel_interleave_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LegacyParallelInterleaveDatasetV2", input_dataset=input_dataset,
                                             other_arguments=other_arguments,
                                             cycle_length=cycle_length,
                                             block_length=block_length,
                                             buffer_output_elements=buffer_output_elements,
                                             prefetch_input_elements=prefetch_input_elements,
                                             f=f, output_types=output_types,
                                             output_shapes=output_shapes,
                                             deterministic=deterministic,
                                             metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "deterministic",
              _op.get_attr("deterministic"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LegacyParallelInterleaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LegacyParallelInterleaveDatasetV2 = tf_export("raw_ops.LegacyParallelInterleaveDatasetV2")(_ops.to_raw_op(legacy_parallel_interleave_dataset_v2))


def legacy_parallel_interleave_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'legacy_parallel_interleave_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'legacy_parallel_interleave_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  buffer_output_elements = _ops.convert_to_tensor(buffer_output_elements, _dtypes.int64)
  prefetch_input_elements = _ops.convert_to_tensor(prefetch_input_elements, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, buffer_output_elements, prefetch_input_elements]
  _attrs = ("f", f, "deterministic", deterministic, "Targuments",
  _attr_Targuments, "output_types", output_types, "output_shapes",
  output_shapes, "metadata", metadata)
  _result = _execute.execute(b"LegacyParallelInterleaveDatasetV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LegacyParallelInterleaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def list_dataset(tensors, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits each of `tensors` once.

  Args:
    tensors: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ListDataset", name, tensors, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return list_dataset_eager_fallback(
          tensors, output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'list_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'list_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ListDataset", tensors=tensors, output_types=output_types,
                       output_shapes=output_shapes, metadata=metadata,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput_types", _op.get_attr("Tinput_types"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ListDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ListDataset = tf_export("raw_ops.ListDataset")(_ops.to_raw_op(list_dataset))


def list_dataset_eager_fallback(tensors, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'list_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'list_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tinput_types, tensors = _execute.convert_to_mixed_eager_tensors(tensors, ctx)
  _inputs_flat = list(tensors)
  _attrs = ("Tinput_types", _attr_Tinput_types, "output_types", output_types,
  "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ListDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ListDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def list_snapshot_chunks_dataset(snapshot_path: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    snapshot_path: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ListSnapshotChunksDataset", name, snapshot_path,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return list_snapshot_chunks_dataset_eager_fallback(
          snapshot_path, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'list_snapshot_chunks_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'list_snapshot_chunks_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ListSnapshotChunksDataset", snapshot_path=snapshot_path,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ListSnapshotChunksDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ListSnapshotChunksDataset = tf_export("raw_ops.ListSnapshotChunksDataset")(_ops.to_raw_op(list_snapshot_chunks_dataset))


def list_snapshot_chunks_dataset_eager_fallback(snapshot_path: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'list_snapshot_chunks_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'list_snapshot_chunks_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  snapshot_path = _ops.convert_to_tensor(snapshot_path, _dtypes.string)
  _inputs_flat = [snapshot_path]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ListSnapshotChunksDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ListSnapshotChunksDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def load_dataset(path: Annotated[Any, _atypes.String], reader_func_other_args, output_types, output_shapes, reader_func, compression:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    path: A `Tensor` of type `string`.
    reader_func_other_args: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reader_func: A function decorated with @Defun.
    compression: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadDataset", name, path, reader_func_other_args,
        "output_types", output_types, "output_shapes", output_shapes,
        "compression", compression, "reader_func", reader_func)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return load_dataset_eager_fallback(
          path, reader_func_other_args, output_types=output_types,
          output_shapes=output_shapes, compression=compression,
          reader_func=reader_func, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'load_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'load_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadDataset", path=path,
                       reader_func_other_args=reader_func_other_args,
                       output_types=output_types, output_shapes=output_shapes,
                       reader_func=reader_func, compression=compression,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "compression",
              _op.get_attr("compression"), "reader_func",
              _op.get_attr("reader_func"), "Treader_func_args",
              _op.get_attr("Treader_func_args"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LoadDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LoadDataset = tf_export("raw_ops.LoadDataset")(_ops.to_raw_op(load_dataset))


def load_dataset_eager_fallback(path: Annotated[Any, _atypes.String], reader_func_other_args, output_types, output_shapes, reader_func, compression: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'load_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'load_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  _attr_Treader_func_args, reader_func_other_args = _execute.convert_to_mixed_eager_tensors(reader_func_other_args, ctx)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [path] + list(reader_func_other_args)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "compression", compression, "reader_func", reader_func, "Treader_func_args",
  _attr_Treader_func_args)
  _result = _execute.execute(b"LoadDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LoadDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def map_and_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], f, output_types, output_shapes, preserve_cardinality:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that fuses mapping with batching.

  Creates a dataset that applies `f` to the outputs of `input_dataset` and then
  batches `batch_size` of them.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `batch_size * num_parallel_batches` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when building a closure
      for `f`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch. It determines the number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    num_parallel_calls: A `Tensor` of type `int64`.
      A scalar representing the maximum number of parallel invocations of the `map_fn`
      function. Applying the `map_fn` on consecutive input elements in parallel has
      the potential to improve input pipeline throughput.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    f: A function decorated with @Defun.
      A function to apply to the outputs of `input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapAndBatchDataset", name, input_dataset, other_arguments,
        batch_size, num_parallel_calls, drop_remainder, "f", f,
        "output_types", output_types, "output_shapes", output_shapes,
        "preserve_cardinality", preserve_cardinality, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_and_batch_dataset_eager_fallback(
          input_dataset, other_arguments, batch_size, num_parallel_calls,
          drop_remainder, f=f, output_types=output_types,
          output_shapes=output_shapes,
          preserve_cardinality=preserve_cardinality, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_and_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_and_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapAndBatchDataset", input_dataset=input_dataset,
                              other_arguments=other_arguments,
                              batch_size=batch_size,
                              num_parallel_calls=num_parallel_calls,
                              drop_remainder=drop_remainder, f=f,
                              output_types=output_types,
                              output_shapes=output_shapes,
                              preserve_cardinality=preserve_cardinality,
                              metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapAndBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MapAndBatchDataset = tf_export("raw_ops.MapAndBatchDataset")(_ops.to_raw_op(map_and_batch_dataset))


def map_and_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], f, output_types, output_shapes, preserve_cardinality: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_and_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_and_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset] + list(other_arguments) + [batch_size, num_parallel_calls, drop_remainder]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "preserve_cardinality",
  preserve_cardinality, "metadata", metadata)
  _result = _execute.execute(b"MapAndBatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapAndBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def matching_files_dataset(patterns: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    patterns: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatchingFilesDataset", name, patterns)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return matching_files_dataset_eager_fallback(
          patterns, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatchingFilesDataset", patterns=patterns, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatchingFilesDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatchingFilesDataset = tf_export("raw_ops.MatchingFilesDataset")(_ops.to_raw_op(matching_files_dataset))


def matching_files_dataset_eager_fallback(patterns: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Variant]:
  patterns = _ops.convert_to_tensor(patterns, _dtypes.string)
  _inputs_flat = [patterns]
  _attrs = None
  _result = _execute.execute(b"MatchingFilesDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatchingFilesDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def max_intra_op_parallelism_dataset(input_dataset: Annotated[Any, _atypes.Variant], max_intra_op_parallelism: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that overrides the maximum intra-op parallelism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    max_intra_op_parallelism: A `Tensor` of type `int64`.
      Identifies the maximum intra-op parallelism to use.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxIntraOpParallelismDataset", name, input_dataset,
        max_intra_op_parallelism, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_intra_op_parallelism_dataset_eager_fallback(
          input_dataset, max_intra_op_parallelism, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'max_intra_op_parallelism_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'max_intra_op_parallelism_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxIntraOpParallelismDataset", input_dataset=input_dataset,
                                        max_intra_op_parallelism=max_intra_op_parallelism,
                                        output_types=output_types,
                                        output_shapes=output_shapes,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxIntraOpParallelismDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxIntraOpParallelismDataset = tf_export("raw_ops.MaxIntraOpParallelismDataset")(_ops.to_raw_op(max_intra_op_parallelism_dataset))


def max_intra_op_parallelism_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], max_intra_op_parallelism: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'max_intra_op_parallelism_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'max_intra_op_parallelism_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  max_intra_op_parallelism = _ops.convert_to_tensor(max_intra_op_parallelism, _dtypes.int64)
  _inputs_flat = [input_dataset, max_intra_op_parallelism]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"MaxIntraOpParallelismDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxIntraOpParallelismDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def non_serializable_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonSerializableDataset", name, input_dataset, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_serializable_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'non_serializable_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'non_serializable_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonSerializableDataset", input_dataset=input_dataset,
                                  output_types=output_types,
                                  output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonSerializableDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonSerializableDataset = tf_export("raw_ops.NonSerializableDataset")(_ops.to_raw_op(non_serializable_dataset))


def non_serializable_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'non_serializable_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'non_serializable_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"NonSerializableDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonSerializableDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_interleave_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], sloppy: Annotated[Any, _atypes.Bool], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, with the exception
  that if retrieving the next value from a dataset would cause the requester to
  block, it will skip that input dataset. This dataset is especially useful
  when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
  allows the training step to proceed so long as some data is available.

  !! WARNING !! If the `sloppy` parameter is set to `True`, the operation of this
  dataset will not be deterministic!

  This dataset has been superseded by `ParallelInterleaveDatasetV2`.  New code
  should use `ParallelInterleaveDatasetV2`.

  The Python API `tf.data.experimental.parallel_interleave` creates instances of
  this op. `tf.data.experimental.parallel_interleave` is a deprecated API.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDataset` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    sloppy: A `Tensor` of type `bool`.
      If `True`, return elements as they become available, even if that means returning
      these elements in a non-deterministic order. Sloppy operation may result in better
      performance in the presence of stragglers, but the dataset will still block if
      all of its open streams are blocked.
      If `False`, always return elements in a deterministic order.
    buffer_output_elements: A `Tensor` of type `int64`.
      The number of elements each iterator being interleaved should buffer (similar
      to the `.prefetch()` transformation for each interleaved iterator).
    prefetch_input_elements: A `Tensor` of type `int64`.
      Determines the number of iterators to prefetch, allowing buffers to warm up and
      data to be pre-fetched without blocking the main thread.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelInterleaveDataset", name, input_dataset,
        other_arguments, cycle_length, block_length, sloppy,
        buffer_output_elements, prefetch_input_elements, "f", f,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_interleave_dataset_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length, sloppy,
          buffer_output_elements, prefetch_input_elements, f=f,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelInterleaveDataset", input_dataset=input_dataset,
                                     other_arguments=other_arguments,
                                     cycle_length=cycle_length,
                                     block_length=block_length, sloppy=sloppy,
                                     buffer_output_elements=buffer_output_elements,
                                     prefetch_input_elements=prefetch_input_elements,
                                     f=f, output_types=output_types,
                                     output_shapes=output_shapes,
                                     metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelInterleaveDataset = tf_export("raw_ops.ParallelInterleaveDataset")(_ops.to_raw_op(parallel_interleave_dataset))


def parallel_interleave_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], sloppy: Annotated[Any, _atypes.Bool], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  sloppy = _ops.convert_to_tensor(sloppy, _dtypes.bool)
  buffer_output_elements = _ops.convert_to_tensor(buffer_output_elements, _dtypes.int64)
  prefetch_input_elements = _ops.convert_to_tensor(prefetch_input_elements, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ParallelInterleaveDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelInterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parse_example_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, sloppy:bool=False, ragged_keys=[], ragged_value_types=[], ragged_split_types=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_parallel_calls: A `Tensor` of type `int64`.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    sparse_keys: A list of `strings`.
      A list of string keys in the examples features.
      The results for these keys will be returned as `SparseTensor` objects.
    dense_keys: A list of `strings`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples features associated with dense values.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    sloppy: An optional `bool`. Defaults to `False`.
    ragged_keys: An optional list of `strings`. Defaults to `[]`.
    ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseExampleDataset", name, input_dataset, num_parallel_calls,
        dense_defaults, "sparse_keys", sparse_keys, "dense_keys", dense_keys,
        "sparse_types", sparse_types, "dense_shapes", dense_shapes,
        "output_types", output_types, "output_shapes", output_shapes,
        "sloppy", sloppy, "ragged_keys", ragged_keys, "ragged_value_types",
        ragged_value_types, "ragged_split_types", ragged_split_types)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_example_dataset_eager_fallback(
          input_dataset, num_parallel_calls, dense_defaults,
          sparse_keys=sparse_keys, dense_keys=dense_keys,
          sparse_types=sparse_types, dense_shapes=dense_shapes,
          output_types=output_types, output_shapes=output_shapes,
          sloppy=sloppy, ragged_keys=ragged_keys,
          ragged_value_types=ragged_value_types,
          ragged_split_types=ragged_split_types, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example_dataset' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example_dataset' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_dataset' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_dataset' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parse_example_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parse_example_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if ragged_keys is None:
    ragged_keys = []
  if not isinstance(ragged_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_keys' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_keys)
  ragged_keys = [_execute.make_str(_s, "ragged_keys") for _s in ragged_keys]
  if ragged_value_types is None:
    ragged_value_types = []
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if ragged_split_types is None:
    ragged_split_types = []
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseExampleDataset", input_dataset=input_dataset,
                               num_parallel_calls=num_parallel_calls,
                               dense_defaults=dense_defaults,
                               sparse_keys=sparse_keys, dense_keys=dense_keys,
                               sparse_types=sparse_types,
                               dense_shapes=dense_shapes,
                               output_types=output_types,
                               output_shapes=output_shapes, sloppy=sloppy,
                               ragged_keys=ragged_keys,
                               ragged_value_types=ragged_value_types,
                               ragged_split_types=ragged_split_types,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sparse_keys", _op.get_attr("sparse_keys"), "dense_keys",
              _op.get_attr("dense_keys"), "sparse_types",
              _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
              "dense_shapes", _op.get_attr("dense_shapes"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "sloppy",
              _op._get_attr_bool("sloppy"), "ragged_keys",
              _op.get_attr("ragged_keys"), "ragged_value_types",
              _op.get_attr("ragged_value_types"), "ragged_split_types",
              _op.get_attr("ragged_split_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseExampleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParseExampleDataset = tf_export("raw_ops.ParseExampleDataset")(_ops.to_raw_op(parse_example_dataset))


def parse_example_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, sloppy: bool, ragged_keys, ragged_value_types, ragged_split_types, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example_dataset' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example_dataset' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_dataset' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_dataset' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parse_example_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parse_example_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if ragged_keys is None:
    ragged_keys = []
  if not isinstance(ragged_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_keys' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_keys)
  ragged_keys = [_execute.make_str(_s, "ragged_keys") for _s in ragged_keys]
  if ragged_value_types is None:
    ragged_value_types = []
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if ragged_split_types is None:
    ragged_split_types = []
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_dataset' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset, num_parallel_calls] + list(dense_defaults)
  _attrs = ("sparse_keys", sparse_keys, "dense_keys", dense_keys,
  "sparse_types", sparse_types, "Tdense", _attr_Tdense, "dense_shapes",
  dense_shapes, "output_types", output_types, "output_shapes", output_shapes,
  "sloppy", sloppy, "ragged_keys", ragged_keys, "ragged_value_types",
  ragged_value_types, "ragged_split_types", ragged_split_types)
  _result = _execute.execute(b"ParseExampleDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseExampleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parse_example_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, deterministic:str="default", ragged_keys=[], ragged_value_types=[], ragged_split_types=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_parallel_calls: A `Tensor` of type `int64`.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    sparse_keys: A list of `strings`.
      A list of string keys in the examples features.
      The results for these keys will be returned as `SparseTensor` objects.
    dense_keys: A list of `strings`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples features associated with dense values.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the dataset is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    ragged_keys: An optional list of `strings`. Defaults to `[]`.
    ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseExampleDatasetV2", name, input_dataset,
        num_parallel_calls, dense_defaults, "sparse_keys", sparse_keys,
        "dense_keys", dense_keys, "sparse_types", sparse_types,
        "dense_shapes", dense_shapes, "output_types", output_types,
        "output_shapes", output_shapes, "deterministic", deterministic,
        "ragged_keys", ragged_keys, "ragged_value_types", ragged_value_types,
        "ragged_split_types", ragged_split_types)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_example_dataset_v2_eager_fallback(
          input_dataset, num_parallel_calls, dense_defaults,
          sparse_keys=sparse_keys, dense_keys=dense_keys,
          sparse_types=sparse_types, dense_shapes=dense_shapes,
          output_types=output_types, output_shapes=output_shapes,
          deterministic=deterministic, ragged_keys=ragged_keys,
          ragged_value_types=ragged_value_types,
          ragged_split_types=ragged_split_types, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_dataset_v2' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parse_example_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if ragged_keys is None:
    ragged_keys = []
  if not isinstance(ragged_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_keys)
  ragged_keys = [_execute.make_str(_s, "ragged_keys") for _s in ragged_keys]
  if ragged_value_types is None:
    ragged_value_types = []
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if ragged_split_types is None:
    ragged_split_types = []
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseExampleDatasetV2", input_dataset=input_dataset,
                                 num_parallel_calls=num_parallel_calls,
                                 dense_defaults=dense_defaults,
                                 sparse_keys=sparse_keys,
                                 dense_keys=dense_keys,
                                 sparse_types=sparse_types,
                                 dense_shapes=dense_shapes,
                                 output_types=output_types,
                                 output_shapes=output_shapes,
                                 deterministic=deterministic,
                                 ragged_keys=ragged_keys,
                                 ragged_value_types=ragged_value_types,
                                 ragged_split_types=ragged_split_types,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sparse_keys", _op.get_attr("sparse_keys"), "dense_keys",
              _op.get_attr("dense_keys"), "sparse_types",
              _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
              "dense_shapes", _op.get_attr("dense_shapes"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "deterministic",
              _op.get_attr("deterministic"), "ragged_keys",
              _op.get_attr("ragged_keys"), "ragged_value_types",
              _op.get_attr("ragged_value_types"), "ragged_split_types",
              _op.get_attr("ragged_split_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseExampleDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParseExampleDatasetV2 = tf_export("raw_ops.ParseExampleDatasetV2")(_ops.to_raw_op(parse_example_dataset_v2))


def parse_example_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_parallel_calls: Annotated[Any, _atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, deterministic: str, ragged_keys, ragged_value_types, ragged_split_types, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_dataset_v2' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parse_example_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if ragged_keys is None:
    ragged_keys = []
  if not isinstance(ragged_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_keys' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_keys)
  ragged_keys = [_execute.make_str(_s, "ragged_keys") for _s in ragged_keys]
  if ragged_value_types is None:
    ragged_value_types = []
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if ragged_split_types is None:
    ragged_split_types = []
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_dataset_v2' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset, num_parallel_calls] + list(dense_defaults)
  _attrs = ("sparse_keys", sparse_keys, "dense_keys", dense_keys,
  "sparse_types", sparse_types, "Tdense", _attr_Tdense, "dense_shapes",
  dense_shapes, "output_types", output_types, "output_shapes", output_shapes,
  "deterministic", deterministic, "ragged_keys", ragged_keys,
  "ragged_value_types", ragged_value_types, "ragged_split_types",
  ragged_split_types)
  _result = _execute.execute(b"ParseExampleDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseExampleDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def private_thread_pool_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_threads: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_threads: A `Tensor` of type `int64`.
      Identifies the number of threads to use for the private threadpool.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PrivateThreadPoolDataset", name, input_dataset, num_threads,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return private_thread_pool_dataset_eager_fallback(
          input_dataset, num_threads, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'private_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'private_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PrivateThreadPoolDataset", input_dataset=input_dataset,
                                    num_threads=num_threads,
                                    output_types=output_types,
                                    output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PrivateThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PrivateThreadPoolDataset = tf_export("raw_ops.PrivateThreadPoolDataset")(_ops.to_raw_op(private_thread_pool_dataset))


def private_thread_pool_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_threads: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'private_thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'private_thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_threads = _ops.convert_to_tensor(num_threads, _dtypes.int64)
  _inputs_flat = [input_dataset, num_threads]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"PrivateThreadPoolDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PrivateThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def random_dataset(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a Dataset that returns pseudorandom numbers.

  Creates a Dataset that returns a stream of uniformly distributed
  pseudorandom 64-bit signed integers.

  In the TensorFlow Python API, you can instantiate this dataset via the
  class `tf.data.experimental.RandomDataset`.

  Instances of this dataset are also created as a result of the
  `hoist_random_uniform` static optimization. Whether this optimization is
  performed is determined by the `experimental_optimization.hoist_random_uniform`
  option of `tf.data.Options`.

  Args:
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either seed or
      seed2 is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomDataset", name, seed, seed2, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_dataset_eager_fallback(
          seed, seed2, output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'random_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'random_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomDataset", seed=seed, seed2=seed2, output_types=output_types,
                         output_shapes=output_shapes, metadata=metadata,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomDataset = tf_export("raw_ops.RandomDataset")(_ops.to_raw_op(random_dataset))


def random_dataset_eager_fallback(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'random_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'random_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  _inputs_flat = [seed, seed2]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"RandomDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def random_dataset_v2(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, rerandomize_each_iteration:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a Dataset that returns pseudorandom numbers.

  Creates a Dataset that returns a stream of uniformly distributed
  pseudorandom 64-bit signed integers. It accepts a boolean attribute that
  determines if the random number generators are re-applied at each epoch. The
  default value is True which means that the seeds are applied and the same
  sequence of random numbers are generated at each epoch. If set to False, the
  seeds are not re-applied and a different sequence of random numbers are
  generated at each epoch.

  In the TensorFlow Python API, you can instantiate this dataset via the
  class `tf.data.experimental.RandomDatasetV2`.

  Args:
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either seed or
      seed2 is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    seed_generator: A `Tensor` of type `resource`.
      A resource for the random number seed generator.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    rerandomize_each_iteration: An optional `bool`. Defaults to `False`.
      A boolean attribute to rerandomize the sequence of random numbers generated
      at each epoch.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomDatasetV2", name, seed, seed2, seed_generator,
        "rerandomize_each_iteration", rerandomize_each_iteration,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_dataset_v2_eager_fallback(
          seed, seed2, seed_generator,
          rerandomize_each_iteration=rerandomize_each_iteration,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'random_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'random_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if rerandomize_each_iteration is None:
    rerandomize_each_iteration = False
  rerandomize_each_iteration = _execute.make_bool(rerandomize_each_iteration, "rerandomize_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomDatasetV2", seed=seed, seed2=seed2,
                           seed_generator=seed_generator,
                           output_types=output_types,
                           output_shapes=output_shapes,
                           rerandomize_each_iteration=rerandomize_each_iteration,
                           metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("rerandomize_each_iteration",
              _op._get_attr_bool("rerandomize_each_iteration"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomDatasetV2 = tf_export("raw_ops.RandomDatasetV2")(_ops.to_raw_op(random_dataset_v2))


def random_dataset_v2_eager_fallback(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, rerandomize_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'random_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'random_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if rerandomize_each_iteration is None:
    rerandomize_each_iteration = False
  rerandomize_each_iteration = _execute.make_bool(rerandomize_each_iteration, "rerandomize_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
  _inputs_flat = [seed, seed2, seed_generator]
  _attrs = ("rerandomize_each_iteration", rerandomize_each_iteration,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"RandomDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def rebatch_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_replicas: Annotated[Any, _atypes.Int64], output_types, output_shapes, use_fallback:bool=True, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that changes the batch size.

  Creates a dataset that changes the batch size of the dataset to current batch
  size // num_workers.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    num_replicas: A `Tensor` of type `int64`.
      A scalar representing the number of replicas to distribute this batch across. As
      a result of this transformation the current batch size would end up being
      divided  by this parameter.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_fallback: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RebatchDataset", name, input_dataset, num_replicas,
        "output_types", output_types, "output_shapes", output_shapes,
        "use_fallback", use_fallback)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rebatch_dataset_eager_fallback(
          input_dataset, num_replicas, output_types=output_types,
          output_shapes=output_shapes, use_fallback=use_fallback, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rebatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rebatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_fallback is None:
    use_fallback = True
  use_fallback = _execute.make_bool(use_fallback, "use_fallback")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RebatchDataset", input_dataset=input_dataset,
                          num_replicas=num_replicas,
                          output_types=output_types,
                          output_shapes=output_shapes,
                          use_fallback=use_fallback, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_fallback",
              _op._get_attr_bool("use_fallback"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RebatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RebatchDataset = tf_export("raw_ops.RebatchDataset")(_ops.to_raw_op(rebatch_dataset))


def rebatch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_replicas: Annotated[Any, _atypes.Int64], output_types, output_shapes, use_fallback: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rebatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rebatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_fallback is None:
    use_fallback = True
  use_fallback = _execute.make_bool(use_fallback, "use_fallback")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_replicas = _ops.convert_to_tensor(num_replicas, _dtypes.int64)
  _inputs_flat = [input_dataset, num_replicas]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "use_fallback", use_fallback)
  _result = _execute.execute(b"RebatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RebatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def rebatch_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], batch_sizes: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that changes the batch size.

  Creates a dataset that rebatches elements from `input_dataset` into new batch
  sizes.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    batch_sizes: A `Tensor` of type `int64`.
      A vector of integers representing the size of batches to produce. These values
      are cycled through in order.
    drop_remainder: A `Tensor` of type `bool`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RebatchDatasetV2", name, input_dataset, batch_sizes,
        drop_remainder, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rebatch_dataset_v2_eager_fallback(
          input_dataset, batch_sizes, drop_remainder,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rebatch_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rebatch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RebatchDatasetV2", input_dataset=input_dataset,
                            batch_sizes=batch_sizes,
                            drop_remainder=drop_remainder,
                            output_types=output_types,
                            output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RebatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RebatchDatasetV2 = tf_export("raw_ops.RebatchDatasetV2")(_ops.to_raw_op(rebatch_dataset_v2))


def rebatch_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_sizes: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rebatch_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rebatch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_sizes = _ops.convert_to_tensor(batch_sizes, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset, batch_sizes, drop_remainder]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"RebatchDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RebatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def register_dataset(dataset: Annotated[Any, _atypes.Variant], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], external_state_policy: int, element_spec:str="", metadata:str="", name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Registers a dataset with the tf.data service.

  Args:
    dataset: A `Tensor` of type `variant`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    external_state_policy: An `int`.
    element_spec: An optional `string`. Defaults to `""`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RegisterDataset", name, dataset, address, protocol,
        "external_state_policy", external_state_policy, "element_spec",
        element_spec, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return register_dataset_eager_fallback(
          dataset, address, protocol,
          external_state_policy=external_state_policy,
          element_spec=element_spec, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if element_spec is None:
    element_spec = ""
  element_spec = _execute.make_str(element_spec, "element_spec")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RegisterDataset", dataset=dataset, address=address,
                           protocol=protocol,
                           external_state_policy=external_state_policy,
                           element_spec=element_spec, metadata=metadata,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("external_state_policy",
              _op._get_attr_int("external_state_policy"), "element_spec",
              _op.get_attr("element_spec"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RegisterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RegisterDataset = tf_export("raw_ops.RegisterDataset")(_ops.to_raw_op(register_dataset))


def register_dataset_eager_fallback(dataset: Annotated[Any, _atypes.Variant], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], external_state_policy: int, element_spec: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Int64]:
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if element_spec is None:
    element_spec = ""
  element_spec = _execute.make_str(element_spec, "element_spec")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  _inputs_flat = [dataset, address, protocol]
  _attrs = ("external_state_policy", external_state_policy, "element_spec",
  element_spec, "metadata", metadata)
  _result = _execute.execute(b"RegisterDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RegisterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def register_dataset_v2(dataset: Annotated[Any, _atypes.Variant], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], external_state_policy: int, element_spec:str="", requested_dataset_id:str="", metadata:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Registers a dataset with the tf.data service.

  Args:
    dataset: A `Tensor` of type `variant`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    external_state_policy: An `int`.
    element_spec: An optional `string`. Defaults to `""`.
    requested_dataset_id: An optional `string`. Defaults to `""`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RegisterDatasetV2", name, dataset, address, protocol,
        "external_state_policy", external_state_policy, "element_spec",
        element_spec, "requested_dataset_id", requested_dataset_id,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return register_dataset_v2_eager_fallback(
          dataset, address, protocol,
          external_state_policy=external_state_policy,
          element_spec=element_spec,
          requested_dataset_id=requested_dataset_id, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if element_spec is None:
    element_spec = ""
  element_spec = _execute.make_str(element_spec, "element_spec")
  if requested_dataset_id is None:
    requested_dataset_id = ""
  requested_dataset_id = _execute.make_str(requested_dataset_id, "requested_dataset_id")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RegisterDatasetV2", dataset=dataset, address=address,
                             protocol=protocol,
                             external_state_policy=external_state_policy,
                             element_spec=element_spec,
                             requested_dataset_id=requested_dataset_id,
                             metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("external_state_policy",
              _op._get_attr_int("external_state_policy"), "element_spec",
              _op.get_attr("element_spec"), "requested_dataset_id",
              _op.get_attr("requested_dataset_id"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RegisterDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RegisterDatasetV2 = tf_export("raw_ops.RegisterDatasetV2")(_ops.to_raw_op(register_dataset_v2))


def register_dataset_v2_eager_fallback(dataset: Annotated[Any, _atypes.Variant], address: Annotated[Any, _atypes.String], protocol: Annotated[Any, _atypes.String], external_state_policy: int, element_spec: str, requested_dataset_id: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.String]:
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if element_spec is None:
    element_spec = ""
  element_spec = _execute.make_str(element_spec, "element_spec")
  if requested_dataset_id is None:
    requested_dataset_id = ""
  requested_dataset_id = _execute.make_str(requested_dataset_id, "requested_dataset_id")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  address = _ops.convert_to_tensor(address, _dtypes.string)
  protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
  _inputs_flat = [dataset, address, protocol]
  _attrs = ("external_state_policy", external_state_policy, "element_spec",
  element_spec, "requested_dataset_id", requested_dataset_id, "metadata",
  metadata)
  _result = _execute.execute(b"RegisterDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RegisterDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sampling_dataset(input_dataset: Annotated[Any, _atypes.Variant], rate: Annotated[Any, _atypes.Float32], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that takes a Bernoulli sample of the contents of another dataset.

  There is no transformation in the `tf.data` Python API for creating this dataset.
  Instead, it is created as a result of the `filter_with_random_uniform_fusion`
  static optimization. Whether this optimization is performed is determined by the
  `experimental_optimization.filter_with_random_uniform_fusion` option of
  `tf.data.Options`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    rate: A `Tensor` of type `float32`.
      A scalar representing the sample rate. Each element of `input_dataset` is
      retained with this probability, independent of all other elements.
    seed: A `Tensor` of type `int64`.
      A scalar representing seed of random number generator.
    seed2: A `Tensor` of type `int64`.
      A scalar representing seed2 of random number generator.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SamplingDataset", name, input_dataset, rate, seed, seed2,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sampling_dataset_eager_fallback(
          input_dataset, rate, seed, seed2, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sampling_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sampling_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SamplingDataset", input_dataset=input_dataset, rate=rate, seed=seed,
                           seed2=seed2, output_types=output_types,
                           output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SamplingDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SamplingDataset = tf_export("raw_ops.SamplingDataset")(_ops.to_raw_op(sampling_dataset))


def sampling_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], rate: Annotated[Any, _atypes.Float32], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sampling_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sampling_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  rate = _ops.convert_to_tensor(rate, _dtypes.float32)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  _inputs_flat = [input_dataset, rate, seed, seed2]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"SamplingDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SamplingDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def save_dataset(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], shard_func_other_args, shard_func, compression:str="", use_shard_func:bool=True, name=None):
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    path: A `Tensor` of type `string`.
    shard_func_other_args: A list of `Tensor` objects.
    shard_func: A function decorated with @Defun.
    compression: An optional `string`. Defaults to `""`.
    use_shard_func: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SaveDataset", name, input_dataset, path, shard_func_other_args,
        "compression", compression, "shard_func", shard_func,
        "use_shard_func", use_shard_func)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return save_dataset_eager_fallback(
          input_dataset, path, shard_func_other_args, compression=compression,
          shard_func=shard_func, use_shard_func=use_shard_func, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if use_shard_func is None:
    use_shard_func = True
  use_shard_func = _execute.make_bool(use_shard_func, "use_shard_func")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SaveDataset", input_dataset=input_dataset, path=path,
                       shard_func_other_args=shard_func_other_args,
                       shard_func=shard_func, compression=compression,
                       use_shard_func=use_shard_func, name=name)
  return _op
SaveDataset = tf_export("raw_ops.SaveDataset")(_ops.to_raw_op(save_dataset))


def save_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], shard_func_other_args, shard_func, compression: str, use_shard_func: bool, name, ctx):
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if use_shard_func is None:
    use_shard_func = True
  use_shard_func = _execute.make_bool(use_shard_func, "use_shard_func")
  _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [input_dataset, path] + list(shard_func_other_args)
  _attrs = ("compression", compression, "shard_func", shard_func,
  "use_shard_func", use_shard_func, "Tshard_func_args",
  _attr_Tshard_func_args)
  _result = _execute.execute(b"SaveDataset", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def save_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], shard_func_other_args, shard_func, output_types, output_shapes, compression:str="", use_shard_func:bool=True, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    path: A `Tensor` of type `string`.
    shard_func_other_args: A list of `Tensor` objects.
    shard_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    compression: An optional `string`. Defaults to `""`.
    use_shard_func: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SaveDatasetV2", name, input_dataset, path,
        shard_func_other_args, "compression", compression, "shard_func",
        shard_func, "use_shard_func", use_shard_func, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return save_dataset_v2_eager_fallback(
          input_dataset, path, shard_func_other_args, compression=compression,
          shard_func=shard_func, use_shard_func=use_shard_func,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'save_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'save_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if use_shard_func is None:
    use_shard_func = True
  use_shard_func = _execute.make_bool(use_shard_func, "use_shard_func")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SaveDatasetV2", input_dataset=input_dataset, path=path,
                         shard_func_other_args=shard_func_other_args,
                         shard_func=shard_func, output_types=output_types,
                         output_shapes=output_shapes, compression=compression,
                         use_shard_func=use_shard_func, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("compression", _op.get_attr("compression"), "shard_func",
              _op.get_attr("shard_func"), "use_shard_func",
              _op._get_attr_bool("use_shard_func"), "Tshard_func_args",
              _op.get_attr("Tshard_func_args"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SaveDatasetV2 = tf_export("raw_ops.SaveDatasetV2")(_ops.to_raw_op(save_dataset_v2))


def save_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], shard_func_other_args, shard_func, output_types, output_shapes, compression: str, use_shard_func: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'save_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'save_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if use_shard_func is None:
    use_shard_func = True
  use_shard_func = _execute.make_bool(use_shard_func, "use_shard_func")
  _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [input_dataset, path] + list(shard_func_other_args)
  _attrs = ("compression", compression, "shard_func", shard_func,
  "use_shard_func", use_shard_func, "Tshard_func_args",
  _attr_Tshard_func_args, "output_types", output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"SaveDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def scan_dataset(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, preserve_cardinality:bool=False, use_default_device:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset successively reduces `f` over the elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    initial_state: A list of `Tensor` objects.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    use_default_device: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScanDataset", name, input_dataset, initial_state,
        other_arguments, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "preserve_cardinality",
        preserve_cardinality, "use_default_device", use_default_device,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return scan_dataset_eager_fallback(
          input_dataset, initial_state, other_arguments, f=f,
          output_types=output_types, output_shapes=output_shapes,
          preserve_cardinality=preserve_cardinality,
          use_default_device=use_default_device, metadata=metadata, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'scan_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'scan_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if use_default_device is None:
    use_default_device = True
  use_default_device = _execute.make_bool(use_default_device, "use_default_device")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScanDataset", input_dataset=input_dataset,
                       initial_state=initial_state,
                       other_arguments=other_arguments, f=f,
                       output_types=output_types, output_shapes=output_shapes,
                       preserve_cardinality=preserve_cardinality,
                       use_default_device=use_default_device,
                       metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Tstate", _op.get_attr("Tstate"),
              "Targuments", _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"),
              "use_default_device", _op._get_attr_bool("use_default_device"),
              "metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScanDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScanDataset = tf_export("raw_ops.ScanDataset")(_ops.to_raw_op(scan_dataset))


def scan_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, preserve_cardinality: bool, use_default_device: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'scan_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'scan_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if use_default_device is None:
    use_default_device = True
  use_default_device = _execute.make_bool(use_default_device, "use_default_device")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tstate, initial_state = _execute.convert_to_mixed_eager_tensors(initial_state, ctx)
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(initial_state) + list(other_arguments)
  _attrs = ("f", f, "Tstate", _attr_Tstate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes,
  "preserve_cardinality", preserve_cardinality, "use_default_device",
  use_default_device, "metadata", metadata)
  _result = _execute.execute(b"ScanDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScanDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def set_stats_aggregator_dataset(input_dataset: Annotated[Any, _atypes.Variant], stats_aggregator: Annotated[Any, _atypes.Resource], tag: Annotated[Any, _atypes.String], counter_prefix: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    stats_aggregator: A `Tensor` of type `resource`.
    tag: A `Tensor` of type `string`.
    counter_prefix: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SetStatsAggregatorDataset", name, input_dataset,
        stats_aggregator, tag, counter_prefix, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return set_stats_aggregator_dataset_eager_fallback(
          input_dataset, stats_aggregator, tag, counter_prefix,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'set_stats_aggregator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'set_stats_aggregator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SetStatsAggregatorDataset", input_dataset=input_dataset,
                                     stats_aggregator=stats_aggregator,
                                     tag=tag, counter_prefix=counter_prefix,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SetStatsAggregatorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SetStatsAggregatorDataset = tf_export("raw_ops.SetStatsAggregatorDataset")(_ops.to_raw_op(set_stats_aggregator_dataset))


def set_stats_aggregator_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], stats_aggregator: Annotated[Any, _atypes.Resource], tag: Annotated[Any, _atypes.String], counter_prefix: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'set_stats_aggregator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'set_stats_aggregator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  stats_aggregator = _ops.convert_to_tensor(stats_aggregator, _dtypes.resource)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  counter_prefix = _ops.convert_to_tensor(counter_prefix, _dtypes.string)
  _inputs_flat = [input_dataset, stats_aggregator, tag, counter_prefix]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"SetStatsAggregatorDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SetStatsAggregatorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sleep_dataset(input_dataset: Annotated[Any, _atypes.Variant], sleep_microseconds: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    sleep_microseconds: A `Tensor` of type `int64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SleepDataset", name, input_dataset, sleep_microseconds,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sleep_dataset_eager_fallback(
          input_dataset, sleep_microseconds, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sleep_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sleep_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SleepDataset", input_dataset=input_dataset,
                        sleep_microseconds=sleep_microseconds,
                        output_types=output_types,
                        output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SleepDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SleepDataset = tf_export("raw_ops.SleepDataset")(_ops.to_raw_op(sleep_dataset))


def sleep_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], sleep_microseconds: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sleep_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sleep_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  sleep_microseconds = _ops.convert_to_tensor(sleep_microseconds, _dtypes.int64)
  _inputs_flat = [input_dataset, sleep_microseconds]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"SleepDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SleepDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sliding_window_dataset(input_dataset: Annotated[Any, _atypes.Variant], window_size: Annotated[Any, _atypes.Int64], window_shift: Annotated[Any, _atypes.Int64], window_stride: Annotated[Any, _atypes.Int64], output_types, output_shapes, drop_remainder:bool=True, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that passes a sliding window over `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    window_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements in the
      sliding window.
    window_shift: A `Tensor` of type `int64`.
      A scalar representing the steps moving the sliding window
      forward in one iteration. It must be positive.
    window_stride: A `Tensor` of type `int64`.
      A scalar representing the stride of the input elements of the sliding window.
      It must be positive.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    drop_remainder: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SlidingWindowDataset", name, input_dataset, window_size,
        window_shift, window_stride, "drop_remainder", drop_remainder,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sliding_window_dataset_eager_fallback(
          input_dataset, window_size, window_shift, window_stride,
          drop_remainder=drop_remainder, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sliding_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sliding_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if drop_remainder is None:
    drop_remainder = True
  drop_remainder = _execute.make_bool(drop_remainder, "drop_remainder")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SlidingWindowDataset", input_dataset=input_dataset,
                                window_size=window_size,
                                window_shift=window_shift,
                                window_stride=window_stride,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                drop_remainder=drop_remainder, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("drop_remainder", _op._get_attr_bool("drop_remainder"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SlidingWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SlidingWindowDataset = tf_export("raw_ops.SlidingWindowDataset")(_ops.to_raw_op(sliding_window_dataset))


def sliding_window_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], window_size: Annotated[Any, _atypes.Int64], window_shift: Annotated[Any, _atypes.Int64], window_stride: Annotated[Any, _atypes.Int64], output_types, output_shapes, drop_remainder: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sliding_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sliding_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if drop_remainder is None:
    drop_remainder = True
  drop_remainder = _execute.make_bool(drop_remainder, "drop_remainder")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  window_size = _ops.convert_to_tensor(window_size, _dtypes.int64)
  window_shift = _ops.convert_to_tensor(window_shift, _dtypes.int64)
  window_stride = _ops.convert_to_tensor(window_stride, _dtypes.int64)
  _inputs_flat = [input_dataset, window_size, window_shift, window_stride]
  _attrs = ("drop_remainder", drop_remainder, "output_types", output_types,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"SlidingWindowDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SlidingWindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def snapshot_chunk_dataset(chunk_file: Annotated[Any, _atypes.String], output_types, output_shapes, compression:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    chunk_file: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    compression: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SnapshotChunkDataset", name, chunk_file, "output_types",
        output_types, "output_shapes", output_shapes, "compression",
        compression)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_chunk_dataset_eager_fallback(
          chunk_file, output_types=output_types, output_shapes=output_shapes,
          compression=compression, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_chunk_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_chunk_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SnapshotChunkDataset", chunk_file=chunk_file,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                compression=compression, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "compression",
              _op.get_attr("compression"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SnapshotChunkDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SnapshotChunkDataset = tf_export("raw_ops.SnapshotChunkDataset")(_ops.to_raw_op(snapshot_chunk_dataset))


def snapshot_chunk_dataset_eager_fallback(chunk_file: Annotated[Any, _atypes.String], output_types, output_shapes, compression: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_chunk_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_chunk_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  chunk_file = _ops.convert_to_tensor(chunk_file, _dtypes.string)
  _inputs_flat = [chunk_file]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "compression", compression)
  _result = _execute.execute(b"SnapshotChunkDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SnapshotChunkDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def snapshot_dataset(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], output_types, output_shapes, compression:str="", reader_path_prefix:str="", writer_path_prefix:str="", shard_size_bytes:int=10737418240, pending_snapshot_expiry_seconds:int=86400, num_reader_threads:int=1, reader_buffer_size:int=1, num_writer_threads:int=1, writer_buffer_size:int=1, shuffle_on_read:bool=False, seed:int=0, seed2:int=0, mode:str="auto", snapshot_name:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that will write to / read from a snapshot.

  This dataset attempts to determine whether a valid snapshot exists at the
  `snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
  If not, it will run the preprocessing pipeline as usual, and write out a
  snapshot of the data processed for future use.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    path: A `Tensor` of type `string`.
      The path we should write snapshots to / read snapshots from.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    compression: An optional `string`. Defaults to `""`.
    reader_path_prefix: An optional `string`. Defaults to `""`.
    writer_path_prefix: An optional `string`. Defaults to `""`.
    shard_size_bytes: An optional `int`. Defaults to `10737418240`.
    pending_snapshot_expiry_seconds: An optional `int`. Defaults to `86400`.
    num_reader_threads: An optional `int`. Defaults to `1`.
    reader_buffer_size: An optional `int`. Defaults to `1`.
    num_writer_threads: An optional `int`. Defaults to `1`.
    writer_buffer_size: An optional `int`. Defaults to `1`.
    shuffle_on_read: An optional `bool`. Defaults to `False`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    mode: An optional `string`. Defaults to `"auto"`.
    snapshot_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SnapshotDataset", name, input_dataset, path, "output_types",
        output_types, "output_shapes", output_shapes, "compression",
        compression, "reader_path_prefix", reader_path_prefix,
        "writer_path_prefix", writer_path_prefix, "shard_size_bytes",
        shard_size_bytes, "pending_snapshot_expiry_seconds",
        pending_snapshot_expiry_seconds, "num_reader_threads",
        num_reader_threads, "reader_buffer_size", reader_buffer_size,
        "num_writer_threads", num_writer_threads, "writer_buffer_size",
        writer_buffer_size, "shuffle_on_read", shuffle_on_read, "seed", seed,
        "seed2", seed2, "mode", mode, "snapshot_name", snapshot_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_dataset_eager_fallback(
          input_dataset, path, output_types=output_types,
          output_shapes=output_shapes, compression=compression,
          reader_path_prefix=reader_path_prefix,
          writer_path_prefix=writer_path_prefix,
          shard_size_bytes=shard_size_bytes,
          pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
          num_reader_threads=num_reader_threads,
          reader_buffer_size=reader_buffer_size,
          num_writer_threads=num_writer_threads,
          writer_buffer_size=writer_buffer_size,
          shuffle_on_read=shuffle_on_read, seed=seed, seed2=seed2, mode=mode,
          snapshot_name=snapshot_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if reader_path_prefix is None:
    reader_path_prefix = ""
  reader_path_prefix = _execute.make_str(reader_path_prefix, "reader_path_prefix")
  if writer_path_prefix is None:
    writer_path_prefix = ""
  writer_path_prefix = _execute.make_str(writer_path_prefix, "writer_path_prefix")
  if shard_size_bytes is None:
    shard_size_bytes = 10737418240
  shard_size_bytes = _execute.make_int(shard_size_bytes, "shard_size_bytes")
  if pending_snapshot_expiry_seconds is None:
    pending_snapshot_expiry_seconds = 86400
  pending_snapshot_expiry_seconds = _execute.make_int(pending_snapshot_expiry_seconds, "pending_snapshot_expiry_seconds")
  if num_reader_threads is None:
    num_reader_threads = 1
  num_reader_threads = _execute.make_int(num_reader_threads, "num_reader_threads")
  if reader_buffer_size is None:
    reader_buffer_size = 1
  reader_buffer_size = _execute.make_int(reader_buffer_size, "reader_buffer_size")
  if num_writer_threads is None:
    num_writer_threads = 1
  num_writer_threads = _execute.make_int(num_writer_threads, "num_writer_threads")
  if writer_buffer_size is None:
    writer_buffer_size = 1
  writer_buffer_size = _execute.make_int(writer_buffer_size, "writer_buffer_size")
  if shuffle_on_read is None:
    shuffle_on_read = False
  shuffle_on_read = _execute.make_bool(shuffle_on_read, "shuffle_on_read")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if mode is None:
    mode = "auto"
  mode = _execute.make_str(mode, "mode")
  if snapshot_name is None:
    snapshot_name = ""
  snapshot_name = _execute.make_str(snapshot_name, "snapshot_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SnapshotDataset", input_dataset=input_dataset, path=path,
                           output_types=output_types,
                           output_shapes=output_shapes,
                           compression=compression,
                           reader_path_prefix=reader_path_prefix,
                           writer_path_prefix=writer_path_prefix,
                           shard_size_bytes=shard_size_bytes,
                           pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
                           num_reader_threads=num_reader_threads,
                           reader_buffer_size=reader_buffer_size,
                           num_writer_threads=num_writer_threads,
                           writer_buffer_size=writer_buffer_size,
                           shuffle_on_read=shuffle_on_read, seed=seed,
                           seed2=seed2, mode=mode,
                           snapshot_name=snapshot_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "compression",
              _op.get_attr("compression"), "reader_path_prefix",
              _op.get_attr("reader_path_prefix"), "writer_path_prefix",
              _op.get_attr("writer_path_prefix"), "shard_size_bytes",
              _op._get_attr_int("shard_size_bytes"),
              "pending_snapshot_expiry_seconds",
              _op._get_attr_int("pending_snapshot_expiry_seconds"),
              "num_reader_threads", _op._get_attr_int("num_reader_threads"),
              "reader_buffer_size", _op._get_attr_int("reader_buffer_size"),
              "num_writer_threads", _op._get_attr_int("num_writer_threads"),
              "writer_buffer_size", _op._get_attr_int("writer_buffer_size"),
              "shuffle_on_read", _op._get_attr_bool("shuffle_on_read"),
              "seed", _op._get_attr_int("seed"), "seed2",
              _op._get_attr_int("seed2"), "mode", _op.get_attr("mode"),
              "snapshot_name", _op.get_attr("snapshot_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SnapshotDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SnapshotDataset = tf_export("raw_ops.SnapshotDataset")(_ops.to_raw_op(snapshot_dataset))


def snapshot_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], output_types, output_shapes, compression: str, reader_path_prefix: str, writer_path_prefix: str, shard_size_bytes: int, pending_snapshot_expiry_seconds: int, num_reader_threads: int, reader_buffer_size: int, num_writer_threads: int, writer_buffer_size: int, shuffle_on_read: bool, seed: int, seed2: int, mode: str, snapshot_name: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if reader_path_prefix is None:
    reader_path_prefix = ""
  reader_path_prefix = _execute.make_str(reader_path_prefix, "reader_path_prefix")
  if writer_path_prefix is None:
    writer_path_prefix = ""
  writer_path_prefix = _execute.make_str(writer_path_prefix, "writer_path_prefix")
  if shard_size_bytes is None:
    shard_size_bytes = 10737418240
  shard_size_bytes = _execute.make_int(shard_size_bytes, "shard_size_bytes")
  if pending_snapshot_expiry_seconds is None:
    pending_snapshot_expiry_seconds = 86400
  pending_snapshot_expiry_seconds = _execute.make_int(pending_snapshot_expiry_seconds, "pending_snapshot_expiry_seconds")
  if num_reader_threads is None:
    num_reader_threads = 1
  num_reader_threads = _execute.make_int(num_reader_threads, "num_reader_threads")
  if reader_buffer_size is None:
    reader_buffer_size = 1
  reader_buffer_size = _execute.make_int(reader_buffer_size, "reader_buffer_size")
  if num_writer_threads is None:
    num_writer_threads = 1
  num_writer_threads = _execute.make_int(num_writer_threads, "num_writer_threads")
  if writer_buffer_size is None:
    writer_buffer_size = 1
  writer_buffer_size = _execute.make_int(writer_buffer_size, "writer_buffer_size")
  if shuffle_on_read is None:
    shuffle_on_read = False
  shuffle_on_read = _execute.make_bool(shuffle_on_read, "shuffle_on_read")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if mode is None:
    mode = "auto"
  mode = _execute.make_str(mode, "mode")
  if snapshot_name is None:
    snapshot_name = ""
  snapshot_name = _execute.make_str(snapshot_name, "snapshot_name")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [input_dataset, path]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "compression", compression, "reader_path_prefix", reader_path_prefix,
  "writer_path_prefix", writer_path_prefix, "shard_size_bytes",
  shard_size_bytes, "pending_snapshot_expiry_seconds",
  pending_snapshot_expiry_seconds, "num_reader_threads", num_reader_threads,
  "reader_buffer_size", reader_buffer_size, "num_writer_threads",
  num_writer_threads, "writer_buffer_size", writer_buffer_size,
  "shuffle_on_read", shuffle_on_read, "seed", seed, "seed2", seed2, "mode",
  mode, "snapshot_name", snapshot_name)
  _result = _execute.execute(b"SnapshotDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SnapshotDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def snapshot_dataset_reader(shard_dir: Annotated[Any, _atypes.String], start_index: Annotated[Any, _atypes.Int64], output_types, output_shapes, version: int, compression:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    shard_dir: A `Tensor` of type `string`.
    start_index: A `Tensor` of type `int64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    version: An `int`.
    compression: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SnapshotDatasetReader", name, shard_dir, start_index,
        "output_types", output_types, "output_shapes", output_shapes,
        "compression", compression, "version", version)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_dataset_reader_eager_fallback(
          shard_dir, start_index, output_types=output_types,
          output_shapes=output_shapes, compression=compression,
          version=version, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset_reader' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset_reader' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  version = _execute.make_int(version, "version")
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SnapshotDatasetReader", shard_dir=shard_dir, start_index=start_index,
                                 output_types=output_types,
                                 output_shapes=output_shapes, version=version,
                                 compression=compression, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "compression",
              _op.get_attr("compression"), "version",
              _op._get_attr_int("version"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SnapshotDatasetReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SnapshotDatasetReader = tf_export("raw_ops.SnapshotDatasetReader")(_ops.to_raw_op(snapshot_dataset_reader))


def snapshot_dataset_reader_eager_fallback(shard_dir: Annotated[Any, _atypes.String], start_index: Annotated[Any, _atypes.Int64], output_types, output_shapes, version: int, compression: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset_reader' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset_reader' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  version = _execute.make_int(version, "version")
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  shard_dir = _ops.convert_to_tensor(shard_dir, _dtypes.string)
  start_index = _ops.convert_to_tensor(start_index, _dtypes.int64)
  _inputs_flat = [shard_dir, start_index]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "compression", compression, "version", version)
  _result = _execute.execute(b"SnapshotDatasetReader", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SnapshotDatasetReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def snapshot_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], reader_func_other_args, shard_func_other_args, output_types, output_shapes, reader_func, shard_func, compression:str="", reader_prefix:str="", writer_prefix:str="", hash_valid:bool=False, hash:int=0, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that will write to / read from a snapshot.

  This dataset attempts to determine whether a valid snapshot exists at the
  `snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
  If not, it will run the preprocessing pipeline as usual, and write out a
  snapshot of the data processed for future use.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    path: A `Tensor` of type `string`.
      The path we should write snapshots to / read snapshots from.
    reader_func_other_args: A list of `Tensor` objects.
    shard_func_other_args: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reader_func: A function decorated with @Defun.
      Optional. A function to control how to read data from snapshot shards.
    shard_func: A function decorated with @Defun.
      Optional. A function to control how to shard data when writing a snapshot.
    compression: An optional `string`. Defaults to `""`.
      The type of compression to be applied to the saved snapshot files.
    reader_prefix: An optional `string`. Defaults to `""`.
    writer_prefix: An optional `string`. Defaults to `""`.
    hash_valid: An optional `bool`. Defaults to `False`.
    hash: An optional `int`. Defaults to `0`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SnapshotDatasetV2", name, input_dataset, path,
        reader_func_other_args, shard_func_other_args, "output_types",
        output_types, "output_shapes", output_shapes, "compression",
        compression, "reader_prefix", reader_prefix, "writer_prefix",
        writer_prefix, "hash_valid", hash_valid, "hash", hash, "reader_func",
        reader_func, "shard_func", shard_func, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_dataset_v2_eager_fallback(
          input_dataset, path, reader_func_other_args, shard_func_other_args,
          output_types=output_types, output_shapes=output_shapes,
          compression=compression, reader_prefix=reader_prefix,
          writer_prefix=writer_prefix, hash_valid=hash_valid, hash=hash,
          reader_func=reader_func, shard_func=shard_func, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if reader_prefix is None:
    reader_prefix = ""
  reader_prefix = _execute.make_str(reader_prefix, "reader_prefix")
  if writer_prefix is None:
    writer_prefix = ""
  writer_prefix = _execute.make_str(writer_prefix, "writer_prefix")
  if hash_valid is None:
    hash_valid = False
  hash_valid = _execute.make_bool(hash_valid, "hash_valid")
  if hash is None:
    hash = 0
  hash = _execute.make_int(hash, "hash")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SnapshotDatasetV2", input_dataset=input_dataset, path=path,
                             reader_func_other_args=reader_func_other_args,
                             shard_func_other_args=shard_func_other_args,
                             output_types=output_types,
                             output_shapes=output_shapes,
                             reader_func=reader_func, shard_func=shard_func,
                             compression=compression,
                             reader_prefix=reader_prefix,
                             writer_prefix=writer_prefix,
                             hash_valid=hash_valid, hash=hash,
                             metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "compression",
              _op.get_attr("compression"), "reader_prefix",
              _op.get_attr("reader_prefix"), "writer_prefix",
              _op.get_attr("writer_prefix"), "hash_valid",
              _op._get_attr_bool("hash_valid"), "hash",
              _op._get_attr_int("hash"), "reader_func",
              _op.get_attr("reader_func"), "shard_func",
              _op.get_attr("shard_func"), "Treader_func_args",
              _op.get_attr("Treader_func_args"), "Tshard_func_args",
              _op.get_attr("Tshard_func_args"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SnapshotDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SnapshotDatasetV2 = tf_export("raw_ops.SnapshotDatasetV2")(_ops.to_raw_op(snapshot_dataset_v2))


def snapshot_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], path: Annotated[Any, _atypes.String], reader_func_other_args, shard_func_other_args, output_types, output_shapes, reader_func, shard_func, compression: str, reader_prefix: str, writer_prefix: str, hash_valid: bool, hash: int, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if compression is None:
    compression = ""
  compression = _execute.make_str(compression, "compression")
  if reader_prefix is None:
    reader_prefix = ""
  reader_prefix = _execute.make_str(reader_prefix, "reader_prefix")
  if writer_prefix is None:
    writer_prefix = ""
  writer_prefix = _execute.make_str(writer_prefix, "writer_prefix")
  if hash_valid is None:
    hash_valid = False
  hash_valid = _execute.make_bool(hash_valid, "hash_valid")
  if hash is None:
    hash = 0
  hash = _execute.make_int(hash, "hash")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Treader_func_args, reader_func_other_args = _execute.convert_to_mixed_eager_tensors(reader_func_other_args, ctx)
  _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [input_dataset, path] + list(reader_func_other_args) + list(shard_func_other_args)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "compression", compression, "reader_prefix", reader_prefix, "writer_prefix",
  writer_prefix, "hash_valid", hash_valid, "hash", hash, "reader_func",
  reader_func, "shard_func", shard_func, "Treader_func_args",
  _attr_Treader_func_args, "Tshard_func_args", _attr_Tshard_func_args,
  "metadata", metadata)
  _result = _execute.execute(b"SnapshotDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SnapshotDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def snapshot_nested_dataset_reader(inputs: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    inputs: A list of at least 1 `Tensor` objects with type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SnapshotNestedDatasetReader", name, inputs, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return snapshot_nested_dataset_reader_eager_fallback(
          inputs, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SnapshotNestedDatasetReader", inputs=inputs,
                                       output_types=output_types,
                                       output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SnapshotNestedDatasetReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SnapshotNestedDatasetReader = tf_export("raw_ops.SnapshotNestedDatasetReader")(_ops.to_raw_op(snapshot_nested_dataset_reader))


def snapshot_nested_dataset_reader_eager_fallback(inputs: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'snapshot_nested_dataset_reader' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  inputs = _ops.convert_n_to_tensor(inputs, _dtypes.variant)
  _inputs_flat = list(inputs)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes, "N",
  _attr_N)
  _result = _execute.execute(b"SnapshotNestedDatasetReader", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SnapshotNestedDatasetReader", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def sql_dataset(driver_name: Annotated[Any, _atypes.String], data_source_name: Annotated[Any, _atypes.String], query: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that executes a SQL query and emits rows of the result set.

  Args:
    driver_name: A `Tensor` of type `string`.
      The database type. Currently, the only supported type is 'sqlite'.
    data_source_name: A `Tensor` of type `string`.
      A connection string to connect to the database.
    query: A `Tensor` of type `string`. A SQL query to execute.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SqlDataset", name, driver_name, data_source_name, query,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sql_dataset_eager_fallback(
          driver_name, data_source_name, query, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sql_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sql_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SqlDataset", driver_name=driver_name,
                      data_source_name=data_source_name, query=query,
                      output_types=output_types, output_shapes=output_shapes,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SqlDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SqlDataset = tf_export("raw_ops.SqlDataset")(_ops.to_raw_op(sql_dataset))


def sql_dataset_eager_fallback(driver_name: Annotated[Any, _atypes.String], data_source_name: Annotated[Any, _atypes.String], query: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sql_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sql_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  driver_name = _ops.convert_to_tensor(driver_name, _dtypes.string)
  data_source_name = _ops.convert_to_tensor(data_source_name, _dtypes.string)
  query = _ops.convert_to_tensor(query, _dtypes.string)
  _inputs_flat = [driver_name, data_source_name, query]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"SqlDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SqlDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def stats_aggregator_handle(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a statistics manager resource.

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatsAggregatorHandle", name, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stats_aggregator_handle_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatsAggregatorHandle", container=container, shared_name=shared_name,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatsAggregatorHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatsAggregatorHandle = tf_export("raw_ops.StatsAggregatorHandle")(_ops.to_raw_op(stats_aggregator_handle))


def stats_aggregator_handle_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StatsAggregatorHandle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatsAggregatorHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def stats_aggregator_handle_v2(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatsAggregatorHandleV2", name, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stats_aggregator_handle_v2_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatsAggregatorHandleV2", container=container,
                                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatsAggregatorHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatsAggregatorHandleV2 = tf_export("raw_ops.StatsAggregatorHandleV2")(_ops.to_raw_op(stats_aggregator_handle_v2))


def stats_aggregator_handle_v2_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StatsAggregatorHandleV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatsAggregatorHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def stats_aggregator_set_summary_writer(stats_aggregator: Annotated[Any, _atypes.Resource], summary: Annotated[Any, _atypes.Resource], name=None):
  r"""Set a summary_writer_interface to record statistics using given stats_aggregator.

  Args:
    stats_aggregator: A `Tensor` of type `resource`.
    summary: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatsAggregatorSetSummaryWriter", name, stats_aggregator,
        summary)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stats_aggregator_set_summary_writer_eager_fallback(
          stats_aggregator, summary, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatsAggregatorSetSummaryWriter", stats_aggregator=stats_aggregator,
                                           summary=summary, name=name)
  return _op
StatsAggregatorSetSummaryWriter = tf_export("raw_ops.StatsAggregatorSetSummaryWriter")(_ops.to_raw_op(stats_aggregator_set_summary_writer))


def stats_aggregator_set_summary_writer_eager_fallback(stats_aggregator: Annotated[Any, _atypes.Resource], summary: Annotated[Any, _atypes.Resource], name, ctx):
  stats_aggregator = _ops.convert_to_tensor(stats_aggregator, _dtypes.resource)
  summary = _ops.convert_to_tensor(summary, _dtypes.resource)
  _inputs_flat = [stats_aggregator, summary]
  _attrs = None
  _result = _execute.execute(b"StatsAggregatorSetSummaryWriter", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def stats_aggregator_summary(iterator: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Produces a summary of any statistics recorded by the given statistics manager.

  Args:
    iterator: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatsAggregatorSummary", name, iterator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stats_aggregator_summary_eager_fallback(
          iterator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatsAggregatorSummary", iterator=iterator, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatsAggregatorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StatsAggregatorSummary = tf_export("raw_ops.StatsAggregatorSummary")(_ops.to_raw_op(stats_aggregator_summary))


def stats_aggregator_summary_eager_fallback(iterator: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = None
  _result = _execute.execute(b"StatsAggregatorSummary", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatsAggregatorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def take_while_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that stops iteration when predicate` is false.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TakeWhileDataset", name, input_dataset, other_arguments,
        "predicate", predicate, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return take_while_dataset_eager_fallback(
          input_dataset, other_arguments, predicate=predicate,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'take_while_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'take_while_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TakeWhileDataset", input_dataset=input_dataset,
                            other_arguments=other_arguments,
                            predicate=predicate, output_types=output_types,
                            output_shapes=output_shapes, metadata=metadata,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("predicate", _op.get_attr("predicate"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TakeWhileDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TakeWhileDataset = tf_export("raw_ops.TakeWhileDataset")(_ops.to_raw_op(take_while_dataset))


def take_while_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'take_while_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'take_while_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("predicate", predicate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"TakeWhileDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TakeWhileDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def thread_pool_dataset(input_dataset: Annotated[Any, _atypes.Variant], thread_pool: Annotated[Any, _atypes.Resource], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    thread_pool: A `Tensor` of type `resource`.
      A resource produced by the ThreadPoolHandle op.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ThreadPoolDataset", name, input_dataset, thread_pool,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return thread_pool_dataset_eager_fallback(
          input_dataset, thread_pool, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ThreadPoolDataset", input_dataset=input_dataset,
                             thread_pool=thread_pool,
                             output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ThreadPoolDataset = tf_export("raw_ops.ThreadPoolDataset")(_ops.to_raw_op(thread_pool_dataset))


def thread_pool_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], thread_pool: Annotated[Any, _atypes.Resource], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  thread_pool = _ops.convert_to_tensor(thread_pool, _dtypes.resource)
  _inputs_flat = [input_dataset, thread_pool]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ThreadPoolDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ThreadPoolDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def thread_pool_handle(num_threads: int, display_name: str, max_intra_op_parallelism:int=1, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    num_threads: An `int`. The number of threads in the thread pool.
    display_name: A `string`.
      A human-readable name for the threads that may be visible in some
      visualizations.
      threadpool.
    max_intra_op_parallelism: An optional `int`. Defaults to `1`.
      The maximum degree of parallelism to use within operations that execute on this
      threadpool.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ThreadPoolHandle", name, "num_threads", num_threads,
        "max_intra_op_parallelism", max_intra_op_parallelism, "display_name",
        display_name, "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return thread_pool_handle_eager_fallback(
          num_threads=num_threads,
          max_intra_op_parallelism=max_intra_op_parallelism,
          display_name=display_name, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_threads = _execute.make_int(num_threads, "num_threads")
  display_name = _execute.make_str(display_name, "display_name")
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ThreadPoolHandle", num_threads=num_threads,
                            display_name=display_name,
                            max_intra_op_parallelism=max_intra_op_parallelism,
                            container=container, shared_name=shared_name,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_threads", _op._get_attr_int("num_threads"),
              "max_intra_op_parallelism",
              _op._get_attr_int("max_intra_op_parallelism"), "display_name",
              _op.get_attr("display_name"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ThreadPoolHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ThreadPoolHandle = tf_export("raw_ops.ThreadPoolHandle")(_ops.to_raw_op(thread_pool_handle))


def thread_pool_handle_eager_fallback(num_threads: int, display_name: str, max_intra_op_parallelism: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  num_threads = _execute.make_int(num_threads, "num_threads")
  display_name = _execute.make_str(display_name, "display_name")
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("num_threads", num_threads, "max_intra_op_parallelism",
  max_intra_op_parallelism, "display_name", display_name, "container",
  container, "shared_name", shared_name)
  _result = _execute.execute(b"ThreadPoolHandle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ThreadPoolHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def unbatch_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""A dataset that splits the elements of its input into multiple elements.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnbatchDataset", name, input_dataset, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unbatch_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'unbatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'unbatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnbatchDataset", input_dataset=input_dataset,
                          output_types=output_types,
                          output_shapes=output_shapes, metadata=metadata,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnbatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnbatchDataset = tf_export("raw_ops.UnbatchDataset")(_ops.to_raw_op(unbatch_dataset))


def unbatch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'unbatch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'unbatch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"UnbatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnbatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uncompress_element(compressed: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None):
  r"""Uncompresses a compressed dataset element.

  Args:
    compressed: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UncompressElement", name, compressed, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uncompress_element_eager_fallback(
          compressed, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'uncompress_element' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'uncompress_element' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UncompressElement", compressed=compressed, output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UncompressElement", _inputs_flat, _attrs, _result)
  return _result

UncompressElement = tf_export("raw_ops.UncompressElement")(_ops.to_raw_op(uncompress_element))


def uncompress_element_eager_fallback(compressed: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'uncompress_element' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'uncompress_element' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  compressed = _ops.convert_to_tensor(compressed, _dtypes.variant)
  _inputs_flat = [compressed]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"UncompressElement", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UncompressElement", _inputs_flat, _attrs, _result)
  return _result


def unique_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that contains the unique elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniqueDataset", name, input_dataset, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unique_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'unique_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'unique_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniqueDataset", input_dataset=input_dataset,
                         output_types=output_types,
                         output_shapes=output_shapes, metadata=metadata,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniqueDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniqueDataset = tf_export("raw_ops.UniqueDataset")(_ops.to_raw_op(unique_dataset))


def unique_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'unique_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'unique_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"UniqueDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniqueDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def weighted_flat_map_dataset(input_datasets: Annotated[List[Any], _atypes.Variant], weights: Annotated[List[Any], _atypes.Float64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_datasets: A list of at least 2 `Tensor` objects with type `variant`.
    weights: A list of at least 2 `Tensor` objects with type `float64`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WeightedFlatMapDataset", name, input_datasets, weights,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return weighted_flat_map_dataset_eager_fallback(
          input_datasets, weights, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % weights)
  _attr_M = len(weights)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WeightedFlatMapDataset", input_datasets=input_datasets,
                                  weights=weights, output_types=output_types,
                                  output_shapes=output_shapes,
                                  metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "M", _op._get_attr_int("M"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WeightedFlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WeightedFlatMapDataset = tf_export("raw_ops.WeightedFlatMapDataset")(_ops.to_raw_op(weighted_flat_map_dataset))


def weighted_flat_map_dataset_eager_fallback(input_datasets: Annotated[List[Any], _atypes.Variant], weights: Annotated[List[Any], _atypes.Float64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  if not isinstance(weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'weights' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % weights)
  _attr_M = len(weights)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'weighted_flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_datasets = _ops.convert_n_to_tensor(input_datasets, _dtypes.variant)
  weights = _ops.convert_n_to_tensor(weights, _dtypes.float64)
  _inputs_flat = list(input_datasets) + list(weights)
  _attrs = ("N", _attr_N, "M", _attr_M, "output_types", output_types,
  "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"WeightedFlatMapDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WeightedFlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

