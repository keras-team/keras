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
_ConvertToCooTensorOutput = collections.namedtuple(
    "ConvertToCooTensor",
    ["row_ids", "col_ids", "gains"])


def convert_to_coo_tensor(indices_or_row_splits: Annotated[Any, _atypes.Int32], values: Annotated[Any, _atypes.Int32], weights: Annotated[Any, _atypes.Float32], sample_count: int, combiner: str, name=None):
  r"""TODO: add doc.

  Args:
    indices_or_row_splits: A `Tensor` of type `int32`.
    values: A `Tensor` of type `int32`.
    weights: A `Tensor` of type `float32`.
    sample_count: An `int` that is `>= 1`.
    combiner: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_ids, col_ids, gains).

    row_ids: A `Tensor` of type `int32`.
    col_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConvertToCooTensor", name, indices_or_row_splits, values,
        weights, "sample_count", sample_count, "combiner", combiner)
      _result = _ConvertToCooTensorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return convert_to_coo_tensor_eager_fallback(
          indices_or_row_splits, values, weights, sample_count=sample_count,
          combiner=combiner, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_count = _execute.make_int(sample_count, "sample_count")
  combiner = _execute.make_str(combiner, "combiner")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConvertToCooTensor", indices_or_row_splits=indices_or_row_splits,
                              values=values, weights=weights,
                              sample_count=sample_count, combiner=combiner,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count", _op._get_attr_int("sample_count"), "combiner",
              _op.get_attr("combiner"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConvertToCooTensor", _inputs_flat, _attrs, _result)
  _result = _ConvertToCooTensorOutput._make(_result)
  return _result

ConvertToCooTensor = tf_export("raw_ops.ConvertToCooTensor")(_ops.to_raw_op(convert_to_coo_tensor))


def convert_to_coo_tensor_eager_fallback(indices_or_row_splits: Annotated[Any, _atypes.Int32], values: Annotated[Any, _atypes.Int32], weights: Annotated[Any, _atypes.Float32], sample_count: int, combiner: str, name, ctx):
  sample_count = _execute.make_int(sample_count, "sample_count")
  combiner = _execute.make_str(combiner, "combiner")
  indices_or_row_splits = _ops.convert_to_tensor(indices_or_row_splits, _dtypes.int32)
  values = _ops.convert_to_tensor(values, _dtypes.int32)
  weights = _ops.convert_to_tensor(weights, _dtypes.float32)
  _inputs_flat = [indices_or_row_splits, values, weights]
  _attrs = ("sample_count", sample_count, "combiner", combiner)
  _result = _execute.execute(b"ConvertToCooTensor", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConvertToCooTensor", _inputs_flat, _attrs, _result)
  _result = _ConvertToCooTensorOutput._make(_result)
  return _result

_ConvertToListOfSparseCoreCooTensorsOutput = collections.namedtuple(
    "ConvertToListOfSparseCoreCooTensors",
    ["row_ids_list", "col_ids_list", "gains_list"])


def convert_to_list_of_sparse_core_coo_tensors(indices_or_row_splits: Annotated[Any, _atypes.Int32], values: Annotated[Any, _atypes.Int32], weights: Annotated[Any, _atypes.Float32], sample_count: int, num_sc_per_chip: int, row_offset: int, col_offset: int, col_shift: int, num_sc_shards: int, stacked_table_sample_count: int, combiner: str, name=None):
  r"""TODO: add doc.

  Args:
    indices_or_row_splits: A `Tensor` of type `int32`.
    values: A `Tensor` of type `int32`.
    weights: A `Tensor` of type `float32`.
    sample_count: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    row_offset: An `int` that is `>= 0`.
    col_offset: An `int` that is `>= 0`.
    col_shift: An `int` that is `>= 0`.
    num_sc_shards: An `int` that is `>= 1`.
    stacked_table_sample_count: An `int` that is `>= 1`.
    combiner: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_ids_list, col_ids_list, gains_list).

    row_ids_list: A list of `num_sc_per_chip` `Tensor` objects with type `int32`.
    col_ids_list: A list of `num_sc_per_chip` `Tensor` objects with type `int32`.
    gains_list: A list of `num_sc_per_chip` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConvertToListOfSparseCoreCooTensors", name,
        indices_or_row_splits, values, weights, "sample_count", sample_count,
        "num_sc_per_chip", num_sc_per_chip, "row_offset", row_offset,
        "col_offset", col_offset, "col_shift", col_shift, "num_sc_shards",
        num_sc_shards, "stacked_table_sample_count",
        stacked_table_sample_count, "combiner", combiner)
      _result = _ConvertToListOfSparseCoreCooTensorsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return convert_to_list_of_sparse_core_coo_tensors_eager_fallback(
          indices_or_row_splits, values, weights, sample_count=sample_count,
          num_sc_per_chip=num_sc_per_chip, row_offset=row_offset,
          col_offset=col_offset, col_shift=col_shift,
          num_sc_shards=num_sc_shards,
          stacked_table_sample_count=stacked_table_sample_count,
          combiner=combiner, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  row_offset = _execute.make_int(row_offset, "row_offset")
  col_offset = _execute.make_int(col_offset, "col_offset")
  col_shift = _execute.make_int(col_shift, "col_shift")
  num_sc_shards = _execute.make_int(num_sc_shards, "num_sc_shards")
  stacked_table_sample_count = _execute.make_int(stacked_table_sample_count, "stacked_table_sample_count")
  combiner = _execute.make_str(combiner, "combiner")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConvertToListOfSparseCoreCooTensors", indices_or_row_splits=indices_or_row_splits,
                                               values=values, weights=weights,
                                               sample_count=sample_count,
                                               num_sc_per_chip=num_sc_per_chip,
                                               row_offset=row_offset,
                                               col_offset=col_offset,
                                               col_shift=col_shift,
                                               num_sc_shards=num_sc_shards,
                                               stacked_table_sample_count=stacked_table_sample_count,
                                               combiner=combiner, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count", _op._get_attr_int("sample_count"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "row_offset", _op._get_attr_int("row_offset"), "col_offset",
              _op._get_attr_int("col_offset"), "col_shift",
              _op._get_attr_int("col_shift"), "num_sc_shards",
              _op._get_attr_int("num_sc_shards"),
              "stacked_table_sample_count",
              _op._get_attr_int("stacked_table_sample_count"), "combiner",
              _op.get_attr("combiner"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConvertToListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sc_per_chip]] + _result[num_sc_per_chip:]
  _result = _result[:1] + [_result[1:1 + num_sc_per_chip]] + _result[1 + num_sc_per_chip:]
  _result = _result[:2] + [_result[2:]]
  _result = _ConvertToListOfSparseCoreCooTensorsOutput._make(_result)
  return _result

ConvertToListOfSparseCoreCooTensors = tf_export("raw_ops.ConvertToListOfSparseCoreCooTensors")(_ops.to_raw_op(convert_to_list_of_sparse_core_coo_tensors))


def convert_to_list_of_sparse_core_coo_tensors_eager_fallback(indices_or_row_splits: Annotated[Any, _atypes.Int32], values: Annotated[Any, _atypes.Int32], weights: Annotated[Any, _atypes.Float32], sample_count: int, num_sc_per_chip: int, row_offset: int, col_offset: int, col_shift: int, num_sc_shards: int, stacked_table_sample_count: int, combiner: str, name, ctx):
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  row_offset = _execute.make_int(row_offset, "row_offset")
  col_offset = _execute.make_int(col_offset, "col_offset")
  col_shift = _execute.make_int(col_shift, "col_shift")
  num_sc_shards = _execute.make_int(num_sc_shards, "num_sc_shards")
  stacked_table_sample_count = _execute.make_int(stacked_table_sample_count, "stacked_table_sample_count")
  combiner = _execute.make_str(combiner, "combiner")
  indices_or_row_splits = _ops.convert_to_tensor(indices_or_row_splits, _dtypes.int32)
  values = _ops.convert_to_tensor(values, _dtypes.int32)
  weights = _ops.convert_to_tensor(weights, _dtypes.float32)
  _inputs_flat = [indices_or_row_splits, values, weights]
  _attrs = ("sample_count", sample_count, "num_sc_per_chip", num_sc_per_chip,
  "row_offset", row_offset, "col_offset", col_offset, "col_shift", col_shift,
  "num_sc_shards", num_sc_shards, "stacked_table_sample_count",
  stacked_table_sample_count, "combiner", combiner)
  _result = _execute.execute(b"ConvertToListOfSparseCoreCooTensors",
                             num_sc_per_chip + num_sc_per_chip +
                             num_sc_per_chip, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConvertToListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sc_per_chip]] + _result[num_sc_per_chip:]
  _result = _result[:1] + [_result[1:1 + num_sc_per_chip]] + _result[1 + num_sc_per_chip:]
  _result = _result[:2] + [_result[2:]]
  _result = _ConvertToListOfSparseCoreCooTensorsOutput._make(_result)
  return _result

_ConvertToSparseCoreCsrWrappedCooTensorOutput = collections.namedtuple(
    "ConvertToSparseCoreCsrWrappedCooTensor",
    ["row_pointers", "sorted_sample_ids", "sorted_token_ids", "sorted_gains", "row_pointers_unpadded_size", "ids_unpadded_size", "num_minibatches_per_sc"])


def convert_to_sparse_core_csr_wrapped_coo_tensor(sorted_row_ids_list: Annotated[List[Any], _atypes.Int32], sorted_col_ids_list: Annotated[List[Any], _atypes.Int32], sorted_gains_list: Annotated[List[Any], _atypes.Float32], id_counts_list: Annotated[List[Any], _atypes.Int32], splits: Annotated[Any, _atypes.Int64], sample_count_per_sc: int, num_replica: int, max_minibatches_per_sc: int, max_ids_per_chip_per_sample: int, table_vocab_size: int, feature_width: int, table_name: str, allow_id_dropping: bool, name=None):
  r"""TODO: add doc.

  Args:
    sorted_row_ids_list: A list of at least 1 `Tensor` objects with type `int32`.
    sorted_col_ids_list: A list with the same length as `sorted_row_ids_list` of `Tensor` objects with type `int32`.
    sorted_gains_list: A list with the same length as `sorted_row_ids_list` of `Tensor` objects with type `float32`.
    id_counts_list: A list with the same length as `sorted_row_ids_list` of `Tensor` objects with type `int32`.
    splits: A `Tensor` of type `int64`.
    sample_count_per_sc: An `int` that is `>= 1`.
    num_replica: An `int` that is `>= 1`.
    max_minibatches_per_sc: An `int` that is `>= 1`.
    max_ids_per_chip_per_sample: An `int` that is `>= 1`.
    table_vocab_size: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    table_name: A `string`.
    allow_id_dropping: A `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, row_pointers_unpadded_size, ids_unpadded_size, num_minibatches_per_sc).

    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    row_pointers_unpadded_size: A `Tensor` of type `int32`.
    ids_unpadded_size: A `Tensor` of type `int32`.
    num_minibatches_per_sc: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConvertToSparseCoreCsrWrappedCooTensor", name,
        sorted_row_ids_list, sorted_col_ids_list, sorted_gains_list,
        id_counts_list, splits, "sample_count_per_sc", sample_count_per_sc,
        "num_replica", num_replica, "max_minibatches_per_sc",
        max_minibatches_per_sc, "max_ids_per_chip_per_sample",
        max_ids_per_chip_per_sample, "table_vocab_size", table_vocab_size,
        "feature_width", feature_width, "table_name", table_name,
        "allow_id_dropping", allow_id_dropping)
      _result = _ConvertToSparseCoreCsrWrappedCooTensorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return convert_to_sparse_core_csr_wrapped_coo_tensor_eager_fallback(
          sorted_row_ids_list, sorted_col_ids_list, sorted_gains_list,
          id_counts_list, splits, sample_count_per_sc=sample_count_per_sc,
          num_replica=num_replica,
          max_minibatches_per_sc=max_minibatches_per_sc,
          max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
          table_vocab_size=table_vocab_size, feature_width=feature_width,
          table_name=table_name, allow_id_dropping=allow_id_dropping,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sorted_row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_row_ids_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_row_ids_list)
  _attr_num_sc_per_chip = len(sorted_row_ids_list)
  if not isinstance(sorted_col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_col_ids_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_col_ids_list)
  if len(sorted_col_ids_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'sorted_col_ids_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(sorted_col_ids_list), _attr_num_sc_per_chip))
  if not isinstance(sorted_gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_gains_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_gains_list)
  if len(sorted_gains_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'sorted_gains_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(sorted_gains_list), _attr_num_sc_per_chip))
  if not isinstance(id_counts_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'id_counts_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % id_counts_list)
  if len(id_counts_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'id_counts_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(id_counts_list), _attr_num_sc_per_chip))
  sample_count_per_sc = _execute.make_int(sample_count_per_sc, "sample_count_per_sc")
  num_replica = _execute.make_int(num_replica, "num_replica")
  max_minibatches_per_sc = _execute.make_int(max_minibatches_per_sc, "max_minibatches_per_sc")
  max_ids_per_chip_per_sample = _execute.make_int(max_ids_per_chip_per_sample, "max_ids_per_chip_per_sample")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  table_name = _execute.make_str(table_name, "table_name")
  allow_id_dropping = _execute.make_bool(allow_id_dropping, "allow_id_dropping")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConvertToSparseCoreCsrWrappedCooTensor", sorted_row_ids_list=sorted_row_ids_list,
                                                  sorted_col_ids_list=sorted_col_ids_list,
                                                  sorted_gains_list=sorted_gains_list,
                                                  id_counts_list=id_counts_list,
                                                  splits=splits,
                                                  sample_count_per_sc=sample_count_per_sc,
                                                  num_replica=num_replica,
                                                  max_minibatches_per_sc=max_minibatches_per_sc,
                                                  max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
                                                  table_vocab_size=table_vocab_size,
                                                  feature_width=feature_width,
                                                  table_name=table_name,
                                                  allow_id_dropping=allow_id_dropping,
                                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count_per_sc", _op._get_attr_int("sample_count_per_sc"),
              "num_replica", _op._get_attr_int("num_replica"),
              "max_minibatches_per_sc",
              _op._get_attr_int("max_minibatches_per_sc"),
              "max_ids_per_chip_per_sample",
              _op._get_attr_int("max_ids_per_chip_per_sample"),
              "table_vocab_size", _op._get_attr_int("table_vocab_size"),
              "feature_width", _op._get_attr_int("feature_width"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "table_name", _op.get_attr("table_name"), "allow_id_dropping",
              _op._get_attr_bool("allow_id_dropping"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConvertToSparseCoreCsrWrappedCooTensor", _inputs_flat, _attrs, _result)
  _result = _ConvertToSparseCoreCsrWrappedCooTensorOutput._make(_result)
  return _result

ConvertToSparseCoreCsrWrappedCooTensor = tf_export("raw_ops.ConvertToSparseCoreCsrWrappedCooTensor")(_ops.to_raw_op(convert_to_sparse_core_csr_wrapped_coo_tensor))


def convert_to_sparse_core_csr_wrapped_coo_tensor_eager_fallback(sorted_row_ids_list: Annotated[List[Any], _atypes.Int32], sorted_col_ids_list: Annotated[List[Any], _atypes.Int32], sorted_gains_list: Annotated[List[Any], _atypes.Float32], id_counts_list: Annotated[List[Any], _atypes.Int32], splits: Annotated[Any, _atypes.Int64], sample_count_per_sc: int, num_replica: int, max_minibatches_per_sc: int, max_ids_per_chip_per_sample: int, table_vocab_size: int, feature_width: int, table_name: str, allow_id_dropping: bool, name, ctx):
  if not isinstance(sorted_row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_row_ids_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_row_ids_list)
  _attr_num_sc_per_chip = len(sorted_row_ids_list)
  if not isinstance(sorted_col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_col_ids_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_col_ids_list)
  if len(sorted_col_ids_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'sorted_col_ids_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(sorted_col_ids_list), _attr_num_sc_per_chip))
  if not isinstance(sorted_gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sorted_gains_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % sorted_gains_list)
  if len(sorted_gains_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'sorted_gains_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(sorted_gains_list), _attr_num_sc_per_chip))
  if not isinstance(id_counts_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'id_counts_list' argument to "
        "'convert_to_sparse_core_csr_wrapped_coo_tensor' Op, not %r." % id_counts_list)
  if len(id_counts_list) != _attr_num_sc_per_chip:
    raise ValueError(
        "List argument 'id_counts_list' to 'convert_to_sparse_core_csr_wrapped_coo_tensor' Op with length %d "
        "must match length %d of argument 'sorted_row_ids_list'." %
        (len(id_counts_list), _attr_num_sc_per_chip))
  sample_count_per_sc = _execute.make_int(sample_count_per_sc, "sample_count_per_sc")
  num_replica = _execute.make_int(num_replica, "num_replica")
  max_minibatches_per_sc = _execute.make_int(max_minibatches_per_sc, "max_minibatches_per_sc")
  max_ids_per_chip_per_sample = _execute.make_int(max_ids_per_chip_per_sample, "max_ids_per_chip_per_sample")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  table_name = _execute.make_str(table_name, "table_name")
  allow_id_dropping = _execute.make_bool(allow_id_dropping, "allow_id_dropping")
  sorted_row_ids_list = _ops.convert_n_to_tensor(sorted_row_ids_list, _dtypes.int32)
  sorted_col_ids_list = _ops.convert_n_to_tensor(sorted_col_ids_list, _dtypes.int32)
  sorted_gains_list = _ops.convert_n_to_tensor(sorted_gains_list, _dtypes.float32)
  id_counts_list = _ops.convert_n_to_tensor(id_counts_list, _dtypes.int32)
  splits = _ops.convert_to_tensor(splits, _dtypes.int64)
  _inputs_flat = list(sorted_row_ids_list) + list(sorted_col_ids_list) + list(sorted_gains_list) + list(id_counts_list) + [splits]
  _attrs = ("sample_count_per_sc", sample_count_per_sc, "num_replica",
  num_replica, "max_minibatches_per_sc", max_minibatches_per_sc,
  "max_ids_per_chip_per_sample", max_ids_per_chip_per_sample,
  "table_vocab_size", table_vocab_size, "feature_width", feature_width,
  "num_sc_per_chip", _attr_num_sc_per_chip, "table_name", table_name,
  "allow_id_dropping", allow_id_dropping)
  _result = _execute.execute(b"ConvertToSparseCoreCsrWrappedCooTensor", 7,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConvertToSparseCoreCsrWrappedCooTensor", _inputs_flat, _attrs, _result)
  _result = _ConvertToSparseCoreCsrWrappedCooTensorOutput._make(_result)
  return _result

_GetMinibatchSplitsWithPhysicalReplicaOutput = collections.namedtuple(
    "GetMinibatchSplitsWithPhysicalReplica",
    ["sorted_row_ids", "sorted_col_ids", "sorted_gains", "splits", "id_counts", "max_ids", "max_uniques"])


def get_minibatch_splits_with_physical_replica(program_key: Annotated[Any, _atypes.String], row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.Int32], gains: Annotated[Any, _atypes.Float32], sample_count: int, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_splits: str, name=None):
  r"""TODO: add doc.

  Args:
    program_key: A `Tensor` of type `string`.
    row_ids: A `Tensor` of type `int32`.
    col_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    sample_count: An `int` that is `>= 1`.
    num_replica: An `int` that is `>= 1`.
    table_vocab_size: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    table_name: A `string`.
    mini_batch_splits: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sorted_row_ids, sorted_col_ids, sorted_gains, splits, id_counts, max_ids, max_uniques).

    sorted_row_ids: A `Tensor` of type `int32`.
    sorted_col_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    splits: A `Tensor` of type `int64`.
    id_counts: A `Tensor` of type `int32`.
    max_ids: A `Tensor` of type `int32`.
    max_uniques: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetMinibatchSplitsWithPhysicalReplica", name, program_key,
        row_ids, col_ids, gains, "sample_count", sample_count, "num_replica",
        num_replica, "table_vocab_size", table_vocab_size, "feature_width",
        feature_width, "num_sc_per_chip", num_sc_per_chip, "table_name",
        table_name, "mini_batch_splits", mini_batch_splits)
      _result = _GetMinibatchSplitsWithPhysicalReplicaOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_minibatch_splits_with_physical_replica_eager_fallback(
          program_key, row_ids, col_ids, gains, sample_count=sample_count,
          num_replica=num_replica, table_vocab_size=table_vocab_size,
          feature_width=feature_width, num_sc_per_chip=num_sc_per_chip,
          table_name=table_name, mini_batch_splits=mini_batch_splits,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_splits = _execute.make_str(mini_batch_splits, "mini_batch_splits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetMinibatchSplitsWithPhysicalReplica", program_key=program_key,
                                                 row_ids=row_ids,
                                                 col_ids=col_ids, gains=gains,
                                                 sample_count=sample_count,
                                                 num_replica=num_replica,
                                                 table_vocab_size=table_vocab_size,
                                                 feature_width=feature_width,
                                                 num_sc_per_chip=num_sc_per_chip,
                                                 table_name=table_name,
                                                 mini_batch_splits=mini_batch_splits,
                                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count", _op._get_attr_int("sample_count"),
              "num_replica", _op._get_attr_int("num_replica"),
              "table_vocab_size", _op._get_attr_int("table_vocab_size"),
              "feature_width", _op._get_attr_int("feature_width"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "table_name", _op.get_attr("table_name"), "mini_batch_splits",
              _op.get_attr("mini_batch_splits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetMinibatchSplitsWithPhysicalReplica", _inputs_flat, _attrs, _result)
  _result = _GetMinibatchSplitsWithPhysicalReplicaOutput._make(_result)
  return _result

GetMinibatchSplitsWithPhysicalReplica = tf_export("raw_ops.GetMinibatchSplitsWithPhysicalReplica")(_ops.to_raw_op(get_minibatch_splits_with_physical_replica))


def get_minibatch_splits_with_physical_replica_eager_fallback(program_key: Annotated[Any, _atypes.String], row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.Int32], gains: Annotated[Any, _atypes.Float32], sample_count: int, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_splits: str, name, ctx):
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_splits = _execute.make_str(mini_batch_splits, "mini_batch_splits")
  program_key = _ops.convert_to_tensor(program_key, _dtypes.string)
  row_ids = _ops.convert_to_tensor(row_ids, _dtypes.int32)
  col_ids = _ops.convert_to_tensor(col_ids, _dtypes.int32)
  gains = _ops.convert_to_tensor(gains, _dtypes.float32)
  _inputs_flat = [program_key, row_ids, col_ids, gains]
  _attrs = ("sample_count", sample_count, "num_replica", num_replica,
  "table_vocab_size", table_vocab_size, "feature_width", feature_width,
  "num_sc_per_chip", num_sc_per_chip, "table_name", table_name,
  "mini_batch_splits", mini_batch_splits)
  _result = _execute.execute(b"GetMinibatchSplitsWithPhysicalReplica", 7,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetMinibatchSplitsWithPhysicalReplica", _inputs_flat, _attrs, _result)
  _result = _GetMinibatchSplitsWithPhysicalReplicaOutput._make(_result)
  return _result

_GetMinibatchesInCsrWithPhysicalReplicaOutput = collections.namedtuple(
    "GetMinibatchesInCsrWithPhysicalReplica",
    ["row_pointers", "sorted_sample_ids", "sorted_token_ids", "sorted_gains", "row_pointers_unpadded_size", "ids_unpadded_size", "num_minibatches_per_physical_sparse_core"])


def get_minibatches_in_csr_with_physical_replica(program_key: Annotated[Any, _atypes.String], row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.Int32], gains: Annotated[Any, _atypes.Float32], splits: Annotated[Any, _atypes.Int64], id_counts: Annotated[Any, _atypes.Int32], sample_count: int, num_replica: int, max_minibatches_per_sc: int, max_ids_per_chip_per_sample: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_in_csr: str, name=None):
  r"""TODO: add doc.

  Args:
    program_key: A `Tensor` of type `string`.
    row_ids: A `Tensor` of type `int32`.
    col_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    splits: A `Tensor` of type `int64`.
    id_counts: A `Tensor` of type `int32`.
    sample_count: An `int` that is `>= 1`.
    num_replica: An `int` that is `>= 1`.
    max_minibatches_per_sc: An `int` that is `>= 1`.
    max_ids_per_chip_per_sample: An `int` that is `>= 1`.
    table_vocab_size: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    table_name: A `string`.
    mini_batch_in_csr: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, row_pointers_unpadded_size, ids_unpadded_size, num_minibatches_per_physical_sparse_core).

    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    row_pointers_unpadded_size: A `Tensor` of type `int32`.
    ids_unpadded_size: A `Tensor` of type `int32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetMinibatchesInCsrWithPhysicalReplica", name, program_key,
        row_ids, col_ids, gains, splits, id_counts, "sample_count",
        sample_count, "num_replica", num_replica, "max_minibatches_per_sc",
        max_minibatches_per_sc, "max_ids_per_chip_per_sample",
        max_ids_per_chip_per_sample, "table_vocab_size", table_vocab_size,
        "feature_width", feature_width, "num_sc_per_chip", num_sc_per_chip,
        "table_name", table_name, "mini_batch_in_csr", mini_batch_in_csr)
      _result = _GetMinibatchesInCsrWithPhysicalReplicaOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_minibatches_in_csr_with_physical_replica_eager_fallback(
          program_key, row_ids, col_ids, gains, splits, id_counts,
          sample_count=sample_count, num_replica=num_replica,
          max_minibatches_per_sc=max_minibatches_per_sc,
          max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
          table_vocab_size=table_vocab_size, feature_width=feature_width,
          num_sc_per_chip=num_sc_per_chip, table_name=table_name,
          mini_batch_in_csr=mini_batch_in_csr, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  max_minibatches_per_sc = _execute.make_int(max_minibatches_per_sc, "max_minibatches_per_sc")
  max_ids_per_chip_per_sample = _execute.make_int(max_ids_per_chip_per_sample, "max_ids_per_chip_per_sample")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_in_csr = _execute.make_str(mini_batch_in_csr, "mini_batch_in_csr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetMinibatchesInCsrWithPhysicalReplica", program_key=program_key,
                                                  row_ids=row_ids,
                                                  col_ids=col_ids,
                                                  gains=gains, splits=splits,
                                                  id_counts=id_counts,
                                                  sample_count=sample_count,
                                                  num_replica=num_replica,
                                                  max_minibatches_per_sc=max_minibatches_per_sc,
                                                  max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
                                                  table_vocab_size=table_vocab_size,
                                                  feature_width=feature_width,
                                                  num_sc_per_chip=num_sc_per_chip,
                                                  table_name=table_name,
                                                  mini_batch_in_csr=mini_batch_in_csr,
                                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count", _op._get_attr_int("sample_count"),
              "num_replica", _op._get_attr_int("num_replica"),
              "max_minibatches_per_sc",
              _op._get_attr_int("max_minibatches_per_sc"),
              "max_ids_per_chip_per_sample",
              _op._get_attr_int("max_ids_per_chip_per_sample"),
              "table_vocab_size", _op._get_attr_int("table_vocab_size"),
              "feature_width", _op._get_attr_int("feature_width"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "table_name", _op.get_attr("table_name"), "mini_batch_in_csr",
              _op.get_attr("mini_batch_in_csr"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetMinibatchesInCsrWithPhysicalReplica", _inputs_flat, _attrs, _result)
  _result = _GetMinibatchesInCsrWithPhysicalReplicaOutput._make(_result)
  return _result

GetMinibatchesInCsrWithPhysicalReplica = tf_export("raw_ops.GetMinibatchesInCsrWithPhysicalReplica")(_ops.to_raw_op(get_minibatches_in_csr_with_physical_replica))


def get_minibatches_in_csr_with_physical_replica_eager_fallback(program_key: Annotated[Any, _atypes.String], row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.Int32], gains: Annotated[Any, _atypes.Float32], splits: Annotated[Any, _atypes.Int64], id_counts: Annotated[Any, _atypes.Int32], sample_count: int, num_replica: int, max_minibatches_per_sc: int, max_ids_per_chip_per_sample: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_in_csr: str, name, ctx):
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  max_minibatches_per_sc = _execute.make_int(max_minibatches_per_sc, "max_minibatches_per_sc")
  max_ids_per_chip_per_sample = _execute.make_int(max_ids_per_chip_per_sample, "max_ids_per_chip_per_sample")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_in_csr = _execute.make_str(mini_batch_in_csr, "mini_batch_in_csr")
  program_key = _ops.convert_to_tensor(program_key, _dtypes.string)
  row_ids = _ops.convert_to_tensor(row_ids, _dtypes.int32)
  col_ids = _ops.convert_to_tensor(col_ids, _dtypes.int32)
  gains = _ops.convert_to_tensor(gains, _dtypes.float32)
  splits = _ops.convert_to_tensor(splits, _dtypes.int64)
  id_counts = _ops.convert_to_tensor(id_counts, _dtypes.int32)
  _inputs_flat = [program_key, row_ids, col_ids, gains, splits, id_counts]
  _attrs = ("sample_count", sample_count, "num_replica", num_replica,
  "max_minibatches_per_sc", max_minibatches_per_sc,
  "max_ids_per_chip_per_sample", max_ids_per_chip_per_sample,
  "table_vocab_size", table_vocab_size, "feature_width", feature_width,
  "num_sc_per_chip", num_sc_per_chip, "table_name", table_name,
  "mini_batch_in_csr", mini_batch_in_csr)
  _result = _execute.execute(b"GetMinibatchesInCsrWithPhysicalReplica", 7,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetMinibatchesInCsrWithPhysicalReplica", _inputs_flat, _attrs, _result)
  _result = _GetMinibatchesInCsrWithPhysicalReplicaOutput._make(_result)
  return _result

_GetStatsFromListOfSparseCoreCooTensorsOutput = collections.namedtuple(
    "GetStatsFromListOfSparseCoreCooTensors",
    ["max_ids_per_sparse_core", "max_unique_ids_per_sparse_core"])


def get_stats_from_list_of_sparse_core_coo_tensors(row_ids_list: Annotated[List[Any], _atypes.Int32], col_ids_list: Annotated[List[Any], _atypes.Int32], gains_list: Annotated[List[Any], _atypes.Float32], sample_count_list, col_offset_list, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, name=None):
  r"""TODO: add doc.

  Args:
    row_ids_list: A list of at least 1 `Tensor` objects with type `int32`.
    col_ids_list: A list with the same length as `row_ids_list` of `Tensor` objects with type `int32`.
    gains_list: A list with the same length as `row_ids_list` of `Tensor` objects with type `float32`.
    sample_count_list: A list of `ints`.
    col_offset_list: A list of `ints`.
    num_replica: An `int` that is `>= 1`.
    table_vocab_size: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    table_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (max_ids_per_sparse_core, max_unique_ids_per_sparse_core).

    max_ids_per_sparse_core: A `Tensor` of type `int32`.
    max_unique_ids_per_sparse_core: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetStatsFromListOfSparseCoreCooTensors", name, row_ids_list,
        col_ids_list, gains_list, "sample_count_list", sample_count_list,
        "col_offset_list", col_offset_list, "num_replica", num_replica,
        "table_vocab_size", table_vocab_size, "feature_width", feature_width,
        "num_sc_per_chip", num_sc_per_chip, "table_name", table_name)
      _result = _GetStatsFromListOfSparseCoreCooTensorsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_stats_from_list_of_sparse_core_coo_tensors_eager_fallback(
          row_ids_list, col_ids_list, gains_list,
          sample_count_list=sample_count_list,
          col_offset_list=col_offset_list, num_replica=num_replica,
          table_vocab_size=table_vocab_size, feature_width=feature_width,
          num_sc_per_chip=num_sc_per_chip, table_name=table_name, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_ids_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % row_ids_list)
  _attr_N = len(row_ids_list)
  if not isinstance(col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_ids_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % col_ids_list)
  if len(col_ids_list) != _attr_N:
    raise ValueError(
        "List argument 'col_ids_list' to 'get_stats_from_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(col_ids_list), _attr_N))
  if not isinstance(gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % gains_list)
  if len(gains_list) != _attr_N:
    raise ValueError(
        "List argument 'gains_list' to 'get_stats_from_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(gains_list), _attr_N))
  if not isinstance(sample_count_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_count_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % sample_count_list)
  sample_count_list = [_execute.make_int(_i, "sample_count_list") for _i in sample_count_list]
  if not isinstance(col_offset_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_offset_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % col_offset_list)
  col_offset_list = [_execute.make_int(_i, "col_offset_list") for _i in col_offset_list]
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetStatsFromListOfSparseCoreCooTensors", row_ids_list=row_ids_list,
                                                  col_ids_list=col_ids_list,
                                                  gains_list=gains_list,
                                                  sample_count_list=sample_count_list,
                                                  col_offset_list=col_offset_list,
                                                  num_replica=num_replica,
                                                  table_vocab_size=table_vocab_size,
                                                  feature_width=feature_width,
                                                  num_sc_per_chip=num_sc_per_chip,
                                                  table_name=table_name,
                                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count_list", _op.get_attr("sample_count_list"),
              "col_offset_list", _op.get_attr("col_offset_list"),
              "num_replica", _op._get_attr_int("num_replica"),
              "table_vocab_size", _op._get_attr_int("table_vocab_size"),
              "feature_width", _op._get_attr_int("feature_width"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "table_name", _op.get_attr("table_name"), "N",
              _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetStatsFromListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = _GetStatsFromListOfSparseCoreCooTensorsOutput._make(_result)
  return _result

GetStatsFromListOfSparseCoreCooTensors = tf_export("raw_ops.GetStatsFromListOfSparseCoreCooTensors")(_ops.to_raw_op(get_stats_from_list_of_sparse_core_coo_tensors))


def get_stats_from_list_of_sparse_core_coo_tensors_eager_fallback(row_ids_list: Annotated[List[Any], _atypes.Int32], col_ids_list: Annotated[List[Any], _atypes.Int32], gains_list: Annotated[List[Any], _atypes.Float32], sample_count_list, col_offset_list, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, table_name: str, name, ctx):
  if not isinstance(row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_ids_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % row_ids_list)
  _attr_N = len(row_ids_list)
  if not isinstance(col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_ids_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % col_ids_list)
  if len(col_ids_list) != _attr_N:
    raise ValueError(
        "List argument 'col_ids_list' to 'get_stats_from_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(col_ids_list), _attr_N))
  if not isinstance(gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % gains_list)
  if len(gains_list) != _attr_N:
    raise ValueError(
        "List argument 'gains_list' to 'get_stats_from_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(gains_list), _attr_N))
  if not isinstance(sample_count_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_count_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % sample_count_list)
  sample_count_list = [_execute.make_int(_i, "sample_count_list") for _i in sample_count_list]
  if not isinstance(col_offset_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_offset_list' argument to "
        "'get_stats_from_list_of_sparse_core_coo_tensors' Op, not %r." % col_offset_list)
  col_offset_list = [_execute.make_int(_i, "col_offset_list") for _i in col_offset_list]
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  row_ids_list = _ops.convert_n_to_tensor(row_ids_list, _dtypes.int32)
  col_ids_list = _ops.convert_n_to_tensor(col_ids_list, _dtypes.int32)
  gains_list = _ops.convert_n_to_tensor(gains_list, _dtypes.float32)
  _inputs_flat = list(row_ids_list) + list(col_ids_list) + list(gains_list)
  _attrs = ("sample_count_list", sample_count_list, "col_offset_list",
  col_offset_list, "num_replica", num_replica, "table_vocab_size",
  table_vocab_size, "feature_width", feature_width, "num_sc_per_chip",
  num_sc_per_chip, "table_name", table_name, "N", _attr_N)
  _result = _execute.execute(b"GetStatsFromListOfSparseCoreCooTensors", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetStatsFromListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = _GetStatsFromListOfSparseCoreCooTensorsOutput._make(_result)
  return _result


def global_iter_id(name=None) -> Annotated[Any, _atypes.Int64]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GlobalIterId", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return global_iter_id_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GlobalIterId", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GlobalIterId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GlobalIterId = tf_export("raw_ops.GlobalIterId")(_ops.to_raw_op(global_iter_id))


def global_iter_id_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int64]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"GlobalIterId", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GlobalIterId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SortListOfSparseCoreCooTensorsOutput = collections.namedtuple(
    "SortListOfSparseCoreCooTensors",
    ["sorted_row_ids", "sorted_col_ids", "sorted_gains", "id_counts"])


def sort_list_of_sparse_core_coo_tensors(row_ids_list: Annotated[List[Any], _atypes.Int32], col_ids_list: Annotated[List[Any], _atypes.Int32], gains_list: Annotated[List[Any], _atypes.Float32], sample_count_list, col_offset_list, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, name=None):
  r"""TODO: add doc.

  Args:
    row_ids_list: A list of at least 1 `Tensor` objects with type `int32`.
    col_ids_list: A list with the same length as `row_ids_list` of `Tensor` objects with type `int32`.
    gains_list: A list with the same length as `row_ids_list` of `Tensor` objects with type `float32`.
    sample_count_list: A list of `ints`.
    col_offset_list: A list of `ints`.
    num_replica: An `int` that is `>= 1`.
    table_vocab_size: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sorted_row_ids, sorted_col_ids, sorted_gains, id_counts).

    sorted_row_ids: A `Tensor` of type `int32`.
    sorted_col_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    id_counts: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SortListOfSparseCoreCooTensors", name, row_ids_list,
        col_ids_list, gains_list, "sample_count_list", sample_count_list,
        "col_offset_list", col_offset_list, "num_replica", num_replica,
        "table_vocab_size", table_vocab_size, "feature_width", feature_width,
        "num_sc_per_chip", num_sc_per_chip, "max_ids_per_sparse_core",
        max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
        max_unique_ids_per_sparse_core, "table_name", table_name)
      _result = _SortListOfSparseCoreCooTensorsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sort_list_of_sparse_core_coo_tensors_eager_fallback(
          row_ids_list, col_ids_list, gains_list,
          sample_count_list=sample_count_list,
          col_offset_list=col_offset_list, num_replica=num_replica,
          table_vocab_size=table_vocab_size, feature_width=feature_width,
          num_sc_per_chip=num_sc_per_chip,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_ids_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % row_ids_list)
  _attr_N = len(row_ids_list)
  if not isinstance(col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_ids_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % col_ids_list)
  if len(col_ids_list) != _attr_N:
    raise ValueError(
        "List argument 'col_ids_list' to 'sort_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(col_ids_list), _attr_N))
  if not isinstance(gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % gains_list)
  if len(gains_list) != _attr_N:
    raise ValueError(
        "List argument 'gains_list' to 'sort_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(gains_list), _attr_N))
  if not isinstance(sample_count_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_count_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % sample_count_list)
  sample_count_list = [_execute.make_int(_i, "sample_count_list") for _i in sample_count_list]
  if not isinstance(col_offset_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_offset_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % col_offset_list)
  col_offset_list = [_execute.make_int(_i, "col_offset_list") for _i in col_offset_list]
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SortListOfSparseCoreCooTensors", row_ids_list=row_ids_list,
                                          col_ids_list=col_ids_list,
                                          gains_list=gains_list,
                                          sample_count_list=sample_count_list,
                                          col_offset_list=col_offset_list,
                                          num_replica=num_replica,
                                          table_vocab_size=table_vocab_size,
                                          feature_width=feature_width,
                                          num_sc_per_chip=num_sc_per_chip,
                                          max_ids_per_sparse_core=max_ids_per_sparse_core,
                                          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                          table_name=table_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_count_list", _op.get_attr("sample_count_list"),
              "col_offset_list", _op.get_attr("col_offset_list"),
              "num_replica", _op._get_attr_int("num_replica"),
              "table_vocab_size", _op._get_attr_int("table_vocab_size"),
              "feature_width", _op._get_attr_int("feature_width"),
              "num_sc_per_chip", _op._get_attr_int("num_sc_per_chip"),
              "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"), "N",
              _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SortListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = _SortListOfSparseCoreCooTensorsOutput._make(_result)
  return _result

SortListOfSparseCoreCooTensors = tf_export("raw_ops.SortListOfSparseCoreCooTensors")(_ops.to_raw_op(sort_list_of_sparse_core_coo_tensors))


def sort_list_of_sparse_core_coo_tensors_eager_fallback(row_ids_list: Annotated[List[Any], _atypes.Int32], col_ids_list: Annotated[List[Any], _atypes.Int32], gains_list: Annotated[List[Any], _atypes.Float32], sample_count_list, col_offset_list, num_replica: int, table_vocab_size: int, feature_width: int, num_sc_per_chip: int, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, name, ctx):
  if not isinstance(row_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_ids_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % row_ids_list)
  _attr_N = len(row_ids_list)
  if not isinstance(col_ids_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_ids_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % col_ids_list)
  if len(col_ids_list) != _attr_N:
    raise ValueError(
        "List argument 'col_ids_list' to 'sort_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(col_ids_list), _attr_N))
  if not isinstance(gains_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'gains_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % gains_list)
  if len(gains_list) != _attr_N:
    raise ValueError(
        "List argument 'gains_list' to 'sort_list_of_sparse_core_coo_tensors' Op with length %d "
        "must match length %d of argument 'row_ids_list'." %
        (len(gains_list), _attr_N))
  if not isinstance(sample_count_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_count_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % sample_count_list)
  sample_count_list = [_execute.make_int(_i, "sample_count_list") for _i in sample_count_list]
  if not isinstance(col_offset_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'col_offset_list' argument to "
        "'sort_list_of_sparse_core_coo_tensors' Op, not %r." % col_offset_list)
  col_offset_list = [_execute.make_int(_i, "col_offset_list") for _i in col_offset_list]
  num_replica = _execute.make_int(num_replica, "num_replica")
  table_vocab_size = _execute.make_int(table_vocab_size, "table_vocab_size")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  row_ids_list = _ops.convert_n_to_tensor(row_ids_list, _dtypes.int32)
  col_ids_list = _ops.convert_n_to_tensor(col_ids_list, _dtypes.int32)
  gains_list = _ops.convert_n_to_tensor(gains_list, _dtypes.float32)
  _inputs_flat = list(row_ids_list) + list(col_ids_list) + list(gains_list)
  _attrs = ("sample_count_list", sample_count_list, "col_offset_list",
  col_offset_list, "num_replica", num_replica, "table_vocab_size",
  table_vocab_size, "feature_width", feature_width, "num_sc_per_chip",
  num_sc_per_chip, "max_ids_per_sparse_core", max_ids_per_sparse_core,
  "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
  "table_name", table_name, "N", _attr_N)
  _result = _execute.execute(b"SortListOfSparseCoreCooTensors", 4,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SortListOfSparseCoreCooTensors", _inputs_flat, _attrs, _result)
  _result = _SortListOfSparseCoreCooTensorsOutput._make(_result)
  return _result


def store_minibatch_statistics_in_fdo(program_key: Annotated[Any, _atypes.String], max_ids: Annotated[Any, _atypes.Int32], max_uniques: Annotated[Any, _atypes.Int32], sample_count: int, num_replica: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_splits: str, name=None):
  r"""TODO: add doc.

  Args:
    program_key: A `Tensor` of type `string`.
    max_ids: A `Tensor` of type `int32`.
    max_uniques: A `Tensor` of type `int32`.
    sample_count: An `int` that is `>= 1`.
    num_replica: An `int` that is `>= 1`.
    feature_width: An `int` that is `>= 1`.
    num_sc_per_chip: An `int` that is `>= 1`.
    table_name: A `string`.
    mini_batch_splits: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StoreMinibatchStatisticsInFdo", name, program_key, max_ids,
        max_uniques, "sample_count", sample_count, "num_replica", num_replica,
        "feature_width", feature_width, "num_sc_per_chip", num_sc_per_chip,
        "table_name", table_name, "mini_batch_splits", mini_batch_splits)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return store_minibatch_statistics_in_fdo_eager_fallback(
          program_key, max_ids, max_uniques, sample_count=sample_count,
          num_replica=num_replica, feature_width=feature_width,
          num_sc_per_chip=num_sc_per_chip, table_name=table_name,
          mini_batch_splits=mini_batch_splits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_splits = _execute.make_str(mini_batch_splits, "mini_batch_splits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StoreMinibatchStatisticsInFdo", program_key=program_key,
                                         max_ids=max_ids,
                                         max_uniques=max_uniques,
                                         sample_count=sample_count,
                                         num_replica=num_replica,
                                         feature_width=feature_width,
                                         num_sc_per_chip=num_sc_per_chip,
                                         table_name=table_name,
                                         mini_batch_splits=mini_batch_splits,
                                         name=name)
  return _op
StoreMinibatchStatisticsInFdo = tf_export("raw_ops.StoreMinibatchStatisticsInFdo")(_ops.to_raw_op(store_minibatch_statistics_in_fdo))


def store_minibatch_statistics_in_fdo_eager_fallback(program_key: Annotated[Any, _atypes.String], max_ids: Annotated[Any, _atypes.Int32], max_uniques: Annotated[Any, _atypes.Int32], sample_count: int, num_replica: int, feature_width: int, num_sc_per_chip: int, table_name: str, mini_batch_splits: str, name, ctx):
  sample_count = _execute.make_int(sample_count, "sample_count")
  num_replica = _execute.make_int(num_replica, "num_replica")
  feature_width = _execute.make_int(feature_width, "feature_width")
  num_sc_per_chip = _execute.make_int(num_sc_per_chip, "num_sc_per_chip")
  table_name = _execute.make_str(table_name, "table_name")
  mini_batch_splits = _execute.make_str(mini_batch_splits, "mini_batch_splits")
  program_key = _ops.convert_to_tensor(program_key, _dtypes.string)
  max_ids = _ops.convert_to_tensor(max_ids, _dtypes.int32)
  max_uniques = _ops.convert_to_tensor(max_uniques, _dtypes.int32)
  _inputs_flat = [program_key, max_ids, max_uniques]
  _attrs = ("sample_count", sample_count, "num_replica", num_replica,
  "feature_width", feature_width, "num_sc_per_chip", num_sc_per_chip,
  "table_name", table_name, "mini_batch_splits", mini_batch_splits)
  _result = _execute.execute(b"StoreMinibatchStatisticsInFdo", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def tpu_annotate_tensors_with_dynamic_shape(tensors, name=None):
  r"""TODO: add doc.

  Args:
    tensors: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `tensors`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUAnnotateTensorsWithDynamicShape", name, tensors)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_annotate_tensors_with_dynamic_shape_eager_fallback(
          tensors, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUAnnotateTensorsWithDynamicShape", tensors=tensors, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUAnnotateTensorsWithDynamicShape", _inputs_flat, _attrs, _result)
  return _result

TPUAnnotateTensorsWithDynamicShape = tf_export("raw_ops.TPUAnnotateTensorsWithDynamicShape")(_ops.to_raw_op(tpu_annotate_tensors_with_dynamic_shape))


def tpu_annotate_tensors_with_dynamic_shape_eager_fallback(tensors, name, ctx):
  _attr_T, tensors = _execute.convert_to_mixed_eager_tensors(tensors, ctx)
  _inputs_flat = list(tensors)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TPUAnnotateTensorsWithDynamicShape",
                             len(tensors), inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUAnnotateTensorsWithDynamicShape", _inputs_flat, _attrs, _result)
  return _result


def tpu_copy_with_dynamic_shape(tensors, unpadded_sizes: Annotated[List[Any], _atypes.Int32], name=None):
  r"""Op that copies host tensor to device with dynamic shape support.
For internal use only.

  Args:
    tensors: A list of `Tensor` objects.
    unpadded_sizes: A list of `Tensor` objects with type `int32`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `tensors`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUCopyWithDynamicShape", name, tensors, unpadded_sizes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_copy_with_dynamic_shape_eager_fallback(
          tensors, unpadded_sizes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(unpadded_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'unpadded_sizes' argument to "
        "'tpu_copy_with_dynamic_shape' Op, not %r." % unpadded_sizes)
  _attr_N = len(unpadded_sizes)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUCopyWithDynamicShape", tensors=tensors,
                                   unpadded_sizes=unpadded_sizes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUCopyWithDynamicShape", _inputs_flat, _attrs, _result)
  return _result

TPUCopyWithDynamicShape = tf_export("raw_ops.TPUCopyWithDynamicShape")(_ops.to_raw_op(tpu_copy_with_dynamic_shape))


def tpu_copy_with_dynamic_shape_eager_fallback(tensors, unpadded_sizes: Annotated[List[Any], _atypes.Int32], name, ctx):
  if not isinstance(unpadded_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'unpadded_sizes' argument to "
        "'tpu_copy_with_dynamic_shape' Op, not %r." % unpadded_sizes)
  _attr_N = len(unpadded_sizes)
  _attr_T, tensors = _execute.convert_to_mixed_eager_tensors(tensors, ctx)
  unpadded_sizes = _ops.convert_n_to_tensor(unpadded_sizes, _dtypes.int32)
  _inputs_flat = list(tensors) + list(unpadded_sizes)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"TPUCopyWithDynamicShape", len(tensors),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUCopyWithDynamicShape", _inputs_flat, _attrs, _result)
  return _result

_XlaSparseCoreAdagradOutput = collections.namedtuple(
    "XlaSparseCoreAdagrad",
    ["updated_embedding_table", "updated_accumulator"])


def xla_sparse_core_adagrad(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, name=None):
  r"""TODO: add doc.

  Args:
    indices: A `Tensor` of type `int32`.
    gradient: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    feature_width: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseCoreAdagrad", name, indices, gradient, learning_rate,
        accumulator, embedding_table, "feature_width", feature_width)
      _result = _XlaSparseCoreAdagradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_core_adagrad_eager_fallback(
          indices, gradient, learning_rate, accumulator, embedding_table,
          feature_width=feature_width, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  feature_width = _execute.make_int(feature_width, "feature_width")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseCoreAdagrad", indices=indices, gradient=gradient,
                                learning_rate=learning_rate,
                                accumulator=accumulator,
                                embedding_table=embedding_table,
                                feature_width=feature_width, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_width", _op._get_attr_int("feature_width"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseCoreAdagrad", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdagradOutput._make(_result)
  return _result

XlaSparseCoreAdagrad = tf_export("raw_ops.XlaSparseCoreAdagrad")(_ops.to_raw_op(xla_sparse_core_adagrad))


def xla_sparse_core_adagrad_eager_fallback(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, name, ctx):
  feature_width = _execute.make_int(feature_width, "feature_width")
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  gradient = _ops.convert_to_tensor(gradient, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  _inputs_flat = [indices, gradient, learning_rate, accumulator, embedding_table]
  _attrs = ("feature_width", feature_width)
  _result = _execute.execute(b"XlaSparseCoreAdagrad", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseCoreAdagrad", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdagradOutput._make(_result)
  return _result

_XlaSparseCoreAdagradMomentumOutput = collections.namedtuple(
    "XlaSparseCoreAdagradMomentum",
    ["updated_embedding_table", "updated_accumulator", "updated_momentum"])


def xla_sparse_core_adagrad_momentum(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], beta_1: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momentum: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, use_nesterov: bool, beta_2: float, exponent: float, name=None):
  r"""TODO: add doc.

  Args:
    indices: A `Tensor` of type `int32`.
    gradient: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    beta_1: A `Tensor` of type `float32`.
    epsilon: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    momentum: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    feature_width: An `int`.
    use_nesterov: A `bool`.
    beta_2: A `float`.
    exponent: A `float`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_momentum).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_momentum: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseCoreAdagradMomentum", name, indices, gradient,
        learning_rate, beta_1, epsilon, accumulator, momentum,
        embedding_table, "feature_width", feature_width, "use_nesterov",
        use_nesterov, "beta_2", beta_2, "exponent", exponent)
      _result = _XlaSparseCoreAdagradMomentumOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_core_adagrad_momentum_eager_fallback(
          indices, gradient, learning_rate, beta_1, epsilon, accumulator,
          momentum, embedding_table, feature_width=feature_width,
          use_nesterov=use_nesterov, beta_2=beta_2, exponent=exponent,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  feature_width = _execute.make_int(feature_width, "feature_width")
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  beta_2 = _execute.make_float(beta_2, "beta_2")
  exponent = _execute.make_float(exponent, "exponent")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseCoreAdagradMomentum", indices=indices, gradient=gradient,
                                        learning_rate=learning_rate,
                                        beta_1=beta_1, epsilon=epsilon,
                                        accumulator=accumulator,
                                        momentum=momentum,
                                        embedding_table=embedding_table,
                                        feature_width=feature_width,
                                        use_nesterov=use_nesterov,
                                        beta_2=beta_2, exponent=exponent,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_width", _op._get_attr_int("feature_width"),
              "use_nesterov", _op._get_attr_bool("use_nesterov"), "beta_2",
              _op.get_attr("beta_2"), "exponent", _op.get_attr("exponent"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseCoreAdagradMomentum", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdagradMomentumOutput._make(_result)
  return _result

XlaSparseCoreAdagradMomentum = tf_export("raw_ops.XlaSparseCoreAdagradMomentum")(_ops.to_raw_op(xla_sparse_core_adagrad_momentum))


def xla_sparse_core_adagrad_momentum_eager_fallback(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], beta_1: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momentum: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, use_nesterov: bool, beta_2: float, exponent: float, name, ctx):
  feature_width = _execute.make_int(feature_width, "feature_width")
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  beta_2 = _execute.make_float(beta_2, "beta_2")
  exponent = _execute.make_float(exponent, "exponent")
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  gradient = _ops.convert_to_tensor(gradient, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  beta_1 = _ops.convert_to_tensor(beta_1, _dtypes.float32)
  epsilon = _ops.convert_to_tensor(epsilon, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  momentum = _ops.convert_to_tensor(momentum, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  _inputs_flat = [indices, gradient, learning_rate, beta_1, epsilon, accumulator, momentum, embedding_table]
  _attrs = ("feature_width", feature_width, "use_nesterov", use_nesterov,
  "beta_2", beta_2, "exponent", exponent)
  _result = _execute.execute(b"XlaSparseCoreAdagradMomentum", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseCoreAdagradMomentum", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdagradMomentumOutput._make(_result)
  return _result

_XlaSparseCoreAdamOutput = collections.namedtuple(
    "XlaSparseCoreAdam",
    ["updated_embedding_table", "updated_velocity", "updated_momentum"])


def xla_sparse_core_adam(embedding_table: Annotated[Any, _atypes.Float32], indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], momentum: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], beta_1: Annotated[Any, _atypes.Float32], beta_2: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], feature_width: int, use_sum_inside_sqrt: bool, name=None):
  r"""TODO: add doc.

  Args:
    embedding_table: A `Tensor` of type `float32`.
    indices: A `Tensor` of type `int32`.
    gradient: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    momentum: A `Tensor` of type `float32`.
    velocity: A `Tensor` of type `float32`.
    beta_1: A `Tensor` of type `float32`.
    beta_2: A `Tensor` of type `float32`.
    epsilon: A `Tensor` of type `float32`.
    feature_width: An `int`.
    use_sum_inside_sqrt: A `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_velocity, updated_momentum).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_velocity: A `Tensor` of type `float32`.
    updated_momentum: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseCoreAdam", name, embedding_table, indices, gradient,
        learning_rate, momentum, velocity, beta_1, beta_2, epsilon,
        "feature_width", feature_width, "use_sum_inside_sqrt",
        use_sum_inside_sqrt)
      _result = _XlaSparseCoreAdamOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_core_adam_eager_fallback(
          embedding_table, indices, gradient, learning_rate, momentum,
          velocity, beta_1, beta_2, epsilon, feature_width=feature_width,
          use_sum_inside_sqrt=use_sum_inside_sqrt, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  feature_width = _execute.make_int(feature_width, "feature_width")
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseCoreAdam", embedding_table=embedding_table, indices=indices,
                             gradient=gradient, learning_rate=learning_rate,
                             momentum=momentum, velocity=velocity,
                             beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                             feature_width=feature_width,
                             use_sum_inside_sqrt=use_sum_inside_sqrt,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_width", _op._get_attr_int("feature_width"),
              "use_sum_inside_sqrt",
              _op._get_attr_bool("use_sum_inside_sqrt"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseCoreAdam", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdamOutput._make(_result)
  return _result

XlaSparseCoreAdam = tf_export("raw_ops.XlaSparseCoreAdam")(_ops.to_raw_op(xla_sparse_core_adam))


def xla_sparse_core_adam_eager_fallback(embedding_table: Annotated[Any, _atypes.Float32], indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], momentum: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], beta_1: Annotated[Any, _atypes.Float32], beta_2: Annotated[Any, _atypes.Float32], epsilon: Annotated[Any, _atypes.Float32], feature_width: int, use_sum_inside_sqrt: bool, name, ctx):
  feature_width = _execute.make_int(feature_width, "feature_width")
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  gradient = _ops.convert_to_tensor(gradient, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  momentum = _ops.convert_to_tensor(momentum, _dtypes.float32)
  velocity = _ops.convert_to_tensor(velocity, _dtypes.float32)
  beta_1 = _ops.convert_to_tensor(beta_1, _dtypes.float32)
  beta_2 = _ops.convert_to_tensor(beta_2, _dtypes.float32)
  epsilon = _ops.convert_to_tensor(epsilon, _dtypes.float32)
  _inputs_flat = [embedding_table, indices, gradient, learning_rate, momentum, velocity, beta_1, beta_2, epsilon]
  _attrs = ("feature_width", feature_width, "use_sum_inside_sqrt",
  use_sum_inside_sqrt)
  _result = _execute.execute(b"XlaSparseCoreAdam", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseCoreAdam", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreAdamOutput._make(_result)
  return _result

_XlaSparseCoreFtrlOutput = collections.namedtuple(
    "XlaSparseCoreFtrl",
    ["updated_embedding_table", "updated_accumulator", "updated_linear"])


def xla_sparse_core_ftrl(embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], beta: Annotated[Any, _atypes.Float32], learning_rate_power: Annotated[Any, _atypes.Float32], l2_regularization_strength: Annotated[Any, _atypes.Float32], feature_width: int, multiply_linear_by_learning_rate: bool, l1_regularization_strength: float, name=None):
  r"""TODO: add doc.

  Args:
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    linear: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    indices: A `Tensor` of type `int32`.
    gradient: A `Tensor` of type `float32`.
    beta: A `Tensor` of type `float32`.
    learning_rate_power: A `Tensor` of type `float32`.
    l2_regularization_strength: A `Tensor` of type `float32`.
    feature_width: An `int`.
    multiply_linear_by_learning_rate: A `bool`.
    l1_regularization_strength: A `float`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_linear).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_linear: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseCoreFtrl", name, embedding_table, accumulator, linear,
        learning_rate, indices, gradient, beta, learning_rate_power,
        l2_regularization_strength, "feature_width", feature_width,
        "multiply_linear_by_learning_rate", multiply_linear_by_learning_rate,
        "l1_regularization_strength", l1_regularization_strength)
      _result = _XlaSparseCoreFtrlOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_core_ftrl_eager_fallback(
          embedding_table, accumulator, linear, learning_rate, indices,
          gradient, beta, learning_rate_power, l2_regularization_strength,
          feature_width=feature_width,
          multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
          l1_regularization_strength=l1_regularization_strength, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  feature_width = _execute.make_int(feature_width, "feature_width")
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseCoreFtrl", embedding_table=embedding_table,
                             accumulator=accumulator, linear=linear,
                             learning_rate=learning_rate, indices=indices,
                             gradient=gradient, beta=beta,
                             learning_rate_power=learning_rate_power,
                             l2_regularization_strength=l2_regularization_strength,
                             feature_width=feature_width,
                             multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
                             l1_regularization_strength=l1_regularization_strength,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_width", _op._get_attr_int("feature_width"),
              "multiply_linear_by_learning_rate",
              _op._get_attr_bool("multiply_linear_by_learning_rate"),
              "l1_regularization_strength",
              _op.get_attr("l1_regularization_strength"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseCoreFtrl", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreFtrlOutput._make(_result)
  return _result

XlaSparseCoreFtrl = tf_export("raw_ops.XlaSparseCoreFtrl")(_ops.to_raw_op(xla_sparse_core_ftrl))


def xla_sparse_core_ftrl_eager_fallback(embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], beta: Annotated[Any, _atypes.Float32], learning_rate_power: Annotated[Any, _atypes.Float32], l2_regularization_strength: Annotated[Any, _atypes.Float32], feature_width: int, multiply_linear_by_learning_rate: bool, l1_regularization_strength: float, name, ctx):
  feature_width = _execute.make_int(feature_width, "feature_width")
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  linear = _ops.convert_to_tensor(linear, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  gradient = _ops.convert_to_tensor(gradient, _dtypes.float32)
  beta = _ops.convert_to_tensor(beta, _dtypes.float32)
  learning_rate_power = _ops.convert_to_tensor(learning_rate_power, _dtypes.float32)
  l2_regularization_strength = _ops.convert_to_tensor(l2_regularization_strength, _dtypes.float32)
  _inputs_flat = [embedding_table, accumulator, linear, learning_rate, indices, gradient, beta, learning_rate_power, l2_regularization_strength]
  _attrs = ("feature_width", feature_width,
  "multiply_linear_by_learning_rate", multiply_linear_by_learning_rate,
  "l1_regularization_strength", l1_regularization_strength)
  _result = _execute.execute(b"XlaSparseCoreFtrl", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseCoreFtrl", _inputs_flat, _attrs, _result)
  _result = _XlaSparseCoreFtrlOutput._make(_result)
  return _result


def xla_sparse_core_sgd(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    indices: A `Tensor` of type `int32`.
    gradient: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    feature_width: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseCoreSgd", name, indices, gradient, learning_rate,
        embedding_table, "feature_width", feature_width)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_core_sgd_eager_fallback(
          indices, gradient, learning_rate, embedding_table,
          feature_width=feature_width, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  feature_width = _execute.make_int(feature_width, "feature_width")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseCoreSgd", indices=indices, gradient=gradient,
                            learning_rate=learning_rate,
                            embedding_table=embedding_table,
                            feature_width=feature_width, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_width", _op._get_attr_int("feature_width"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseCoreSgd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSparseCoreSgd = tf_export("raw_ops.XlaSparseCoreSgd")(_ops.to_raw_op(xla_sparse_core_sgd))


def xla_sparse_core_sgd_eager_fallback(indices: Annotated[Any, _atypes.Int32], gradient: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], feature_width: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  feature_width = _execute.make_int(feature_width, "feature_width")
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  gradient = _ops.convert_to_tensor(gradient, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  _inputs_flat = [indices, gradient, learning_rate, embedding_table]
  _attrs = ("feature_width", feature_width)
  _result = _execute.execute(b"XlaSparseCoreSgd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseCoreSgd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaSparseDenseMatmulOutput = collections.namedtuple(
    "XlaSparseDenseMatmul",
    ["activations", "row_pointers", "sorted_embedding_ids", "sorted_sample_ids", "sorted_gains"])


def xla_sparse_dense_matmul(row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.UInt32], values: Annotated[Any, _atypes.Float32], offsets: Annotated[Any, _atypes.UInt32], embedding_table: Annotated[Any, _atypes.Float32], max_ids_per_partition: int, max_unique_ids_per_partition: int, input_size: int, name=None):
  r"""TODO: add doc.

  Args:
    row_ids: A `Tensor` of type `int32`.
    col_ids: A `Tensor` of type `uint32`.
    values: A `Tensor` of type `float32`.
    offsets: A `Tensor` of type `uint32`.
    embedding_table: A `Tensor` of type `float32`.
    max_ids_per_partition: An `int` that is `>= 0`.
    max_unique_ids_per_partition: An `int` that is `>= 0`.
    input_size: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, row_pointers, sorted_embedding_ids, sorted_sample_ids, sorted_gains).

    activations: A `Tensor` of type `float32`.
    row_pointers: A `Tensor` of type `int32`.
    sorted_embedding_ids: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmul", name, row_ids, col_ids, values, offsets,
        embedding_table, "max_ids_per_partition", max_ids_per_partition,
        "max_unique_ids_per_partition", max_unique_ids_per_partition,
        "input_size", input_size)
      _result = _XlaSparseDenseMatmulOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_eager_fallback(
          row_ids, col_ids, values, offsets, embedding_table,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
          input_size=input_size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  max_ids_per_partition = _execute.make_int(max_ids_per_partition, "max_ids_per_partition")
  max_unique_ids_per_partition = _execute.make_int(max_unique_ids_per_partition, "max_unique_ids_per_partition")
  input_size = _execute.make_int(input_size, "input_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmul", row_ids=row_ids, col_ids=col_ids,
                                values=values, offsets=offsets,
                                embedding_table=embedding_table,
                                max_ids_per_partition=max_ids_per_partition,
                                max_unique_ids_per_partition=max_unique_ids_per_partition,
                                input_size=input_size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_ids_per_partition",
              _op._get_attr_int("max_ids_per_partition"),
              "max_unique_ids_per_partition",
              _op._get_attr_int("max_unique_ids_per_partition"), "input_size",
              _op._get_attr_int("input_size"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmul", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulOutput._make(_result)
  return _result

XlaSparseDenseMatmul = tf_export("raw_ops.XlaSparseDenseMatmul")(_ops.to_raw_op(xla_sparse_dense_matmul))


def xla_sparse_dense_matmul_eager_fallback(row_ids: Annotated[Any, _atypes.Int32], col_ids: Annotated[Any, _atypes.UInt32], values: Annotated[Any, _atypes.Float32], offsets: Annotated[Any, _atypes.UInt32], embedding_table: Annotated[Any, _atypes.Float32], max_ids_per_partition: int, max_unique_ids_per_partition: int, input_size: int, name, ctx):
  max_ids_per_partition = _execute.make_int(max_ids_per_partition, "max_ids_per_partition")
  max_unique_ids_per_partition = _execute.make_int(max_unique_ids_per_partition, "max_unique_ids_per_partition")
  input_size = _execute.make_int(input_size, "input_size")
  row_ids = _ops.convert_to_tensor(row_ids, _dtypes.int32)
  col_ids = _ops.convert_to_tensor(col_ids, _dtypes.uint32)
  values = _ops.convert_to_tensor(values, _dtypes.float32)
  offsets = _ops.convert_to_tensor(offsets, _dtypes.uint32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  _inputs_flat = [row_ids, col_ids, values, offsets, embedding_table]
  _attrs = ("max_ids_per_partition", max_ids_per_partition,
  "max_unique_ids_per_partition", max_unique_ids_per_partition, "input_size",
  input_size)
  _result = _execute.execute(b"XlaSparseDenseMatmul", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmul", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdagradAndCsrInputOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdagradAndCsrInput",
    ["updated_embedding_table", "updated_accumulator"])


def xla_sparse_dense_matmul_grad_with_adagrad_and_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithAdagradAndCsrInput", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        num_minibatches_per_physical_sparse_core, "clip_weight_min",
        clip_weight_min, "clip_weight_max", clip_weight_max, "table_name",
        table_name)
      _result = _XlaSparseDenseMatmulGradWithAdagradAndCsrInputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adagrad_and_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          num_minibatches_per_physical_sparse_core,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdagradAndCsrInput", row_pointers=row_pointers,
                                                          sorted_sample_ids=sorted_sample_ids,
                                                          sorted_token_ids=sorted_token_ids,
                                                          sorted_gains=sorted_gains,
                                                          activation_gradients=activation_gradients,
                                                          learning_rate=learning_rate,
                                                          embedding_table=embedding_table,
                                                          accumulator=accumulator,
                                                          num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                          table_name=table_name,
                                                          clip_weight_min=clip_weight_min,
                                                          clip_weight_max=clip_weight_max,
                                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("clip_weight_min", _op.get_attr("clip_weight_min"),
              "clip_weight_max", _op.get_attr("clip_weight_max"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradAndCsrInputOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdagradAndCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdagradAndCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adagrad_and_csr_input))


def xla_sparse_dense_matmul_grad_with_adagrad_and_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, num_minibatches_per_physical_sparse_core]
  _attrs = ("clip_weight_min", clip_weight_min, "clip_weight_max",
  clip_weight_max, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdagradAndCsrInput",
                             2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradAndCsrInputOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize",
    ["updated_embedding_table", "updated_accumulator"])


def xla_sparse_dense_matmul_grad_with_adagrad_and_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        num_minibatches_per_physical_sparse_core, "clip_weight_min",
        clip_weight_min, "clip_weight_max", clip_weight_max,
        "max_ids_per_sparse_core", max_ids_per_sparse_core,
        "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
        "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adagrad_and_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          num_minibatches_per_physical_sparse_core,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize", row_pointers=row_pointers,
                                                                  sorted_sample_ids=sorted_sample_ids,
                                                                  sorted_token_ids=sorted_token_ids,
                                                                  sorted_gains=sorted_gains,
                                                                  activation_gradients=activation_gradients,
                                                                  learning_rate=learning_rate,
                                                                  embedding_table=embedding_table,
                                                                  accumulator=accumulator,
                                                                  num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                                  max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                                  max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                                  table_name=table_name,
                                                                  clip_weight_min=clip_weight_min,
                                                                  clip_weight_max=clip_weight_max,
                                                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("clip_weight_min", _op.get_attr("clip_weight_min"),
              "clip_weight_max", _op.get_attr("clip_weight_max"),
              "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adagrad_and_static_buffer_size))


def xla_sparse_dense_matmul_grad_with_adagrad_and_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, num_minibatches_per_physical_sparse_core]
  _attrs = ("clip_weight_min", clip_weight_min, "clip_weight_max",
  clip_weight_max, "max_ids_per_sparse_core", max_ids_per_sparse_core,
  "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
  "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize",
                             2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput",
    ["updated_embedding_table", "updated_accumulator", "updated_momenta"])


def xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_nesterov: bool, exponent: float, beta1: float, beta2: float, epsilon: float, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    use_nesterov: A `bool`.
    exponent: A `float`.
    beta1: A `float`.
    beta2: A `float`.
    epsilon: A `float`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_momenta).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_momenta: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        momenta, num_minibatches_per_physical_sparse_core, "use_nesterov",
        use_nesterov, "exponent", exponent, "beta1", beta1, "beta2", beta2,
        "epsilon", epsilon, "clip_weight_min", clip_weight_min,
        "clip_weight_max", clip_weight_max, "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          momenta, num_minibatches_per_physical_sparse_core,
          use_nesterov=use_nesterov, exponent=exponent, beta1=beta1,
          beta2=beta2, epsilon=epsilon, clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max, table_name=table_name, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  exponent = _execute.make_float(exponent, "exponent")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput", row_pointers=row_pointers,
                                                                  sorted_sample_ids=sorted_sample_ids,
                                                                  sorted_token_ids=sorted_token_ids,
                                                                  sorted_gains=sorted_gains,
                                                                  activation_gradients=activation_gradients,
                                                                  learning_rate=learning_rate,
                                                                  embedding_table=embedding_table,
                                                                  accumulator=accumulator,
                                                                  momenta=momenta,
                                                                  num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                                  use_nesterov=use_nesterov,
                                                                  exponent=exponent,
                                                                  beta1=beta1,
                                                                  beta2=beta2,
                                                                  epsilon=epsilon,
                                                                  table_name=table_name,
                                                                  clip_weight_min=clip_weight_min,
                                                                  clip_weight_max=clip_weight_max,
                                                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_nesterov", _op._get_attr_bool("use_nesterov"), "exponent",
              _op.get_attr("exponent"), "beta1", _op.get_attr("beta1"),
              "beta2", _op.get_attr("beta2"), "epsilon",
              _op.get_attr("epsilon"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "table_name",
              _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_csr_input))


def xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_nesterov: bool, exponent: float, beta1: float, beta2: float, epsilon: float, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  exponent = _execute.make_float(exponent, "exponent")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, momenta, num_minibatches_per_physical_sparse_core]
  _attrs = ("use_nesterov", use_nesterov, "exponent", exponent, "beta1",
  beta1, "beta2", beta2, "epsilon", epsilon, "clip_weight_min",
  clip_weight_min, "clip_weight_max", clip_weight_max, "table_name",
  table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
    ["updated_embedding_table", "updated_accumulator", "updated_momenta"])


def xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_nesterov: bool, exponent: float, beta1: float, beta2: float, epsilon: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    use_nesterov: A `bool`.
    exponent: A `float`.
    beta1: A `float`.
    beta2: A `float`.
    epsilon: A `float`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_momenta).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_momenta: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx,
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
        name, row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        momenta, num_minibatches_per_physical_sparse_core, "use_nesterov",
        use_nesterov, "exponent", exponent, "beta1", beta1, "beta2", beta2,
        "epsilon", epsilon, "clip_weight_min", clip_weight_min,
        "clip_weight_max", clip_weight_max, "max_ids_per_sparse_core",
        max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
        max_unique_ids_per_sparse_core, "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          momenta, num_minibatches_per_physical_sparse_core,
          use_nesterov=use_nesterov, exponent=exponent, beta1=beta1,
          beta2=beta2, epsilon=epsilon, clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  exponent = _execute.make_float(exponent, "exponent")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize", row_pointers=row_pointers,
                                                                          sorted_sample_ids=sorted_sample_ids,
                                                                          sorted_token_ids=sorted_token_ids,
                                                                          sorted_gains=sorted_gains,
                                                                          activation_gradients=activation_gradients,
                                                                          learning_rate=learning_rate,
                                                                          embedding_table=embedding_table,
                                                                          accumulator=accumulator,
                                                                          momenta=momenta,
                                                                          num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                                          use_nesterov=use_nesterov,
                                                                          exponent=exponent,
                                                                          beta1=beta1,
                                                                          beta2=beta2,
                                                                          epsilon=epsilon,
                                                                          max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                                          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                                          table_name=table_name,
                                                                          clip_weight_min=clip_weight_min,
                                                                          clip_weight_max=clip_weight_max,
                                                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_nesterov", _op._get_attr_bool("use_nesterov"), "exponent",
              _op.get_attr("exponent"), "beta1", _op.get_attr("beta1"),
              "beta2", _op.get_attr("beta2"), "epsilon",
              _op.get_attr("epsilon"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size))


def xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_nesterov: bool, exponent: float, beta1: float, beta2: float, epsilon: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  exponent = _execute.make_float(exponent, "exponent")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, momenta, num_minibatches_per_physical_sparse_core]
  _attrs = ("use_nesterov", use_nesterov, "exponent", exponent, "beta1",
  beta1, "beta2", beta2, "epsilon", epsilon, "clip_weight_min",
  clip_weight_min, "clip_weight_max", clip_weight_max,
  "max_ids_per_sparse_core", max_ids_per_sparse_core,
  "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
  "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdamAndCsrInputOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdamAndCsrInput",
    ["updated_embedding_table", "updated_momenta", "updated_velocity"])


def xla_sparse_dense_matmul_grad_with_adam_and_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_sum_inside_sqrt: bool, beta1: float, beta2: float, epsilon: float, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    velocity: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    use_sum_inside_sqrt: A `bool`.
    beta1: A `float`.
    beta2: A `float`.
    epsilon: A `float`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_momenta, updated_velocity).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_momenta: A `Tensor` of type `float32`.
    updated_velocity: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithAdamAndCsrInput", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, momenta,
        velocity, num_minibatches_per_physical_sparse_core,
        "use_sum_inside_sqrt", use_sum_inside_sqrt, "beta1", beta1, "beta2",
        beta2, "epsilon", epsilon, "clip_weight_min", clip_weight_min,
        "clip_weight_max", clip_weight_max, "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithAdamAndCsrInputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adam_and_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, momenta,
          velocity, num_minibatches_per_physical_sparse_core,
          use_sum_inside_sqrt=use_sum_inside_sqrt, beta1=beta1, beta2=beta2,
          epsilon=epsilon, clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max, table_name=table_name, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdamAndCsrInput", row_pointers=row_pointers,
                                                       sorted_sample_ids=sorted_sample_ids,
                                                       sorted_token_ids=sorted_token_ids,
                                                       sorted_gains=sorted_gains,
                                                       activation_gradients=activation_gradients,
                                                       learning_rate=learning_rate,
                                                       embedding_table=embedding_table,
                                                       momenta=momenta,
                                                       velocity=velocity,
                                                       num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                       use_sum_inside_sqrt=use_sum_inside_sqrt,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       epsilon=epsilon,
                                                       table_name=table_name,
                                                       clip_weight_min=clip_weight_min,
                                                       clip_weight_max=clip_weight_max,
                                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_sum_inside_sqrt",
              _op._get_attr_bool("use_sum_inside_sqrt"), "beta1",
              _op.get_attr("beta1"), "beta2", _op.get_attr("beta2"),
              "epsilon", _op.get_attr("epsilon"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "table_name",
              _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdamAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdamAndCsrInputOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdamAndCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdamAndCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adam_and_csr_input))


def xla_sparse_dense_matmul_grad_with_adam_and_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_sum_inside_sqrt: bool, beta1: float, beta2: float, epsilon: float, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  velocity = _ops.convert_to_tensor(velocity, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, momenta, velocity, num_minibatches_per_physical_sparse_core]
  _attrs = ("use_sum_inside_sqrt", use_sum_inside_sqrt, "beta1", beta1,
  "beta2", beta2, "epsilon", epsilon, "clip_weight_min", clip_weight_min,
  "clip_weight_max", clip_weight_max, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdamAndCsrInput",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdamAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdamAndCsrInputOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize",
    ["updated_embedding_table", "updated_momenta", "updated_velocity"])


def xla_sparse_dense_matmul_grad_with_adam_and_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_sum_inside_sqrt: bool, beta1: float, beta2: float, epsilon: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    velocity: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    use_sum_inside_sqrt: A `bool`.
    beta1: A `float`.
    beta2: A `float`.
    epsilon: A `float`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_momenta, updated_velocity).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_momenta: A `Tensor` of type `float32`.
    updated_velocity: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, momenta,
        velocity, num_minibatches_per_physical_sparse_core,
        "use_sum_inside_sqrt", use_sum_inside_sqrt, "beta1", beta1, "beta2",
        beta2, "epsilon", epsilon, "clip_weight_min", clip_weight_min,
        "clip_weight_max", clip_weight_max, "max_ids_per_sparse_core",
        max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
        max_unique_ids_per_sparse_core, "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_adam_and_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, momenta,
          velocity, num_minibatches_per_physical_sparse_core,
          use_sum_inside_sqrt=use_sum_inside_sqrt, beta1=beta1, beta2=beta2,
          epsilon=epsilon, clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize", row_pointers=row_pointers,
                                                               sorted_sample_ids=sorted_sample_ids,
                                                               sorted_token_ids=sorted_token_ids,
                                                               sorted_gains=sorted_gains,
                                                               activation_gradients=activation_gradients,
                                                               learning_rate=learning_rate,
                                                               embedding_table=embedding_table,
                                                               momenta=momenta,
                                                               velocity=velocity,
                                                               num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                               use_sum_inside_sqrt=use_sum_inside_sqrt,
                                                               beta1=beta1,
                                                               beta2=beta2,
                                                               epsilon=epsilon,
                                                               max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                               max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                               table_name=table_name,
                                                               clip_weight_min=clip_weight_min,
                                                               clip_weight_max=clip_weight_max,
                                                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_sum_inside_sqrt",
              _op._get_attr_bool("use_sum_inside_sqrt"), "beta1",
              _op.get_attr("beta1"), "beta2", _op.get_attr("beta2"),
              "epsilon", _op.get_attr("epsilon"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_adam_and_static_buffer_size))


def xla_sparse_dense_matmul_grad_with_adam_and_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], momenta: Annotated[Any, _atypes.Float32], velocity: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], use_sum_inside_sqrt: bool, beta1: float, beta2: float, epsilon: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  use_sum_inside_sqrt = _execute.make_bool(use_sum_inside_sqrt, "use_sum_inside_sqrt")
  beta1 = _execute.make_float(beta1, "beta1")
  beta2 = _execute.make_float(beta2, "beta2")
  epsilon = _execute.make_float(epsilon, "epsilon")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  momenta = _ops.convert_to_tensor(momenta, _dtypes.float32)
  velocity = _ops.convert_to_tensor(velocity, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, momenta, velocity, num_minibatches_per_physical_sparse_core]
  _attrs = ("use_sum_inside_sqrt", use_sum_inside_sqrt, "beta1", beta1,
  "beta2", beta2, "epsilon", epsilon, "clip_weight_min", clip_weight_min,
  "clip_weight_max", clip_weight_max, "max_ids_per_sparse_core",
  max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
  max_unique_ids_per_sparse_core, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOutput._make(_result)
  return _result


def xla_sparse_dense_matmul_grad_with_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], tables: Annotated[List[Any], _atypes.Float32], hyperparameters: Annotated[List[Any], _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], custom_computation, table_name: str, name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    tables: A list of at least 1 `Tensor` objects with type `float32`.
    hyperparameters: A list of at least 1 `Tensor` objects with type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    custom_computation: A function decorated with @Defun.
    table_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `tables` of `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithCsrInput", name, row_pointers,
        sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, tables, hyperparameters,
        num_minibatches_per_physical_sparse_core, "custom_computation",
        custom_computation, "table_name", table_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, tables, hyperparameters,
          num_minibatches_per_physical_sparse_core,
          custom_computation=custom_computation, table_name=table_name,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(tables, (list, tuple)):
    raise TypeError(
        "Expected list for 'tables' argument to "
        "'xla_sparse_dense_matmul_grad_with_csr_input' Op, not %r." % tables)
  _attr_N = len(tables)
  if not isinstance(hyperparameters, (list, tuple)):
    raise TypeError(
        "Expected list for 'hyperparameters' argument to "
        "'xla_sparse_dense_matmul_grad_with_csr_input' Op, not %r." % hyperparameters)
  _attr_M = len(hyperparameters)
  table_name = _execute.make_str(table_name, "table_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithCsrInput", row_pointers=row_pointers,
                                                sorted_sample_ids=sorted_sample_ids,
                                                sorted_token_ids=sorted_token_ids,
                                                sorted_gains=sorted_gains,
                                                activation_gradients=activation_gradients,
                                                tables=tables,
                                                hyperparameters=hyperparameters,
                                                num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                custom_computation=custom_computation,
                                                table_name=table_name,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "M", _op._get_attr_int("M"),
              "custom_computation", _op.get_attr("custom_computation"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithCsrInput", _inputs_flat, _attrs, _result)
  return _result

XlaSparseDenseMatmulGradWithCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_csr_input))


def xla_sparse_dense_matmul_grad_with_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], tables: Annotated[List[Any], _atypes.Float32], hyperparameters: Annotated[List[Any], _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], custom_computation, table_name: str, name, ctx):
  if not isinstance(tables, (list, tuple)):
    raise TypeError(
        "Expected list for 'tables' argument to "
        "'xla_sparse_dense_matmul_grad_with_csr_input' Op, not %r." % tables)
  _attr_N = len(tables)
  if not isinstance(hyperparameters, (list, tuple)):
    raise TypeError(
        "Expected list for 'hyperparameters' argument to "
        "'xla_sparse_dense_matmul_grad_with_csr_input' Op, not %r." % hyperparameters)
  _attr_M = len(hyperparameters)
  table_name = _execute.make_str(table_name, "table_name")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  tables = _ops.convert_n_to_tensor(tables, _dtypes.float32)
  hyperparameters = _ops.convert_n_to_tensor(hyperparameters, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients] + list(tables) + list(hyperparameters) + [num_minibatches_per_physical_sparse_core]
  _attrs = ("N", _attr_N, "M", _attr_M, "custom_computation",
  custom_computation, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithCsrInput", _attr_N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithCsrInput", _inputs_flat, _attrs, _result)
  return _result

_XlaSparseDenseMatmulGradWithFtrlAndCsrInputOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithFtrlAndCsrInput",
    ["updated_embedding_table", "updated_accumulator", "updated_linear"])


def xla_sparse_dense_matmul_grad_with_ftrl_and_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], multiply_linear_by_learning_rate: bool, beta: float, learning_rate_power: float, l1_regularization_strength: float, l2_regularization_strength: float, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    linear: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    multiply_linear_by_learning_rate: A `bool`.
    beta: A `float`.
    learning_rate_power: A `float`.
    l1_regularization_strength: A `float`.
    l2_regularization_strength: A `float`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_linear).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_linear: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithFtrlAndCsrInput", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        linear, num_minibatches_per_physical_sparse_core,
        "multiply_linear_by_learning_rate", multiply_linear_by_learning_rate,
        "beta", beta, "learning_rate_power", learning_rate_power,
        "l1_regularization_strength", l1_regularization_strength,
        "l2_regularization_strength", l2_regularization_strength,
        "clip_weight_min", clip_weight_min, "clip_weight_max",
        clip_weight_max, "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithFtrlAndCsrInputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_ftrl_and_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          linear, num_minibatches_per_physical_sparse_core,
          multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
          beta=beta, learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  beta = _execute.make_float(beta, "beta")
  learning_rate_power = _execute.make_float(learning_rate_power, "learning_rate_power")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  l2_regularization_strength = _execute.make_float(l2_regularization_strength, "l2_regularization_strength")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithFtrlAndCsrInput", row_pointers=row_pointers,
                                                       sorted_sample_ids=sorted_sample_ids,
                                                       sorted_token_ids=sorted_token_ids,
                                                       sorted_gains=sorted_gains,
                                                       activation_gradients=activation_gradients,
                                                       learning_rate=learning_rate,
                                                       embedding_table=embedding_table,
                                                       accumulator=accumulator,
                                                       linear=linear,
                                                       num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                       multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
                                                       beta=beta,
                                                       learning_rate_power=learning_rate_power,
                                                       l1_regularization_strength=l1_regularization_strength,
                                                       l2_regularization_strength=l2_regularization_strength,
                                                       table_name=table_name,
                                                       clip_weight_min=clip_weight_min,
                                                       clip_weight_max=clip_weight_max,
                                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("multiply_linear_by_learning_rate",
              _op._get_attr_bool("multiply_linear_by_learning_rate"), "beta",
              _op.get_attr("beta"), "learning_rate_power",
              _op.get_attr("learning_rate_power"),
              "l1_regularization_strength",
              _op.get_attr("l1_regularization_strength"),
              "l2_regularization_strength",
              _op.get_attr("l2_regularization_strength"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "table_name",
              _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithFtrlAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithFtrlAndCsrInputOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithFtrlAndCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithFtrlAndCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_ftrl_and_csr_input))


def xla_sparse_dense_matmul_grad_with_ftrl_and_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], multiply_linear_by_learning_rate: bool, beta: float, learning_rate_power: float, l1_regularization_strength: float, l2_regularization_strength: float, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  beta = _execute.make_float(beta, "beta")
  learning_rate_power = _execute.make_float(learning_rate_power, "learning_rate_power")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  l2_regularization_strength = _execute.make_float(l2_regularization_strength, "l2_regularization_strength")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  linear = _ops.convert_to_tensor(linear, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, linear, num_minibatches_per_physical_sparse_core]
  _attrs = ("multiply_linear_by_learning_rate",
  multiply_linear_by_learning_rate, "beta", beta, "learning_rate_power",
  learning_rate_power, "l1_regularization_strength",
  l1_regularization_strength, "l2_regularization_strength",
  l2_regularization_strength, "clip_weight_min", clip_weight_min,
  "clip_weight_max", clip_weight_max, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithFtrlAndCsrInput",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithFtrlAndCsrInput", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithFtrlAndCsrInputOutput._make(_result)
  return _result

_XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOutput = collections.namedtuple(
    "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize",
    ["updated_embedding_table", "updated_accumulator", "updated_linear"])


def xla_sparse_dense_matmul_grad_with_ftrl_and_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], multiply_linear_by_learning_rate: bool, beta: float, learning_rate_power: float, l1_regularization_strength: float, l2_regularization_strength: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None):
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    accumulator: A `Tensor` of type `float32`.
    linear: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    multiply_linear_by_learning_rate: A `bool`.
    beta: A `float`.
    learning_rate_power: A `float`.
    l1_regularization_strength: A `float`.
    l2_regularization_strength: A `float`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (updated_embedding_table, updated_accumulator, updated_linear).

    updated_embedding_table: A `Tensor` of type `float32`.
    updated_accumulator: A `Tensor` of type `float32`.
    updated_linear: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table, accumulator,
        linear, num_minibatches_per_physical_sparse_core,
        "multiply_linear_by_learning_rate", multiply_linear_by_learning_rate,
        "beta", beta, "learning_rate_power", learning_rate_power,
        "l1_regularization_strength", l1_regularization_strength,
        "l2_regularization_strength", l2_regularization_strength,
        "clip_weight_min", clip_weight_min, "clip_weight_max",
        clip_weight_max, "max_ids_per_sparse_core", max_ids_per_sparse_core,
        "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
        "table_name", table_name)
      _result = _XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_ftrl_and_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table, accumulator,
          linear, num_minibatches_per_physical_sparse_core,
          multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
          beta=beta, learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  beta = _execute.make_float(beta, "beta")
  learning_rate_power = _execute.make_float(learning_rate_power, "learning_rate_power")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  l2_regularization_strength = _execute.make_float(l2_regularization_strength, "l2_regularization_strength")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize", row_pointers=row_pointers,
                                                               sorted_sample_ids=sorted_sample_ids,
                                                               sorted_token_ids=sorted_token_ids,
                                                               sorted_gains=sorted_gains,
                                                               activation_gradients=activation_gradients,
                                                               learning_rate=learning_rate,
                                                               embedding_table=embedding_table,
                                                               accumulator=accumulator,
                                                               linear=linear,
                                                               num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                               multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
                                                               beta=beta,
                                                               learning_rate_power=learning_rate_power,
                                                               l1_regularization_strength=l1_regularization_strength,
                                                               l2_regularization_strength=l2_regularization_strength,
                                                               max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                               max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                               table_name=table_name,
                                                               clip_weight_min=clip_weight_min,
                                                               clip_weight_max=clip_weight_max,
                                                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("multiply_linear_by_learning_rate",
              _op._get_attr_bool("multiply_linear_by_learning_rate"), "beta",
              _op.get_attr("beta"), "learning_rate_power",
              _op.get_attr("learning_rate_power"),
              "l1_regularization_strength",
              _op.get_attr("l1_regularization_strength"),
              "l2_regularization_strength",
              _op.get_attr("l2_regularization_strength"), "clip_weight_min",
              _op.get_attr("clip_weight_min"), "clip_weight_max",
              _op.get_attr("clip_weight_max"), "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOutput._make(_result)
  return _result

XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_ftrl_and_static_buffer_size))


def xla_sparse_dense_matmul_grad_with_ftrl_and_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], accumulator: Annotated[Any, _atypes.Float32], linear: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], multiply_linear_by_learning_rate: bool, beta: float, learning_rate_power: float, l1_regularization_strength: float, l2_regularization_strength: float, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx):
  multiply_linear_by_learning_rate = _execute.make_bool(multiply_linear_by_learning_rate, "multiply_linear_by_learning_rate")
  beta = _execute.make_float(beta, "beta")
  learning_rate_power = _execute.make_float(learning_rate_power, "learning_rate_power")
  l1_regularization_strength = _execute.make_float(l1_regularization_strength, "l1_regularization_strength")
  l2_regularization_strength = _execute.make_float(l2_regularization_strength, "l2_regularization_strength")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  accumulator = _ops.convert_to_tensor(accumulator, _dtypes.float32)
  linear = _ops.convert_to_tensor(linear, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, accumulator, linear, num_minibatches_per_physical_sparse_core]
  _attrs = ("multiply_linear_by_learning_rate",
  multiply_linear_by_learning_rate, "beta", beta, "learning_rate_power",
  learning_rate_power, "l1_regularization_strength",
  l1_regularization_strength, "l2_regularization_strength",
  l2_regularization_strength, "clip_weight_min", clip_weight_min,
  "clip_weight_max", clip_weight_max, "max_ids_per_sparse_core",
  max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
  max_unique_ids_per_sparse_core, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result = _XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOutput._make(_result)
  return _result


def xla_sparse_dense_matmul_grad_with_sgd_and_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithSgdAndCsrInput", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table,
        num_minibatches_per_physical_sparse_core, "clip_weight_min",
        clip_weight_min, "clip_weight_max", clip_weight_max, "table_name",
        table_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_sgd_and_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table,
          num_minibatches_per_physical_sparse_core,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithSgdAndCsrInput", row_pointers=row_pointers,
                                                      sorted_sample_ids=sorted_sample_ids,
                                                      sorted_token_ids=sorted_token_ids,
                                                      sorted_gains=sorted_gains,
                                                      activation_gradients=activation_gradients,
                                                      learning_rate=learning_rate,
                                                      embedding_table=embedding_table,
                                                      num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                      table_name=table_name,
                                                      clip_weight_min=clip_weight_min,
                                                      clip_weight_max=clip_weight_max,
                                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("clip_weight_min", _op.get_attr("clip_weight_min"),
              "clip_weight_max", _op.get_attr("clip_weight_max"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithSgdAndCsrInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSparseDenseMatmulGradWithSgdAndCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulGradWithSgdAndCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_sgd_and_csr_input))


def xla_sparse_dense_matmul_grad_with_sgd_and_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx) -> Annotated[Any, _atypes.Float32]:
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, num_minibatches_per_physical_sparse_core]
  _attrs = ("clip_weight_min", clip_weight_min, "clip_weight_max",
  clip_weight_max, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithSgdAndCsrInput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithSgdAndCsrInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def xla_sparse_dense_matmul_grad_with_sgd_and_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min:float=float('-inf'), clip_weight_max:float=float('inf'), name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    activation_gradients: A `Tensor` of type `float32`.
    learning_rate: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    clip_weight_min: An optional `float`. Defaults to `float('-inf')`.
    clip_weight_max: An optional `float`. Defaults to `float('inf')`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize", name,
        row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
        activation_gradients, learning_rate, embedding_table,
        num_minibatches_per_physical_sparse_core, "clip_weight_min",
        clip_weight_min, "clip_weight_max", clip_weight_max,
        "max_ids_per_sparse_core", max_ids_per_sparse_core,
        "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
        "table_name", table_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_grad_with_sgd_and_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          activation_gradients, learning_rate, embedding_table,
          num_minibatches_per_physical_sparse_core,
          clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize", row_pointers=row_pointers,
                                                              sorted_sample_ids=sorted_sample_ids,
                                                              sorted_token_ids=sorted_token_ids,
                                                              sorted_gains=sorted_gains,
                                                              activation_gradients=activation_gradients,
                                                              learning_rate=learning_rate,
                                                              embedding_table=embedding_table,
                                                              num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                              max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                              max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                              table_name=table_name,
                                                              clip_weight_min=clip_weight_min,
                                                              clip_weight_max=clip_weight_max,
                                                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("clip_weight_min", _op.get_attr("clip_weight_min"),
              "clip_weight_max", _op.get_attr("clip_weight_max"),
              "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_grad_with_sgd_and_static_buffer_size))


def xla_sparse_dense_matmul_grad_with_sgd_and_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], activation_gradients: Annotated[Any, _atypes.Float32], learning_rate: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, clip_weight_min: float, clip_weight_max: float, name, ctx) -> Annotated[Any, _atypes.Float32]:
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  if clip_weight_min is None:
    clip_weight_min = float('-inf')
  clip_weight_min = _execute.make_float(clip_weight_min, "clip_weight_min")
  if clip_weight_max is None:
    clip_weight_max = float('inf')
  clip_weight_max = _execute.make_float(clip_weight_max, "clip_weight_max")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  activation_gradients = _ops.convert_to_tensor(activation_gradients, _dtypes.float32)
  learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, activation_gradients, learning_rate, embedding_table, num_minibatches_per_physical_sparse_core]
  _attrs = ("clip_weight_min", clip_weight_min, "clip_weight_max",
  clip_weight_max, "max_ids_per_sparse_core", max_ids_per_sparse_core,
  "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
  "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize",
                             1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def xla_sparse_dense_matmul_with_csr_input(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], input_size: int, quantization_config_low: float, quantization_config_high: float, quantization_config_num_buckets: int, table_name: str, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    input_size: An `int` that is `>= 0`.
    quantization_config_low: A `float`.
    quantization_config_high: A `float`.
    quantization_config_num_buckets: An `int` that is `>= 0`.
    table_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulWithCsrInput", name, row_pointers,
        sorted_sample_ids, sorted_token_ids, sorted_gains, embedding_table,
        num_minibatches_per_physical_sparse_core, "input_size", input_size,
        "quantization_config_low", quantization_config_low,
        "quantization_config_high", quantization_config_high,
        "quantization_config_num_buckets", quantization_config_num_buckets,
        "table_name", table_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_with_csr_input_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          embedding_table, num_minibatches_per_physical_sparse_core,
          input_size=input_size,
          quantization_config_low=quantization_config_low,
          quantization_config_high=quantization_config_high,
          quantization_config_num_buckets=quantization_config_num_buckets,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  input_size = _execute.make_int(input_size, "input_size")
  quantization_config_low = _execute.make_float(quantization_config_low, "quantization_config_low")
  quantization_config_high = _execute.make_float(quantization_config_high, "quantization_config_high")
  quantization_config_num_buckets = _execute.make_int(quantization_config_num_buckets, "quantization_config_num_buckets")
  table_name = _execute.make_str(table_name, "table_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulWithCsrInput", row_pointers=row_pointers,
                                            sorted_sample_ids=sorted_sample_ids,
                                            sorted_token_ids=sorted_token_ids,
                                            sorted_gains=sorted_gains,
                                            embedding_table=embedding_table,
                                            num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                            input_size=input_size,
                                            quantization_config_low=quantization_config_low,
                                            quantization_config_high=quantization_config_high,
                                            quantization_config_num_buckets=quantization_config_num_buckets,
                                            table_name=table_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_size", _op._get_attr_int("input_size"),
              "quantization_config_low",
              _op.get_attr("quantization_config_low"),
              "quantization_config_high",
              _op.get_attr("quantization_config_high"),
              "quantization_config_num_buckets",
              _op._get_attr_int("quantization_config_num_buckets"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulWithCsrInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSparseDenseMatmulWithCsrInput = tf_export("raw_ops.XlaSparseDenseMatmulWithCsrInput")(_ops.to_raw_op(xla_sparse_dense_matmul_with_csr_input))


def xla_sparse_dense_matmul_with_csr_input_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], input_size: int, quantization_config_low: float, quantization_config_high: float, quantization_config_num_buckets: int, table_name: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  input_size = _execute.make_int(input_size, "input_size")
  quantization_config_low = _execute.make_float(quantization_config_low, "quantization_config_low")
  quantization_config_high = _execute.make_float(quantization_config_high, "quantization_config_high")
  quantization_config_num_buckets = _execute.make_int(quantization_config_num_buckets, "quantization_config_num_buckets")
  table_name = _execute.make_str(table_name, "table_name")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, embedding_table, num_minibatches_per_physical_sparse_core]
  _attrs = ("input_size", input_size, "quantization_config_low",
  quantization_config_low, "quantization_config_high",
  quantization_config_high, "quantization_config_num_buckets",
  quantization_config_num_buckets, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulWithCsrInput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulWithCsrInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def xla_sparse_dense_matmul_with_static_buffer_size(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], input_size: int, quantization_config_low: float, quantization_config_high: float, quantization_config_num_buckets: int, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    row_pointers: A `Tensor` of type `int32`.
    sorted_sample_ids: A `Tensor` of type `int32`.
    sorted_token_ids: A `Tensor` of type `int32`.
    sorted_gains: A `Tensor` of type `float32`.
    embedding_table: A `Tensor` of type `float32`.
    num_minibatches_per_physical_sparse_core: A `Tensor` of type `int32`.
    input_size: An `int` that is `>= 0`.
    quantization_config_low: A `float`.
    quantization_config_high: A `float`.
    quantization_config_num_buckets: An `int` that is `>= 0`.
    max_ids_per_sparse_core: An `int` that is `>= 1`.
    max_unique_ids_per_sparse_core: An `int` that is `>= 1`.
    table_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSparseDenseMatmulWithStaticBufferSize", name, row_pointers,
        sorted_sample_ids, sorted_token_ids, sorted_gains, embedding_table,
        num_minibatches_per_physical_sparse_core, "input_size", input_size,
        "quantization_config_low", quantization_config_low,
        "quantization_config_high", quantization_config_high,
        "quantization_config_num_buckets", quantization_config_num_buckets,
        "max_ids_per_sparse_core", max_ids_per_sparse_core,
        "max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core,
        "table_name", table_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xla_sparse_dense_matmul_with_static_buffer_size_eager_fallback(
          row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains,
          embedding_table, num_minibatches_per_physical_sparse_core,
          input_size=input_size,
          quantization_config_low=quantization_config_low,
          quantization_config_high=quantization_config_high,
          quantization_config_num_buckets=quantization_config_num_buckets,
          max_ids_per_sparse_core=max_ids_per_sparse_core,
          max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
          table_name=table_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  input_size = _execute.make_int(input_size, "input_size")
  quantization_config_low = _execute.make_float(quantization_config_low, "quantization_config_low")
  quantization_config_high = _execute.make_float(quantization_config_high, "quantization_config_high")
  quantization_config_num_buckets = _execute.make_int(quantization_config_num_buckets, "quantization_config_num_buckets")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSparseDenseMatmulWithStaticBufferSize", row_pointers=row_pointers,
                                                    sorted_sample_ids=sorted_sample_ids,
                                                    sorted_token_ids=sorted_token_ids,
                                                    sorted_gains=sorted_gains,
                                                    embedding_table=embedding_table,
                                                    num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                                                    input_size=input_size,
                                                    quantization_config_low=quantization_config_low,
                                                    quantization_config_high=quantization_config_high,
                                                    quantization_config_num_buckets=quantization_config_num_buckets,
                                                    max_ids_per_sparse_core=max_ids_per_sparse_core,
                                                    max_unique_ids_per_sparse_core=max_unique_ids_per_sparse_core,
                                                    table_name=table_name,
                                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_size", _op._get_attr_int("input_size"),
              "quantization_config_low",
              _op.get_attr("quantization_config_low"),
              "quantization_config_high",
              _op.get_attr("quantization_config_high"),
              "quantization_config_num_buckets",
              _op._get_attr_int("quantization_config_num_buckets"),
              "max_ids_per_sparse_core",
              _op._get_attr_int("max_ids_per_sparse_core"),
              "max_unique_ids_per_sparse_core",
              _op._get_attr_int("max_unique_ids_per_sparse_core"),
              "table_name", _op.get_attr("table_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSparseDenseMatmulWithStaticBufferSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSparseDenseMatmulWithStaticBufferSize = tf_export("raw_ops.XlaSparseDenseMatmulWithStaticBufferSize")(_ops.to_raw_op(xla_sparse_dense_matmul_with_static_buffer_size))


def xla_sparse_dense_matmul_with_static_buffer_size_eager_fallback(row_pointers: Annotated[Any, _atypes.Int32], sorted_sample_ids: Annotated[Any, _atypes.Int32], sorted_token_ids: Annotated[Any, _atypes.Int32], sorted_gains: Annotated[Any, _atypes.Float32], embedding_table: Annotated[Any, _atypes.Float32], num_minibatches_per_physical_sparse_core: Annotated[Any, _atypes.Int32], input_size: int, quantization_config_low: float, quantization_config_high: float, quantization_config_num_buckets: int, max_ids_per_sparse_core: int, max_unique_ids_per_sparse_core: int, table_name: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  input_size = _execute.make_int(input_size, "input_size")
  quantization_config_low = _execute.make_float(quantization_config_low, "quantization_config_low")
  quantization_config_high = _execute.make_float(quantization_config_high, "quantization_config_high")
  quantization_config_num_buckets = _execute.make_int(quantization_config_num_buckets, "quantization_config_num_buckets")
  max_ids_per_sparse_core = _execute.make_int(max_ids_per_sparse_core, "max_ids_per_sparse_core")
  max_unique_ids_per_sparse_core = _execute.make_int(max_unique_ids_per_sparse_core, "max_unique_ids_per_sparse_core")
  table_name = _execute.make_str(table_name, "table_name")
  row_pointers = _ops.convert_to_tensor(row_pointers, _dtypes.int32)
  sorted_sample_ids = _ops.convert_to_tensor(sorted_sample_ids, _dtypes.int32)
  sorted_token_ids = _ops.convert_to_tensor(sorted_token_ids, _dtypes.int32)
  sorted_gains = _ops.convert_to_tensor(sorted_gains, _dtypes.float32)
  embedding_table = _ops.convert_to_tensor(embedding_table, _dtypes.float32)
  num_minibatches_per_physical_sparse_core = _ops.convert_to_tensor(num_minibatches_per_physical_sparse_core, _dtypes.int32)
  _inputs_flat = [row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, embedding_table, num_minibatches_per_physical_sparse_core]
  _attrs = ("input_size", input_size, "quantization_config_low",
  quantization_config_low, "quantization_config_high",
  quantization_config_high, "quantization_config_num_buckets",
  quantization_config_num_buckets, "max_ids_per_sparse_core",
  max_ids_per_sparse_core, "max_unique_ids_per_sparse_core",
  max_unique_ids_per_sparse_core, "table_name", table_name)
  _result = _execute.execute(b"XlaSparseDenseMatmulWithStaticBufferSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSparseDenseMatmulWithStaticBufferSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

