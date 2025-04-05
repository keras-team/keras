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
_RaggedTensorFromVariantOutput = collections.namedtuple(
    "RaggedTensorFromVariant",
    ["output_nested_splits", "output_dense_values"])


TV_RaggedTensorFromVariant_Tvalues = TypeVar("TV_RaggedTensorFromVariant_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedTensorFromVariant_Tsplits = TypeVar("TV_RaggedTensorFromVariant_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_tensor_from_variant(encoded_ragged: Annotated[Any, _atypes.Variant], input_ragged_rank: int, output_ragged_rank: int, Tvalues: TV_RaggedTensorFromVariant_Tvalues, Tsplits:TV_RaggedTensorFromVariant_Tsplits=_dtypes.int64, name=None):
  r"""Decodes a `variant` Tensor into a `RaggedTensor`.

  Decodes the given `variant` Tensor and returns a `RaggedTensor`. The input
  could be a scalar, meaning it encodes a single `RaggedTensor` with ragged_rank
  `output_ragged_rank`. It could also have an arbitrary rank, in which case each
  element is decoded into a `RaggedTensor` with ragged_rank `input_ragged_rank`
  and these are then stacked according to the input shape to output a single
  `RaggedTensor` with ragged_rank `output_ragged_rank`. Each `variant` element in
  the input Tensor is decoded by retrieving from the element a 1-D `variant`
  Tensor with `input_ragged_rank + 1` Tensors, corresponding to the splits and
  values of the decoded `RaggedTensor`. If `input_ragged_rank` is -1, then it is
  inferred as `output_ragged_rank` - `rank(encoded_ragged)`. See
  `RaggedTensorToVariant` for the corresponding encoding logic.

  Args:
    encoded_ragged: A `Tensor` of type `variant`.
      A `variant` Tensor containing encoded `RaggedTensor`s.
    input_ragged_rank: An `int` that is `>= -1`.
      The ragged rank of each encoded `RaggedTensor` component in the input. If set to
      -1, this is inferred as `output_ragged_rank` - `rank(encoded_ragged)`
    output_ragged_rank: An `int` that is `>= 0`.
      The expected ragged rank of the output `RaggedTensor`. The following must hold:
      `output_ragged_rank = rank(encoded_ragged) + input_ragged_rank`.
    Tvalues: A `tf.DType`.
    Tsplits: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_nested_splits, output_dense_values).

    output_nested_splits: A list of `output_ragged_rank` `Tensor` objects with type `Tsplits`.
    output_dense_values: A `Tensor` of type `Tvalues`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedTensorFromVariant", name, encoded_ragged,
        "input_ragged_rank", input_ragged_rank, "output_ragged_rank",
        output_ragged_rank, "Tvalues", Tvalues, "Tsplits", Tsplits)
      _result = _RaggedTensorFromVariantOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_tensor_from_variant_eager_fallback(
          encoded_ragged, input_ragged_rank=input_ragged_rank,
          output_ragged_rank=output_ragged_rank, Tvalues=Tvalues,
          Tsplits=Tsplits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  input_ragged_rank = _execute.make_int(input_ragged_rank, "input_ragged_rank")
  output_ragged_rank = _execute.make_int(output_ragged_rank, "output_ragged_rank")
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedTensorFromVariant", encoded_ragged=encoded_ragged,
                                   input_ragged_rank=input_ragged_rank,
                                   output_ragged_rank=output_ragged_rank,
                                   Tvalues=Tvalues, Tsplits=Tsplits,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_ragged_rank", _op._get_attr_int("input_ragged_rank"),
              "output_ragged_rank", _op._get_attr_int("output_ragged_rank"),
              "Tvalues", _op._get_attr_type("Tvalues"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedTensorFromVariant", _inputs_flat, _attrs, _result)
  _result = [_result[:output_ragged_rank]] + _result[output_ragged_rank:]
  _result = _RaggedTensorFromVariantOutput._make(_result)
  return _result

RaggedTensorFromVariant = tf_export("raw_ops.RaggedTensorFromVariant")(_ops.to_raw_op(ragged_tensor_from_variant))


def ragged_tensor_from_variant_eager_fallback(encoded_ragged: Annotated[Any, _atypes.Variant], input_ragged_rank: int, output_ragged_rank: int, Tvalues: TV_RaggedTensorFromVariant_Tvalues, Tsplits: TV_RaggedTensorFromVariant_Tsplits, name, ctx):
  input_ragged_rank = _execute.make_int(input_ragged_rank, "input_ragged_rank")
  output_ragged_rank = _execute.make_int(output_ragged_rank, "output_ragged_rank")
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  encoded_ragged = _ops.convert_to_tensor(encoded_ragged, _dtypes.variant)
  _inputs_flat = [encoded_ragged]
  _attrs = ("input_ragged_rank", input_ragged_rank, "output_ragged_rank",
  output_ragged_rank, "Tvalues", Tvalues, "Tsplits", Tsplits)
  _result = _execute.execute(b"RaggedTensorFromVariant", output_ragged_rank +
                             1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedTensorFromVariant", _inputs_flat, _attrs, _result)
  _result = [_result[:output_ragged_rank]] + _result[output_ragged_rank:]
  _result = _RaggedTensorFromVariantOutput._make(_result)
  return _result

_RaggedTensorToSparseOutput = collections.namedtuple(
    "RaggedTensorToSparse",
    ["sparse_indices", "sparse_values", "sparse_dense_shape"])


TV_RaggedTensorToSparse_T = TypeVar("TV_RaggedTensorToSparse_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedTensorToSparse_Tsplits = TypeVar("TV_RaggedTensorToSparse_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_tensor_to_sparse(rt_nested_splits: Annotated[List[Any], TV_RaggedTensorToSparse_Tsplits], rt_dense_values: Annotated[Any, TV_RaggedTensorToSparse_T], name=None):
  r"""Converts a `RaggedTensor` into a `SparseTensor` with the same values.

  input=ragged.from_nested_row_splits(rt_dense_values, rt_nested_splits)
  output=SparseTensor(indices=sparse_indices, values=sparse_values,
                      dense_shape=sparse_dense_shape)

  Args:
    rt_nested_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      The `row_splits` for the `RaggedTensor`.
    rt_dense_values: A `Tensor`. The `flat_values` for the `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_dense_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor`. Has the same type as `rt_dense_values`.
    sparse_dense_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedTensorToSparse", name, rt_nested_splits, rt_dense_values)
      _result = _RaggedTensorToSparseOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_tensor_to_sparse_eager_fallback(
          rt_nested_splits, rt_dense_values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(rt_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'rt_nested_splits' argument to "
        "'ragged_tensor_to_sparse' Op, not %r." % rt_nested_splits)
  _attr_RAGGED_RANK = len(rt_nested_splits)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedTensorToSparse", rt_nested_splits=rt_nested_splits,
                                rt_dense_values=rt_dense_values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("RAGGED_RANK", _op._get_attr_int("RAGGED_RANK"), "T",
              _op._get_attr_type("T"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedTensorToSparse", _inputs_flat, _attrs, _result)
  _result = _RaggedTensorToSparseOutput._make(_result)
  return _result

RaggedTensorToSparse = tf_export("raw_ops.RaggedTensorToSparse")(_ops.to_raw_op(ragged_tensor_to_sparse))


def ragged_tensor_to_sparse_eager_fallback(rt_nested_splits: Annotated[List[Any], TV_RaggedTensorToSparse_Tsplits], rt_dense_values: Annotated[Any, TV_RaggedTensorToSparse_T], name, ctx):
  if not isinstance(rt_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'rt_nested_splits' argument to "
        "'ragged_tensor_to_sparse' Op, not %r." % rt_nested_splits)
  _attr_RAGGED_RANK = len(rt_nested_splits)
  _attr_T, (rt_dense_values,) = _execute.args_to_matching_eager([rt_dense_values], ctx, [])
  _attr_Tsplits, rt_nested_splits = _execute.args_to_matching_eager(list(rt_nested_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = list(rt_nested_splits) + [rt_dense_values]
  _attrs = ("RAGGED_RANK", _attr_RAGGED_RANK, "T", _attr_T, "Tsplits",
  _attr_Tsplits)
  _result = _execute.execute(b"RaggedTensorToSparse", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedTensorToSparse", _inputs_flat, _attrs, _result)
  _result = _RaggedTensorToSparseOutput._make(_result)
  return _result


TV_RaggedTensorToTensor_T = TypeVar("TV_RaggedTensorToTensor_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedTensorToTensor_Tindex = TypeVar("TV_RaggedTensorToTensor_Tindex", _atypes.Int32, _atypes.Int64)
TV_RaggedTensorToTensor_Tshape = TypeVar("TV_RaggedTensorToTensor_Tshape", _atypes.Int32, _atypes.Int64)

def ragged_tensor_to_tensor(shape: Annotated[Any, TV_RaggedTensorToTensor_Tshape], values: Annotated[Any, TV_RaggedTensorToTensor_T], default_value: Annotated[Any, TV_RaggedTensorToTensor_T], row_partition_tensors: Annotated[List[Any], TV_RaggedTensorToTensor_Tindex], row_partition_types, name=None) -> Annotated[Any, TV_RaggedTensorToTensor_T]:
  r"""Create a dense tensor from a ragged tensor, possibly altering its shape.

  The `ragged_to_dense` op creates a dense tensor from a list of row partition
  tensors, a value vector, and default values. If the shape is unspecified, the
  minimal shape required to contain all the elements in the ragged tensor (the
  natural shape) will be used. If some dimensions are left unspecified, then the
  size of the natural shape is used in that dimension.

  The default_value will be broadcast to the output shape. After that, the values
  from the ragged tensor overwrite the default values. Note that the default_value
  must have less dimensions than the value.

  The row partition tensors are in the order of the dimensions.
  At present, the types can be:
  * "ROW_SPLITS": the row_splits tensor from the ragged tensor.
  * "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
  * "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
    is preceded by "FIRST_DIM_SIZE".

  Args:
    shape: A `Tensor`. Must be one of the following types: `int64`, `int32`.
      The desired shape of the output tensor. If left unspecified (empty),
      the minimal shape required to contain all the elements in the ragged tensor
      (the natural shape) will be used. If some dimensions are left unspecified, then
      the size of the natural shape is used in that dimension.

      Note that dense dimensions cannot be modified by the shape argument. Trying to
      change the size of a dense dimension will cause the op to fail.
      Examples:
      natural shape: [4, 5, 6]
      shape: -1
      output shape: [4, 5, 6]

      natural shape: [4, 5, 6]
      shape: [3, -1, 2]
      output shape: [3, 5, 2]

      natural shape: [4, 5, 6]
      shape: [3, 7, 2]
      output shape: [3, 7, 2]
    values: A `Tensor`.
      A 1D tensor representing the values of the ragged tensor.
    default_value: A `Tensor`. Must have the same type as `values`.
      The default_value when the shape is larger than the ragged tensor. The
      default_value is broadcast until it is the shape of the output tensor, and
      then overwritten by values in the ragged tensor. The default value must be
      compatible with this broadcast operation, and must have fewer dimensions than
      the value tensor.
    row_partition_tensors: A list of at least 1 `Tensor` objects with the same type in: `int64`, `int32`.
    row_partition_types: A list of `strings`.
      The types of the row partition tensors. At present, these can be:
      * "ROW_SPLITS": the row_splits tensor from the ragged tensor.
      * "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
      * "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
        is preceeded by "FIRST_DIM_SIZE".
      The tensors are in the order of the dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedTensorToTensor", name, shape, values, default_value,
        row_partition_tensors, "row_partition_types", row_partition_types)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_tensor_to_tensor_eager_fallback(
          shape, values, default_value, row_partition_tensors,
          row_partition_types=row_partition_types, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(row_partition_tensors, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_partition_tensors' argument to "
        "'ragged_tensor_to_tensor' Op, not %r." % row_partition_tensors)
  _attr_num_row_partition_tensors = len(row_partition_tensors)
  if not isinstance(row_partition_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_partition_types' argument to "
        "'ragged_tensor_to_tensor' Op, not %r." % row_partition_types)
  row_partition_types = [_execute.make_str(_s, "row_partition_types") for _s in row_partition_types]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedTensorToTensor", shape=shape, values=values,
                                default_value=default_value,
                                row_partition_tensors=row_partition_tensors,
                                row_partition_types=row_partition_types,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindex",
              _op._get_attr_type("Tindex"), "Tshape",
              _op._get_attr_type("Tshape"), "num_row_partition_tensors",
              _op._get_attr_int("num_row_partition_tensors"),
              "row_partition_types", _op.get_attr("row_partition_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedTensorToTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RaggedTensorToTensor = tf_export("raw_ops.RaggedTensorToTensor")(_ops.to_raw_op(ragged_tensor_to_tensor))


def ragged_tensor_to_tensor_eager_fallback(shape: Annotated[Any, TV_RaggedTensorToTensor_Tshape], values: Annotated[Any, TV_RaggedTensorToTensor_T], default_value: Annotated[Any, TV_RaggedTensorToTensor_T], row_partition_tensors: Annotated[List[Any], TV_RaggedTensorToTensor_Tindex], row_partition_types, name, ctx) -> Annotated[Any, TV_RaggedTensorToTensor_T]:
  if not isinstance(row_partition_tensors, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_partition_tensors' argument to "
        "'ragged_tensor_to_tensor' Op, not %r." % row_partition_tensors)
  _attr_num_row_partition_tensors = len(row_partition_tensors)
  if not isinstance(row_partition_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'row_partition_types' argument to "
        "'ragged_tensor_to_tensor' Op, not %r." % row_partition_types)
  row_partition_types = [_execute.make_str(_s, "row_partition_types") for _s in row_partition_types]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([values, default_value], ctx, [])
  (values, default_value) = _inputs_T
  _attr_Tindex, row_partition_tensors = _execute.args_to_matching_eager(list(row_partition_tensors), ctx, [_dtypes.int64, _dtypes.int32, ])
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int64, _dtypes.int32, ])
  _inputs_flat = [shape, values, default_value] + list(row_partition_tensors)
  _attrs = ("T", _attr_T, "Tindex", _attr_Tindex, "Tshape", _attr_Tshape,
  "num_row_partition_tensors", _attr_num_row_partition_tensors,
  "row_partition_types", row_partition_types)
  _result = _execute.execute(b"RaggedTensorToTensor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedTensorToTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RaggedTensorToVariant_Tvalues = TypeVar("TV_RaggedTensorToVariant_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedTensorToVariant_Tsplits = TypeVar("TV_RaggedTensorToVariant_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_tensor_to_variant(rt_nested_splits: Annotated[List[Any], TV_RaggedTensorToVariant_Tsplits], rt_dense_values: Annotated[Any, TV_RaggedTensorToVariant_Tvalues], batched_input: bool, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Encodes a `RaggedTensor` into a `variant` Tensor.

  
  Encodes the given `RaggedTensor` and returns a `variant` Tensor. If
  `batched_input` is True, then input `RaggedTensor` is unbatched along the
  zero-th dimension, each component `RaggedTensor` is encoded into a scalar
  `variant` Tensor, and these are stacked to return a 1-D `variant` Tensor.
  If `batched_input` is False, then the input `RaggedTensor` is encoded as is and
  a scalar `variant` Tensor is returned. A `RaggedTensor` is encoded by first
  creating a 1-D `variant` Tensor with `ragged_rank + 1` elements, containing the
  splits and values Tensors of the `RaggedTensor`. Then the 1-D `variant` Tensor
  is wrapped in a scalar `variant` Tensor. See `RaggedTensorFromVariant` for the
  corresponding decoding logic.

  Args:
    rt_nested_splits: A list of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of one or more Tensors representing the splits of the input
      `RaggedTensor`.
    rt_dense_values: A `Tensor`.
      A Tensor representing the values of the input `RaggedTensor`.
    batched_input: A `bool`.
      A `bool` denoting whether the input is a batched `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedTensorToVariant", name, rt_nested_splits,
        rt_dense_values, "batched_input", batched_input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_tensor_to_variant_eager_fallback(
          rt_nested_splits, rt_dense_values, batched_input=batched_input,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(rt_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'rt_nested_splits' argument to "
        "'ragged_tensor_to_variant' Op, not %r." % rt_nested_splits)
  _attr_RAGGED_RANK = len(rt_nested_splits)
  batched_input = _execute.make_bool(batched_input, "batched_input")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedTensorToVariant", rt_nested_splits=rt_nested_splits,
                                 rt_dense_values=rt_dense_values,
                                 batched_input=batched_input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("RAGGED_RANK", _op._get_attr_int("RAGGED_RANK"), "Tvalues",
              _op._get_attr_type("Tvalues"), "Tsplits",
              _op._get_attr_type("Tsplits"), "batched_input",
              _op._get_attr_bool("batched_input"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedTensorToVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RaggedTensorToVariant = tf_export("raw_ops.RaggedTensorToVariant")(_ops.to_raw_op(ragged_tensor_to_variant))


def ragged_tensor_to_variant_eager_fallback(rt_nested_splits: Annotated[List[Any], TV_RaggedTensorToVariant_Tsplits], rt_dense_values: Annotated[Any, TV_RaggedTensorToVariant_Tvalues], batched_input: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(rt_nested_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'rt_nested_splits' argument to "
        "'ragged_tensor_to_variant' Op, not %r." % rt_nested_splits)
  _attr_RAGGED_RANK = len(rt_nested_splits)
  batched_input = _execute.make_bool(batched_input, "batched_input")
  _attr_Tvalues, (rt_dense_values,) = _execute.args_to_matching_eager([rt_dense_values], ctx, [])
  _attr_Tsplits, rt_nested_splits = _execute.args_to_matching_eager(list(rt_nested_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = list(rt_nested_splits) + [rt_dense_values]
  _attrs = ("RAGGED_RANK", _attr_RAGGED_RANK, "Tvalues", _attr_Tvalues,
  "Tsplits", _attr_Tsplits, "batched_input", batched_input)
  _result = _execute.execute(b"RaggedTensorToVariant", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedTensorToVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RaggedTensorToVariantGradient_Tvalues = TypeVar("TV_RaggedTensorToVariantGradient_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RaggedTensorToVariantGradient_Tsplits = TypeVar("TV_RaggedTensorToVariantGradient_Tsplits", _atypes.Int32, _atypes.Int64)

def ragged_tensor_to_variant_gradient(encoded_ragged_grad: Annotated[Any, _atypes.Variant], row_splits: Annotated[Any, TV_RaggedTensorToVariantGradient_Tsplits], dense_values_shape: Annotated[Any, _atypes.Int32], Tvalues: TV_RaggedTensorToVariantGradient_Tvalues, name=None) -> Annotated[Any, TV_RaggedTensorToVariantGradient_Tvalues]:
  r"""Helper used to compute the gradient for `RaggedTensorToVariant`.

  Computes the gradient for the dense_values input to the RaggedTensorToVariant
  op, given the variant-encoded ragged gradients of the outputs, along with
  the outer row-splits and the shape of the dense-values that were provided as
  inputs to the RaggedTensorToVariant op.

  Args:
    encoded_ragged_grad: A `Tensor` of type `variant`.
      A `variant` Tensor containing encoded `RaggedTensor` gradients.
    row_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Outermost row-splits that were used as input to the RaggedTensorToVariant op.
    dense_values_shape: A `Tensor` of type `int32`.
      Shape of the dense_values that was used as an input to the
      RaggedTensorToVariant op.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tvalues`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedTensorToVariantGradient", name, encoded_ragged_grad,
        row_splits, dense_values_shape, "Tvalues", Tvalues)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_tensor_to_variant_gradient_eager_fallback(
          encoded_ragged_grad, row_splits, dense_values_shape,
          Tvalues=Tvalues, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedTensorToVariantGradient", encoded_ragged_grad=encoded_ragged_grad,
                                         row_splits=row_splits,
                                         dense_values_shape=dense_values_shape,
                                         Tvalues=Tvalues, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tvalues", _op._get_attr_type("Tvalues"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedTensorToVariantGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RaggedTensorToVariantGradient = tf_export("raw_ops.RaggedTensorToVariantGradient")(_ops.to_raw_op(ragged_tensor_to_variant_gradient))


def ragged_tensor_to_variant_gradient_eager_fallback(encoded_ragged_grad: Annotated[Any, _atypes.Variant], row_splits: Annotated[Any, TV_RaggedTensorToVariantGradient_Tsplits], dense_values_shape: Annotated[Any, _atypes.Int32], Tvalues: TV_RaggedTensorToVariantGradient_Tvalues, name, ctx) -> Annotated[Any, TV_RaggedTensorToVariantGradient_Tvalues]:
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  _attr_Tsplits, (row_splits,) = _execute.args_to_matching_eager([row_splits], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  encoded_ragged_grad = _ops.convert_to_tensor(encoded_ragged_grad, _dtypes.variant)
  dense_values_shape = _ops.convert_to_tensor(dense_values_shape, _dtypes.int32)
  _inputs_flat = [encoded_ragged_grad, row_splits, dense_values_shape]
  _attrs = ("Tvalues", Tvalues, "Tsplits", _attr_Tsplits)
  _result = _execute.execute(b"RaggedTensorToVariantGradient", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedTensorToVariantGradient", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

