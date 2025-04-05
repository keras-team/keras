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

def decode_csv(records: Annotated[Any, _atypes.String], record_defaults, field_delim:str=",", use_quote_delim:bool=True, na_value:str="", select_cols=[], name=None):
  r"""Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `float64`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or an empty vector if the column is
      required.
    field_delim: An optional `string`. Defaults to `","`.
      char delimiter to separate fields in a record.
    use_quote_delim: An optional `bool`. Defaults to `True`.
      If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).
    na_value: An optional `string`. Defaults to `""`.
      Additional string to recognize as NA/NaN.
    select_cols: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeCSV", name, records, record_defaults, "field_delim",
        field_delim, "use_quote_delim", use_quote_delim, "na_value", na_value,
        "select_cols", select_cols)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_csv_eager_fallback(
          records, record_defaults, field_delim=field_delim,
          use_quote_delim=use_quote_delim, na_value=na_value,
          select_cols=select_cols, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if field_delim is None:
    field_delim = ","
  field_delim = _execute.make_str(field_delim, "field_delim")
  if use_quote_delim is None:
    use_quote_delim = True
  use_quote_delim = _execute.make_bool(use_quote_delim, "use_quote_delim")
  if na_value is None:
    na_value = ""
  na_value = _execute.make_str(na_value, "na_value")
  if select_cols is None:
    select_cols = []
  if not isinstance(select_cols, (list, tuple)):
    raise TypeError(
        "Expected list for 'select_cols' argument to "
        "'decode_csv' Op, not %r." % select_cols)
  select_cols = [_execute.make_int(_i, "select_cols") for _i in select_cols]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeCSV", records=records, record_defaults=record_defaults,
                     field_delim=field_delim, use_quote_delim=use_quote_delim,
                     na_value=na_value, select_cols=select_cols, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("OUT_TYPE", _op.get_attr("OUT_TYPE"), "field_delim",
              _op.get_attr("field_delim"), "use_quote_delim",
              _op._get_attr_bool("use_quote_delim"), "na_value",
              _op.get_attr("na_value"), "select_cols",
              _op.get_attr("select_cols"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeCSV", _inputs_flat, _attrs, _result)
  return _result

DecodeCSV = tf_export("raw_ops.DecodeCSV")(_ops.to_raw_op(decode_csv))


def decode_csv_eager_fallback(records: Annotated[Any, _atypes.String], record_defaults, field_delim: str, use_quote_delim: bool, na_value: str, select_cols, name, ctx):
  if field_delim is None:
    field_delim = ","
  field_delim = _execute.make_str(field_delim, "field_delim")
  if use_quote_delim is None:
    use_quote_delim = True
  use_quote_delim = _execute.make_bool(use_quote_delim, "use_quote_delim")
  if na_value is None:
    na_value = ""
  na_value = _execute.make_str(na_value, "na_value")
  if select_cols is None:
    select_cols = []
  if not isinstance(select_cols, (list, tuple)):
    raise TypeError(
        "Expected list for 'select_cols' argument to "
        "'decode_csv' Op, not %r." % select_cols)
  select_cols = [_execute.make_int(_i, "select_cols") for _i in select_cols]
  _attr_OUT_TYPE, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
  records = _ops.convert_to_tensor(records, _dtypes.string)
  _inputs_flat = [records] + list(record_defaults)
  _attrs = ("OUT_TYPE", _attr_OUT_TYPE, "field_delim", field_delim,
  "use_quote_delim", use_quote_delim, "na_value", na_value, "select_cols",
  select_cols)
  _result = _execute.execute(b"DecodeCSV", len(record_defaults),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeCSV", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.decode_compressed', v1=['io.decode_compressed', 'decode_compressed'])
@deprecated_endpoints('decode_compressed')
def decode_compressed(bytes: Annotated[Any, _atypes.String], compression_type:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Decompress strings.

  This op decompresses each element of the `bytes` input `Tensor`, which
  is assumed to be compressed using the given `compression_type`.

  The `output` is a string `Tensor` of the same shape as `bytes`,
  each element containing the decompressed data from the corresponding
  element in `bytes`.

  Args:
    bytes: A `Tensor` of type `string`.
      A Tensor of string which is compressed.
    compression_type: An optional `string`. Defaults to `""`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeCompressed", name, bytes, "compression_type",
        compression_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_decode_compressed(
          (bytes, compression_type, name,), None)
      if _result is not NotImplemented:
        return _result
      return decode_compressed_eager_fallback(
          bytes, compression_type=compression_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            decode_compressed, (), dict(bytes=bytes,
                                        compression_type=compression_type,
                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_decode_compressed(
        (bytes, compression_type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeCompressed", bytes=bytes, compression_type=compression_type,
                            name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          decode_compressed, (), dict(bytes=bytes,
                                      compression_type=compression_type,
                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("compression_type", _op.get_attr("compression_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeCompressed", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeCompressed = tf_export("raw_ops.DecodeCompressed")(_ops.to_raw_op(decode_compressed))
_dispatcher_for_decode_compressed = decode_compressed._tf_type_based_dispatcher.Dispatch


def decode_compressed_eager_fallback(bytes: Annotated[Any, _atypes.String], compression_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if compression_type is None:
    compression_type = ""
  compression_type = _execute.make_str(compression_type, "compression_type")
  bytes = _ops.convert_to_tensor(bytes, _dtypes.string)
  _inputs_flat = [bytes]
  _attrs = ("compression_type", compression_type)
  _result = _execute.execute(b"DecodeCompressed", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeCompressed", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def decode_json_example(json_examples: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Convert JSON-encoded Example records to binary protocol buffer strings.

  
  Note: This is **not** a general purpose JSON parsing op.

  This op converts JSON-serialized
  `tf.train.Example` (created with `json_format.MessageToJson`, following the
  [standard JSON mapping](https://developers.google.com/protocol-buffers/docs/proto3#json))
  to a binary-serialized `tf.train.Example` (equivalent to
  `Example.SerializeToString()`) suitable for conversion to tensors with
  `tf.io.parse_example`.

  Args:
    json_examples: A `Tensor` of type `string`.
      Each string is a JSON object serialized according to the JSON
      mapping of the Example proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeJSONExample", name, json_examples)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_json_example_eager_fallback(
          json_examples, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeJSONExample", json_examples=json_examples, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeJSONExample", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeJSONExample = tf_export("raw_ops.DecodeJSONExample")(_ops.to_raw_op(decode_json_example))


def decode_json_example_eager_fallback(json_examples: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  json_examples = _ops.convert_to_tensor(json_examples, _dtypes.string)
  _inputs_flat = [json_examples]
  _attrs = None
  _result = _execute.execute(b"DecodeJSONExample", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeJSONExample", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DecodePaddedRaw_out_type = TypeVar("TV_DecodePaddedRaw_out_type", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def decode_padded_raw(input_bytes: Annotated[Any, _atypes.String], fixed_length: Annotated[Any, _atypes.Int32], out_type: TV_DecodePaddedRaw_out_type, little_endian:bool=True, name=None) -> Annotated[Any, TV_DecodePaddedRaw_out_type]:
  r"""Reinterpret the bytes of a string as a vector of numbers.

  Args:
    input_bytes: A `Tensor` of type `string`. Tensor of string to be decoded.
    fixed_length: A `Tensor` of type `int32`.
      Length in bytes for each element of the decoded output. Must be a multiple
      of the size of the output type.
    out_type: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64, tf.bfloat16`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input `input_bytes` is in little-endian order. Ignored for
      `out_type` values that are stored in a single byte, like `uint8`
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodePaddedRaw", name, input_bytes, fixed_length, "out_type",
        out_type, "little_endian", little_endian)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_padded_raw_eager_fallback(
          input_bytes, fixed_length, out_type=out_type,
          little_endian=little_endian, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  if little_endian is None:
    little_endian = True
  little_endian = _execute.make_bool(little_endian, "little_endian")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodePaddedRaw", input_bytes=input_bytes, fixed_length=fixed_length,
                           out_type=out_type, little_endian=little_endian,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("out_type", _op._get_attr_type("out_type"), "little_endian",
              _op._get_attr_bool("little_endian"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodePaddedRaw", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodePaddedRaw = tf_export("raw_ops.DecodePaddedRaw")(_ops.to_raw_op(decode_padded_raw))


def decode_padded_raw_eager_fallback(input_bytes: Annotated[Any, _atypes.String], fixed_length: Annotated[Any, _atypes.Int32], out_type: TV_DecodePaddedRaw_out_type, little_endian: bool, name, ctx) -> Annotated[Any, TV_DecodePaddedRaw_out_type]:
  out_type = _execute.make_type(out_type, "out_type")
  if little_endian is None:
    little_endian = True
  little_endian = _execute.make_bool(little_endian, "little_endian")
  input_bytes = _ops.convert_to_tensor(input_bytes, _dtypes.string)
  fixed_length = _ops.convert_to_tensor(fixed_length, _dtypes.int32)
  _inputs_flat = [input_bytes, fixed_length]
  _attrs = ("out_type", out_type, "little_endian", little_endian)
  _result = _execute.execute(b"DecodePaddedRaw", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodePaddedRaw", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DecodeRaw_out_type = TypeVar("TV_DecodeRaw_out_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def decode_raw(bytes: Annotated[Any, _atypes.String], out_type: TV_DecodeRaw_out_type, little_endian:bool=True, name=None) -> Annotated[Any, TV_DecodeRaw_out_type]:
  r"""Reinterpret the bytes of a string as a vector of numbers.

  Args:
    bytes: A `Tensor` of type `string`.
      All the elements must have the same length.
    out_type: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64, tf.complex64, tf.complex128, tf.bool, tf.bfloat16`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input `bytes` are in little-endian order.
      Ignored for `out_type` values that are stored in a single byte like
      `uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeRaw", name, bytes, "out_type", out_type, "little_endian",
        little_endian)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_raw_eager_fallback(
          bytes, out_type=out_type, little_endian=little_endian, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  if little_endian is None:
    little_endian = True
  little_endian = _execute.make_bool(little_endian, "little_endian")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeRaw", bytes=bytes, out_type=out_type,
                     little_endian=little_endian, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("out_type", _op._get_attr_type("out_type"), "little_endian",
              _op._get_attr_bool("little_endian"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeRaw", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeRaw = tf_export("raw_ops.DecodeRaw")(_ops.to_raw_op(decode_raw))


def decode_raw_eager_fallback(bytes: Annotated[Any, _atypes.String], out_type: TV_DecodeRaw_out_type, little_endian: bool, name, ctx) -> Annotated[Any, TV_DecodeRaw_out_type]:
  out_type = _execute.make_type(out_type, "out_type")
  if little_endian is None:
    little_endian = True
  little_endian = _execute.make_bool(little_endian, "little_endian")
  bytes = _ops.convert_to_tensor(bytes, _dtypes.string)
  _inputs_flat = [bytes]
  _attrs = ("out_type", out_type, "little_endian", little_endian)
  _result = _execute.execute(b"DecodeRaw", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeRaw", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_ParseExampleOutput = collections.namedtuple(
    "ParseExample",
    ["sparse_indices", "sparse_values", "sparse_shapes", "dense_values"])


def parse_example(serialized: Annotated[Any, _atypes.String], names: Annotated[Any, _atypes.String], sparse_keys: Annotated[List[Any], _atypes.String], dense_keys: Annotated[List[Any], _atypes.String], dense_defaults, sparse_types, dense_shapes, name=None):
  r"""Transforms a vector of brain.Example protos (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A vector containing a batch of binary serialized Example protos.
    names: A `Tensor` of type `string`.
      A vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Nsparse string Tensors (scalars).
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A list of `Tensor` objects with type `string`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples' features associated with dense values.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ndense Tensors (some may be empty).
      dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
      then the shape of dense_defaults[j] must match that of dense_shapes[j].
      If dense_shapes[j] has an undefined major dimension (variable strides dense
      feature), dense_defaults[j] must contain a single element:
      the padding element.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of Nsparse types; the data types of data in each Feature
      given in sparse_keys.
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      A list of Ndense shapes; the shapes of data in each Feature
      given in dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always equal dense_shapes[j].NumEntries().
      If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
      This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
      the shape of the output Tensor dense_values[j] will be
      (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
      of elements of length D1 * .... * DN, across all minibatch entries
      in the input.  Any minibatch entry with less than M blocks of elements of
      length D1 * ... * DN will be padded with the corresponding default_value
      scalar element along the second dimension.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values).

    sparse_indices: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseExample", name, serialized, names, sparse_keys,
        dense_keys, dense_defaults, "sparse_types", sparse_types,
        "dense_shapes", dense_shapes)
      _result = _ParseExampleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_example_eager_fallback(
          serialized, names, sparse_keys, dense_keys, dense_defaults,
          sparse_types=sparse_types, dense_shapes=dense_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example' Op, not %r." % sparse_keys)
  _attr_Nsparse = len(sparse_keys)
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example' Op, not %r." % dense_keys)
  _attr_Ndense = len(dense_keys)
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseExample", serialized=serialized, names=names,
                        sparse_keys=sparse_keys, dense_keys=dense_keys,
                        dense_defaults=dense_defaults,
                        sparse_types=sparse_types, dense_shapes=dense_shapes,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Nsparse", _op._get_attr_int("Nsparse"), "Ndense",
              _op._get_attr_int("Ndense"), "sparse_types",
              _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
              "dense_shapes", _op.get_attr("dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseExample", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_Nsparse]] + _result[_attr_Nsparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + _attr_Nsparse]] + _result[2 + _attr_Nsparse:]
  _result = _result[:3] + [_result[3:]]
  _result = _ParseExampleOutput._make(_result)
  return _result

ParseExample = tf_export("raw_ops.ParseExample")(_ops.to_raw_op(parse_example))


def parse_example_eager_fallback(serialized: Annotated[Any, _atypes.String], names: Annotated[Any, _atypes.String], sparse_keys: Annotated[List[Any], _atypes.String], dense_keys: Annotated[List[Any], _atypes.String], dense_defaults, sparse_types, dense_shapes, name, ctx):
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_example' Op, not %r." % sparse_keys)
  _attr_Nsparse = len(sparse_keys)
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_example' Op, not %r." % dense_keys)
  _attr_Ndense = len(dense_keys)
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  names = _ops.convert_to_tensor(names, _dtypes.string)
  sparse_keys = _ops.convert_n_to_tensor(sparse_keys, _dtypes.string)
  dense_keys = _ops.convert_n_to_tensor(dense_keys, _dtypes.string)
  _inputs_flat = [serialized, names] + list(sparse_keys) + list(dense_keys) + list(dense_defaults)
  _attrs = ("Nsparse", _attr_Nsparse, "Ndense", _attr_Ndense, "sparse_types",
  sparse_types, "Tdense", _attr_Tdense, "dense_shapes", dense_shapes)
  _result = _execute.execute(b"ParseExample", _attr_Nsparse +
                             len(sparse_types) + _attr_Nsparse +
                             len(dense_defaults), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseExample", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_Nsparse]] + _result[_attr_Nsparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + _attr_Nsparse]] + _result[2 + _attr_Nsparse:]
  _result = _result[:3] + [_result[3:]]
  _result = _ParseExampleOutput._make(_result)
  return _result

_ParseExampleV2Output = collections.namedtuple(
    "ParseExampleV2",
    ["sparse_indices", "sparse_values", "sparse_shapes", "dense_values", "ragged_values", "ragged_row_splits"])


def parse_example_v2(serialized: Annotated[Any, _atypes.String], names: Annotated[Any, _atypes.String], sparse_keys: Annotated[Any, _atypes.String], dense_keys: Annotated[Any, _atypes.String], ragged_keys: Annotated[Any, _atypes.String], dense_defaults, num_sparse: int, sparse_types, ragged_value_types, ragged_split_types, dense_shapes, name=None):
  r"""Transforms a vector of tf.Example protos (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar or vector containing binary serialized Example protos.
    names: A `Tensor` of type `string`.
      A tensor containing the names of the serialized protos.
      Corresponds 1:1 with the `serialized` tensor.
      May contain, for example, table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no names are available.
      If non-empty, this tensor must have the same shape as "serialized".
    sparse_keys: A `Tensor` of type `string`. Vector of strings.
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A `Tensor` of type `string`. Vector of strings.
      The keys expected in the Examples' features associated with dense values.
    ragged_keys: A `Tensor` of type `string`. Vector of strings.
      The keys expected in the Examples' features associated with ragged values.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Tensors (some may be empty).  Corresponds 1:1 with `dense_keys`.
      dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
      then the shape of dense_defaults[j] must match that of dense_shapes[j].
      If dense_shapes[j] has an undefined major dimension (variable strides dense
      feature), dense_defaults[j] must contain a single element:
      the padding element.
    num_sparse: An `int` that is `>= 0`. The number of sparse keys.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `num_sparse` types; the data types of data in each Feature
      given in sparse_keys.
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    ragged_value_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `num_ragged` types; the data types of data in each Feature
      given in ragged_keys (where `num_ragged = sparse_keys.size()`).
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    ragged_split_types: A list of `tf.DTypes` from: `tf.int32, tf.int64`.
      A list of `num_ragged` types; the data types of row_splits in each Feature
      given in ragged_keys (where `num_ragged = sparse_keys.size()`).
      May be DT_INT32 or DT_INT64.
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      A list of `num_dense` shapes; the shapes of data in each Feature
      given in dense_keys (where `num_dense = dense_keys.size()`).
      The number of elements in the Feature corresponding to dense_key[j]
      must always equal dense_shapes[j].NumEntries().
      If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
      This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
      the shape of the output Tensor dense_values[j] will be
      (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
      of elements of length D1 * .... * DN, across all minibatch entries
      in the input.  Any minibatch entry with less than M blocks of elements of
      length D1 * ... * DN will be padded with the corresponding default_value
      scalar element along the second dimension.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values, ragged_values, ragged_row_splits).

    sparse_indices: A list of `num_sparse` `Tensor` objects with type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list of `num_sparse` `Tensor` objects with type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
    ragged_values: A list of `Tensor` objects of type `ragged_value_types`.
    ragged_row_splits: A list of `Tensor` objects of type `ragged_split_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseExampleV2", name, serialized, names, sparse_keys,
        dense_keys, ragged_keys, dense_defaults, "num_sparse", num_sparse,
        "sparse_types", sparse_types, "ragged_value_types",
        ragged_value_types, "ragged_split_types", ragged_split_types,
        "dense_shapes", dense_shapes)
      _result = _ParseExampleV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_example_v2_eager_fallback(
          serialized, names, sparse_keys, dense_keys, ragged_keys,
          dense_defaults, num_sparse=num_sparse, sparse_types=sparse_types,
          ragged_value_types=ragged_value_types,
          ragged_split_types=ragged_split_types, dense_shapes=dense_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_sparse = _execute.make_int(num_sparse, "num_sparse")
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_v2' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_v2' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_v2' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_v2' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseExampleV2", serialized=serialized, names=names,
                          sparse_keys=sparse_keys, dense_keys=dense_keys,
                          ragged_keys=ragged_keys,
                          dense_defaults=dense_defaults,
                          num_sparse=num_sparse, sparse_types=sparse_types,
                          ragged_value_types=ragged_value_types,
                          ragged_split_types=ragged_split_types,
                          dense_shapes=dense_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tdense", _op.get_attr("Tdense"), "num_sparse",
              _op._get_attr_int("num_sparse"), "sparse_types",
              _op.get_attr("sparse_types"), "ragged_value_types",
              _op.get_attr("ragged_value_types"), "ragged_split_types",
              _op.get_attr("ragged_split_types"), "dense_shapes",
              _op.get_attr("dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseExampleV2", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sparse]] + _result[num_sparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
  _result = _result[:3] + [_result[3:3 + len(dense_defaults)]] + _result[3 + len(dense_defaults):]
  _result = _result[:4] + [_result[4:4 + len(ragged_value_types)]] + _result[4 + len(ragged_value_types):]
  _result = _result[:5] + [_result[5:]]
  _result = _ParseExampleV2Output._make(_result)
  return _result

ParseExampleV2 = tf_export("raw_ops.ParseExampleV2")(_ops.to_raw_op(parse_example_v2))


def parse_example_v2_eager_fallback(serialized: Annotated[Any, _atypes.String], names: Annotated[Any, _atypes.String], sparse_keys: Annotated[Any, _atypes.String], dense_keys: Annotated[Any, _atypes.String], ragged_keys: Annotated[Any, _atypes.String], dense_defaults, num_sparse: int, sparse_types, ragged_value_types, ragged_split_types, dense_shapes, name, ctx):
  num_sparse = _execute.make_int(num_sparse, "num_sparse")
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_example_v2' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_value_types' argument to "
        "'parse_example_v2' Op, not %r." % ragged_value_types)
  ragged_value_types = [_execute.make_type(_t, "ragged_value_types") for _t in ragged_value_types]
  if not isinstance(ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'ragged_split_types' argument to "
        "'parse_example_v2' Op, not %r." % ragged_split_types)
  ragged_split_types = [_execute.make_type(_t, "ragged_split_types") for _t in ragged_split_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_example_v2' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  names = _ops.convert_to_tensor(names, _dtypes.string)
  sparse_keys = _ops.convert_to_tensor(sparse_keys, _dtypes.string)
  dense_keys = _ops.convert_to_tensor(dense_keys, _dtypes.string)
  ragged_keys = _ops.convert_to_tensor(ragged_keys, _dtypes.string)
  _inputs_flat = [serialized, names, sparse_keys, dense_keys, ragged_keys] + list(dense_defaults)
  _attrs = ("Tdense", _attr_Tdense, "num_sparse", num_sparse, "sparse_types",
  sparse_types, "ragged_value_types", ragged_value_types,
  "ragged_split_types", ragged_split_types, "dense_shapes", dense_shapes)
  _result = _execute.execute(b"ParseExampleV2", num_sparse + len(sparse_types)
                             + num_sparse + len(dense_defaults) +
                             len(ragged_value_types) +
                             len(ragged_split_types), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseExampleV2", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sparse]] + _result[num_sparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
  _result = _result[:3] + [_result[3:3 + len(dense_defaults)]] + _result[3 + len(dense_defaults):]
  _result = _result[:4] + [_result[4:4 + len(ragged_value_types)]] + _result[4 + len(ragged_value_types):]
  _result = _result[:5] + [_result[5:]]
  _result = _ParseExampleV2Output._make(_result)
  return _result

_ParseSequenceExampleOutput = collections.namedtuple(
    "ParseSequenceExample",
    ["context_sparse_indices", "context_sparse_values", "context_sparse_shapes", "context_dense_values", "feature_list_sparse_indices", "feature_list_sparse_values", "feature_list_sparse_shapes", "feature_list_dense_values", "feature_list_dense_lengths"])


def parse_sequence_example(serialized: Annotated[Any, _atypes.String], debug_name: Annotated[Any, _atypes.String], context_dense_defaults, feature_list_dense_missing_assumed_empty, context_sparse_keys, context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys, Ncontext_sparse:int=0, Ncontext_dense:int=0, Nfeature_list_sparse:int=0, Nfeature_list_dense:int=0, context_sparse_types=[], feature_list_dense_types=[], context_dense_shapes=[], feature_list_sparse_types=[], feature_list_dense_shapes=[], name=None):
  r"""Transforms a vector of brain.SequenceExample protos (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A vector containing binary serialized SequenceExample protos.
    debug_name: A `Tensor` of type `string`.
      A vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) name for the
      corresponding serialized proto.  This is purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no name is available.
    context_dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ncontext_dense Tensors (some may be empty).
      context_dense_defaults[j] provides default values
      when the SequenceExample's context map lacks context_dense_key[j].
      If an empty Tensor is provided for context_dense_defaults[j],
      then the Feature context_dense_keys[j] is required.
      The input type is inferred from context_dense_defaults[j], even when it's
      empty.  If context_dense_defaults[j] is not empty, its shape must match
      context_dense_shapes[j].
    feature_list_dense_missing_assumed_empty: A list of `strings`.
      A vector listing the
      FeatureList keys which may be missing from the SequenceExamples.  If the
      associated FeatureList is missing, it is treated as empty.  By default,
      any FeatureList not listed in this vector must exist in the SequenceExamples.
    context_sparse_keys: A list of `strings`.
      A list of Ncontext_sparse string Tensors (scalars).
      The keys expected in the Examples' features associated with context_sparse
      values.
    context_dense_keys: A list of `strings`.
      A list of Ncontext_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' context features associated with
      dense values.
    feature_list_sparse_keys: A list of `strings`.
      A list of Nfeature_list_sparse string Tensors
      (scalars).  The keys expected in the FeatureLists associated with sparse
      values.
    feature_list_dense_keys: A list of `strings`.
      A list of Nfeature_list_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' feature_lists associated
      with lists of dense values.
    Ncontext_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    Ncontext_dense: An optional `int` that is `>= 0`. Defaults to `0`.
    Nfeature_list_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    Nfeature_list_dense: An optional `int` that is `>= 0`. Defaults to `0`.
    context_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Ncontext_sparse types; the data types of data in
      each context Feature given in context_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    context_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Ncontext_dense shapes; the shapes of data in
      each context Feature given in context_dense_keys.
      The number of elements in the Feature corresponding to context_dense_key[j]
      must always equal context_dense_shapes[j].NumEntries().
      The shape of context_dense_values[j] will match context_dense_shapes[j].
    feature_list_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Nfeature_list_sparse types; the data types
      of data in each FeatureList given in feature_list_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Nfeature_list_dense shapes; the shapes of
      data in each FeatureList given in feature_list_dense_keys.
      The shape of each Feature in the FeatureList corresponding to
      feature_list_dense_key[j] must always equal
      feature_list_dense_shapes[j].NumEntries().
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values, feature_list_dense_lengths).

    context_sparse_indices: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_sparse_values: A list of `Tensor` objects of type `context_sparse_types`.
    context_sparse_shapes: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_dense_values: A list of `Tensor` objects. Has the same type as `context_dense_defaults`.
    feature_list_sparse_indices: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_sparse_values: A list of `Tensor` objects of type `feature_list_sparse_types`.
    feature_list_sparse_shapes: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_dense_values: A list of `Tensor` objects of type `feature_list_dense_types`.
    feature_list_dense_lengths: A list of `Nfeature_list_dense` `Tensor` objects with type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseSequenceExample", name, serialized, debug_name,
        context_dense_defaults, "feature_list_dense_missing_assumed_empty",
        feature_list_dense_missing_assumed_empty, "context_sparse_keys",
        context_sparse_keys, "context_dense_keys", context_dense_keys,
        "feature_list_sparse_keys", feature_list_sparse_keys,
        "feature_list_dense_keys", feature_list_dense_keys, "Ncontext_sparse",
        Ncontext_sparse, "Ncontext_dense", Ncontext_dense,
        "Nfeature_list_sparse", Nfeature_list_sparse, "Nfeature_list_dense",
        Nfeature_list_dense, "context_sparse_types", context_sparse_types,
        "feature_list_dense_types", feature_list_dense_types,
        "context_dense_shapes", context_dense_shapes,
        "feature_list_sparse_types", feature_list_sparse_types,
        "feature_list_dense_shapes", feature_list_dense_shapes)
      _result = _ParseSequenceExampleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_sequence_example_eager_fallback(
          serialized, debug_name, context_dense_defaults,
          feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty,
          context_sparse_keys=context_sparse_keys,
          context_dense_keys=context_dense_keys,
          feature_list_sparse_keys=feature_list_sparse_keys,
          feature_list_dense_keys=feature_list_dense_keys,
          Ncontext_sparse=Ncontext_sparse, Ncontext_dense=Ncontext_dense,
          Nfeature_list_sparse=Nfeature_list_sparse,
          Nfeature_list_dense=Nfeature_list_dense,
          context_sparse_types=context_sparse_types,
          feature_list_dense_types=feature_list_dense_types,
          context_dense_shapes=context_dense_shapes,
          feature_list_sparse_types=feature_list_sparse_types,
          feature_list_dense_shapes=feature_list_dense_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(feature_list_dense_missing_assumed_empty, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_missing_assumed_empty' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_missing_assumed_empty)
  feature_list_dense_missing_assumed_empty = [_execute.make_str(_s, "feature_list_dense_missing_assumed_empty") for _s in feature_list_dense_missing_assumed_empty]
  if not isinstance(context_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_keys' argument to "
        "'parse_sequence_example' Op, not %r." % context_sparse_keys)
  context_sparse_keys = [_execute.make_str(_s, "context_sparse_keys") for _s in context_sparse_keys]
  if not isinstance(context_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_keys' argument to "
        "'parse_sequence_example' Op, not %r." % context_dense_keys)
  context_dense_keys = [_execute.make_str(_s, "context_dense_keys") for _s in context_dense_keys]
  if not isinstance(feature_list_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_keys' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_sparse_keys)
  feature_list_sparse_keys = [_execute.make_str(_s, "feature_list_sparse_keys") for _s in feature_list_sparse_keys]
  if not isinstance(feature_list_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_keys' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_keys)
  feature_list_dense_keys = [_execute.make_str(_s, "feature_list_dense_keys") for _s in feature_list_dense_keys]
  if Ncontext_sparse is None:
    Ncontext_sparse = 0
  Ncontext_sparse = _execute.make_int(Ncontext_sparse, "Ncontext_sparse")
  if Ncontext_dense is None:
    Ncontext_dense = 0
  Ncontext_dense = _execute.make_int(Ncontext_dense, "Ncontext_dense")
  if Nfeature_list_sparse is None:
    Nfeature_list_sparse = 0
  Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, "Nfeature_list_sparse")
  if Nfeature_list_dense is None:
    Nfeature_list_dense = 0
  Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, "Nfeature_list_dense")
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_sequence_example' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_sequence_example' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseSequenceExample", serialized=serialized, debug_name=debug_name,
                                context_dense_defaults=context_dense_defaults,
                                feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty,
                                context_sparse_keys=context_sparse_keys,
                                context_dense_keys=context_dense_keys,
                                feature_list_sparse_keys=feature_list_sparse_keys,
                                feature_list_dense_keys=feature_list_dense_keys,
                                Ncontext_sparse=Ncontext_sparse,
                                Ncontext_dense=Ncontext_dense,
                                Nfeature_list_sparse=Nfeature_list_sparse,
                                Nfeature_list_dense=Nfeature_list_dense,
                                context_sparse_types=context_sparse_types,
                                feature_list_dense_types=feature_list_dense_types,
                                context_dense_shapes=context_dense_shapes,
                                feature_list_sparse_types=feature_list_sparse_types,
                                feature_list_dense_shapes=feature_list_dense_shapes,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("feature_list_dense_missing_assumed_empty",
              _op.get_attr("feature_list_dense_missing_assumed_empty"),
              "context_sparse_keys", _op.get_attr("context_sparse_keys"),
              "context_dense_keys", _op.get_attr("context_dense_keys"),
              "feature_list_sparse_keys",
              _op.get_attr("feature_list_sparse_keys"),
              "feature_list_dense_keys",
              _op.get_attr("feature_list_dense_keys"), "Ncontext_sparse",
              _op._get_attr_int("Ncontext_sparse"), "Ncontext_dense",
              _op._get_attr_int("Ncontext_dense"), "Nfeature_list_sparse",
              _op._get_attr_int("Nfeature_list_sparse"),
              "Nfeature_list_dense", _op._get_attr_int("Nfeature_list_dense"),
              "context_sparse_types", _op.get_attr("context_sparse_types"),
              "Tcontext_dense", _op.get_attr("Tcontext_dense"),
              "feature_list_dense_types",
              _op.get_attr("feature_list_dense_types"),
              "context_dense_shapes", _op.get_attr("context_dense_shapes"),
              "feature_list_sparse_types",
              _op.get_attr("feature_list_sparse_types"),
              "feature_list_dense_shapes",
              _op.get_attr("feature_list_dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseSequenceExample", _inputs_flat, _attrs, _result)
  _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + Nfeature_list_sparse]] + _result[4 + Nfeature_list_sparse:]
  _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
  _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:7 + len(feature_list_dense_types)]] + _result[7 + len(feature_list_dense_types):]
  _result = _result[:8] + [_result[8:]]
  _result = _ParseSequenceExampleOutput._make(_result)
  return _result

ParseSequenceExample = tf_export("raw_ops.ParseSequenceExample")(_ops.to_raw_op(parse_sequence_example))


def parse_sequence_example_eager_fallback(serialized: Annotated[Any, _atypes.String], debug_name: Annotated[Any, _atypes.String], context_dense_defaults, feature_list_dense_missing_assumed_empty, context_sparse_keys, context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys, Ncontext_sparse: int, Ncontext_dense: int, Nfeature_list_sparse: int, Nfeature_list_dense: int, context_sparse_types, feature_list_dense_types, context_dense_shapes, feature_list_sparse_types, feature_list_dense_shapes, name, ctx):
  if not isinstance(feature_list_dense_missing_assumed_empty, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_missing_assumed_empty' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_missing_assumed_empty)
  feature_list_dense_missing_assumed_empty = [_execute.make_str(_s, "feature_list_dense_missing_assumed_empty") for _s in feature_list_dense_missing_assumed_empty]
  if not isinstance(context_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_keys' argument to "
        "'parse_sequence_example' Op, not %r." % context_sparse_keys)
  context_sparse_keys = [_execute.make_str(_s, "context_sparse_keys") for _s in context_sparse_keys]
  if not isinstance(context_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_keys' argument to "
        "'parse_sequence_example' Op, not %r." % context_dense_keys)
  context_dense_keys = [_execute.make_str(_s, "context_dense_keys") for _s in context_dense_keys]
  if not isinstance(feature_list_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_keys' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_sparse_keys)
  feature_list_sparse_keys = [_execute.make_str(_s, "feature_list_sparse_keys") for _s in feature_list_sparse_keys]
  if not isinstance(feature_list_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_keys' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_keys)
  feature_list_dense_keys = [_execute.make_str(_s, "feature_list_dense_keys") for _s in feature_list_dense_keys]
  if Ncontext_sparse is None:
    Ncontext_sparse = 0
  Ncontext_sparse = _execute.make_int(Ncontext_sparse, "Ncontext_sparse")
  if Ncontext_dense is None:
    Ncontext_dense = 0
  Ncontext_dense = _execute.make_int(Ncontext_dense, "Ncontext_dense")
  if Nfeature_list_sparse is None:
    Nfeature_list_sparse = 0
  Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, "Nfeature_list_sparse")
  if Nfeature_list_dense is None:
    Nfeature_list_dense = 0
  Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, "Nfeature_list_dense")
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_sequence_example' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_sequence_example' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_sequence_example' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _attr_Tcontext_dense, context_dense_defaults = _execute.convert_to_mixed_eager_tensors(context_dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  debug_name = _ops.convert_to_tensor(debug_name, _dtypes.string)
  _inputs_flat = [serialized, debug_name] + list(context_dense_defaults)
  _attrs = ("feature_list_dense_missing_assumed_empty",
  feature_list_dense_missing_assumed_empty, "context_sparse_keys",
  context_sparse_keys, "context_dense_keys", context_dense_keys,
  "feature_list_sparse_keys", feature_list_sparse_keys,
  "feature_list_dense_keys", feature_list_dense_keys, "Ncontext_sparse",
  Ncontext_sparse, "Ncontext_dense", Ncontext_dense, "Nfeature_list_sparse",
  Nfeature_list_sparse, "Nfeature_list_dense", Nfeature_list_dense,
  "context_sparse_types", context_sparse_types, "Tcontext_dense",
  _attr_Tcontext_dense, "feature_list_dense_types", feature_list_dense_types,
  "context_dense_shapes", context_dense_shapes, "feature_list_sparse_types",
  feature_list_sparse_types, "feature_list_dense_shapes",
  feature_list_dense_shapes)
  _result = _execute.execute(b"ParseSequenceExample", Ncontext_sparse +
                             len(context_sparse_types) + Ncontext_sparse +
                             len(context_dense_defaults) +
                             Nfeature_list_sparse +
                             len(feature_list_sparse_types) +
                             Nfeature_list_sparse +
                             len(feature_list_dense_types) +
                             Nfeature_list_dense, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseSequenceExample", _inputs_flat, _attrs, _result)
  _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + Nfeature_list_sparse]] + _result[4 + Nfeature_list_sparse:]
  _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
  _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:7 + len(feature_list_dense_types)]] + _result[7 + len(feature_list_dense_types):]
  _result = _result[:8] + [_result[8:]]
  _result = _ParseSequenceExampleOutput._make(_result)
  return _result

_ParseSequenceExampleV2Output = collections.namedtuple(
    "ParseSequenceExampleV2",
    ["context_sparse_indices", "context_sparse_values", "context_sparse_shapes", "context_dense_values", "context_ragged_values", "context_ragged_row_splits", "feature_list_sparse_indices", "feature_list_sparse_values", "feature_list_sparse_shapes", "feature_list_dense_values", "feature_list_dense_lengths", "feature_list_ragged_values", "feature_list_ragged_outer_splits", "feature_list_ragged_inner_splits"])


def parse_sequence_example_v2(serialized: Annotated[Any, _atypes.String], debug_name: Annotated[Any, _atypes.String], context_sparse_keys: Annotated[Any, _atypes.String], context_dense_keys: Annotated[Any, _atypes.String], context_ragged_keys: Annotated[Any, _atypes.String], feature_list_sparse_keys: Annotated[Any, _atypes.String], feature_list_dense_keys: Annotated[Any, _atypes.String], feature_list_ragged_keys: Annotated[Any, _atypes.String], feature_list_dense_missing_assumed_empty: Annotated[Any, _atypes.Bool], context_dense_defaults, Ncontext_sparse:int=0, context_sparse_types=[], context_ragged_value_types=[], context_ragged_split_types=[], context_dense_shapes=[], Nfeature_list_sparse:int=0, Nfeature_list_dense:int=0, feature_list_dense_types=[], feature_list_sparse_types=[], feature_list_ragged_value_types=[], feature_list_ragged_split_types=[], feature_list_dense_shapes=[], name=None):
  r"""Transforms a vector of tf.io.SequenceExample protos (as strings) into
typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar or vector containing binary serialized SequenceExample protos.
    debug_name: A `Tensor` of type `string`.
      A scalar or vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) name for the
      corresponding serialized proto.  This is purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no name is available.
    context_sparse_keys: A `Tensor` of type `string`.
      The keys expected in the Examples' features associated with context_sparse
      values.
    context_dense_keys: A `Tensor` of type `string`.
      The keys expected in the SequenceExamples' context features associated with
      dense values.
    context_ragged_keys: A `Tensor` of type `string`.
      The keys expected in the Examples' features associated with context_ragged
      values.
    feature_list_sparse_keys: A `Tensor` of type `string`.
      The keys expected in the FeatureLists associated with sparse values.
    feature_list_dense_keys: A `Tensor` of type `string`.
      The keys expected in the SequenceExamples' feature_lists associated
      with lists of dense values.
    feature_list_ragged_keys: A `Tensor` of type `string`.
      The keys expected in the FeatureLists associated with ragged values.
    feature_list_dense_missing_assumed_empty: A `Tensor` of type `bool`.
      A vector corresponding 1:1 with feature_list_dense_keys, indicating which
      features may be missing from the SequenceExamples.  If the associated
      FeatureList is missing, it is treated as empty.
    context_dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ncontext_dense Tensors (some may be empty).
      context_dense_defaults[j] provides default values
      when the SequenceExample's context map lacks context_dense_key[j].
      If an empty Tensor is provided for context_dense_defaults[j],
      then the Feature context_dense_keys[j] is required.
      The input type is inferred from context_dense_defaults[j], even when it's
      empty.  If context_dense_defaults[j] is not empty, its shape must match
      context_dense_shapes[j].
    Ncontext_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    context_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Ncontext_sparse types; the data types of data in
      each context Feature given in context_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    context_ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      RaggedTensor.value dtypes for the ragged context features.
    context_ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
      RaggedTensor.row_split dtypes for the ragged context features.
    context_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Ncontext_dense shapes; the shapes of data in
      each context Feature given in context_dense_keys.
      The number of elements in the Feature corresponding to context_dense_key[j]
      must always equal context_dense_shapes[j].NumEntries().
      The shape of context_dense_values[j] will match context_dense_shapes[j].
    Nfeature_list_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    Nfeature_list_dense: An optional `int` that is `>= 0`. Defaults to `0`.
    feature_list_dense_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    feature_list_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Nfeature_list_sparse types; the data types
      of data in each FeatureList given in feature_list_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      RaggedTensor.value dtypes for the ragged FeatureList features.
    feature_list_ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
      RaggedTensor.row_split dtypes for the ragged FeatureList features.
    feature_list_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Nfeature_list_dense shapes; the shapes of
      data in each FeatureList given in feature_list_dense_keys.
      The shape of each Feature in the FeatureList corresponding to
      feature_list_dense_key[j] must always equal
      feature_list_dense_shapes[j].NumEntries().
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, context_ragged_values, context_ragged_row_splits, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values, feature_list_dense_lengths, feature_list_ragged_values, feature_list_ragged_outer_splits, feature_list_ragged_inner_splits).

    context_sparse_indices: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_sparse_values: A list of `Tensor` objects of type `context_sparse_types`.
    context_sparse_shapes: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_dense_values: A list of `Tensor` objects. Has the same type as `context_dense_defaults`.
    context_ragged_values: A list of `Tensor` objects of type `context_ragged_value_types`.
    context_ragged_row_splits: A list of `Tensor` objects of type `context_ragged_split_types`.
    feature_list_sparse_indices: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_sparse_values: A list of `Tensor` objects of type `feature_list_sparse_types`.
    feature_list_sparse_shapes: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_dense_values: A list of `Tensor` objects of type `feature_list_dense_types`.
    feature_list_dense_lengths: A list of `Nfeature_list_dense` `Tensor` objects with type `int64`.
    feature_list_ragged_values: A list of `Tensor` objects of type `feature_list_ragged_value_types`.
    feature_list_ragged_outer_splits: A list of `Tensor` objects of type `feature_list_ragged_split_types`.
    feature_list_ragged_inner_splits: A list of `Tensor` objects of type `feature_list_ragged_split_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseSequenceExampleV2", name, serialized, debug_name,
        context_sparse_keys, context_dense_keys, context_ragged_keys,
        feature_list_sparse_keys, feature_list_dense_keys,
        feature_list_ragged_keys, feature_list_dense_missing_assumed_empty,
        context_dense_defaults, "Ncontext_sparse", Ncontext_sparse,
        "context_sparse_types", context_sparse_types,
        "context_ragged_value_types", context_ragged_value_types,
        "context_ragged_split_types", context_ragged_split_types,
        "context_dense_shapes", context_dense_shapes, "Nfeature_list_sparse",
        Nfeature_list_sparse, "Nfeature_list_dense", Nfeature_list_dense,
        "feature_list_dense_types", feature_list_dense_types,
        "feature_list_sparse_types", feature_list_sparse_types,
        "feature_list_ragged_value_types", feature_list_ragged_value_types,
        "feature_list_ragged_split_types", feature_list_ragged_split_types,
        "feature_list_dense_shapes", feature_list_dense_shapes)
      _result = _ParseSequenceExampleV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_sequence_example_v2_eager_fallback(
          serialized, debug_name, context_sparse_keys, context_dense_keys,
          context_ragged_keys, feature_list_sparse_keys,
          feature_list_dense_keys, feature_list_ragged_keys,
          feature_list_dense_missing_assumed_empty, context_dense_defaults,
          Ncontext_sparse=Ncontext_sparse,
          context_sparse_types=context_sparse_types,
          context_ragged_value_types=context_ragged_value_types,
          context_ragged_split_types=context_ragged_split_types,
          context_dense_shapes=context_dense_shapes,
          Nfeature_list_sparse=Nfeature_list_sparse,
          Nfeature_list_dense=Nfeature_list_dense,
          feature_list_dense_types=feature_list_dense_types,
          feature_list_sparse_types=feature_list_sparse_types,
          feature_list_ragged_value_types=feature_list_ragged_value_types,
          feature_list_ragged_split_types=feature_list_ragged_split_types,
          feature_list_dense_shapes=feature_list_dense_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Ncontext_sparse is None:
    Ncontext_sparse = 0
  Ncontext_sparse = _execute.make_int(Ncontext_sparse, "Ncontext_sparse")
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if context_ragged_value_types is None:
    context_ragged_value_types = []
  if not isinstance(context_ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_ragged_value_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_ragged_value_types)
  context_ragged_value_types = [_execute.make_type(_t, "context_ragged_value_types") for _t in context_ragged_value_types]
  if context_ragged_split_types is None:
    context_ragged_split_types = []
  if not isinstance(context_ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_ragged_split_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_ragged_split_types)
  context_ragged_split_types = [_execute.make_type(_t, "context_ragged_split_types") for _t in context_ragged_split_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if Nfeature_list_sparse is None:
    Nfeature_list_sparse = 0
  Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, "Nfeature_list_sparse")
  if Nfeature_list_dense is None:
    Nfeature_list_dense = 0
  Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, "Nfeature_list_dense")
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_ragged_value_types is None:
    feature_list_ragged_value_types = []
  if not isinstance(feature_list_ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_ragged_value_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_value_types)
  feature_list_ragged_value_types = [_execute.make_type(_t, "feature_list_ragged_value_types") for _t in feature_list_ragged_value_types]
  if feature_list_ragged_split_types is None:
    feature_list_ragged_split_types = []
  if not isinstance(feature_list_ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_ragged_split_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_split_types)
  feature_list_ragged_split_types = [_execute.make_type(_t, "feature_list_ragged_split_types") for _t in feature_list_ragged_split_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseSequenceExampleV2", serialized=serialized,
                                  debug_name=debug_name,
                                  context_sparse_keys=context_sparse_keys,
                                  context_dense_keys=context_dense_keys,
                                  context_ragged_keys=context_ragged_keys,
                                  feature_list_sparse_keys=feature_list_sparse_keys,
                                  feature_list_dense_keys=feature_list_dense_keys,
                                  feature_list_ragged_keys=feature_list_ragged_keys,
                                  feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty,
                                  context_dense_defaults=context_dense_defaults,
                                  Ncontext_sparse=Ncontext_sparse,
                                  context_sparse_types=context_sparse_types,
                                  context_ragged_value_types=context_ragged_value_types,
                                  context_ragged_split_types=context_ragged_split_types,
                                  context_dense_shapes=context_dense_shapes,
                                  Nfeature_list_sparse=Nfeature_list_sparse,
                                  Nfeature_list_dense=Nfeature_list_dense,
                                  feature_list_dense_types=feature_list_dense_types,
                                  feature_list_sparse_types=feature_list_sparse_types,
                                  feature_list_ragged_value_types=feature_list_ragged_value_types,
                                  feature_list_ragged_split_types=feature_list_ragged_split_types,
                                  feature_list_dense_shapes=feature_list_dense_shapes,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Ncontext_sparse", _op._get_attr_int("Ncontext_sparse"),
              "Tcontext_dense", _op.get_attr("Tcontext_dense"),
              "context_sparse_types", _op.get_attr("context_sparse_types"),
              "context_ragged_value_types",
              _op.get_attr("context_ragged_value_types"),
              "context_ragged_split_types",
              _op.get_attr("context_ragged_split_types"),
              "context_dense_shapes", _op.get_attr("context_dense_shapes"),
              "Nfeature_list_sparse",
              _op._get_attr_int("Nfeature_list_sparse"),
              "Nfeature_list_dense", _op._get_attr_int("Nfeature_list_dense"),
              "feature_list_dense_types",
              _op.get_attr("feature_list_dense_types"),
              "feature_list_sparse_types",
              _op.get_attr("feature_list_sparse_types"),
              "feature_list_ragged_value_types",
              _op.get_attr("feature_list_ragged_value_types"),
              "feature_list_ragged_split_types",
              _op.get_attr("feature_list_ragged_split_types"),
              "feature_list_dense_shapes",
              _op.get_attr("feature_list_dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseSequenceExampleV2", _inputs_flat, _attrs, _result)
  _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + len(context_ragged_value_types)]] + _result[4 + len(context_ragged_value_types):]
  _result = _result[:5] + [_result[5:5 + len(context_ragged_split_types)]] + _result[5 + len(context_ragged_split_types):]
  _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:7 + len(feature_list_sparse_types)]] + _result[7 + len(feature_list_sparse_types):]
  _result = _result[:8] + [_result[8:8 + Nfeature_list_sparse]] + _result[8 + Nfeature_list_sparse:]
  _result = _result[:9] + [_result[9:9 + len(feature_list_dense_types)]] + _result[9 + len(feature_list_dense_types):]
  _result = _result[:10] + [_result[10:10 + Nfeature_list_dense]] + _result[10 + Nfeature_list_dense:]
  _result = _result[:11] + [_result[11:11 + len(feature_list_ragged_value_types)]] + _result[11 + len(feature_list_ragged_value_types):]
  _result = _result[:12] + [_result[12:12 + len(feature_list_ragged_split_types)]] + _result[12 + len(feature_list_ragged_split_types):]
  _result = _result[:13] + [_result[13:]]
  _result = _ParseSequenceExampleV2Output._make(_result)
  return _result

ParseSequenceExampleV2 = tf_export("raw_ops.ParseSequenceExampleV2")(_ops.to_raw_op(parse_sequence_example_v2))


def parse_sequence_example_v2_eager_fallback(serialized: Annotated[Any, _atypes.String], debug_name: Annotated[Any, _atypes.String], context_sparse_keys: Annotated[Any, _atypes.String], context_dense_keys: Annotated[Any, _atypes.String], context_ragged_keys: Annotated[Any, _atypes.String], feature_list_sparse_keys: Annotated[Any, _atypes.String], feature_list_dense_keys: Annotated[Any, _atypes.String], feature_list_ragged_keys: Annotated[Any, _atypes.String], feature_list_dense_missing_assumed_empty: Annotated[Any, _atypes.Bool], context_dense_defaults, Ncontext_sparse: int, context_sparse_types, context_ragged_value_types, context_ragged_split_types, context_dense_shapes, Nfeature_list_sparse: int, Nfeature_list_dense: int, feature_list_dense_types, feature_list_sparse_types, feature_list_ragged_value_types, feature_list_ragged_split_types, feature_list_dense_shapes, name, ctx):
  if Ncontext_sparse is None:
    Ncontext_sparse = 0
  Ncontext_sparse = _execute.make_int(Ncontext_sparse, "Ncontext_sparse")
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if context_ragged_value_types is None:
    context_ragged_value_types = []
  if not isinstance(context_ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_ragged_value_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_ragged_value_types)
  context_ragged_value_types = [_execute.make_type(_t, "context_ragged_value_types") for _t in context_ragged_value_types]
  if context_ragged_split_types is None:
    context_ragged_split_types = []
  if not isinstance(context_ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_ragged_split_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_ragged_split_types)
  context_ragged_split_types = [_execute.make_type(_t, "context_ragged_split_types") for _t in context_ragged_split_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_sequence_example_v2' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if Nfeature_list_sparse is None:
    Nfeature_list_sparse = 0
  Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, "Nfeature_list_sparse")
  if Nfeature_list_dense is None:
    Nfeature_list_dense = 0
  Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, "Nfeature_list_dense")
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_ragged_value_types is None:
    feature_list_ragged_value_types = []
  if not isinstance(feature_list_ragged_value_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_ragged_value_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_value_types)
  feature_list_ragged_value_types = [_execute.make_type(_t, "feature_list_ragged_value_types") for _t in feature_list_ragged_value_types]
  if feature_list_ragged_split_types is None:
    feature_list_ragged_split_types = []
  if not isinstance(feature_list_ragged_split_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_ragged_split_types' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_split_types)
  feature_list_ragged_split_types = [_execute.make_type(_t, "feature_list_ragged_split_types") for _t in feature_list_ragged_split_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_sequence_example_v2' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _attr_Tcontext_dense, context_dense_defaults = _execute.convert_to_mixed_eager_tensors(context_dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  debug_name = _ops.convert_to_tensor(debug_name, _dtypes.string)
  context_sparse_keys = _ops.convert_to_tensor(context_sparse_keys, _dtypes.string)
  context_dense_keys = _ops.convert_to_tensor(context_dense_keys, _dtypes.string)
  context_ragged_keys = _ops.convert_to_tensor(context_ragged_keys, _dtypes.string)
  feature_list_sparse_keys = _ops.convert_to_tensor(feature_list_sparse_keys, _dtypes.string)
  feature_list_dense_keys = _ops.convert_to_tensor(feature_list_dense_keys, _dtypes.string)
  feature_list_ragged_keys = _ops.convert_to_tensor(feature_list_ragged_keys, _dtypes.string)
  feature_list_dense_missing_assumed_empty = _ops.convert_to_tensor(feature_list_dense_missing_assumed_empty, _dtypes.bool)
  _inputs_flat = [serialized, debug_name, context_sparse_keys, context_dense_keys, context_ragged_keys, feature_list_sparse_keys, feature_list_dense_keys, feature_list_ragged_keys, feature_list_dense_missing_assumed_empty] + list(context_dense_defaults)
  _attrs = ("Ncontext_sparse", Ncontext_sparse, "Tcontext_dense",
  _attr_Tcontext_dense, "context_sparse_types", context_sparse_types,
  "context_ragged_value_types", context_ragged_value_types,
  "context_ragged_split_types", context_ragged_split_types,
  "context_dense_shapes", context_dense_shapes, "Nfeature_list_sparse",
  Nfeature_list_sparse, "Nfeature_list_dense", Nfeature_list_dense,
  "feature_list_dense_types", feature_list_dense_types,
  "feature_list_sparse_types", feature_list_sparse_types,
  "feature_list_ragged_value_types", feature_list_ragged_value_types,
  "feature_list_ragged_split_types", feature_list_ragged_split_types,
  "feature_list_dense_shapes", feature_list_dense_shapes)
  _result = _execute.execute(b"ParseSequenceExampleV2", Ncontext_sparse +
                             len(context_sparse_types) + Ncontext_sparse +
                             len(context_dense_defaults) +
                             len(context_ragged_value_types) +
                             len(context_ragged_split_types) +
                             Nfeature_list_sparse +
                             len(feature_list_sparse_types) +
                             Nfeature_list_sparse +
                             len(feature_list_dense_types) +
                             Nfeature_list_dense +
                             len(feature_list_ragged_value_types) +
                             len(feature_list_ragged_split_types) +
                             len(feature_list_ragged_split_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseSequenceExampleV2", _inputs_flat, _attrs, _result)
  _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + len(context_ragged_value_types)]] + _result[4 + len(context_ragged_value_types):]
  _result = _result[:5] + [_result[5:5 + len(context_ragged_split_types)]] + _result[5 + len(context_ragged_split_types):]
  _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:7 + len(feature_list_sparse_types)]] + _result[7 + len(feature_list_sparse_types):]
  _result = _result[:8] + [_result[8:8 + Nfeature_list_sparse]] + _result[8 + Nfeature_list_sparse:]
  _result = _result[:9] + [_result[9:9 + len(feature_list_dense_types)]] + _result[9 + len(feature_list_dense_types):]
  _result = _result[:10] + [_result[10:10 + Nfeature_list_dense]] + _result[10 + Nfeature_list_dense:]
  _result = _result[:11] + [_result[11:11 + len(feature_list_ragged_value_types)]] + _result[11 + len(feature_list_ragged_value_types):]
  _result = _result[:12] + [_result[12:12 + len(feature_list_ragged_split_types)]] + _result[12 + len(feature_list_ragged_split_types):]
  _result = _result[:13] + [_result[13:]]
  _result = _ParseSequenceExampleV2Output._make(_result)
  return _result

_ParseSingleExampleOutput = collections.namedtuple(
    "ParseSingleExample",
    ["sparse_indices", "sparse_values", "sparse_shapes", "dense_values"])


def parse_single_example(serialized: Annotated[Any, _atypes.String], dense_defaults, num_sparse: int, sparse_keys, dense_keys, sparse_types, dense_shapes, name=None):
  r"""Transforms a tf.Example proto (as a string) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A vector containing a batch of binary serialized Example protos.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Tensors (some may be empty), whose length matches
      the length of `dense_keys`. dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
      then the shape of dense_defaults[j] must match that of dense_shapes[j].
      If dense_shapes[j] has an undefined major dimension (variable strides dense
      feature), dense_defaults[j] must contain a single element:
      the padding element.
    num_sparse: An `int` that is `>= 0`.
      The number of sparse features to be parsed from the example. This
      must match the lengths of `sparse_keys` and `sparse_types`.
    sparse_keys: A list of `strings`. A list of `num_sparse` strings.
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A list of `strings`.
      The keys expected in the Examples' features associated with dense
      values.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `num_sparse` types; the data types of data in each
      Feature given in sparse_keys.
      Currently the ParseSingleExample op supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of data in each Feature given in dense_keys.
      The length of this list must match the length of `dense_keys`.  The
      number of elements in the Feature corresponding to dense_key[j] must
      always equal dense_shapes[j].NumEntries().  If dense_shapes[j] ==
      (D0, D1, ..., DN) then the shape of output Tensor dense_values[j]
      will be (D0, D1, ..., DN): In the case dense_shapes[j] = (-1, D1,
      ..., DN), the shape of the output Tensor dense_values[j] will be (M,
      D1, .., DN), where M is the number of blocks of elements of length
      D1 * .... * DN, in the input.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values).

    sparse_indices: A list of `num_sparse` `Tensor` objects with type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list of `num_sparse` `Tensor` objects with type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseSingleExample", name, serialized, dense_defaults,
        "num_sparse", num_sparse, "sparse_keys", sparse_keys, "dense_keys",
        dense_keys, "sparse_types", sparse_types, "dense_shapes",
        dense_shapes)
      _result = _ParseSingleExampleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_single_example_eager_fallback(
          serialized, dense_defaults, num_sparse=num_sparse,
          sparse_keys=sparse_keys, dense_keys=dense_keys,
          sparse_types=sparse_types, dense_shapes=dense_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_sparse = _execute.make_int(num_sparse, "num_sparse")
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_single_example' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_single_example' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_single_example' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_single_example' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseSingleExample", serialized=serialized,
                              dense_defaults=dense_defaults,
                              num_sparse=num_sparse, sparse_keys=sparse_keys,
                              dense_keys=dense_keys,
                              sparse_types=sparse_types,
                              dense_shapes=dense_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_sparse", _op._get_attr_int("num_sparse"), "sparse_keys",
              _op.get_attr("sparse_keys"), "dense_keys",
              _op.get_attr("dense_keys"), "sparse_types",
              _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
              "dense_shapes", _op.get_attr("dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseSingleExample", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sparse]] + _result[num_sparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
  _result = _result[:3] + [_result[3:]]
  _result = _ParseSingleExampleOutput._make(_result)
  return _result

ParseSingleExample = tf_export("raw_ops.ParseSingleExample")(_ops.to_raw_op(parse_single_example))


def parse_single_example_eager_fallback(serialized: Annotated[Any, _atypes.String], dense_defaults, num_sparse: int, sparse_keys, dense_keys, sparse_types, dense_shapes, name, ctx):
  num_sparse = _execute.make_int(num_sparse, "num_sparse")
  if not isinstance(sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_keys' argument to "
        "'parse_single_example' Op, not %r." % sparse_keys)
  sparse_keys = [_execute.make_str(_s, "sparse_keys") for _s in sparse_keys]
  if not isinstance(dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_keys' argument to "
        "'parse_single_example' Op, not %r." % dense_keys)
  dense_keys = [_execute.make_str(_s, "dense_keys") for _s in dense_keys]
  if not isinstance(sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'sparse_types' argument to "
        "'parse_single_example' Op, not %r." % sparse_types)
  sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
  if not isinstance(dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dense_shapes' argument to "
        "'parse_single_example' Op, not %r." % dense_shapes)
  dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
  _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  _inputs_flat = [serialized] + list(dense_defaults)
  _attrs = ("num_sparse", num_sparse, "sparse_keys", sparse_keys,
  "dense_keys", dense_keys, "sparse_types", sparse_types, "Tdense",
  _attr_Tdense, "dense_shapes", dense_shapes)
  _result = _execute.execute(b"ParseSingleExample", num_sparse +
                             len(sparse_types) + num_sparse +
                             len(dense_defaults), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseSingleExample", _inputs_flat, _attrs, _result)
  _result = [_result[:num_sparse]] + _result[num_sparse:]
  _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
  _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
  _result = _result[:3] + [_result[3:]]
  _result = _ParseSingleExampleOutput._make(_result)
  return _result

_ParseSingleSequenceExampleOutput = collections.namedtuple(
    "ParseSingleSequenceExample",
    ["context_sparse_indices", "context_sparse_values", "context_sparse_shapes", "context_dense_values", "feature_list_sparse_indices", "feature_list_sparse_values", "feature_list_sparse_shapes", "feature_list_dense_values"])


def parse_single_sequence_example(serialized: Annotated[Any, _atypes.String], feature_list_dense_missing_assumed_empty: Annotated[Any, _atypes.String], context_sparse_keys: Annotated[List[Any], _atypes.String], context_dense_keys: Annotated[List[Any], _atypes.String], feature_list_sparse_keys: Annotated[List[Any], _atypes.String], feature_list_dense_keys: Annotated[List[Any], _atypes.String], context_dense_defaults, debug_name: Annotated[Any, _atypes.String], context_sparse_types=[], feature_list_dense_types=[], context_dense_shapes=[], feature_list_sparse_types=[], feature_list_dense_shapes=[], name=None):
  r"""Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar containing a binary serialized SequenceExample proto.
    feature_list_dense_missing_assumed_empty: A `Tensor` of type `string`.
      A vector listing the
      FeatureList keys which may be missing from the SequenceExample.  If the
      associated FeatureList is missing, it is treated as empty.  By default,
      any FeatureList not listed in this vector must exist in the SequenceExample.
    context_sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Ncontext_sparse string Tensors (scalars).
      The keys expected in the Examples' features associated with context_sparse
      values.
    context_dense_keys: A list of `Tensor` objects with type `string`.
      A list of Ncontext_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' context features associated with
      dense values.
    feature_list_sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Nfeature_list_sparse string Tensors
      (scalars).  The keys expected in the FeatureLists associated with sparse
      values.
    feature_list_dense_keys: A list of `Tensor` objects with type `string`.
      A list of Nfeature_list_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' feature_lists associated
      with lists of dense values.
    context_dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ncontext_dense Tensors (some may be empty).
      context_dense_defaults[j] provides default values
      when the SequenceExample's context map lacks context_dense_key[j].
      If an empty Tensor is provided for context_dense_defaults[j],
      then the Feature context_dense_keys[j] is required.
      The input type is inferred from context_dense_defaults[j], even when it's
      empty.  If context_dense_defaults[j] is not empty, its shape must match
      context_dense_shapes[j].
    debug_name: A `Tensor` of type `string`.
      A scalar containing the name of the serialized proto.
      May contain, for example, table key (descriptive) name for the
      corresponding serialized proto.  This is purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty scalar if no name is available.
    context_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Ncontext_sparse types; the data types of data in
      each context Feature given in context_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    context_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Ncontext_dense shapes; the shapes of data in
      each context Feature given in context_dense_keys.
      The number of elements in the Feature corresponding to context_dense_key[j]
      must always equal context_dense_shapes[j].NumEntries().
      The shape of context_dense_values[j] will match context_dense_shapes[j].
    feature_list_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Nfeature_list_sparse types; the data types
      of data in each FeatureList given in feature_list_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Nfeature_list_dense shapes; the shapes of
      data in each FeatureList given in feature_list_dense_keys.
      The shape of each Feature in the FeatureList corresponding to
      feature_list_dense_key[j] must always equal
      feature_list_dense_shapes[j].NumEntries().
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values).

    context_sparse_indices: A list with the same length as `context_sparse_keys` of `Tensor` objects with type `int64`.
    context_sparse_values: A list of `Tensor` objects of type `context_sparse_types`.
    context_sparse_shapes: A list with the same length as `context_sparse_keys` of `Tensor` objects with type `int64`.
    context_dense_values: A list of `Tensor` objects. Has the same type as `context_dense_defaults`.
    feature_list_sparse_indices: A list with the same length as `feature_list_sparse_keys` of `Tensor` objects with type `int64`.
    feature_list_sparse_values: A list of `Tensor` objects of type `feature_list_sparse_types`.
    feature_list_sparse_shapes: A list with the same length as `feature_list_sparse_keys` of `Tensor` objects with type `int64`.
    feature_list_dense_values: A list of `Tensor` objects of type `feature_list_dense_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseSingleSequenceExample", name, serialized,
        feature_list_dense_missing_assumed_empty, context_sparse_keys,
        context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys,
        context_dense_defaults, debug_name, "context_sparse_types",
        context_sparse_types, "feature_list_dense_types",
        feature_list_dense_types, "context_dense_shapes",
        context_dense_shapes, "feature_list_sparse_types",
        feature_list_sparse_types, "feature_list_dense_shapes",
        feature_list_dense_shapes)
      _result = _ParseSingleSequenceExampleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parse_single_sequence_example_eager_fallback(
          serialized, feature_list_dense_missing_assumed_empty,
          context_sparse_keys, context_dense_keys, feature_list_sparse_keys,
          feature_list_dense_keys, context_dense_defaults, debug_name,
          context_sparse_types=context_sparse_types,
          feature_list_dense_types=feature_list_dense_types,
          context_dense_shapes=context_dense_shapes,
          feature_list_sparse_types=feature_list_sparse_types,
          feature_list_dense_shapes=feature_list_dense_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(context_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_sparse_keys)
  _attr_Ncontext_sparse = len(context_sparse_keys)
  if not isinstance(context_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_dense_keys)
  _attr_Ncontext_dense = len(context_dense_keys)
  if not isinstance(feature_list_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_sparse_keys)
  _attr_Nfeature_list_sparse = len(feature_list_sparse_keys)
  if not isinstance(feature_list_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_keys)
  _attr_Nfeature_list_dense = len(feature_list_dense_keys)
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseSingleSequenceExample", serialized=serialized,
                                      feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty,
                                      context_sparse_keys=context_sparse_keys,
                                      context_dense_keys=context_dense_keys,
                                      feature_list_sparse_keys=feature_list_sparse_keys,
                                      feature_list_dense_keys=feature_list_dense_keys,
                                      context_dense_defaults=context_dense_defaults,
                                      debug_name=debug_name,
                                      context_sparse_types=context_sparse_types,
                                      feature_list_dense_types=feature_list_dense_types,
                                      context_dense_shapes=context_dense_shapes,
                                      feature_list_sparse_types=feature_list_sparse_types,
                                      feature_list_dense_shapes=feature_list_dense_shapes,
                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Ncontext_sparse", _op._get_attr_int("Ncontext_sparse"),
              "Ncontext_dense", _op._get_attr_int("Ncontext_dense"),
              "Nfeature_list_sparse",
              _op._get_attr_int("Nfeature_list_sparse"),
              "Nfeature_list_dense", _op._get_attr_int("Nfeature_list_dense"),
              "context_sparse_types", _op.get_attr("context_sparse_types"),
              "Tcontext_dense", _op.get_attr("Tcontext_dense"),
              "feature_list_dense_types",
              _op.get_attr("feature_list_dense_types"),
              "context_dense_shapes", _op.get_attr("context_dense_shapes"),
              "feature_list_sparse_types",
              _op.get_attr("feature_list_sparse_types"),
              "feature_list_dense_shapes",
              _op.get_attr("feature_list_dense_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseSingleSequenceExample", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_Ncontext_sparse]] + _result[_attr_Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + _attr_Ncontext_sparse]] + _result[2 + _attr_Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + _attr_Nfeature_list_sparse]] + _result[4 + _attr_Nfeature_list_sparse:]
  _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
  _result = _result[:6] + [_result[6:6 + _attr_Nfeature_list_sparse]] + _result[6 + _attr_Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:]]
  _result = _ParseSingleSequenceExampleOutput._make(_result)
  return _result

ParseSingleSequenceExample = tf_export("raw_ops.ParseSingleSequenceExample")(_ops.to_raw_op(parse_single_sequence_example))


def parse_single_sequence_example_eager_fallback(serialized: Annotated[Any, _atypes.String], feature_list_dense_missing_assumed_empty: Annotated[Any, _atypes.String], context_sparse_keys: Annotated[List[Any], _atypes.String], context_dense_keys: Annotated[List[Any], _atypes.String], feature_list_sparse_keys: Annotated[List[Any], _atypes.String], feature_list_dense_keys: Annotated[List[Any], _atypes.String], context_dense_defaults, debug_name: Annotated[Any, _atypes.String], context_sparse_types, feature_list_dense_types, context_dense_shapes, feature_list_sparse_types, feature_list_dense_shapes, name, ctx):
  if not isinstance(context_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_sparse_keys)
  _attr_Ncontext_sparse = len(context_sparse_keys)
  if not isinstance(context_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_dense_keys)
  _attr_Ncontext_dense = len(context_dense_keys)
  if not isinstance(feature_list_sparse_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_sparse_keys)
  _attr_Nfeature_list_sparse = len(feature_list_sparse_keys)
  if not isinstance(feature_list_dense_keys, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_keys' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_keys)
  _attr_Nfeature_list_dense = len(feature_list_dense_keys)
  if context_sparse_types is None:
    context_sparse_types = []
  if not isinstance(context_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_sparse_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_sparse_types)
  context_sparse_types = [_execute.make_type(_t, "context_sparse_types") for _t in context_sparse_types]
  if feature_list_dense_types is None:
    feature_list_dense_types = []
  if not isinstance(feature_list_dense_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_types)
  feature_list_dense_types = [_execute.make_type(_t, "feature_list_dense_types") for _t in feature_list_dense_types]
  if context_dense_shapes is None:
    context_dense_shapes = []
  if not isinstance(context_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'context_dense_shapes' argument to "
        "'parse_single_sequence_example' Op, not %r." % context_dense_shapes)
  context_dense_shapes = [_execute.make_shape(_s, "context_dense_shapes") for _s in context_dense_shapes]
  if feature_list_sparse_types is None:
    feature_list_sparse_types = []
  if not isinstance(feature_list_sparse_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_sparse_types' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_sparse_types)
  feature_list_sparse_types = [_execute.make_type(_t, "feature_list_sparse_types") for _t in feature_list_sparse_types]
  if feature_list_dense_shapes is None:
    feature_list_dense_shapes = []
  if not isinstance(feature_list_dense_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'feature_list_dense_shapes' argument to "
        "'parse_single_sequence_example' Op, not %r." % feature_list_dense_shapes)
  feature_list_dense_shapes = [_execute.make_shape(_s, "feature_list_dense_shapes") for _s in feature_list_dense_shapes]
  _attr_Tcontext_dense, context_dense_defaults = _execute.convert_to_mixed_eager_tensors(context_dense_defaults, ctx)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  feature_list_dense_missing_assumed_empty = _ops.convert_to_tensor(feature_list_dense_missing_assumed_empty, _dtypes.string)
  context_sparse_keys = _ops.convert_n_to_tensor(context_sparse_keys, _dtypes.string)
  context_dense_keys = _ops.convert_n_to_tensor(context_dense_keys, _dtypes.string)
  feature_list_sparse_keys = _ops.convert_n_to_tensor(feature_list_sparse_keys, _dtypes.string)
  feature_list_dense_keys = _ops.convert_n_to_tensor(feature_list_dense_keys, _dtypes.string)
  debug_name = _ops.convert_to_tensor(debug_name, _dtypes.string)
  _inputs_flat = [serialized, feature_list_dense_missing_assumed_empty] + list(context_sparse_keys) + list(context_dense_keys) + list(feature_list_sparse_keys) + list(feature_list_dense_keys) + list(context_dense_defaults) + [debug_name]
  _attrs = ("Ncontext_sparse", _attr_Ncontext_sparse, "Ncontext_dense",
  _attr_Ncontext_dense, "Nfeature_list_sparse", _attr_Nfeature_list_sparse,
  "Nfeature_list_dense", _attr_Nfeature_list_dense, "context_sparse_types",
  context_sparse_types, "Tcontext_dense", _attr_Tcontext_dense,
  "feature_list_dense_types", feature_list_dense_types,
  "context_dense_shapes", context_dense_shapes, "feature_list_sparse_types",
  feature_list_sparse_types, "feature_list_dense_shapes",
  feature_list_dense_shapes)
  _result = _execute.execute(b"ParseSingleSequenceExample",
                             _attr_Ncontext_sparse + len(context_sparse_types)
                             + _attr_Ncontext_sparse +
                             len(context_dense_defaults) +
                             _attr_Nfeature_list_sparse +
                             len(feature_list_sparse_types) +
                             _attr_Nfeature_list_sparse +
                             len(feature_list_dense_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseSingleSequenceExample", _inputs_flat, _attrs, _result)
  _result = [_result[:_attr_Ncontext_sparse]] + _result[_attr_Ncontext_sparse:]
  _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
  _result = _result[:2] + [_result[2:2 + _attr_Ncontext_sparse]] + _result[2 + _attr_Ncontext_sparse:]
  _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
  _result = _result[:4] + [_result[4:4 + _attr_Nfeature_list_sparse]] + _result[4 + _attr_Nfeature_list_sparse:]
  _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
  _result = _result[:6] + [_result[6:6 + _attr_Nfeature_list_sparse]] + _result[6 + _attr_Nfeature_list_sparse:]
  _result = _result[:7] + [_result[7:]]
  _result = _ParseSingleSequenceExampleOutput._make(_result)
  return _result


TV_ParseTensor_out_type = TypeVar("TV_ParseTensor_out_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.parse_tensor', v1=['io.parse_tensor', 'parse_tensor'])
@deprecated_endpoints('parse_tensor')
def parse_tensor(serialized: Annotated[Any, _atypes.String], out_type: TV_ParseTensor_out_type, name=None) -> Annotated[Any, TV_ParseTensor_out_type]:
  r"""Transforms a serialized tensorflow.TensorProto proto into a Tensor.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar string containing a serialized TensorProto proto.
    out_type: A `tf.DType`.
      The type of the serialized tensor.  The provided type must match the
      type of the serialized tensor and no implicit conversion will take place.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParseTensor", name, serialized, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_parse_tensor(
          (serialized, out_type, name,), None)
      if _result is not NotImplemented:
        return _result
      return parse_tensor_eager_fallback(
          serialized, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            parse_tensor, (), dict(serialized=serialized, out_type=out_type,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_parse_tensor(
        (serialized, out_type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParseTensor", serialized=serialized, out_type=out_type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          parse_tensor, (), dict(serialized=serialized, out_type=out_type,
                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("out_type", _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParseTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParseTensor = tf_export("raw_ops.ParseTensor")(_ops.to_raw_op(parse_tensor))
_dispatcher_for_parse_tensor = parse_tensor._tf_type_based_dispatcher.Dispatch


def parse_tensor_eager_fallback(serialized: Annotated[Any, _atypes.String], out_type: TV_ParseTensor_out_type, name, ctx) -> Annotated[Any, TV_ParseTensor_out_type]:
  out_type = _execute.make_type(out_type, "out_type")
  serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
  _inputs_flat = [serialized]
  _attrs = ("out_type", out_type)
  _result = _execute.execute(b"ParseTensor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParseTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SerializeTensor_T = TypeVar("TV_SerializeTensor_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def serialize_tensor(tensor: Annotated[Any, TV_SerializeTensor_T], name=None) -> Annotated[Any, _atypes.String]:
  r"""Transforms a Tensor into a serialized TensorProto proto.

  Args:
    tensor: A `Tensor`. A Tensor of type `T`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SerializeTensor", name, tensor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return serialize_tensor_eager_fallback(
          tensor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SerializeTensor", tensor=tensor, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SerializeTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SerializeTensor = tf_export("raw_ops.SerializeTensor")(_ops.to_raw_op(serialize_tensor))


def serialize_tensor_eager_fallback(tensor: Annotated[Any, TV_SerializeTensor_T], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SerializeTensor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SerializeTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_StringToNumber_out_type = TypeVar("TV_StringToNumber_out_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)

def string_to_number(string_tensor: Annotated[Any, _atypes.String], out_type:TV_StringToNumber_out_type=_dtypes.float32, name=None) -> Annotated[Any, TV_StringToNumber_out_type]:
  r"""Converts each string in the input Tensor to the specified numeric type.

  (Note that int32 overflow results in an error while float overflow
  results in a rounded value.)

  Example:

  >>> strings = ["5.0", "3.0", "7.0"]
  >>> tf.strings.to_number(strings)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([5., 3., 7.], dtype=float32)>

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.int64, tf.uint32, tf.uint64`. Defaults to `tf.float32`.
      The numeric type to interpret each string in `string_tensor` as.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringToNumber", name, string_tensor, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_to_number_eager_fallback(
          string_tensor, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.float32
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringToNumber", string_tensor=string_tensor, out_type=out_type,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("out_type", _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringToNumber", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringToNumber = tf_export("raw_ops.StringToNumber")(_ops.to_raw_op(string_to_number))


def string_to_number_eager_fallback(string_tensor: Annotated[Any, _atypes.String], out_type: TV_StringToNumber_out_type, name, ctx) -> Annotated[Any, TV_StringToNumber_out_type]:
  if out_type is None:
    out_type = _dtypes.float32
  out_type = _execute.make_type(out_type, "out_type")
  string_tensor = _ops.convert_to_tensor(string_tensor, _dtypes.string)
  _inputs_flat = [string_tensor]
  _attrs = ("out_type", out_type)
  _result = _execute.execute(b"StringToNumber", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringToNumber", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

