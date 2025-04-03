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

def empty_tensor_map(name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates and returns an empty tensor map.

  handle: an empty tensor map

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EmptyTensorMap", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return empty_tensor_map_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EmptyTensorMap", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EmptyTensorMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EmptyTensorMap = tf_export("raw_ops.EmptyTensorMap")(_ops.to_raw_op(empty_tensor_map))


def empty_tensor_map_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Variant]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"EmptyTensorMap", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EmptyTensorMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorMapErase_key_dtype = TypeVar("TV_TensorMapErase_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorMapErase_value_dtype = TypeVar("TV_TensorMapErase_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_map_erase(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapErase_key_dtype], value_dtype: TV_TensorMapErase_value_dtype, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Returns a tensor map with item from given key erased.

  input_handle: the original map
  output_handle: the map with value from given key removed
  key: the key of the value to be erased

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapErase", name, input_handle, key, "value_dtype",
        value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_erase_eager_fallback(
          input_handle, key, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapErase", input_handle=input_handle, key=key,
                          value_dtype=value_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapErase", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapErase = tf_export("raw_ops.TensorMapErase")(_ops.to_raw_op(tensor_map_erase))


def tensor_map_erase_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapErase_key_dtype], value_dtype: TV_TensorMapErase_value_dtype, name, ctx) -> Annotated[Any, _atypes.Variant]:
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"TensorMapErase", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapErase", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorMapHasKey_key_dtype = TypeVar("TV_TensorMapHasKey_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_map_has_key(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapHasKey_key_dtype], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns whether the given key exists in the map.

  input_handle: the input map
  key: the key to check
  has_key: whether the key is already in the map or not

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapHasKey", name, input_handle, key)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_has_key_eager_fallback(
          input_handle, key, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapHasKey", input_handle=input_handle, key=key, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapHasKey", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapHasKey = tf_export("raw_ops.TensorMapHasKey")(_ops.to_raw_op(tensor_map_has_key))


def tensor_map_has_key_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapHasKey_key_dtype], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype)
  _result = _execute.execute(b"TensorMapHasKey", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapHasKey", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorMapInsert_key_dtype = TypeVar("TV_TensorMapInsert_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorMapInsert_value_dtype = TypeVar("TV_TensorMapInsert_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_map_insert(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapInsert_key_dtype], value: Annotated[Any, TV_TensorMapInsert_value_dtype], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Returns a map that is the 'input_handle' with the given key-value pair inserted.

  input_handle: the original map
  output_handle: the map with key and value inserted
  key: the key to be inserted
  value: the value to be inserted

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapInsert", name, input_handle, key, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_insert_eager_fallback(
          input_handle, key, value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapInsert", input_handle=input_handle, key=key, value=value,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapInsert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapInsert = tf_export("raw_ops.TensorMapInsert")(_ops.to_raw_op(tensor_map_insert))


def tensor_map_insert_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapInsert_key_dtype], value: Annotated[Any, TV_TensorMapInsert_value_dtype], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  _attr_value_dtype, (value,) = _execute.args_to_matching_eager([value], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key, value]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", _attr_value_dtype)
  _result = _execute.execute(b"TensorMapInsert", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapInsert", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorMapLookup_key_dtype = TypeVar("TV_TensorMapLookup_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorMapLookup_value_dtype = TypeVar("TV_TensorMapLookup_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_map_lookup(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapLookup_key_dtype], value_dtype: TV_TensorMapLookup_value_dtype, name=None) -> Annotated[Any, TV_TensorMapLookup_value_dtype]:
  r"""Returns the value from a given key in a tensor map.

  input_handle: the input map
  key: the key to be looked up
  value: the value found from the given key

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `value_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapLookup", name, input_handle, key, "value_dtype",
        value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_lookup_eager_fallback(
          input_handle, key, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapLookup", input_handle=input_handle, key=key,
                           value_dtype=value_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapLookup", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapLookup = tf_export("raw_ops.TensorMapLookup")(_ops.to_raw_op(tensor_map_lookup))


def tensor_map_lookup_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], key: Annotated[Any, TV_TensorMapLookup_key_dtype], value_dtype: TV_TensorMapLookup_value_dtype, name, ctx) -> Annotated[Any, TV_TensorMapLookup_value_dtype]:
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, key]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"TensorMapLookup", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapLookup", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_map_size(input_handle: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the number of tensors in the input tensor map.

  input_handle: the input map
  size: the number of tensors in the map

  Args:
    input_handle: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapSize", name, input_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_size_eager_fallback(
          input_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapSize", input_handle=input_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapSize = tf_export("raw_ops.TensorMapSize")(_ops.to_raw_op(tensor_map_size))


def tensor_map_size_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int32]:
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = None
  _result = _execute.execute(b"TensorMapSize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorMapStackKeys_key_dtype = TypeVar("TV_TensorMapStackKeys_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_map_stack_keys(input_handle: Annotated[Any, _atypes.Variant], key_dtype: TV_TensorMapStackKeys_key_dtype, name=None) -> Annotated[Any, TV_TensorMapStackKeys_key_dtype]:
  r"""Returns a Tensor stack of all keys in a tensor map.

  input_handle: the input map
  keys: the returned Tensor of all keys in the map

  Args:
    input_handle: A `Tensor` of type `variant`.
    key_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `key_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorMapStackKeys", name, input_handle, "key_dtype",
        key_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_map_stack_keys_eager_fallback(
          input_handle, key_dtype=key_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorMapStackKeys", input_handle=input_handle, key_dtype=key_dtype,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorMapStackKeys", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorMapStackKeys = tf_export("raw_ops.TensorMapStackKeys")(_ops.to_raw_op(tensor_map_stack_keys))


def tensor_map_stack_keys_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], key_dtype: TV_TensorMapStackKeys_key_dtype, name, ctx) -> Annotated[Any, TV_TensorMapStackKeys_key_dtype]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = ("key_dtype", key_dtype)
  _result = _execute.execute(b"TensorMapStackKeys", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorMapStackKeys", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

