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

TV_AnonymousHashTable_key_dtype = TypeVar("TV_AnonymousHashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_AnonymousHashTable_value_dtype = TypeVar("TV_AnonymousHashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def anonymous_hash_table(key_dtype: TV_AnonymousHashTable_key_dtype, value_dtype: TV_AnonymousHashTable_value_dtype, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a uninitialized anonymous hash table.

  This op creates a new anonymous hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle.  Before using the table you will have
  to initialize it.  After initialization the table will be
  immutable. The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousHashTable", name, "key_dtype", key_dtype,
        "value_dtype", value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_hash_table_eager_fallback(
          key_dtype=key_dtype, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousHashTable", key_dtype=key_dtype, value_dtype=value_dtype,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousHashTable = tf_export("raw_ops.AnonymousHashTable")(_ops.to_raw_op(anonymous_hash_table))


def anonymous_hash_table_eager_fallback(key_dtype: TV_AnonymousHashTable_key_dtype, value_dtype: TV_AnonymousHashTable_value_dtype, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _inputs_flat = []
  _attrs = ("key_dtype", key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"AnonymousHashTable", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AnonymousMutableDenseHashTable_key_dtype = TypeVar("TV_AnonymousMutableDenseHashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_AnonymousMutableDenseHashTable_value_dtype = TypeVar("TV_AnonymousMutableDenseHashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def anonymous_mutable_dense_hash_table(empty_key: Annotated[Any, TV_AnonymousMutableDenseHashTable_key_dtype], deleted_key: Annotated[Any, TV_AnonymousMutableDenseHashTable_key_dtype], value_dtype: TV_AnonymousMutableDenseHashTable_value_dtype, value_shape=[], initial_num_buckets:int=131072, max_load_factor:float=0.8, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty anonymous mutable hash table that uses tensors as the backing store.

  This op creates a new anonymous mutable hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle. Each value must be a scalar.
  Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    deleted_key: A `Tensor`. Must have the same type as `empty_key`.
    value_dtype: A `tf.DType`. Type of the table values.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMutableDenseHashTable", name, empty_key, deleted_key,
        "value_dtype", value_dtype, "value_shape", value_shape,
        "initial_num_buckets", initial_num_buckets, "max_load_factor",
        max_load_factor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_mutable_dense_hash_table_eager_fallback(
          empty_key, deleted_key, value_dtype=value_dtype,
          value_shape=value_shape, initial_num_buckets=initial_num_buckets,
          max_load_factor=max_load_factor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  if initial_num_buckets is None:
    initial_num_buckets = 131072
  initial_num_buckets = _execute.make_int(initial_num_buckets, "initial_num_buckets")
  if max_load_factor is None:
    max_load_factor = 0.8
  max_load_factor = _execute.make_float(max_load_factor, "max_load_factor")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMutableDenseHashTable", empty_key=empty_key,
                                          deleted_key=deleted_key,
                                          value_dtype=value_dtype,
                                          value_shape=value_shape,
                                          initial_num_buckets=initial_num_buckets,
                                          max_load_factor=max_load_factor,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"), "initial_num_buckets",
              _op._get_attr_int("initial_num_buckets"), "max_load_factor",
              _op.get_attr("max_load_factor"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMutableDenseHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousMutableDenseHashTable = tf_export("raw_ops.AnonymousMutableDenseHashTable")(_ops.to_raw_op(anonymous_mutable_dense_hash_table))


def anonymous_mutable_dense_hash_table_eager_fallback(empty_key: Annotated[Any, TV_AnonymousMutableDenseHashTable_key_dtype], deleted_key: Annotated[Any, TV_AnonymousMutableDenseHashTable_key_dtype], value_dtype: TV_AnonymousMutableDenseHashTable_value_dtype, value_shape, initial_num_buckets: int, max_load_factor: float, name, ctx) -> Annotated[Any, _atypes.Resource]:
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  if initial_num_buckets is None:
    initial_num_buckets = 131072
  initial_num_buckets = _execute.make_int(initial_num_buckets, "initial_num_buckets")
  if max_load_factor is None:
    max_load_factor = 0.8
  max_load_factor = _execute.make_float(max_load_factor, "max_load_factor")
  _attr_key_dtype, _inputs_key_dtype = _execute.args_to_matching_eager([empty_key, deleted_key], ctx, [])
  (empty_key, deleted_key) = _inputs_key_dtype
  _inputs_flat = [empty_key, deleted_key]
  _attrs = ("key_dtype", _attr_key_dtype, "value_dtype", value_dtype,
  "value_shape", value_shape, "initial_num_buckets", initial_num_buckets,
  "max_load_factor", max_load_factor)
  _result = _execute.execute(b"AnonymousMutableDenseHashTable", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMutableDenseHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AnonymousMutableHashTable_key_dtype = TypeVar("TV_AnonymousMutableHashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_AnonymousMutableHashTable_value_dtype = TypeVar("TV_AnonymousMutableHashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def anonymous_mutable_hash_table(key_dtype: TV_AnonymousMutableHashTable_key_dtype, value_dtype: TV_AnonymousMutableHashTable_value_dtype, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty anonymous mutable hash table.

  This op creates a new anonymous mutable hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle. Each value must be a scalar.
  Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.
  The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMutableHashTable", name, "key_dtype", key_dtype,
        "value_dtype", value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_mutable_hash_table_eager_fallback(
          key_dtype=key_dtype, value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMutableHashTable", key_dtype=key_dtype,
                                     value_dtype=value_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMutableHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousMutableHashTable = tf_export("raw_ops.AnonymousMutableHashTable")(_ops.to_raw_op(anonymous_mutable_hash_table))


def anonymous_mutable_hash_table_eager_fallback(key_dtype: TV_AnonymousMutableHashTable_key_dtype, value_dtype: TV_AnonymousMutableHashTable_value_dtype, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  _inputs_flat = []
  _attrs = ("key_dtype", key_dtype, "value_dtype", value_dtype)
  _result = _execute.execute(b"AnonymousMutableHashTable", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMutableHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AnonymousMutableHashTableOfTensors_key_dtype = TypeVar("TV_AnonymousMutableHashTableOfTensors_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_AnonymousMutableHashTableOfTensors_value_dtype = TypeVar("TV_AnonymousMutableHashTableOfTensors_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def anonymous_mutable_hash_table_of_tensors(key_dtype: TV_AnonymousMutableHashTableOfTensors_key_dtype, value_dtype: TV_AnonymousMutableHashTableOfTensors_value_dtype, value_shape=[], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty anonymous mutable hash table of vector values.

  This op creates a new anonymous mutable hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle. Each value must be a vector.
  Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.
  The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMutableHashTableOfTensors", name, "key_dtype",
        key_dtype, "value_dtype", value_dtype, "value_shape", value_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_mutable_hash_table_of_tensors_eager_fallback(
          key_dtype=key_dtype, value_dtype=value_dtype,
          value_shape=value_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMutableHashTableOfTensors", key_dtype=key_dtype,
                                              value_dtype=value_dtype,
                                              value_shape=value_shape,
                                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("key_dtype", _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMutableHashTableOfTensors", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousMutableHashTableOfTensors = tf_export("raw_ops.AnonymousMutableHashTableOfTensors")(_ops.to_raw_op(anonymous_mutable_hash_table_of_tensors))


def anonymous_mutable_hash_table_of_tensors_eager_fallback(key_dtype: TV_AnonymousMutableHashTableOfTensors_key_dtype, value_dtype: TV_AnonymousMutableHashTableOfTensors_value_dtype, value_shape, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  _inputs_flat = []
  _attrs = ("key_dtype", key_dtype, "value_dtype", value_dtype, "value_shape",
  value_shape)
  _result = _execute.execute(b"AnonymousMutableHashTableOfTensors", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMutableHashTableOfTensors", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_HashTable_key_dtype = TypeVar("TV_HashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_HashTable_value_dtype = TypeVar("TV_HashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def hash_table(key_dtype: TV_HashTable_key_dtype, value_dtype: TV_HashTable_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Creates a non-initialized hash table.

  This op creates a hash table, specifying the type of its keys and values.
  Before using the table you will have to initialize it.  After initialization the
  table will be immutable.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("hash_table op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "HashTable", key_dtype=key_dtype, value_dtype=value_dtype,
                     container=container, shared_name=shared_name,
                     use_node_name_sharing=use_node_name_sharing, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "HashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

HashTable = tf_export("raw_ops.HashTable")(_ops.to_raw_op(hash_table))


def hash_table_eager_fallback(key_dtype: TV_HashTable_key_dtype, value_dtype: TV_HashTable_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("hash_table op does not support eager execution. Arg 'table_handle' is a ref.")

TV_HashTableV2_key_dtype = TypeVar("TV_HashTableV2_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_HashTableV2_value_dtype = TypeVar("TV_HashTableV2_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def hash_table_v2(key_dtype: TV_HashTableV2_key_dtype, value_dtype: TV_HashTableV2_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a non-initialized hash table.

  This op creates a hash table, specifying the type of its keys and values.
  Before using the table you will have to initialize it.  After initialization the
  table will be immutable.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "HashTableV2", name, "container", container, "shared_name",
        shared_name, "use_node_name_sharing", use_node_name_sharing,
        "key_dtype", key_dtype, "value_dtype", value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return hash_table_v2_eager_fallback(
          container=container, shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing, key_dtype=key_dtype,
          value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "HashTableV2", key_dtype=key_dtype, value_dtype=value_dtype,
                       container=container, shared_name=shared_name,
                       use_node_name_sharing=use_node_name_sharing, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "HashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

HashTableV2 = tf_export("raw_ops.HashTableV2")(_ops.to_raw_op(hash_table_v2))


def hash_table_v2_eager_fallback(key_dtype: TV_HashTableV2_key_dtype, value_dtype: TV_HashTableV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name,
  "use_node_name_sharing", use_node_name_sharing, "key_dtype", key_dtype,
  "value_dtype", value_dtype)
  _result = _execute.execute(b"HashTableV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "HashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InitializeTable_Tkey = TypeVar("TV_InitializeTable_Tkey", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_InitializeTable_Tval = TypeVar("TV_InitializeTable_Tval", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def initialize_table(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_InitializeTable_Tkey], values: Annotated[Any, TV_InitializeTable_Tval], name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("initialize_table op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTable", table_handle=table_handle, keys=keys,
                           values=values, name=name)
  return _op
InitializeTable = tf_export("raw_ops.InitializeTable")(_ops.to_raw_op(initialize_table))


def initialize_table_eager_fallback(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_InitializeTable_Tkey], values: Annotated[Any, TV_InitializeTable_Tval], name, ctx):
  raise RuntimeError("initialize_table op does not support eager execution. Arg 'table_handle' is a ref.")

def initialize_table_from_text_file(table_handle: Annotated[Any, _atypes.String], filename: Annotated[Any, _atypes.String], key_index: int, value_index: int, vocab_size:int=-1, delimiter:str="\t", offset:int=0, name=None):
  r"""Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\t"`.
      Delimiter to separate fields in a line.
    offset: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("initialize_table_from_text_file op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  key_index = _execute.make_int(key_index, "key_index")
  value_index = _execute.make_int(value_index, "value_index")
  if vocab_size is None:
    vocab_size = -1
  vocab_size = _execute.make_int(vocab_size, "vocab_size")
  if delimiter is None:
    delimiter = "\t"
  delimiter = _execute.make_str(delimiter, "delimiter")
  if offset is None:
    offset = 0
  offset = _execute.make_int(offset, "offset")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTableFromTextFile", table_handle=table_handle,
                                       filename=filename, key_index=key_index,
                                       value_index=value_index,
                                       vocab_size=vocab_size,
                                       delimiter=delimiter, offset=offset,
                                       name=name)
  return _op
InitializeTableFromTextFile = tf_export("raw_ops.InitializeTableFromTextFile")(_ops.to_raw_op(initialize_table_from_text_file))


def initialize_table_from_text_file_eager_fallback(table_handle: Annotated[Any, _atypes.String], filename: Annotated[Any, _atypes.String], key_index: int, value_index: int, vocab_size: int, delimiter: str, offset: int, name, ctx):
  raise RuntimeError("initialize_table_from_text_file op does not support eager execution. Arg 'table_handle' is a ref.")

def initialize_table_from_text_file_v2(table_handle: Annotated[Any, _atypes.Resource], filename: Annotated[Any, _atypes.String], key_index: int, value_index: int, vocab_size:int=-1, delimiter:str="\t", offset:int=0, name=None):
  r"""Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type `resource`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\t"`.
      Delimiter to separate fields in a line.
    offset: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InitializeTableFromTextFileV2", name, table_handle, filename,
        "key_index", key_index, "value_index", value_index, "vocab_size",
        vocab_size, "delimiter", delimiter, "offset", offset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return initialize_table_from_text_file_v2_eager_fallback(
          table_handle, filename, key_index=key_index,
          value_index=value_index, vocab_size=vocab_size, delimiter=delimiter,
          offset=offset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_index = _execute.make_int(key_index, "key_index")
  value_index = _execute.make_int(value_index, "value_index")
  if vocab_size is None:
    vocab_size = -1
  vocab_size = _execute.make_int(vocab_size, "vocab_size")
  if delimiter is None:
    delimiter = "\t"
  delimiter = _execute.make_str(delimiter, "delimiter")
  if offset is None:
    offset = 0
  offset = _execute.make_int(offset, "offset")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTableFromTextFileV2", table_handle=table_handle,
                                         filename=filename,
                                         key_index=key_index,
                                         value_index=value_index,
                                         vocab_size=vocab_size,
                                         delimiter=delimiter, offset=offset,
                                         name=name)
  return _op
InitializeTableFromTextFileV2 = tf_export("raw_ops.InitializeTableFromTextFileV2")(_ops.to_raw_op(initialize_table_from_text_file_v2))


def initialize_table_from_text_file_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], filename: Annotated[Any, _atypes.String], key_index: int, value_index: int, vocab_size: int, delimiter: str, offset: int, name, ctx):
  key_index = _execute.make_int(key_index, "key_index")
  value_index = _execute.make_int(value_index, "value_index")
  if vocab_size is None:
    vocab_size = -1
  vocab_size = _execute.make_int(vocab_size, "vocab_size")
  if delimiter is None:
    delimiter = "\t"
  delimiter = _execute.make_str(delimiter, "delimiter")
  if offset is None:
    offset = 0
  offset = _execute.make_int(offset, "offset")
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [table_handle, filename]
  _attrs = ("key_index", key_index, "value_index", value_index, "vocab_size",
  vocab_size, "delimiter", delimiter, "offset", offset)
  _result = _execute.execute(b"InitializeTableFromTextFileV2", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_InitializeTableV2_Tkey = TypeVar("TV_InitializeTableV2_Tkey", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_InitializeTableV2_Tval = TypeVar("TV_InitializeTableV2_Tval", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def initialize_table_v2(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_InitializeTableV2_Tkey], values: Annotated[Any, TV_InitializeTableV2_Tval], name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type `resource`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InitializeTableV2", name, table_handle, keys, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return initialize_table_v2_eager_fallback(
          table_handle, keys, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTableV2", table_handle=table_handle, keys=keys,
                             values=values, name=name)
  return _op
InitializeTableV2 = tf_export("raw_ops.InitializeTableV2")(_ops.to_raw_op(initialize_table_v2))


def initialize_table_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_InitializeTableV2_Tkey], values: Annotated[Any, TV_InitializeTableV2_Tval], name, ctx):
  _attr_Tkey, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
  _attr_Tval, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle, keys, values]
  _attrs = ("Tkey", _attr_Tkey, "Tval", _attr_Tval)
  _result = _execute.execute(b"InitializeTableV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

_LookupTableExportOutput = collections.namedtuple(
    "LookupTableExport",
    ["keys", "values"])


TV_LookupTableExport_Tkeys = TypeVar("TV_LookupTableExport_Tkeys", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableExport_Tvalues = TypeVar("TV_LookupTableExport_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_export(table_handle: Annotated[Any, _atypes.String], Tkeys: TV_LookupTableExport_Tkeys, Tvalues: TV_LookupTableExport_Tvalues, name=None):
  r"""Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `Tkeys`.
    values: A `Tensor` of type `Tvalues`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lookup_table_export op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  Tkeys = _execute.make_type(Tkeys, "Tkeys")
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableExport", table_handle=table_handle, Tkeys=Tkeys,
                             Tvalues=Tvalues, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tkeys", _op._get_attr_type("Tkeys"), "Tvalues",
              _op._get_attr_type("Tvalues"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableExport", _inputs_flat, _attrs, _result)
  _result = _LookupTableExportOutput._make(_result)
  return _result

LookupTableExport = tf_export("raw_ops.LookupTableExport")(_ops.to_raw_op(lookup_table_export))


def lookup_table_export_eager_fallback(table_handle: Annotated[Any, _atypes.String], Tkeys: TV_LookupTableExport_Tkeys, Tvalues: TV_LookupTableExport_Tvalues, name, ctx):
  raise RuntimeError("lookup_table_export op does not support eager execution. Arg 'table_handle' is a ref.")
_LookupTableExportV2Output = collections.namedtuple(
    "LookupTableExportV2",
    ["keys", "values"])


TV_LookupTableExportV2_Tkeys = TypeVar("TV_LookupTableExportV2_Tkeys", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableExportV2_Tvalues = TypeVar("TV_LookupTableExportV2_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_export_v2(table_handle: Annotated[Any, _atypes.Resource], Tkeys: TV_LookupTableExportV2_Tkeys, Tvalues: TV_LookupTableExportV2_Tvalues, name=None):
  r"""Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `Tkeys`.
    values: A `Tensor` of type `Tvalues`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableExportV2", name, table_handle, "Tkeys", Tkeys,
        "Tvalues", Tvalues)
      _result = _LookupTableExportV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_export_v2_eager_fallback(
          table_handle, Tkeys=Tkeys, Tvalues=Tvalues, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tkeys = _execute.make_type(Tkeys, "Tkeys")
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableExportV2", table_handle=table_handle, Tkeys=Tkeys,
                               Tvalues=Tvalues, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tkeys", _op._get_attr_type("Tkeys"), "Tvalues",
              _op._get_attr_type("Tvalues"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableExportV2", _inputs_flat, _attrs, _result)
  _result = _LookupTableExportV2Output._make(_result)
  return _result

LookupTableExportV2 = tf_export("raw_ops.LookupTableExportV2")(_ops.to_raw_op(lookup_table_export_v2))


def lookup_table_export_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], Tkeys: TV_LookupTableExportV2_Tkeys, Tvalues: TV_LookupTableExportV2_Tvalues, name, ctx):
  Tkeys = _execute.make_type(Tkeys, "Tkeys")
  Tvalues = _execute.make_type(Tvalues, "Tvalues")
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle]
  _attrs = ("Tkeys", Tkeys, "Tvalues", Tvalues)
  _result = _execute.execute(b"LookupTableExportV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LookupTableExportV2", _inputs_flat, _attrs, _result)
  _result = _LookupTableExportV2Output._make(_result)
  return _result


TV_LookupTableFind_Tin = TypeVar("TV_LookupTableFind_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableFind_Tout = TypeVar("TV_LookupTableFind_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_find(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableFind_Tin], default_value: Annotated[Any, TV_LookupTableFind_Tout], name=None) -> Annotated[Any, TV_LookupTableFind_Tout]:
  r"""Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lookup_table_find op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableFind", table_handle=table_handle, keys=keys,
                           default_value=default_value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableFind", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LookupTableFind = tf_export("raw_ops.LookupTableFind")(_ops.to_raw_op(lookup_table_find))


def lookup_table_find_eager_fallback(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableFind_Tin], default_value: Annotated[Any, TV_LookupTableFind_Tout], name, ctx) -> Annotated[Any, TV_LookupTableFind_Tout]:
  raise RuntimeError("lookup_table_find op does not support eager execution. Arg 'table_handle' is a ref.")

TV_LookupTableFindV2_Tin = TypeVar("TV_LookupTableFindV2_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableFindV2_Tout = TypeVar("TV_LookupTableFindV2_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_find_v2(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableFindV2_Tin], default_value: Annotated[Any, TV_LookupTableFindV2_Tout], name=None) -> Annotated[Any, TV_LookupTableFindV2_Tout]:
  r"""Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableFindV2", name, table_handle, keys, default_value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_find_v2_eager_fallback(
          table_handle, keys, default_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableFindV2", table_handle=table_handle, keys=keys,
                             default_value=default_value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableFindV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LookupTableFindV2 = tf_export("raw_ops.LookupTableFindV2")(_ops.to_raw_op(lookup_table_find_v2))


def lookup_table_find_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableFindV2_Tin], default_value: Annotated[Any, TV_LookupTableFindV2_Tout], name, ctx) -> Annotated[Any, TV_LookupTableFindV2_Tout]:
  _attr_Tin, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
  _attr_Tout, (default_value,) = _execute.args_to_matching_eager([default_value], ctx, [])
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle, keys, default_value]
  _attrs = ("Tin", _attr_Tin, "Tout", _attr_Tout)
  _result = _execute.execute(b"LookupTableFindV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LookupTableFindV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LookupTableImport_Tin = TypeVar("TV_LookupTableImport_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableImport_Tout = TypeVar("TV_LookupTableImport_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_import(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableImport_Tin], values: Annotated[Any, TV_LookupTableImport_Tout], name=None):
  r"""Replaces the contents of the table with the specified keys and values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lookup_table_import op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableImport", table_handle=table_handle, keys=keys,
                             values=values, name=name)
  return _op
LookupTableImport = tf_export("raw_ops.LookupTableImport")(_ops.to_raw_op(lookup_table_import))


def lookup_table_import_eager_fallback(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableImport_Tin], values: Annotated[Any, TV_LookupTableImport_Tout], name, ctx):
  raise RuntimeError("lookup_table_import op does not support eager execution. Arg 'table_handle' is a ref.")

TV_LookupTableImportV2_Tin = TypeVar("TV_LookupTableImportV2_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableImportV2_Tout = TypeVar("TV_LookupTableImportV2_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_import_v2(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableImportV2_Tin], values: Annotated[Any, TV_LookupTableImportV2_Tout], name=None):
  r"""Replaces the contents of the table with the specified keys and values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableImportV2", name, table_handle, keys, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_import_v2_eager_fallback(
          table_handle, keys, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableImportV2", table_handle=table_handle, keys=keys,
                               values=values, name=name)
  return _op
LookupTableImportV2 = tf_export("raw_ops.LookupTableImportV2")(_ops.to_raw_op(lookup_table_import_v2))


def lookup_table_import_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableImportV2_Tin], values: Annotated[Any, TV_LookupTableImportV2_Tout], name, ctx):
  _attr_Tin, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
  _attr_Tout, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle, keys, values]
  _attrs = ("Tin", _attr_Tin, "Tout", _attr_Tout)
  _result = _execute.execute(b"LookupTableImportV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_LookupTableInsert_Tin = TypeVar("TV_LookupTableInsert_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableInsert_Tout = TypeVar("TV_LookupTableInsert_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_insert(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableInsert_Tin], values: Annotated[Any, TV_LookupTableInsert_Tout], name=None):
  r"""Updates the table to associates keys with values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lookup_table_insert op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableInsert", table_handle=table_handle, keys=keys,
                             values=values, name=name)
  return _op
LookupTableInsert = tf_export("raw_ops.LookupTableInsert")(_ops.to_raw_op(lookup_table_insert))


def lookup_table_insert_eager_fallback(table_handle: Annotated[Any, _atypes.String], keys: Annotated[Any, TV_LookupTableInsert_Tin], values: Annotated[Any, TV_LookupTableInsert_Tout], name, ctx):
  raise RuntimeError("lookup_table_insert op does not support eager execution. Arg 'table_handle' is a ref.")

TV_LookupTableInsertV2_Tin = TypeVar("TV_LookupTableInsertV2_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_LookupTableInsertV2_Tout = TypeVar("TV_LookupTableInsertV2_Tout", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_insert_v2(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableInsertV2_Tin], values: Annotated[Any, TV_LookupTableInsertV2_Tout], name=None):
  r"""Updates the table to associates keys with values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableInsertV2", name, table_handle, keys, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_insert_v2_eager_fallback(
          table_handle, keys, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableInsertV2", table_handle=table_handle, keys=keys,
                               values=values, name=name)
  return _op
LookupTableInsertV2 = tf_export("raw_ops.LookupTableInsertV2")(_ops.to_raw_op(lookup_table_insert_v2))


def lookup_table_insert_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableInsertV2_Tin], values: Annotated[Any, TV_LookupTableInsertV2_Tout], name, ctx):
  _attr_Tin, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
  _attr_Tout, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle, keys, values]
  _attrs = ("Tin", _attr_Tin, "Tout", _attr_Tout)
  _result = _execute.execute(b"LookupTableInsertV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_LookupTableRemoveV2_Tin = TypeVar("TV_LookupTableRemoveV2_Tin", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def lookup_table_remove_v2(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableRemoveV2_Tin], name=None):
  r"""Removes keys and its associated values from a table.

  The tensor `keys` must of the same type as the keys of the table. Keys not
  already in the table are silently ignored.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys of the elements to remove.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableRemoveV2", name, table_handle, keys)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_remove_v2_eager_fallback(
          table_handle, keys, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableRemoveV2", table_handle=table_handle, keys=keys,
                               name=name)
  return _op
LookupTableRemoveV2 = tf_export("raw_ops.LookupTableRemoveV2")(_ops.to_raw_op(lookup_table_remove_v2))


def lookup_table_remove_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], keys: Annotated[Any, TV_LookupTableRemoveV2_Tin], name, ctx):
  _attr_Tin, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle, keys]
  _attrs = ("Tin", _attr_Tin)
  _result = _execute.execute(b"LookupTableRemoveV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def lookup_table_size(table_handle: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("lookup_table_size op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableSize", table_handle=table_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LookupTableSize = tf_export("raw_ops.LookupTableSize")(_ops.to_raw_op(lookup_table_size))


def lookup_table_size_eager_fallback(table_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int64]:
  raise RuntimeError("lookup_table_size op does not support eager execution. Arg 'table_handle' is a ref.")

def lookup_table_size_v2(table_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LookupTableSizeV2", name, table_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lookup_table_size_v2_eager_fallback(
          table_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LookupTableSizeV2", table_handle=table_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LookupTableSizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LookupTableSizeV2 = tf_export("raw_ops.LookupTableSizeV2")(_ops.to_raw_op(lookup_table_size_v2))


def lookup_table_size_v2_eager_fallback(table_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int64]:
  table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
  _inputs_flat = [table_handle]
  _attrs = None
  _result = _execute.execute(b"LookupTableSizeV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LookupTableSizeV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MutableDenseHashTable_key_dtype = TypeVar("TV_MutableDenseHashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableDenseHashTable_value_dtype = TypeVar("TV_MutableDenseHashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_dense_hash_table(empty_key: Annotated[Any, TV_MutableDenseHashTable_key_dtype], value_dtype: TV_MutableDenseHashTable_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, value_shape=[], initial_num_buckets:int=131072, max_load_factor:float=0.8, name=None) -> Annotated[Any, _atypes.String]:
  r"""Creates an empty hash table that uses tensors as the backing store.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("mutable_dense_hash_table op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  if initial_num_buckets is None:
    initial_num_buckets = 131072
  initial_num_buckets = _execute.make_int(initial_num_buckets, "initial_num_buckets")
  if max_load_factor is None:
    max_load_factor = 0.8
  max_load_factor = _execute.make_float(max_load_factor, "max_load_factor")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableDenseHashTable", empty_key=empty_key, value_dtype=value_dtype,
                                 container=container, shared_name=shared_name,
                                 use_node_name_sharing=use_node_name_sharing,
                                 value_shape=value_shape,
                                 initial_num_buckets=initial_num_buckets,
                                 max_load_factor=max_load_factor, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"), "initial_num_buckets",
              _op._get_attr_int("initial_num_buckets"), "max_load_factor",
              _op.get_attr("max_load_factor"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableDenseHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableDenseHashTable = tf_export("raw_ops.MutableDenseHashTable")(_ops.to_raw_op(mutable_dense_hash_table))


def mutable_dense_hash_table_eager_fallback(empty_key: Annotated[Any, TV_MutableDenseHashTable_key_dtype], value_dtype: TV_MutableDenseHashTable_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, initial_num_buckets: int, max_load_factor: float, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("mutable_dense_hash_table op does not support eager execution. Arg 'table_handle' is a ref.")

TV_MutableDenseHashTableV2_key_dtype = TypeVar("TV_MutableDenseHashTableV2_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableDenseHashTableV2_value_dtype = TypeVar("TV_MutableDenseHashTableV2_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_dense_hash_table_v2(empty_key: Annotated[Any, TV_MutableDenseHashTableV2_key_dtype], deleted_key: Annotated[Any, TV_MutableDenseHashTableV2_key_dtype], value_dtype: TV_MutableDenseHashTableV2_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, value_shape=[], initial_num_buckets:int=131072, max_load_factor:float=0.8, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty hash table that uses tensors as the backing store.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    deleted_key: A `Tensor`. Must have the same type as `empty_key`.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MutableDenseHashTableV2", name, empty_key, deleted_key,
        "container", container, "shared_name", shared_name,
        "use_node_name_sharing", use_node_name_sharing, "value_dtype",
        value_dtype, "value_shape", value_shape, "initial_num_buckets",
        initial_num_buckets, "max_load_factor", max_load_factor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mutable_dense_hash_table_v2_eager_fallback(
          empty_key, deleted_key, container=container,
          shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing,
          value_dtype=value_dtype, value_shape=value_shape,
          initial_num_buckets=initial_num_buckets,
          max_load_factor=max_load_factor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  if initial_num_buckets is None:
    initial_num_buckets = 131072
  initial_num_buckets = _execute.make_int(initial_num_buckets, "initial_num_buckets")
  if max_load_factor is None:
    max_load_factor = 0.8
  max_load_factor = _execute.make_float(max_load_factor, "max_load_factor")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableDenseHashTableV2", empty_key=empty_key,
                                   deleted_key=deleted_key,
                                   value_dtype=value_dtype,
                                   container=container,
                                   shared_name=shared_name,
                                   use_node_name_sharing=use_node_name_sharing,
                                   value_shape=value_shape,
                                   initial_num_buckets=initial_num_buckets,
                                   max_load_factor=max_load_factor, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"), "initial_num_buckets",
              _op._get_attr_int("initial_num_buckets"), "max_load_factor",
              _op.get_attr("max_load_factor"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableDenseHashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableDenseHashTableV2 = tf_export("raw_ops.MutableDenseHashTableV2")(_ops.to_raw_op(mutable_dense_hash_table_v2))


def mutable_dense_hash_table_v2_eager_fallback(empty_key: Annotated[Any, TV_MutableDenseHashTableV2_key_dtype], deleted_key: Annotated[Any, TV_MutableDenseHashTableV2_key_dtype], value_dtype: TV_MutableDenseHashTableV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, initial_num_buckets: int, max_load_factor: float, name, ctx) -> Annotated[Any, _atypes.Resource]:
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  if initial_num_buckets is None:
    initial_num_buckets = 131072
  initial_num_buckets = _execute.make_int(initial_num_buckets, "initial_num_buckets")
  if max_load_factor is None:
    max_load_factor = 0.8
  max_load_factor = _execute.make_float(max_load_factor, "max_load_factor")
  _attr_key_dtype, _inputs_key_dtype = _execute.args_to_matching_eager([empty_key, deleted_key], ctx, [])
  (empty_key, deleted_key) = _inputs_key_dtype
  _inputs_flat = [empty_key, deleted_key]
  _attrs = ("container", container, "shared_name", shared_name,
  "use_node_name_sharing", use_node_name_sharing, "key_dtype",
  _attr_key_dtype, "value_dtype", value_dtype, "value_shape", value_shape,
  "initial_num_buckets", initial_num_buckets, "max_load_factor",
  max_load_factor)
  _result = _execute.execute(b"MutableDenseHashTableV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MutableDenseHashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MutableHashTable_key_dtype = TypeVar("TV_MutableHashTable_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableHashTable_value_dtype = TypeVar("TV_MutableHashTable_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_hash_table(key_dtype: TV_MutableHashTable_key_dtype, value_dtype: TV_MutableHashTable_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("mutable_hash_table op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableHashTable", key_dtype=key_dtype, value_dtype=value_dtype,
                            container=container, shared_name=shared_name,
                            use_node_name_sharing=use_node_name_sharing,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableHashTable", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableHashTable = tf_export("raw_ops.MutableHashTable")(_ops.to_raw_op(mutable_hash_table))


def mutable_hash_table_eager_fallback(key_dtype: TV_MutableHashTable_key_dtype, value_dtype: TV_MutableHashTable_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("mutable_hash_table op does not support eager execution. Arg 'table_handle' is a ref.")

TV_MutableHashTableOfTensors_key_dtype = TypeVar("TV_MutableHashTableOfTensors_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableHashTableOfTensors_value_dtype = TypeVar("TV_MutableHashTableOfTensors_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_hash_table_of_tensors(key_dtype: TV_MutableHashTableOfTensors_key_dtype, value_dtype: TV_MutableHashTableOfTensors_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, value_shape=[], name=None) -> Annotated[Any, _atypes.String]:
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a vector. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("mutable_hash_table_of_tensors op does not support eager execution. Arg 'table_handle' is a ref.")
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableHashTableOfTensors", key_dtype=key_dtype,
                                     value_dtype=value_dtype,
                                     container=container,
                                     shared_name=shared_name,
                                     use_node_name_sharing=use_node_name_sharing,
                                     value_shape=value_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableHashTableOfTensors", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableHashTableOfTensors = tf_export("raw_ops.MutableHashTableOfTensors")(_ops.to_raw_op(mutable_hash_table_of_tensors))


def mutable_hash_table_of_tensors_eager_fallback(key_dtype: TV_MutableHashTableOfTensors_key_dtype, value_dtype: TV_MutableHashTableOfTensors_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, name, ctx) -> Annotated[Any, _atypes.String]:
  raise RuntimeError("mutable_hash_table_of_tensors op does not support eager execution. Arg 'table_handle' is a ref.")

TV_MutableHashTableOfTensorsV2_key_dtype = TypeVar("TV_MutableHashTableOfTensorsV2_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableHashTableOfTensorsV2_value_dtype = TypeVar("TV_MutableHashTableOfTensorsV2_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_hash_table_of_tensors_v2(key_dtype: TV_MutableHashTableOfTensorsV2_key_dtype, value_dtype: TV_MutableHashTableOfTensorsV2_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, value_shape=[], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a vector. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MutableHashTableOfTensorsV2", name, "container", container,
        "shared_name", shared_name, "use_node_name_sharing",
        use_node_name_sharing, "key_dtype", key_dtype, "value_dtype",
        value_dtype, "value_shape", value_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mutable_hash_table_of_tensors_v2_eager_fallback(
          container=container, shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing, key_dtype=key_dtype,
          value_dtype=value_dtype, value_shape=value_shape, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableHashTableOfTensorsV2", key_dtype=key_dtype,
                                       value_dtype=value_dtype,
                                       container=container,
                                       shared_name=shared_name,
                                       use_node_name_sharing=use_node_name_sharing,
                                       value_shape=value_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"), "value_shape",
              _op.get_attr("value_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableHashTableOfTensorsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableHashTableOfTensorsV2 = tf_export("raw_ops.MutableHashTableOfTensorsV2")(_ops.to_raw_op(mutable_hash_table_of_tensors_v2))


def mutable_hash_table_of_tensors_v2_eager_fallback(key_dtype: TV_MutableHashTableOfTensorsV2_key_dtype, value_dtype: TV_MutableHashTableOfTensorsV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  if value_shape is None:
    value_shape = []
  value_shape = _execute.make_shape(value_shape, "value_shape")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name,
  "use_node_name_sharing", use_node_name_sharing, "key_dtype", key_dtype,
  "value_dtype", value_dtype, "value_shape", value_shape)
  _result = _execute.execute(b"MutableHashTableOfTensorsV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MutableHashTableOfTensorsV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MutableHashTableV2_key_dtype = TypeVar("TV_MutableHashTableV2_key_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_MutableHashTableV2_value_dtype = TypeVar("TV_MutableHashTableV2_value_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def mutable_hash_table_v2(key_dtype: TV_MutableHashTableV2_key_dtype, value_dtype: TV_MutableHashTableV2_value_dtype, container:str="", shared_name:str="", use_node_name_sharing:bool=False, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MutableHashTableV2", name, "container", container,
        "shared_name", shared_name, "use_node_name_sharing",
        use_node_name_sharing, "key_dtype", key_dtype, "value_dtype",
        value_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mutable_hash_table_v2_eager_fallback(
          container=container, shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing, key_dtype=key_dtype,
          value_dtype=value_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MutableHashTableV2", key_dtype=key_dtype, value_dtype=value_dtype,
                              container=container, shared_name=shared_name,
                              use_node_name_sharing=use_node_name_sharing,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "use_node_name_sharing",
              _op._get_attr_bool("use_node_name_sharing"), "key_dtype",
              _op._get_attr_type("key_dtype"), "value_dtype",
              _op._get_attr_type("value_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MutableHashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MutableHashTableV2 = tf_export("raw_ops.MutableHashTableV2")(_ops.to_raw_op(mutable_hash_table_v2))


def mutable_hash_table_v2_eager_fallback(key_dtype: TV_MutableHashTableV2_key_dtype, value_dtype: TV_MutableHashTableV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, name, ctx) -> Annotated[Any, _atypes.Resource]:
  key_dtype = _execute.make_type(key_dtype, "key_dtype")
  value_dtype = _execute.make_type(value_dtype, "value_dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if use_node_name_sharing is None:
    use_node_name_sharing = False
  use_node_name_sharing = _execute.make_bool(use_node_name_sharing, "use_node_name_sharing")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name,
  "use_node_name_sharing", use_node_name_sharing, "key_dtype", key_dtype,
  "value_dtype", value_dtype)
  _result = _execute.execute(b"MutableHashTableV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MutableHashTableV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

