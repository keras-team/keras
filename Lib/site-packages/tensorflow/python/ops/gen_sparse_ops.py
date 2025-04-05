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

TV_AddManySparseToTensorsMap_T = TypeVar("TV_AddManySparseToTensorsMap_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def add_many_sparse_to_tensors_map(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_AddManySparseToTensorsMap_T], sparse_shape: Annotated[Any, _atypes.Int64], container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

  A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
  `sparse_values`, and `sparse_shape`, where

  ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```

  An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
  having a first `sparse_indices` column taking values between `[0, N)`, where
  the minibatch size `N == sparse_shape[0]`.

  The input `SparseTensor` must have rank `R` greater than 1, and the first
  dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The stored
  `SparseTensor` objects pointed to by each row of the output `sparse_handles`
  will have rank `R-1`.

  The `SparseTensor` values can then be read out as part of a minibatch by passing
  the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
  the correct `SparseTensorsMap` is accessed, ensure that the same
  `container` and `shared_name` are passed to that Op.  If no `shared_name`
  is provided here, instead use the *name* of the Operation created by calling
  `AddManySparseToTensorsMap` as the `shared_name` passed to
  `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
      `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
      The minibatch size `N == sparse_shape[0]`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` created by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` created by this op.
      If blank, the new Operation's unique name is used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AddManySparseToTensorsMap", name, sparse_indices,
        sparse_values, sparse_shape, "container", container, "shared_name",
        shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return add_many_sparse_to_tensors_map_eager_fallback(
          sparse_indices, sparse_values, sparse_shape, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
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
        "AddManySparseToTensorsMap", sparse_indices=sparse_indices,
                                     sparse_values=sparse_values,
                                     sparse_shape=sparse_shape,
                                     container=container,
                                     shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AddManySparseToTensorsMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AddManySparseToTensorsMap = tf_export("raw_ops.AddManySparseToTensorsMap")(_ops.to_raw_op(add_many_sparse_to_tensors_map))


def add_many_sparse_to_tensors_map_eager_fallback(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_AddManySparseToTensorsMap_T], sparse_shape: Annotated[Any, _atypes.Int64], container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int64]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_T, (sparse_values,) = _execute.args_to_matching_eager([sparse_values], ctx, [])
  sparse_indices = _ops.convert_to_tensor(sparse_indices, _dtypes.int64)
  sparse_shape = _ops.convert_to_tensor(sparse_shape, _dtypes.int64)
  _inputs_flat = [sparse_indices, sparse_values, sparse_shape]
  _attrs = ("T", _attr_T, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"AddManySparseToTensorsMap", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AddManySparseToTensorsMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AddSparseToTensorsMap_T = TypeVar("TV_AddSparseToTensorsMap_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def add_sparse_to_tensors_map(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_AddSparseToTensorsMap_T], sparse_shape: Annotated[Any, _atypes.Int64], container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

  A `SparseTensor` is represented by three tensors: `sparse_indices`,
  `sparse_values`, and `sparse_shape`.

  This operator takes the given `SparseTensor` and adds it to a container
  object (a `SparseTensorsMap`).  A unique key within this container is generated
  in the form of an `int64`, and this is the value that is returned.

  The `SparseTensor` can then be read out as part of a minibatch by passing
  the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
  the correct `SparseTensorsMap` is accessed, ensure that the same
  `container` and `shared_name` are passed to that Op.  If no `shared_name`
  is provided here, instead use the *name* of the Operation created by calling
  `AddSparseToTensorsMap` as the `shared_name` passed to
  `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`.
    sparse_values: A `Tensor`. 1-D.  The `values` of the `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` created by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` created by this op.
      If blank, the new Operation's unique name is used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AddSparseToTensorsMap", name, sparse_indices, sparse_values,
        sparse_shape, "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return add_sparse_to_tensors_map_eager_fallback(
          sparse_indices, sparse_values, sparse_shape, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
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
        "AddSparseToTensorsMap", sparse_indices=sparse_indices,
                                 sparse_values=sparse_values,
                                 sparse_shape=sparse_shape,
                                 container=container, shared_name=shared_name,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AddSparseToTensorsMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AddSparseToTensorsMap = tf_export("raw_ops.AddSparseToTensorsMap")(_ops.to_raw_op(add_sparse_to_tensors_map))


def add_sparse_to_tensors_map_eager_fallback(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_AddSparseToTensorsMap_T], sparse_shape: Annotated[Any, _atypes.Int64], container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Int64]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_T, (sparse_values,) = _execute.args_to_matching_eager([sparse_values], ctx, [])
  sparse_indices = _ops.convert_to_tensor(sparse_indices, _dtypes.int64)
  sparse_shape = _ops.convert_to_tensor(sparse_shape, _dtypes.int64)
  _inputs_flat = [sparse_indices, sparse_values, sparse_shape]
  _attrs = ("T", _attr_T, "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"AddSparseToTensorsMap", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AddSparseToTensorsMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_DeserializeManySparseOutput = collections.namedtuple(
    "DeserializeManySparse",
    ["sparse_indices", "sparse_values", "sparse_shape"])


TV_DeserializeManySparse_dtype = TypeVar("TV_DeserializeManySparse_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def deserialize_many_sparse(serialized_sparse: Annotated[Any, _atypes.String], dtype: TV_DeserializeManySparse_dtype, name=None):
  r"""Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `SerializeSparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the serialized input is a `[2 x 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: A `Tensor` of type `string`.
      2-D, The `N` serialized `SparseTensor` objects.
      Must have 3 columns.
    dtype: A `tf.DType`. The `dtype` of the serialized `SparseTensor` objects.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeserializeManySparse", name, serialized_sparse, "dtype",
        dtype)
      _result = _DeserializeManySparseOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return deserialize_many_sparse_eager_fallback(
          serialized_sparse, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeserializeManySparse", serialized_sparse=serialized_sparse,
                                 dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DeserializeManySparse", _inputs_flat, _attrs, _result)
  _result = _DeserializeManySparseOutput._make(_result)
  return _result

DeserializeManySparse = tf_export("raw_ops.DeserializeManySparse")(_ops.to_raw_op(deserialize_many_sparse))


def deserialize_many_sparse_eager_fallback(serialized_sparse: Annotated[Any, _atypes.String], dtype: TV_DeserializeManySparse_dtype, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  serialized_sparse = _ops.convert_to_tensor(serialized_sparse, _dtypes.string)
  _inputs_flat = [serialized_sparse]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"DeserializeManySparse", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DeserializeManySparse", _inputs_flat, _attrs, _result)
  _result = _DeserializeManySparseOutput._make(_result)
  return _result

_DeserializeSparseOutput = collections.namedtuple(
    "DeserializeSparse",
    ["sparse_indices", "sparse_values", "sparse_shape"])


TV_DeserializeSparse_dtype = TypeVar("TV_DeserializeSparse_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_DeserializeSparse_Tserialized = TypeVar("TV_DeserializeSparse_Tserialized", _atypes.String, _atypes.Variant)

def deserialize_sparse(serialized_sparse: Annotated[Any, TV_DeserializeSparse_Tserialized], dtype: TV_DeserializeSparse_dtype, name=None):
  r"""Deserialize `SparseTensor` objects.

  The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
  the last dimension stores serialized `SparseTensor` objects and the other N
  dimensions (N >= 0) correspond to a batch. The ranks of the original
  `SparseTensor` objects must all match. When the final `SparseTensor` is
  created, its rank is the rank of the incoming `SparseTensor` objects plus N;
  the sparse tensors have been concatenated along new dimensions, one for each
  batch.

  The output `SparseTensor` object's shape values for the original dimensions
  are the max across the input `SparseTensor` objects' shape values for the
  corresponding dimensions. The new dimensions match the size of the batch.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the serialized input is a `[2 x 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: A `Tensor`. Must be one of the following types: `string`, `variant`.
      The serialized `SparseTensor` objects. The last dimension
      must have 3 columns.
    dtype: A `tf.DType`. The `dtype` of the serialized `SparseTensor` objects.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeserializeSparse", name, serialized_sparse, "dtype", dtype)
      _result = _DeserializeSparseOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return deserialize_sparse_eager_fallback(
          serialized_sparse, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeserializeSparse", serialized_sparse=serialized_sparse, dtype=dtype,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tserialized",
              _op._get_attr_type("Tserialized"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DeserializeSparse", _inputs_flat, _attrs, _result)
  _result = _DeserializeSparseOutput._make(_result)
  return _result

DeserializeSparse = tf_export("raw_ops.DeserializeSparse")(_ops.to_raw_op(deserialize_sparse))


def deserialize_sparse_eager_fallback(serialized_sparse: Annotated[Any, TV_DeserializeSparse_Tserialized], dtype: TV_DeserializeSparse_dtype, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tserialized, (serialized_sparse,) = _execute.args_to_matching_eager([serialized_sparse], ctx, [_dtypes.string, _dtypes.variant, ], _dtypes.string)
  _inputs_flat = [serialized_sparse]
  _attrs = ("dtype", dtype, "Tserialized", _attr_Tserialized)
  _result = _execute.execute(b"DeserializeSparse", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DeserializeSparse", _inputs_flat, _attrs, _result)
  _result = _DeserializeSparseOutput._make(_result)
  return _result


TV_SerializeManySparse_T = TypeVar("TV_SerializeManySparse_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SerializeManySparse_out_type = TypeVar("TV_SerializeManySparse_out_type", _atypes.String, _atypes.Variant)

def serialize_many_sparse(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_SerializeManySparse_T], sparse_shape: Annotated[Any, _atypes.Int64], out_type:TV_SerializeManySparse_out_type=_dtypes.string, name=None) -> Annotated[Any, TV_SerializeManySparse_out_type]:
  r"""Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of `serialized_sparse` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
    out_type: An optional `tf.DType` from: `tf.string, tf.variant`. Defaults to `tf.string`.
      The `dtype` to use for serialization; the supported types are `string`
      (default) and `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SerializeManySparse", name, sparse_indices, sparse_values,
        sparse_shape, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return serialize_many_sparse_eager_fallback(
          sparse_indices, sparse_values, sparse_shape, out_type=out_type,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.string
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SerializeManySparse", sparse_indices=sparse_indices,
                               sparse_values=sparse_values,
                               sparse_shape=sparse_shape, out_type=out_type,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SerializeManySparse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SerializeManySparse = tf_export("raw_ops.SerializeManySparse")(_ops.to_raw_op(serialize_many_sparse))


def serialize_many_sparse_eager_fallback(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_SerializeManySparse_T], sparse_shape: Annotated[Any, _atypes.Int64], out_type: TV_SerializeManySparse_out_type, name, ctx) -> Annotated[Any, TV_SerializeManySparse_out_type]:
  if out_type is None:
    out_type = _dtypes.string
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, (sparse_values,) = _execute.args_to_matching_eager([sparse_values], ctx, [])
  sparse_indices = _ops.convert_to_tensor(sparse_indices, _dtypes.int64)
  sparse_shape = _ops.convert_to_tensor(sparse_shape, _dtypes.int64)
  _inputs_flat = [sparse_indices, sparse_values, sparse_shape]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"SerializeManySparse", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SerializeManySparse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SerializeSparse_T = TypeVar("TV_SerializeSparse_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SerializeSparse_out_type = TypeVar("TV_SerializeSparse_out_type", _atypes.String, _atypes.Variant)

def serialize_sparse(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_SerializeSparse_T], sparse_shape: Annotated[Any, _atypes.Int64], out_type:TV_SerializeSparse_out_type=_dtypes.string, name=None) -> Annotated[Any, TV_SerializeSparse_out_type]:
  r"""Serialize a `SparseTensor` into a `[3]` `Tensor` object.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`.
    sparse_values: A `Tensor`. 1-D.  The `values` of the `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`.
    out_type: An optional `tf.DType` from: `tf.string, tf.variant`. Defaults to `tf.string`.
      The `dtype` to use for serialization; the supported types are `string`
      (default) and `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SerializeSparse", name, sparse_indices, sparse_values,
        sparse_shape, "out_type", out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return serialize_sparse_eager_fallback(
          sparse_indices, sparse_values, sparse_shape, out_type=out_type,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.string
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SerializeSparse", sparse_indices=sparse_indices,
                           sparse_values=sparse_values,
                           sparse_shape=sparse_shape, out_type=out_type,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SerializeSparse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SerializeSparse = tf_export("raw_ops.SerializeSparse")(_ops.to_raw_op(serialize_sparse))


def serialize_sparse_eager_fallback(sparse_indices: Annotated[Any, _atypes.Int64], sparse_values: Annotated[Any, TV_SerializeSparse_T], sparse_shape: Annotated[Any, _atypes.Int64], out_type: TV_SerializeSparse_out_type, name, ctx) -> Annotated[Any, TV_SerializeSparse_out_type]:
  if out_type is None:
    out_type = _dtypes.string
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, (sparse_values,) = _execute.args_to_matching_eager([sparse_values], ctx, [])
  sparse_indices = _ops.convert_to_tensor(sparse_indices, _dtypes.int64)
  sparse_shape = _ops.convert_to_tensor(sparse_shape, _dtypes.int64)
  _inputs_flat = [sparse_indices, sparse_values, sparse_shape]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"SerializeSparse", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SerializeSparse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseAddOutput = collections.namedtuple(
    "SparseAdd",
    ["sum_indices", "sum_values", "sum_shape"])


TV_SparseAdd_T = TypeVar("TV_SparseAdd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseAdd_Treal = TypeVar("TV_SparseAdd_Treal", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_add(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseAdd_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseAdd_T], b_shape: Annotated[Any, _atypes.Int64], thresh: Annotated[Any, TV_SparseAdd_Treal], name=None):
  r"""Adds two `SparseTensor` objects to produce another `SparseTensor`.

  The input `SparseTensor` objects' indices are assumed ordered in standard
  lexicographic order.  If this is not the case, before this step run
  `SparseReorder` to restore index ordering.

  By default, if two values sum to zero at some index, the output `SparseTensor`
  would still include that particular location in its index, storing a zero in the
  corresponding value slot.  To override this, callers can specify `thresh`,
  indicating that if the sum has a magnitude strictly smaller than `thresh`, its
  corresponding value and index would then not be included.  In particular,
  `thresh == 0` (default) means everything is kept and actual thresholding happens
  only for a positive value.

  In the following shapes, `nnz` is the count after taking `thresh` into account.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
    b_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
    b_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
    thresh: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      0-D.  The magnitude threshold that determines if an output value/index
      pair takes space.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sum_indices, sum_values, sum_shape).

    sum_indices: A `Tensor` of type `int64`.
    sum_values: A `Tensor`. Has the same type as `a_values`.
    sum_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseAdd", name, a_indices, a_values, a_shape, b_indices,
        b_values, b_shape, thresh)
      _result = _SparseAddOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_add_eager_fallback(
          a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseAdd", a_indices=a_indices, a_values=a_values, a_shape=a_shape,
                     b_indices=b_indices, b_values=b_values, b_shape=b_shape,
                     thresh=thresh, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Treal",
              _op._get_attr_type("Treal"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseAdd", _inputs_flat, _attrs, _result)
  _result = _SparseAddOutput._make(_result)
  return _result

SparseAdd = tf_export("raw_ops.SparseAdd")(_ops.to_raw_op(sparse_add))


def sparse_add_eager_fallback(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseAdd_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseAdd_T], b_shape: Annotated[Any, _atypes.Int64], thresh: Annotated[Any, TV_SparseAdd_Treal], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (a_values, b_values) = _inputs_T
  _attr_Treal, (thresh,) = _execute.args_to_matching_eager([thresh], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  a_indices = _ops.convert_to_tensor(a_indices, _dtypes.int64)
  a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
  b_indices = _ops.convert_to_tensor(b_indices, _dtypes.int64)
  b_shape = _ops.convert_to_tensor(b_shape, _dtypes.int64)
  _inputs_flat = [a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh]
  _attrs = ("T", _attr_T, "Treal", _attr_Treal)
  _result = _execute.execute(b"SparseAdd", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseAdd", _inputs_flat, _attrs, _result)
  _result = _SparseAddOutput._make(_result)
  return _result

_SparseAddGradOutput = collections.namedtuple(
    "SparseAddGrad",
    ["a_val_grad", "b_val_grad"])


TV_SparseAddGrad_T = TypeVar("TV_SparseAddGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_add_grad(backprop_val_grad: Annotated[Any, TV_SparseAddGrad_T], a_indices: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], sum_indices: Annotated[Any, _atypes.Int64], name=None):
  r"""The gradient operator for the SparseAdd op.

  The SparseAdd op calculates A + B, where A, B, and the sum are all represented
  as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
  non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
  values of A and B.

  Args:
    backprop_val_grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D with shape `[nnz(sum)]`.  The gradient with respect to
      the non-empty values of the sum.
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
    b_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
    sum_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the sum `SparseTensor`, size
      `[nnz(sum), ndims]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a_val_grad, b_val_grad).

    a_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`.
    b_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseAddGrad", name, backprop_val_grad, a_indices, b_indices,
        sum_indices)
      _result = _SparseAddGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_add_grad_eager_fallback(
          backprop_val_grad, a_indices, b_indices, sum_indices, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseAddGrad", backprop_val_grad=backprop_val_grad,
                         a_indices=a_indices, b_indices=b_indices,
                         sum_indices=sum_indices, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseAddGrad", _inputs_flat, _attrs, _result)
  _result = _SparseAddGradOutput._make(_result)
  return _result

SparseAddGrad = tf_export("raw_ops.SparseAddGrad")(_ops.to_raw_op(sparse_add_grad))


def sparse_add_grad_eager_fallback(backprop_val_grad: Annotated[Any, TV_SparseAddGrad_T], a_indices: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], sum_indices: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_T, (backprop_val_grad,) = _execute.args_to_matching_eager([backprop_val_grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  a_indices = _ops.convert_to_tensor(a_indices, _dtypes.int64)
  b_indices = _ops.convert_to_tensor(b_indices, _dtypes.int64)
  sum_indices = _ops.convert_to_tensor(sum_indices, _dtypes.int64)
  _inputs_flat = [backprop_val_grad, a_indices, b_indices, sum_indices]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseAddGrad", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseAddGrad", _inputs_flat, _attrs, _result)
  _result = _SparseAddGradOutput._make(_result)
  return _result

_SparseConcatOutput = collections.namedtuple(
    "SparseConcat",
    ["output_indices", "output_values", "output_shape"])


TV_SparseConcat_T = TypeVar("TV_SparseConcat_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_concat(indices: Annotated[List[Any], _atypes.Int64], values: Annotated[List[Any], TV_SparseConcat_T], shapes: Annotated[List[Any], _atypes.Int64], concat_dim: int, name=None):
  r"""Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of these sparse tensors.
  It is assumed that each input is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    indices: A list of at least 2 `Tensor` objects with type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list with the same length as `indices` of `Tensor` objects with the same type.
      1-D.  Non-empty values of each `SparseTensor`.
    shapes: A list with the same length as `indices` of `Tensor` objects with type `int64`.
      1-D.  Shapes of each `SparseTensor`.
    concat_dim: An `int`.
      Dimension to concatenate along. Must be in range [-rank, rank),
      where rank is the number of dimensions in each input `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `values`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseConcat", name, indices, values, shapes, "concat_dim",
        concat_dim)
      _result = _SparseConcatOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_concat_eager_fallback(
          indices, values, shapes, concat_dim=concat_dim, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_concat' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_concat' Op, not %r." % values)
  if len(values) != _attr_N:
    raise ValueError(
        "List argument 'values' to 'sparse_concat' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(values), _attr_N))
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_concat' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_concat' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  concat_dim = _execute.make_int(concat_dim, "concat_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseConcat", indices=indices, values=values, shapes=shapes,
                        concat_dim=concat_dim, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("concat_dim", _op._get_attr_int("concat_dim"), "N",
              _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseConcat", _inputs_flat, _attrs, _result)
  _result = _SparseConcatOutput._make(_result)
  return _result

SparseConcat = tf_export("raw_ops.SparseConcat")(_ops.to_raw_op(sparse_concat))


def sparse_concat_eager_fallback(indices: Annotated[List[Any], _atypes.Int64], values: Annotated[List[Any], TV_SparseConcat_T], shapes: Annotated[List[Any], _atypes.Int64], concat_dim: int, name, ctx):
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_concat' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_concat' Op, not %r." % values)
  if len(values) != _attr_N:
    raise ValueError(
        "List argument 'values' to 'sparse_concat' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(values), _attr_N))
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_concat' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_concat' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  concat_dim = _execute.make_int(concat_dim, "concat_dim")
  _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
  shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
  _inputs_flat = list(indices) + list(values) + list(shapes)
  _attrs = ("concat_dim", concat_dim, "N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"SparseConcat", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseConcat", _inputs_flat, _attrs, _result)
  _result = _SparseConcatOutput._make(_result)
  return _result

_SparseCrossOutput = collections.namedtuple(
    "SparseCross",
    ["output_indices", "output_values", "output_shape"])


TV_SparseCross_out_type = TypeVar("TV_SparseCross_out_type", _atypes.Int64, _atypes.String)
TV_SparseCross_internal_type = TypeVar("TV_SparseCross_internal_type", _atypes.Int64, _atypes.String)

def sparse_cross(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, hashed_output: bool, num_buckets: int, hash_key: int, out_type: TV_SparseCross_out_type, internal_type: TV_SparseCross_internal_type, name=None):
  r"""Generates sparse cross from a list of sparse and dense tensors.

  The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
  representing features of one feature column. It outputs a 2D `SparseTensor` with
  the batchwise crosses of these features.

  For example, if the inputs are

      inputs[0]: SparseTensor with shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      inputs[1]: SparseTensor with shape = [2, 1]
      [0, 0]: "d"
      [1, 0]: "e"

      inputs[2]: Tensor [["f"], ["g"]]

  then the output will be

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  if hashed_output=true then the output will be

      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))

  Args:
    indices: A list of `Tensor` objects with type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list of `Tensor` objects with types from: `int64`, `string`.
      1-D.   values of each `SparseTensor`.
    shapes: A list with the same length as `indices` of `Tensor` objects with type `int64`.
      1-D.   Shapes of each `SparseTensor`.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      2-D.    Columns represented by dense `Tensor`.
    hashed_output: A `bool`.
      If true, returns the hash of the cross instead of the string.
      This will allow us avoiding string manipulations.
    num_buckets: An `int` that is `>= 0`. It is used if hashed_output is true.
      output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
    hash_key: An `int`.
      Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints.
    out_type: A `tf.DType` from: `tf.int64, tf.string`.
    internal_type: A `tf.DType` from: `tf.int64, tf.string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor` of type `out_type`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseCross", name, indices, values, shapes, dense_inputs,
        "hashed_output", hashed_output, "num_buckets", num_buckets,
        "hash_key", hash_key, "out_type", out_type, "internal_type",
        internal_type)
      _result = _SparseCrossOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_cross_eager_fallback(
          indices, values, shapes, dense_inputs, hashed_output=hashed_output,
          num_buckets=num_buckets, hash_key=hash_key, out_type=out_type,
          internal_type=internal_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  hashed_output = _execute.make_bool(hashed_output, "hashed_output")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  hash_key = _execute.make_int(hash_key, "hash_key")
  out_type = _execute.make_type(out_type, "out_type")
  internal_type = _execute.make_type(internal_type, "internal_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseCross", indices=indices, values=values, shapes=shapes,
                       dense_inputs=dense_inputs, hashed_output=hashed_output,
                       num_buckets=num_buckets, hash_key=hash_key,
                       out_type=out_type, internal_type=internal_type,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "hashed_output",
              _op._get_attr_bool("hashed_output"), "num_buckets",
              _op._get_attr_int("num_buckets"), "hash_key",
              _op._get_attr_int("hash_key"), "sparse_types",
              _op.get_attr("sparse_types"), "dense_types",
              _op.get_attr("dense_types"), "out_type",
              _op._get_attr_type("out_type"), "internal_type",
              _op._get_attr_type("internal_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseCross", _inputs_flat, _attrs, _result)
  _result = _SparseCrossOutput._make(_result)
  return _result

SparseCross = tf_export("raw_ops.SparseCross")(_ops.to_raw_op(sparse_cross))


def sparse_cross_eager_fallback(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, hashed_output: bool, num_buckets: int, hash_key: int, out_type: TV_SparseCross_out_type, internal_type: TV_SparseCross_internal_type, name, ctx):
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  hashed_output = _execute.make_bool(hashed_output, "hashed_output")
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  hash_key = _execute.make_int(hash_key, "hash_key")
  out_type = _execute.make_type(out_type, "out_type")
  internal_type = _execute.make_type(internal_type, "internal_type")
  _attr_sparse_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
  shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
  _inputs_flat = list(indices) + list(values) + list(shapes) + list(dense_inputs)
  _attrs = ("N", _attr_N, "hashed_output", hashed_output, "num_buckets",
  num_buckets, "hash_key", hash_key, "sparse_types", _attr_sparse_types,
  "dense_types", _attr_dense_types, "out_type", out_type, "internal_type",
  internal_type)
  _result = _execute.execute(b"SparseCross", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseCross", _inputs_flat, _attrs, _result)
  _result = _SparseCrossOutput._make(_result)
  return _result

_SparseCrossHashedOutput = collections.namedtuple(
    "SparseCrossHashed",
    ["output_indices", "output_values", "output_shape"])


def sparse_cross_hashed(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, num_buckets: Annotated[Any, _atypes.Int64], strong_hash: Annotated[Any, _atypes.Bool], salt: Annotated[Any, _atypes.Int64], name=None):
  r"""Generates sparse cross from a list of sparse and dense tensors.

  The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
  representing features of one feature column. It outputs a 2D `SparseTensor` with
  the batchwise crosses of these features.

  For example, if the inputs are

      inputs[0]: SparseTensor with shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      inputs[1]: SparseTensor with shape = [2, 1]
      [0, 0]: "d"
      [1, 0]: "e"

      inputs[2]: Tensor [["f"], ["g"]]

  then the output will be

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  if hashed_output=true then the output will be

      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))

  Args:
    indices: A list of `Tensor` objects with type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list of `Tensor` objects with types from: `int64`, `string`.
      1-D.   values of each `SparseTensor`.
    shapes: A list with the same length as `indices` of `Tensor` objects with type `int64`.
      1-D.   Shapes of each `SparseTensor`.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      2-D.    Columns represented by dense `Tensor`.
    num_buckets: A `Tensor` of type `int64`.
      It is used if hashed_output is true.
      output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
    strong_hash: A `Tensor` of type `bool`.
      boolean, if true, siphash with salt will be used instead of farmhash.
    salt: A `Tensor` of type `int64`.
      Specify the salt that will be used by the siphash function.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor` of type `int64`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseCrossHashed", name, indices, values, shapes,
        dense_inputs, num_buckets, strong_hash, salt)
      _result = _SparseCrossHashedOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_cross_hashed_eager_fallback(
          indices, values, shapes, dense_inputs, num_buckets, strong_hash,
          salt, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross_hashed' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross_hashed' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross_hashed' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseCrossHashed", indices=indices, values=values, shapes=shapes,
                             dense_inputs=dense_inputs,
                             num_buckets=num_buckets, strong_hash=strong_hash,
                             salt=salt, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "sparse_types",
              _op.get_attr("sparse_types"), "dense_types",
              _op.get_attr("dense_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseCrossHashed", _inputs_flat, _attrs, _result)
  _result = _SparseCrossHashedOutput._make(_result)
  return _result

SparseCrossHashed = tf_export("raw_ops.SparseCrossHashed")(_ops.to_raw_op(sparse_cross_hashed))


def sparse_cross_hashed_eager_fallback(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, num_buckets: Annotated[Any, _atypes.Int64], strong_hash: Annotated[Any, _atypes.Bool], salt: Annotated[Any, _atypes.Int64], name, ctx):
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross_hashed' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross_hashed' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross_hashed' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  _attr_sparse_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
  shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
  num_buckets = _ops.convert_to_tensor(num_buckets, _dtypes.int64)
  strong_hash = _ops.convert_to_tensor(strong_hash, _dtypes.bool)
  salt = _ops.convert_to_tensor(salt, _dtypes.int64)
  _inputs_flat = list(indices) + list(values) + list(shapes) + list(dense_inputs) + [num_buckets, strong_hash, salt]
  _attrs = ("N", _attr_N, "sparse_types", _attr_sparse_types, "dense_types",
  _attr_dense_types)
  _result = _execute.execute(b"SparseCrossHashed", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseCrossHashed", _inputs_flat, _attrs, _result)
  _result = _SparseCrossHashedOutput._make(_result)
  return _result

_SparseCrossV2Output = collections.namedtuple(
    "SparseCrossV2",
    ["output_indices", "output_values", "output_shape"])


def sparse_cross_v2(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, sep: Annotated[Any, _atypes.String], name=None):
  r"""Generates sparse cross from a list of sparse and dense tensors.

  The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
  representing features of one feature column. It outputs a 2D `SparseTensor` with
  the batchwise crosses of these features.

  For example, if the inputs are

      inputs[0]: SparseTensor with shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      inputs[1]: SparseTensor with shape = [2, 1]
      [0, 0]: "d"
      [1, 0]: "e"

      inputs[2]: Tensor [["f"], ["g"]]

  then the output will be

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  if hashed_output=true then the output will be

      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))

  Args:
    indices: A list of `Tensor` objects with type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list of `Tensor` objects with types from: `int64`, `string`.
      1-D.   values of each `SparseTensor`.
    shapes: A list with the same length as `indices` of `Tensor` objects with type `int64`.
      1-D.   Shapes of each `SparseTensor`.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      2-D.    Columns represented by dense `Tensor`.
    sep: A `Tensor` of type `string`.
      string used when joining a list of string inputs, can be used as separator later.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor` of type `string`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseCrossV2", name, indices, values, shapes, dense_inputs,
        sep)
      _result = _SparseCrossV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_cross_v2_eager_fallback(
          indices, values, shapes, dense_inputs, sep, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross_v2' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross_v2' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross_v2' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseCrossV2", indices=indices, values=values, shapes=shapes,
                         dense_inputs=dense_inputs, sep=sep, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "sparse_types",
              _op.get_attr("sparse_types"), "dense_types",
              _op.get_attr("dense_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseCrossV2", _inputs_flat, _attrs, _result)
  _result = _SparseCrossV2Output._make(_result)
  return _result

SparseCrossV2 = tf_export("raw_ops.SparseCrossV2")(_ops.to_raw_op(sparse_cross_v2))


def sparse_cross_v2_eager_fallback(indices: Annotated[List[Any], _atypes.Int64], values, shapes: Annotated[List[Any], _atypes.Int64], dense_inputs, sep: Annotated[Any, _atypes.String], name, ctx):
  if not isinstance(indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'indices' argument to "
        "'sparse_cross_v2' Op, not %r." % indices)
  _attr_N = len(indices)
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'sparse_cross_v2' Op, not %r." % shapes)
  if len(shapes) != _attr_N:
    raise ValueError(
        "List argument 'shapes' to 'sparse_cross_v2' Op with length %d "
        "must match length %d of argument 'indices'." %
        (len(shapes), _attr_N))
  _attr_sparse_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
  indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
  shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
  sep = _ops.convert_to_tensor(sep, _dtypes.string)
  _inputs_flat = list(indices) + list(values) + list(shapes) + list(dense_inputs) + [sep]
  _attrs = ("N", _attr_N, "sparse_types", _attr_sparse_types, "dense_types",
  _attr_dense_types)
  _result = _execute.execute(b"SparseCrossV2", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseCrossV2", _inputs_flat, _attrs, _result)
  _result = _SparseCrossV2Output._make(_result)
  return _result


TV_SparseDenseCwiseAdd_T = TypeVar("TV_SparseDenseCwiseAdd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_dense_cwise_add(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseAdd_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseAdd_T], name=None) -> Annotated[Any, TV_SparseDenseCwiseAdd_T]:
  r"""Adds up a SparseTensor and a dense Tensor, using these special rules:

  (1) Broadcasts the dense side to have the same shape as the sparse side, if
      eligible;
  (2) Then, only the dense values pointed to by the indices of the SparseTensor
      participate in the cwise addition.

  By these rules, the result is a logical SparseTensor with exactly the same
  indices and shape, but possibly with different non-zero values.  The output of
  this Op is the resultant non-zero values.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseDenseCwiseAdd", name, sp_indices, sp_values, sp_shape,
        dense)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_dense_cwise_add_eager_fallback(
          sp_indices, sp_values, sp_shape, dense, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseDenseCwiseAdd", sp_indices=sp_indices, sp_values=sp_values,
                               sp_shape=sp_shape, dense=dense, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseDenseCwiseAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseDenseCwiseAdd = tf_export("raw_ops.SparseDenseCwiseAdd")(_ops.to_raw_op(sparse_dense_cwise_add))


def sparse_dense_cwise_add_eager_fallback(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseAdd_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseAdd_T], name, ctx) -> Annotated[Any, TV_SparseDenseCwiseAdd_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sp_values, dense], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (sp_values, dense) = _inputs_T
  sp_indices = _ops.convert_to_tensor(sp_indices, _dtypes.int64)
  sp_shape = _ops.convert_to_tensor(sp_shape, _dtypes.int64)
  _inputs_flat = [sp_indices, sp_values, sp_shape, dense]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseDenseCwiseAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseDenseCwiseAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseDenseCwiseDiv_T = TypeVar("TV_SparseDenseCwiseDiv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_dense_cwise_div(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseDiv_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseDiv_T], name=None) -> Annotated[Any, TV_SparseDenseCwiseDiv_T]:
  r"""Component-wise divides a SparseTensor by a dense Tensor.

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseDenseCwiseDiv", name, sp_indices, sp_values, sp_shape,
        dense)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_dense_cwise_div_eager_fallback(
          sp_indices, sp_values, sp_shape, dense, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseDenseCwiseDiv", sp_indices=sp_indices, sp_values=sp_values,
                               sp_shape=sp_shape, dense=dense, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseDenseCwiseDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseDenseCwiseDiv = tf_export("raw_ops.SparseDenseCwiseDiv")(_ops.to_raw_op(sparse_dense_cwise_div))


def sparse_dense_cwise_div_eager_fallback(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseDiv_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseDiv_T], name, ctx) -> Annotated[Any, TV_SparseDenseCwiseDiv_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sp_values, dense], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (sp_values, dense) = _inputs_T
  sp_indices = _ops.convert_to_tensor(sp_indices, _dtypes.int64)
  sp_shape = _ops.convert_to_tensor(sp_shape, _dtypes.int64)
  _inputs_flat = [sp_indices, sp_values, sp_shape, dense]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseDenseCwiseDiv", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseDenseCwiseDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseDenseCwiseMul_T = TypeVar("TV_SparseDenseCwiseMul_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_dense_cwise_mul(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseMul_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseMul_T], name=None) -> Annotated[Any, TV_SparseDenseCwiseMul_T]:
  r"""Component-wise multiplies a SparseTensor by a dense Tensor.

  The output locations corresponding to the implicitly zero elements in the sparse
  tensor will be zero (i.e., will not take up storage space), regardless of the
  contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseDenseCwiseMul", name, sp_indices, sp_values, sp_shape,
        dense)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_dense_cwise_mul_eager_fallback(
          sp_indices, sp_values, sp_shape, dense, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseDenseCwiseMul", sp_indices=sp_indices, sp_values=sp_values,
                               sp_shape=sp_shape, dense=dense, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseDenseCwiseMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseDenseCwiseMul = tf_export("raw_ops.SparseDenseCwiseMul")(_ops.to_raw_op(sparse_dense_cwise_mul))


def sparse_dense_cwise_mul_eager_fallback(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseDenseCwiseMul_T], sp_shape: Annotated[Any, _atypes.Int64], dense: Annotated[Any, TV_SparseDenseCwiseMul_T], name, ctx) -> Annotated[Any, TV_SparseDenseCwiseMul_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sp_values, dense], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (sp_values, dense) = _inputs_T
  sp_indices = _ops.convert_to_tensor(sp_indices, _dtypes.int64)
  sp_shape = _ops.convert_to_tensor(sp_shape, _dtypes.int64)
  _inputs_flat = [sp_indices, sp_values, sp_shape, dense]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseDenseCwiseMul", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseDenseCwiseMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseFillEmptyRowsOutput = collections.namedtuple(
    "SparseFillEmptyRows",
    ["output_indices", "output_values", "empty_row_indicator", "reverse_index_map"])


TV_SparseFillEmptyRows_T = TypeVar("TV_SparseFillEmptyRows_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_fill_empty_rows(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseFillEmptyRows_T], dense_shape: Annotated[Any, _atypes.Int64], default_value: Annotated[Any, TV_SparseFillEmptyRows_T], name=None):
  r"""Fills empty rows in the input 2-D `SparseTensor` with a default value.

  The input `SparseTensor` is represented via the tuple of inputs
  (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
  same `dense_shape` but with indices `output_indices` and values
  `output_values`.

  This op inserts a single entry for every row that doesn't have any values.
  The index is created as `[row, 0, ..., 0]` and the inserted value
  is `default_value`.

  For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

      [0, 1]: a
      [0, 3]: b
      [1, 0]: default_value
      [2, 0]: c
      [3, 1]: d
      [4, 0]: default_value

  The output `SparseTensor` will be in row-major order and will have the
  same shape as the input.

  This op also returns an indicator vector shaped `[dense_shape[0]]` such that

      empty_row_indicator[i] = True iff row i was an empty row.

  And a reverse index map vector shaped `[indices.shape[0]]` that is used during
  backpropagation,

      reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]

  Args:
    indices: A `Tensor` of type `int64`.
      2-D. the indices of the sparse tensor.
    values: A `Tensor`. 1-D. the values of the sparse tensor.
    dense_shape: A `Tensor` of type `int64`.
      1-D. the shape of the sparse tensor.
    default_value: A `Tensor`. Must have the same type as `values`.
      0-D. default value to insert into location `[row, 0, ..., 0]`
        for rows missing from the input sparse tensor.
      output indices: 2-D. the indices of the filled sparse tensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, empty_row_indicator, reverse_index_map).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `values`.
    empty_row_indicator: A `Tensor` of type `bool`.
    reverse_index_map: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseFillEmptyRows", name, indices, values, dense_shape,
        default_value)
      _result = _SparseFillEmptyRowsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_fill_empty_rows_eager_fallback(
          indices, values, dense_shape, default_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseFillEmptyRows", indices=indices, values=values,
                               dense_shape=dense_shape,
                               default_value=default_value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseFillEmptyRows", _inputs_flat, _attrs, _result)
  _result = _SparseFillEmptyRowsOutput._make(_result)
  return _result

SparseFillEmptyRows = tf_export("raw_ops.SparseFillEmptyRows")(_ops.to_raw_op(sparse_fill_empty_rows))


def sparse_fill_empty_rows_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseFillEmptyRows_T], dense_shape: Annotated[Any, _atypes.Int64], default_value: Annotated[Any, TV_SparseFillEmptyRows_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([values, default_value], ctx, [])
  (values, default_value) = _inputs_T
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [indices, values, dense_shape, default_value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseFillEmptyRows", 4, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseFillEmptyRows", _inputs_flat, _attrs, _result)
  _result = _SparseFillEmptyRowsOutput._make(_result)
  return _result

_SparseFillEmptyRowsGradOutput = collections.namedtuple(
    "SparseFillEmptyRowsGrad",
    ["d_values", "d_default_value"])


TV_SparseFillEmptyRowsGrad_T = TypeVar("TV_SparseFillEmptyRowsGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_fill_empty_rows_grad(reverse_index_map: Annotated[Any, _atypes.Int64], grad_values: Annotated[Any, TV_SparseFillEmptyRowsGrad_T], name=None):
  r"""The gradient of SparseFillEmptyRows.

  Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
  shaped `[N_full]`, where `N_full >= N` and copies data into either
  `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
  `d_default_value` is a scalar.

    d_values[j] = grad_values[reverse_index_map[j]]
    d_default_value = sum_{k : 0 .. N_full - 1} (
       grad_values[k] * 1{k not in reverse_index_map})

  Args:
    reverse_index_map: A `Tensor` of type `int64`.
      1-D.  The reverse index map from SparseFillEmptyRows.
    grad_values: A `Tensor`. 1-D.  The gradients from backprop.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_values, d_default_value).

    d_values: A `Tensor`. Has the same type as `grad_values`.
    d_default_value: A `Tensor`. Has the same type as `grad_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseFillEmptyRowsGrad", name, reverse_index_map, grad_values)
      _result = _SparseFillEmptyRowsGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_fill_empty_rows_grad_eager_fallback(
          reverse_index_map, grad_values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseFillEmptyRowsGrad", reverse_index_map=reverse_index_map,
                                   grad_values=grad_values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseFillEmptyRowsGrad", _inputs_flat, _attrs, _result)
  _result = _SparseFillEmptyRowsGradOutput._make(_result)
  return _result

SparseFillEmptyRowsGrad = tf_export("raw_ops.SparseFillEmptyRowsGrad")(_ops.to_raw_op(sparse_fill_empty_rows_grad))


def sparse_fill_empty_rows_grad_eager_fallback(reverse_index_map: Annotated[Any, _atypes.Int64], grad_values: Annotated[Any, TV_SparseFillEmptyRowsGrad_T], name, ctx):
  _attr_T, (grad_values,) = _execute.args_to_matching_eager([grad_values], ctx, [])
  reverse_index_map = _ops.convert_to_tensor(reverse_index_map, _dtypes.int64)
  _inputs_flat = [reverse_index_map, grad_values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseFillEmptyRowsGrad", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseFillEmptyRowsGrad", _inputs_flat, _attrs, _result)
  _result = _SparseFillEmptyRowsGradOutput._make(_result)
  return _result


TV_SparseReduceMax_T = TypeVar("TV_SparseReduceMax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_reduce_max(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceMax_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims:bool=False, name=None) -> Annotated[Any, TV_SparseReduceMax_T]:
  r"""Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReduceMax", name, input_indices, input_values,
        input_shape, reduction_axes, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reduce_max_eager_fallback(
          input_indices, input_values, input_shape, reduction_axes,
          keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReduceMax", input_indices=input_indices,
                           input_values=input_values, input_shape=input_shape,
                           reduction_axes=reduction_axes, keep_dims=keep_dims,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReduceMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseReduceMax = tf_export("raw_ops.SparseReduceMax")(_ops.to_raw_op(sparse_reduce_max))


def sparse_reduce_max_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceMax_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims: bool, name, ctx) -> Annotated[Any, TV_SparseReduceMax_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  reduction_axes = _ops.convert_to_tensor(reduction_axes, _dtypes.int32)
  _inputs_flat = [input_indices, input_values, input_shape, reduction_axes]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T)
  _result = _execute.execute(b"SparseReduceMax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReduceMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseReduceMaxSparseOutput = collections.namedtuple(
    "SparseReduceMaxSparse",
    ["output_indices", "output_values", "output_shape"])


TV_SparseReduceMaxSparse_T = TypeVar("TV_SparseReduceMaxSparse_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_reduce_max_sparse(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceMaxSparse_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims:bool=False, name=None):
  r"""Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
  SparseTensor.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `input_values`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReduceMaxSparse", name, input_indices, input_values,
        input_shape, reduction_axes, "keep_dims", keep_dims)
      _result = _SparseReduceMaxSparseOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reduce_max_sparse_eager_fallback(
          input_indices, input_values, input_shape, reduction_axes,
          keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReduceMaxSparse", input_indices=input_indices,
                                 input_values=input_values,
                                 input_shape=input_shape,
                                 reduction_axes=reduction_axes,
                                 keep_dims=keep_dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReduceMaxSparse", _inputs_flat, _attrs, _result)
  _result = _SparseReduceMaxSparseOutput._make(_result)
  return _result

SparseReduceMaxSparse = tf_export("raw_ops.SparseReduceMaxSparse")(_ops.to_raw_op(sparse_reduce_max_sparse))


def sparse_reduce_max_sparse_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceMaxSparse_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims: bool, name, ctx):
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  reduction_axes = _ops.convert_to_tensor(reduction_axes, _dtypes.int32)
  _inputs_flat = [input_indices, input_values, input_shape, reduction_axes]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T)
  _result = _execute.execute(b"SparseReduceMaxSparse", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReduceMaxSparse", _inputs_flat, _attrs, _result)
  _result = _SparseReduceMaxSparseOutput._make(_result)
  return _result


TV_SparseReduceSum_T = TypeVar("TV_SparseReduceSum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_reduce_sum(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceSum_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims:bool=False, name=None) -> Annotated[Any, TV_SparseReduceSum_T]:
  r"""Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReduceSum", name, input_indices, input_values,
        input_shape, reduction_axes, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reduce_sum_eager_fallback(
          input_indices, input_values, input_shape, reduction_axes,
          keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReduceSum", input_indices=input_indices,
                           input_values=input_values, input_shape=input_shape,
                           reduction_axes=reduction_axes, keep_dims=keep_dims,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReduceSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseReduceSum = tf_export("raw_ops.SparseReduceSum")(_ops.to_raw_op(sparse_reduce_sum))


def sparse_reduce_sum_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceSum_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims: bool, name, ctx) -> Annotated[Any, TV_SparseReduceSum_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  reduction_axes = _ops.convert_to_tensor(reduction_axes, _dtypes.int32)
  _inputs_flat = [input_indices, input_values, input_shape, reduction_axes]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T)
  _result = _execute.execute(b"SparseReduceSum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReduceSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseReduceSumSparseOutput = collections.namedtuple(
    "SparseReduceSumSparse",
    ["output_indices", "output_values", "output_shape"])


TV_SparseReduceSumSparse_T = TypeVar("TV_SparseReduceSumSparse_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_reduce_sum_sparse(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceSumSparse_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims:bool=False, name=None):
  r"""Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
  SparseTensor.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `input_values`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReduceSumSparse", name, input_indices, input_values,
        input_shape, reduction_axes, "keep_dims", keep_dims)
      _result = _SparseReduceSumSparseOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reduce_sum_sparse_eager_fallback(
          input_indices, input_values, input_shape, reduction_axes,
          keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReduceSumSparse", input_indices=input_indices,
                                 input_values=input_values,
                                 input_shape=input_shape,
                                 reduction_axes=reduction_axes,
                                 keep_dims=keep_dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReduceSumSparse", _inputs_flat, _attrs, _result)
  _result = _SparseReduceSumSparseOutput._make(_result)
  return _result

SparseReduceSumSparse = tf_export("raw_ops.SparseReduceSumSparse")(_ops.to_raw_op(sparse_reduce_sum_sparse))


def sparse_reduce_sum_sparse_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReduceSumSparse_T], input_shape: Annotated[Any, _atypes.Int64], reduction_axes: Annotated[Any, _atypes.Int32], keep_dims: bool, name, ctx):
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  reduction_axes = _ops.convert_to_tensor(reduction_axes, _dtypes.int32)
  _inputs_flat = [input_indices, input_values, input_shape, reduction_axes]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T)
  _result = _execute.execute(b"SparseReduceSumSparse", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReduceSumSparse", _inputs_flat, _attrs, _result)
  _result = _SparseReduceSumSparseOutput._make(_result)
  return _result

_SparseReorderOutput = collections.namedtuple(
    "SparseReorder",
    ["output_indices", "output_values"])


TV_SparseReorder_T = TypeVar("TV_SparseReorder_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_reorder(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReorder_T], input_shape: Annotated[Any, _atypes.Int64], name=None):
  r"""Reorders a SparseTensor into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering along
  increasing dimension number. The only time ordering can be violated is during
  manual manipulation of the indices and values vectors to add entries.

  Reordering does not affect the shape of the SparseTensor.

  If the tensor has rank `R` and `N` non-empty values, `input_indices` has
  shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `input_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReorder", name, input_indices, input_values, input_shape)
      _result = _SparseReorderOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reorder_eager_fallback(
          input_indices, input_values, input_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReorder", input_indices=input_indices,
                         input_values=input_values, input_shape=input_shape,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReorder", _inputs_flat, _attrs, _result)
  _result = _SparseReorderOutput._make(_result)
  return _result

SparseReorder = tf_export("raw_ops.SparseReorder")(_ops.to_raw_op(sparse_reorder))


def sparse_reorder_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_values: Annotated[Any, TV_SparseReorder_T], input_shape: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  _inputs_flat = [input_indices, input_values, input_shape]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseReorder", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReorder", _inputs_flat, _attrs, _result)
  _result = _SparseReorderOutput._make(_result)
  return _result

_SparseReshapeOutput = collections.namedtuple(
    "SparseReshape",
    ["output_indices", "output_shape"])


def sparse_reshape(input_indices: Annotated[Any, _atypes.Int64], input_shape: Annotated[Any, _atypes.Int64], new_shape: Annotated[Any, _atypes.Int64], name=None):
  r"""Reshapes a SparseTensor to represent values in a new dense shape.

  This operation has the same semantics as reshape on the represented dense
  tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

  If one component of `new_shape` is the special value -1, the size of that
  dimension is computed so that the total dense size remains constant.  At
  most one component of `new_shape` can be -1.  The number of dense elements
  implied by `new_shape` must be the same as the number of dense elements
  originally implied by `input_shape`.

  Reshaping does not affect the order of values in the SparseTensor.

  If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
  has length `R_out`, then `input_indices` has shape `[N, R_in]`,
  `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
  `output_shape` has length `R_out`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R_in` matrix with the indices of non-empty values in a
      SparseTensor.
    input_shape: A `Tensor` of type `int64`.
      1-D.  `R_in` vector with the input SparseTensor's dense shape.
    new_shape: A `Tensor` of type `int64`.
      1-D.  `R_out` vector with the requested new dense shape.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseReshape", name, input_indices, input_shape, new_shape)
      _result = _SparseReshapeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_reshape_eager_fallback(
          input_indices, input_shape, new_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseReshape", input_indices=input_indices, input_shape=input_shape,
                         new_shape=new_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseReshape", _inputs_flat, _attrs, _result)
  _result = _SparseReshapeOutput._make(_result)
  return _result

SparseReshape = tf_export("raw_ops.SparseReshape")(_ops.to_raw_op(sparse_reshape))


def sparse_reshape_eager_fallback(input_indices: Annotated[Any, _atypes.Int64], input_shape: Annotated[Any, _atypes.Int64], new_shape: Annotated[Any, _atypes.Int64], name, ctx):
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
  new_shape = _ops.convert_to_tensor(new_shape, _dtypes.int64)
  _inputs_flat = [input_indices, input_shape, new_shape]
  _attrs = None
  _result = _execute.execute(b"SparseReshape", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseReshape", _inputs_flat, _attrs, _result)
  _result = _SparseReshapeOutput._make(_result)
  return _result

_SparseSliceOutput = collections.namedtuple(
    "SparseSlice",
    ["output_indices", "output_values", "output_shape"])


TV_SparseSlice_T = TypeVar("TV_SparseSlice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_slice(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseSlice_T], shape: Annotated[Any, _atypes.Int64], start: Annotated[Any, _atypes.Int64], size: Annotated[Any, _atypes.Int64], name=None):
  r"""Slice a `SparseTensor` based on the `start` and `size`.

  For example, if the input is

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
      [    a  ]
      [b c    ]

      sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
      [ d e  ]
      [      ]

  Args:
    indices: A `Tensor` of type `int64`.
      2-D tensor represents the indices of the sparse tensor.
    values: A `Tensor`. 1-D tensor represents the values of the sparse tensor.
    shape: A `Tensor` of type `int64`.
      1-D. tensor represents the shape of the sparse tensor.
    start: A `Tensor` of type `int64`.
      1-D. tensor represents the start of the slice.
    size: A `Tensor` of type `int64`.
      1-D. tensor represents the size of the slice.
      output indices: A list of 1-D tensors represents the indices of the output
      sparse tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `values`.
    output_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSlice", name, indices, values, shape, start, size)
      _result = _SparseSliceOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_slice_eager_fallback(
          indices, values, shape, start, size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSlice", indices=indices, values=values, shape=shape,
                       start=start, size=size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSlice", _inputs_flat, _attrs, _result)
  _result = _SparseSliceOutput._make(_result)
  return _result

SparseSlice = tf_export("raw_ops.SparseSlice")(_ops.to_raw_op(sparse_slice))


def sparse_slice_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseSlice_T], shape: Annotated[Any, _atypes.Int64], start: Annotated[Any, _atypes.Int64], size: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  shape = _ops.convert_to_tensor(shape, _dtypes.int64)
  start = _ops.convert_to_tensor(start, _dtypes.int64)
  size = _ops.convert_to_tensor(size, _dtypes.int64)
  _inputs_flat = [indices, values, shape, start, size]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseSlice", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSlice", _inputs_flat, _attrs, _result)
  _result = _SparseSliceOutput._make(_result)
  return _result


TV_SparseSliceGrad_T = TypeVar("TV_SparseSliceGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_slice_grad(backprop_val_grad: Annotated[Any, TV_SparseSliceGrad_T], input_indices: Annotated[Any, _atypes.Int64], input_start: Annotated[Any, _atypes.Int64], output_indices: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, TV_SparseSliceGrad_T]:
  r"""The gradient operator for the SparseSlice op.

  This op takes in the upstream gradient w.r.t. non-empty values of
  the sliced `SparseTensor`, and outputs the gradients w.r.t.
  the non-empty values of input `SparseTensor`.

  Args:
    backprop_val_grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D. The gradient with respect to
      the non-empty values of the sliced `SparseTensor`.
    input_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the input `SparseTensor`.
    input_start: A `Tensor` of type `int64`.
      1-D. tensor represents the start of the slice.
    output_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the sliced `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `backprop_val_grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSliceGrad", name, backprop_val_grad, input_indices,
        input_start, output_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_slice_grad_eager_fallback(
          backprop_val_grad, input_indices, input_start, output_indices,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSliceGrad", backprop_val_grad=backprop_val_grad,
                           input_indices=input_indices,
                           input_start=input_start,
                           output_indices=output_indices, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSliceGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSliceGrad = tf_export("raw_ops.SparseSliceGrad")(_ops.to_raw_op(sparse_slice_grad))


def sparse_slice_grad_eager_fallback(backprop_val_grad: Annotated[Any, TV_SparseSliceGrad_T], input_indices: Annotated[Any, _atypes.Int64], input_start: Annotated[Any, _atypes.Int64], output_indices: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, TV_SparseSliceGrad_T]:
  _attr_T, (backprop_val_grad,) = _execute.args_to_matching_eager([backprop_val_grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
  input_start = _ops.convert_to_tensor(input_start, _dtypes.int64)
  output_indices = _ops.convert_to_tensor(output_indices, _dtypes.int64)
  _inputs_flat = [backprop_val_grad, input_indices, input_start, output_indices]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseSliceGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSliceGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSoftmax_T = TypeVar("TV_SparseSoftmax_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def sparse_softmax(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseSoftmax_T], sp_shape: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, TV_SparseSoftmax_T]:
  r"""Applies softmax to a batched N-D `SparseTensor`.

  The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
  (where `N >= 2`), and with indices sorted in the canonical lexicographic order.

  This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
  logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
  zero elements do not participate*.  Specifically, the algorithm is equivalent
  to the following:

    (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
        with shape `[B, C]`, along the size-C dimension;
    (2) Masks out the original implicitly-zero locations;
    (3) Renormalizes the remaining elements.

  Hence, the `SparseTensor` result has exactly the same non-zero indices and
  shape.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `NNZ x R` matrix with the indices of non-empty values in a
      SparseTensor, in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSoftmax", name, sp_indices, sp_values, sp_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_softmax_eager_fallback(
          sp_indices, sp_values, sp_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSoftmax", sp_indices=sp_indices, sp_values=sp_values,
                         sp_shape=sp_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSoftmax = tf_export("raw_ops.SparseSoftmax")(_ops.to_raw_op(sparse_softmax))


def sparse_softmax_eager_fallback(sp_indices: Annotated[Any, _atypes.Int64], sp_values: Annotated[Any, TV_SparseSoftmax_T], sp_shape: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, TV_SparseSoftmax_T]:
  _attr_T, (sp_values,) = _execute.args_to_matching_eager([sp_values], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  sp_indices = _ops.convert_to_tensor(sp_indices, _dtypes.int64)
  sp_shape = _ops.convert_to_tensor(sp_shape, _dtypes.int64)
  _inputs_flat = [sp_indices, sp_values, sp_shape]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseSoftmax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseSparseMaximumOutput = collections.namedtuple(
    "SparseSparseMaximum",
    ["output_indices", "output_values"])


TV_SparseSparseMaximum_T = TypeVar("TV_SparseSparseMaximum_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_sparse_maximum(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseSparseMaximum_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseSparseMaximum_T], b_shape: Annotated[Any, _atypes.Int64], name=None):
  r"""Returns the element-wise max of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, in the canonical lexicographic ordering.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `a_indices`.
    a_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    b_indices: A `Tensor` of type `int64`.
      counterpart to `a_indices` for the other operand.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      counterpart to `a_values` for the other operand; must be of the same dtype.
    b_shape: A `Tensor` of type `int64`.
      counterpart to `a_shape` for the other operand; the two shapes must be equal.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `a_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSparseMaximum", name, a_indices, a_values, a_shape,
        b_indices, b_values, b_shape)
      _result = _SparseSparseMaximumOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_sparse_maximum_eager_fallback(
          a_indices, a_values, a_shape, b_indices, b_values, b_shape,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSparseMaximum", a_indices=a_indices, a_values=a_values,
                               a_shape=a_shape, b_indices=b_indices,
                               b_values=b_values, b_shape=b_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSparseMaximum", _inputs_flat, _attrs, _result)
  _result = _SparseSparseMaximumOutput._make(_result)
  return _result

SparseSparseMaximum = tf_export("raw_ops.SparseSparseMaximum")(_ops.to_raw_op(sparse_sparse_maximum))


def sparse_sparse_maximum_eager_fallback(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseSparseMaximum_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseSparseMaximum_T], b_shape: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (a_values, b_values) = _inputs_T
  a_indices = _ops.convert_to_tensor(a_indices, _dtypes.int64)
  a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
  b_indices = _ops.convert_to_tensor(b_indices, _dtypes.int64)
  b_shape = _ops.convert_to_tensor(b_shape, _dtypes.int64)
  _inputs_flat = [a_indices, a_values, a_shape, b_indices, b_values, b_shape]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseSparseMaximum", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSparseMaximum", _inputs_flat, _attrs, _result)
  _result = _SparseSparseMaximumOutput._make(_result)
  return _result

_SparseSparseMinimumOutput = collections.namedtuple(
    "SparseSparseMinimum",
    ["output_indices", "output_values"])


TV_SparseSparseMinimum_T = TypeVar("TV_SparseSparseMinimum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sparse_sparse_minimum(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseSparseMinimum_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseSparseMinimum_T], b_shape: Annotated[Any, _atypes.Int64], name=None):
  r"""Returns the element-wise min of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, in the canonical lexicographic ordering.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `a_indices`.
    a_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    b_indices: A `Tensor` of type `int64`.
      counterpart to `a_indices` for the other operand.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      counterpart to `a_values` for the other operand; must be of the same dtype.
    b_shape: A `Tensor` of type `int64`.
      counterpart to `a_shape` for the other operand; the two shapes must be equal.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `a_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSparseMinimum", name, a_indices, a_values, a_shape,
        b_indices, b_values, b_shape)
      _result = _SparseSparseMinimumOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_sparse_minimum_eager_fallback(
          a_indices, a_values, a_shape, b_indices, b_values, b_shape,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSparseMinimum", a_indices=a_indices, a_values=a_values,
                               a_shape=a_shape, b_indices=b_indices,
                               b_values=b_values, b_shape=b_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSparseMinimum", _inputs_flat, _attrs, _result)
  _result = _SparseSparseMinimumOutput._make(_result)
  return _result

SparseSparseMinimum = tf_export("raw_ops.SparseSparseMinimum")(_ops.to_raw_op(sparse_sparse_minimum))


def sparse_sparse_minimum_eager_fallback(a_indices: Annotated[Any, _atypes.Int64], a_values: Annotated[Any, TV_SparseSparseMinimum_T], a_shape: Annotated[Any, _atypes.Int64], b_indices: Annotated[Any, _atypes.Int64], b_values: Annotated[Any, TV_SparseSparseMinimum_T], b_shape: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (a_values, b_values) = _inputs_T
  a_indices = _ops.convert_to_tensor(a_indices, _dtypes.int64)
  a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
  b_indices = _ops.convert_to_tensor(b_indices, _dtypes.int64)
  b_shape = _ops.convert_to_tensor(b_shape, _dtypes.int64)
  _inputs_flat = [a_indices, a_values, a_shape, b_indices, b_values, b_shape]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SparseSparseMinimum", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSparseMinimum", _inputs_flat, _attrs, _result)
  _result = _SparseSparseMinimumOutput._make(_result)
  return _result

_SparseSplitOutput = collections.namedtuple(
    "SparseSplit",
    ["output_indices", "output_values", "output_shape"])


TV_SparseSplit_T = TypeVar("TV_SparseSplit_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_split(split_dim: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseSplit_T], shape: Annotated[Any, _atypes.Int64], num_split: int, name=None):
  r"""Split a `SparseTensor` into `num_split` tensors along one dimension.

  If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
  `[0 : shape[split_dim] % num_split]` gets one extra dimension.
  For example, if `split_dim = 1` and `num_split = 2` and the input is

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] = shape = [2, 4]
      [    a  ]
      [b c    ]

      output_tensor[1] = shape = [2, 3]
      [ d e  ]
      [      ]

  Args:
    split_dim: A `Tensor` of type `int64`.
      0-D.  The dimension along which to split.  Must be in the range
      `[0, rank(shape))`.
    indices: A `Tensor` of type `int64`.
      2-D tensor represents the indices of the sparse tensor.
    values: A `Tensor`. 1-D tensor represents the values of the sparse tensor.
    shape: A `Tensor` of type `int64`.
      1-D. tensor represents the shape of the sparse tensor.
      output indices: A list of 1-D tensors represents the indices of the output
      sparse tensors.
    num_split: An `int` that is `>= 1`. The number of ways to split.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).

    output_indices: A list of `num_split` `Tensor` objects with type `int64`.
    output_values: A list of `num_split` `Tensor` objects with the same type as `values`.
    output_shape: A list of `num_split` `Tensor` objects with type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSplit", name, split_dim, indices, values, shape,
        "num_split", num_split)
      _result = _SparseSplitOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_split_eager_fallback(
          split_dim, indices, values, shape, num_split=num_split, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_split = _execute.make_int(num_split, "num_split")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSplit", split_dim=split_dim, indices=indices, values=values,
                       shape=shape, num_split=num_split, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_split", _op._get_attr_int("num_split"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSplit", _inputs_flat, _attrs, _result)
  _result = [_result[:num_split]] + _result[num_split:]
  _result = _result[:1] + [_result[1:1 + num_split]] + _result[1 + num_split:]
  _result = _result[:2] + [_result[2:]]
  _result = _SparseSplitOutput._make(_result)
  return _result

SparseSplit = tf_export("raw_ops.SparseSplit")(_ops.to_raw_op(sparse_split))


def sparse_split_eager_fallback(split_dim: Annotated[Any, _atypes.Int64], indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseSplit_T], shape: Annotated[Any, _atypes.Int64], num_split: int, name, ctx):
  num_split = _execute.make_int(num_split, "num_split")
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  split_dim = _ops.convert_to_tensor(split_dim, _dtypes.int64)
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  shape = _ops.convert_to_tensor(shape, _dtypes.int64)
  _inputs_flat = [split_dim, indices, values, shape]
  _attrs = ("num_split", num_split, "T", _attr_T)
  _result = _execute.execute(b"SparseSplit", num_split + num_split +
                             num_split, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSplit", _inputs_flat, _attrs, _result)
  _result = [_result[:num_split]] + _result[num_split:]
  _result = _result[:1] + [_result[1:1 + num_split]] + _result[1 + num_split:]
  _result = _result[:2] + [_result[2:]]
  _result = _SparseSplitOutput._make(_result)
  return _result


TV_SparseTensorDenseAdd_T = TypeVar("TV_SparseTensorDenseAdd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseTensorDenseAdd_Tindices = TypeVar("TV_SparseTensorDenseAdd_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_tensor_dense_add(a_indices: Annotated[Any, TV_SparseTensorDenseAdd_Tindices], a_values: Annotated[Any, TV_SparseTensorDenseAdd_T], a_shape: Annotated[Any, TV_SparseTensorDenseAdd_Tindices], b: Annotated[Any, TV_SparseTensorDenseAdd_T], name=None) -> Annotated[Any, TV_SparseTensorDenseAdd_T]:
  r"""Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

  This Op does not require `a_indices` be sorted in standard lexicographic order.

  Args:
    a_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
    a_shape: A `Tensor`. Must have the same type as `a_indices`.
      1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
    b: A `Tensor`. Must have the same type as `a_values`.
      `ndims`-D Tensor.  With shape `a_shape`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseTensorDenseAdd", name, a_indices, a_values, a_shape, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_tensor_dense_add_eager_fallback(
          a_indices, a_values, a_shape, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTensorDenseAdd", a_indices=a_indices, a_values=a_values,
                                a_shape=a_shape, b=b, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseTensorDenseAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseTensorDenseAdd = tf_export("raw_ops.SparseTensorDenseAdd")(_ops.to_raw_op(sparse_tensor_dense_add))


def sparse_tensor_dense_add_eager_fallback(a_indices: Annotated[Any, TV_SparseTensorDenseAdd_Tindices], a_values: Annotated[Any, TV_SparseTensorDenseAdd_T], a_shape: Annotated[Any, TV_SparseTensorDenseAdd_Tindices], b: Annotated[Any, TV_SparseTensorDenseAdd_T], name, ctx) -> Annotated[Any, TV_SparseTensorDenseAdd_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (a_values, b) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([a_indices, a_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  (a_indices, a_shape) = _inputs_Tindices
  _inputs_flat = [a_indices, a_values, a_shape, b]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SparseTensorDenseAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseTensorDenseAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseTensorDenseMatMul_T = TypeVar("TV_SparseTensorDenseMatMul_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SparseTensorDenseMatMul_Tindices = TypeVar("TV_SparseTensorDenseMatMul_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_tensor_dense_mat_mul(a_indices: Annotated[Any, TV_SparseTensorDenseMatMul_Tindices], a_values: Annotated[Any, TV_SparseTensorDenseMatMul_T], a_shape: Annotated[Any, _atypes.Int64], b: Annotated[Any, TV_SparseTensorDenseMatMul_T], adjoint_a:bool=False, adjoint_b:bool=False, name=None) -> Annotated[Any, TV_SparseTensorDenseMatMul_T]:
  r"""Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

  No validity checking is performed on the indices of A.  However, the following
  input format is recommended for optimal behavior:

  if adjoint_a == false:
    A should be sorted in lexicographically increasing order.  Use SparseReorder
    if you're not sure.
  if adjoint_a == true:
    A should be sorted in order of increasing dimension 1 (i.e., "column major"
    order instead of "row major" order).

  Args:
    a_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
    a_values: A `Tensor`.
      1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
    b: A `Tensor`. Must have the same type as `a_values`.
      2-D.  A dense Matrix.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Use the adjoint of A in the matrix multiply.  If A is complex, this
      is transpose(conj(A)).  Otherwise it's transpose(A).
    adjoint_b: An optional `bool`. Defaults to `False`.
      Use the adjoint of B in the matrix multiply.  If B is complex, this
      is transpose(conj(B)).  Otherwise it's transpose(B).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseTensorDenseMatMul", name, a_indices, a_values, a_shape,
        b, "adjoint_a", adjoint_a, "adjoint_b", adjoint_b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_tensor_dense_mat_mul_eager_fallback(
          a_indices, a_values, a_shape, b, adjoint_a=adjoint_a,
          adjoint_b=adjoint_b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTensorDenseMatMul", a_indices=a_indices, a_values=a_values,
                                   a_shape=a_shape, b=b, adjoint_a=adjoint_a,
                                   adjoint_b=adjoint_b, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "adjoint_a",
              _op._get_attr_bool("adjoint_a"), "adjoint_b",
              _op._get_attr_bool("adjoint_b"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseTensorDenseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseTensorDenseMatMul = tf_export("raw_ops.SparseTensorDenseMatMul")(_ops.to_raw_op(sparse_tensor_dense_mat_mul))


def sparse_tensor_dense_mat_mul_eager_fallback(a_indices: Annotated[Any, TV_SparseTensorDenseMatMul_Tindices], a_values: Annotated[Any, TV_SparseTensorDenseMatMul_T], a_shape: Annotated[Any, _atypes.Int64], b: Annotated[Any, TV_SparseTensorDenseMatMul_T], adjoint_a: bool, adjoint_b: bool, name, ctx) -> Annotated[Any, TV_SparseTensorDenseMatMul_T]:
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b], ctx, [])
  (a_values, b) = _inputs_T
  _attr_Tindices, (a_indices,) = _execute.args_to_matching_eager([a_indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
  _inputs_flat = [a_indices, a_values, a_shape, b]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "adjoint_a", adjoint_a,
  "adjoint_b", adjoint_b)
  _result = _execute.execute(b"SparseTensorDenseMatMul", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseTensorDenseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseToDense_T = TypeVar("TV_SparseToDense_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_SparseToDense_Tindices = TypeVar("TV_SparseToDense_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_to_dense(sparse_indices: Annotated[Any, TV_SparseToDense_Tindices], output_shape: Annotated[Any, TV_SparseToDense_Tindices], sparse_values: Annotated[Any, TV_SparseToDense_T], default_value: Annotated[Any, TV_SparseToDense_T], validate_indices:bool=True, name=None) -> Annotated[Any, TV_SparseToDense_T]:
  r"""Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values` is a
  scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is true, these properties
  are checked during execution.

  Args:
    sparse_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
      index where `sparse_values[i]` will be placed.
    output_shape: A `Tensor`. Must have the same type as `sparse_indices`.
      1-D.  Shape of the dense output tensor.
    sparse_values: A `Tensor`.
      1-D.  Values corresponding to each row of `sparse_indices`,
      or a scalar value to be used for all sparse indices.
    default_value: A `Tensor`. Must have the same type as `sparse_values`.
      Scalar value to set for indices not specified in
      `sparse_indices`.
    validate_indices: An optional `bool`. Defaults to `True`.
      If true, indices are checked to make sure they are sorted in
      lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sparse_values`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseToDense", name, sparse_indices, output_shape,
        sparse_values, default_value, "validate_indices", validate_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_to_dense_eager_fallback(
          sparse_indices, output_shape, sparse_values, default_value,
          validate_indices=validate_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseToDense", sparse_indices=sparse_indices,
                         output_shape=output_shape,
                         sparse_values=sparse_values,
                         default_value=default_value,
                         validate_indices=validate_indices, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("validate_indices", _op._get_attr_bool("validate_indices"), "T",
              _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseToDense", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseToDense = tf_export("raw_ops.SparseToDense")(_ops.to_raw_op(sparse_to_dense))


def sparse_to_dense_eager_fallback(sparse_indices: Annotated[Any, TV_SparseToDense_Tindices], output_shape: Annotated[Any, TV_SparseToDense_Tindices], sparse_values: Annotated[Any, TV_SparseToDense_T], default_value: Annotated[Any, TV_SparseToDense_T], validate_indices: bool, name, ctx) -> Annotated[Any, TV_SparseToDense_T]:
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([sparse_values, default_value], ctx, [])
  (sparse_values, default_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([sparse_indices, output_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  (sparse_indices, output_shape) = _inputs_Tindices
  _inputs_flat = [sparse_indices, output_shape, sparse_values, default_value]
  _attrs = ("validate_indices", validate_indices, "T", _attr_T, "Tindices",
  _attr_Tindices)
  _result = _execute.execute(b"SparseToDense", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseToDense", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TakeManySparseFromTensorsMapOutput = collections.namedtuple(
    "TakeManySparseFromTensorsMap",
    ["sparse_indices", "sparse_values", "sparse_shape"])


TV_TakeManySparseFromTensorsMap_dtype = TypeVar("TV_TakeManySparseFromTensorsMap_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def take_many_sparse_from_tensors_map(sparse_handles: Annotated[Any, _atypes.Int64], dtype: TV_TakeManySparseFromTensorsMap_dtype, container:str="", shared_name:str="", name=None):
  r"""Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

  The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
  `N` is the minibatch size and the rows correspond to the output handles of
  `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
  original `SparseTensor` objects that went into the given input ops must all
  match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension on the left).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the handles represent an input, which is a `[2, 3]` matrix
  representing two original `SparseTensor` objects:

  ```
      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]
  ```

  and

  ```
      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]
  ```

  then the final `SparseTensor` will be:

  ```
      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]
  ```

  Args:
    sparse_handles: A `Tensor` of type `int64`.
      1-D, The `N` serialized `SparseTensor` objects.
      Shape: `[N]`.
    dtype: A `tf.DType`.
      The `dtype` of the `SparseTensor` objects stored in the
      `SparseTensorsMap`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` read by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` read by this op.
      It should not be blank; rather the `shared_name` or unique Operation name
      of the Op that created the original `SparseTensorsMap` should be used.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TakeManySparseFromTensorsMap", name, sparse_handles, "dtype",
        dtype, "container", container, "shared_name", shared_name)
      _result = _TakeManySparseFromTensorsMapOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return take_many_sparse_from_tensors_map_eager_fallback(
          sparse_handles, dtype=dtype, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TakeManySparseFromTensorsMap", sparse_handles=sparse_handles,
                                        dtype=dtype, container=container,
                                        shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TakeManySparseFromTensorsMap", _inputs_flat, _attrs, _result)
  _result = _TakeManySparseFromTensorsMapOutput._make(_result)
  return _result

TakeManySparseFromTensorsMap = tf_export("raw_ops.TakeManySparseFromTensorsMap")(_ops.to_raw_op(take_many_sparse_from_tensors_map))


def take_many_sparse_from_tensors_map_eager_fallback(sparse_handles: Annotated[Any, _atypes.Int64], dtype: TV_TakeManySparseFromTensorsMap_dtype, container: str, shared_name: str, name, ctx):
  dtype = _execute.make_type(dtype, "dtype")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  sparse_handles = _ops.convert_to_tensor(sparse_handles, _dtypes.int64)
  _inputs_flat = [sparse_handles]
  _attrs = ("dtype", dtype, "container", container, "shared_name",
  shared_name)
  _result = _execute.execute(b"TakeManySparseFromTensorsMap", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TakeManySparseFromTensorsMap", _inputs_flat, _attrs, _result)
  _result = _TakeManySparseFromTensorsMapOutput._make(_result)
  return _result

