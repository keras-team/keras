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

TV_EmptyTensorList_element_dtype = TypeVar("TV_EmptyTensorList_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_EmptyTensorList_shape_type = TypeVar("TV_EmptyTensorList_shape_type", _atypes.Int32, _atypes.Int64)

def empty_tensor_list(element_shape: Annotated[Any, TV_EmptyTensorList_shape_type], max_num_elements: Annotated[Any, _atypes.Int32], element_dtype: TV_EmptyTensorList_element_dtype, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates and returns an empty tensor list.

  All list elements must be tensors of dtype element_dtype and shape compatible
  with element_shape.

  handle: an empty tensor list.
  element_dtype: the type of elements in the list.
  element_shape: a shape compatible with that of elements in the list.

  Args:
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    max_num_elements: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EmptyTensorList", name, element_shape, max_num_elements,
        "element_dtype", element_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return empty_tensor_list_eager_fallback(
          element_shape, max_num_elements, element_dtype=element_dtype,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EmptyTensorList", element_shape=element_shape,
                           max_num_elements=max_num_elements,
                           element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EmptyTensorList", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EmptyTensorList = tf_export("raw_ops.EmptyTensorList")(_ops.to_raw_op(empty_tensor_list))


def empty_tensor_list_eager_fallback(element_shape: Annotated[Any, TV_EmptyTensorList_shape_type], max_num_elements: Annotated[Any, _atypes.Int32], element_dtype: TV_EmptyTensorList_element_dtype, name, ctx) -> Annotated[Any, _atypes.Variant]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  max_num_elements = _ops.convert_to_tensor(max_num_elements, _dtypes.int32)
  _inputs_flat = [element_shape, max_num_elements]
  _attrs = ("element_dtype", element_dtype, "shape_type", _attr_shape_type)
  _result = _execute.execute(b"EmptyTensorList", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EmptyTensorList", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TensorListConcatOutput = collections.namedtuple(
    "TensorListConcat",
    ["tensor", "lengths"])


TV_TensorListConcat_element_dtype = TypeVar("TV_TensorListConcat_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_concat(input_handle: Annotated[Any, _atypes.Variant], element_dtype: TV_TensorListConcat_element_dtype, element_shape=None, name=None):
  r"""Concats all tensors in the list along the 0th dimension.

  Requires that all tensors have the same shape except the first dimension.

  input_handle: The input list.
  tensor: The concated result.
  lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (tensor, lengths).

    tensor: A `Tensor` of type `element_dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListConcat", name, input_handle, "element_dtype",
        element_dtype, "element_shape", element_shape)
      _result = _TensorListConcatOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_concat_eager_fallback(
          input_handle, element_dtype=element_dtype,
          element_shape=element_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListConcat", input_handle=input_handle,
                            element_dtype=element_dtype,
                            element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "element_shape", _op.get_attr("element_shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListConcat", _inputs_flat, _attrs, _result)
  _result = _TensorListConcatOutput._make(_result)
  return _result

TensorListConcat = tf_export("raw_ops.TensorListConcat")(_ops.to_raw_op(tensor_list_concat))


def tensor_list_concat_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], element_dtype: TV_TensorListConcat_element_dtype, element_shape, name, ctx):
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  if element_shape is None:
    element_shape = None
  element_shape = _execute.make_shape(element_shape, "element_shape")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = ("element_dtype", element_dtype, "element_shape", element_shape)
  _result = _execute.execute(b"TensorListConcat", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListConcat", _inputs_flat, _attrs, _result)
  _result = _TensorListConcatOutput._make(_result)
  return _result


TV_TensorListConcatLists_element_dtype = TypeVar("TV_TensorListConcatLists_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_concat_lists(input_a: Annotated[Any, _atypes.Variant], input_b: Annotated[Any, _atypes.Variant], element_dtype: TV_TensorListConcatLists_element_dtype, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_a: A `Tensor` of type `variant`.
    input_b: A `Tensor` of type `variant`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListConcatLists", name, input_a, input_b,
        "element_dtype", element_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_concat_lists_eager_fallback(
          input_a, input_b, element_dtype=element_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListConcatLists", input_a=input_a, input_b=input_b,
                                 element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListConcatLists", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListConcatLists = tf_export("raw_ops.TensorListConcatLists")(_ops.to_raw_op(tensor_list_concat_lists))


def tensor_list_concat_lists_eager_fallback(input_a: Annotated[Any, _atypes.Variant], input_b: Annotated[Any, _atypes.Variant], element_dtype: TV_TensorListConcatLists_element_dtype, name, ctx) -> Annotated[Any, _atypes.Variant]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  input_a = _ops.convert_to_tensor(input_a, _dtypes.variant)
  input_b = _ops.convert_to_tensor(input_b, _dtypes.variant)
  _inputs_flat = [input_a, input_b]
  _attrs = ("element_dtype", element_dtype)
  _result = _execute.execute(b"TensorListConcatLists", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListConcatLists", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TensorListConcatV2Output = collections.namedtuple(
    "TensorListConcatV2",
    ["tensor", "lengths"])


TV_TensorListConcatV2_element_dtype = TypeVar("TV_TensorListConcatV2_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListConcatV2_shape_type = TypeVar("TV_TensorListConcatV2_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_concat_v2(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, TV_TensorListConcatV2_shape_type], leading_dims: Annotated[Any, _atypes.Int64], element_dtype: TV_TensorListConcatV2_element_dtype, name=None):
  r"""Concats all tensors in the list along the 0th dimension.

  Requires that all tensors have the same shape except the first dimension.

  input_handle: The input list.
  element_shape: The shape of the uninitialized elements in the list. If the first
    dimension is not -1, it is assumed that all list elements have the same
    leading dim.
  leading_dims: The list of leading dims of uninitialized list elements. Used if
    the leading dim of input_handle.element_shape or the element_shape input arg
    is not already set.
  tensor: The concated result.
  lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    leading_dims: A `Tensor` of type `int64`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (tensor, lengths).

    tensor: A `Tensor` of type `element_dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListConcatV2", name, input_handle, element_shape,
        leading_dims, "element_dtype", element_dtype)
      _result = _TensorListConcatV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_concat_v2_eager_fallback(
          input_handle, element_shape, leading_dims,
          element_dtype=element_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListConcatV2", input_handle=input_handle,
                              element_shape=element_shape,
                              leading_dims=leading_dims,
                              element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListConcatV2", _inputs_flat, _attrs, _result)
  _result = _TensorListConcatV2Output._make(_result)
  return _result

TensorListConcatV2 = tf_export("raw_ops.TensorListConcatV2")(_ops.to_raw_op(tensor_list_concat_v2))


def tensor_list_concat_v2_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, TV_TensorListConcatV2_shape_type], leading_dims: Annotated[Any, _atypes.Int64], element_dtype: TV_TensorListConcatV2_element_dtype, name, ctx):
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  leading_dims = _ops.convert_to_tensor(leading_dims, _dtypes.int64)
  _inputs_flat = [input_handle, element_shape, leading_dims]
  _attrs = ("element_dtype", element_dtype, "shape_type", _attr_shape_type)
  _result = _execute.execute(b"TensorListConcatV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListConcatV2", _inputs_flat, _attrs, _result)
  _result = _TensorListConcatV2Output._make(_result)
  return _result


TV_TensorListElementShape_shape_type = TypeVar("TV_TensorListElementShape_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_element_shape(input_handle: Annotated[Any, _atypes.Variant], shape_type: TV_TensorListElementShape_shape_type, name=None) -> Annotated[Any, TV_TensorListElementShape_shape_type]:
  r"""The shape of the elements of the given list, as a tensor.

    input_handle: the list
    element_shape: the shape of elements of the list

  Args:
    input_handle: A `Tensor` of type `variant`.
    shape_type: A `tf.DType` from: `tf.int32, tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `shape_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListElementShape", name, input_handle, "shape_type",
        shape_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_element_shape_eager_fallback(
          input_handle, shape_type=shape_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shape_type = _execute.make_type(shape_type, "shape_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListElementShape", input_handle=input_handle,
                                  shape_type=shape_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListElementShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListElementShape = tf_export("raw_ops.TensorListElementShape")(_ops.to_raw_op(tensor_list_element_shape))


def tensor_list_element_shape_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], shape_type: TV_TensorListElementShape_shape_type, name, ctx) -> Annotated[Any, TV_TensorListElementShape_shape_type]:
  shape_type = _execute.make_type(shape_type, "shape_type")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = ("shape_type", shape_type)
  _result = _execute.execute(b"TensorListElementShape", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListElementShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListFromTensor_element_dtype = TypeVar("TV_TensorListFromTensor_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListFromTensor_shape_type = TypeVar("TV_TensorListFromTensor_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_from_tensor(tensor: Annotated[Any, TV_TensorListFromTensor_element_dtype], element_shape: Annotated[Any, TV_TensorListFromTensor_shape_type], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a TensorList which, when stacked, has the value of `tensor`.

  Each tensor in the result list corresponds to one row of the input tensor.

  tensor: The input tensor.
  output_handle: The list.

  Args:
    tensor: A `Tensor`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListFromTensor", name, tensor, element_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_from_tensor_eager_fallback(
          tensor, element_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListFromTensor", tensor=tensor, element_shape=element_shape,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListFromTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListFromTensor = tf_export("raw_ops.TensorListFromTensor")(_ops.to_raw_op(tensor_list_from_tensor))


def tensor_list_from_tensor_eager_fallback(tensor: Annotated[Any, TV_TensorListFromTensor_element_dtype], element_shape: Annotated[Any, TV_TensorListFromTensor_shape_type], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [tensor, element_shape]
  _attrs = ("element_dtype", _attr_element_dtype, "shape_type",
  _attr_shape_type)
  _result = _execute.execute(b"TensorListFromTensor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListFromTensor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListGather_element_dtype = TypeVar("TV_TensorListGather_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_gather(input_handle: Annotated[Any, _atypes.Variant], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListGather_element_dtype, name=None) -> Annotated[Any, TV_TensorListGather_element_dtype]:
  r"""Creates a Tensor by indexing into the TensorList.

  Each row in the produced Tensor corresponds to the element in the TensorList
  specified by the given index (see `tf.gather`).

  input_handle: The input tensor list.
  indices: The indices used to index into the list.
  values: The tensor.

  Args:
    input_handle: A `Tensor` of type `variant`.
    indices: A `Tensor` of type `int32`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `element_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListGather", name, input_handle, indices, element_shape,
        "element_dtype", element_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_gather_eager_fallback(
          input_handle, indices, element_shape, element_dtype=element_dtype,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListGather", input_handle=input_handle, indices=indices,
                            element_shape=element_shape,
                            element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListGather = tf_export("raw_ops.TensorListGather")(_ops.to_raw_op(tensor_list_gather))


def tensor_list_gather_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListGather_element_dtype, name, ctx) -> Annotated[Any, TV_TensorListGather_element_dtype]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  element_shape = _ops.convert_to_tensor(element_shape, _dtypes.int32)
  _inputs_flat = [input_handle, indices, element_shape]
  _attrs = ("element_dtype", element_dtype)
  _result = _execute.execute(b"TensorListGather", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListGetItem_element_dtype = TypeVar("TV_TensorListGetItem_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_get_item(input_handle: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListGetItem_element_dtype, name=None) -> Annotated[Any, TV_TensorListGetItem_element_dtype]:
  r"""Returns the item in the list with the given index.

  input_handle: the list
  index: the position in the list from which an element will be retrieved
  item: the element at that position

  Args:
    input_handle: A `Tensor` of type `variant`.
    index: A `Tensor` of type `int32`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `element_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListGetItem", name, input_handle, index, element_shape,
        "element_dtype", element_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_get_item_eager_fallback(
          input_handle, index, element_shape, element_dtype=element_dtype,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListGetItem", input_handle=input_handle, index=index,
                             element_shape=element_shape,
                             element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListGetItem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListGetItem = tf_export("raw_ops.TensorListGetItem")(_ops.to_raw_op(tensor_list_get_item))


def tensor_list_get_item_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListGetItem_element_dtype, name, ctx) -> Annotated[Any, TV_TensorListGetItem_element_dtype]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  element_shape = _ops.convert_to_tensor(element_shape, _dtypes.int32)
  _inputs_flat = [input_handle, index, element_shape]
  _attrs = ("element_dtype", element_dtype)
  _result = _execute.execute(b"TensorListGetItem", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListGetItem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_list_length(input_handle: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Returns the number of tensors in the input tensor list.

  input_handle: the input list
  length: the number of tensors in the list

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
        _ctx, "TensorListLength", name, input_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_length_eager_fallback(
          input_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListLength", input_handle=input_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListLength", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListLength = tf_export("raw_ops.TensorListLength")(_ops.to_raw_op(tensor_list_length))


def tensor_list_length_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Int32]:
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = None
  _result = _execute.execute(b"TensorListLength", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListLength", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TensorListPopBackOutput = collections.namedtuple(
    "TensorListPopBack",
    ["output_handle", "tensor"])


TV_TensorListPopBack_element_dtype = TypeVar("TV_TensorListPopBack_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_pop_back(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListPopBack_element_dtype, name=None):
  r"""Returns the last element of the input list as well as a list with all but that element.

  Fails if the list is empty.

  input_handle: the input list
  tensor: the withdrawn last element of the list
  element_dtype: the type of elements in the list
  element_shape: the shape of the output tensor

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_handle, tensor).

    output_handle: A `Tensor` of type `variant`.
    tensor: A `Tensor` of type `element_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListPopBack", name, input_handle, element_shape,
        "element_dtype", element_dtype)
      _result = _TensorListPopBackOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_pop_back_eager_fallback(
          input_handle, element_shape, element_dtype=element_dtype, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListPopBack", input_handle=input_handle,
                             element_shape=element_shape,
                             element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListPopBack", _inputs_flat, _attrs, _result)
  _result = _TensorListPopBackOutput._make(_result)
  return _result

TensorListPopBack = tf_export("raw_ops.TensorListPopBack")(_ops.to_raw_op(tensor_list_pop_back))


def tensor_list_pop_back_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListPopBack_element_dtype, name, ctx):
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  element_shape = _ops.convert_to_tensor(element_shape, _dtypes.int32)
  _inputs_flat = [input_handle, element_shape]
  _attrs = ("element_dtype", element_dtype)
  _result = _execute.execute(b"TensorListPopBack", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListPopBack", _inputs_flat, _attrs, _result)
  _result = _TensorListPopBackOutput._make(_result)
  return _result


TV_TensorListPushBack_element_dtype = TypeVar("TV_TensorListPushBack_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_push_back(input_handle: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListPushBack_element_dtype], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.

  tensor: The tensor to put on the list.
  input_handle: The old list.
  output_handle: A list with the elements of the old list followed by tensor.
  element_dtype: the type of elements in the list.
  element_shape: a shape compatible with that of elements in the list.

  Args:
    input_handle: A `Tensor` of type `variant`.
    tensor: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListPushBack", name, input_handle, tensor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_push_back_eager_fallback(
          input_handle, tensor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListPushBack", input_handle=input_handle, tensor=tensor,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListPushBack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListPushBack = tf_export("raw_ops.TensorListPushBack")(_ops.to_raw_op(tensor_list_push_back))


def tensor_list_push_back_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListPushBack_element_dtype], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle, tensor]
  _attrs = ("element_dtype", _attr_element_dtype)
  _result = _execute.execute(b"TensorListPushBack", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListPushBack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListPushBackBatch_element_dtype = TypeVar("TV_TensorListPushBackBatch_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_push_back_batch(input_handles: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListPushBackBatch_element_dtype], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_handles: A `Tensor` of type `variant`.
    tensor: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListPushBackBatch", name, input_handles, tensor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_push_back_batch_eager_fallback(
          input_handles, tensor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListPushBackBatch", input_handles=input_handles, tensor=tensor,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListPushBackBatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListPushBackBatch = tf_export("raw_ops.TensorListPushBackBatch")(_ops.to_raw_op(tensor_list_push_back_batch))


def tensor_list_push_back_batch_eager_fallback(input_handles: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListPushBackBatch_element_dtype], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  input_handles = _ops.convert_to_tensor(input_handles, _dtypes.variant)
  _inputs_flat = [input_handles, tensor]
  _attrs = ("element_dtype", _attr_element_dtype)
  _result = _execute.execute(b"TensorListPushBackBatch", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListPushBackBatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListReserve_element_dtype = TypeVar("TV_TensorListReserve_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListReserve_shape_type = TypeVar("TV_TensorListReserve_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_reserve(element_shape: Annotated[Any, TV_TensorListReserve_shape_type], num_elements: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListReserve_element_dtype, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""List of the given size with empty elements.

  element_shape: the shape of the future elements of the list
  num_elements: the number of elements to reserve
  handle: the output list
  element_dtype: the desired type of elements in the list.

  Args:
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    num_elements: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListReserve", name, element_shape, num_elements,
        "element_dtype", element_dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_reserve_eager_fallback(
          element_shape, num_elements, element_dtype=element_dtype, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListReserve", element_shape=element_shape,
                             num_elements=num_elements,
                             element_dtype=element_dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListReserve", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListReserve = tf_export("raw_ops.TensorListReserve")(_ops.to_raw_op(tensor_list_reserve))


def tensor_list_reserve_eager_fallback(element_shape: Annotated[Any, TV_TensorListReserve_shape_type], num_elements: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListReserve_element_dtype, name, ctx) -> Annotated[Any, _atypes.Variant]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  num_elements = _ops.convert_to_tensor(num_elements, _dtypes.int32)
  _inputs_flat = [element_shape, num_elements]
  _attrs = ("element_dtype", element_dtype, "shape_type", _attr_shape_type)
  _result = _execute.execute(b"TensorListReserve", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListReserve", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_list_resize(input_handle: Annotated[Any, _atypes.Variant], size: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Resizes the list.

  
  input_handle: the input list
  size: size of the output list

  Args:
    input_handle: A `Tensor` of type `variant`.
    size: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListResize", name, input_handle, size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_resize_eager_fallback(
          input_handle, size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListResize", input_handle=input_handle, size=size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListResize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListResize = tf_export("raw_ops.TensorListResize")(_ops.to_raw_op(tensor_list_resize))


def tensor_list_resize_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], size: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Variant]:
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [input_handle, size]
  _attrs = None
  _result = _execute.execute(b"TensorListResize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListResize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListScatter_element_dtype = TypeVar("TV_TensorListScatter_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListScatter_shape_type = TypeVar("TV_TensorListScatter_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_scatter(tensor: Annotated[Any, TV_TensorListScatter_element_dtype], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, TV_TensorListScatter_shape_type], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a TensorList by indexing into a Tensor.

  Each member of the TensorList corresponds to one row of the input tensor,
  specified by the given index (see `tf.gather`).

  tensor: The input tensor.
  indices: The indices used to index into the list.
  element_shape: The shape of the elements in the list (can be less specified than
    the shape of the tensor).
  output_handle: The TensorList.

  Args:
    tensor: A `Tensor`.
    indices: A `Tensor` of type `int32`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListScatter", name, tensor, indices, element_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_scatter_eager_fallback(
          tensor, indices, element_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListScatter", tensor=tensor, indices=indices,
                             element_shape=element_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListScatter = tf_export("raw_ops.TensorListScatter")(_ops.to_raw_op(tensor_list_scatter))


def tensor_list_scatter_eager_fallback(tensor: Annotated[Any, TV_TensorListScatter_element_dtype], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, TV_TensorListScatter_shape_type], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [tensor, indices, element_shape]
  _attrs = ("element_dtype", _attr_element_dtype, "shape_type",
  _attr_shape_type)
  _result = _execute.execute(b"TensorListScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListScatterIntoExistingList_element_dtype = TypeVar("TV_TensorListScatterIntoExistingList_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_scatter_into_existing_list(input_handle: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListScatterIntoExistingList_element_dtype], indices: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Scatters tensor at indices in an input list.

  Each member of the TensorList corresponds to one row of the input tensor,
  specified by the given index (see `tf.gather`).

  input_handle: The list to scatter into.
  tensor: The input tensor.
  indices: The indices used to index into the list.
  output_handle: The TensorList.

  Args:
    input_handle: A `Tensor` of type `variant`.
    tensor: A `Tensor`.
    indices: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListScatterIntoExistingList", name, input_handle, tensor,
        indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_scatter_into_existing_list_eager_fallback(
          input_handle, tensor, indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListScatterIntoExistingList", input_handle=input_handle,
                                             tensor=tensor, indices=indices,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListScatterIntoExistingList", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListScatterIntoExistingList = tf_export("raw_ops.TensorListScatterIntoExistingList")(_ops.to_raw_op(tensor_list_scatter_into_existing_list))


def tensor_list_scatter_into_existing_list_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], tensor: Annotated[Any, TV_TensorListScatterIntoExistingList_element_dtype], indices: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  _inputs_flat = [input_handle, tensor, indices]
  _attrs = ("element_dtype", _attr_element_dtype)
  _result = _execute.execute(b"TensorListScatterIntoExistingList", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListScatterIntoExistingList", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListScatterV2_element_dtype = TypeVar("TV_TensorListScatterV2_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListScatterV2_shape_type = TypeVar("TV_TensorListScatterV2_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_scatter_v2(tensor: Annotated[Any, TV_TensorListScatterV2_element_dtype], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, TV_TensorListScatterV2_shape_type], num_elements: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a TensorList by indexing into a Tensor.

  Each member of the TensorList corresponds to one row of the input tensor,
  specified by the given index (see `tf.gather`).

  tensor: The input tensor.
  indices: The indices used to index into the list.
  element_shape: The shape of the elements in the list (can be less specified than
    the shape of the tensor).
  num_elements: The size of the output list. Must be large enough to accommodate
    the largest index in indices. If -1, the list is just large enough to include
    the largest index in indices.
  output_handle: The TensorList.

  Args:
    tensor: A `Tensor`.
    indices: A `Tensor` of type `int32`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    num_elements: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListScatterV2", name, tensor, indices, element_shape,
        num_elements)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_scatter_v2_eager_fallback(
          tensor, indices, element_shape, num_elements, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListScatterV2", tensor=tensor, indices=indices,
                               element_shape=element_shape,
                               num_elements=num_elements, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListScatterV2 = tf_export("raw_ops.TensorListScatterV2")(_ops.to_raw_op(tensor_list_scatter_v2))


def tensor_list_scatter_v2_eager_fallback(tensor: Annotated[Any, TV_TensorListScatterV2_element_dtype], indices: Annotated[Any, _atypes.Int32], element_shape: Annotated[Any, TV_TensorListScatterV2_shape_type], num_elements: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int32)
  num_elements = _ops.convert_to_tensor(num_elements, _dtypes.int32)
  _inputs_flat = [tensor, indices, element_shape, num_elements]
  _attrs = ("element_dtype", _attr_element_dtype, "shape_type",
  _attr_shape_type)
  _result = _execute.execute(b"TensorListScatterV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListScatterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListSetItem_element_dtype = TypeVar("TV_TensorListSetItem_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_set_item(input_handle: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], item: Annotated[Any, TV_TensorListSetItem_element_dtype], resize_if_index_out_of_bounds:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Sets the index-th position of the list to contain the given tensor.

  input_handle: the list
  index: the position in the list to which the tensor will be assigned
  item: the element to be assigned to that position
  output_handle: the new list, with the element in the proper position

  Args:
    input_handle: A `Tensor` of type `variant`.
    index: A `Tensor` of type `int32`.
    item: A `Tensor`.
    resize_if_index_out_of_bounds: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListSetItem", name, input_handle, index, item,
        "resize_if_index_out_of_bounds", resize_if_index_out_of_bounds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_set_item_eager_fallback(
          input_handle, index, item,
          resize_if_index_out_of_bounds=resize_if_index_out_of_bounds,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if resize_if_index_out_of_bounds is None:
    resize_if_index_out_of_bounds = False
  resize_if_index_out_of_bounds = _execute.make_bool(resize_if_index_out_of_bounds, "resize_if_index_out_of_bounds")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListSetItem", input_handle=input_handle, index=index,
                             item=item,
                             resize_if_index_out_of_bounds=resize_if_index_out_of_bounds,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "resize_if_index_out_of_bounds",
              _op._get_attr_bool("resize_if_index_out_of_bounds"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListSetItem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListSetItem = tf_export("raw_ops.TensorListSetItem")(_ops.to_raw_op(tensor_list_set_item))


def tensor_list_set_item_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], index: Annotated[Any, _atypes.Int32], item: Annotated[Any, TV_TensorListSetItem_element_dtype], resize_if_index_out_of_bounds: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if resize_if_index_out_of_bounds is None:
    resize_if_index_out_of_bounds = False
  resize_if_index_out_of_bounds = _execute.make_bool(resize_if_index_out_of_bounds, "resize_if_index_out_of_bounds")
  _attr_element_dtype, (item,) = _execute.args_to_matching_eager([item], ctx, [])
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  index = _ops.convert_to_tensor(index, _dtypes.int32)
  _inputs_flat = [input_handle, index, item]
  _attrs = ("element_dtype", _attr_element_dtype,
  "resize_if_index_out_of_bounds", resize_if_index_out_of_bounds)
  _result = _execute.execute(b"TensorListSetItem", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListSetItem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListSplit_element_dtype = TypeVar("TV_TensorListSplit_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_TensorListSplit_shape_type = TypeVar("TV_TensorListSplit_shape_type", _atypes.Int32, _atypes.Int64)

def tensor_list_split(tensor: Annotated[Any, TV_TensorListSplit_element_dtype], element_shape: Annotated[Any, TV_TensorListSplit_shape_type], lengths: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Splits a tensor into a list.

  list[i] corresponds to lengths[i] tensors from the input tensor.
  The tensor must have rank at least 1 and contain exactly sum(lengths) elements.

  tensor: The input tensor.
  element_shape: A shape compatible with that of elements in the tensor.
  lengths: Vector of sizes of the 0th dimension of tensors in the list.
  output_handle: The list.

  Args:
    tensor: A `Tensor`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    lengths: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListSplit", name, tensor, element_shape, lengths)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_split_eager_fallback(
          tensor, element_shape, lengths, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListSplit", tensor=tensor, element_shape=element_shape,
                           lengths=lengths, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "shape_type", _op._get_attr_type("shape_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListSplit", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListSplit = tf_export("raw_ops.TensorListSplit")(_ops.to_raw_op(tensor_list_split))


def tensor_list_split_eager_fallback(tensor: Annotated[Any, TV_TensorListSplit_element_dtype], element_shape: Annotated[Any, TV_TensorListSplit_shape_type], lengths: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64, ])
  lengths = _ops.convert_to_tensor(lengths, _dtypes.int64)
  _inputs_flat = [tensor, element_shape, lengths]
  _attrs = ("element_dtype", _attr_element_dtype, "shape_type",
  _attr_shape_type)
  _result = _execute.execute(b"TensorListSplit", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListSplit", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorListStack_element_dtype = TypeVar("TV_TensorListStack_element_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_list_stack(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListStack_element_dtype, num_elements:int=-1, name=None) -> Annotated[Any, TV_TensorListStack_element_dtype]:
  r"""Stacks all tensors in the list.

  Requires that all tensors have the same shape.

  input_handle: the input list
  tensor: the gathered result
  num_elements: optional. If not -1, the number of elements in the list.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    num_elements: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `element_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorListStack", name, input_handle, element_shape,
        "element_dtype", element_dtype, "num_elements", num_elements)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_list_stack_eager_fallback(
          input_handle, element_shape, element_dtype=element_dtype,
          num_elements=num_elements, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  if num_elements is None:
    num_elements = -1
  num_elements = _execute.make_int(num_elements, "num_elements")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorListStack", input_handle=input_handle,
                           element_shape=element_shape,
                           element_dtype=element_dtype,
                           num_elements=num_elements, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("element_dtype", _op._get_attr_type("element_dtype"),
              "num_elements", _op._get_attr_int("num_elements"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorListStack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorListStack = tf_export("raw_ops.TensorListStack")(_ops.to_raw_op(tensor_list_stack))


def tensor_list_stack_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], element_shape: Annotated[Any, _atypes.Int32], element_dtype: TV_TensorListStack_element_dtype, num_elements: int, name, ctx) -> Annotated[Any, TV_TensorListStack_element_dtype]:
  element_dtype = _execute.make_type(element_dtype, "element_dtype")
  if num_elements is None:
    num_elements = -1
  num_elements = _execute.make_int(num_elements, "num_elements")
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  element_shape = _ops.convert_to_tensor(element_shape, _dtypes.int32)
  _inputs_flat = [input_handle, element_shape]
  _attrs = ("element_dtype", element_dtype, "num_elements", num_elements)
  _result = _execute.execute(b"TensorListStack", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorListStack", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

