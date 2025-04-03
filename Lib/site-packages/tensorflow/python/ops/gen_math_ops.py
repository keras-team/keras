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

TV_Abs_T = TypeVar("TV_Abs_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

def _abs(x: Annotated[Any, TV_Abs_T], name=None) -> Annotated[Any, TV_Abs_T]:
  r"""Computes the absolute value of a tensor.

  Given a tensor `x`, this operation returns a tensor containing the absolute
  value of each element in `x`. For example, if x is an input element and y is
  an output element, this operation computes \\(y = |x|\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Abs", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _abs_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Abs", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Abs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Abs = tf_export("raw_ops.Abs")(_ops.to_raw_op(_abs))


def _abs_eager_fallback(x: Annotated[Any, TV_Abs_T], name, ctx) -> Annotated[Any, TV_Abs_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Abs", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Abs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AccumulateNV2_T = TypeVar("TV_AccumulateNV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def accumulate_nv2(inputs: Annotated[List[Any], TV_AccumulateNV2_T], shape, name=None) -> Annotated[Any, TV_AccumulateNV2_T]:
  r"""Returns the element-wise sum of a list of tensors.

  `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
  wait for all of its inputs to be ready before beginning to sum. This can
  save memory if inputs are ready at different times, since minimum temporary
  storage is proportional to the output size rather than the inputs size.

  Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.

  Returns a `Tensor` of same shape and type as the elements of `inputs`.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A list of `Tensor` objects, each with same shape and type.
    shape: A `tf.TensorShape` or list of `ints`.
      Shape of elements of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AccumulateNV2", name, inputs, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return accumulate_nv2_eager_fallback(
          inputs, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'accumulate_nv2' Op, not %r." % inputs)
  _attr_N = len(inputs)
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AccumulateNV2", inputs=inputs, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AccumulateNV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AccumulateNV2 = tf_export("raw_ops.AccumulateNV2")(_ops.to_raw_op(accumulate_nv2))


def accumulate_nv2_eager_fallback(inputs: Annotated[List[Any], TV_AccumulateNV2_T], shape, name, ctx) -> Annotated[Any, TV_AccumulateNV2_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'accumulate_nv2' Op, not %r." % inputs)
  _attr_N = len(inputs)
  shape = _execute.make_shape(shape, "shape")
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T, "shape", shape)
  _result = _execute.execute(b"AccumulateNV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AccumulateNV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Acos_T = TypeVar("TV_Acos_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def acos(x: Annotated[Any, TV_Acos_T], name=None) -> Annotated[Any, TV_Acos_T]:
  r"""Computes acos of x element-wise.

  
    Provided an input tensor, the `tf.math.acos` operation returns the inverse cosine of each element of the tensor. If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.

    Input range is `[-1, 1]` and the output has a range of `[0, pi]`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Acos", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return acos_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Acos", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Acos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Acos = tf_export("raw_ops.Acos")(_ops.to_raw_op(acos))


def acos_eager_fallback(x: Annotated[Any, TV_Acos_T], name, ctx) -> Annotated[Any, TV_Acos_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Acos", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Acos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Acosh_T = TypeVar("TV_Acosh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.acosh', 'acosh')
def acosh(x: Annotated[Any, TV_Acosh_T], name=None) -> Annotated[Any, TV_Acosh_T]:
  r"""Computes inverse hyperbolic cosine of x element-wise.

  Given an input tensor, the function computes inverse hyperbolic cosine of every element.
  Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.

  ```python
  x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
  tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Acosh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_acosh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return acosh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            acosh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_acosh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Acosh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          acosh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Acosh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Acosh = tf_export("raw_ops.Acosh")(_ops.to_raw_op(acosh))
_dispatcher_for_acosh = acosh._tf_type_based_dispatcher.Dispatch


def acosh_eager_fallback(x: Annotated[Any, TV_Acosh_T], name, ctx) -> Annotated[Any, TV_Acosh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Acosh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Acosh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Add_T = TypeVar("TV_Add_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt8)

def add(x: Annotated[Any, TV_Add_T], y: Annotated[Any, TV_Add_T], name=None) -> Annotated[Any, TV_Add_T]:
  r"""Returns x + y element-wise.

  *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Given two input tensors, the `tf.add` operation computes the sum for every element in the tensor.

  Both input and output have a range `(-inf, inf)`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Add", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return add_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Add", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Add", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Add = tf_export("raw_ops.Add")(_ops.to_raw_op(add))


def add_eager_fallback(x: Annotated[Any, TV_Add_T], y: Annotated[Any, TV_Add_T], name, ctx) -> Annotated[Any, TV_Add_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, _dtypes.string, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Add", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Add", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AddN_T = TypeVar("TV_AddN_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def add_n(inputs: Annotated[List[Any], TV_AddN_T], name=None) -> Annotated[Any, TV_AddN_T]:
  r"""Add all input tensors element wise.

    Inputs must be of same size and shape.

    ```python
    x = [9, 7, 10]
    tf.math.add_n(x) ==> 26
    ```

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AddN", name, inputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return add_n_eager_fallback(
          inputs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'add_n' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AddN", inputs=inputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AddN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AddN = tf_export("raw_ops.AddN")(_ops.to_raw_op(add_n))


def add_n_eager_fallback(inputs: Annotated[List[Any], TV_AddN_T], name, ctx) -> Annotated[Any, TV_AddN_T]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'add_n' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.variant, ])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"AddN", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AddN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AddV2_T = TypeVar("TV_AddV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def add_v2(x: Annotated[Any, TV_AddV2_T], y: Annotated[Any, TV_AddV2_T], name=None) -> Annotated[Any, TV_AddV2_T]:
  r"""Returns x + y element-wise.

  *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AddV2", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return add_v2_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AddV2", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AddV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AddV2 = tf_export("raw_ops.AddV2")(_ops.to_raw_op(add_v2))


def add_v2_eager_fallback(x: Annotated[Any, TV_AddV2_T], y: Annotated[Any, TV_AddV2_T], name, ctx) -> Annotated[Any, TV_AddV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AddV2", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AddV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_All_Tidx = TypeVar("TV_All_Tidx", _atypes.Int32, _atypes.Int64)

def _all(input: Annotated[Any, _atypes.Bool], axis: Annotated[Any, TV_All_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "All", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _all_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "All", input=input, reduction_indices=axis, keep_dims=keep_dims,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "Tidx",
              _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "All", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

All = tf_export("raw_ops.All")(_ops.to_raw_op(_all))


def _all_eager_fallback(input: Annotated[Any, _atypes.Bool], axis: Annotated[Any, TV_All_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  input = _ops.convert_to_tensor(input, _dtypes.bool)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"All", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "All", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Angle_T = TypeVar("TV_Angle_T", _atypes.Complex128, _atypes.Complex64)
TV_Angle_Tout = TypeVar("TV_Angle_Tout", _atypes.Float32, _atypes.Float64)

def angle(input: Annotated[Any, TV_Angle_T], Tout:TV_Angle_Tout=_dtypes.float32, name=None) -> Annotated[Any, TV_Angle_Tout]:
  r"""Returns the argument of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the argument of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part.

  The argument returned by this operation is of the form \\(atan2(b, a)\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.math.angle(input) ==> [2.0132, 1.056]
  ```

  @compatibility(numpy)
  Equivalent to np.angle.
  @end_compatibility

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Angle", name, input, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return angle_eager_fallback(
          input, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Angle", input=input, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Angle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Angle = tf_export("raw_ops.Angle")(_ops.to_raw_op(angle))


def angle_eager_fallback(input: Annotated[Any, TV_Angle_T], Tout: TV_Angle_Tout, name, ctx) -> Annotated[Any, TV_Angle_Tout]:
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "Tout", Tout)
  _result = _execute.execute(b"Angle", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Angle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Any_Tidx = TypeVar("TV_Any_Tidx", _atypes.Int32, _atypes.Int64)

def _any(input: Annotated[Any, _atypes.Bool], axis: Annotated[Any, TV_Any_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Any", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _any_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Any", input=input, reduction_indices=axis, keep_dims=keep_dims,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "Tidx",
              _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Any", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Any = tf_export("raw_ops.Any")(_ops.to_raw_op(_any))


def _any_eager_fallback(input: Annotated[Any, _atypes.Bool], axis: Annotated[Any, TV_Any_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  input = _ops.convert_to_tensor(input, _dtypes.bool)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Any", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Any", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ApproximateEqual_T = TypeVar("TV_ApproximateEqual_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def approximate_equal(x: Annotated[Any, TV_ApproximateEqual_T], y: Annotated[Any, TV_ApproximateEqual_T], tolerance:float=1e-05, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of abs(x-y) < tolerance element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    tolerance: An optional `float`. Defaults to `1e-05`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ApproximateEqual", name, x, y, "tolerance", tolerance)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return approximate_equal_eager_fallback(
          x, y, tolerance=tolerance, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if tolerance is None:
    tolerance = 1e-05
  tolerance = _execute.make_float(tolerance, "tolerance")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApproximateEqual", x=x, y=y, tolerance=tolerance, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "tolerance",
              _op.get_attr("tolerance"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApproximateEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApproximateEqual = tf_export("raw_ops.ApproximateEqual")(_ops.to_raw_op(approximate_equal))


def approximate_equal_eager_fallback(x: Annotated[Any, TV_ApproximateEqual_T], y: Annotated[Any, TV_ApproximateEqual_T], tolerance: float, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if tolerance is None:
    tolerance = 1e-05
  tolerance = _execute.make_float(tolerance, "tolerance")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "tolerance", tolerance)
  _result = _execute.execute(b"ApproximateEqual", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ApproximateEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ArgMax_T = TypeVar("TV_ArgMax_T", _atypes.BFloat16, _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ArgMax_Tidx = TypeVar("TV_ArgMax_Tidx", _atypes.Int16, _atypes.Int32, _atypes.Int64)
TV_ArgMax_output_type = TypeVar("TV_ArgMax_output_type", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.UInt16)

def arg_max(input: Annotated[Any, TV_ArgMax_T], dimension: Annotated[Any, TV_ArgMax_Tidx], output_type:TV_ArgMax_output_type=_dtypes.int64, name=None) -> Annotated[Any, TV_ArgMax_output_type]:
  r"""Returns the index with the largest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmax(input = a)
    c = tf.keras.backend.eval(b)
    # c = 4
    # here a[4] = 166.32 which is the largest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`, `bool`.
    dimension: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      int16, int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int16, tf.uint16, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ArgMax", name, input, dimension, "output_type", output_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return arg_max_eager_fallback(
          input, dimension, output_type=output_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_type is None:
    output_type = _dtypes.int64
  output_type = _execute.make_type(output_type, "output_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ArgMax", input=input, dimension=dimension, output_type=output_type,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "output_type",
              _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ArgMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ArgMax = tf_export("raw_ops.ArgMax")(_ops.to_raw_op(arg_max))


def arg_max_eager_fallback(input: Annotated[Any, TV_ArgMax_T], dimension: Annotated[Any, TV_ArgMax_Tidx], output_type: TV_ArgMax_output_type, name, ctx) -> Annotated[Any, TV_ArgMax_output_type]:
  if output_type is None:
    output_type = _dtypes.int64
  output_type = _execute.make_type(output_type, "output_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, _dtypes.bool, ])
  _attr_Tidx, (dimension,) = _execute.args_to_matching_eager([dimension], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, dimension]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "output_type", output_type)
  _result = _execute.execute(b"ArgMax", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ArgMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ArgMin_T = TypeVar("TV_ArgMin_T", _atypes.BFloat16, _atypes.Bool, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ArgMin_Tidx = TypeVar("TV_ArgMin_Tidx", _atypes.Int32, _atypes.Int64)
TV_ArgMin_output_type = TypeVar("TV_ArgMin_output_type", _atypes.Int32, _atypes.Int64)

def arg_min(input: Annotated[Any, TV_ArgMin_T], dimension: Annotated[Any, TV_ArgMin_Tidx], output_type:TV_ArgMin_output_type=_dtypes.int64, name=None) -> Annotated[Any, TV_ArgMin_output_type]:
  r"""Returns the index with the smallest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmin(input = a)
    c = tf.keras.backend.eval(b)
    # c = 0
    # here a[0] = 1 which is the smallest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`, `bool`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ArgMin", name, input, dimension, "output_type", output_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return arg_min_eager_fallback(
          input, dimension, output_type=output_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_type is None:
    output_type = _dtypes.int64
  output_type = _execute.make_type(output_type, "output_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ArgMin", input=input, dimension=dimension, output_type=output_type,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "output_type",
              _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ArgMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ArgMin = tf_export("raw_ops.ArgMin")(_ops.to_raw_op(arg_min))


def arg_min_eager_fallback(input: Annotated[Any, TV_ArgMin_T], dimension: Annotated[Any, TV_ArgMin_Tidx], output_type: TV_ArgMin_output_type, name, ctx) -> Annotated[Any, TV_ArgMin_output_type]:
  if output_type is None:
    output_type = _dtypes.int64
  output_type = _execute.make_type(output_type, "output_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, _dtypes.bool, ])
  _attr_Tidx, (dimension,) = _execute.args_to_matching_eager([dimension], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, dimension]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "output_type", output_type)
  _result = _execute.execute(b"ArgMin", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ArgMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Asin_T = TypeVar("TV_Asin_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.asin', 'asin')
def asin(x: Annotated[Any, TV_Asin_T], name=None) -> Annotated[Any, TV_Asin_T]:
  r"""Computes the trignometric inverse sine of x element-wise.

  The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
  if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.

  **Note**: The output of `tf.math.asin` will lie within the invertible range
  of sine, i.e [-pi/2, pi/2].

  For example:

  ```python
  # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
  x = tf.constant([1.047, 0.785])
  y = tf.math.sin(x) # [0.8659266, 0.7068252]

  tf.math.asin(y) # [1.047, 0.785] = x
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Asin", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_asin(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return asin_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            asin, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_asin(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Asin", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          asin, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Asin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Asin = tf_export("raw_ops.Asin")(_ops.to_raw_op(asin))
_dispatcher_for_asin = asin._tf_type_based_dispatcher.Dispatch


def asin_eager_fallback(x: Annotated[Any, TV_Asin_T], name, ctx) -> Annotated[Any, TV_Asin_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Asin", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Asin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Asinh_T = TypeVar("TV_Asinh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.asinh', 'asinh')
def asinh(x: Annotated[Any, TV_Asinh_T], name=None) -> Annotated[Any, TV_Asinh_T]:
  r"""Computes inverse hyperbolic sine of x element-wise.

    Given an input tensor, this function computes inverse hyperbolic sine
    for every element in the tensor. Both input and output has a range of
    `[-inf, inf]`.

    ```python
    x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
    tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Asinh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_asinh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return asinh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            asinh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_asinh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Asinh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          asinh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Asinh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Asinh = tf_export("raw_ops.Asinh")(_ops.to_raw_op(asinh))
_dispatcher_for_asinh = asinh._tf_type_based_dispatcher.Dispatch


def asinh_eager_fallback(x: Annotated[Any, TV_Asinh_T], name, ctx) -> Annotated[Any, TV_Asinh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Asinh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Asinh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Atan_T = TypeVar("TV_Atan_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.atan', 'atan')
def atan(x: Annotated[Any, TV_Atan_T], name=None) -> Annotated[Any, TV_Atan_T]:
  r"""Computes the trignometric inverse tangent of x element-wise.

  The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
  if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.

  **Note**: The output of `tf.math.atan` will lie within the invertible range
  of tan, i.e (-pi/2, pi/2).

  For example:

  ```python
  # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
  x = tf.constant([1.047, 0.785])
  y = tf.math.tan(x) # [1.731261, 0.99920404]

  tf.math.atan(y) # [1.047, 0.785] = x
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Atan", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_atan(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return atan_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            atan, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_atan(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Atan", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          atan, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Atan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Atan = tf_export("raw_ops.Atan")(_ops.to_raw_op(atan))
_dispatcher_for_atan = atan._tf_type_based_dispatcher.Dispatch


def atan_eager_fallback(x: Annotated[Any, TV_Atan_T], name, ctx) -> Annotated[Any, TV_Atan_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Atan", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Atan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Atan2_T = TypeVar("TV_Atan2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.atan2', 'atan2')
def atan2(y: Annotated[Any, TV_Atan2_T], x: Annotated[Any, TV_Atan2_T], name=None) -> Annotated[Any, TV_Atan2_T]:
  r"""Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

  This is the angle \\( \theta \in [-\pi, \pi] \\) such that
  \\[ x = r \cos(\theta) \\]
  and
  \\[ y = r \sin(\theta) \\]
  where \\(r = \sqrt{x^2 + y^2} \\).

  For example:

  >>> x = [1., 1.]
  >>> y = [1., -1.]
  >>> print((tf.math.atan2(y,x) * (180 / np.pi)).numpy())
  [ 45. -45.]

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Atan2", name, y, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_atan2(
          (y, x, name,), None)
      if _result is not NotImplemented:
        return _result
      return atan2_eager_fallback(
          y, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            atan2, (), dict(y=y, x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_atan2(
        (y, x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Atan2", y=y, x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          atan2, (), dict(y=y, x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Atan2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Atan2 = tf_export("raw_ops.Atan2")(_ops.to_raw_op(atan2))
_dispatcher_for_atan2 = atan2._tf_type_based_dispatcher.Dispatch


def atan2_eager_fallback(y: Annotated[Any, TV_Atan2_T], x: Annotated[Any, TV_Atan2_T], name, ctx) -> Annotated[Any, TV_Atan2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (y, x) = _inputs_T
  _inputs_flat = [y, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Atan2", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Atan2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Atanh_T = TypeVar("TV_Atanh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.atanh', 'atanh')
def atanh(x: Annotated[Any, TV_Atanh_T], name=None) -> Annotated[Any, TV_Atanh_T]:
  r"""Computes inverse hyperbolic tangent of x element-wise.

    Given an input tensor, this function computes inverse hyperbolic tangent
    for every element in the tensor. Input range is `[-1,1]` and output range is
    `[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
    input is `1`, output will be `inf`. Values outside the range will have
    `nan` as output.

    ```python
    x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
    tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Atanh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_atanh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return atanh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            atanh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_atanh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Atanh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          atanh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Atanh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Atanh = tf_export("raw_ops.Atanh")(_ops.to_raw_op(atanh))
_dispatcher_for_atanh = atanh._tf_type_based_dispatcher.Dispatch


def atanh_eager_fallback(x: Annotated[Any, TV_Atanh_T], name, ctx) -> Annotated[Any, TV_Atanh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Atanh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Atanh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatMul_T = TypeVar("TV_BatchMatMul_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

def batch_mat_mul(x: Annotated[Any, TV_BatchMatMul_T], y: Annotated[Any, TV_BatchMatMul_T], adj_x:bool=False, adj_y:bool=False, grad_x:bool=False, grad_y:bool=False, name=None) -> Annotated[Any, TV_BatchMatMul_T]:
  r"""Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    grad_x: An optional `bool`. Defaults to `False`.
    grad_y: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatMul", name, x, y, "adj_x", adj_x, "adj_y", adj_y,
        "grad_x", grad_x, "grad_y", grad_y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_mat_mul_eager_fallback(
          x, y, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x, grad_y=grad_y,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatMul", x=x, y=y, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x,
                       grad_y=grad_y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "adj_x",
              _op._get_attr_bool("adj_x"), "adj_y",
              _op._get_attr_bool("adj_y"), "grad_x",
              _op._get_attr_bool("grad_x"), "grad_y",
              _op._get_attr_bool("grad_y"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatMul = tf_export("raw_ops.BatchMatMul")(_ops.to_raw_op(batch_mat_mul))


def batch_mat_mul_eager_fallback(x: Annotated[Any, TV_BatchMatMul_T], y: Annotated[Any, TV_BatchMatMul_T], adj_x: bool, adj_y: bool, grad_x: bool, grad_y: bool, name, ctx) -> Annotated[Any, TV_BatchMatMul_T]:
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "adj_x", adj_x, "adj_y", adj_y, "grad_x", grad_x,
  "grad_y", grad_y)
  _result = _execute.execute(b"BatchMatMul", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatMulV2_T = TypeVar("TV_BatchMatMulV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def batch_mat_mul_v2(x: Annotated[Any, TV_BatchMatMulV2_T], y: Annotated[Any, TV_BatchMatMulV2_T], adj_x:bool=False, adj_y:bool=False, grad_x:bool=False, grad_y:bool=False, name=None) -> Annotated[Any, TV_BatchMatMulV2_T]:
  r"""Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
  about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    grad_x: An optional `bool`. Defaults to `False`.
    grad_y: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatMulV2", name, x, y, "adj_x", adj_x, "adj_y", adj_y,
        "grad_x", grad_x, "grad_y", grad_y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_mat_mul_v2_eager_fallback(
          x, y, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x, grad_y=grad_y,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatMulV2", x=x, y=y, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x,
                         grad_y=grad_y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "adj_x",
              _op._get_attr_bool("adj_x"), "adj_y",
              _op._get_attr_bool("adj_y"), "grad_x",
              _op._get_attr_bool("grad_x"), "grad_y",
              _op._get_attr_bool("grad_y"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatMulV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatMulV2 = tf_export("raw_ops.BatchMatMulV2")(_ops.to_raw_op(batch_mat_mul_v2))


def batch_mat_mul_v2_eager_fallback(x: Annotated[Any, TV_BatchMatMulV2_T], y: Annotated[Any, TV_BatchMatMulV2_T], adj_x: bool, adj_y: bool, grad_x: bool, grad_y: bool, name, ctx) -> Annotated[Any, TV_BatchMatMulV2_T]:
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "adj_x", adj_x, "adj_y", adj_y, "grad_x", grad_x,
  "grad_y", grad_y)
  _result = _execute.execute(b"BatchMatMulV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatMulV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchMatMulV3_Ta = TypeVar("TV_BatchMatMulV3_Ta", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)
TV_BatchMatMulV3_Tb = TypeVar("TV_BatchMatMulV3_Tb", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)
TV_BatchMatMulV3_Tout = TypeVar("TV_BatchMatMulV3_Tout", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64)

def batch_mat_mul_v3(x: Annotated[Any, TV_BatchMatMulV3_Ta], y: Annotated[Any, TV_BatchMatMulV3_Tb], Tout: TV_BatchMatMulV3_Tout, adj_x:bool=False, adj_y:bool=False, grad_x:bool=False, grad_y:bool=False, name=None) -> Annotated[Any, TV_BatchMatMulV3_Tout]:
  r"""Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  *NOTE*: `BatchMatMulV3` supports broadcasting in the batch dimensions. More
  about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_y, c_y]`.
    Tout: A `tf.DType` from: `tf.bfloat16, tf.half, tf.float32, tf.float64, tf.int16, tf.int32, tf.int64, tf.complex64, tf.complex128`.
      If not spcified, Tout is the same type to input type.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    grad_x: An optional `bool`. Defaults to `False`.
    grad_y: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchMatMulV3", name, x, y, "Tout", Tout, "adj_x", adj_x,
        "adj_y", adj_y, "grad_x", grad_x, "grad_y", grad_y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_mat_mul_v3_eager_fallback(
          x, y, Tout=Tout, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x,
          grad_y=grad_y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchMatMulV3", x=x, y=y, Tout=Tout, adj_x=adj_x, adj_y=adj_y,
                         grad_x=grad_x, grad_y=grad_y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Ta", _op._get_attr_type("Ta"), "Tb", _op._get_attr_type("Tb"),
              "Tout", _op._get_attr_type("Tout"), "adj_x",
              _op._get_attr_bool("adj_x"), "adj_y",
              _op._get_attr_bool("adj_y"), "grad_x",
              _op._get_attr_bool("grad_x"), "grad_y",
              _op._get_attr_bool("grad_y"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchMatMulV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchMatMulV3 = tf_export("raw_ops.BatchMatMulV3")(_ops.to_raw_op(batch_mat_mul_v3))


def batch_mat_mul_v3_eager_fallback(x: Annotated[Any, TV_BatchMatMulV3_Ta], y: Annotated[Any, TV_BatchMatMulV3_Tb], Tout: TV_BatchMatMulV3_Tout, adj_x: bool, adj_y: bool, grad_x: bool, grad_y: bool, name, ctx) -> Annotated[Any, TV_BatchMatMulV3_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  if adj_x is None:
    adj_x = False
  adj_x = _execute.make_bool(adj_x, "adj_x")
  if adj_y is None:
    adj_y = False
  adj_y = _execute.make_bool(adj_y, "adj_y")
  if grad_x is None:
    grad_x = False
  grad_x = _execute.make_bool(grad_x, "grad_x")
  if grad_y is None:
    grad_y = False
  grad_y = _execute.make_bool(grad_y, "grad_y")
  _attr_Ta, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _attr_Tb, (y,) = _execute.args_to_matching_eager([y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x, y]
  _attrs = ("Ta", _attr_Ta, "Tb", _attr_Tb, "Tout", Tout, "adj_x", adj_x,
  "adj_y", adj_y, "grad_x", grad_x, "grad_y", grad_y)
  _result = _execute.execute(b"BatchMatMulV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchMatMulV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Betainc_T = TypeVar("TV_Betainc_T", _atypes.Float32, _atypes.Float64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.betainc', v1=['math.betainc', 'betainc'])
@deprecated_endpoints('betainc')
def betainc(a: Annotated[Any, TV_Betainc_T], b: Annotated[Any, TV_Betainc_T], x: Annotated[Any, TV_Betainc_T], name=None) -> Annotated[Any, TV_Betainc_T]:
  r"""Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

  The regularized incomplete beta integral is defined as:


  \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)

  where


  \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)


  is the incomplete beta function and \\(B(a, b)\\) is the *complete*
  beta function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    b: A `Tensor`. Must have the same type as `a`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Betainc", name, a, b, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_betainc(
          (a, b, x, name,), None)
      if _result is not NotImplemented:
        return _result
      return betainc_eager_fallback(
          a, b, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            betainc, (), dict(a=a, b=b, x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_betainc(
        (a, b, x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Betainc", a=a, b=b, x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          betainc, (), dict(a=a, b=b, x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Betainc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Betainc = tf_export("raw_ops.Betainc")(_ops.to_raw_op(betainc))
_dispatcher_for_betainc = betainc._tf_type_based_dispatcher.Dispatch


def betainc_eager_fallback(a: Annotated[Any, TV_Betainc_T], b: Annotated[Any, TV_Betainc_T], x: Annotated[Any, TV_Betainc_T], name, ctx) -> Annotated[Any, TV_Betainc_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b, x], ctx, [_dtypes.float32, _dtypes.float64, ])
  (a, b, x) = _inputs_T
  _inputs_flat = [a, b, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Betainc", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Betainc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Bincount_T = TypeVar("TV_Bincount_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def bincount(arr: Annotated[Any, _atypes.Int32], size: Annotated[Any, _atypes.Int32], weights: Annotated[Any, TV_Bincount_T], name=None) -> Annotated[Any, TV_Bincount_T]:
  r"""Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    arr: A `Tensor` of type `int32`. int32 `Tensor`.
    size: A `Tensor` of type `int32`. non-negative int32 scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Bincount", name, arr, size, weights)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bincount_eager_fallback(
          arr, size, weights, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Bincount", arr=arr, size=size, weights=weights, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Bincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Bincount = tf_export("raw_ops.Bincount")(_ops.to_raw_op(bincount))


def bincount_eager_fallback(arr: Annotated[Any, _atypes.Int32], size: Annotated[Any, _atypes.Int32], weights: Annotated[Any, TV_Bincount_T], name, ctx) -> Annotated[Any, TV_Bincount_T]:
  _attr_T, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  arr = _ops.convert_to_tensor(arr, _dtypes.int32)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [arr, size, weights]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Bincount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Bincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Bucketize_T = TypeVar("TV_Bucketize_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def bucketize(input: Annotated[Any, TV_Bucketize_T], boundaries, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Bucketizes 'input' based on 'boundaries'.

  For example, if the inputs are
      boundaries = [0, 10, 100]
      input = [[-5, 10000]
               [150,   10]
               [5,    100]]

  then the output will be
      output = [[0, 3]
                [3, 2]
                [1, 3]]

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Any shape of Tensor contains with int or float type.
    boundaries: A list of `floats`.
      A sorted list of floats gives the boundary of the buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Bucketize", name, input, "boundaries", boundaries)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bucketize_eager_fallback(
          input, boundaries=boundaries, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'boundaries' argument to "
        "'bucketize' Op, not %r." % boundaries)
  boundaries = [_execute.make_float(_f, "boundaries") for _f in boundaries]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Bucketize", input=input, boundaries=boundaries, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "boundaries",
              _op.get_attr("boundaries"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Bucketize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Bucketize = tf_export("raw_ops.Bucketize")(_ops.to_raw_op(bucketize))


def bucketize_eager_fallback(input: Annotated[Any, TV_Bucketize_T], boundaries, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(boundaries, (list, tuple)):
    raise TypeError(
        "Expected list for 'boundaries' argument to "
        "'bucketize' Op, not %r." % boundaries)
  boundaries = [_execute.make_float(_f, "boundaries") for _f in boundaries]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "boundaries", boundaries)
  _result = _execute.execute(b"Bucketize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Bucketize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cast_SrcT = TypeVar("TV_Cast_SrcT", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_Cast_DstT = TypeVar("TV_Cast_DstT", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def cast(x: Annotated[Any, TV_Cast_SrcT], DstT: TV_Cast_DstT, Truncate:bool=False, name=None) -> Annotated[Any, TV_Cast_DstT]:
  r"""Cast x of type SrcT to y of DstT.

  Args:
    x: A `Tensor`.
    DstT: A `tf.DType`.
    Truncate: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `DstT`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cast", name, x, "DstT", DstT, "Truncate", Truncate)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cast_eager_fallback(
          x, DstT=DstT, Truncate=Truncate, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  DstT = _execute.make_type(DstT, "DstT")
  if Truncate is None:
    Truncate = False
  Truncate = _execute.make_bool(Truncate, "Truncate")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cast", x=x, DstT=DstT, Truncate=Truncate, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("SrcT", _op._get_attr_type("SrcT"), "DstT",
              _op._get_attr_type("DstT"), "Truncate",
              _op._get_attr_bool("Truncate"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cast = tf_export("raw_ops.Cast")(_ops.to_raw_op(cast))


def cast_eager_fallback(x: Annotated[Any, TV_Cast_SrcT], DstT: TV_Cast_DstT, Truncate: bool, name, ctx) -> Annotated[Any, TV_Cast_DstT]:
  DstT = _execute.make_type(DstT, "DstT")
  if Truncate is None:
    Truncate = False
  Truncate = _execute.make_bool(Truncate, "Truncate")
  _attr_SrcT, (x,) = _execute.args_to_matching_eager([x], ctx, [])
  _inputs_flat = [x]
  _attrs = ("SrcT", _attr_SrcT, "DstT", DstT, "Truncate", Truncate)
  _result = _execute.execute(b"Cast", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Ceil_T = TypeVar("TV_Ceil_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def ceil(x: Annotated[Any, TV_Ceil_T], name=None) -> Annotated[Any, TV_Ceil_T]:
  r"""Returns element-wise smallest integer not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Ceil", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ceil_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Ceil", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Ceil", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Ceil = tf_export("raw_ops.Ceil")(_ops.to_raw_op(ceil))


def ceil_eager_fallback(x: Annotated[Any, TV_Ceil_T], name, ctx) -> Annotated[Any, TV_Ceil_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Ceil", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Ceil", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ClipByValue_T = TypeVar("TV_ClipByValue_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def _clip_by_value(t: Annotated[Any, TV_ClipByValue_T], clip_value_min: Annotated[Any, TV_ClipByValue_T], clip_value_max: Annotated[Any, TV_ClipByValue_T], name=None) -> Annotated[Any, TV_ClipByValue_T]:
  r"""Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`.
    clip_value_min: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ClipByValue", name, t, clip_value_min, clip_value_max)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _clip_by_value_eager_fallback(
          t, clip_value_min, clip_value_max, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ClipByValue", t=t, clip_value_min=clip_value_min,
                       clip_value_max=clip_value_max, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ClipByValue = tf_export("raw_ops.ClipByValue")(_ops.to_raw_op(_clip_by_value))


def _clip_by_value_eager_fallback(t: Annotated[Any, TV_ClipByValue_T], clip_value_min: Annotated[Any, TV_ClipByValue_T], clip_value_max: Annotated[Any, TV_ClipByValue_T], name, ctx) -> Annotated[Any, TV_ClipByValue_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([t, clip_value_min, clip_value_max], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (t, clip_value_min, clip_value_max) = _inputs_T
  _inputs_flat = [t, clip_value_min, clip_value_max]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ClipByValue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Complex_T = TypeVar("TV_Complex_T", _atypes.Float32, _atypes.Float64)
TV_Complex_Tout = TypeVar("TV_Complex_Tout", _atypes.Complex128, _atypes.Complex64)

def _complex(real: Annotated[Any, TV_Complex_T], imag: Annotated[Any, TV_Complex_T], Tout:TV_Complex_Tout=_dtypes.complex64, name=None) -> Annotated[Any, TV_Complex_Tout]:
  r"""Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    Tout: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Complex", name, real, imag, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _complex_eager_fallback(
          real, imag, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tout is None:
    Tout = _dtypes.complex64
  Tout = _execute.make_type(Tout, "Tout")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Complex", real=real, imag=imag, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Complex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Complex = tf_export("raw_ops.Complex")(_ops.to_raw_op(_complex))


def _complex_eager_fallback(real: Annotated[Any, TV_Complex_T], imag: Annotated[Any, TV_Complex_T], Tout: TV_Complex_Tout, name, ctx) -> Annotated[Any, TV_Complex_Tout]:
  if Tout is None:
    Tout = _dtypes.complex64
  Tout = _execute.make_type(Tout, "Tout")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([real, imag], ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  (real, imag) = _inputs_T
  _inputs_flat = [real, imag]
  _attrs = ("T", _attr_T, "Tout", Tout)
  _result = _execute.execute(b"Complex", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Complex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ComplexAbs_T = TypeVar("TV_ComplexAbs_T", _atypes.Complex128, _atypes.Complex64)
TV_ComplexAbs_Tout = TypeVar("TV_ComplexAbs_Tout", _atypes.Float32, _atypes.Float64)

def complex_abs(x: Annotated[Any, TV_ComplexAbs_T], Tout:TV_ComplexAbs_Tout=_dtypes.float32, name=None) -> Annotated[Any, TV_ComplexAbs_Tout]:
  r"""Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` or `double` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
  value is computed as \\( \sqrt{a^2 + b^2}\\).

  For example:

  >>> x = tf.complex(3.0, 4.0)
  >>> print((tf.raw_ops.ComplexAbs(x=x, Tout=tf.dtypes.float32, name=None)).numpy())
  5.0

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ComplexAbs", name, x, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return complex_abs_eager_fallback(
          x, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ComplexAbs", x=x, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ComplexAbs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ComplexAbs = tf_export("raw_ops.ComplexAbs")(_ops.to_raw_op(complex_abs))


def complex_abs_eager_fallback(x: Annotated[Any, TV_ComplexAbs_T], Tout: TV_ComplexAbs_Tout, name, ctx) -> Annotated[Any, TV_ComplexAbs_Tout]:
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T, "Tout", Tout)
  _result = _execute.execute(b"ComplexAbs", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ComplexAbs", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conj_T = TypeVar("TV_Conj_T", _atypes.Complex128, _atypes.Complex64, _atypes.Variant)

def conj(input: Annotated[Any, TV_Conj_T], name=None) -> Annotated[Any, TV_Conj_T]:
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conj", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conj_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conj", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conj", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conj = tf_export("raw_ops.Conj")(_ops.to_raw_op(conj))


def conj_eager_fallback(input: Annotated[Any, TV_Conj_T], name, ctx) -> Annotated[Any, TV_Conj_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, _dtypes.variant, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Conj", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conj", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cos_T = TypeVar("TV_Cos_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.cos', 'cos')
def cos(x: Annotated[Any, TV_Cos_T], name=None) -> Annotated[Any, TV_Cos_T]:
  r"""Computes cos of x element-wise.

    Given an input tensor, this function computes cosine of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `[-1,1]`. If input lies outside the boundary, `nan`
    is returned.

    ```python
    x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cos", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_cos(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return cos_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            cos, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_cos(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cos", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          cos, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cos = tf_export("raw_ops.Cos")(_ops.to_raw_op(cos))
_dispatcher_for_cos = cos._tf_type_based_dispatcher.Dispatch


def cos_eager_fallback(x: Annotated[Any, TV_Cos_T], name, ctx) -> Annotated[Any, TV_Cos_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Cos", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cos", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cosh_T = TypeVar("TV_Cosh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.cosh', 'cosh')
def cosh(x: Annotated[Any, TV_Cosh_T], name=None) -> Annotated[Any, TV_Cosh_T]:
  r"""Computes hyperbolic cosine of x element-wise.

    Given an input tensor, this function computes hyperbolic cosine of every
    element in the tensor. Input range is `[-inf, inf]` and output range
    is `[1, inf]`.

    ```python
    x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
    tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cosh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_cosh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return cosh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            cosh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_cosh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cosh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          cosh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cosh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cosh = tf_export("raw_ops.Cosh")(_ops.to_raw_op(cosh))
_dispatcher_for_cosh = cosh._tf_type_based_dispatcher.Dispatch


def cosh_eager_fallback(x: Annotated[Any, TV_Cosh_T], name, ctx) -> Annotated[Any, TV_Cosh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Cosh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cosh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cross_T = TypeVar("TV_Cross_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('linalg.cross', v1=['linalg.cross', 'cross'])
@deprecated_endpoints('cross')
def cross(a: Annotated[Any, TV_Cross_T], b: Annotated[Any, TV_Cross_T], name=None) -> Annotated[Any, TV_Cross_T]:
  r"""Compute the pairwise cross product.

  `a` and `b` must be the same shape; they can either be simple 3-element vectors,
  or any shape where the innermost dimension is 3. In the latter case, each pair
  of corresponding 3-element vectors is cross-multiplied independently.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A tensor containing 3-element vectors.
    b: A `Tensor`. Must have the same type as `a`.
      Another tensor, of same type and shape as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cross", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_cross(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return cross_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            cross, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_cross(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cross", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          cross, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cross", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cross = tf_export("raw_ops.Cross")(_ops.to_raw_op(cross))
_dispatcher_for_cross = cross._tf_type_based_dispatcher.Dispatch


def cross_eager_fallback(a: Annotated[Any, TV_Cross_T], b: Annotated[Any, TV_Cross_T], name, ctx) -> Annotated[Any, TV_Cross_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (a, b) = _inputs_T
  _inputs_flat = [a, b]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Cross", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cross", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cumprod_T = TypeVar("TV_Cumprod_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Cumprod_Tidx = TypeVar("TV_Cumprod_Tidx", _atypes.Int32, _atypes.Int64)

def cumprod(x: Annotated[Any, TV_Cumprod_T], axis: Annotated[Any, TV_Cumprod_Tidx], exclusive:bool=False, reverse:bool=False, name=None) -> Annotated[Any, TV_Cumprod_T]:
  r"""Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed instead:

  ```python
  tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:

  ```python
  tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumprod.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cumprod", name, x, axis, "exclusive", exclusive, "reverse",
        reverse)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cumprod_eager_fallback(
          x, axis, exclusive=exclusive, reverse=reverse, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cumprod", x=x, axis=axis, exclusive=exclusive, reverse=reverse,
                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("exclusive", _op._get_attr_bool("exclusive"), "reverse",
              _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"),
              "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cumprod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cumprod = tf_export("raw_ops.Cumprod")(_ops.to_raw_op(cumprod))


def cumprod_eager_fallback(x: Annotated[Any, TV_Cumprod_T], axis: Annotated[Any, TV_Cumprod_Tidx], exclusive: bool, reverse: bool, name, ctx) -> Annotated[Any, TV_Cumprod_T]:
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x, axis]
  _attrs = ("exclusive", exclusive, "reverse", reverse, "T", _attr_T, "Tidx",
  _attr_Tidx)
  _result = _execute.execute(b"Cumprod", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cumprod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Cumsum_T = TypeVar("TV_Cumsum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Cumsum_Tidx = TypeVar("TV_Cumsum_Tidx", _atypes.Int32, _atypes.Int64)

def cumsum(x: Annotated[Any, TV_Cumsum_T], axis: Annotated[Any, TV_Cumsum_Tidx], exclusive:bool=False, reverse:bool=False, name=None) -> Annotated[Any, TV_Cumsum_T]:
  r"""Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
  performed instead:

  ```python
  tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  ```python
  tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumsum.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Cumsum", name, x, axis, "exclusive", exclusive, "reverse",
        reverse)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cumsum_eager_fallback(
          x, axis, exclusive=exclusive, reverse=reverse, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Cumsum", x=x, axis=axis, exclusive=exclusive, reverse=reverse,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("exclusive", _op._get_attr_bool("exclusive"), "reverse",
              _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"),
              "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Cumsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Cumsum = tf_export("raw_ops.Cumsum")(_ops.to_raw_op(cumsum))


def cumsum_eager_fallback(x: Annotated[Any, TV_Cumsum_T], axis: Annotated[Any, TV_Cumsum_Tidx], exclusive: bool, reverse: bool, name, ctx) -> Annotated[Any, TV_Cumsum_T]:
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x, axis]
  _attrs = ("exclusive", exclusive, "reverse", reverse, "T", _attr_T, "Tidx",
  _attr_Tidx)
  _result = _execute.execute(b"Cumsum", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Cumsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CumulativeLogsumexp_T = TypeVar("TV_CumulativeLogsumexp_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_CumulativeLogsumexp_Tidx = TypeVar("TV_CumulativeLogsumexp_Tidx", _atypes.Int32, _atypes.Int64)

def cumulative_logsumexp(x: Annotated[Any, TV_CumulativeLogsumexp_T], axis: Annotated[Any, TV_CumulativeLogsumexp_Tidx], exclusive:bool=False, reverse:bool=False, name=None) -> Annotated[Any, TV_CumulativeLogsumexp_T]:
  r"""Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumulative log-sum-exp,
  which means that the first
  element of the input is identical to the first element of the output:
  ```python
  tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
  performed instead:
  ```python
  tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
  ```
  Note that the neutral element of the log-sum-exp operation is `-inf`,
  however, for performance reasons, the minimal value representable by the
  floating point type is used instead.

  By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
  opposite direction.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumulative log-sum-exp.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CumulativeLogsumexp", name, x, axis, "exclusive", exclusive,
        "reverse", reverse)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cumulative_logsumexp_eager_fallback(
          x, axis, exclusive=exclusive, reverse=reverse, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CumulativeLogsumexp", x=x, axis=axis, exclusive=exclusive,
                               reverse=reverse, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("exclusive", _op._get_attr_bool("exclusive"), "reverse",
              _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"),
              "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CumulativeLogsumexp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CumulativeLogsumexp = tf_export("raw_ops.CumulativeLogsumexp")(_ops.to_raw_op(cumulative_logsumexp))


def cumulative_logsumexp_eager_fallback(x: Annotated[Any, TV_CumulativeLogsumexp_T], axis: Annotated[Any, TV_CumulativeLogsumexp_Tidx], exclusive: bool, reverse: bool, name, ctx) -> Annotated[Any, TV_CumulativeLogsumexp_T]:
  if exclusive is None:
    exclusive = False
  exclusive = _execute.make_bool(exclusive, "exclusive")
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x, axis]
  _attrs = ("exclusive", exclusive, "reverse", reverse, "T", _attr_T, "Tidx",
  _attr_Tidx)
  _result = _execute.execute(b"CumulativeLogsumexp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CumulativeLogsumexp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DenseBincount_Tidx = TypeVar("TV_DenseBincount_Tidx", _atypes.Int32, _atypes.Int64)
TV_DenseBincount_T = TypeVar("TV_DenseBincount_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def dense_bincount(input: Annotated[Any, TV_DenseBincount_Tidx], size: Annotated[Any, TV_DenseBincount_Tidx], weights: Annotated[Any, TV_DenseBincount_T], binary_output:bool=False, name=None) -> Annotated[Any, TV_DenseBincount_T]:
  r"""Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1D or 2D int `Tensor`.
    size: A `Tensor`. Must have the same type as `input`.
      non-negative int scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    binary_output: An optional `bool`. Defaults to `False`.
      bool; Whether the kernel should count the appearance or number of occurrences.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DenseBincount", name, input, size, weights, "binary_output",
        binary_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dense_bincount_eager_fallback(
          input, size, weights, binary_output=binary_output, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseBincount", input=input, size=size, weights=weights,
                         binary_output=binary_output, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"), "T",
              _op._get_attr_type("T"), "binary_output",
              _op._get_attr_bool("binary_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DenseBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DenseBincount = tf_export("raw_ops.DenseBincount")(_ops.to_raw_op(dense_bincount))


def dense_bincount_eager_fallback(input: Annotated[Any, TV_DenseBincount_Tidx], size: Annotated[Any, TV_DenseBincount_Tidx], weights: Annotated[Any, TV_DenseBincount_T], binary_output: bool, name, ctx) -> Annotated[Any, TV_DenseBincount_T]:
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([input, size], ctx, [_dtypes.int32, _dtypes.int64, ])
  (input, size) = _inputs_Tidx
  _attr_T, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [input, size, weights]
  _attrs = ("Tidx", _attr_Tidx, "T", _attr_T, "binary_output", binary_output)
  _result = _execute.execute(b"DenseBincount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DenseBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Digamma_T = TypeVar("TV_Digamma_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.digamma', v1=['math.digamma', 'digamma'])
@deprecated_endpoints('digamma')
def digamma(x: Annotated[Any, TV_Digamma_T], name=None) -> Annotated[Any, TV_Digamma_T]:
  r"""Computes Psi, the derivative of Lgamma (the log of the absolute value of

  `Gamma(x)`), element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Digamma", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_digamma(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return digamma_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            digamma, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_digamma(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Digamma", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          digamma, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Digamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Digamma = tf_export("raw_ops.Digamma")(_ops.to_raw_op(digamma))
_dispatcher_for_digamma = digamma._tf_type_based_dispatcher.Dispatch


def digamma_eager_fallback(x: Annotated[Any, TV_Digamma_T], name, ctx) -> Annotated[Any, TV_Digamma_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Digamma", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Digamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Div_T = TypeVar("TV_Div_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def div(x: Annotated[Any, TV_Div_T], y: Annotated[Any, TV_Div_T], name=None) -> Annotated[Any, TV_Div_T]:
  r"""Returns x / y element-wise.

  *NOTE*: `Div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `uint64`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Div", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Div", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Div", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Div = tf_export("raw_ops.Div")(_ops.to_raw_op(div))


def div_eager_fallback(x: Annotated[Any, TV_Div_T], y: Annotated[Any, TV_Div_T], name, ctx) -> Annotated[Any, TV_Div_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.uint64, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Div", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Div", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DivNoNan_T = TypeVar("TV_DivNoNan_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def div_no_nan(x: Annotated[Any, TV_DivNoNan_T], y: Annotated[Any, TV_DivNoNan_T], name=None) -> Annotated[Any, TV_DivNoNan_T]:
  r"""Returns 0 if the denominator is zero.

  
  *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `bfloat16`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DivNoNan", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return div_no_nan_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DivNoNan", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DivNoNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DivNoNan = tf_export("raw_ops.DivNoNan")(_ops.to_raw_op(div_no_nan))


def div_no_nan_eager_fallback(x: Annotated[Any, TV_DivNoNan_T], y: Annotated[Any, TV_DivNoNan_T], name, ctx) -> Annotated[Any, TV_DivNoNan_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.half, _dtypes.float32, _dtypes.bfloat16, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DivNoNan", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DivNoNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Equal_T = TypeVar("TV_Equal_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def equal(x: Annotated[Any, TV_Equal_T], y: Annotated[Any, TV_Equal_T], incompatible_shape_error:bool=True, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x == y) element-wise.

  *NOTE*: `Equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ```python
  x = tf.constant([2, 4])
  y = tf.constant(2)
  tf.math.equal(x, y) ==> array([True, False])

  x = tf.constant([2, 4])
  y = tf.constant([2, 4])
  tf.math.equal(x, y) ==> array([True,  True])
  ```

  Args:
    x: A `Tensor`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Equal", name, x, y, "incompatible_shape_error",
        incompatible_shape_error)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return equal_eager_fallback(
          x, y, incompatible_shape_error=incompatible_shape_error, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if incompatible_shape_error is None:
    incompatible_shape_error = True
  incompatible_shape_error = _execute.make_bool(incompatible_shape_error, "incompatible_shape_error")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Equal", x=x, y=y, incompatible_shape_error=incompatible_shape_error,
                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "incompatible_shape_error",
              _op._get_attr_bool("incompatible_shape_error"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Equal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Equal = tf_export("raw_ops.Equal")(_ops.to_raw_op(equal))


def equal_eager_fallback(x: Annotated[Any, TV_Equal_T], y: Annotated[Any, TV_Equal_T], incompatible_shape_error: bool, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if incompatible_shape_error is None:
    incompatible_shape_error = True
  incompatible_shape_error = _execute.make_bool(incompatible_shape_error, "incompatible_shape_error")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "incompatible_shape_error",
  incompatible_shape_error)
  _result = _execute.execute(b"Equal", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Equal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Erf_T = TypeVar("TV_Erf_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.erf', v1=['math.erf', 'erf'])
@deprecated_endpoints('erf')
def erf(x: Annotated[Any, TV_Erf_T], name=None) -> Annotated[Any, TV_Erf_T]:
  r"""Computes the [Gauss error function](https://en.wikipedia.org/wiki/Error_function) of `x` element-wise. In statistics, for non-negative values of $x$, the error function has the following interpretation: for a random variable $Y$ that is normally distributed with mean 0 and variance $1/\sqrt{2}$, $erf(x)$ is the probability that $Y$ falls in the range $[−x, x]$.

  For example:

  >>> tf.math.erf([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]])
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 0.8427007,  0.9953223,  0.999978 ],
         [ 0.       , -0.8427007, -0.9953223]], dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Erf", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_erf(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return erf_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            erf, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_erf(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Erf", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          erf, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Erf", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Erf = tf_export("raw_ops.Erf")(_ops.to_raw_op(erf))
_dispatcher_for_erf = erf._tf_type_based_dispatcher.Dispatch


def erf_eager_fallback(x: Annotated[Any, TV_Erf_T], name, ctx) -> Annotated[Any, TV_Erf_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Erf", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Erf", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Erfc_T = TypeVar("TV_Erfc_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.erfc', v1=['math.erfc', 'erfc'])
@deprecated_endpoints('erfc')
def erfc(x: Annotated[Any, TV_Erfc_T], name=None) -> Annotated[Any, TV_Erfc_T]:
  r"""Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Erfc", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_erfc(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return erfc_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            erfc, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_erfc(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Erfc", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          erfc, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Erfc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Erfc = tf_export("raw_ops.Erfc")(_ops.to_raw_op(erfc))
_dispatcher_for_erfc = erfc._tf_type_based_dispatcher.Dispatch


def erfc_eager_fallback(x: Annotated[Any, TV_Erfc_T], name, ctx) -> Annotated[Any, TV_Erfc_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Erfc", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Erfc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Erfinv_T = TypeVar("TV_Erfinv_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def erfinv(x: Annotated[Any, TV_Erfinv_T], name=None) -> Annotated[Any, TV_Erfinv_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Erfinv", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return erfinv_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Erfinv", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Erfinv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Erfinv = tf_export("raw_ops.Erfinv")(_ops.to_raw_op(erfinv))


def erfinv_eager_fallback(x: Annotated[Any, TV_Erfinv_T], name, ctx) -> Annotated[Any, TV_Erfinv_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Erfinv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Erfinv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_EuclideanNorm_T = TypeVar("TV_EuclideanNorm_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_EuclideanNorm_Tidx = TypeVar("TV_EuclideanNorm_Tidx", _atypes.Int32, _atypes.Int64)

def euclidean_norm(input: Annotated[Any, TV_EuclideanNorm_T], axis: Annotated[Any, TV_EuclideanNorm_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_EuclideanNorm_T]:
  r"""Computes the euclidean norm of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EuclideanNorm", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return euclidean_norm_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EuclideanNorm", input=input, reduction_indices=axis,
                         keep_dims=keep_dims, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EuclideanNorm", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EuclideanNorm = tf_export("raw_ops.EuclideanNorm")(_ops.to_raw_op(euclidean_norm))


def euclidean_norm_eager_fallback(input: Annotated[Any, TV_EuclideanNorm_T], axis: Annotated[Any, TV_EuclideanNorm_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_EuclideanNorm_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"EuclideanNorm", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EuclideanNorm", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Exp_T = TypeVar("TV_Exp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def exp(x: Annotated[Any, TV_Exp_T], name=None) -> Annotated[Any, TV_Exp_T]:
  r"""Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of every element in the input tensor.
    i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
    `e` denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    ```python
    x = tf.constant(2.0)
    tf.math.exp(x) ==> 7.389056

    x = tf.constant([2.0, 8.0])
    tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
    ```

    For complex numbers, the exponential value is calculated as follows:

    ```
    e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
    ```

    Let's consider complex number 1+1j as an example.
    e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)

    ```python
    x = tf.constant(1 + 1j)
    tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Exp", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return exp_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Exp", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Exp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Exp = tf_export("raw_ops.Exp")(_ops.to_raw_op(exp))


def exp_eager_fallback(x: Annotated[Any, TV_Exp_T], name, ctx) -> Annotated[Any, TV_Exp_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Exp", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Exp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Expm1_T = TypeVar("TV_Expm1_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.expm1', v1=['math.expm1', 'expm1'])
@deprecated_endpoints('expm1')
def expm1(x: Annotated[Any, TV_Expm1_T], name=None) -> Annotated[Any, TV_Expm1_T]:
  r"""Computes `exp(x) - 1` element-wise.

    i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
    `e` denotes Euler's number and is approximately equal to 2.718281.

    ```python
    x = tf.constant(2.0)
    tf.math.expm1(x) ==> 6.389056

    x = tf.constant([2.0, 8.0])
    tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

    x = tf.constant(1 + 1j)
    tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Expm1", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_expm1(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return expm1_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            expm1, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_expm1(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Expm1", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          expm1, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Expm1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Expm1 = tf_export("raw_ops.Expm1")(_ops.to_raw_op(expm1))
_dispatcher_for_expm1 = expm1._tf_type_based_dispatcher.Dispatch


def expm1_eager_fallback(x: Annotated[Any, TV_Expm1_T], name, ctx) -> Annotated[Any, TV_Expm1_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Expm1", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Expm1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Floor_T = TypeVar("TV_Floor_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def floor(x: Annotated[Any, TV_Floor_T], name=None) -> Annotated[Any, TV_Floor_T]:
  r"""Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Floor", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return floor_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Floor", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Floor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Floor = tf_export("raw_ops.Floor")(_ops.to_raw_op(floor))


def floor_eager_fallback(x: Annotated[Any, TV_Floor_T], name, ctx) -> Annotated[Any, TV_Floor_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Floor", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Floor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FloorDiv_T = TypeVar("TV_FloorDiv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export(v1=['floor_div'])
@deprecated_endpoints('floor_div')
def floor_div(x: Annotated[Any, TV_FloorDiv_T], y: Annotated[Any, TV_FloorDiv_T], name=None) -> Annotated[Any, TV_FloorDiv_T]:
  r"""Returns x // y element-wise.

  *NOTE*: `floor_div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `uint64`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FloorDiv", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_floor_div(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return floor_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            floor_div, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_floor_div(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FloorDiv", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          floor_div, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FloorDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FloorDiv = tf_export("raw_ops.FloorDiv")(_ops.to_raw_op(floor_div))
_dispatcher_for_floor_div = floor_div._tf_type_based_dispatcher.Dispatch


def floor_div_eager_fallback(x: Annotated[Any, TV_FloorDiv_T], y: Annotated[Any, TV_FloorDiv_T], name, ctx) -> Annotated[Any, TV_FloorDiv_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.uint64, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FloorDiv", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FloorDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FloorMod_T = TypeVar("TV_FloorMod_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.floormod', 'math.mod', v1=['math.floormod', 'floormod', 'math.mod', 'mod'])
@deprecated_endpoints('floormod', 'mod')
def floor_mod(x: Annotated[Any, TV_FloorMod_T], y: Annotated[Any, TV_FloorMod_T], name=None) -> Annotated[Any, TV_FloorMod_T]:
  r"""Returns element-wise remainder of division.

  This follows Python semantics in that the
  result here is consistent with a flooring divide. E.g.
  `floor(x / y) * y + floormod(x, y) = x`, regardless of the signs of x and y.

  *NOTE*: `math.floormod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bfloat16`, `half`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FloorMod", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_floor_mod(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return floor_mod_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            floor_mod, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_floor_mod(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FloorMod", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          floor_mod, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FloorMod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FloorMod = tf_export("raw_ops.FloorMod")(_ops.to_raw_op(floor_mod))
_dispatcher_for_floor_mod = floor_mod._tf_type_based_dispatcher.Dispatch


def floor_mod_eager_fallback(x: Annotated[Any, TV_FloorMod_T], y: Annotated[Any, TV_FloorMod_T], name, ctx) -> Annotated[Any, TV_FloorMod_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"FloorMod", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FloorMod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Greater_T = TypeVar("TV_Greater_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.greater', 'greater')
def greater(x: Annotated[Any, TV_Greater_T], y: Annotated[Any, TV_Greater_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x > y) element-wise.

  *NOTE*: `math.greater` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Example:

  ```python
  x = tf.constant([5, 4, 6])
  y = tf.constant([5, 2, 5])
  tf.math.greater(x, y) ==> [False, True, True]

  x = tf.constant([5, 4, 6])
  y = tf.constant([5])
  tf.math.greater(x, y) ==> [False, False, True]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Greater", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_greater(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return greater_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            greater, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_greater(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Greater", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          greater, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Greater", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Greater = tf_export("raw_ops.Greater")(_ops.to_raw_op(greater))
_dispatcher_for_greater = greater._tf_type_based_dispatcher.Dispatch


def greater_eager_fallback(x: Annotated[Any, TV_Greater_T], y: Annotated[Any, TV_Greater_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Greater", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Greater", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_GreaterEqual_T = TypeVar("TV_GreaterEqual_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.greater_equal', 'greater_equal')
def greater_equal(x: Annotated[Any, TV_GreaterEqual_T], y: Annotated[Any, TV_GreaterEqual_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x >= y) element-wise.

  *NOTE*: `math.greater_equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Example:

  ```python
  x = tf.constant([5, 4, 6, 7])
  y = tf.constant([5, 2, 5, 10])
  tf.math.greater_equal(x, y) ==> [True, True, True, False]

  x = tf.constant([5, 4, 6, 7])
  y = tf.constant([5])
  tf.math.greater_equal(x, y) ==> [True, False, True, True]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GreaterEqual", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_greater_equal(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return greater_equal_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            greater_equal, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_greater_equal(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GreaterEqual", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          greater_equal, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GreaterEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GreaterEqual = tf_export("raw_ops.GreaterEqual")(_ops.to_raw_op(greater_equal))
_dispatcher_for_greater_equal = greater_equal._tf_type_based_dispatcher.Dispatch


def greater_equal_eager_fallback(x: Annotated[Any, TV_GreaterEqual_T], y: Annotated[Any, TV_GreaterEqual_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"GreaterEqual", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GreaterEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_HistogramFixedWidth_T = TypeVar("TV_HistogramFixedWidth_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)
TV_HistogramFixedWidth_dtype = TypeVar("TV_HistogramFixedWidth_dtype", _atypes.Int32, _atypes.Int64)

def _histogram_fixed_width(values: Annotated[Any, TV_HistogramFixedWidth_T], value_range: Annotated[Any, TV_HistogramFixedWidth_T], nbins: Annotated[Any, _atypes.Int32], dtype:TV_HistogramFixedWidth_dtype=_dtypes.int32, name=None) -> Annotated[Any, TV_HistogramFixedWidth_dtype]:
  r"""Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fall into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.get_default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```

  Args:
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Numeric `Tensor`.
    value_range: A `Tensor`. Must have the same type as `values`.
      Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins: A `Tensor` of type `int32`.
      Scalar `int32 Tensor`.  Number of histogram bins.
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "HistogramFixedWidth", name, values, value_range, nbins,
        "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _histogram_fixed_width_eager_fallback(
          values, value_range, nbins, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.int32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "HistogramFixedWidth", values=values, value_range=value_range,
                               nbins=nbins, dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "dtype",
              _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "HistogramFixedWidth", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

HistogramFixedWidth = tf_export("raw_ops.HistogramFixedWidth")(_ops.to_raw_op(_histogram_fixed_width))


def _histogram_fixed_width_eager_fallback(values: Annotated[Any, TV_HistogramFixedWidth_T], value_range: Annotated[Any, TV_HistogramFixedWidth_T], nbins: Annotated[Any, _atypes.Int32], dtype: TV_HistogramFixedWidth_dtype, name, ctx) -> Annotated[Any, TV_HistogramFixedWidth_dtype]:
  if dtype is None:
    dtype = _dtypes.int32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([values, value_range], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  (values, value_range) = _inputs_T
  nbins = _ops.convert_to_tensor(nbins, _dtypes.int32)
  _inputs_flat = [values, value_range, nbins]
  _attrs = ("T", _attr_T, "dtype", dtype)
  _result = _execute.execute(b"HistogramFixedWidth", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "HistogramFixedWidth", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Igamma_T = TypeVar("TV_Igamma_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.igamma', v1=['math.igamma', 'igamma'])
@deprecated_endpoints('igamma')
def igamma(a: Annotated[Any, TV_Igamma_T], x: Annotated[Any, TV_Igamma_T], name=None) -> Annotated[Any, TV_Igamma_T]:
  r"""Compute the lower regularized incomplete Gamma function `P(a, x)`.

  The lower regularized incomplete Gamma function is defined as:


  \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)

  where

  \\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)

  is the lower incomplete Gamma function.

  Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Igamma", name, a, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_igamma(
          (a, x, name,), None)
      if _result is not NotImplemented:
        return _result
      return igamma_eager_fallback(
          a, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            igamma, (), dict(a=a, x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_igamma(
        (a, x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Igamma", a=a, x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          igamma, (), dict(a=a, x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Igamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Igamma = tf_export("raw_ops.Igamma")(_ops.to_raw_op(igamma))
_dispatcher_for_igamma = igamma._tf_type_based_dispatcher.Dispatch


def igamma_eager_fallback(a: Annotated[Any, TV_Igamma_T], x: Annotated[Any, TV_Igamma_T], name, ctx) -> Annotated[Any, TV_Igamma_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (a, x) = _inputs_T
  _inputs_flat = [a, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Igamma", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Igamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IgammaGradA_T = TypeVar("TV_IgammaGradA_T", _atypes.Float32, _atypes.Float64)

def igamma_grad_a(a: Annotated[Any, TV_IgammaGradA_T], x: Annotated[Any, TV_IgammaGradA_T], name=None) -> Annotated[Any, TV_IgammaGradA_T]:
  r"""Computes the gradient of `igamma(a, x)` wrt `a`.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IgammaGradA", name, a, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return igamma_grad_a_eager_fallback(
          a, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IgammaGradA", a=a, x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IgammaGradA", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IgammaGradA = tf_export("raw_ops.IgammaGradA")(_ops.to_raw_op(igamma_grad_a))


def igamma_grad_a_eager_fallback(a: Annotated[Any, TV_IgammaGradA_T], x: Annotated[Any, TV_IgammaGradA_T], name, ctx) -> Annotated[Any, TV_IgammaGradA_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, x], ctx, [_dtypes.float32, _dtypes.float64, ])
  (a, x) = _inputs_T
  _inputs_flat = [a, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"IgammaGradA", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IgammaGradA", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Igammac_T = TypeVar("TV_Igammac_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.igammac', v1=['math.igammac', 'igammac'])
@deprecated_endpoints('igammac')
def igammac(a: Annotated[Any, TV_Igammac_T], x: Annotated[Any, TV_Igammac_T], name=None) -> Annotated[Any, TV_Igammac_T]:
  r"""Compute the upper regularized incomplete Gamma function `Q(a, x)`.

  The upper regularized incomplete Gamma function is defined as:

  \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)

  where

  \\(Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt\\)

  is the upper incomplete Gamma function.

  Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Igammac", name, a, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_igammac(
          (a, x, name,), None)
      if _result is not NotImplemented:
        return _result
      return igammac_eager_fallback(
          a, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            igammac, (), dict(a=a, x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_igammac(
        (a, x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Igammac", a=a, x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          igammac, (), dict(a=a, x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Igammac", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Igammac = tf_export("raw_ops.Igammac")(_ops.to_raw_op(igammac))
_dispatcher_for_igammac = igammac._tf_type_based_dispatcher.Dispatch


def igammac_eager_fallback(a: Annotated[Any, TV_Igammac_T], x: Annotated[Any, TV_Igammac_T], name, ctx) -> Annotated[Any, TV_Igammac_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (a, x) = _inputs_T
  _inputs_flat = [a, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Igammac", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Igammac", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Imag_T = TypeVar("TV_Imag_T", _atypes.Complex128, _atypes.Complex64)
TV_Imag_Tout = TypeVar("TV_Imag_Tout", _atypes.Float32, _atypes.Float64)

def imag(input: Annotated[Any, TV_Imag_T], Tout:TV_Imag_Tout=_dtypes.float32, name=None) -> Annotated[Any, TV_Imag_Tout]:
  r"""Returns the imaginary part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the imaginary part of each element in `input`. All
  elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(input) ==> [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Imag", name, input, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return imag_eager_fallback(
          input, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Imag", input=input, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Imag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Imag = tf_export("raw_ops.Imag")(_ops.to_raw_op(imag))


def imag_eager_fallback(input: Annotated[Any, TV_Imag_T], Tout: TV_Imag_Tout, name, ctx) -> Annotated[Any, TV_Imag_Tout]:
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "Tout", Tout)
  _result = _execute.execute(b"Imag", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Imag", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Inv_T = TypeVar("TV_Inv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

def inv(x: Annotated[Any, TV_Inv_T], name=None) -> Annotated[Any, TV_Inv_T]:
  r"""Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Inv", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return inv_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Inv", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Inv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Inv = tf_export("raw_ops.Inv")(_ops.to_raw_op(inv))


def inv_eager_fallback(x: Annotated[Any, TV_Inv_T], name, ctx) -> Annotated[Any, TV_Inv_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Inv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Inv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InvGrad_T = TypeVar("TV_InvGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def inv_grad(y: Annotated[Any, TV_InvGrad_T], dy: Annotated[Any, TV_InvGrad_T], name=None) -> Annotated[Any, TV_InvGrad_T]:
  r"""Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InvGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return inv_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InvGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InvGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InvGrad = tf_export("raw_ops.InvGrad")(_ops.to_raw_op(inv_grad))


def inv_grad_eager_fallback(y: Annotated[Any, TV_InvGrad_T], dy: Annotated[Any, TV_InvGrad_T], name, ctx) -> Annotated[Any, TV_InvGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InvGrad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InvGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IsFinite_T = TypeVar("TV_IsFinite_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.is_finite', v1=['math.is_finite', 'debugging.is_finite', 'is_finite'])
@deprecated_endpoints('debugging.is_finite', 'is_finite')
def is_finite(x: Annotated[Any, TV_IsFinite_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns which elements of x are finite.

  @compatibility(numpy)
  Equivalent to np.isfinite
  @end_compatibility

  Example:

  ```python
  x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
  tf.math.is_finite(x) ==> [True, True, True, False, False]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsFinite", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_finite(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return is_finite_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_finite, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_finite(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsFinite", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_finite, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsFinite", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsFinite = tf_export("raw_ops.IsFinite")(_ops.to_raw_op(is_finite))
_dispatcher_for_is_finite = is_finite._tf_type_based_dispatcher.Dispatch


def is_finite_eager_fallback(x: Annotated[Any, TV_IsFinite_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"IsFinite", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsFinite", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IsInf_T = TypeVar("TV_IsInf_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.is_inf', v1=['math.is_inf', 'debugging.is_inf', 'is_inf'])
@deprecated_endpoints('debugging.is_inf', 'is_inf')
def is_inf(x: Annotated[Any, TV_IsInf_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns which elements of x are Inf.

  @compatibility(numpy)
  Equivalent to np.isinf
  @end_compatibility

  Example:

  ```python
  x = tf.constant([5.0, np.inf, 6.8, np.inf])
  tf.math.is_inf(x) ==> [False, True, False, True]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsInf", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_inf(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return is_inf_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_inf, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_inf(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsInf", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_inf, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsInf", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsInf = tf_export("raw_ops.IsInf")(_ops.to_raw_op(is_inf))
_dispatcher_for_is_inf = is_inf._tf_type_based_dispatcher.Dispatch


def is_inf_eager_fallback(x: Annotated[Any, TV_IsInf_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"IsInf", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsInf", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IsNan_T = TypeVar("TV_IsNan_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.is_nan', v1=['math.is_nan', 'debugging.is_nan', 'is_nan'])
@deprecated_endpoints('debugging.is_nan', 'is_nan')
def is_nan(x: Annotated[Any, TV_IsNan_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns which elements of x are NaN.

  @compatibility(numpy)
  Equivalent to np.isnan
  @end_compatibility

  Example:

  ```python
  x = tf.constant([5.0, np.nan, 6.8, np.nan, np.inf])
  tf.math.is_nan(x) ==> [False, True, False, True, False]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsNan", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_nan(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return is_nan_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_nan, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_nan(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsNan", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_nan, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsNan = tf_export("raw_ops.IsNan")(_ops.to_raw_op(is_nan))
_dispatcher_for_is_nan = is_nan._tf_type_based_dispatcher.Dispatch


def is_nan_eager_fallback(x: Annotated[Any, TV_IsNan_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"IsNan", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Less_T = TypeVar("TV_Less_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.less', 'less')
def less(x: Annotated[Any, TV_Less_T], y: Annotated[Any, TV_Less_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x < y) element-wise.

  *NOTE*: `math.less` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Example:

  ```python
  x = tf.constant([5, 4, 6])
  y = tf.constant([5])
  tf.math.less(x, y) ==> [False, True, False]

  x = tf.constant([5, 4, 6])
  y = tf.constant([5, 6, 7])
  tf.math.less(x, y) ==> [False, True, True]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Less", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_less(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return less_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            less, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_less(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Less", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          less, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Less", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Less = tf_export("raw_ops.Less")(_ops.to_raw_op(less))
_dispatcher_for_less = less._tf_type_based_dispatcher.Dispatch


def less_eager_fallback(x: Annotated[Any, TV_Less_T], y: Annotated[Any, TV_Less_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Less", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Less", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LessEqual_T = TypeVar("TV_LessEqual_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.less_equal', 'less_equal')
def less_equal(x: Annotated[Any, TV_LessEqual_T], y: Annotated[Any, TV_LessEqual_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x <= y) element-wise.

  *NOTE*: `math.less_equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Example:

  ```python
  x = tf.constant([5, 4, 6])
  y = tf.constant([5])
  tf.math.less_equal(x, y) ==> [True, True, False]

  x = tf.constant([5, 4, 6])
  y = tf.constant([5, 6, 6])
  tf.math.less_equal(x, y) ==> [True, True, True]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LessEqual", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_less_equal(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return less_equal_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            less_equal, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_less_equal(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LessEqual", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          less_equal, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LessEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LessEqual = tf_export("raw_ops.LessEqual")(_ops.to_raw_op(less_equal))
_dispatcher_for_less_equal = less_equal._tf_type_based_dispatcher.Dispatch


def less_equal_eager_fallback(x: Annotated[Any, TV_LessEqual_T], y: Annotated[Any, TV_LessEqual_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"LessEqual", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LessEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Lgamma_T = TypeVar("TV_Lgamma_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.lgamma', v1=['math.lgamma', 'lgamma'])
@deprecated_endpoints('lgamma')
def lgamma(x: Annotated[Any, TV_Lgamma_T], name=None) -> Annotated[Any, TV_Lgamma_T]:
  r"""Computes the log of the absolute value of `Gamma(x)` element-wise.

    For positive numbers, this function computes log((input - 1)!) for every element in the tensor.
    `lgamma(5) = log((5-1)!) = log(4!) = log(24) = 3.1780539`

  Example:

  ```python
  x = tf.constant([0, 0.5, 1, 4.5, -4, -5.6])
  tf.math.lgamma(x) ==> [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Lgamma", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_lgamma(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return lgamma_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            lgamma, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_lgamma(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Lgamma", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          lgamma, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Lgamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Lgamma = tf_export("raw_ops.Lgamma")(_ops.to_raw_op(lgamma))
_dispatcher_for_lgamma = lgamma._tf_type_based_dispatcher.Dispatch


def lgamma_eager_fallback(x: Annotated[Any, TV_Lgamma_T], name, ctx) -> Annotated[Any, TV_Lgamma_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Lgamma", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Lgamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LinSpace_T = TypeVar("TV_LinSpace_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_LinSpace_Tidx = TypeVar("TV_LinSpace_Tidx", _atypes.Int32, _atypes.Int64)

def lin_space(start: Annotated[Any, TV_LinSpace_T], stop: Annotated[Any, TV_LinSpace_T], num: Annotated[Any, TV_LinSpace_Tidx], name=None) -> Annotated[Any, TV_LinSpace_T]:
  r"""Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by
  `(stop - start) / (num - 1)`, so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      0-D tensor. First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      0-D tensor. Last entry in the range.
    num: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D tensor. Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LinSpace", name, start, stop, num)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lin_space_eager_fallback(
          start, stop, num, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LinSpace", start=start, stop=stop, num=num, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LinSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LinSpace = tf_export("raw_ops.LinSpace")(_ops.to_raw_op(lin_space))


def lin_space_eager_fallback(start: Annotated[Any, TV_LinSpace_T], stop: Annotated[Any, TV_LinSpace_T], num: Annotated[Any, TV_LinSpace_Tidx], name, ctx) -> Annotated[Any, TV_LinSpace_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([start, stop], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (start, stop) = _inputs_T
  _attr_Tidx, (num,) = _execute.args_to_matching_eager([num], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [start, stop, num]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"LinSpace", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LinSpace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Log_T = TypeVar("TV_Log_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.log', v1=['math.log', 'log'])
@deprecated_endpoints('log')
def log(x: Annotated[Any, TV_Log_T], name=None) -> Annotated[Any, TV_Log_T]:
  r"""Computes natural logarithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Example:
  >>> x = tf.constant([0, 0.5, 1, 5])
  >>> tf.math.log(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([      -inf, -0.6931472,  0.       ,  1.609438 ], dtype=float32)>

  See: https://en.wikipedia.org/wiki/Logarithm

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Log", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_log(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return log_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            log, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_log(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Log", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          log, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Log", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Log = tf_export("raw_ops.Log")(_ops.to_raw_op(log))
_dispatcher_for_log = log._tf_type_based_dispatcher.Dispatch


def log_eager_fallback(x: Annotated[Any, TV_Log_T], name, ctx) -> Annotated[Any, TV_Log_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Log", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Log", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Log1p_T = TypeVar("TV_Log1p_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.log1p', v1=['math.log1p', 'log1p'])
@deprecated_endpoints('log1p')
def log1p(x: Annotated[Any, TV_Log1p_T], name=None) -> Annotated[Any, TV_Log1p_T]:
  r"""Computes natural logarithm of (1 + x) element-wise.

  I.e., \\(y = \log_e (1 + x)\\).

  Example:
  >>> x = tf.constant([0, 0.5, 1, 5])
  >>> tf.math.log1p(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.       , 0.4054651, 0.6931472, 1.7917595], dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Log1p", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_log1p(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return log1p_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            log1p, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_log1p(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Log1p", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          log1p, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Log1p", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Log1p = tf_export("raw_ops.Log1p")(_ops.to_raw_op(log1p))
_dispatcher_for_log1p = log1p._tf_type_based_dispatcher.Dispatch


def log1p_eager_fallback(x: Annotated[Any, TV_Log1p_T], name, ctx) -> Annotated[Any, TV_Log1p_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Log1p", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Log1p", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.logical_and', 'logical_and')
def logical_and(x: Annotated[Any, _atypes.Bool], y: Annotated[Any, _atypes.Bool], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of x AND y element-wise.

  Logical AND function.

  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `x` and `y` can be:

    - Two single elements of type `bool`.
    - One `tf.Tensor` of type `bool` and one single `bool`, where the result will
      be calculated by applying logical AND with the single element to each
      element in the larger Tensor.
    - Two `tf.Tensor` objects of type `bool` of the same shape. In this case,
      the result will be the element-wise logical AND of the two input tensors.

  You can also use the `&` operator instead.

  Usage:

    >>> a = tf.constant([True])
    >>> b = tf.constant([False])
    >>> tf.math.logical_and(a, b)
    <tf.Tensor: shape=(1,), dtype=bool, numpy=array([False])>
    >>> a & b
    <tf.Tensor: shape=(1,), dtype=bool, numpy=array([False])>

    >>> c = tf.constant([True])
    >>> x = tf.constant([False, True, True, False])
    >>> tf.math.logical_and(c, x)
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False,  True,  True, False])>
    >>> c & x
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False,  True,  True, False])>

    >>> y = tf.constant([False, False, True, True])
    >>> z = tf.constant([False, True, False, True])
    >>> tf.math.logical_and(y, z)
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, False, False, True])>
    >>> y & z
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, False, False, True])>

    This op also supports broadcasting

    >>> tf.logical_and([[True, False]], [[True], [False]])
    <tf.Tensor: shape=(2, 2), dtype=bool, numpy=
      array([[ True, False],
             [False, False]])>

  The reduction version of this elementwise operation is `tf.math.reduce_all`.

  Args:
      x: A `tf.Tensor` of type bool.
      y: A `tf.Tensor` of type bool.
      name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of type bool with the shape that `x` and `y` broadcast to.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LogicalAnd", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_logical_and(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return logical_and_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            logical_and, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_logical_and(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LogicalAnd", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          logical_and, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LogicalAnd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LogicalAnd = tf_export("raw_ops.LogicalAnd")(_ops.to_raw_op(logical_and))
_dispatcher_for_logical_and = logical_and._tf_type_based_dispatcher.Dispatch


def logical_and_eager_fallback(x: Annotated[Any, _atypes.Bool], y: Annotated[Any, _atypes.Bool], name, ctx) -> Annotated[Any, _atypes.Bool]:
  x = _ops.convert_to_tensor(x, _dtypes.bool)
  y = _ops.convert_to_tensor(y, _dtypes.bool)
  _inputs_flat = [x, y]
  _attrs = None
  _result = _execute.execute(b"LogicalAnd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LogicalAnd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.logical_not', 'logical_not')
def logical_not(x: Annotated[Any, _atypes.Bool], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of `NOT x` element-wise.

  Example:

  >>> tf.math.logical_not(tf.constant([True, False]))
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>

  Args:
    x: A `Tensor` of type `bool`. A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LogicalNot", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_logical_not(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return logical_not_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            logical_not, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_logical_not(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LogicalNot", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          logical_not, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LogicalNot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LogicalNot = tf_export("raw_ops.LogicalNot")(_ops.to_raw_op(logical_not))
_dispatcher_for_logical_not = logical_not._tf_type_based_dispatcher.Dispatch


def logical_not_eager_fallback(x: Annotated[Any, _atypes.Bool], name, ctx) -> Annotated[Any, _atypes.Bool]:
  x = _ops.convert_to_tensor(x, _dtypes.bool)
  _inputs_flat = [x]
  _attrs = None
  _result = _execute.execute(b"LogicalNot", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LogicalNot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.logical_or', 'logical_or')
def logical_or(x: Annotated[Any, _atypes.Bool], y: Annotated[Any, _atypes.Bool], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of x OR y element-wise.

  Logical OR function.

  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `x` and `y` can be:

  - Two single elements of type `bool`.
  - One `tf.Tensor` of type `bool` and one single `bool`, where the result will
    be calculated by applying logical OR with the single element to each
    element in the larger Tensor.
  - Two `tf.Tensor` objects of type `bool` of the same shape. In this case,
    the result will be the element-wise logical OR of the two input tensors.

  You can also use the `|` operator instead.

  Usage:

    >>> a = tf.constant([True])
    >>> b = tf.constant([False])
    >>> tf.math.logical_or(a, b)
    <tf.Tensor: shape=(1,), dtype=bool, numpy=array([ True])>
    >>> a | b
    <tf.Tensor: shape=(1,), dtype=bool, numpy=array([ True])>

    >>> c = tf.constant([False])
    >>> x = tf.constant([False, True, True, False])
    >>> tf.math.logical_or(c, x)
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, True,  True, False])>
    >>> c | x
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, True,  True, False])>

    >>> y = tf.constant([False, False, True, True])
    >>> z = tf.constant([False, True, False, True])
    >>> tf.math.logical_or(y, z)
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, True, True, True])>
    >>> y | z
    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, True, True, True])>

    This op also supports broadcasting

    >>> tf.logical_or([[True, False]], [[True], [False]])
    <tf.Tensor: shape=(2, 2), dtype=bool, numpy=
    array([[ True,  True],
         [ True, False]])>

  The reduction version of this elementwise operation is `tf.math.reduce_any`.

  Args:
      x: A `tf.Tensor` of type bool.
      y: A `tf.Tensor` of type bool.
      name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of type bool with the shape that `x` and `y` broadcast to.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LogicalOr", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_logical_or(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return logical_or_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            logical_or, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_logical_or(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LogicalOr", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          logical_or, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LogicalOr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LogicalOr = tf_export("raw_ops.LogicalOr")(_ops.to_raw_op(logical_or))
_dispatcher_for_logical_or = logical_or._tf_type_based_dispatcher.Dispatch


def logical_or_eager_fallback(x: Annotated[Any, _atypes.Bool], y: Annotated[Any, _atypes.Bool], name, ctx) -> Annotated[Any, _atypes.Bool]:
  x = _ops.convert_to_tensor(x, _dtypes.bool)
  y = _ops.convert_to_tensor(y, _dtypes.bool)
  _inputs_flat = [x, y]
  _attrs = None
  _result = _execute.execute(b"LogicalOr", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LogicalOr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MatMul_T = TypeVar("TV_MatMul_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def mat_mul(a: Annotated[Any, TV_MatMul_T], b: Annotated[Any, TV_MatMul_T], transpose_a:bool=False, transpose_b:bool=False, grad_a:bool=False, grad_b:bool=False, name=None) -> Annotated[Any, TV_MatMul_T]:
  r"""Multiply the matrix "a" by the matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of
  "a" (after being transposed if transpose_a is true) must match the
  outer dimension of "b" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, "a" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, "b" is transposed before multiplication.
    grad_a: An optional `bool`. Defaults to `False`.
    grad_b: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MatMul", name, a, b, "transpose_a", transpose_a, "transpose_b",
        transpose_b, "grad_a", grad_a, "grad_b", grad_b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mat_mul_eager_fallback(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b,
          grad_a=grad_a, grad_b=grad_b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if grad_a is None:
    grad_a = False
  grad_a = _execute.make_bool(grad_a, "grad_a")
  if grad_b is None:
    grad_b = False
  grad_b = _execute.make_bool(grad_b, "grad_b")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MatMul", a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b,
                  grad_a=grad_a, grad_b=grad_b, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "T", _op._get_attr_type("T"),
              "grad_a", _op._get_attr_bool("grad_a"), "grad_b",
              _op._get_attr_bool("grad_b"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MatMul = tf_export("raw_ops.MatMul")(_ops.to_raw_op(mat_mul))


def mat_mul_eager_fallback(a: Annotated[Any, TV_MatMul_T], b: Annotated[Any, TV_MatMul_T], transpose_a: bool, transpose_b: bool, grad_a: bool, grad_b: bool, name, ctx) -> Annotated[Any, TV_MatMul_T]:
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if grad_a is None:
    grad_a = False
  grad_a = _execute.make_bool(grad_a, "grad_a")
  if grad_b is None:
    grad_b = False
  grad_b = _execute.make_bool(grad_b, "grad_b")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, ])
  (a, b) = _inputs_T
  _inputs_flat = [a, b]
  _attrs = ("transpose_a", transpose_a, "transpose_b", transpose_b, "T",
  _attr_T, "grad_a", grad_a, "grad_b", grad_b)
  _result = _execute.execute(b"MatMul", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Max_T = TypeVar("TV_Max_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Max_Tidx = TypeVar("TV_Max_Tidx", _atypes.Int32, _atypes.Int64)

def _max(input: Annotated[Any, TV_Max_T], axis: Annotated[Any, TV_Max_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_Max_T]:
  r"""Computes the maximum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Max", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _max_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Max", input=input, reduction_indices=axis, keep_dims=keep_dims,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Max", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Max = tf_export("raw_ops.Max")(_ops.to_raw_op(_max))


def _max_eager_fallback(input: Annotated[Any, TV_Max_T], axis: Annotated[Any, TV_Max_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_Max_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Max", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Max", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Maximum_T = TypeVar("TV_Maximum_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.maximum', 'maximum')
def maximum(x: Annotated[Any, TV_Maximum_T], y: Annotated[Any, TV_Maximum_T], name=None) -> Annotated[Any, TV_Maximum_T]:
  r"""Returns the max of x and y (i.e. x > y ? x : y) element-wise.

  Example:

  >>> x = tf.constant([0., 0., 0., 0.])
  >>> y = tf.constant([-2., 0., 2., 5.])
  >>> tf.math.maximum(x, y)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 0., 2., 5.], dtype=float32)>

  Note that `maximum` supports [broadcast semantics](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) for `x` and `y`.

  >>> x = tf.constant([-5., 0., 0., 0.])
  >>> y = tf.constant([-3.])
  >>> tf.math.maximum(x, y)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-3., 0., 0., 0.], dtype=float32)>

  The reduction version of this elementwise operation is `tf.math.reduce_max`

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Maximum", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_maximum(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return maximum_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            maximum, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_maximum(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Maximum", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          maximum, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Maximum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Maximum = tf_export("raw_ops.Maximum")(_ops.to_raw_op(maximum))
_dispatcher_for_maximum = maximum._tf_type_based_dispatcher.Dispatch


def maximum_eager_fallback(x: Annotated[Any, TV_Maximum_T], y: Annotated[Any, TV_Maximum_T], name, ctx) -> Annotated[Any, TV_Maximum_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Maximum", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Maximum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Mean_T = TypeVar("TV_Mean_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Mean_Tidx = TypeVar("TV_Mean_Tidx", _atypes.Int32, _atypes.Int64)

def mean(input: Annotated[Any, TV_Mean_T], axis: Annotated[Any, TV_Mean_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_Mean_T]:
  r"""Computes the mean of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Mean", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mean_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Mean", input=input, reduction_indices=axis, keep_dims=keep_dims,
                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Mean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Mean = tf_export("raw_ops.Mean")(_ops.to_raw_op(mean))


def mean_eager_fallback(input: Annotated[Any, TV_Mean_T], axis: Annotated[Any, TV_Mean_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_Mean_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Mean", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Mean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Min_T = TypeVar("TV_Min_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Min_Tidx = TypeVar("TV_Min_Tidx", _atypes.Int32, _atypes.Int64)

def _min(input: Annotated[Any, TV_Min_T], axis: Annotated[Any, TV_Min_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_Min_T]:
  r"""Computes the minimum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Min", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _min_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Min", input=input, reduction_indices=axis, keep_dims=keep_dims,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Min", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Min = tf_export("raw_ops.Min")(_ops.to_raw_op(_min))


def _min_eager_fallback(input: Annotated[Any, TV_Min_T], axis: Annotated[Any, TV_Min_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_Min_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Min", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Min", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Minimum_T = TypeVar("TV_Minimum_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.minimum', 'minimum')
def minimum(x: Annotated[Any, TV_Minimum_T], y: Annotated[Any, TV_Minimum_T], name=None) -> Annotated[Any, TV_Minimum_T]:
  r"""Returns the min of x and y (i.e. x < y ? x : y) element-wise.

  Both inputs are number-type tensors (except complex).  `minimum` expects that
  both tensors have the same `dtype`.

  Examples:

  >>> x = tf.constant([0., 0., 0., 0.])
  >>> y = tf.constant([-5., -2., 0., 3.])
  >>> tf.math.minimum(x, y)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-5., -2., 0., 0.], dtype=float32)>

  Note that `minimum` supports [broadcast semantics](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) for `x` and `y`.

  >>> x = tf.constant([-5., 0., 0., 0.])
  >>> y = tf.constant([-3.])
  >>> tf.math.minimum(x, y)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-5., -3., -3., -3.], dtype=float32)>

  The reduction version of this elementwise operation is `tf.math.reduce_min`

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Minimum", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_minimum(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return minimum_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            minimum, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_minimum(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Minimum", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          minimum, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Minimum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Minimum = tf_export("raw_ops.Minimum")(_ops.to_raw_op(minimum))
_dispatcher_for_minimum = minimum._tf_type_based_dispatcher.Dispatch


def minimum_eager_fallback(x: Annotated[Any, TV_Minimum_T], y: Annotated[Any, TV_Minimum_T], name, ctx) -> Annotated[Any, TV_Minimum_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Minimum", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Minimum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Mod_T = TypeVar("TV_Mod_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Half, _atypes.Int32, _atypes.Int64)

def mod(x: Annotated[Any, TV_Mod_T], y: Annotated[Any, TV_Mod_T], name=None) -> Annotated[Any, TV_Mod_T]:
  r"""Returns element-wise remainder of division. This emulates C semantics in that

  the result here is consistent with a truncating divide. E.g.
  `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.

  *NOTE*: `Mod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `half`, `half`, `bfloat16`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Mod", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mod_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Mod", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Mod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Mod = tf_export("raw_ops.Mod")(_ops.to_raw_op(mod))


def mod_eager_fallback(x: Annotated[Any, TV_Mod_T], y: Annotated[Any, TV_Mod_T], name, ctx) -> Annotated[Any, TV_Mod_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Mod", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Mod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Mul_T = TypeVar("TV_Mul_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def mul(x: Annotated[Any, TV_Mul_T], y: Annotated[Any, TV_Mul_T], name=None) -> Annotated[Any, TV_Mul_T]:
  r"""Returns x * y element-wise.

  *NOTE*: `Multiply` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `uint64`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Mul", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mul_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Mul", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Mul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Mul = tf_export("raw_ops.Mul")(_ops.to_raw_op(mul))


def mul_eager_fallback(x: Annotated[Any, TV_Mul_T], y: Annotated[Any, TV_Mul_T], name, ctx) -> Annotated[Any, TV_Mul_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.uint64, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Mul", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Mul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MulNoNan_T = TypeVar("TV_MulNoNan_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def mul_no_nan(x: Annotated[Any, TV_MulNoNan_T], y: Annotated[Any, TV_MulNoNan_T], name=None) -> Annotated[Any, TV_MulNoNan_T]:
  r"""Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.

  *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MulNoNan", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mul_no_nan_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MulNoNan", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MulNoNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MulNoNan = tf_export("raw_ops.MulNoNan")(_ops.to_raw_op(mul_no_nan))


def mul_no_nan_eager_fallback(x: Annotated[Any, TV_MulNoNan_T], y: Annotated[Any, TV_MulNoNan_T], name, ctx) -> Annotated[Any, TV_MulNoNan_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"MulNoNan", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MulNoNan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Ndtri_T = TypeVar("TV_Ndtri_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def ndtri(x: Annotated[Any, TV_Ndtri_T], name=None) -> Annotated[Any, TV_Ndtri_T]:
  r"""TODO: add doc.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Ndtri", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ndtri_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Ndtri", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Ndtri", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Ndtri = tf_export("raw_ops.Ndtri")(_ops.to_raw_op(ndtri))


def ndtri_eager_fallback(x: Annotated[Any, TV_Ndtri_T], name, ctx) -> Annotated[Any, TV_Ndtri_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Ndtri", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Ndtri", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Neg_T = TypeVar("TV_Neg_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.negative', 'negative')
def neg(x: Annotated[Any, TV_Neg_T], name=None) -> Annotated[Any, TV_Neg_T]:
  r"""Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Neg", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_neg(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return neg_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            neg, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_neg(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Neg", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          neg, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Neg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Neg = tf_export("raw_ops.Neg")(_ops.to_raw_op(neg))
_dispatcher_for_neg = neg._tf_type_based_dispatcher.Dispatch


def neg_eager_fallback(x: Annotated[Any, TV_Neg_T], name, ctx) -> Annotated[Any, TV_Neg_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Neg", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Neg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NextAfter_T = TypeVar("TV_NextAfter_T", _atypes.Float32, _atypes.Float64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.nextafter')
def next_after(x1: Annotated[Any, TV_NextAfter_T], x2: Annotated[Any, TV_NextAfter_T], name=None) -> Annotated[Any, TV_NextAfter_T]:
  r"""Returns the next representable value of `x1` in the direction of `x2`, element-wise.

  This operation returns the same result as the C++ std::nextafter function.

  It can also return a subnormal number.

  @compatibility(cpp)
  Equivalent to C++ std::nextafter function.
  @end_compatibility

  Args:
    x1: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    x2: A `Tensor`. Must have the same type as `x1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x1`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NextAfter", name, x1, x2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_next_after(
          (x1, x2, name,), None)
      if _result is not NotImplemented:
        return _result
      return next_after_eager_fallback(
          x1, x2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            next_after, (), dict(x1=x1, x2=x2, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_next_after(
        (x1, x2, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NextAfter", x1=x1, x2=x2, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          next_after, (), dict(x1=x1, x2=x2, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NextAfter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NextAfter = tf_export("raw_ops.NextAfter")(_ops.to_raw_op(next_after))
_dispatcher_for_next_after = next_after._tf_type_based_dispatcher.Dispatch


def next_after_eager_fallback(x1: Annotated[Any, TV_NextAfter_T], x2: Annotated[Any, TV_NextAfter_T], name, ctx) -> Annotated[Any, TV_NextAfter_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x1, x2], ctx, [_dtypes.float64, _dtypes.float32, ], _dtypes.float32)
  (x1, x2) = _inputs_T
  _inputs_flat = [x1, x2]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"NextAfter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NextAfter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NotEqual_T = TypeVar("TV_NotEqual_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def not_equal(x: Annotated[Any, TV_NotEqual_T], y: Annotated[Any, TV_NotEqual_T], incompatible_shape_error:bool=True, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns the truth value of (x != y) element-wise.

  *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NotEqual", name, x, y, "incompatible_shape_error",
        incompatible_shape_error)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return not_equal_eager_fallback(
          x, y, incompatible_shape_error=incompatible_shape_error, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if incompatible_shape_error is None:
    incompatible_shape_error = True
  incompatible_shape_error = _execute.make_bool(incompatible_shape_error, "incompatible_shape_error")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NotEqual", x=x, y=y,
                    incompatible_shape_error=incompatible_shape_error,
                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "incompatible_shape_error",
              _op._get_attr_bool("incompatible_shape_error"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NotEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NotEqual = tf_export("raw_ops.NotEqual")(_ops.to_raw_op(not_equal))


def not_equal_eager_fallback(x: Annotated[Any, TV_NotEqual_T], y: Annotated[Any, TV_NotEqual_T], incompatible_shape_error: bool, name, ctx) -> Annotated[Any, _atypes.Bool]:
  if incompatible_shape_error is None:
    incompatible_shape_error = True
  incompatible_shape_error = _execute.make_bool(incompatible_shape_error, "incompatible_shape_error")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T, "incompatible_shape_error",
  incompatible_shape_error)
  _result = _execute.execute(b"NotEqual", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NotEqual", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Polygamma_T = TypeVar("TV_Polygamma_T", _atypes.Float32, _atypes.Float64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.polygamma', v1=['math.polygamma', 'polygamma'])
@deprecated_endpoints('polygamma')
def polygamma(a: Annotated[Any, TV_Polygamma_T], x: Annotated[Any, TV_Polygamma_T], name=None) -> Annotated[Any, TV_Polygamma_T]:
  r"""Compute the polygamma function \\(\psi^{(n)}(x)\\).

  The polygamma function is defined as:


  \\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)

  where \\(\psi(x)\\) is the digamma function.
  The polygamma function is defined only for non-negative integer orders \\a\\.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Polygamma", name, a, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_polygamma(
          (a, x, name,), None)
      if _result is not NotImplemented:
        return _result
      return polygamma_eager_fallback(
          a, x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            polygamma, (), dict(a=a, x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_polygamma(
        (a, x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Polygamma", a=a, x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          polygamma, (), dict(a=a, x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Polygamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Polygamma = tf_export("raw_ops.Polygamma")(_ops.to_raw_op(polygamma))
_dispatcher_for_polygamma = polygamma._tf_type_based_dispatcher.Dispatch


def polygamma_eager_fallback(a: Annotated[Any, TV_Polygamma_T], x: Annotated[Any, TV_Polygamma_T], name, ctx) -> Annotated[Any, TV_Polygamma_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, x], ctx, [_dtypes.float32, _dtypes.float64, ])
  (a, x) = _inputs_T
  _inputs_flat = [a, x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Polygamma", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Polygamma", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Pow_T = TypeVar("TV_Pow_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

def _pow(x: Annotated[Any, TV_Pow_T], y: Annotated[Any, TV_Pow_T], name=None) -> Annotated[Any, TV_Pow_T]:
  r"""Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Pow", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _pow_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Pow", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Pow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Pow = tf_export("raw_ops.Pow")(_ops.to_raw_op(_pow))


def _pow_eager_fallback(x: Annotated[Any, TV_Pow_T], y: Annotated[Any, TV_Pow_T], name, ctx) -> Annotated[Any, TV_Pow_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Pow", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Pow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Prod_T = TypeVar("TV_Prod_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Prod_Tidx = TypeVar("TV_Prod_Tidx", _atypes.Int32, _atypes.Int64)

def prod(input: Annotated[Any, TV_Prod_T], axis: Annotated[Any, TV_Prod_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_Prod_T]:
  r"""Computes the product of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Prod", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return prod_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Prod", input=input, reduction_indices=axis, keep_dims=keep_dims,
                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Prod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Prod = tf_export("raw_ops.Prod")(_ops.to_raw_op(prod))


def prod_eager_fallback(input: Annotated[Any, TV_Prod_T], axis: Annotated[Any, TV_Prod_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_Prod_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Prod", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Prod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_QuantizeDownAndShrinkRangeOutput = collections.namedtuple(
    "QuantizeDownAndShrinkRange",
    ["output", "output_min", "output_max"])


TV_QuantizeDownAndShrinkRange_Tinput = TypeVar("TV_QuantizeDownAndShrinkRange_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizeDownAndShrinkRange_out_type = TypeVar("TV_QuantizeDownAndShrinkRange_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantize_down_and_shrink_range(input: Annotated[Any, TV_QuantizeDownAndShrinkRange_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], out_type: TV_QuantizeDownAndShrinkRange_out_type, name=None):
  r"""Convert the quantized 'input' tensor into a lower-precision 'output', using the

  actual distribution of the values to maximize the usage of the lower bit depth
  and adjusting the output min and max ranges accordingly.

  [input_min, input_max] are scalar floats that specify the range for the float
  interpretation of the 'input' data. For example, if input_min is -1.0f and
  input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  This operator tries to squeeze as much precision as possible into an output with
  a lower bit depth by calculating the actual min and max values found in the
  data. For example, maybe that quint16 input has no values lower than 16,384 and
  none higher than 49,152. That means only half the range is actually needed, all
  the float interpretations are between -0.5f and 0.5f, so if we want to compress
  the data into a quint8 output, we can use that range rather than the theoretical
  -1.0f to 1.0f that is suggested by the input min and max.

  In practice, this is most useful for taking output from operations like
  QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
  may have large potential output ranges, but in practice have a distribution of
  input values that only uses a small fraction of the possible range. By feeding
  that output into this operator, we can reduce it from 32 bits down to 8 with
  minimal loss of accuracy.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizeDownAndShrinkRange", name, input, input_min, input_max,
        "out_type", out_type)
      _result = _QuantizeDownAndShrinkRangeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantize_down_and_shrink_range_eager_fallback(
          input, input_min, input_max, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizeDownAndShrinkRange", input=input, input_min=input_min,
                                      input_max=input_max, out_type=out_type,
                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizeDownAndShrinkRange", _inputs_flat, _attrs, _result)
  _result = _QuantizeDownAndShrinkRangeOutput._make(_result)
  return _result

QuantizeDownAndShrinkRange = tf_export("raw_ops.QuantizeDownAndShrinkRange")(_ops.to_raw_op(quantize_down_and_shrink_range))


def quantize_down_and_shrink_range_eager_fallback(input: Annotated[Any, TV_QuantizeDownAndShrinkRange_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], out_type: TV_QuantizeDownAndShrinkRange_out_type, name, ctx):
  out_type = _execute.make_type(out_type, "out_type")
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  _inputs_flat = [input, input_min, input_max]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type)
  _result = _execute.execute(b"QuantizeDownAndShrinkRange", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizeDownAndShrinkRange", _inputs_flat, _attrs, _result)
  _result = _QuantizeDownAndShrinkRangeOutput._make(_result)
  return _result

_QuantizedAddOutput = collections.namedtuple(
    "QuantizedAdd",
    ["z", "min_z", "max_z"])


TV_QuantizedAdd_T1 = TypeVar("TV_QuantizedAdd_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedAdd_T2 = TypeVar("TV_QuantizedAdd_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedAdd_Toutput = TypeVar("TV_QuantizedAdd_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_add(x: Annotated[Any, TV_QuantizedAdd_T1], y: Annotated[Any, TV_QuantizedAdd_T2], min_x: Annotated[Any, _atypes.Float32], max_x: Annotated[Any, _atypes.Float32], min_y: Annotated[Any, _atypes.Float32], max_y: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedAdd_Toutput=_dtypes.qint32, name=None):
  r"""Returns x + y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedAdd", name, x, y, min_x, max_x, min_y, max_y,
        "Toutput", Toutput)
      _result = _QuantizedAddOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_add_eager_fallback(
          x, y, min_x, max_x, min_y, max_y, Toutput=Toutput, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedAdd", x=x, y=y, min_x=min_x, max_x=max_x, min_y=min_y,
                        max_y=max_y, Toutput=Toutput, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Toutput", _op._get_attr_type("Toutput"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedAdd", _inputs_flat, _attrs, _result)
  _result = _QuantizedAddOutput._make(_result)
  return _result

QuantizedAdd = tf_export("raw_ops.QuantizedAdd")(_ops.to_raw_op(quantized_add))


def quantized_add_eager_fallback(x: Annotated[Any, TV_QuantizedAdd_T1], y: Annotated[Any, TV_QuantizedAdd_T2], min_x: Annotated[Any, _atypes.Float32], max_x: Annotated[Any, _atypes.Float32], min_y: Annotated[Any, _atypes.Float32], max_y: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedAdd_Toutput, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  _attr_T1, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (y,) = _execute.args_to_matching_eager([y], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_x = _ops.convert_to_tensor(min_x, _dtypes.float32)
  max_x = _ops.convert_to_tensor(max_x, _dtypes.float32)
  min_y = _ops.convert_to_tensor(min_y, _dtypes.float32)
  max_y = _ops.convert_to_tensor(max_y, _dtypes.float32)
  _inputs_flat = [x, y, min_x, max_x, min_y, max_y]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Toutput", Toutput)
  _result = _execute.execute(b"QuantizedAdd", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedAdd", _inputs_flat, _attrs, _result)
  _result = _QuantizedAddOutput._make(_result)
  return _result

_QuantizedMatMulOutput = collections.namedtuple(
    "QuantizedMatMul",
    ["out", "min_out", "max_out"])


TV_QuantizedMatMul_T1 = TypeVar("TV_QuantizedMatMul_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMul_T2 = TypeVar("TV_QuantizedMatMul_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMul_Toutput = TypeVar("TV_QuantizedMatMul_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMul_Tactivation = TypeVar("TV_QuantizedMatMul_Tactivation", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mat_mul(a: Annotated[Any, TV_QuantizedMatMul_T1], b: Annotated[Any, TV_QuantizedMatMul_T2], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMatMul_Toutput=_dtypes.qint32, transpose_a:bool=False, transpose_b:bool=False, Tactivation:TV_QuantizedMatMul_Tactivation=_dtypes.quint8, name=None):
  r"""Perform a quantized matrix multiplication of  `a` by the matrix `b`.

  The inputs must be two-dimensional matrices and the inner dimension of
  `a` (after being transposed if `transpose_a` is non-zero) must match the
  outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero).

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    Tactivation: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of output produced by activation function
      following this operation.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedMatMul", name, a, b, min_a, max_a, min_b, max_b,
        "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b",
        transpose_b, "Tactivation", Tactivation)
      _result = _QuantizedMatMulOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_eager_fallback(
          a, b, min_a, max_a, min_b, max_b, Toutput=Toutput,
          transpose_a=transpose_a, transpose_b=transpose_b,
          Tactivation=Tactivation, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if Tactivation is None:
    Tactivation = _dtypes.quint8
  Tactivation = _execute.make_type(Tactivation, "Tactivation")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMul", a=a, b=b, min_a=min_a, max_a=max_a, min_b=min_b,
                           max_b=max_b, Toutput=Toutput,
                           transpose_a=transpose_a, transpose_b=transpose_b,
                           Tactivation=Tactivation, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Toutput", _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "Tactivation",
              _op._get_attr_type("Tactivation"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMul", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulOutput._make(_result)
  return _result

QuantizedMatMul = tf_export("raw_ops.QuantizedMatMul")(_ops.to_raw_op(quantized_mat_mul))


def quantized_mat_mul_eager_fallback(a: Annotated[Any, TV_QuantizedMatMul_T1], b: Annotated[Any, TV_QuantizedMatMul_T2], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMul_Toutput, transpose_a: bool, transpose_b: bool, Tactivation: TV_QuantizedMatMul_Tactivation, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if Tactivation is None:
    Tactivation = _dtypes.quint8
  Tactivation = _execute.make_type(Tactivation, "Tactivation")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  _inputs_flat = [a, b, min_a, max_a, min_b, max_b]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Toutput", Toutput, "transpose_a",
  transpose_a, "transpose_b", transpose_b, "Tactivation", Tactivation)
  _result = _execute.execute(b"QuantizedMatMul", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMul", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulOutput._make(_result)
  return _result

_QuantizedMulOutput = collections.namedtuple(
    "QuantizedMul",
    ["z", "min_z", "max_z"])


TV_QuantizedMul_T1 = TypeVar("TV_QuantizedMul_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMul_T2 = TypeVar("TV_QuantizedMul_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMul_Toutput = TypeVar("TV_QuantizedMul_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mul(x: Annotated[Any, TV_QuantizedMul_T1], y: Annotated[Any, TV_QuantizedMul_T2], min_x: Annotated[Any, _atypes.Float32], max_x: Annotated[Any, _atypes.Float32], min_y: Annotated[Any, _atypes.Float32], max_y: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMul_Toutput=_dtypes.qint32, name=None):
  r"""Returns x * y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedMul", name, x, y, min_x, max_x, min_y, max_y,
        "Toutput", Toutput)
      _result = _QuantizedMulOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mul_eager_fallback(
          x, y, min_x, max_x, min_y, max_y, Toutput=Toutput, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMul", x=x, y=y, min_x=min_x, max_x=max_x, min_y=min_y,
                        max_y=max_y, Toutput=Toutput, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Toutput", _op._get_attr_type("Toutput"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMul", _inputs_flat, _attrs, _result)
  _result = _QuantizedMulOutput._make(_result)
  return _result

QuantizedMul = tf_export("raw_ops.QuantizedMul")(_ops.to_raw_op(quantized_mul))


def quantized_mul_eager_fallback(x: Annotated[Any, TV_QuantizedMul_T1], y: Annotated[Any, TV_QuantizedMul_T2], min_x: Annotated[Any, _atypes.Float32], max_x: Annotated[Any, _atypes.Float32], min_y: Annotated[Any, _atypes.Float32], max_y: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMul_Toutput, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  _attr_T1, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (y,) = _execute.args_to_matching_eager([y], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_x = _ops.convert_to_tensor(min_x, _dtypes.float32)
  max_x = _ops.convert_to_tensor(max_x, _dtypes.float32)
  min_y = _ops.convert_to_tensor(min_y, _dtypes.float32)
  max_y = _ops.convert_to_tensor(max_y, _dtypes.float32)
  _inputs_flat = [x, y, min_x, max_x, min_y, max_y]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Toutput", Toutput)
  _result = _execute.execute(b"QuantizedMul", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMul", _inputs_flat, _attrs, _result)
  _result = _QuantizedMulOutput._make(_result)
  return _result


TV_RaggedBincount_Tidx = TypeVar("TV_RaggedBincount_Tidx", _atypes.Int32, _atypes.Int64)
TV_RaggedBincount_T = TypeVar("TV_RaggedBincount_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def ragged_bincount(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedBincount_Tidx], size: Annotated[Any, TV_RaggedBincount_Tidx], weights: Annotated[Any, TV_RaggedBincount_T], binary_output:bool=False, name=None) -> Annotated[Any, TV_RaggedBincount_T]:
  r"""Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    splits: A `Tensor` of type `int64`. 1D int64 `Tensor`.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2D int `Tensor`.
    size: A `Tensor`. Must have the same type as `values`.
      non-negative int scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `input`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    binary_output: An optional `bool`. Defaults to `False`.
      bool; Whether the kernel should count the appearance or number of occurrences.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RaggedBincount", name, splits, values, size, weights,
        "binary_output", binary_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return ragged_bincount_eager_fallback(
          splits, values, size, weights, binary_output=binary_output,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RaggedBincount", splits=splits, values=values, size=size,
                          weights=weights, binary_output=binary_output,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"), "T",
              _op._get_attr_type("T"), "binary_output",
              _op._get_attr_bool("binary_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RaggedBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RaggedBincount = tf_export("raw_ops.RaggedBincount")(_ops.to_raw_op(ragged_bincount))


def ragged_bincount_eager_fallback(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedBincount_Tidx], size: Annotated[Any, TV_RaggedBincount_Tidx], weights: Annotated[Any, TV_RaggedBincount_T], binary_output: bool, name, ctx) -> Annotated[Any, TV_RaggedBincount_T]:
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([values, size], ctx, [_dtypes.int32, _dtypes.int64, ])
  (values, size) = _inputs_Tidx
  _attr_T, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  splits = _ops.convert_to_tensor(splits, _dtypes.int64)
  _inputs_flat = [splits, values, size, weights]
  _attrs = ("Tidx", _attr_Tidx, "T", _attr_T, "binary_output", binary_output)
  _result = _execute.execute(b"RaggedBincount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RaggedBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Range_Tidx = TypeVar("TV_Range_Tidx", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32)

def _range(start: Annotated[Any, TV_Range_Tidx], limit: Annotated[Any, TV_Range_Tidx], delta: Annotated[Any, TV_Range_Tidx], name=None) -> Annotated[Any, TV_Range_Tidx]:
  r"""Creates a sequence of numbers.

  This operation creates a sequence of numbers that begins at `start` and
  extends by increments of `delta` up to but not including `limit`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint16`, `uint32`.
      0-D (scalar). First entry in the sequence.
    limit: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Upper limit of sequence, exclusive.
    delta: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Optional. Default is 1. Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Range", name, start, limit, delta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _range_eager_fallback(
          start, limit, delta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Range", start=start, limit=limit, delta=delta, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Range", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Range = tf_export("raw_ops.Range")(_ops.to_raw_op(_range))


def _range_eager_fallback(start: Annotated[Any, TV_Range_Tidx], limit: Annotated[Any, TV_Range_Tidx], delta: Annotated[Any, TV_Range_Tidx], name, ctx) -> Annotated[Any, TV_Range_Tidx]:
  _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([start, limit, delta], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint16, _dtypes.uint32, ], _dtypes.int32)
  (start, limit, delta) = _inputs_Tidx
  _inputs_flat = [start, limit, delta]
  _attrs = ("Tidx", _attr_Tidx)
  _result = _execute.execute(b"Range", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Range", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Real_T = TypeVar("TV_Real_T", _atypes.Complex128, _atypes.Complex64)
TV_Real_Tout = TypeVar("TV_Real_Tout", _atypes.Float32, _atypes.Float64)

def real(input: Annotated[Any, TV_Real_T], Tout:TV_Real_Tout=_dtypes.float32, name=None) -> Annotated[Any, TV_Real_Tout]:
  r"""Returns the real part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the real part of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
   part returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(input) ==> [-2.25, 3.25]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Real", name, input, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return real_eager_fallback(
          input, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Real", input=input, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tout",
              _op._get_attr_type("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Real", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Real = tf_export("raw_ops.Real")(_ops.to_raw_op(real))


def real_eager_fallback(input: Annotated[Any, TV_Real_T], Tout: TV_Real_Tout, name, ctx) -> Annotated[Any, TV_Real_Tout]:
  if Tout is None:
    Tout = _dtypes.float32
  Tout = _execute.make_type(Tout, "Tout")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "Tout", Tout)
  _result = _execute.execute(b"Real", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Real", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RealDiv_T = TypeVar("TV_RealDiv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('realdiv')
def real_div(x: Annotated[Any, TV_RealDiv_T], y: Annotated[Any, TV_RealDiv_T], name=None) -> Annotated[Any, TV_RealDiv_T]:
  r"""Returns x / y element-wise for real types.

  If `x` and `y` are reals, this will return the floating-point division.

  *NOTE*: `Div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `uint64`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RealDiv", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_real_div(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return real_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            real_div, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_real_div(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RealDiv", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          real_div, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RealDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RealDiv = tf_export("raw_ops.RealDiv")(_ops.to_raw_op(real_div))
_dispatcher_for_real_div = real_div._tf_type_based_dispatcher.Dispatch


def real_div_eager_fallback(x: Annotated[Any, TV_RealDiv_T], y: Annotated[Any, TV_RealDiv_T], name, ctx) -> Annotated[Any, TV_RealDiv_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.uint64, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RealDiv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RealDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Reciprocal_T = TypeVar("TV_Reciprocal_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.reciprocal', v1=['math.reciprocal', 'reciprocal'])
@deprecated_endpoints('reciprocal')
def reciprocal(x: Annotated[Any, TV_Reciprocal_T], name=None) -> Annotated[Any, TV_Reciprocal_T]:
  r"""Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Reciprocal", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_reciprocal(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return reciprocal_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            reciprocal, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_reciprocal(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Reciprocal", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          reciprocal, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Reciprocal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Reciprocal = tf_export("raw_ops.Reciprocal")(_ops.to_raw_op(reciprocal))
_dispatcher_for_reciprocal = reciprocal._tf_type_based_dispatcher.Dispatch


def reciprocal_eager_fallback(x: Annotated[Any, TV_Reciprocal_T], name, ctx) -> Annotated[Any, TV_Reciprocal_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Reciprocal", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Reciprocal", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ReciprocalGrad_T = TypeVar("TV_ReciprocalGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def reciprocal_grad(y: Annotated[Any, TV_ReciprocalGrad_T], dy: Annotated[Any, TV_ReciprocalGrad_T], name=None) -> Annotated[Any, TV_ReciprocalGrad_T]:
  r"""Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReciprocalGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reciprocal_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReciprocalGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReciprocalGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReciprocalGrad = tf_export("raw_ops.ReciprocalGrad")(_ops.to_raw_op(reciprocal_grad))


def reciprocal_grad_eager_fallback(y: Annotated[Any, TV_ReciprocalGrad_T], dy: Annotated[Any, TV_ReciprocalGrad_T], name, ctx) -> Annotated[Any, TV_ReciprocalGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ReciprocalGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReciprocalGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_RequantizationRangeOutput = collections.namedtuple(
    "RequantizationRange",
    ["output_min", "output_max"])


TV_RequantizationRange_Tinput = TypeVar("TV_RequantizationRange_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def requantization_range(input: Annotated[Any, TV_RequantizationRange_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], name=None):
  r"""Computes a range that covers the actual values present in a quantized tensor.

  Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
  range that covers the actual values present in that tensor. This op is typically
  used to produce the `requested_output_min` and `requested_output_max` for
  `Requantize`.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RequantizationRange", name, input, input_min, input_max)
      _result = _RequantizationRangeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return requantization_range_eager_fallback(
          input, input_min, input_max, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RequantizationRange", input=input, input_min=input_min,
                               input_max=input_max, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RequantizationRange", _inputs_flat, _attrs, _result)
  _result = _RequantizationRangeOutput._make(_result)
  return _result

RequantizationRange = tf_export("raw_ops.RequantizationRange")(_ops.to_raw_op(requantization_range))


def requantization_range_eager_fallback(input: Annotated[Any, TV_RequantizationRange_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], name, ctx):
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  _inputs_flat = [input, input_min, input_max]
  _attrs = ("Tinput", _attr_Tinput)
  _result = _execute.execute(b"RequantizationRange", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RequantizationRange", _inputs_flat, _attrs, _result)
  _result = _RequantizationRangeOutput._make(_result)
  return _result

_RequantizationRangePerChannelOutput = collections.namedtuple(
    "RequantizationRangePerChannel",
    ["output_min", "output_max"])


TV_RequantizationRangePerChannel_T = TypeVar("TV_RequantizationRangePerChannel_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def requantization_range_per_channel(input: Annotated[Any, TV_RequantizationRangePerChannel_T], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], clip_value_max: float, name=None):
  r"""Computes requantization range per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    clip_value_max: A `float`.
      The maximum value of the output that needs to be clipped.
      Example: set this to 6 for Relu6.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RequantizationRangePerChannel", name, input, input_min,
        input_max, "clip_value_max", clip_value_max)
      _result = _RequantizationRangePerChannelOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return requantization_range_per_channel_eager_fallback(
          input, input_min, input_max, clip_value_max=clip_value_max,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  clip_value_max = _execute.make_float(clip_value_max, "clip_value_max")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RequantizationRangePerChannel", input=input, input_min=input_min,
                                         input_max=input_max,
                                         clip_value_max=clip_value_max,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "clip_value_max",
              _op.get_attr("clip_value_max"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RequantizationRangePerChannel", _inputs_flat, _attrs, _result)
  _result = _RequantizationRangePerChannelOutput._make(_result)
  return _result

RequantizationRangePerChannel = tf_export("raw_ops.RequantizationRangePerChannel")(_ops.to_raw_op(requantization_range_per_channel))


def requantization_range_per_channel_eager_fallback(input: Annotated[Any, TV_RequantizationRangePerChannel_T], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], clip_value_max: float, name, ctx):
  clip_value_max = _execute.make_float(clip_value_max, "clip_value_max")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ], _dtypes.qint32)
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  _inputs_flat = [input, input_min, input_max]
  _attrs = ("T", _attr_T, "clip_value_max", clip_value_max)
  _result = _execute.execute(b"RequantizationRangePerChannel", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RequantizationRangePerChannel", _inputs_flat, _attrs, _result)
  _result = _RequantizationRangePerChannelOutput._make(_result)
  return _result

_RequantizeOutput = collections.namedtuple(
    "Requantize",
    ["output", "output_min", "output_max"])


TV_Requantize_Tinput = TypeVar("TV_Requantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_Requantize_out_type = TypeVar("TV_Requantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def requantize(input: Annotated[Any, TV_Requantize_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], requested_output_min: Annotated[Any, _atypes.Float32], requested_output_max: Annotated[Any, _atypes.Float32], out_type: TV_Requantize_out_type, name=None):
  r"""Converts the quantized `input` tensor into a lower-precision `output`.

  Converts the quantized `input` tensor into a lower-precision `output`, using the
  output range specified with `requested_output_min` and `requested_output_max`.

  `[input_min, input_max]` are scalar floats that specify the range for the float
  interpretation of the `input` data. For example, if `input_min` is -1.0f and
  `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    requested_output_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized output value represents.
    requested_output_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized output value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Requantize", name, input, input_min, input_max,
        requested_output_min, requested_output_max, "out_type", out_type)
      _result = _RequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return requantize_eager_fallback(
          input, input_min, input_max, requested_output_min,
          requested_output_max, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Requantize", input=input, input_min=input_min, input_max=input_max,
                      requested_output_min=requested_output_min,
                      requested_output_max=requested_output_max,
                      out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Requantize", _inputs_flat, _attrs, _result)
  _result = _RequantizeOutput._make(_result)
  return _result

Requantize = tf_export("raw_ops.Requantize")(_ops.to_raw_op(requantize))


def requantize_eager_fallback(input: Annotated[Any, TV_Requantize_Tinput], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], requested_output_min: Annotated[Any, _atypes.Float32], requested_output_max: Annotated[Any, _atypes.Float32], out_type: TV_Requantize_out_type, name, ctx):
  out_type = _execute.make_type(out_type, "out_type")
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  requested_output_min = _ops.convert_to_tensor(requested_output_min, _dtypes.float32)
  requested_output_max = _ops.convert_to_tensor(requested_output_max, _dtypes.float32)
  _inputs_flat = [input, input_min, input_max, requested_output_min, requested_output_max]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type)
  _result = _execute.execute(b"Requantize", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Requantize", _inputs_flat, _attrs, _result)
  _result = _RequantizeOutput._make(_result)
  return _result

_RequantizePerChannelOutput = collections.namedtuple(
    "RequantizePerChannel",
    ["output", "output_min", "output_max"])


TV_RequantizePerChannel_T = TypeVar("TV_RequantizePerChannel_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_RequantizePerChannel_out_type = TypeVar("TV_RequantizePerChannel_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def requantize_per_channel(input: Annotated[Any, TV_RequantizePerChannel_T], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], requested_output_min: Annotated[Any, _atypes.Float32], requested_output_max: Annotated[Any, _atypes.Float32], out_type:TV_RequantizePerChannel_out_type=_dtypes.quint8, name=None):
  r"""Requantizes input with min and max values known per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    requested_output_min: A `Tensor` of type `float32`.
      The minimum value of the output tensor requested.
    requested_output_max: A `Tensor` of type `float32`.
      The maximum value of the output tensor requested.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The quantized type of output tensor that needs to be converted.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RequantizePerChannel", name, input, input_min, input_max,
        requested_output_min, requested_output_max, "out_type", out_type)
      _result = _RequantizePerChannelOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return requantize_per_channel_eager_fallback(
          input, input_min, input_max, requested_output_min,
          requested_output_max, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RequantizePerChannel", input=input, input_min=input_min,
                                input_max=input_max,
                                requested_output_min=requested_output_min,
                                requested_output_max=requested_output_max,
                                out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RequantizePerChannel", _inputs_flat, _attrs, _result)
  _result = _RequantizePerChannelOutput._make(_result)
  return _result

RequantizePerChannel = tf_export("raw_ops.RequantizePerChannel")(_ops.to_raw_op(requantize_per_channel))


def requantize_per_channel_eager_fallback(input: Annotated[Any, TV_RequantizePerChannel_T], input_min: Annotated[Any, _atypes.Float32], input_max: Annotated[Any, _atypes.Float32], requested_output_min: Annotated[Any, _atypes.Float32], requested_output_max: Annotated[Any, _atypes.Float32], out_type: TV_RequantizePerChannel_out_type, name, ctx):
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ], _dtypes.qint32)
  input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
  input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
  requested_output_min = _ops.convert_to_tensor(requested_output_min, _dtypes.float32)
  requested_output_max = _ops.convert_to_tensor(requested_output_max, _dtypes.float32)
  _inputs_flat = [input, input_min, input_max, requested_output_min, requested_output_max]
  _attrs = ("T", _attr_T, "out_type", out_type)
  _result = _execute.execute(b"RequantizePerChannel", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RequantizePerChannel", _inputs_flat, _attrs, _result)
  _result = _RequantizePerChannelOutput._make(_result)
  return _result


TV_Rint_T = TypeVar("TV_Rint_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.rint', v1=['math.rint', 'rint'])
@deprecated_endpoints('rint')
def rint(x: Annotated[Any, TV_Rint_T], name=None) -> Annotated[Any, TV_Rint_T]:
  r"""Returns element-wise integer closest to x.

  If the result is midway between two representable values,
  the even representable is chosen.
  For example:

  ```
  rint(-1.5) ==> -2.0
  rint(0.5000001) ==> 1.0
  rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Rint", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rint(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return rint_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rint, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rint(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Rint", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rint, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Rint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Rint = tf_export("raw_ops.Rint")(_ops.to_raw_op(rint))
_dispatcher_for_rint = rint._tf_type_based_dispatcher.Dispatch


def rint_eager_fallback(x: Annotated[Any, TV_Rint_T], name, ctx) -> Annotated[Any, TV_Rint_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Rint", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Rint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Round_T = TypeVar("TV_Round_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

def round(x: Annotated[Any, TV_Round_T], name=None) -> Annotated[Any, TV_Round_T]:
  r"""Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use std::cint.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Round", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return round_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Round", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Round", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Round = tf_export("raw_ops.Round")(_ops.to_raw_op(round))


def round_eager_fallback(x: Annotated[Any, TV_Round_T], name, ctx) -> Annotated[Any, TV_Round_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Round", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Round", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Rsqrt_T = TypeVar("TV_Rsqrt_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def rsqrt(x: Annotated[Any, TV_Rsqrt_T], name=None) -> Annotated[Any, TV_Rsqrt_T]:
  r"""Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Rsqrt", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rsqrt_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Rsqrt", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Rsqrt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Rsqrt = tf_export("raw_ops.Rsqrt")(_ops.to_raw_op(rsqrt))


def rsqrt_eager_fallback(x: Annotated[Any, TV_Rsqrt_T], name, ctx) -> Annotated[Any, TV_Rsqrt_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Rsqrt", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Rsqrt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RsqrtGrad_T = TypeVar("TV_RsqrtGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def rsqrt_grad(y: Annotated[Any, TV_RsqrtGrad_T], dy: Annotated[Any, TV_RsqrtGrad_T], name=None) -> Annotated[Any, TV_RsqrtGrad_T]:
  r"""Computes the gradient for the rsqrt of `x` wrt its input.

  Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RsqrtGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rsqrt_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RsqrtGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RsqrtGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RsqrtGrad = tf_export("raw_ops.RsqrtGrad")(_ops.to_raw_op(rsqrt_grad))


def rsqrt_grad_eager_fallback(y: Annotated[Any, TV_RsqrtGrad_T], dy: Annotated[Any, TV_RsqrtGrad_T], name, ctx) -> Annotated[Any, TV_RsqrtGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RsqrtGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RsqrtGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentMax_T = TypeVar("TV_SegmentMax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentMax_Tindices = TypeVar("TV_SegmentMax_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.segment_max', v1=['math.segment_max', 'segment_max'])
@deprecated_endpoints('segment_max')
def segment_max(data: Annotated[Any, TV_SegmentMax_T], segment_ids: Annotated[Any, TV_SegmentMax_Tindices], name=None) -> Annotated[Any, TV_SegmentMax_T]:
  r"""Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  If the max is empty for a given segment ID `i`, `output[i] = 0`.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> tf.math.segment_max(c, tf.constant([0, 0, 1])).numpy()
  array([[4, 3, 3, 4],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentMax", name, data, segment_ids)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_segment_max(
          (data, segment_ids, name,), None)
      if _result is not NotImplemented:
        return _result
      return segment_max_eager_fallback(
          data, segment_ids, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            segment_max, (), dict(data=data, segment_ids=segment_ids,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_segment_max(
        (data, segment_ids, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentMax", data=data, segment_ids=segment_ids, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          segment_max, (), dict(data=data, segment_ids=segment_ids, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentMax = tf_export("raw_ops.SegmentMax")(_ops.to_raw_op(segment_max))
_dispatcher_for_segment_max = segment_max._tf_type_based_dispatcher.Dispatch


def segment_max_eager_fallback(data: Annotated[Any, TV_SegmentMax_T], segment_ids: Annotated[Any, TV_SegmentMax_Tindices], name, ctx) -> Annotated[Any, TV_SegmentMax_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [data, segment_ids]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SegmentMax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentMaxV2_T = TypeVar("TV_SegmentMaxV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentMaxV2_Tindices = TypeVar("TV_SegmentMaxV2_Tindices", _atypes.Int32, _atypes.Int64)
TV_SegmentMaxV2_Tnumsegments = TypeVar("TV_SegmentMaxV2_Tnumsegments", _atypes.Int32, _atypes.Int64)

def segment_max_v2(data: Annotated[Any, TV_SegmentMaxV2_T], segment_ids: Annotated[Any, TV_SegmentMaxV2_Tindices], num_segments: Annotated[Any, TV_SegmentMaxV2_Tnumsegments], name=None) -> Annotated[Any, TV_SegmentMaxV2_T]:
  r"""Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  Note: That this op is currently only supported with jit_compile=True.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  The only difference with SegmentMax is the additional input  `num_segments`.
  This helps in evaluating the output shape in compile time.
  `num_segments` should be consistent with segment_ids.
  e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
  With inconsistent num_segments, the op still runs. only difference is,
  the output takes the size of num_segments irrespective of size of segment_ids and data.
  for num_segments less than expected output size, the last elements are ignored
  for num_segments more than the expected output size, last elements are assigned 
  smallest possible value for the specific numeric type.

  For example:

  >>> @tf.function(jit_compile=True)
  ... def test(c):
  ...   return tf.raw_ops.SegmentMaxV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> test(c).numpy()
  array([[4, 3, 3, 4],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentMaxV2", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return segment_max_v2_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentMaxV2", data=data, segment_ids=segment_ids,
                        num_segments=num_segments, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentMaxV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentMaxV2 = tf_export("raw_ops.SegmentMaxV2")(_ops.to_raw_op(segment_max_v2))


def segment_max_v2_eager_fallback(data: Annotated[Any, TV_SegmentMaxV2_T], segment_ids: Annotated[Any, TV_SegmentMaxV2_Tindices], num_segments: Annotated[Any, TV_SegmentMaxV2_Tnumsegments], name, ctx) -> Annotated[Any, TV_SegmentMaxV2_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"SegmentMaxV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentMaxV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentMean_T = TypeVar("TV_SegmentMean_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentMean_Tindices = TypeVar("TV_SegmentMean_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.segment_mean', v1=['math.segment_mean', 'segment_mean'])
@deprecated_endpoints('segment_mean')
def segment_mean(data: Annotated[Any, TV_SegmentMean_T], segment_ids: Annotated[Any, TV_SegmentMean_Tindices], name=None) -> Annotated[Any, TV_SegmentMean_T]:
  r"""Computes the mean along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  If the mean is empty for a given segment ID `i`, `output[i] = 0`.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as a smaller following index when computing the numerator
  of the mean.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> tf.math.segment_mean(c, tf.constant([0, 0, 1])).numpy()
  array([[2.5, 2.5, 2.5, 2.5],
         [5., 6., 7., 8.]], dtype=float32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentMean", name, data, segment_ids)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_segment_mean(
          (data, segment_ids, name,), None)
      if _result is not NotImplemented:
        return _result
      return segment_mean_eager_fallback(
          data, segment_ids, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            segment_mean, (), dict(data=data, segment_ids=segment_ids,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_segment_mean(
        (data, segment_ids, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentMean", data=data, segment_ids=segment_ids, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          segment_mean, (), dict(data=data, segment_ids=segment_ids,
                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentMean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentMean = tf_export("raw_ops.SegmentMean")(_ops.to_raw_op(segment_mean))
_dispatcher_for_segment_mean = segment_mean._tf_type_based_dispatcher.Dispatch


def segment_mean_eager_fallback(data: Annotated[Any, TV_SegmentMean_T], segment_ids: Annotated[Any, TV_SegmentMean_Tindices], name, ctx) -> Annotated[Any, TV_SegmentMean_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [data, segment_ids]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SegmentMean", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentMean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentMin_T = TypeVar("TV_SegmentMin_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentMin_Tindices = TypeVar("TV_SegmentMin_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.segment_min', v1=['math.segment_min', 'segment_min'])
@deprecated_endpoints('segment_min')
def segment_min(data: Annotated[Any, TV_SegmentMin_T], segment_ids: Annotated[Any, TV_SegmentMin_Tindices], name=None) -> Annotated[Any, TV_SegmentMin_T]:
  r"""Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  If the min is empty for a given segment ID `i`, `output[i] = 0`.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> tf.math.segment_min(c, tf.constant([0, 0, 1])).numpy()
  array([[1, 2, 2, 1],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentMin", name, data, segment_ids)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_segment_min(
          (data, segment_ids, name,), None)
      if _result is not NotImplemented:
        return _result
      return segment_min_eager_fallback(
          data, segment_ids, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            segment_min, (), dict(data=data, segment_ids=segment_ids,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_segment_min(
        (data, segment_ids, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentMin", data=data, segment_ids=segment_ids, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          segment_min, (), dict(data=data, segment_ids=segment_ids, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentMin = tf_export("raw_ops.SegmentMin")(_ops.to_raw_op(segment_min))
_dispatcher_for_segment_min = segment_min._tf_type_based_dispatcher.Dispatch


def segment_min_eager_fallback(data: Annotated[Any, TV_SegmentMin_T], segment_ids: Annotated[Any, TV_SegmentMin_Tindices], name, ctx) -> Annotated[Any, TV_SegmentMin_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [data, segment_ids]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SegmentMin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentMinV2_T = TypeVar("TV_SegmentMinV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentMinV2_Tindices = TypeVar("TV_SegmentMinV2_Tindices", _atypes.Int32, _atypes.Int64)
TV_SegmentMinV2_Tnumsegments = TypeVar("TV_SegmentMinV2_Tnumsegments", _atypes.Int32, _atypes.Int64)

def segment_min_v2(data: Annotated[Any, TV_SegmentMinV2_T], segment_ids: Annotated[Any, TV_SegmentMinV2_Tindices], num_segments: Annotated[Any, TV_SegmentMinV2_Tnumsegments], name=None) -> Annotated[Any, TV_SegmentMinV2_T]:
  r"""Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  If the minimum is empty for a given segment ID `i`, it outputs the largest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::max()`.

  Note: That this op is currently only supported with jit_compile=True.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  The only difference with SegmentMin is the additional input  `num_segments`.
  This helps in evaluating the output shape in compile time.
  `num_segments` should be consistent with segment_ids.
  e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
  With inconsistent num_segments, the op still runs. only difference is,
  the output takes the size of num_segments irrespective of size of segment_ids and data.
  for num_segments less than expected output size, the last elements are ignored
  for num_segments more than the expected output size, last elements are assigned 
  the largest possible value for the specific numeric type.

  For example:

  >>> @tf.function(jit_compile=True)
  ... def test(c):
  ...   return tf.raw_ops.SegmentMinV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> test(c).numpy()
  array([[1, 2, 2, 1],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentMinV2", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return segment_min_v2_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentMinV2", data=data, segment_ids=segment_ids,
                        num_segments=num_segments, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentMinV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentMinV2 = tf_export("raw_ops.SegmentMinV2")(_ops.to_raw_op(segment_min_v2))


def segment_min_v2_eager_fallback(data: Annotated[Any, TV_SegmentMinV2_T], segment_ids: Annotated[Any, TV_SegmentMinV2_Tindices], num_segments: Annotated[Any, TV_SegmentMinV2_Tnumsegments], name, ctx) -> Annotated[Any, TV_SegmentMinV2_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"SegmentMinV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentMinV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentProd_T = TypeVar("TV_SegmentProd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentProd_Tindices = TypeVar("TV_SegmentProd_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.segment_prod', v1=['math.segment_prod', 'segment_prod'])
@deprecated_endpoints('segment_prod')
def segment_prod(data: Annotated[Any, TV_SegmentProd_T], segment_ids: Annotated[Any, TV_SegmentProd_Tindices], name=None) -> Annotated[Any, TV_SegmentProd_T]:
  r"""Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  If the product is empty for a given segment ID `i`, `output[i] = 1`.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> tf.math.segment_prod(c, tf.constant([0, 0, 1])).numpy()
  array([[4, 6, 6, 4],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentProd", name, data, segment_ids)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_segment_prod(
          (data, segment_ids, name,), None)
      if _result is not NotImplemented:
        return _result
      return segment_prod_eager_fallback(
          data, segment_ids, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            segment_prod, (), dict(data=data, segment_ids=segment_ids,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_segment_prod(
        (data, segment_ids, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentProd", data=data, segment_ids=segment_ids, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          segment_prod, (), dict(data=data, segment_ids=segment_ids,
                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentProd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentProd = tf_export("raw_ops.SegmentProd")(_ops.to_raw_op(segment_prod))
_dispatcher_for_segment_prod = segment_prod._tf_type_based_dispatcher.Dispatch


def segment_prod_eager_fallback(data: Annotated[Any, TV_SegmentProd_T], segment_ids: Annotated[Any, TV_SegmentProd_Tindices], name, ctx) -> Annotated[Any, TV_SegmentProd_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [data, segment_ids]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SegmentProd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentProd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentProdV2_T = TypeVar("TV_SegmentProdV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentProdV2_Tindices = TypeVar("TV_SegmentProdV2_Tindices", _atypes.Int32, _atypes.Int64)
TV_SegmentProdV2_Tnumsegments = TypeVar("TV_SegmentProdV2_Tnumsegments", _atypes.Int32, _atypes.Int64)

def segment_prod_v2(data: Annotated[Any, TV_SegmentProdV2_T], segment_ids: Annotated[Any, TV_SegmentProdV2_Tindices], num_segments: Annotated[Any, TV_SegmentProdV2_Tnumsegments], name=None) -> Annotated[Any, TV_SegmentProdV2_T]:
  r"""Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  If the product is empty for a given segment ID `i`, `output[i] = 1`.

  Note: That this op is currently only supported with jit_compile=True.

  The only difference with SegmentProd is the additional input  `num_segments`.
  This helps in evaluating the output shape in compile time.
  `num_segments` should be consistent with segment_ids.
  e.g. Max(segment_ids) - 1 should be equal to `num_segments` for a 1-d segment_ids
  With inconsistent num_segments, the op still runs. only difference is, 
  the output takes the size of num_segments irrespective of size of segment_ids and data.
  for num_segments less than expected output size, the last elements are ignored
  for num_segments more than the expected output size, last elements are assigned 1.

  For example:

  >>> @tf.function(jit_compile=True)
  ... def test(c):
  ...   return tf.raw_ops.SegmentProdV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> test(c).numpy()
  array([[4, 6, 6, 4],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentProdV2", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return segment_prod_v2_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentProdV2", data=data, segment_ids=segment_ids,
                         num_segments=num_segments, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentProdV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentProdV2 = tf_export("raw_ops.SegmentProdV2")(_ops.to_raw_op(segment_prod_v2))


def segment_prod_v2_eager_fallback(data: Annotated[Any, TV_SegmentProdV2_T], segment_ids: Annotated[Any, TV_SegmentProdV2_Tindices], num_segments: Annotated[Any, TV_SegmentProdV2_Tnumsegments], name, ctx) -> Annotated[Any, TV_SegmentProdV2_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"SegmentProdV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentProdV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentSum_T = TypeVar("TV_SegmentSum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentSum_Tindices = TypeVar("TV_SegmentSum_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.segment_sum', v1=['math.segment_sum', 'segment_sum'])
@deprecated_endpoints('segment_sum')
def segment_sum(data: Annotated[Any, TV_SegmentSum_T], segment_ids: Annotated[Any, TV_SegmentSum_Tindices], name=None) -> Annotated[Any, TV_SegmentSum_T]:
  r"""Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> tf.math.segment_sum(c, tf.constant([0, 0, 1])).numpy()
  array([[5, 5, 5, 5],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentSum", name, data, segment_ids)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_segment_sum(
          (data, segment_ids, name,), None)
      if _result is not NotImplemented:
        return _result
      return segment_sum_eager_fallback(
          data, segment_ids, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            segment_sum, (), dict(data=data, segment_ids=segment_ids,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_segment_sum(
        (data, segment_ids, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentSum", data=data, segment_ids=segment_ids, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          segment_sum, (), dict(data=data, segment_ids=segment_ids, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentSum = tf_export("raw_ops.SegmentSum")(_ops.to_raw_op(segment_sum))
_dispatcher_for_segment_sum = segment_sum._tf_type_based_dispatcher.Dispatch


def segment_sum_eager_fallback(data: Annotated[Any, TV_SegmentSum_T], segment_ids: Annotated[Any, TV_SegmentSum_Tindices], name, ctx) -> Annotated[Any, TV_SegmentSum_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [data, segment_ids]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"SegmentSum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SegmentSumV2_T = TypeVar("TV_SegmentSumV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SegmentSumV2_Tindices = TypeVar("TV_SegmentSumV2_Tindices", _atypes.Int32, _atypes.Int64)
TV_SegmentSumV2_Tnumsegments = TypeVar("TV_SegmentSumV2_Tnumsegments", _atypes.Int32, _atypes.Int64)

def segment_sum_v2(data: Annotated[Any, TV_SegmentSumV2_T], segment_ids: Annotated[Any, TV_SegmentSumV2_Tindices], num_segments: Annotated[Any, TV_SegmentSumV2_Tnumsegments], name=None) -> Annotated[Any, TV_SegmentSumV2_T]:
  r"""Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  Note that this op is currently only supported with jit_compile=True.
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SegmentSumV2", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return segment_sum_v2_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SegmentSumV2", data=data, segment_ids=segment_ids,
                        num_segments=num_segments, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SegmentSumV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SegmentSumV2 = tf_export("raw_ops.SegmentSumV2")(_ops.to_raw_op(segment_sum_v2))


def segment_sum_v2_eager_fallback(data: Annotated[Any, TV_SegmentSumV2_T], segment_ids: Annotated[Any, TV_SegmentSumV2_Tindices], num_segments: Annotated[Any, TV_SegmentSumV2_Tnumsegments], name, ctx) -> Annotated[Any, TV_SegmentSumV2_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"SegmentSumV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SegmentSumV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Select_T = TypeVar("TV_Select_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def select(condition: Annotated[Any, _atypes.Bool], x: Annotated[Any, TV_Select_T], y: Annotated[Any, TV_Select_T], name=None) -> Annotated[Any, TV_Select_T]:
  r"""Selects elements from `x` or `y`, depending on `condition`.

  The `x`, and `y` tensors must all have the same shape, and the
  output will also have that shape.

  The `condition` tensor must be a scalar if `x` and `y` are scalars.
  If `x` and `y` are vectors or higher rank, then `condition` must be either a
  scalar, a vector with size matching the first dimension of `x`, or must have
  the same shape as `x`.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be
  taken from `x` (if true) or `y` (if false).

  If `condition` is a vector and `x` and `y` are higher rank matrices, then
  it chooses which row (outer dimension) to copy from `x` and `y`.
  If `condition` has the same shape as `x` and `y`, then it chooses which
  element to copy from `x` and `y`.

  For example:

  ```python
  # 'condition' tensor is [[True,  False]
  #                        [False, True]]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e)  # => [[1, 6], [7, 4]]


  # 'condition' tensor is [True, False]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e) ==> [[1, 2],
                               [7, 8]]

  ```

  Args:
    condition: A `Tensor` of type `bool`.
    x:  A `Tensor` which may have the same shape as `condition`.
      If `condition` is rank 1, `x` may have higher rank,
      but its first dimension must match the size of `condition`.
    y:  A `Tensor` with the same type and shape as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Select", name, condition, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return select_eager_fallback(
          condition, x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Select", condition=condition, t=x, e=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Select", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Select = tf_export("raw_ops.Select")(_ops.to_raw_op(select))


def select_eager_fallback(condition: Annotated[Any, _atypes.Bool], x: Annotated[Any, TV_Select_T], y: Annotated[Any, TV_Select_T], name, ctx) -> Annotated[Any, TV_Select_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
  (x, y) = _inputs_T
  condition = _ops.convert_to_tensor(condition, _dtypes.bool)
  _inputs_flat = [condition, x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Select", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Select", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SelectV2_T = TypeVar("TV_SelectV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def select_v2(condition: Annotated[Any, _atypes.Bool], t: Annotated[Any, TV_SelectV2_T], e: Annotated[Any, TV_SelectV2_T], name=None) -> Annotated[Any, TV_SelectV2_T]:
  r"""TODO: add doc.

  Args:
    condition: A `Tensor` of type `bool`.
    t: A `Tensor`.
    e: A `Tensor`. Must have the same type as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SelectV2", name, condition, t, e)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return select_v2_eager_fallback(
          condition, t, e, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SelectV2", condition=condition, t=t, e=e, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SelectV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SelectV2 = tf_export("raw_ops.SelectV2")(_ops.to_raw_op(select_v2))


def select_v2_eager_fallback(condition: Annotated[Any, _atypes.Bool], t: Annotated[Any, TV_SelectV2_T], e: Annotated[Any, TV_SelectV2_T], name, ctx) -> Annotated[Any, TV_SelectV2_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([t, e], ctx, [])
  (t, e) = _inputs_T
  condition = _ops.convert_to_tensor(condition, _dtypes.bool)
  _inputs_flat = [condition, t, e]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SelectV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SelectV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sigmoid_T = TypeVar("TV_Sigmoid_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def sigmoid(x: Annotated[Any, TV_Sigmoid_T], name=None) -> Annotated[Any, TV_Sigmoid_T]:
  r"""Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sigmoid", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sigmoid_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sigmoid", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sigmoid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sigmoid = tf_export("raw_ops.Sigmoid")(_ops.to_raw_op(sigmoid))


def sigmoid_eager_fallback(x: Annotated[Any, TV_Sigmoid_T], name, ctx) -> Annotated[Any, TV_Sigmoid_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sigmoid", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sigmoid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SigmoidGrad_T = TypeVar("TV_SigmoidGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def sigmoid_grad(y: Annotated[Any, TV_SigmoidGrad_T], dy: Annotated[Any, TV_SigmoidGrad_T], name=None) -> Annotated[Any, TV_SigmoidGrad_T]:
  r"""Computes the gradient of the sigmoid of `x` wrt its input.

  Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
  `dy` is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SigmoidGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sigmoid_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SigmoidGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SigmoidGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SigmoidGrad = tf_export("raw_ops.SigmoidGrad")(_ops.to_raw_op(sigmoid_grad))


def sigmoid_grad_eager_fallback(y: Annotated[Any, TV_SigmoidGrad_T], dy: Annotated[Any, TV_SigmoidGrad_T], name, ctx) -> Annotated[Any, TV_SigmoidGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SigmoidGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SigmoidGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sign_T = TypeVar("TV_Sign_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8)

def sign(x: Annotated[Any, TV_Sign_T], name=None) -> Annotated[Any, TV_Sign_T]:
  r"""Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Example usage:
  >>> tf.math.sign([0., 2., -3.])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sign", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sign_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sign", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sign = tf_export("raw_ops.Sign")(_ops.to_raw_op(sign))


def sign_eager_fallback(x: Annotated[Any, TV_Sign_T], name, ctx) -> Annotated[Any, TV_Sign_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sign", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sin_T = TypeVar("TV_Sin_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.sin', 'sin')
def sin(x: Annotated[Any, TV_Sin_T], name=None) -> Annotated[Any, TV_Sin_T]:
  r"""Computes sine of x element-wise.

    Given an input tensor, this function computes sine of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `[-1,1]`.

    ```python
    x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
    tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sin", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_sin(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return sin_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            sin, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_sin(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sin", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          sin, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sin = tf_export("raw_ops.Sin")(_ops.to_raw_op(sin))
_dispatcher_for_sin = sin._tf_type_based_dispatcher.Dispatch


def sin_eager_fallback(x: Annotated[Any, TV_Sin_T], name, ctx) -> Annotated[Any, TV_Sin_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sin", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sinh_T = TypeVar("TV_Sinh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.sinh', 'sinh')
def sinh(x: Annotated[Any, TV_Sinh_T], name=None) -> Annotated[Any, TV_Sinh_T]:
  r"""Computes hyperbolic sine of x element-wise.

    Given an input tensor, this function computes hyperbolic sine of every
    element in the tensor. Input range is `[-inf,inf]` and output range
    is `[-inf,inf]`.

    ```python
    x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
    tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sinh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_sinh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return sinh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            sinh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_sinh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sinh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          sinh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sinh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sinh = tf_export("raw_ops.Sinh")(_ops.to_raw_op(sinh))
_dispatcher_for_sinh = sinh._tf_type_based_dispatcher.Dispatch


def sinh_eager_fallback(x: Annotated[Any, TV_Sinh_T], name, ctx) -> Annotated[Any, TV_Sinh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sinh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sinh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SobolSample_dtype = TypeVar("TV_SobolSample_dtype", _atypes.Float32, _atypes.Float64)

def sobol_sample(dim: Annotated[Any, _atypes.Int32], num_results: Annotated[Any, _atypes.Int32], skip: Annotated[Any, _atypes.Int32], dtype:TV_SobolSample_dtype=_dtypes.float32, name=None) -> Annotated[Any, TV_SobolSample_dtype]:
  r"""Generates points from the Sobol sequence.

  Creates a Sobol sequence with `num_results` samples. Each sample has dimension
  `dim`. Skips the first `skip` samples.

  Args:
    dim: A `Tensor` of type `int32`.
      Positive scalar `Tensor` representing each sample's dimension.
    num_results: A `Tensor` of type `int32`.
      Positive scalar `Tensor` of dtype int32. The number of Sobol points to return
      in the output.
    skip: A `Tensor` of type `int32`.
      Positive scalar `Tensor` of dtype int32. The number of initial points of the
      Sobol sequence to skip.
    dtype: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the sample. One of: `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SobolSample", name, dim, num_results, skip, "dtype", dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sobol_sample_eager_fallback(
          dim, num_results, skip, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SobolSample", dim=dim, num_results=num_results, skip=skip,
                       dtype=dtype, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SobolSample", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SobolSample = tf_export("raw_ops.SobolSample")(_ops.to_raw_op(sobol_sample))


def sobol_sample_eager_fallback(dim: Annotated[Any, _atypes.Int32], num_results: Annotated[Any, _atypes.Int32], skip: Annotated[Any, _atypes.Int32], dtype: TV_SobolSample_dtype, name, ctx) -> Annotated[Any, TV_SobolSample_dtype]:
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  dim = _ops.convert_to_tensor(dim, _dtypes.int32)
  num_results = _ops.convert_to_tensor(num_results, _dtypes.int32)
  skip = _ops.convert_to_tensor(skip, _dtypes.int32)
  _inputs_flat = [dim, num_results, skip]
  _attrs = ("dtype", dtype)
  _result = _execute.execute(b"SobolSample", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SobolSample", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseBincount_Tidx = TypeVar("TV_SparseBincount_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseBincount_T = TypeVar("TV_SparseBincount_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def sparse_bincount(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseBincount_Tidx], dense_shape: Annotated[Any, _atypes.Int64], size: Annotated[Any, TV_SparseBincount_Tidx], weights: Annotated[Any, TV_SparseBincount_T], binary_output:bool=False, name=None) -> Annotated[Any, TV_SparseBincount_T]:
  r"""Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    indices: A `Tensor` of type `int64`. 2D int64 `Tensor`.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1D int `Tensor`.
    dense_shape: A `Tensor` of type `int64`. 1D int64 `Tensor`.
    size: A `Tensor`. Must have the same type as `values`.
      non-negative int scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `input`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    binary_output: An optional `bool`. Defaults to `False`.
      bool; Whether the kernel should count the appearance or number of occurrences.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseBincount", name, indices, values, dense_shape, size,
        weights, "binary_output", binary_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_bincount_eager_fallback(
          indices, values, dense_shape, size, weights,
          binary_output=binary_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseBincount", indices=indices, values=values,
                          dense_shape=dense_shape, size=size, weights=weights,
                          binary_output=binary_output, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tidx", _op._get_attr_type("Tidx"), "T",
              _op._get_attr_type("T"), "binary_output",
              _op._get_attr_bool("binary_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseBincount = tf_export("raw_ops.SparseBincount")(_ops.to_raw_op(sparse_bincount))


def sparse_bincount_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseBincount_Tidx], dense_shape: Annotated[Any, _atypes.Int64], size: Annotated[Any, TV_SparseBincount_Tidx], weights: Annotated[Any, TV_SparseBincount_T], binary_output: bool, name, ctx) -> Annotated[Any, TV_SparseBincount_T]:
  if binary_output is None:
    binary_output = False
  binary_output = _execute.make_bool(binary_output, "binary_output")
  _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([values, size], ctx, [_dtypes.int32, _dtypes.int64, ])
  (values, size) = _inputs_Tidx
  _attr_T, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [indices, values, dense_shape, size, weights]
  _attrs = ("Tidx", _attr_Tidx, "T", _attr_T, "binary_output", binary_output)
  _result = _execute.execute(b"SparseBincount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseBincount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseMatMul_Ta = TypeVar("TV_SparseMatMul_Ta", _atypes.BFloat16, _atypes.Float32)
TV_SparseMatMul_Tb = TypeVar("TV_SparseMatMul_Tb", _atypes.BFloat16, _atypes.Float32)

def sparse_mat_mul(a: Annotated[Any, TV_SparseMatMul_Ta], b: Annotated[Any, TV_SparseMatMul_Tb], transpose_a:bool=False, transpose_b:bool=False, a_is_sparse:bool=False, b_is_sparse:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Multiply matrix "a" by matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of "a" must
  match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
  `SparseTensor`s.  This op is optimized for the case where at least one of "a" or
  "b" is sparse, in the sense that they have a large proportion of zero values.
  The breakeven for using this versus a dense matrix multiply on one platform was
  30% zero values in the sparse matrix.

  The gradient computation of this operation will only take advantage of sparsity
  in the input gradient when that gradient comes from a Relu.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseMatMul", name, a, b, "transpose_a", transpose_a,
        "transpose_b", transpose_b, "a_is_sparse", a_is_sparse, "b_is_sparse",
        b_is_sparse)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_mat_mul_eager_fallback(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b,
          a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if a_is_sparse is None:
    a_is_sparse = False
  a_is_sparse = _execute.make_bool(a_is_sparse, "a_is_sparse")
  if b_is_sparse is None:
    b_is_sparse = False
  b_is_sparse = _execute.make_bool(b_is_sparse, "b_is_sparse")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseMatMul", a=a, b=b, transpose_a=transpose_a,
                        transpose_b=transpose_b, a_is_sparse=a_is_sparse,
                        b_is_sparse=b_is_sparse, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "a_is_sparse",
              _op._get_attr_bool("a_is_sparse"), "b_is_sparse",
              _op._get_attr_bool("b_is_sparse"), "Ta",
              _op._get_attr_type("Ta"), "Tb", _op._get_attr_type("Tb"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseMatMul = tf_export("raw_ops.SparseMatMul")(_ops.to_raw_op(sparse_mat_mul))


def sparse_mat_mul_eager_fallback(a: Annotated[Any, TV_SparseMatMul_Ta], b: Annotated[Any, TV_SparseMatMul_Tb], transpose_a: bool, transpose_b: bool, a_is_sparse: bool, b_is_sparse: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if a_is_sparse is None:
    a_is_sparse = False
  a_is_sparse = _execute.make_bool(a_is_sparse, "a_is_sparse")
  if b_is_sparse is None:
    b_is_sparse = False
  b_is_sparse = _execute.make_bool(b_is_sparse, "b_is_sparse")
  _attr_Ta, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.bfloat16, ], _dtypes.float32)
  _attr_Tb, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.float32, _dtypes.bfloat16, ], _dtypes.float32)
  _inputs_flat = [a, b]
  _attrs = ("transpose_a", transpose_a, "transpose_b", transpose_b,
  "a_is_sparse", a_is_sparse, "b_is_sparse", b_is_sparse, "Ta", _attr_Ta,
  "Tb", _attr_Tb)
  _result = _execute.execute(b"SparseMatMul", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseMatMul", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentMean_T = TypeVar("TV_SparseSegmentMean_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentMean_Tidx = TypeVar("TV_SparseSegmentMean_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentMean_Tsegmentids = TypeVar("TV_SparseSegmentMean_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_mean(data: Annotated[Any, TV_SparseSegmentMean_T], indices: Annotated[Any, TV_SparseSegmentMean_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMean_Tsegmentids], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentMean_T]:
  r"""Computes the mean along sparse segments of a tensor.

  See `tf.sparse.segment_sum` for usage examples.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentMean", name, data, indices, segment_ids,
        "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_mean_eager_fallback(
          data, indices, segment_ids, sparse_gradient=sparse_gradient,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentMean", data=data, indices=indices,
                             segment_ids=segment_ids,
                             sparse_gradient=sparse_gradient, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentMean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentMean = tf_export("raw_ops.SparseSegmentMean")(_ops.to_raw_op(sparse_segment_mean))


def sparse_segment_mean_eager_fallback(data: Annotated[Any, TV_SparseSegmentMean_T], indices: Annotated[Any, TV_SparseSegmentMean_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMean_Tsegmentids], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentMean_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids, "sparse_gradient", sparse_gradient)
  _result = _execute.execute(b"SparseSegmentMean", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentMean", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentMeanGrad_T = TypeVar("TV_SparseSegmentMeanGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentMeanGrad_Tidx = TypeVar("TV_SparseSegmentMeanGrad_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentMeanGrad_Tsegmentids = TypeVar("TV_SparseSegmentMeanGrad_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_mean_grad(grad: Annotated[Any, TV_SparseSegmentMeanGrad_T], indices: Annotated[Any, TV_SparseSegmentMeanGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_SparseSegmentMeanGrad_T]:
  r"""Computes gradients for SparseSegmentMean.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentMeanGrad", name, grad, indices, segment_ids,
        output_dim0)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_mean_grad_eager_fallback(
          grad, indices, segment_ids, output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentMeanGrad", grad=grad, indices=indices,
                                 segment_ids=segment_ids,
                                 output_dim0=output_dim0, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentMeanGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentMeanGrad = tf_export("raw_ops.SparseSegmentMeanGrad")(_ops.to_raw_op(sparse_segment_mean_grad))


def sparse_segment_mean_grad_eager_fallback(grad: Annotated[Any, TV_SparseSegmentMeanGrad_T], indices: Annotated[Any, TV_SparseSegmentMeanGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_SparseSegmentMeanGrad_T]:
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  output_dim0 = _ops.convert_to_tensor(output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentMeanGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentMeanGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseSegmentMeanGradV2Output = collections.namedtuple(
    "SparseSegmentMeanGradV2",
    ["output", "sorted_unique_indices"])


TV_SparseSegmentMeanGradV2_T = TypeVar("TV_SparseSegmentMeanGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentMeanGradV2_Tidx = TypeVar("TV_SparseSegmentMeanGradV2_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentMeanGradV2_Tsegmentids = TypeVar("TV_SparseSegmentMeanGradV2_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_mean_grad_v2(grad: Annotated[Any, TV_SparseSegmentMeanGradV2_T], indices: Annotated[Any, TV_SparseSegmentMeanGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name=None):
  r"""Computes gradients for SparseSegmentMean.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is the number of unique indexes in "indices". Also returns vector
  "sorted_unique_indices" containing the corresponding indexes from "indices".

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    dense_output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, sorted_unique_indices).

    output: A `Tensor`. Has the same type as `grad`.
    sorted_unique_indices: A `Tensor`. Has the same type as `indices`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentMeanGradV2", name, grad, indices, segment_ids,
        dense_output_dim0)
      _result = _SparseSegmentMeanGradV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_mean_grad_v2_eager_fallback(
          grad, indices, segment_ids, dense_output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentMeanGradV2", grad=grad, indices=indices,
                                   segment_ids=segment_ids,
                                   dense_output_dim0=dense_output_dim0,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentMeanGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentMeanGradV2Output._make(_result)
  return _result

SparseSegmentMeanGradV2 = tf_export("raw_ops.SparseSegmentMeanGradV2")(_ops.to_raw_op(sparse_segment_mean_grad_v2))


def sparse_segment_mean_grad_v2_eager_fallback(grad: Annotated[Any, TV_SparseSegmentMeanGradV2_T], indices: Annotated[Any, TV_SparseSegmentMeanGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name, ctx):
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  dense_output_dim0 = _ops.convert_to_tensor(dense_output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, dense_output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentMeanGradV2", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentMeanGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentMeanGradV2Output._make(_result)
  return _result


TV_SparseSegmentMeanWithNumSegments_T = TypeVar("TV_SparseSegmentMeanWithNumSegments_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentMeanWithNumSegments_Tidx = TypeVar("TV_SparseSegmentMeanWithNumSegments_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentMeanWithNumSegments_Tnumsegments = TypeVar("TV_SparseSegmentMeanWithNumSegments_Tnumsegments", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentMeanWithNumSegments_Tsegmentids = TypeVar("TV_SparseSegmentMeanWithNumSegments_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_mean_with_num_segments(data: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tnumsegments], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentMeanWithNumSegments_T]:
  r"""Computes the mean along sparse segments of a tensor.

  Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
  missing, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentMeanWithNumSegments", name, data, indices,
        segment_ids, num_segments, "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_mean_with_num_segments_eager_fallback(
          data, indices, segment_ids, num_segments,
          sparse_gradient=sparse_gradient, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentMeanWithNumSegments", data=data, indices=indices,
                                            segment_ids=segment_ids,
                                            num_segments=num_segments,
                                            sparse_gradient=sparse_gradient,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentMeanWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentMeanWithNumSegments = tf_export("raw_ops.SparseSegmentMeanWithNumSegments")(_ops.to_raw_op(sparse_segment_mean_with_num_segments))


def sparse_segment_mean_with_num_segments_eager_fallback(data: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentMeanWithNumSegments_Tnumsegments], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentMeanWithNumSegments_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tnumsegments",
  _attr_Tnumsegments, "Tsegmentids", _attr_Tsegmentids, "sparse_gradient",
  sparse_gradient)
  _result = _execute.execute(b"SparseSegmentMeanWithNumSegments", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentMeanWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentSqrtN_T = TypeVar("TV_SparseSegmentSqrtN_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSqrtN_Tidx = TypeVar("TV_SparseSegmentSqrtN_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSqrtN_Tsegmentids = TypeVar("TV_SparseSegmentSqrtN_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sqrt_n(data: Annotated[Any, TV_SparseSegmentSqrtN_T], indices: Annotated[Any, TV_SparseSegmentSqrtN_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtN_Tsegmentids], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentSqrtN_T]:
  r"""Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  See `tf.sparse.segment_sum` for usage examples.

  Args:
    data: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSqrtN", name, data, indices, segment_ids,
        "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sqrt_n_eager_fallback(
          data, indices, segment_ids, sparse_gradient=sparse_gradient,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSqrtN", data=data, indices=indices,
                              segment_ids=segment_ids,
                              sparse_gradient=sparse_gradient, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSqrtN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSqrtN = tf_export("raw_ops.SparseSegmentSqrtN")(_ops.to_raw_op(sparse_segment_sqrt_n))


def sparse_segment_sqrt_n_eager_fallback(data: Annotated[Any, TV_SparseSegmentSqrtN_T], indices: Annotated[Any, TV_SparseSegmentSqrtN_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtN_Tsegmentids], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentSqrtN_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids, "sparse_gradient", sparse_gradient)
  _result = _execute.execute(b"SparseSegmentSqrtN", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSqrtN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentSqrtNGrad_T = TypeVar("TV_SparseSegmentSqrtNGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSqrtNGrad_Tidx = TypeVar("TV_SparseSegmentSqrtNGrad_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSqrtNGrad_Tsegmentids = TypeVar("TV_SparseSegmentSqrtNGrad_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sqrt_n_grad(grad: Annotated[Any, TV_SparseSegmentSqrtNGrad_T], indices: Annotated[Any, TV_SparseSegmentSqrtNGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_SparseSegmentSqrtNGrad_T]:
  r"""Computes gradients for SparseSegmentSqrtN.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSqrtNGrad", name, grad, indices, segment_ids,
        output_dim0)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sqrt_n_grad_eager_fallback(
          grad, indices, segment_ids, output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSqrtNGrad", grad=grad, indices=indices,
                                  segment_ids=segment_ids,
                                  output_dim0=output_dim0, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSqrtNGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSqrtNGrad = tf_export("raw_ops.SparseSegmentSqrtNGrad")(_ops.to_raw_op(sparse_segment_sqrt_n_grad))


def sparse_segment_sqrt_n_grad_eager_fallback(grad: Annotated[Any, TV_SparseSegmentSqrtNGrad_T], indices: Annotated[Any, TV_SparseSegmentSqrtNGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_SparseSegmentSqrtNGrad_T]:
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  output_dim0 = _ops.convert_to_tensor(output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentSqrtNGrad", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSqrtNGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseSegmentSqrtNGradV2Output = collections.namedtuple(
    "SparseSegmentSqrtNGradV2",
    ["output", "sorted_unique_indices"])


TV_SparseSegmentSqrtNGradV2_T = TypeVar("TV_SparseSegmentSqrtNGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSqrtNGradV2_Tidx = TypeVar("TV_SparseSegmentSqrtNGradV2_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSqrtNGradV2_Tsegmentids = TypeVar("TV_SparseSegmentSqrtNGradV2_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sqrt_n_grad_v2(grad: Annotated[Any, TV_SparseSegmentSqrtNGradV2_T], indices: Annotated[Any, TV_SparseSegmentSqrtNGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name=None):
  r"""Computes gradients for SparseSegmentSqrtN.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is the number of unique indexes in "indices". Also returns vector
  "sorted_unique_indices" containing the corresponding indexes from "indices".

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    dense_output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, sorted_unique_indices).

    output: A `Tensor`. Has the same type as `grad`.
    sorted_unique_indices: A `Tensor`. Has the same type as `indices`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSqrtNGradV2", name, grad, indices, segment_ids,
        dense_output_dim0)
      _result = _SparseSegmentSqrtNGradV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sqrt_n_grad_v2_eager_fallback(
          grad, indices, segment_ids, dense_output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSqrtNGradV2", grad=grad, indices=indices,
                                    segment_ids=segment_ids,
                                    dense_output_dim0=dense_output_dim0,
                                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSqrtNGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentSqrtNGradV2Output._make(_result)
  return _result

SparseSegmentSqrtNGradV2 = tf_export("raw_ops.SparseSegmentSqrtNGradV2")(_ops.to_raw_op(sparse_segment_sqrt_n_grad_v2))


def sparse_segment_sqrt_n_grad_v2_eager_fallback(grad: Annotated[Any, TV_SparseSegmentSqrtNGradV2_T], indices: Annotated[Any, TV_SparseSegmentSqrtNGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name, ctx):
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  dense_output_dim0 = _ops.convert_to_tensor(dense_output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, dense_output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentSqrtNGradV2", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSqrtNGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentSqrtNGradV2Output._make(_result)
  return _result


TV_SparseSegmentSqrtNWithNumSegments_T = TypeVar("TV_SparseSegmentSqrtNWithNumSegments_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSqrtNWithNumSegments_Tidx = TypeVar("TV_SparseSegmentSqrtNWithNumSegments_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments = TypeVar("TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids = TypeVar("TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sqrt_n_with_num_segments(data: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_T]:
  r"""Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
  missing, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSqrtNWithNumSegments", name, data, indices,
        segment_ids, num_segments, "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sqrt_n_with_num_segments_eager_fallback(
          data, indices, segment_ids, num_segments,
          sparse_gradient=sparse_gradient, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSqrtNWithNumSegments", data=data, indices=indices,
                                             segment_ids=segment_ids,
                                             num_segments=num_segments,
                                             sparse_gradient=sparse_gradient,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSqrtNWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSqrtNWithNumSegments = tf_export("raw_ops.SparseSegmentSqrtNWithNumSegments")(_ops.to_raw_op(sparse_segment_sqrt_n_with_num_segments))


def sparse_segment_sqrt_n_with_num_segments_eager_fallback(data: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentSqrtNWithNumSegments_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tnumsegments",
  _attr_Tnumsegments, "Tsegmentids", _attr_Tsegmentids, "sparse_gradient",
  sparse_gradient)
  _result = _execute.execute(b"SparseSegmentSqrtNWithNumSegments", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSqrtNWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentSum_T = TypeVar("TV_SparseSegmentSum_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseSegmentSum_Tidx = TypeVar("TV_SparseSegmentSum_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSum_Tsegmentids = TypeVar("TV_SparseSegmentSum_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sum(data: Annotated[Any, TV_SparseSegmentSum_T], indices: Annotated[Any, TV_SparseSegmentSum_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSum_Tsegmentids], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentSum_T]:
  r"""Computes the sum along sparse segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSum", name, data, indices, segment_ids,
        "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sum_eager_fallback(
          data, indices, segment_ids, sparse_gradient=sparse_gradient,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSum", data=data, indices=indices,
                            segment_ids=segment_ids,
                            sparse_gradient=sparse_gradient, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSum = tf_export("raw_ops.SparseSegmentSum")(_ops.to_raw_op(sparse_segment_sum))


def sparse_segment_sum_eager_fallback(data: Annotated[Any, TV_SparseSegmentSum_T], indices: Annotated[Any, TV_SparseSegmentSum_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSum_Tsegmentids], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentSum_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids, "sparse_gradient", sparse_gradient)
  _result = _execute.execute(b"SparseSegmentSum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseSegmentSumGrad_T = TypeVar("TV_SparseSegmentSumGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSumGrad_Tidx = TypeVar("TV_SparseSegmentSumGrad_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSumGrad_Tsegmentids = TypeVar("TV_SparseSegmentSumGrad_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sum_grad(grad: Annotated[Any, TV_SparseSegmentSumGrad_T], indices: Annotated[Any, TV_SparseSegmentSumGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_SparseSegmentSumGrad_T]:
  r"""Computes gradients for SparseSegmentSum.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentSum op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSum op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentSum op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSum op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSumGrad", name, grad, indices, segment_ids,
        output_dim0)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sum_grad_eager_fallback(
          grad, indices, segment_ids, output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSumGrad", grad=grad, indices=indices,
                                segment_ids=segment_ids,
                                output_dim0=output_dim0, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSumGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSumGrad = tf_export("raw_ops.SparseSegmentSumGrad")(_ops.to_raw_op(sparse_segment_sum_grad))


def sparse_segment_sum_grad_eager_fallback(grad: Annotated[Any, TV_SparseSegmentSumGrad_T], indices: Annotated[Any, TV_SparseSegmentSumGrad_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumGrad_Tsegmentids], output_dim0: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_SparseSegmentSumGrad_T]:
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  output_dim0 = _ops.convert_to_tensor(output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentSumGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSumGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseSegmentSumGradV2Output = collections.namedtuple(
    "SparseSegmentSumGradV2",
    ["output", "sorted_unique_indices"])


TV_SparseSegmentSumGradV2_T = TypeVar("TV_SparseSegmentSumGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSegmentSumGradV2_Tidx = TypeVar("TV_SparseSegmentSumGradV2_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSumGradV2_Tsegmentids = TypeVar("TV_SparseSegmentSumGradV2_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sum_grad_v2(grad: Annotated[Any, TV_SparseSegmentSumGradV2_T], indices: Annotated[Any, TV_SparseSegmentSumGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name=None):
  r"""Computes gradients for SparseSegmentSum.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is the number of unique indexes in "indices". Also returns vector
  "sorted_unique_indices" containing the corresponding indexes from "indices".

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentSum op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSum op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentSum op.
    dense_output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSum op.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, sorted_unique_indices).

    output: A `Tensor`. Has the same type as `grad`.
    sorted_unique_indices: A `Tensor`. Has the same type as `indices`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSumGradV2", name, grad, indices, segment_ids,
        dense_output_dim0)
      _result = _SparseSegmentSumGradV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sum_grad_v2_eager_fallback(
          grad, indices, segment_ids, dense_output_dim0, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSumGradV2", grad=grad, indices=indices,
                                  segment_ids=segment_ids,
                                  dense_output_dim0=dense_output_dim0,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSumGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentSumGradV2Output._make(_result)
  return _result

SparseSegmentSumGradV2 = tf_export("raw_ops.SparseSegmentSumGradV2")(_ops.to_raw_op(sparse_segment_sum_grad_v2))


def sparse_segment_sum_grad_v2_eager_fallback(grad: Annotated[Any, TV_SparseSegmentSumGradV2_T], indices: Annotated[Any, TV_SparseSegmentSumGradV2_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumGradV2_Tsegmentids], dense_output_dim0: Annotated[Any, _atypes.Int32], name, ctx):
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  dense_output_dim0 = _ops.convert_to_tensor(dense_output_dim0, _dtypes.int32)
  _inputs_flat = [grad, indices, segment_ids, dense_output_dim0]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tsegmentids",
  _attr_Tsegmentids)
  _result = _execute.execute(b"SparseSegmentSumGradV2", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSumGradV2", _inputs_flat, _attrs, _result)
  _result = _SparseSegmentSumGradV2Output._make(_result)
  return _result


TV_SparseSegmentSumWithNumSegments_T = TypeVar("TV_SparseSegmentSumWithNumSegments_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseSegmentSumWithNumSegments_Tidx = TypeVar("TV_SparseSegmentSumWithNumSegments_Tidx", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSumWithNumSegments_Tnumsegments = TypeVar("TV_SparseSegmentSumWithNumSegments_Tnumsegments", _atypes.Int32, _atypes.Int64)
TV_SparseSegmentSumWithNumSegments_Tsegmentids = TypeVar("TV_SparseSegmentSumWithNumSegments_Tsegmentids", _atypes.Int32, _atypes.Int64)

def sparse_segment_sum_with_num_segments(data: Annotated[Any, TV_SparseSegmentSumWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tnumsegments], sparse_gradient:bool=False, name=None) -> Annotated[Any, TV_SparseSegmentSumWithNumSegments_T]:
  r"""Computes the sum along sparse segments of a tensor.

  Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
  missing, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
  for an explanation of segments.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  tf.sparse_segment_sum_with_num_segments(
      c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
  # => [[0 0 0 0]
  #     [0 0 0 0]
  #     [0 0 0 0]]

  tf.sparse_segment_sum_with_num_segments(c,
                                          tf.constant([0, 1]),
                                          tf.constant([0, 2],
                                          num_segments=4))
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    sparse_gradient: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSegmentSumWithNumSegments", name, data, indices,
        segment_ids, num_segments, "sparse_gradient", sparse_gradient)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_segment_sum_with_num_segments_eager_fallback(
          data, indices, segment_ids, num_segments,
          sparse_gradient=sparse_gradient, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSegmentSumWithNumSegments", data=data, indices=indices,
                                           segment_ids=segment_ids,
                                           num_segments=num_segments,
                                           sparse_gradient=sparse_gradient,
                                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tidx",
              _op._get_attr_type("Tidx"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"), "Tsegmentids",
              _op._get_attr_type("Tsegmentids"), "sparse_gradient",
              _op._get_attr_bool("sparse_gradient"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSegmentSumWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseSegmentSumWithNumSegments = tf_export("raw_ops.SparseSegmentSumWithNumSegments")(_ops.to_raw_op(sparse_segment_sum_with_num_segments))


def sparse_segment_sum_with_num_segments_eager_fallback(data: Annotated[Any, TV_SparseSegmentSumWithNumSegments_T], indices: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tidx], segment_ids: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tsegmentids], num_segments: Annotated[Any, TV_SparseSegmentSumWithNumSegments_Tnumsegments], sparse_gradient: bool, name, ctx) -> Annotated[Any, TV_SparseSegmentSumWithNumSegments_T]:
  if sparse_gradient is None:
    sparse_gradient = False
  sparse_gradient = _execute.make_bool(sparse_gradient, "sparse_gradient")
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, indices, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tidx", _attr_Tidx, "Tnumsegments",
  _attr_Tnumsegments, "Tsegmentids", _attr_Tsegmentids, "sparse_gradient",
  sparse_gradient)
  _result = _execute.execute(b"SparseSegmentSumWithNumSegments", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSegmentSumWithNumSegments", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sqrt_T = TypeVar("TV_Sqrt_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def sqrt(x: Annotated[Any, TV_Sqrt_T], name=None) -> Annotated[Any, TV_Sqrt_T]:
  r"""Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sqrt", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sqrt_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sqrt", x=x, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sqrt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sqrt = tf_export("raw_ops.Sqrt")(_ops.to_raw_op(sqrt))


def sqrt_eager_fallback(x: Annotated[Any, TV_Sqrt_T], name, ctx) -> Annotated[Any, TV_Sqrt_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sqrt", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sqrt", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SqrtGrad_T = TypeVar("TV_SqrtGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def sqrt_grad(y: Annotated[Any, TV_SqrtGrad_T], dy: Annotated[Any, TV_SqrtGrad_T], name=None) -> Annotated[Any, TV_SqrtGrad_T]:
  r"""Computes the gradient for the sqrt of `x` wrt its input.

  Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SqrtGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sqrt_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SqrtGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SqrtGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SqrtGrad = tf_export("raw_ops.SqrtGrad")(_ops.to_raw_op(sqrt_grad))


def sqrt_grad_eager_fallback(y: Annotated[Any, TV_SqrtGrad_T], dy: Annotated[Any, TV_SqrtGrad_T], name, ctx) -> Annotated[Any, TV_SqrtGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SqrtGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SqrtGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Square_T = TypeVar("TV_Square_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.square', 'square')
def square(x: Annotated[Any, TV_Square_T], name=None) -> Annotated[Any, TV_Square_T]:
  r"""Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  >>> tf.math.square([-2., 0., 3.])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([4., 0., 9.], dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Square", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_square(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return square_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            square, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_square(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Square", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          square, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Square", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Square = tf_export("raw_ops.Square")(_ops.to_raw_op(square))
_dispatcher_for_square = square._tf_type_based_dispatcher.Dispatch


def square_eager_fallback(x: Annotated[Any, TV_Square_T], name, ctx) -> Annotated[Any, TV_Square_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Square", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Square", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SquaredDifference_T = TypeVar("TV_SquaredDifference_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.squared_difference', v1=['math.squared_difference', 'squared_difference'])
@deprecated_endpoints('squared_difference')
def squared_difference(x: Annotated[Any, TV_SquaredDifference_T], y: Annotated[Any, TV_SquaredDifference_T], name=None) -> Annotated[Any, TV_SquaredDifference_T]:
  r"""Returns conj(x - y)(x - y) element-wise.

  *NOTE*: `math.squared_difference` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SquaredDifference", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_squared_difference(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return squared_difference_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            squared_difference, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_squared_difference(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SquaredDifference", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          squared_difference, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SquaredDifference", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SquaredDifference = tf_export("raw_ops.SquaredDifference")(_ops.to_raw_op(squared_difference))
_dispatcher_for_squared_difference = squared_difference._tf_type_based_dispatcher.Dispatch


def squared_difference_eager_fallback(x: Annotated[Any, TV_SquaredDifference_T], y: Annotated[Any, TV_SquaredDifference_T], name, ctx) -> Annotated[Any, TV_SquaredDifference_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SquaredDifference", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SquaredDifference", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sub_T = TypeVar("TV_Sub_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def sub(x: Annotated[Any, TV_Sub_T], y: Annotated[Any, TV_Sub_T], name=None) -> Annotated[Any, TV_Sub_T]:
  r"""Returns x - y element-wise.

  *NOTE*: `tf.subtract` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Both input and output have a range `(-inf, inf)`.

  Example usages below.

  Subtract operation between an array and a scalar:

  >>> x = [1, 2, 3, 4, 5]
  >>> y = 1
  >>> tf.subtract(x, y)
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
  >>> tf.subtract(y, x)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>

  Note that binary `-` operator can be used instead:

  >>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
  >>> y = tf.convert_to_tensor(1)
  >>> x - y
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>

  Subtract operation between an array and a tensor of same shape:

  >>> x = [1, 2, 3, 4, 5]
  >>> y = tf.constant([5, 4, 3, 2, 1])
  >>> tf.subtract(y, x)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>

  **Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
  non-tensor, the non-tensor input will adopt (or get casted to) the data type
  of the tensor input. This can potentially cause unwanted overflow or underflow
  conversion.

  For example,

  >>> x = tf.constant([1, 2], dtype=tf.int8)
  >>> y = [2**8 + 1, 2**8 + 2]
  >>> tf.subtract(x, y)
  <tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>

  When subtracting two input values of different shapes, `tf.subtract` follows the
  [general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
  . The two input array shapes are compared element-wise. Starting with the
  trailing dimensions, the two dimensions either have to be equal or one of them
  needs to be `1`.

  For example,

  >>> x = np.ones(6).reshape(2, 3, 1)
  >>> y = np.ones(6).reshape(2, 1, 3)
  >>> tf.subtract(x, y)
  <tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
  array([[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],
         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]])>

  Example with inputs of different dimensions:

  >>> x = np.ones(6).reshape(2, 3, 1)
  >>> y = np.ones(6).reshape(1, 6)
  >>> tf.subtract(x, y)
  <tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
  array([[[0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]]])>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sub", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sub_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sub", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sub = tf_export("raw_ops.Sub")(_ops.to_raw_op(sub))


def sub_eager_fallback(x: Annotated[Any, TV_Sub_T], y: Annotated[Any, TV_Sub_T], name, ctx) -> Annotated[Any, TV_Sub_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, _dtypes.uint32, _dtypes.uint64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Sub", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sub", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Sum_T = TypeVar("TV_Sum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_Sum_Tidx = TypeVar("TV_Sum_Tidx", _atypes.Int32, _atypes.Int64)

def _sum(input: Annotated[Any, TV_Sum_T], axis: Annotated[Any, TV_Sum_Tidx], keep_dims:bool=False, name=None) -> Annotated[Any, TV_Sum_T]:
  r"""Computes the sum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Sum", name, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _sum_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Sum", input=input, reduction_indices=axis, keep_dims=keep_dims,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "T",
              _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Sum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Sum = tf_export("raw_ops.Sum")(_ops.to_raw_op(_sum))


def _sum_eager_fallback(input: Annotated[Any, TV_Sum_T], axis: Annotated[Any, TV_Sum_Tidx], keep_dims: bool, name, ctx) -> Annotated[Any, TV_Sum_T]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, axis]
  _attrs = ("keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx)
  _result = _execute.execute(b"Sum", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Sum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Tan_T = TypeVar("TV_Tan_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.tan', 'tan')
def tan(x: Annotated[Any, TV_Tan_T], name=None) -> Annotated[Any, TV_Tan_T]:
  r"""Computes tan of x element-wise.

    Given an input tensor, this function computes tangent of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
    is returned.

    ```python
    x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Tan", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tan(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return tan_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tan, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tan(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Tan", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tan, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Tan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Tan = tf_export("raw_ops.Tan")(_ops.to_raw_op(tan))
_dispatcher_for_tan = tan._tf_type_based_dispatcher.Dispatch


def tan_eager_fallback(x: Annotated[Any, TV_Tan_T], name, ctx) -> Annotated[Any, TV_Tan_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Tan", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Tan", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Tanh_T = TypeVar("TV_Tanh_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.tanh', 'nn.tanh', 'tanh')
def tanh(x: Annotated[Any, TV_Tanh_T], name=None) -> Annotated[Any, TV_Tanh_T]:
  r"""Computes hyperbolic tangent of `x` element-wise.

    Given an input tensor, this function computes hyperbolic tangent of every
    element in the tensor. Input range is `[-inf, inf]` and
    output range is `[-1,1]`.

    >>> x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
    >>> tf.math.tanh(x)
    <tf.Tensor: shape=(8,), dtype=float32, numpy=
    array([-1.0, -0.99990916, -0.46211717,  0.7615942 ,  0.8336547 ,
            0.9640276 ,  0.9950547 ,  1.0], dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Tanh", name, x)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tanh(
          (x, name,), None)
      if _result is not NotImplemented:
        return _result
      return tanh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tanh, (), dict(x=x, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tanh(
        (x, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Tanh", x=x, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tanh, (), dict(x=x, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Tanh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Tanh = tf_export("raw_ops.Tanh")(_ops.to_raw_op(tanh))
_dispatcher_for_tanh = tanh._tf_type_based_dispatcher.Dispatch


def tanh_eager_fallback(x: Annotated[Any, TV_Tanh_T], name, ctx) -> Annotated[Any, TV_Tanh_T]:
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  _inputs_flat = [x]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Tanh", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Tanh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TanhGrad_T = TypeVar("TV_TanhGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def tanh_grad(y: Annotated[Any, TV_TanhGrad_T], dy: Annotated[Any, TV_TanhGrad_T], name=None) -> Annotated[Any, TV_TanhGrad_T]:
  r"""Computes the gradient for the tanh of `x` wrt its input.

  Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TanhGrad", name, y, dy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tanh_grad_eager_fallback(
          y, dy, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TanhGrad", y=y, dy=dy, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TanhGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TanhGrad = tf_export("raw_ops.TanhGrad")(_ops.to_raw_op(tanh_grad))


def tanh_grad_eager_fallback(y: Annotated[Any, TV_TanhGrad_T], dy: Annotated[Any, TV_TanhGrad_T], name, ctx) -> Annotated[Any, TV_TanhGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y, dy], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (y, dy) = _inputs_T
  _inputs_flat = [y, dy]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TanhGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TanhGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TruncateDiv_T = TypeVar("TV_TruncateDiv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('truncatediv')
def truncate_div(x: Annotated[Any, TV_TruncateDiv_T], y: Annotated[Any, TV_TruncateDiv_T], name=None) -> Annotated[Any, TV_TruncateDiv_T]:
  r"""Returns x / y element-wise, rounded towards zero.

  Truncation designates that negative numbers will round fractional quantities
  toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
  than Python semantics. See `FloorDiv` for a division function that matches
  Python Semantics.

  *NOTE*: `truncatediv` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `uint32`, `uint64`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TruncateDiv", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_truncate_div(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return truncate_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            truncate_div, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_truncate_div(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TruncateDiv", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          truncate_div, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TruncateDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TruncateDiv = tf_export("raw_ops.TruncateDiv")(_ops.to_raw_op(truncate_div))
_dispatcher_for_truncate_div = truncate_div._tf_type_based_dispatcher.Dispatch


def truncate_div_eager_fallback(x: Annotated[Any, TV_TruncateDiv_T], y: Annotated[Any, TV_TruncateDiv_T], name, ctx) -> Annotated[Any, TV_TruncateDiv_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.uint8, _dtypes.int8, _dtypes.uint16, _dtypes.int16, _dtypes.int32, _dtypes.uint32, _dtypes.uint64, _dtypes.int64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TruncateDiv", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TruncateDiv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TruncateMod_T = TypeVar("TV_TruncateMod_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('truncatemod')
def truncate_mod(x: Annotated[Any, TV_TruncateMod_T], y: Annotated[Any, TV_TruncateMod_T], name=None) -> Annotated[Any, TV_TruncateMod_T]:
  r"""Returns element-wise remainder of division. This emulates C semantics in that

  the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
  y + truncate_mod(x, y) = x`.

  *NOTE*: `truncatemod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TruncateMod", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_truncate_mod(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return truncate_mod_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            truncate_mod, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_truncate_mod(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TruncateMod", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          truncate_mod, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TruncateMod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TruncateMod = tf_export("raw_ops.TruncateMod")(_ops.to_raw_op(truncate_mod))
_dispatcher_for_truncate_mod = truncate_mod._tf_type_based_dispatcher.Dispatch


def truncate_mod_eager_fallback(x: Annotated[Any, TV_TruncateMod_T], y: Annotated[Any, TV_TruncateMod_T], name, ctx) -> Annotated[Any, TV_TruncateMod_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TruncateMod", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TruncateMod", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnsortedSegmentMax_T = TypeVar("TV_UnsortedSegmentMax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_UnsortedSegmentMax_Tindices = TypeVar("TV_UnsortedSegmentMax_Tindices", _atypes.Int32, _atypes.Int64)
TV_UnsortedSegmentMax_Tnumsegments = TypeVar("TV_UnsortedSegmentMax_Tnumsegments", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.unsorted_segment_max', v1=['math.unsorted_segment_max', 'unsorted_segment_max'])
@deprecated_endpoints('unsorted_segment_max')
def unsorted_segment_max(data: Annotated[Any, TV_UnsortedSegmentMax_T], segment_ids: Annotated[Any, TV_UnsortedSegmentMax_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentMax_Tnumsegments], name=None) -> Annotated[Any, TV_UnsortedSegmentMax_T]:
  r"""Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to `tf.math.unsorted_segment_sum`,
  Instead of computing the sum over segments, it computes the maximum such that:

  \\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  >>> tf.math.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
  array([[4, 3, 3, 4],
         [5,  6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be in range on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnsortedSegmentMax", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unsorted_segment_max(
          (data, segment_ids, num_segments, name,), None)
      if _result is not NotImplemented:
        return _result
      return unsorted_segment_max_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unsorted_segment_max, (), dict(data=data, segment_ids=segment_ids,
                                           num_segments=num_segments,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unsorted_segment_max(
        (data, segment_ids, num_segments, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnsortedSegmentMax", data=data, segment_ids=segment_ids,
                              num_segments=num_segments, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unsorted_segment_max, (), dict(data=data, segment_ids=segment_ids,
                                         num_segments=num_segments, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnsortedSegmentMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnsortedSegmentMax = tf_export("raw_ops.UnsortedSegmentMax")(_ops.to_raw_op(unsorted_segment_max))
_dispatcher_for_unsorted_segment_max = unsorted_segment_max._tf_type_based_dispatcher.Dispatch


def unsorted_segment_max_eager_fallback(data: Annotated[Any, TV_UnsortedSegmentMax_T], segment_ids: Annotated[Any, TV_UnsortedSegmentMax_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentMax_Tnumsegments], name, ctx) -> Annotated[Any, TV_UnsortedSegmentMax_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"UnsortedSegmentMax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnsortedSegmentMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnsortedSegmentMin_T = TypeVar("TV_UnsortedSegmentMin_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_UnsortedSegmentMin_Tindices = TypeVar("TV_UnsortedSegmentMin_Tindices", _atypes.Int32, _atypes.Int64)
TV_UnsortedSegmentMin_Tnumsegments = TypeVar("TV_UnsortedSegmentMin_Tnumsegments", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.unsorted_segment_min', v1=['math.unsorted_segment_min', 'unsorted_segment_min'])
@deprecated_endpoints('unsorted_segment_min')
def unsorted_segment_min(data: Annotated[Any, TV_UnsortedSegmentMin_T], segment_ids: Annotated[Any, TV_UnsortedSegmentMin_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentMin_Tnumsegments], name=None) -> Annotated[Any, TV_UnsortedSegmentMin_T]:
  r"""Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to `tf.math.unsorted_segment_sum`,
  Instead of computing the sum over segments, it computes the minimum such that:

  \\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the minimum is empty for a given segment ID `i`, it outputs the largest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::max()`.

  For example:

  >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  >>> tf.math.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
  array([[1, 2, 2, 1],
         [5, 6, 7, 8]], dtype=int32)

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be in range on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnsortedSegmentMin", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unsorted_segment_min(
          (data, segment_ids, num_segments, name,), None)
      if _result is not NotImplemented:
        return _result
      return unsorted_segment_min_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unsorted_segment_min, (), dict(data=data, segment_ids=segment_ids,
                                           num_segments=num_segments,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unsorted_segment_min(
        (data, segment_ids, num_segments, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnsortedSegmentMin", data=data, segment_ids=segment_ids,
                              num_segments=num_segments, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unsorted_segment_min, (), dict(data=data, segment_ids=segment_ids,
                                         num_segments=num_segments, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnsortedSegmentMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnsortedSegmentMin = tf_export("raw_ops.UnsortedSegmentMin")(_ops.to_raw_op(unsorted_segment_min))
_dispatcher_for_unsorted_segment_min = unsorted_segment_min._tf_type_based_dispatcher.Dispatch


def unsorted_segment_min_eager_fallback(data: Annotated[Any, TV_UnsortedSegmentMin_T], segment_ids: Annotated[Any, TV_UnsortedSegmentMin_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentMin_Tnumsegments], name, ctx) -> Annotated[Any, TV_UnsortedSegmentMin_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"UnsortedSegmentMin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnsortedSegmentMin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnsortedSegmentProd_T = TypeVar("TV_UnsortedSegmentProd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_UnsortedSegmentProd_Tindices = TypeVar("TV_UnsortedSegmentProd_Tindices", _atypes.Int32, _atypes.Int64)
TV_UnsortedSegmentProd_Tnumsegments = TypeVar("TV_UnsortedSegmentProd_Tnumsegments", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.unsorted_segment_prod', v1=['math.unsorted_segment_prod', 'unsorted_segment_prod'])
@deprecated_endpoints('unsorted_segment_prod')
def unsorted_segment_prod(data: Annotated[Any, TV_UnsortedSegmentProd_T], segment_ids: Annotated[Any, TV_UnsortedSegmentProd_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentProd_Tnumsegments], name=None) -> Annotated[Any, TV_UnsortedSegmentProd_T]:
  r"""Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to `tf.math.unsorted_segment_sum`,
  Instead of computing the sum over segments, it computes the product of all
  entries belonging to a segment such that:

  \\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
  `j...` such that `segment_ids[j...] == i`.

  For example:

  >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  >>> tf.math.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
  array([[4, 6, 6, 4],
         [5, 6, 7, 8]], dtype=int32)

  If there is no entry for a given segment ID `i`, it outputs 1.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.
  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be in range on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnsortedSegmentProd", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unsorted_segment_prod(
          (data, segment_ids, num_segments, name,), None)
      if _result is not NotImplemented:
        return _result
      return unsorted_segment_prod_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unsorted_segment_prod, (), dict(data=data,
                                            segment_ids=segment_ids,
                                            num_segments=num_segments,
                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unsorted_segment_prod(
        (data, segment_ids, num_segments, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnsortedSegmentProd", data=data, segment_ids=segment_ids,
                               num_segments=num_segments, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unsorted_segment_prod, (), dict(data=data, segment_ids=segment_ids,
                                          num_segments=num_segments,
                                          name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnsortedSegmentProd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnsortedSegmentProd = tf_export("raw_ops.UnsortedSegmentProd")(_ops.to_raw_op(unsorted_segment_prod))
_dispatcher_for_unsorted_segment_prod = unsorted_segment_prod._tf_type_based_dispatcher.Dispatch


def unsorted_segment_prod_eager_fallback(data: Annotated[Any, TV_UnsortedSegmentProd_T], segment_ids: Annotated[Any, TV_UnsortedSegmentProd_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentProd_Tnumsegments], name, ctx) -> Annotated[Any, TV_UnsortedSegmentProd_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"UnsortedSegmentProd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnsortedSegmentProd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnsortedSegmentSum_T = TypeVar("TV_UnsortedSegmentSum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_UnsortedSegmentSum_Tindices = TypeVar("TV_UnsortedSegmentSum_Tindices", _atypes.Int16, _atypes.Int32, _atypes.Int64)
TV_UnsortedSegmentSum_Tnumsegments = TypeVar("TV_UnsortedSegmentSum_Tnumsegments", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.unsorted_segment_sum', v1=['math.unsorted_segment_sum', 'unsorted_segment_sum'])
@deprecated_endpoints('unsorted_segment_sum')
def unsorted_segment_sum(data: Annotated[Any, TV_UnsortedSegmentSum_T], segment_ids: Annotated[Any, TV_UnsortedSegmentSum_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentSum_Tnumsegments], name=None) -> Annotated[Any, TV_UnsortedSegmentSum_T]:
  r"""Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
  that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
  range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.
  If the given segment ID `i` is negative, the value is dropped and will not be
  added to the sum of the segment.

  `num_segments` should equal the number of distinct segment IDs.

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
  </div>

  >>> c = [[1,2,3,4], [5,6,7,8], [4,3,2,1]]
  >>> tf.math.unsorted_segment_sum(c, [0, 1, 0], num_segments=2).numpy()
  array([[5, 5, 5, 5],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be in range on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnsortedSegmentSum", name, data, segment_ids, num_segments)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unsorted_segment_sum(
          (data, segment_ids, num_segments, name,), None)
      if _result is not NotImplemented:
        return _result
      return unsorted_segment_sum_eager_fallback(
          data, segment_ids, num_segments, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unsorted_segment_sum, (), dict(data=data, segment_ids=segment_ids,
                                           num_segments=num_segments,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unsorted_segment_sum(
        (data, segment_ids, num_segments, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnsortedSegmentSum", data=data, segment_ids=segment_ids,
                              num_segments=num_segments, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unsorted_segment_sum, (), dict(data=data, segment_ids=segment_ids,
                                         num_segments=num_segments, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnsortedSegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnsortedSegmentSum = tf_export("raw_ops.UnsortedSegmentSum")(_ops.to_raw_op(unsorted_segment_sum))
_dispatcher_for_unsorted_segment_sum = unsorted_segment_sum._tf_type_based_dispatcher.Dispatch


def unsorted_segment_sum_eager_fallback(data: Annotated[Any, TV_UnsortedSegmentSum_T], segment_ids: Annotated[Any, TV_UnsortedSegmentSum_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentSum_Tnumsegments], name, ctx) -> Annotated[Any, TV_UnsortedSegmentSum_T]:
  _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [data, segment_ids, num_segments]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "Tnumsegments",
  _attr_Tnumsegments)
  _result = _execute.execute(b"UnsortedSegmentSum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnsortedSegmentSum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Xdivy_T = TypeVar("TV_Xdivy_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def xdivy(x: Annotated[Any, TV_Xdivy_T], y: Annotated[Any, TV_Xdivy_T], name=None) -> Annotated[Any, TV_Xdivy_T]:
  r"""Returns 0 if x == 0, and x / y otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Xdivy", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xdivy_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Xdivy", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Xdivy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Xdivy = tf_export("raw_ops.Xdivy")(_ops.to_raw_op(xdivy))


def xdivy_eager_fallback(x: Annotated[Any, TV_Xdivy_T], y: Annotated[Any, TV_Xdivy_T], name, ctx) -> Annotated[Any, TV_Xdivy_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Xdivy", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Xdivy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Xlog1py_T = TypeVar("TV_Xlog1py_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

def xlog1py(x: Annotated[Any, TV_Xlog1py_T], y: Annotated[Any, TV_Xlog1py_T], name=None) -> Annotated[Any, TV_Xlog1py_T]:
  r"""Returns 0 if x == 0, and x * log1p(y) otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Xlog1py", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return xlog1py_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Xlog1py", x=x, y=y, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Xlog1py", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Xlog1py = tf_export("raw_ops.Xlog1py")(_ops.to_raw_op(xlog1py))


def xlog1py_eager_fallback(x: Annotated[Any, TV_Xlog1py_T], y: Annotated[Any, TV_Xlog1py_T], name, ctx) -> Annotated[Any, TV_Xlog1py_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Xlog1py", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Xlog1py", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Xlogy_T = TypeVar("TV_Xlogy_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.xlogy')
def xlogy(x: Annotated[Any, TV_Xlogy_T], y: Annotated[Any, TV_Xlogy_T], name=None) -> Annotated[Any, TV_Xlogy_T]:
  r"""Returns 0 if x == 0, and x * log(y) otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Xlogy", name, x, y)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xlogy(
          (x, y, name,), None)
      if _result is not NotImplemented:
        return _result
      return xlogy_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xlogy, (), dict(x=x, y=y, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xlogy(
        (x, y, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Xlogy", x=x, y=y, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xlogy, (), dict(x=x, y=y, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Xlogy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Xlogy = tf_export("raw_ops.Xlogy")(_ops.to_raw_op(xlogy))
_dispatcher_for_xlogy = xlogy._tf_type_based_dispatcher.Dispatch


def xlogy_eager_fallback(x: Annotated[Any, TV_Xlogy_T], y: Annotated[Any, TV_Xlogy_T], name, ctx) -> Annotated[Any, TV_Xlogy_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128, ])
  (x, y) = _inputs_T
  _inputs_flat = [x, y]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Xlogy", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Xlogy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Zeta_T = TypeVar("TV_Zeta_T", _atypes.Float32, _atypes.Float64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.zeta', v1=['math.zeta', 'zeta'])
@deprecated_endpoints('zeta')
def zeta(x: Annotated[Any, TV_Zeta_T], q: Annotated[Any, TV_Zeta_T], name=None) -> Annotated[Any, TV_Zeta_T]:
  r"""Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

  The Hurwitz zeta function is defined as:


  \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    q: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Zeta", name, x, q)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_zeta(
          (x, q, name,), None)
      if _result is not NotImplemented:
        return _result
      return zeta_eager_fallback(
          x, q, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            zeta, (), dict(x=x, q=q, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_zeta(
        (x, q, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Zeta", x=x, q=q, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          zeta, (), dict(x=x, q=q, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Zeta", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Zeta = tf_export("raw_ops.Zeta")(_ops.to_raw_op(zeta))
_dispatcher_for_zeta = zeta._tf_type_based_dispatcher.Dispatch


def zeta_eager_fallback(x: Annotated[Any, TV_Zeta_T], q: Annotated[Any, TV_Zeta_T], name, ctx) -> Annotated[Any, TV_Zeta_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, q], ctx, [_dtypes.float32, _dtypes.float64, ])
  (x, q) = _inputs_T
  _inputs_flat = [x, q]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Zeta", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Zeta", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

