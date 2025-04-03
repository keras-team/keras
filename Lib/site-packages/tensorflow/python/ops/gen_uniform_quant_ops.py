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

TV_UniformDequantize_Tin = TypeVar("TV_UniformDequantize_Tin", _atypes.QInt32, _atypes.QInt8)
TV_UniformDequantize_Tout = TypeVar("TV_UniformDequantize_Tout", bound=_atypes.Float32)

def uniform_dequantize(input: Annotated[Any, TV_UniformDequantize_Tin], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformDequantize_Tout, quantization_min_val: int, quantization_max_val: int, quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformDequantize_Tout]:
  r"""Perform dequantization on the quantized Tensor `input`.

  Given quantized `input` which was quantized using `scales` and `zero_points`, performs dequantization using the formula:
  dequantized_data = (quantized_data - zero_point) * scale.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
      Must be a Tensor of Tin.
    scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing original data that input represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing original data that input represents.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.float32`.
      The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
    quantization_min_val: An `int`.
      The quantization min value that was used when input was quantized.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    quantization_max_val: An `int`.
      The quantization max value that was used when input was quantized.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformDequantize", name, input, scales, zero_points, "Tout",
        Tout, "quantization_axis", quantization_axis, "quantization_min_val",
        quantization_min_val, "quantization_max_val", quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_dequantize_eager_fallback(
          input, scales, zero_points, Tout=Tout,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformDequantize", input=input, scales=scales,
                             zero_points=zero_points, Tout=Tout,
                             quantization_min_val=quantization_min_val,
                             quantization_max_val=quantization_max_val,
                             quantization_axis=quantization_axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformDequantize = tf_export("raw_ops.UniformDequantize")(_ops.to_raw_op(uniform_dequantize))


def uniform_dequantize_eager_fallback(input: Annotated[Any, TV_UniformDequantize_Tin], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformDequantize_Tout, quantization_min_val: int, quantization_max_val: int, quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformDequantize_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.qint32, ])
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [input, scales, zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "quantization_axis",
  quantization_axis, "quantization_min_val", quantization_min_val,
  "quantization_max_val", quantization_max_val)
  _result = _execute.execute(b"UniformDequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantize_Tin = TypeVar("TV_UniformQuantize_Tin", bound=_atypes.Float32)
TV_UniformQuantize_Tout = TypeVar("TV_UniformQuantize_Tout", _atypes.QInt32, _atypes.QInt8)

def uniform_quantize(input: Annotated[Any, TV_UniformQuantize_Tin], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantize_Tout, quantization_min_val: int, quantization_max_val: int, quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantize_Tout]:
  r"""Perform quantization on Tensor `input`.

  Given `input`, `scales` and `zero_points`, performs quantization using the formula:
  quantized_data = floor(input_data * (1.0f / scale) + 0.5f) + zero_point

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      Must be a Tensor of Tin.
    scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale(s) to quantize `input`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as zero_point(s) to quantize `input`.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.float32
    quantization_min_val: An `int`.
      The quantization min value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    quantization_max_val: An `int`.
      The quantization max value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantize", name, input, scales, zero_points, "Tout",
        Tout, "quantization_axis", quantization_axis, "quantization_min_val",
        quantization_min_val, "quantization_max_val", quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantize_eager_fallback(
          input, scales, zero_points, Tout=Tout,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantize", input=input, scales=scales,
                           zero_points=zero_points, Tout=Tout,
                           quantization_min_val=quantization_min_val,
                           quantization_max_val=quantization_max_val,
                           quantization_axis=quantization_axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantize = tf_export("raw_ops.UniformQuantize")(_ops.to_raw_op(uniform_quantize))


def uniform_quantize_eager_fallback(input: Annotated[Any, TV_UniformQuantize_Tin], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantize_Tout, quantization_min_val: int, quantization_max_val: int, quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantize_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, ])
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [input, scales, zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "quantization_axis",
  quantization_axis, "quantization_min_val", quantization_min_val,
  "quantization_max_val", quantization_max_val)
  _result = _execute.execute(b"UniformQuantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedAdd_T = TypeVar("TV_UniformQuantizedAdd_T", bound=_atypes.QInt32)

def uniform_quantized_add(lhs: Annotated[Any, TV_UniformQuantizedAdd_T], rhs: Annotated[Any, TV_UniformQuantizedAdd_T], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis:int=-1, rhs_quantization_axis:int=-1, output_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedAdd_T]:
  r"""Perform quantized add of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized add on `lhs` and `rhs` to make quantized `output`.

  `UniformQuantizedAdd` follows Numpy broadcasting rules.
  The two input array shapes are compared element-wise.
  Starting with the trailing dimensions, the two dimensions either have to be equal or one of them needs to be 1.

  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  ```
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val)
  ```
  `output` is also quantized, using the same formula.

  If `lhs` and `output` is both per-axis quantized, the quantization axis must match.
  Also, if `rhs` and `output` is both per-axis quantized, the quantization axis must match.
  *Match* means the axis must match when adding, regarding the broadcasting.
  i.e. For both operands `lhs` and `rhs`,
  if `operand.quantization_axis` >= 0 and `output.quantization_axis` >= 0,
  `operand.dims` - `operand.quantization_axis` must be equal to `output.dims` - `output.quantization_axis`.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint32`.
      Must be a quantized tensor.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a quantized tensor.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `lhs` represents.
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `lhs` represents.
      Must have same shape with `lhs_scales`.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `rhs` represents.
      Must have same shape with `rhs_scales`.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale factors when quantizing original data that `output` represents.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that output represents.
      Must have same shape with `output_scales`.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in `output`.
      For example, if  `Tout` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in `output`.
      For example, if `Tout` is `qint8`, this must be set to 127.
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `lhs`, only per-tensor quantization is supported.
      Thus, this must be set to -1.
      Other values will raise error at OpKernel construction.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `rhs`, only per-tensor quantization
      or per-channel quantization along `kernel_output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `output`, only per-tensor quantization or per-channel quantization along `output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedAdd", name, lhs, rhs, lhs_scales,
        lhs_zero_points, rhs_scales, rhs_zero_points, output_scales,
        output_zero_points, "lhs_quantization_axis", lhs_quantization_axis,
        "lhs_quantization_min_val", lhs_quantization_min_val,
        "lhs_quantization_max_val", lhs_quantization_max_val,
        "rhs_quantization_axis", rhs_quantization_axis,
        "rhs_quantization_min_val", rhs_quantization_min_val,
        "rhs_quantization_max_val", rhs_quantization_max_val,
        "output_quantization_axis", output_quantization_axis,
        "output_quantization_min_val", output_quantization_min_val,
        "output_quantization_max_val", output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_add_eager_fallback(
          lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points,
          output_scales, output_zero_points,
          lhs_quantization_axis=lhs_quantization_axis,
          lhs_quantization_min_val=lhs_quantization_min_val,
          lhs_quantization_max_val=lhs_quantization_max_val,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedAdd", lhs=lhs, rhs=rhs, lhs_scales=lhs_scales,
                               lhs_zero_points=lhs_zero_points,
                               rhs_scales=rhs_scales,
                               rhs_zero_points=rhs_zero_points,
                               output_scales=output_scales,
                               output_zero_points=output_zero_points,
                               lhs_quantization_min_val=lhs_quantization_min_val,
                               lhs_quantization_max_val=lhs_quantization_max_val,
                               rhs_quantization_min_val=rhs_quantization_min_val,
                               rhs_quantization_max_val=rhs_quantization_max_val,
                               output_quantization_min_val=output_quantization_min_val,
                               output_quantization_max_val=output_quantization_max_val,
                               lhs_quantization_axis=lhs_quantization_axis,
                               rhs_quantization_axis=rhs_quantization_axis,
                               output_quantization_axis=output_quantization_axis,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("lhs_quantization_axis",
              _op._get_attr_int("lhs_quantization_axis"),
              "lhs_quantization_min_val",
              _op._get_attr_int("lhs_quantization_min_val"),
              "lhs_quantization_max_val",
              _op._get_attr_int("lhs_quantization_max_val"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedAdd = tf_export("raw_ops.UniformQuantizedAdd")(_ops.to_raw_op(uniform_quantized_add))


def uniform_quantized_add_eager_fallback(lhs: Annotated[Any, TV_UniformQuantizedAdd_T], rhs: Annotated[Any, TV_UniformQuantizedAdd_T], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis: int, rhs_quantization_axis: int, output_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedAdd_T]:
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint32, ])
  (lhs, rhs) = _inputs_T
  lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
  lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
  _attrs = ("lhs_quantization_axis", lhs_quantization_axis,
  "lhs_quantization_min_val", lhs_quantization_min_val,
  "lhs_quantization_max_val", lhs_quantization_max_val,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val, "T", _attr_T)
  _result = _execute.execute(b"UniformQuantizedAdd", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedClipByValue_T = TypeVar("TV_UniformQuantizedClipByValue_T", bound=_atypes.QInt32)

def uniform_quantized_clip_by_value(operand: Annotated[Any, TV_UniformQuantizedClipByValue_T], min: Annotated[Any, TV_UniformQuantizedClipByValue_T], max: Annotated[Any, TV_UniformQuantizedClipByValue_T], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], quantization_min_val: int, quantization_max_val: int, quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedClipByValue_T]:
  r"""Perform clip by value on the quantized Tensor `operand`.

  Given quantized `operand` which was quantized using `scales` and `zero_points`, performs clip by value using `min` and `max` values.
  If quantization_axis is -1 (per-tensor quantized), the entire operand is clipped using scalar min, max.
  Otherwise (per-channel quantized), the clipping is also done per-channel.

  Args:
    operand: A `Tensor`. Must be one of the following types: `qint32`.
      Must be a Tensor of T.
    min: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    max: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing `operand`, `min` and `max`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing `operand`, `min` and `max`.
      Same shape condition as scales.
    quantization_min_val: An `int`.
      The quantization min value that was used when operand was quantized.
    quantization_max_val: An `int`.
      The quantization max value that was used when operand was quantized.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, operand.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedClipByValue", name, operand, min, max, scales,
        zero_points, "quantization_axis", quantization_axis,
        "quantization_min_val", quantization_min_val, "quantization_max_val",
        quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_clip_by_value_eager_fallback(
          operand, min, max, scales, zero_points,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedClipByValue", operand=operand, min=min, max=max,
                                       scales=scales, zero_points=zero_points,
                                       quantization_min_val=quantization_min_val,
                                       quantization_max_val=quantization_max_val,
                                       quantization_axis=quantization_axis,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedClipByValue = tf_export("raw_ops.UniformQuantizedClipByValue")(_ops.to_raw_op(uniform_quantized_clip_by_value))


def uniform_quantized_clip_by_value_eager_fallback(operand: Annotated[Any, TV_UniformQuantizedClipByValue_T], min: Annotated[Any, TV_UniformQuantizedClipByValue_T], max: Annotated[Any, TV_UniformQuantizedClipByValue_T], scales: Annotated[Any, _atypes.Float32], zero_points: Annotated[Any, _atypes.Int32], quantization_min_val: int, quantization_max_val: int, quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedClipByValue_T]:
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, min, max], ctx, [_dtypes.qint32, ])
  (operand, min, max) = _inputs_T
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [operand, min, max, scales, zero_points]
  _attrs = ("T", _attr_T, "quantization_axis", quantization_axis,
  "quantization_min_val", quantization_min_val, "quantization_max_val",
  quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedClipByValue", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedConvolution_Tin = TypeVar("TV_UniformQuantizedConvolution_Tin", bound=_atypes.QInt8)
TV_UniformQuantizedConvolution_Tout = TypeVar("TV_UniformQuantizedConvolution_Tout", bound=_atypes.QInt32)

def uniform_quantized_convolution(lhs: Annotated[Any, TV_UniformQuantizedConvolution_Tin], rhs: Annotated[Any, TV_UniformQuantizedConvolution_Tin], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedConvolution_Tout, padding: str, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, window_strides=[], explicit_padding=[], lhs_dilation=[], rhs_dilation=[], batch_group_count:int=1, feature_group_count:int=1, dimension_numbers:str="", lhs_quantization_axis:int=-1, rhs_quantization_axis:int=-1, output_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedConvolution_Tout]:
  r"""Perform quantized convolution of quantized Tensor `lhs` and quantized Tensor `rhs`. to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized dot on `lhs` and `rhs` to make quantized `output`.

  `lhs` and `rhs` must be Tensors of same rank, and meet following shape conditions.
  - `lhs_feature` % `feature_group_count` == 0
  - `lhs_feature` % `rhs_input_feature` == 0
  - `lhs_feature` / `feature_group_count` == `rhs_input_feature`
  - `rhs_output_feature` % `feature_group_count` == 0
  - `lhs_batch` % `batch_group_count` == 0
  - `rhs_output_feature` % `batch_group_count` == 0

  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  ```
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val)
  ```
  `output` is also quantized, using the same formula.
  If `rhs` is per-tensor quantized, `output` must be also per-tensor quantized.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a quantized tensor, rank >= 3.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a quantized tensor, same rank as `lhs`.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `lhs` represents.
      Must be a scalar `Tensor` (`lhs` supports only per-tensor quantization).
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `lhs` represents.
      Same shape condition as `lhs_scales`.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
      Must be a scalar `Tensor` for per-tensor quantization,
      or 1D `Tensor` of size `rhs.dim_size(kernel_output_feature_dimension)`, for per-channel quantization.
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `rhs` represents.
      Same shape condition as `rhs_scales`.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale factors when quantizing original data that `output` represents.
      Must be a scalar `Tensor` for per-tensor quantization,
      or 1D `Tensor` of size `rhs.dim_size(kernel_output_feature_dimension)`
      - which is equal to `output.dim_size(output_feature_dimension)`,
      for per-channel quantization.
      If `rhs` is per-tensor quantized, output must be also per-tensor quantized.
      This means that if `rhs_scales` and `rhs_zero_points` are scalar `Tensor`s, `output_scales` and `output_zero_points` must be scalar `Tensor`s as well.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that output represents.
      Same shape condition as `output_scales`.
    Tout: A `tf.DType` from: `tf.qint32`. The type of `output` `Tensor`.
    padding: A `string`.
      string from: `"SAME"`, `"VALID"`, or `"EXPLICIT"`, indicating the type of padding algorithm to use.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in `output`.
      For example, if  `Tout` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in `output`.
      For example, if `Tout` is `qint8`, this must be set to 127.
    window_strides: An optional list of `ints`. Defaults to `[]`.
      The stride of the sliding window for each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of spatial dimensions).
      If an empty list is provided, the stride for each spatial dimension is set to 1.
    explicit_padding: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, must be set as a list indicating
      the explicit paddings at the start and end of each `lhs` spatial dimension.
      Otherwise, this must be empty.

      (If used,) Must be a list of size `2 * (number of lhs spatial dimensions)`,
      where `(explicit_padding[2 * i], explicit_padding[2 * i + 1])` indicates
      `(start_padding, end_padding)` of `spatial_dimensions[i]`.
    lhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of `lhs` spatial dimensions).
      If empty list, the dilation for each `lhs` spatial dimension is set to 1.
    rhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `rhs`.
      Must be an empty list (default) or a list of size (number of `rhs` spatial dimensions).
      If empty list, the dilation for each `rhs` spatial dimension is set to 1.
    batch_group_count: An optional `int`. Defaults to `1`.
      The number of batch groups. Used for grouped filters.
      Must be a divisor of `output_feature`.
    feature_group_count: An optional `int`. Defaults to `1`.
      The number of feature groups. Used for grouped convolutions.
      Must be a divisor of both `lhs_feature` and `output_feature`.
    dimension_numbers: An optional `string`. Defaults to `""`.
      Structure of dimension information for the convolution op.
      Must be an empty string (default) or a serialized string of `tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr` proto.
      If empty string, the default is `("NCHW", "OIHW", "NCHW")` (for a 2D convolution).
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `lhs`, only per-tensor quantization is supported.
      Thus, this must be set to -1.
      Other values will raise error at OpKernel construction.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `rhs`, only per-tensor quantization
      or per-channel quantization along `kernel_output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `output`, only per-tensor quantization or per-channel quantization along `output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedConvolution", name, lhs, rhs, lhs_scales,
        lhs_zero_points, rhs_scales, rhs_zero_points, output_scales,
        output_zero_points, "Tout", Tout, "window_strides", window_strides,
        "padding", padding, "explicit_padding", explicit_padding,
        "lhs_dilation", lhs_dilation, "rhs_dilation", rhs_dilation,
        "batch_group_count", batch_group_count, "feature_group_count",
        feature_group_count, "dimension_numbers", dimension_numbers,
        "lhs_quantization_axis", lhs_quantization_axis,
        "lhs_quantization_min_val", lhs_quantization_min_val,
        "lhs_quantization_max_val", lhs_quantization_max_val,
        "rhs_quantization_axis", rhs_quantization_axis,
        "rhs_quantization_min_val", rhs_quantization_min_val,
        "rhs_quantization_max_val", rhs_quantization_max_val,
        "output_quantization_axis", output_quantization_axis,
        "output_quantization_min_val", output_quantization_min_val,
        "output_quantization_max_val", output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_convolution_eager_fallback(
          lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points,
          output_scales, output_zero_points, Tout=Tout,
          window_strides=window_strides, padding=padding,
          explicit_padding=explicit_padding, lhs_dilation=lhs_dilation,
          rhs_dilation=rhs_dilation, batch_group_count=batch_group_count,
          feature_group_count=feature_group_count,
          dimension_numbers=dimension_numbers,
          lhs_quantization_axis=lhs_quantization_axis,
          lhs_quantization_min_val=lhs_quantization_min_val,
          lhs_quantization_max_val=lhs_quantization_max_val,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  padding = _execute.make_str(padding, "padding")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if window_strides is None:
    window_strides = []
  if not isinstance(window_strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'window_strides' argument to "
        "'uniform_quantized_convolution' Op, not %r." % window_strides)
  window_strides = [_execute.make_int(_i, "window_strides") for _i in window_strides]
  if explicit_padding is None:
    explicit_padding = []
  if not isinstance(explicit_padding, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_padding' argument to "
        "'uniform_quantized_convolution' Op, not %r." % explicit_padding)
  explicit_padding = [_execute.make_int(_i, "explicit_padding") for _i in explicit_padding]
  if lhs_dilation is None:
    lhs_dilation = []
  if not isinstance(lhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'lhs_dilation' argument to "
        "'uniform_quantized_convolution' Op, not %r." % lhs_dilation)
  lhs_dilation = [_execute.make_int(_i, "lhs_dilation") for _i in lhs_dilation]
  if rhs_dilation is None:
    rhs_dilation = []
  if not isinstance(rhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'rhs_dilation' argument to "
        "'uniform_quantized_convolution' Op, not %r." % rhs_dilation)
  rhs_dilation = [_execute.make_int(_i, "rhs_dilation") for _i in rhs_dilation]
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  if feature_group_count is None:
    feature_group_count = 1
  feature_group_count = _execute.make_int(feature_group_count, "feature_group_count")
  if dimension_numbers is None:
    dimension_numbers = ""
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedConvolution", lhs=lhs, rhs=rhs,
                                       lhs_scales=lhs_scales,
                                       lhs_zero_points=lhs_zero_points,
                                       rhs_scales=rhs_scales,
                                       rhs_zero_points=rhs_zero_points,
                                       output_scales=output_scales,
                                       output_zero_points=output_zero_points,
                                       Tout=Tout, padding=padding,
                                       lhs_quantization_min_val=lhs_quantization_min_val,
                                       lhs_quantization_max_val=lhs_quantization_max_val,
                                       rhs_quantization_min_val=rhs_quantization_min_val,
                                       rhs_quantization_max_val=rhs_quantization_max_val,
                                       output_quantization_min_val=output_quantization_min_val,
                                       output_quantization_max_val=output_quantization_max_val,
                                       window_strides=window_strides,
                                       explicit_padding=explicit_padding,
                                       lhs_dilation=lhs_dilation,
                                       rhs_dilation=rhs_dilation,
                                       batch_group_count=batch_group_count,
                                       feature_group_count=feature_group_count,
                                       dimension_numbers=dimension_numbers,
                                       lhs_quantization_axis=lhs_quantization_axis,
                                       rhs_quantization_axis=rhs_quantization_axis,
                                       output_quantization_axis=output_quantization_axis,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "window_strides",
              _op.get_attr("window_strides"), "padding",
              _op.get_attr("padding"), "explicit_padding",
              _op.get_attr("explicit_padding"), "lhs_dilation",
              _op.get_attr("lhs_dilation"), "rhs_dilation",
              _op.get_attr("rhs_dilation"), "batch_group_count",
              _op._get_attr_int("batch_group_count"), "feature_group_count",
              _op._get_attr_int("feature_group_count"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "lhs_quantization_axis",
              _op._get_attr_int("lhs_quantization_axis"),
              "lhs_quantization_min_val",
              _op._get_attr_int("lhs_quantization_min_val"),
              "lhs_quantization_max_val",
              _op._get_attr_int("lhs_quantization_max_val"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedConvolution", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedConvolution = tf_export("raw_ops.UniformQuantizedConvolution")(_ops.to_raw_op(uniform_quantized_convolution))


def uniform_quantized_convolution_eager_fallback(lhs: Annotated[Any, TV_UniformQuantizedConvolution_Tin], rhs: Annotated[Any, TV_UniformQuantizedConvolution_Tin], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedConvolution_Tout, padding: str, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, window_strides, explicit_padding, lhs_dilation, rhs_dilation, batch_group_count: int, feature_group_count: int, dimension_numbers: str, lhs_quantization_axis: int, rhs_quantization_axis: int, output_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedConvolution_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  padding = _execute.make_str(padding, "padding")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if window_strides is None:
    window_strides = []
  if not isinstance(window_strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'window_strides' argument to "
        "'uniform_quantized_convolution' Op, not %r." % window_strides)
  window_strides = [_execute.make_int(_i, "window_strides") for _i in window_strides]
  if explicit_padding is None:
    explicit_padding = []
  if not isinstance(explicit_padding, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_padding' argument to "
        "'uniform_quantized_convolution' Op, not %r." % explicit_padding)
  explicit_padding = [_execute.make_int(_i, "explicit_padding") for _i in explicit_padding]
  if lhs_dilation is None:
    lhs_dilation = []
  if not isinstance(lhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'lhs_dilation' argument to "
        "'uniform_quantized_convolution' Op, not %r." % lhs_dilation)
  lhs_dilation = [_execute.make_int(_i, "lhs_dilation") for _i in lhs_dilation]
  if rhs_dilation is None:
    rhs_dilation = []
  if not isinstance(rhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'rhs_dilation' argument to "
        "'uniform_quantized_convolution' Op, not %r." % rhs_dilation)
  rhs_dilation = [_execute.make_int(_i, "rhs_dilation") for _i in rhs_dilation]
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  if feature_group_count is None:
    feature_group_count = 1
  feature_group_count = _execute.make_int(feature_group_count, "feature_group_count")
  if dimension_numbers is None:
    dimension_numbers = ""
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_Tin, _inputs_Tin = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint8, ])
  (lhs, rhs) = _inputs_Tin
  lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
  lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "window_strides", window_strides,
  "padding", padding, "explicit_padding", explicit_padding, "lhs_dilation",
  lhs_dilation, "rhs_dilation", rhs_dilation, "batch_group_count",
  batch_group_count, "feature_group_count", feature_group_count,
  "dimension_numbers", dimension_numbers, "lhs_quantization_axis",
  lhs_quantization_axis, "lhs_quantization_min_val", lhs_quantization_min_val,
  "lhs_quantization_max_val", lhs_quantization_max_val,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedConvolution", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedConvolution", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedConvolutionHybrid_Tlhs = TypeVar("TV_UniformQuantizedConvolutionHybrid_Tlhs", bound=_atypes.Float32)
TV_UniformQuantizedConvolutionHybrid_Trhs = TypeVar("TV_UniformQuantizedConvolutionHybrid_Trhs", bound=_atypes.QInt8)
TV_UniformQuantizedConvolutionHybrid_Tout = TypeVar("TV_UniformQuantizedConvolutionHybrid_Tout", bound=_atypes.Float32)

def uniform_quantized_convolution_hybrid(lhs: Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Tlhs], rhs: Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Trhs], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedConvolutionHybrid_Tout, padding: str, rhs_quantization_min_val: int, rhs_quantization_max_val: int, window_strides=[], explicit_padding=[], lhs_dilation=[], rhs_dilation=[], batch_group_count:int=1, feature_group_count:int=1, dimension_numbers:str="", rhs_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Tout]:
  r"""Perform hybrid quantized convolution of float Tensor `lhs` and quantized Tensor `rhs`.

  Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`,
  and then performs quantized convolution on quantized `lhs` and `rhs`.

  The internal quantization on `lhs` is a quantization to `Trhs`, dynamic range,
  per-batch (per-axis along axis `dimension_numbers.input_batch_dimension`), asymmetric,
  and not narrow range (the range is [Trhs_MIN, Trhs_MAX]).

  `lhs` and `rhs` must be Tensors of same rank, and meet following shape conditions.
  - lhs_feature % feature_group_count == 0
  - lhs_feature % rhs_input_feature == 0
  - lhs_feature / feature_group_count == rhs_input_feature
  - rhs_output_feature % feature_group_count == 0
  - lhs_batch % batch_group_count == 0
  - rhs_output_feature % batch_group_count == 0

  `rhs` must be quantized Tensor, where its data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`.
      Must be a non-quantized Tensor of `Tlhs`, rank >= 3.
    rhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a quantized Tensor of `Trhs`, same rank as `lhs`.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
      Must be a scalar Tensor for per-tensor quantization,
      or 1D Tensor of size `rhs.dim_size(kernel_output_feature_dimension)`, for per-channel quantization.
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that `rhs` represents.
      Same shape condition as `rhs_scales`.
    Tout: A `tf.DType` from: `tf.float32`. The type of output Tensor.
    padding: A `string`.
      string from: `"SAME"`, `"VALID"`, or `"EXPLICIT"`, indicating the type of padding algorithm to use.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `rhs`.
      For example, if `Trhs` is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `rhs`.
      For example, if `Trhs` is qint8, this must be set to 127.
    window_strides: An optional list of `ints`. Defaults to `[]`.
      The stride of the sliding window for each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of spatial dimensions).
      If an empty list is provided, the stride for each spatial dimension is set to 1.
    explicit_padding: An optional list of `ints`. Defaults to `[]`.
      If `padding` Attr is `"EXPLICIT"`, must be set as a list indicating
      the explicit paddings at the start and end of each lhs spatial dimension.
      Otherwise, this Attr is must be empty.

      (If used,) Must be a list of size 2 * (number of lhs spatial dimensions),
      where (explicit_padding[2 * i], explicit_padding[2 * i + 1]) indicates
      spatial_dimensions[i] (start_padding, end_padding).
    lhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of lhs spatial dimensions).
      If empty list, the dilation for each lhs spatial dimension is set to 1.
    rhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `rhs`.
      Must be an empty list (default) or a list of size (number of rhs spatial dimensions).
      If empty list, the dilation for each rhs spatial dimension is set to 1.
    batch_group_count: An optional `int`. Defaults to `1`.
      The number of batch groups. Used for grouped filters.
      Must be a divisor of output_feature.
    feature_group_count: An optional `int`. Defaults to `1`.
      The number of feature groups. Used for grouped convolutions.
      Must be a divisor of both lhs_feature and output_feature.
    dimension_numbers: An optional `string`. Defaults to `""`.
      Structure of dimension information for the convolution op.
      Must be an empty string (default) or a serialized string of tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr proto.
      If empty string, the default is `("NCHW", "OIHW", "NCHW")` (for a 2D convolution).
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `rhs`, only per-tensor quantization
      or per-channel quantization along kernel_output_feature_dimension is supported.
      Thus, this attribute must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedConvolutionHybrid", name, lhs, rhs, rhs_scales,
        rhs_zero_points, "Tout", Tout, "window_strides", window_strides,
        "padding", padding, "explicit_padding", explicit_padding,
        "lhs_dilation", lhs_dilation, "rhs_dilation", rhs_dilation,
        "batch_group_count", batch_group_count, "feature_group_count",
        feature_group_count, "dimension_numbers", dimension_numbers,
        "rhs_quantization_axis", rhs_quantization_axis,
        "rhs_quantization_min_val", rhs_quantization_min_val,
        "rhs_quantization_max_val", rhs_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_convolution_hybrid_eager_fallback(
          lhs, rhs, rhs_scales, rhs_zero_points, Tout=Tout,
          window_strides=window_strides, padding=padding,
          explicit_padding=explicit_padding, lhs_dilation=lhs_dilation,
          rhs_dilation=rhs_dilation, batch_group_count=batch_group_count,
          feature_group_count=feature_group_count,
          dimension_numbers=dimension_numbers,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  padding = _execute.make_str(padding, "padding")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if window_strides is None:
    window_strides = []
  if not isinstance(window_strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'window_strides' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % window_strides)
  window_strides = [_execute.make_int(_i, "window_strides") for _i in window_strides]
  if explicit_padding is None:
    explicit_padding = []
  if not isinstance(explicit_padding, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_padding' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % explicit_padding)
  explicit_padding = [_execute.make_int(_i, "explicit_padding") for _i in explicit_padding]
  if lhs_dilation is None:
    lhs_dilation = []
  if not isinstance(lhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'lhs_dilation' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % lhs_dilation)
  lhs_dilation = [_execute.make_int(_i, "lhs_dilation") for _i in lhs_dilation]
  if rhs_dilation is None:
    rhs_dilation = []
  if not isinstance(rhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'rhs_dilation' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % rhs_dilation)
  rhs_dilation = [_execute.make_int(_i, "rhs_dilation") for _i in rhs_dilation]
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  if feature_group_count is None:
    feature_group_count = 1
  feature_group_count = _execute.make_int(feature_group_count, "feature_group_count")
  if dimension_numbers is None:
    dimension_numbers = ""
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedConvolutionHybrid", lhs=lhs, rhs=rhs,
                                             rhs_scales=rhs_scales,
                                             rhs_zero_points=rhs_zero_points,
                                             Tout=Tout, padding=padding,
                                             rhs_quantization_min_val=rhs_quantization_min_val,
                                             rhs_quantization_max_val=rhs_quantization_max_val,
                                             window_strides=window_strides,
                                             explicit_padding=explicit_padding,
                                             lhs_dilation=lhs_dilation,
                                             rhs_dilation=rhs_dilation,
                                             batch_group_count=batch_group_count,
                                             feature_group_count=feature_group_count,
                                             dimension_numbers=dimension_numbers,
                                             rhs_quantization_axis=rhs_quantization_axis,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tlhs", _op._get_attr_type("Tlhs"), "Trhs",
              _op._get_attr_type("Trhs"), "Tout", _op._get_attr_type("Tout"),
              "window_strides", _op.get_attr("window_strides"), "padding",
              _op.get_attr("padding"), "explicit_padding",
              _op.get_attr("explicit_padding"), "lhs_dilation",
              _op.get_attr("lhs_dilation"), "rhs_dilation",
              _op.get_attr("rhs_dilation"), "batch_group_count",
              _op._get_attr_int("batch_group_count"), "feature_group_count",
              _op._get_attr_int("feature_group_count"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedConvolutionHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedConvolutionHybrid = tf_export("raw_ops.UniformQuantizedConvolutionHybrid")(_ops.to_raw_op(uniform_quantized_convolution_hybrid))


def uniform_quantized_convolution_hybrid_eager_fallback(lhs: Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Tlhs], rhs: Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Trhs], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedConvolutionHybrid_Tout, padding: str, rhs_quantization_min_val: int, rhs_quantization_max_val: int, window_strides, explicit_padding, lhs_dilation, rhs_dilation, batch_group_count: int, feature_group_count: int, dimension_numbers: str, rhs_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedConvolutionHybrid_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  padding = _execute.make_str(padding, "padding")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if window_strides is None:
    window_strides = []
  if not isinstance(window_strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'window_strides' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % window_strides)
  window_strides = [_execute.make_int(_i, "window_strides") for _i in window_strides]
  if explicit_padding is None:
    explicit_padding = []
  if not isinstance(explicit_padding, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_padding' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % explicit_padding)
  explicit_padding = [_execute.make_int(_i, "explicit_padding") for _i in explicit_padding]
  if lhs_dilation is None:
    lhs_dilation = []
  if not isinstance(lhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'lhs_dilation' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % lhs_dilation)
  lhs_dilation = [_execute.make_int(_i, "lhs_dilation") for _i in lhs_dilation]
  if rhs_dilation is None:
    rhs_dilation = []
  if not isinstance(rhs_dilation, (list, tuple)):
    raise TypeError(
        "Expected list for 'rhs_dilation' argument to "
        "'uniform_quantized_convolution_hybrid' Op, not %r." % rhs_dilation)
  rhs_dilation = [_execute.make_int(_i, "rhs_dilation") for _i in rhs_dilation]
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  if feature_group_count is None:
    feature_group_count = 1
  feature_group_count = _execute.make_int(feature_group_count, "feature_group_count")
  if dimension_numbers is None:
    dimension_numbers = ""
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _attr_Tlhs, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, ])
  _attr_Trhs, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.qint8, ])
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, rhs_scales, rhs_zero_points]
  _attrs = ("Tlhs", _attr_Tlhs, "Trhs", _attr_Trhs, "Tout", Tout,
  "window_strides", window_strides, "padding", padding, "explicit_padding",
  explicit_padding, "lhs_dilation", lhs_dilation, "rhs_dilation",
  rhs_dilation, "batch_group_count", batch_group_count, "feature_group_count",
  feature_group_count, "dimension_numbers", dimension_numbers,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedConvolutionHybrid", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedConvolutionHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedDot_Tin = TypeVar("TV_UniformQuantizedDot_Tin", bound=_atypes.QInt8)
TV_UniformQuantizedDot_Tout = TypeVar("TV_UniformQuantizedDot_Tout", bound=_atypes.QInt32)

def uniform_quantized_dot(lhs: Annotated[Any, TV_UniformQuantizedDot_Tin], rhs: Annotated[Any, TV_UniformQuantizedDot_Tin], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedDot_Tout, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis:int=-1, rhs_quantization_axis:int=-1, output_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedDot_Tout]:
  r"""Perform quantized dot of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized dot on `lhs` and `rhs` to make quantized `output`.
  `lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).
  `output` is also quantized, using the same formula.
  If `rhs` is per-tensor quantized, `output` must be also per-tensor quantized.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a 2D Tensor of Tin.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a 2D Tensor of Tin.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that lhs represents.
      Must be a scalar Tensor (lhs supports only per-tensor quantization).
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that lhs represents.
      Same shape condition as lhs_scales.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that rhs represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that rhs represents.
      Same shape condition as rhs_scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scales when quantizing original data that output represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (output.dim_size(1),) (per-channel quantization).
      If rhs is per-tensor quantized, output must be also per-tensor quantized.
      This means that if rhs_scales and rhs_zero_points are scalar Tensors, output_scales and output_zero_points must be scalar Tensors as well.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that output represents.
      Same shape condition as rhs_scales.
    Tout: A `tf.DType` from: `tf.qint32`. The type of output Tensor.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in lhs.
      For example, if Tin is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Tin is qint8, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to 127.
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op lhs, only per-tensor quantization is supported.
      Thus, this attribute must be set to -1. Other values are rejected.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op output, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedDot", name, lhs, rhs, lhs_scales,
        lhs_zero_points, rhs_scales, rhs_zero_points, output_scales,
        output_zero_points, "Tout", Tout, "lhs_quantization_axis",
        lhs_quantization_axis, "lhs_quantization_min_val",
        lhs_quantization_min_val, "lhs_quantization_max_val",
        lhs_quantization_max_val, "rhs_quantization_axis",
        rhs_quantization_axis, "rhs_quantization_min_val",
        rhs_quantization_min_val, "rhs_quantization_max_val",
        rhs_quantization_max_val, "output_quantization_axis",
        output_quantization_axis, "output_quantization_min_val",
        output_quantization_min_val, "output_quantization_max_val",
        output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_dot_eager_fallback(
          lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points,
          output_scales, output_zero_points, Tout=Tout,
          lhs_quantization_axis=lhs_quantization_axis,
          lhs_quantization_min_val=lhs_quantization_min_val,
          lhs_quantization_max_val=lhs_quantization_max_val,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedDot", lhs=lhs, rhs=rhs, lhs_scales=lhs_scales,
                               lhs_zero_points=lhs_zero_points,
                               rhs_scales=rhs_scales,
                               rhs_zero_points=rhs_zero_points,
                               output_scales=output_scales,
                               output_zero_points=output_zero_points,
                               Tout=Tout,
                               lhs_quantization_min_val=lhs_quantization_min_val,
                               lhs_quantization_max_val=lhs_quantization_max_val,
                               rhs_quantization_min_val=rhs_quantization_min_val,
                               rhs_quantization_max_val=rhs_quantization_max_val,
                               output_quantization_min_val=output_quantization_min_val,
                               output_quantization_max_val=output_quantization_max_val,
                               lhs_quantization_axis=lhs_quantization_axis,
                               rhs_quantization_axis=rhs_quantization_axis,
                               output_quantization_axis=output_quantization_axis,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "lhs_quantization_axis",
              _op._get_attr_int("lhs_quantization_axis"),
              "lhs_quantization_min_val",
              _op._get_attr_int("lhs_quantization_min_val"),
              "lhs_quantization_max_val",
              _op._get_attr_int("lhs_quantization_max_val"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedDot = tf_export("raw_ops.UniformQuantizedDot")(_ops.to_raw_op(uniform_quantized_dot))


def uniform_quantized_dot_eager_fallback(lhs: Annotated[Any, TV_UniformQuantizedDot_Tin], rhs: Annotated[Any, TV_UniformQuantizedDot_Tin], lhs_scales: Annotated[Any, _atypes.Float32], lhs_zero_points: Annotated[Any, _atypes.Int32], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedDot_Tout, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis: int, rhs_quantization_axis: int, output_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedDot_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_Tin, _inputs_Tin = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint8, ])
  (lhs, rhs) = _inputs_Tin
  lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
  lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "lhs_quantization_axis",
  lhs_quantization_axis, "lhs_quantization_min_val", lhs_quantization_min_val,
  "lhs_quantization_max_val", lhs_quantization_max_val,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedDot", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformQuantizedDotHybrid_Tlhs = TypeVar("TV_UniformQuantizedDotHybrid_Tlhs", bound=_atypes.Float32)
TV_UniformQuantizedDotHybrid_Trhs = TypeVar("TV_UniformQuantizedDotHybrid_Trhs", bound=_atypes.QInt8)
TV_UniformQuantizedDotHybrid_Tout = TypeVar("TV_UniformQuantizedDotHybrid_Tout", bound=_atypes.Float32)

def uniform_quantized_dot_hybrid(lhs: Annotated[Any, TV_UniformQuantizedDotHybrid_Tlhs], rhs: Annotated[Any, TV_UniformQuantizedDotHybrid_Trhs], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedDotHybrid_Tout, rhs_quantization_min_val: int, rhs_quantization_max_val: int, rhs_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformQuantizedDotHybrid_Tout]:
  r"""Perform hybrid quantized dot of float Tensor `lhs` and quantized Tensor `rhs`.

  Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`, and then performs quantized dot on quantized lhs and `rhs`.
  The internal quantization on `lhs` is a quantization to qint8, dynamic range, per-batch (per-axis along axis 0), asymmetric, and not narrow range (the range is [-128, 127]).
  `lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
  `rhs` must be quantized Tensor, where its data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`.
      Must be a 2D Tensor of Tlhs.
    rhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a 2D Tensor of Trhs.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that rhs represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that rhs represents.
      Same shape condition as rhs_scales.
    Tout: A `tf.DType` from: `tf.float32`. The type of output Tensor.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to 127.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedDotHybrid", name, lhs, rhs, rhs_scales,
        rhs_zero_points, "Tout", Tout, "rhs_quantization_axis",
        rhs_quantization_axis, "rhs_quantization_min_val",
        rhs_quantization_min_val, "rhs_quantization_max_val",
        rhs_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_dot_hybrid_eager_fallback(
          lhs, rhs, rhs_scales, rhs_zero_points, Tout=Tout,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedDotHybrid", lhs=lhs, rhs=rhs, rhs_scales=rhs_scales,
                                     rhs_zero_points=rhs_zero_points,
                                     Tout=Tout,
                                     rhs_quantization_min_val=rhs_quantization_min_val,
                                     rhs_quantization_max_val=rhs_quantization_max_val,
                                     rhs_quantization_axis=rhs_quantization_axis,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tlhs", _op._get_attr_type("Tlhs"), "Trhs",
              _op._get_attr_type("Trhs"), "Tout", _op._get_attr_type("Tout"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedDotHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedDotHybrid = tf_export("raw_ops.UniformQuantizedDotHybrid")(_ops.to_raw_op(uniform_quantized_dot_hybrid))


def uniform_quantized_dot_hybrid_eager_fallback(lhs: Annotated[Any, TV_UniformQuantizedDotHybrid_Tlhs], rhs: Annotated[Any, TV_UniformQuantizedDotHybrid_Trhs], rhs_scales: Annotated[Any, _atypes.Float32], rhs_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformQuantizedDotHybrid_Tout, rhs_quantization_min_val: int, rhs_quantization_max_val: int, rhs_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformQuantizedDotHybrid_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _attr_Tlhs, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, ])
  _attr_Trhs, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.qint8, ])
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, rhs_scales, rhs_zero_points]
  _attrs = ("Tlhs", _attr_Tlhs, "Trhs", _attr_Trhs, "Tout", Tout,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedDotHybrid", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedDotHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UniformRequantize_Tin = TypeVar("TV_UniformRequantize_Tin", _atypes.QInt32, _atypes.QInt8)
TV_UniformRequantize_Tout = TypeVar("TV_UniformRequantize_Tout", _atypes.QInt32, _atypes.QInt8)

def uniform_requantize(input: Annotated[Any, TV_UniformRequantize_Tin], input_scales: Annotated[Any, _atypes.Float32], input_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformRequantize_Tout, input_quantization_min_val: int, input_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, input_quantization_axis:int=-1, output_quantization_axis:int=-1, name=None) -> Annotated[Any, TV_UniformRequantize_Tout]:
  r"""Given quantized tensor `input`, requantize it with new quantization parameters.

  Given quantized tensor `input`, which was quantized using {input_scales, input_zero_points, input_quantization_axis, input_quantization_min_val, input_quantization_max_val},
  requantize it to a tensor, which is quantized using {output_scales, output_zero_points, output_quantization_axis, output_quantization_min_val, output_quantization_max_val}.
  The requantization is done by using the formula:
  output_quantized_data = clip(
    (input_quantized_data - input_zero_point) * (input_scale / output_scale) + output_zero_point,
    output_quantization_min_val,
    output_quantization_max_val)

  Per-tensor and per-axis quantization supported cases are followings:
  * per-tensor -> per-tensor
  * per-tensor -> per-axis
  * per-axis -> per-axis where input_quantization_axis equals output_quantization_axis.
  i.e. At least one among input_quantization_axis and output_quantization_axis must be -1, or two must be equal.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
      Must be a Tensor of Tin.
    input_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    input_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing original data that `input` represents.
      Same shape condition as scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as new scale(s) to quantize original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as new zero_point(s) to quantize original data that `input` represents.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
    input_quantization_min_val: An `int`.
      The quantization min value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    input_quantization_max_val: An `int`.
      The quantization max value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    output_quantization_min_val: An `int`.
      The new quantization min value to quantize original data that `input` represents.
    output_quantization_max_val: An `int`.
      The new quantization max value to quantize original data that `input` represents.
    input_quantization_axis: An optional `int`. Defaults to `-1`.
      The quantization axis that was used when quantizing original data that `input` represents.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      The new quantization axis to use to quantize original data that `input` represents.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformRequantize", name, input, input_scales,
        input_zero_points, output_scales, output_zero_points, "Tout", Tout,
        "input_quantization_axis", input_quantization_axis,
        "input_quantization_min_val", input_quantization_min_val,
        "input_quantization_max_val", input_quantization_max_val,
        "output_quantization_axis", output_quantization_axis,
        "output_quantization_min_val", output_quantization_min_val,
        "output_quantization_max_val", output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_requantize_eager_fallback(
          input, input_scales, input_zero_points, output_scales,
          output_zero_points, Tout=Tout,
          input_quantization_axis=input_quantization_axis,
          input_quantization_min_val=input_quantization_min_val,
          input_quantization_max_val=input_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  input_quantization_min_val = _execute.make_int(input_quantization_min_val, "input_quantization_min_val")
  input_quantization_max_val = _execute.make_int(input_quantization_max_val, "input_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if input_quantization_axis is None:
    input_quantization_axis = -1
  input_quantization_axis = _execute.make_int(input_quantization_axis, "input_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformRequantize", input=input, input_scales=input_scales,
                             input_zero_points=input_zero_points,
                             output_scales=output_scales,
                             output_zero_points=output_zero_points, Tout=Tout,
                             input_quantization_min_val=input_quantization_min_val,
                             input_quantization_max_val=input_quantization_max_val,
                             output_quantization_min_val=output_quantization_min_val,
                             output_quantization_max_val=output_quantization_max_val,
                             input_quantization_axis=input_quantization_axis,
                             output_quantization_axis=output_quantization_axis,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "input_quantization_axis",
              _op._get_attr_int("input_quantization_axis"),
              "input_quantization_min_val",
              _op._get_attr_int("input_quantization_min_val"),
              "input_quantization_max_val",
              _op._get_attr_int("input_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformRequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformRequantize = tf_export("raw_ops.UniformRequantize")(_ops.to_raw_op(uniform_requantize))


def uniform_requantize_eager_fallback(input: Annotated[Any, TV_UniformRequantize_Tin], input_scales: Annotated[Any, _atypes.Float32], input_zero_points: Annotated[Any, _atypes.Int32], output_scales: Annotated[Any, _atypes.Float32], output_zero_points: Annotated[Any, _atypes.Int32], Tout: TV_UniformRequantize_Tout, input_quantization_min_val: int, input_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, input_quantization_axis: int, output_quantization_axis: int, name, ctx) -> Annotated[Any, TV_UniformRequantize_Tout]:
  Tout = _execute.make_type(Tout, "Tout")
  input_quantization_min_val = _execute.make_int(input_quantization_min_val, "input_quantization_min_val")
  input_quantization_max_val = _execute.make_int(input_quantization_max_val, "input_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if input_quantization_axis is None:
    input_quantization_axis = -1
  input_quantization_axis = _execute.make_int(input_quantization_axis, "input_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.qint32, ])
  input_scales = _ops.convert_to_tensor(input_scales, _dtypes.float32)
  input_zero_points = _ops.convert_to_tensor(input_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [input, input_scales, input_zero_points, output_scales, output_zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "input_quantization_axis",
  input_quantization_axis, "input_quantization_min_val",
  input_quantization_min_val, "input_quantization_max_val",
  input_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val)
  _result = _execute.execute(b"UniformRequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformRequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

